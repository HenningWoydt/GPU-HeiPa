/*******************************************************************************
 * MIT License
 *
 * This file is part of GPU-HeiPa.
 *
 * Copyright (C) 2025 Henning Woydt <henning.woydt@informatik.uni-heidelberg.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef GPU_HEIPA_GRAPH_H
#define GPU_HEIPA_GRAPH_H

#include "host_graph.h"
#include "kokkos_memory_stack.h"
#include "mapping.h"
#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    KOKKOS_INLINE_FUNCTION
    u32 mix32(u32 x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    struct Graph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;
        bool uniform_edge_weights = false;
        bool uniform_vertex_weights = false;

        UnmanagedDeviceU8 memory;
        u64 n_bytes = 0;
        u64 n_pops = 5;

        UnmanagedDeviceWeight weights;
        UnmanagedDeviceU32 neighborhood;
        UnmanagedDeviceVertex edges_u;
        UnmanagedDeviceVertex edges_v;
        UnmanagedDeviceWeight edges_w;
    };

    inline Graph from_HostGraph(const HostGraph &host_g,
                                KokkosMemoryStack &mem_stack,
                                f64 &down_upload_ms) {
        Graph g;
        // initialize the graph
        {
            ScopedTimer _t{"misc", "from_HostGraph", "initialize"};

            g.n = host_g.n;
            g.m = host_g.m;
            g.g_weight = host_g.g_weight;
            g.uniform_edge_weights = host_g.uniform_edge_weights;
            g.uniform_vertex_weights = host_g.uniform_vertex_weights;

            // bytes needed per array
            size_t off_weights = 0;
            size_t bytes_weights = round_up_64(sizeof(weight_t) * (size_t) g.n);

            size_t off_neighborhood = off_weights + bytes_weights;
            size_t bytes_neighborhood = round_up_64(sizeof(u32) * (size_t) (g.n + 1));

            size_t off_edges_v = off_neighborhood + bytes_neighborhood;
            size_t bytes_edges_v = round_up_64(sizeof(vertex_t) * (size_t) g.m);

            size_t off_edges_w = off_edges_v + bytes_edges_v;
            size_t bytes_edges_w = round_up_64(sizeof(weight_t) * (size_t) g.m);

            size_t off_edges_u = off_edges_w + bytes_edges_w;
            size_t bytes_edges_u = round_up_64(sizeof(vertex_t) * (size_t) g.m);

            // total n bytes
            g.n_bytes = off_edges_u + bytes_edges_u;

            // allocate one owning chunk
            g.memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * g.n_bytes), g.n_bytes);
            uint8_t *base = g.memory.data();
            g.n_pops = 1;

            // create unmanaged views into the chunk
            g.weights = UnmanagedDeviceWeight((weight_t *) (base + off_weights), g.n);
            g.neighborhood = UnmanagedDeviceU32((u32 *) (base + off_neighborhood), g.n + 1);
            g.edges_v = UnmanagedDeviceVertex((vertex_t *) (base + off_edges_v), g.m);
            g.edges_w = UnmanagedDeviceWeight((weight_t *) (base + off_edges_w), g.m);
            g.edges_u = UnmanagedDeviceVertex((vertex_t *) (base + off_edges_u), g.m);
        }
        auto p = get_time_point();
        // copy the structure to device
        {
            ScopedTimer _t{"up/download", "from_HostGraph", "copy"};

            if (g.uniform_vertex_weights && g.uniform_edge_weights) {
                // only upload neighborhood and edges_v, fill weights/edges_w with 1 on GPU
                Kokkos::deep_copy(g.neighborhood, host_g.neighborhood);
                Kokkos::deep_copy(g.edges_v, host_g.edges_v);
                Kokkos::parallel_for("fill_weights_1", g.n, KOKKOS_LAMBDA(const vertex_t i) {
                    g.weights(i) = 1;
                });
                Kokkos::parallel_for("fill_edges_w_1", g.m, KOKKOS_LAMBDA(const u32 i) {
                    g.edges_w(i) = 1;
                });
            } else if (g.uniform_edge_weights) {
                // upload weights, neighborhood, edges_v; fill edges_w with 1
                Kokkos::deep_copy(g.weights, host_g.weights);
                Kokkos::deep_copy(g.neighborhood, host_g.neighborhood);
                Kokkos::deep_copy(g.edges_v, host_g.edges_v);
                Kokkos::parallel_for("fill_edges_w_1", g.m, KOKKOS_LAMBDA(const u32 i) {
                    g.edges_w(i) = 1;
                });
            } else if (g.uniform_vertex_weights) {
                // upload neighborhood, edges_v, edges_w; fill weights with 1
                Kokkos::deep_copy(g.neighborhood, host_g.neighborhood);
                Kokkos::deep_copy(g.edges_v, host_g.edges_v);
                Kokkos::deep_copy(g.edges_w, host_g.edges_w);
                Kokkos::parallel_for("fill_weights_1", g.n, KOKKOS_LAMBDA(const vertex_t i) {
                    g.weights(i) = 1;
                });
            } else {
                // upload everything
                Kokkos::deep_copy(Kokkos::subview(g.memory, std::make_pair((size_t) 0, (size_t) host_g.n_bytes)), host_g.memory);
            }
            KOKKOS_PROFILE_FENCE();
        }
        down_upload_ms += get_milli_seconds(p, get_time_point());
        // create the third array
        {
            ScopedTimer _t{"misc", "from_HostGraph", "fill_edges_u"};
            Kokkos::parallel_for("fill_edges_u", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = g.neighborhood(u);
                u32 end = g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    g.edges_u(i) = u;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        return g;
    }

    inline Graph from_Graph_Mapping(const Graph &old_g,
                                    const Mapping &mapping,
                                    KokkosMemoryStack &mem_stack) {
        Graph coarse_g;
        // initialize graphs
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "initialize_graph"};

            coarse_g.n = mapping.coarse_n;
            coarse_g.g_weight = old_g.g_weight;
            coarse_g.uniform_edge_weights = false;
            coarse_g.uniform_vertex_weights = false;

            coarse_g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.n), coarse_g.n);
            coarse_g.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
        }

        UnmanagedDeviceVertex degrees;
        UnmanagedDeviceVertex sum_degrees;

        UnmanagedDeviceU32 hash_offsets;
        UnmanagedDeviceVertex hash_keys;
        UnmanagedDeviceWeight hash_vals;
        // initialize helpers
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "initialize_helpers"};

            degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * (coarse_g.n + 1)), (coarse_g.n + 1));
            sum_degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * (coarse_g.n + 1)), (coarse_g.n + 1));
            hash_offsets = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            hash_keys = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * old_g.m), old_g.m);
            hash_vals = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * old_g.m), old_g.m);
        }

        // set memory to 0
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "initialize_set_0"};

            Kokkos::deep_copy(coarse_g.weights, 0);
            Kokkos::deep_copy(degrees, 0);
            Kokkos::deep_copy(sum_degrees, 0);
            Kokkos::deep_copy(hash_offsets, 0);
            Kokkos::deep_copy(hash_keys, coarse_g.n);
            Kokkos::deep_copy(hash_vals, 0);

            KOKKOS_PROFILE_FENCE();
        }

        // determine weight of new vertices and maximum degree
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "max_degrees"};

            Kokkos::parallel_for("max_degrees", old_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 deg = old_g.neighborhood(u + 1) - old_g.neighborhood(u);
                weight_t u_w = old_g.weights(u);
                vertex_t v = mapping.mapping(u);

                Kokkos::atomic_add(&sum_degrees(v), deg);
                Kokkos::atomic_add(&coarse_g.weights(v), u_w);
            });
            KOKKOS_PROFILE_FENCE();
        }

        // prefix sum over all degrees
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "max_degrees_prefix_sum"};

            Kokkos::parallel_scan("prefix_sum_offsets", coarse_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (final) { hash_offsets(i) = running; }
                running += sum_degrees(i);
            });

            KOKKOS_PROFILE_FENCE();
        }

        // hash edges
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "hash_edges_edge_parallel"};

            Kokkos::parallel_for("hash_edges_ep", old_g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = old_g.edges_u(i);
                vertex_t v = old_g.edges_v(i);
                weight_t w = old_g.edges_w(i);

                vertex_t u_new = mapping.mapping(u);
                vertex_t v_new = mapping.mapping(v);

                if (u_new == v_new) { return; }

                u32 beg = hash_offsets(u_new);
                u32 end = hash_offsets(u_new + 1);
                u32 len = end - beg;

                u32 h = mix32(v_new);
                for (u32 j = 0; j < len; ++j) {
                    u32 idx = beg + ((h + j) % len);
                    vertex_t found_v = hash_keys(idx);

                    if (found_v == v_new) {
                        Kokkos::atomic_add(&hash_vals(idx), w);
                        break;
                    }

                    if (found_v == coarse_g.n) {
                        vertex_t old = Kokkos::atomic_compare_exchange(&hash_keys(idx), coarse_g.n, v_new);
                        if (old == v_new) {
                            Kokkos::atomic_add(&hash_vals(idx), w);
                            break;
                        }

                        if (old == coarse_g.n) {
                            Kokkos::atomic_inc(&degrees(u_new));
                            Kokkos::atomic_add(&hash_vals(idx), w);
                            break;
                        }
                    }
                }
            });

            KOKKOS_PROFILE_FENCE();
        }

        // build offsets
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "build_offsets"};

            Kokkos::parallel_scan("build_offsets", coarse_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (final) { coarse_g.neighborhood(i) = running; }
                running += degrees(i);
            });
            u32 temp;
            Kokkos::deep_copy(temp, Kokkos::subview(coarse_g.neighborhood, coarse_g.n)); // copy final number of edges m
            coarse_g.m = (vertex_t) temp;

            KOKKOS_PROFILE_FENCE();
        }

        // allocate edges for coarse graph
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "allocate_edges"};

            coarse_g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
            coarse_g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
            coarse_g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.m), coarse_g.m);

            KOKKOS_PROFILE_FENCE();
        }

        // fill the coarse graph and u array
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "fill_coarse_graph"};

            Kokkos::parallel_for("fill_coarse_graph", Kokkos::TeamPolicy<>(coarse_g.n, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
                vertex_t u = team.league_rank();
                u32 h_beg = hash_offsets(u);
                u32 h_end = hash_offsets(u + 1);
                u32 write_base = coarse_g.neighborhood(u);

                // count non-null entries with team reduction to get per-thread write offsets
                u32 count = 0;
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, h_beg, h_end), [=](const u32 i, u32 &partial, const bool final_pass) {
                    u32 is_valid = (hash_keys(i) != coarse_g.n) ? 1 : 0;
                    if (final_pass && is_valid) {
                        u32 pos = write_base + partial;
                        coarse_g.edges_v(pos) = hash_keys(i);
                        coarse_g.edges_w(pos) = hash_vals(i);
                        coarse_g.edges_u(pos) = u;
                    }
                    partial += is_valid;
                }, count);
            });

            KOKKOS_PROFILE_FENCE();
        }

        // deallocate mem
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "deallocate"};

            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);

            KOKKOS_PROFILE_FENCE();
        }

        return coarse_g;
    }

    inline Graph make_graph(const vertex_t n,
                            const vertex_t m,
                            const weight_t w,
                            KokkosMemoryStack &mem_stack) {
        Graph g;
        // initialize the graph

        g.n = n;
        g.m = m;
        g.g_weight = w;

        g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.n), g.n);
        g.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
        g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.m), g.m);
        g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        g.n_pops = 5;

        return g;
    }

    inline HostGraph to_host_graph(const Graph &device_g) {
        HostGraph host_g;

        allocate_memory(host_g, device_g.n, device_g.m, device_g.g_weight);

        Kokkos::deep_copy(host_g.weights, device_g.weights);
        Kokkos::deep_copy(host_g.neighborhood, device_g.neighborhood);
        Kokkos::deep_copy(host_g.edges_v, device_g.edges_v);
        Kokkos::deep_copy(host_g.edges_w, device_g.edges_w);
        Kokkos::fence();

        return host_g;
    }

    inline void free_graph(Graph &g,
                           KokkosMemoryStack &mem_stack) {
        for (u64 i = 0; i < g.n_pops; ++i) {
            pop_front(mem_stack);
        }
    }
}

#endif //GPU_HEIPA_GRAPH_H
