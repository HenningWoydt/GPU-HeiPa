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
    struct Graph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;

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
            Kokkos::deep_copy(Kokkos::subview(g.memory, std::make_pair((size_t) 0, (size_t) host_g.n_bytes)), host_g.memory);
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

        // hash edges - edge parallel
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "hash_edges_edge_parallel"};

            Kokkos::parallel_for("hash_edges", old_g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = old_g.edges_u(i);
                vertex_t v = old_g.edges_v(i);
                weight_t w = old_g.edges_w(i);

                vertex_t u_new = mapping.mapping(u);
                vertex_t v_new = mapping.mapping(v);

                if (u_new == v_new) { return; } // this edge vanishes

                u32 beg = hash_offsets(u_new);
                u32 end = hash_offsets(u_new + 1);
                u32 len = end - beg;

                for (u32 j = 0; j < len; ++j) {
                    u32 idx = beg + ((v_new + j) % len);
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

        // fill the coarse graph
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "fill_coarse_graph"};

            Kokkos::parallel_scan("fill_coarse_graph", old_g.m, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                vertex_t v = hash_keys(i);
                if (final && v != coarse_g.n) {
                    coarse_g.edges_v(running) = v;
                    coarse_g.edges_w(running) = hash_vals(i);
                }
                running += v != coarse_g.n;
            });

            KOKKOS_PROFILE_FENCE();
        }

        // fill the u array
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "fill_coarse_u"};

            Kokkos::parallel_for("fill_edges_u", coarse_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = coarse_g.neighborhood(u);
                u32 end = coarse_g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    coarse_g.edges_u(i) = u;
                }
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

            KOKKOS_PROFILE_FENCE();
        }
        assert_back_is_empty(mem_stack);

        return coarse_g;
    }

    inline Graph from_Graph_Mapping_new(const Graph &old_g,
                                        const Mapping &mapping,
                                        KokkosMemoryStack &mem_stack) {
        Graph coarse_g;
        // initialize graphs
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "initialize_graph"};

            coarse_g.n = mapping.coarse_n;
            coarse_g.g_weight = old_g.g_weight;

            coarse_g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.n), coarse_g.n);
            coarse_g.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
        }

        UnmanagedDeviceVertex degrees;
        UnmanagedDeviceVertex sum_degrees;

        UnmanagedDeviceU32 hash_offsets;
        UnmanagedDeviceVertex hash_keys;
        UnmanagedDeviceWeight hash_vals;

        UnmanagedDeviceU32 group_size;
        UnmanagedDeviceU32 group_offsets;
        UnmanagedDeviceU32 group_pos;
        UnmanagedDeviceVertex grouped;

        // initialize helpers
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "initialize_helpers"};

            degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * (coarse_g.n + 1)), (coarse_g.n + 1));
            sum_degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * (coarse_g.n + 1)), (coarse_g.n + 1));

            hash_offsets = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            hash_keys = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * old_g.m), old_g.m);
            hash_vals = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * old_g.m), old_g.m);

            group_size = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            group_offsets = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            group_pos = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            grouped = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * old_g.n), old_g.n);
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

            Kokkos::deep_copy(group_size, 0);
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

        vertex_t max_degree = 0;
        // determine max degree
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "max_degree"};

            Kokkos::parallel_reduce("max_degree", Kokkos::RangePolicy<DeviceExecutionSpace>(0, coarse_g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &lmax) {
                const vertex_t deg = hash_offsets(u + 1) - hash_offsets(u);
                if (deg > lmax) lmax = deg;
            }, Kokkos::Max<vertex_t>(max_degree));

            Kokkos::fence();
            KOKKOS_PROFILE_FENCE();
        }

        // determine how many vertice map to v
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "group_size"};

            Kokkos::parallel_for("group_size", old_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t v = mapping.mapping(u);
                Kokkos::atomic_inc(&group_size(v));
            });

            KOKKOS_PROFILE_FENCE();
        }

        // determine the offsets
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "group_offsets"};

            Kokkos::parallel_scan("group_offsets_scan", coarse_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (final) group_offsets(i) = running;
                if (i < (u32) coarse_g.n) running += group_size(i);
            });

            KOKKOS_PROFILE_FENCE();
        }

        // copy the offsets
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "copy_group_offsets"};

            Kokkos::parallel_for("init_group_pos", coarse_g.n, KOKKOS_LAMBDA(const u32 v) {
                group_pos(v) = group_offsets(v);
            });

            KOKKOS_PROFILE_FENCE();
        }

        // group by mapping
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "groups"};

            Kokkos::parallel_for("scatter_grouped_vertices", old_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                const vertex_t v = mapping.mapping(u);
                const u32 idx = Kokkos::atomic_fetch_inc(&group_pos(v));
                grouped(idx) = u;
            });

            KOKKOS_PROFILE_FENCE();
        }

        ScopedTimer _t_temp{"contraction", "from_Graph_Mapping", "grouped_degrees"};
        UnmanagedDeviceU32 gdeg_ps; // size old_g.n + 1
        UnmanagedDeviceU32 re_offsets; // size coarse_g.n + 1 (optional)

        gdeg_ps = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (old_g.n + 1)), old_g.n + 1);
        re_offsets = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);

        Kokkos::parallel_scan("grouped_deg_ps_no_gdeg", Kokkos::RangePolicy<DeviceExecutionSpace>(0, old_g.n + 1), KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
            if (final) gdeg_ps(i) = running;
            if (i < (u32) old_g.n) {
                const vertex_t u = grouped(i);
                const u32 deg = old_g.neighborhood(u + 1) - old_g.neighborhood(u);
                running += deg;
            }
        });
        KOKKOS_PROFILE_FENCE();

        Kokkos::parallel_for("build_re_offsets", Kokkos::RangePolicy<DeviceExecutionSpace>(0, coarse_g.n + 1), KOKKOS_LAMBDA(const u32 v) {
            const u32 p = group_offsets(v); // index into grouped
            re_offsets(v) = gdeg_ps(p); // corresponding edge start
        });
        KOKKOS_PROFILE_FENCE();

        UnmanagedDeviceVertex re_u, re_v;
        UnmanagedDeviceWeight re_w;

        re_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * old_g.m), old_g.m);
        re_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * old_g.m), old_g.m);
        re_w = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * old_g.m), old_g.m);

        using TeamPolicy = Kokkos::TeamPolicy<DeviceExecutionSpace>;
        using Member = TeamPolicy::member_type;

        UnmanagedDeviceU32 pos_of_u((u32 *) get_chunk_back(mem_stack, sizeof(u32) * old_g.n), old_g.n);

        Kokkos::parallel_for("build_pos_of_u", Kokkos::RangePolicy<DeviceExecutionSpace>(0, old_g.n), KOKKOS_LAMBDA(const u32 p) {
            pos_of_u(grouped(p)) = p;
        });
        KOKKOS_PROFILE_FENCE();

        Kokkos::parallel_for("scatter_edges_reordered_edge", Kokkos::RangePolicy<DeviceExecutionSpace>(0, old_g.m), KOKKOS_LAMBDA(const u32 i) {
            const vertex_t u = old_g.edges_u(i);

            const u32 p = pos_of_u(u);
            const u32 out = gdeg_ps(p);

            const u32 e_beg = old_g.neighborhood(u);
            const u32 t = out + (i - e_beg); // i is within u’s CSR range

            re_u(t) = u; // optional
            re_v(t) = old_g.edges_v(i);
            re_w(t) = old_g.edges_w(i);
        });
        KOKKOS_PROFILE_FENCE();

        _t_temp.stop();


        // hash edge vertex parallel
        {
            // hash edges - vertex parallel
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "hash_edges_vertex_parallel"};

            // Scratch: keys[max_degree] + vals[max_degree] + deg
            const size_t scratch_bytes = size_t(max_degree) * sizeof(vertex_t) + size_t(max_degree) * sizeof(weight_t) + sizeof(u32);
            using TeamPolicy = Kokkos::TeamPolicy<DeviceExecutionSpace>;
            using Member = typename TeamPolicy::member_type;

            Kokkos::parallel_for("hash_edges_small_degree_team", TeamPolicy((int) coarse_g.n, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)), KOKKOS_LAMBDA(const Member &team) {
                vertex_t u_new = (vertex_t) team.league_rank();

                u32 g_beg = group_offsets(u_new);
                u32 g_end = group_offsets(u_new + 1);

                u32 r_beg = hash_offsets(u_new);
                u32 r_end = hash_offsets(u_new + 1);
                u32 r_len = r_end - r_beg;

                if (r_len == 0) { return; }

                // --- scratch allocations ---
                auto *s_keys = (vertex_t *) team.team_shmem().get_shmem(sizeof(vertex_t) * r_len);
                auto *s_vals = (weight_t *) team.team_shmem().get_shmem(sizeof(weight_t) * r_len);
                auto *s_deg = (u32 *) team.team_shmem().get_shmem(sizeof(u32));

                // init scratch
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, r_len), [&](int j) { s_keys[j] = coarse_g.n; });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, r_len), [&](int j) { s_vals[j] = 0; });
                Kokkos::single(Kokkos::PerTeam(team), [&]() { *s_deg = 0; });
                team.team_barrier();

                // --- aggregate neighbors into scratch hash ---
                for (u32 p = g_beg; p < g_end; ++p) {
                    vertex_t u = grouped(p);

                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, old_g.neighborhood(u), old_g.neighborhood(u + 1)), [&](u32 i) {
                        vertex_t v = old_g.edges_v(i);
                        weight_t w = old_g.edges_w(i);

                        vertex_t v_new = mapping.mapping(v);
                        if (v_new == u_new) { return; } // self-loop vanishes

                        for (u32 j = 0; j < r_len; ++j) {
                            u32 idx = (v_new + j) % r_len;
                            vertex_t found_v = s_keys[idx];

                            if (found_v == v_new) {
                                Kokkos::atomic_add(&s_vals[idx], w);
                                break;
                            }

                            if (found_v == coarse_g.n) {
                                vertex_t old = Kokkos::atomic_compare_exchange(&s_keys[idx], coarse_g.n, v_new);
                                if (old == v_new || old == coarse_g.n) {
                                    Kokkos::atomic_add(&s_vals[idx], w);
                                    break;
                                }
                            }
                        }
                    });
                    team.team_barrier(); // keep scratch coherent across batches of u
                }

                u32 row_count = 0;

                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, 0, (int) r_len), [&](const int j, u32 &upd, const bool final) {
                    const bool keep = (s_keys[j] != coarse_g.n);
                    if (final && keep) {
                        const u32 out = upd;
                        hash_keys(r_beg + out) = s_keys[j];
                        hash_vals(r_beg + out) = s_vals[j];
                    }
                    upd += keep ? 1u : 0u;
                }, row_count);

                Kokkos::single(Kokkos::PerTeam(team), [&]() { degrees(u_new) = row_count; });
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

        // fill the coarse graph
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "fill_coarse_graph"};

            Kokkos::parallel_for("materialize_edges", coarse_g.n, KOKKOS_LAMBDA(const vertex_t u_new) {
                u32 out = coarse_g.neighborhood(u_new);

                for (u32 idx = hash_offsets(u_new); idx < hash_offsets(u_new + 1); ++idx) {
                    vertex_t v = hash_keys(idx);
                    weight_t w = hash_vals(idx);
                    if (v != coarse_g.n) {
                        coarse_g.edges_v(out) = v;
                        coarse_g.edges_w(out) = w;
                        ++out;
                    }
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        // fill the u array
        {
            ScopedTimer _t{"contraction", "from_Graph_Mapping", "fill_coarse_u"};

            Kokkos::parallel_for("fill_edges_u", coarse_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = coarse_g.neighborhood(u);
                u32 end = coarse_g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    coarse_g.edges_u(i) = u;
                }
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

            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);

            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);

            KOKKOS_PROFILE_FENCE();
        }
        assert_back_is_empty(mem_stack);

        return coarse_g;
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
