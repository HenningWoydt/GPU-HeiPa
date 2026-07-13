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
#include "../definitions.h"
#include "../utility/util.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include <Kokkos_Sort.hpp>

namespace GPU_HeiPa {
    KOKKOS_INLINE_FUNCTION
    u32 mix32(u32 x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    struct Graph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;
        bool uniform_edge_weights = false;
        bool uniform_vertex_weights = false;

        u64 n_pops = 5;

        UnmanagedDeviceWeight weights;
        UnmanagedDeviceU32 neighborhood;
        UnmanagedDeviceVertex edges_u;
        UnmanagedDeviceVertex edges_v;
        UnmanagedDeviceWeight edges_w;
    };

    inline Graph from_HostGraph(const HostGraph &host_g,
                                KokkosMemoryStack &mem_stack,
                                f64 &down_upload_ms,
                                DeviceExecutionSpace &exec_space) {
        Graph g;
        {
            HEIPA_PROFILE_SCOPE("misc", "from_HostGraph", "initialize");

            g.n = host_g.n;
            g.m = host_g.m;
            g.g_weight = host_g.g_weight;
            g.uniform_edge_weights = host_g.uniform_edge_weights;
            g.uniform_vertex_weights = host_g.uniform_vertex_weights;

            g.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
            g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
            g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
            g.n_pops = 3;

            if (!g.uniform_vertex_weights) {
                g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.n), g.n);
                g.n_pops += 1;
            }
            if (!g.uniform_edge_weights) {
                g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.m), g.m);
                g.n_pops += 1;
            }
        }
        auto p = get_time_point();
        {
            HEIPA_PROFILE_SCOPE("up/download", "from_HostGraph", "copy");

            Kokkos::deep_copy(exec_space, g.neighborhood, host_g.neighborhood);
            Kokkos::deep_copy(exec_space, g.edges_v, host_g.edges_v);

            if (!g.uniform_vertex_weights) {
                Kokkos::deep_copy(exec_space, g.weights, host_g.weights);
            }
            if (!g.uniform_edge_weights) {
                Kokkos::deep_copy(exec_space, g.edges_w, host_g.edges_w);
            }

            exec_space.fence();
            KOKKOS_PROFILE_FENCE(exec_space);
        }
        down_upload_ms += get_milli_seconds(p, get_time_point());
        // create the third array
        {
            HEIPA_PROFILE_SCOPE("misc", "from_HostGraph", "fill_edges_u");

            Kokkos::parallel_for("fill_edges_u", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = g.neighborhood(u);
                u32 end = g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    g.edges_u(i) = u;
                }
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        return g;
    }

    template<bool uniform_vw, bool uniform_ew>
    inline Graph from_Graph_Mapping_small(const Graph &old_g,
                                          const Mapping &mapping,
                                          KokkosMemoryStack &mem_stack,
                                          DeviceExecutionSpace &exec_space,
                                          bool sort_by_degree = true) {
        // initialize graphs
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "initialize_graph");
        Graph coarse_g;

        coarse_g.n = mapping.coarse_n;
        coarse_g.g_weight = old_g.g_weight;
        coarse_g.uniform_edge_weights = false;
        coarse_g.uniform_vertex_weights = false;
        coarse_g.n_pops = 5;

        coarse_g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.n), coarse_g.n);
        coarse_g.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
        KOKKOS_PROFILE_FENCE(exec_space);

        // initialize helpers
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "initialize_helpers");
        UnmanagedDeviceWeight dense_adj;
        UnmanagedDeviceU32 degrees;

        dense_adj = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * coarse_g.n * coarse_g.n), coarse_g.n * coarse_g.n);
        degrees = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
        KOKKOS_PROFILE_FENCE(exec_space);

        // zero mem
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "initialize_set_0");
        Kokkos::deep_copy(exec_space, coarse_g.weights, 0);
        Kokkos::deep_copy(exec_space, dense_adj, 0);
        Kokkos::deep_copy(exec_space, degrees, 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        // get coarse weights
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "coarse_weights");
        Kokkos::parallel_for("coarse_weights", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.n), KOKKOS_LAMBDA(const vertex_t u) {
            weight_t u_w = uniform_vw ? 1 : old_g.weights(u);
            vertex_t v = mapping.mapping(u);
            Kokkos::atomic_add(&coarse_g.weights(v), u_w);
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // fill matrix
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "fill_dense");
        Kokkos::parallel_for("fill_dense", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.m), KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = old_g.edges_u(i);
            vertex_t v = old_g.edges_v(i);
            weight_t w = uniform_ew ? 1 : old_g.edges_w(i);

            vertex_t u_new = mapping.mapping(u);
            vertex_t v_new = mapping.mapping(v);

            if (u_new != v_new) {
                Kokkos::atomic_add(&dense_adj(u_new * coarse_g.n + v_new), w);
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // count new degrees
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "count_degrees");
        Kokkos::parallel_for("count_degrees", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n), KOKKOS_LAMBDA(const vertex_t u) {
            u32 count = 0;
            for (vertex_t v = 0; v < coarse_g.n; ++v) {
                if (dense_adj(u * coarse_g.n + v) > 0) count++;
            }
            degrees(u) = count;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        UnmanagedDeviceVertex d_perm;
        UnmanagedDeviceVertex sorted_old_ids;

        if (sort_by_degree) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "sort_by_degree_csr");

            UnmanagedDeviceU32 bin_counts((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            Kokkos::deep_copy(exec_space, bin_counts, 0);

            sorted_old_ids = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * coarse_g.n), coarse_g.n);
            d_perm = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * coarse_g.n), coarse_g.n);
            UnmanagedDeviceWeight temp_weights((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * coarse_g.n), coarse_g.n);
            UnmanagedDeviceU32 temp_degrees((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            auto c_weights = coarse_g.weights;

            typedef Kokkos::TeamPolicy<DeviceExecutionSpace> TeamPolicy;
            typedef TeamPolicy::member_type TeamMember;
            Kokkos::parallel_for("sort_by_degree_csr_fused", TeamPolicy(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
                // 1. count_bins
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, coarse_g.n), [&](const vertex_t u) {
                    Kokkos::atomic_add(&bin_counts(degrees(u)), 1);
                });
                team.team_barrier();

                // 2. scan_bins
                u32 total = 0;
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, coarse_g.n + 1), [&](const u32 i, u32 &running, const bool final) {
                    u32 val = bin_counts(i);
                    if (final) bin_counts(i) = running;
                    running += val;
                }, total);
                team.team_barrier();

                // 3. scatter_ids
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, coarse_g.n), [&](const vertex_t u) {
                    u32 write_idx = Kokkos::atomic_fetch_add(&bin_counts(degrees(u)), 1);
                    sorted_old_ids(write_idx) = u;
                });
                team.team_barrier();

                // 4. permute_vertices
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, coarse_g.n), [&](const vertex_t new_u) {
                    vertex_t old_u = sorted_old_ids(new_u);
                    d_perm(old_u) = new_u;
                    temp_degrees(new_u) = degrees(old_u);
                    temp_weights(new_u) = c_weights(old_u);
                });
            });

            auto map = mapping.mapping;
            Kokkos::parallel_for("update_mapping", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                map(u) = d_perm(map(u));
            });

            Kokkos::deep_copy(exec_space, coarse_g.weights, temp_weights);
            Kokkos::deep_copy(exec_space, degrees, temp_degrees);

            pop_back(mem_stack); // temp_degrees
            pop_back(mem_stack); // temp_weights
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // prefix sum
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "prefix_sum_offsets");
        Kokkos::parallel_scan("prefix_sum_offsets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n + 1), KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
            if (final) coarse_g.neighborhood(i) = running;
            running += degrees(i);
        });

        u32 temp;
        Kokkos::deep_copy(exec_space, temp, Kokkos::subview(coarse_g.neighborhood, coarse_g.n));
        exec_space.fence();
        coarse_g.m = temp;
        KOKKOS_PROFILE_FENCE(exec_space);

        // allocate edges
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "allocate_edges");
        coarse_g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
        coarse_g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
        coarse_g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.m), coarse_g.m);
        KOKKOS_PROFILE_FENCE(exec_space);

        // compact edges
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small", "compact_edges_small");
        Kokkos::parallel_for("compact_edges_small", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n), KOKKOS_LAMBDA(const vertex_t u) {
            u32 write_idx = coarse_g.neighborhood(u);
            vertex_t old_u = sort_by_degree ? sorted_old_ids(u) : u;

            for (vertex_t old_v = 0; old_v < coarse_g.n; ++old_v) {
                weight_t w = dense_adj(old_u * coarse_g.n + old_v);
                if (w > 0) {
                    coarse_g.edges_v(write_idx) = sort_by_degree ? d_perm(old_v) : old_v;
                    coarse_g.edges_w(write_idx) = w;
                    coarse_g.edges_u(write_idx) = u;
                    write_idx++;
                }
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        if (sort_by_degree) {
            pop_back(mem_stack); // d_perm
            pop_back(mem_stack); // sorted_old_ids
            pop_back(mem_stack); // bin_counts
        }

        return coarse_g;
    }

    template<bool uniform_vw, bool uniform_ew>
    inline Graph from_Graph_Mapping(const Graph &old_g,
                                    const Mapping &mapping,
                                    KokkosMemoryStack &mem_stack,
                                    DeviceExecutionSpace &exec_space) {
        Graph coarse_g;
        // initialize graphs
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "initialize_graph");

            coarse_g.n = mapping.coarse_n;
            coarse_g.g_weight = old_g.g_weight;
            coarse_g.uniform_edge_weights = false;
            coarse_g.uniform_vertex_weights = false;
            coarse_g.n_pops = 5;

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
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "initialize_helpers");

            degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * (coarse_g.n + 1)), (coarse_g.n + 1));
            sum_degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * (coarse_g.n + 1)), (coarse_g.n + 1));
            hash_offsets = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            hash_keys = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * old_g.m), old_g.m);
            hash_vals = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * old_g.m), old_g.m);
        }

        // set memory to 0
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "initialize_set_0");

            Kokkos::deep_copy(exec_space, coarse_g.weights, 0);
            Kokkos::deep_copy(exec_space, sum_degrees, 0);
            Kokkos::deep_copy(exec_space, hash_offsets, 0);
            Kokkos::deep_copy(exec_space, hash_keys, coarse_g.n);
            Kokkos::deep_copy(exec_space, hash_vals, 0);

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // determine weight of new vertices and maximum degree
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "max_degrees");

            Kokkos::parallel_for("max_degrees", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                u32 deg = old_g.neighborhood(u + 1) - old_g.neighborhood(u);
                weight_t u_w = uniform_vw ? 1 : old_g.weights(u);
                vertex_t v = mapping.mapping(u);

                Kokkos::atomic_add(&sum_degrees(v), deg);
                Kokkos::atomic_add(&coarse_g.weights(v), u_w);
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // prefix sum over all degrees
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "max_degrees_prefix_sum");

            Kokkos::parallel_scan("prefix_sum_offsets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n + 1), KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (final) { hash_offsets(i) = running; }
                running += sum_degrees(i);
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // hash edges
        {
            u32 avg_degree = old_g.n > 0 ? (old_g.m / old_g.n) : 0;

            if (avg_degree > 32) {
                // team-parallel over fine vertices: each team cooperatively scans one vertex's adjacency
                HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "hash_edges_team_parallel");

                Kokkos::parallel_for("hash_edges_tp", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, old_g.n, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
                    vertex_t old_u = team.league_rank();
                    vertex_t new_u = mapping.mapping(old_u);
                    u32 beg = hash_offsets(new_u);
                    u32 len = hash_offsets(new_u + 1) - beg;

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, old_g.neighborhood(old_u), old_g.neighborhood(old_u + 1)), [=](const u32 j) {
                        vertex_t v_new = mapping.mapping(old_g.edges_v(j));
                        if (new_u == v_new) return;
                        weight_t w = uniform_ew ? 1 : old_g.edges_w(j);

                        u32 offset = mix32(v_new) % len;
                        while (true) {
                            u32 idx = beg + offset;
                            vertex_t found_v = hash_keys(idx);

                            if (found_v == v_new) {
                                Kokkos::atomic_add(&hash_vals(idx), w);
                                break;
                            }
                            if (found_v == coarse_g.n) {
                                vertex_t old = Kokkos::atomic_compare_exchange(&hash_keys(idx), coarse_g.n, v_new);
                                if (old == v_new || old == coarse_g.n) {
                                    Kokkos::atomic_add(&hash_vals(idx), w);
                                    break;
                                }
                            }
                            offset++;
                            if (offset >= len) offset = 0;
                        }
                    });
                });

                KOKKOS_PROFILE_FENCE(exec_space);
            } else {
                // edge-parallel: one thread per edge, good for low-degree graphs
                HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "hash_edges_edge_parallel");

                Kokkos::parallel_for("hash_edges_ep", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.m), KOKKOS_LAMBDA(const u32 i) {
                    vertex_t u = old_g.edges_u(i);
                    vertex_t v = old_g.edges_v(i);
                    weight_t w = uniform_ew ? 1 : old_g.edges_w(i);

                    vertex_t u_new = mapping.mapping(u);
                    vertex_t v_new = mapping.mapping(v);

                    if (u_new == v_new) { return; }

                    u32 beg = hash_offsets(u_new);
                    u32 len = hash_offsets(u_new + 1) - beg;

                    u32 offset = mix32(v_new) % len;
                    while (true) {
                        u32 idx = beg + offset;
                        vertex_t found_v = hash_keys(idx);

                        if (found_v == v_new) {
                            Kokkos::atomic_add(&hash_vals(idx), w);
                            break;
                        }
                        if (found_v == coarse_g.n) {
                            vertex_t old = Kokkos::atomic_compare_exchange(&hash_keys(idx), coarse_g.n, v_new);
                            if (old == v_new || old == coarse_g.n) {
                                Kokkos::atomic_add(&hash_vals(idx), w);
                                break;
                            }
                        }
                        offset++;
                        if (offset >= len) offset = 0;
                    }
                });

                KOKKOS_PROFILE_FENCE(exec_space);
            }
        }

        // count unique entries per coarse vertex
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "count_unique");

            u32 avg_hash_len = old_g.n > 0 ? (old_g.m / coarse_g.n) : 0;
            if (avg_hash_len >= 12) {
                Kokkos::parallel_for("count_unique", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, coarse_g.n, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
                    const vertex_t u = team.league_rank();
                    u32 h_beg = hash_offsets(u);
                    u32 h_end = hash_offsets(u + 1);
                    u32 count = 0;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, h_beg, h_end), [=](const u32 j, u32 &local) {
                        local += (hash_keys(j) != coarse_g.n);
                    }, count);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        degrees(u) = count;
                    });
                });
            } else {
                Kokkos::parallel_for("count_unique", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                    u32 h_beg = hash_offsets(u);
                    u32 h_end = hash_offsets(u + 1);
                    u32 count = 0;
                    for (u32 j = h_beg; j < h_end; ++j) {
                        count += (hash_keys(j) != coarse_g.n);
                    }
                    degrees(u) = count;
                });
            }

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // build offsets
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "build_offsets");

            Kokkos::parallel_scan("build_offsets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n + 1), KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (final) { coarse_g.neighborhood(i) = running; }
                running += degrees(i);
            });
            u32 temp;
            Kokkos::deep_copy(exec_space, temp, Kokkos::subview(coarse_g.neighborhood, coarse_g.n)); // copy final number of edges m
            exec_space.fence();
            coarse_g.m = temp;

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // allocate edges for coarse graph
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "allocate_edges");

            coarse_g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
            coarse_g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
            coarse_g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.m), coarse_g.m);

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // fill the coarse graph and u array
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "fill_coarse_graph");

            Kokkos::parallel_scan("fill_coarse_graph", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.m), KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                vertex_t v = hash_keys(i);
                if (final && v != coarse_g.n) {
                    coarse_g.edges_v(running) = v;
                    coarse_g.edges_w(running) = hash_vals(i);
                }
                running += v != coarse_g.n;
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // fill the u array
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "fill_coarse_u");

            Kokkos::parallel_for("fill_edges_u", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = coarse_g.neighborhood(u);
                u32 end = coarse_g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    coarse_g.edges_u(i) = u;
                }
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // deallocate mem
        {
            HEIPA_PROFILE_SCOPE("contraction", "from_Graph_Mapping", "deallocate");

            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        return coarse_g;
    }

    inline Graph make_graph(const vertex_t n,
                            const vertex_t m,
                            const weight_t w,
                            bool uniform_vw,
                            bool uniform_ew,
                            KokkosMemoryStack &mem_stack) {
        Graph g;
        // initialize the graph

        g.n = n;
        g.m = m;
        g.g_weight = w;
        g.uniform_vertex_weights = uniform_vw;
        g.uniform_edge_weights = uniform_ew;

        g.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
        g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        g.n_pops = 3;

        if (!g.uniform_vertex_weights) {
            g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.n), g.n);
            g.n_pops += 1;
        }
        if (!g.uniform_edge_weights) {
            g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.m), g.m);
            g.n_pops += 1;
        }

        return g;
    }

    inline HostGraph to_host_graph(const Graph &device_g,
                                   DeviceExecutionSpace &exec_space) {
        HostGraph host_g;
        host_g.uniform_vertex_weights = device_g.uniform_vertex_weights;
        host_g.uniform_edge_weights = device_g.uniform_edge_weights;

        allocate_memory(host_g, device_g.n, device_g.m, device_g.g_weight);

        Kokkos::deep_copy(exec_space, host_g.neighborhood, Kokkos::subview(device_g.neighborhood, std::make_pair(0U, device_g.n + 1)));
        Kokkos::deep_copy(exec_space, host_g.edges_v, Kokkos::subview(device_g.edges_v, std::make_pair(0U, device_g.m)));

        if (device_g.uniform_vertex_weights) {
            Kokkos::deep_copy(exec_space, host_g.weights, 1);
        } else {
            Kokkos::deep_copy(exec_space, host_g.weights, Kokkos::subview(device_g.weights, std::make_pair(0U, device_g.n)));
        }

        if (device_g.uniform_edge_weights) {
            Kokkos::deep_copy(exec_space, host_g.edges_w, 1);
        } else {
            Kokkos::deep_copy(exec_space, host_g.edges_w, Kokkos::subview(device_g.edges_w, std::make_pair(0U, device_g.m)));
        }
        exec_space.fence();

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
