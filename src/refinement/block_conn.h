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

#ifndef GPU_HEIPA_BLOCK_CONN_H
#define GPU_HEIPA_BLOCK_CONN_H

#include <Kokkos_Core.hpp>

#include "../definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    struct BlockConn {
        vertex_t n = 0;
        u32 size = 0;

        UnmanagedDeviceU32 row;
        UnmanagedDeviceU32 sizes;

        UnmanagedDevicePartition ids;
        UnmanagedDeviceWeight weights;
    };

    template<bool uniform_e_weights>
    inline BlockConn init_BlockConn(const Graph &g,
                                    const Partition &partition,
                                    KokkosMemoryStack &mem_stack,
                                    DeviceExecutionSpace &exec_space) {
        BlockConn bc;
        HEIPA_PROFILE_SCOPE("refinement", "BlockConnectivity_fs", "allocate_rows");

        bc.n = g.n;
        bc.row = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
        bc.sizes = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * g.n), g.n);

        KOKKOS_PROFILE_FENCE(exec_space);

        // set rows
        HEIPA_PROFILE_SCOPE("refinement", "BlockConnectivity_fs", "set_rows");
        auto bc_row_init = bc.row;
        auto bc_sizes_init = bc.sizes;
        auto g_neighborhood_init = g.neighborhood;
        auto p_k_init = partition.k;
        
        Kokkos::parallel_scan("set_rows", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n + 1), KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
            if (i == 0) {
                // first slot is 0
                if (final) bc_row_init(0) = 0;
                return;
            }

            const vertex_t u = i - 1;
            const u32 len = g_neighborhood_init(u + 1) - g_neighborhood_init(u);
            const u32 c = len < p_k_init ? len : p_k_init;

            // write inclusive row[i] = running + c
            if (final) {
                bc_row_init(i) = running + c;
                bc_sizes_init(u) = 0; // c;
            }

            running += c;
        });

        Kokkos::deep_copy(exec_space, bc.size, Kokkos::subview(bc.row, g.n));
        exec_space.fence();
        KOKKOS_PROFILE_FENCE(exec_space);

        // allocate rest
        HEIPA_PROFILE_SCOPE("refinement", "BlockConnectivity_fs", "allocate");
        bc.ids = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * bc.size), bc.size);
        bc.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * bc.size), bc.size);
        Kokkos::deep_copy(exec_space, bc.ids, NULL_PART);
        Kokkos::deep_copy(exec_space, bc.weights, 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        auto g_edges_u = g.edges_u;
        auto g_edges_v = g.edges_v;
        auto g_edges_w = g.edges_w;
        auto g_neighborhood = g.neighborhood;
        auto p_map = partition.map;
        auto bc_row = bc.row;
        auto bc_ids = bc.ids;
        auto bc_weights = bc.weights;
        auto bc_sizes = bc.sizes;

        // first fill of the structure
        if (g.m / g.n < 16) {
            HEIPA_PROFILE_SCOPE("refinement", "BlockConnectivity_fs", "fill");
            Kokkos::parallel_for("fill", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.m), KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = g_edges_u(i);
                vertex_t v = g_edges_v(i);
                weight_t w = uniform_e_weights ? 1 : g_edges_w(i);

                u32 r_beg = bc_row(u);
                u32 r_end = bc_row(u + 1);
                u32 r_len = r_end - r_beg;

                partition_t v_id = p_map(v);

                for (u32 j = 0; j < r_len; j++) {
                    u32 idx = r_beg + j;
                    partition_t val = bc_ids(idx);
                    if (val == v_id) {
                        Kokkos::atomic_add(&bc_weights(idx), w);
                        return;
                    }
                    if (val == NULL_PART) {
                        val = Kokkos::atomic_compare_exchange(&bc_ids(idx), NULL_PART, v_id);
                        if (val == NULL_PART) {
                            Kokkos::atomic_add(&bc_weights(idx), w);
                            Kokkos::atomic_inc(&bc_sizes(u));
                            return;
                        }
                        if (val == v_id) {
                            Kokkos::atomic_add(&bc_weights(idx), w);
                            return;
                        }
                    }
                }
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        } else {
            using team_policy = Kokkos::TeamPolicy<DeviceExecutionSpace>;
            using member_type = team_policy::member_type;

            HEIPA_PROFILE_SCOPE("refinement", "BlockConnectivity_fs", "fill_team");
            Kokkos::parallel_for("fill_team_per_vertex", team_policy(exec_space, g.n, Kokkos::AUTO()), KOKKOS_LAMBDA(const member_type &team) {
                vertex_t u = team.league_rank();

                u32 r_beg = bc_row(u);
                u32 r_end = bc_row(u + 1);
                u32 r_len = r_end - r_beg;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_neighborhood(u), g_neighborhood(u + 1)), [&](const u32 i) {
                    vertex_t v = g_edges_v(i);
                    weight_t w = uniform_e_weights ? 1 : g_edges_w(i);
                    partition_t v_id = p_map(v);

                    for (u32 j = 0; j < r_len; ++j) {
                        u32 idx = r_beg + j;
                        partition_t val = bc_ids(idx);

                        if (val == v_id) {
                            Kokkos::atomic_add(&bc_weights(idx), w);
                            return;
                        }

                        if (val == NULL_PART) {
                            val = Kokkos::atomic_compare_exchange(&bc_ids(idx), NULL_PART, v_id);

                            if (val == NULL_PART) {
                                Kokkos::atomic_add(&bc_weights(idx), w);
                                Kokkos::atomic_inc(&bc_sizes(u));
                                return;
                            }

                            if (val == v_id) {
                                Kokkos::atomic_add(&bc_weights(idx), w);
                                return;
                            }
                        }
                    }
                });
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        return bc;
    }

    inline void free_BlockConn(BlockConn &bc,
                               KokkosMemoryStack &mem_stack) {
        HEIPA_PROFILE_SCOPE("refinement", "BlockConnectivity", "free");

        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    template<bool uniform_e_weights>
    inline void update_large(const Graph &g,
                             Partition &partition,
                             UnmanagedDeviceU32 &moved_round,
                             u32 &round,
                             UnmanagedDevicePartition &dest_cache,
                             BlockConn &bc,
                             const DeviceVertex &moves,
                             DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "update_large_mark");
        round += 1;
        u32 total_moves = (u32) moves.extent(0);
        Kokkos::parallel_for("mark", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, (int) total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            u32 i = (u32) t.league_rank();
            vertex_t u = moves(i);
            moved_round(u) = round;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        //recompute conn tables for each vertex adjacent to a moved vertex
        HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "update_large_rebuild");
        auto g_neighborhood = g.neighborhood;
        auto g_edges_v = g.edges_v;
        auto g_edges_w = g.edges_w;
        auto p_map = partition.map;
        auto bc_row = bc.row;
        auto bc_ids = bc.ids;
        auto bc_weights = bc.weights;
        auto bc_sizes = bc.sizes;
        auto p_k = partition.k;

        Kokkos::parallel_for("rebuild", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, (int) g.n, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(partition.k * sizeof(weight_t) + partition.k * sizeof(partition_t) + 4 * sizeof(partition_t))), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            vertex_t u = (vertex_t) t.league_rank();

            bool needs_update = false;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g_neighborhood(u), g_neighborhood(u + 1)), [&](const u32 i, bool &local_update) {
                const vertex_t v = g_edges_v(i);
                local_update = local_update || (moved_round(v) == round);
            }, Kokkos::LOr<bool>(needs_update));

            if (needs_update) {
                u32 r_beg = bc_row(u);
                u32 r_end = bc_row(u + 1);
                u32 r_len = r_end - r_beg;

                // build the row
                weight_t *s_weights = (weight_t *) t.team_shmem().get_shmem(sizeof(weight_t) * r_len);
                partition_t *s_ids = (partition_t *) t.team_shmem().get_shmem(sizeof(partition_t) * r_len);
                u32 *n_needed_slots = (u32 *) t.team_shmem().get_shmem(sizeof(u32));

                // reset weights and ids
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const vertex_t j) {
                    s_weights[j] = 0;
                    s_ids[j] = NULL_PART;
                });
                *n_needed_slots = 0;
                t.team_barrier();

                // construct conn table from scratch in shared memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_neighborhood(u), g_neighborhood(u + 1)), [&](const u32 &i) {
                    vertex_t v = g_edges_v(i);
                    weight_t w = uniform_e_weights ? 1 : g_edges_w(i);
                    partition_t v_id = p_map(v);
                    u32 idx = v_id % r_len;

                    if (r_len == p_k) {
                        if (NULL_PART == Kokkos::atomic_compare_exchange(s_ids + idx, NULL_PART, v_id)) {
                            Kokkos::atomic_add(n_needed_slots, 1);
                        }
                    } else {
                        while (true) {
                            partition_t id = Kokkos::atomic_compare_exchange(s_ids + idx, NULL_PART, v_id);
                            if (id == v_id) { break; }
                            if (id == NULL_PART) {
                                Kokkos::atomic_add(n_needed_slots, 1);
                                break;
                            }
                            idx += 1;
                            if (idx == r_len) { idx = 0; }
                        }
                    }
                    Kokkos::atomic_add(s_weights + idx, w);
                });
                t.team_barrier();

                u32 new_size = *n_needed_slots + ((*n_needed_slots / 4) < 3 ? 3 : (*n_needed_slots / 4));

                if (new_size < r_len) {
                    // reset global memory
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, r_beg, r_end), [&](const u32 i) {
                        bc_weights(i) = 0;
                        bc_ids(i) = NULL_PART;
                    });

                    t.team_barrier();

                    bc_sizes(u) = new_size;
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const u32 &i) {
                        partition_t id = s_ids[i];
                        if (id != NULL_PART) {
                            u32 idx = id % new_size;

                            while (true) {
                                partition_t found_id = Kokkos::atomic_compare_exchange(&bc_ids(r_beg + idx), NULL_PART, id);
                                if (found_id == NULL_PART || found_id == id) { break; }
                                idx += 1;
                                if (idx == new_size) { idx = 0; }
                            }

                            bc_weights(r_beg + idx) = s_weights[i];
                        }
                    });
                } else {
                    bc_sizes(u) = r_len;
                    //copy conn table into global memory
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const u32 i) {
                        bc_weights(r_beg + i) = s_weights[i];
                        bc_ids(r_beg + i) = s_ids[i];
                    });
                }

                // reset cache
                Kokkos::single(Kokkos::PerTeam(t), [=]() {
                    dest_cache(u) = NULL_PART;
                });
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);
    }

    template<bool uniform_e_weights>
    inline void update_small(const Graph &g,
                             Partition &partition,
                             UnmanagedDevicePartition &dest_part,
                             UnmanagedDevicePartition &dest_cache,
                             BlockConn &bc,
                             const DeviceVertex &moves,
                             DeviceExecutionSpace &exec_space) {
        u32 total_moves = (u32) moves.extent(0);

        HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "update_small_remove_weight");
        auto g_neighborhood = g.neighborhood;
        auto g_edges_v = g.edges_v;
        auto g_edges_w = g.edges_w;
        auto bc_row = bc.row;
        auto bc_sizes = bc.sizes;
        auto bc_ids = bc.ids;
        auto bc_weights = bc.weights;
        auto p_k = partition.k;
        auto p_map = partition.map;

        Kokkos::parallel_for("remove_weight", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, (int) total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            vertex_t u = moves((u32) t.league_rank());
            partition_t old_u_id = dest_part(u);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_neighborhood(u), g_neighborhood(u + 1)), [=](const u32 i) {
                vertex_t v = g_edges_v(i);
                weight_t w = uniform_e_weights ? 1 : g_edges_w(i);

                u32 r_beg = bc_row(v);
                u32 size = bc_sizes(v);

                // find correct idx
                partition_t idx = old_u_id % size;
                while (Kokkos::atomic_load(&bc_ids(r_beg + idx)) != old_u_id) {
                    idx += 1;
                    if (idx == size) { idx = 0; }
                }

                // remove weight
                weight_t id_w = Kokkos::atomic_fetch_add(&bc_weights(r_beg + idx), -w);

                if (size != p_k && id_w == w) { bc_ids(r_beg + idx) = HASH_RECLAIM; }
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "update_small_add_weight");
        Kokkos::parallel_for("add_weight", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, (int) total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            vertex_t u = moves((u32) t.league_rank());
            partition_t new_u_id = p_map(u);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_neighborhood(u), g_neighborhood(u + 1)), [=](const u32 i) {
                vertex_t v = g_edges_v(i);
                weight_t w = uniform_e_weights ? 1 : g_edges_w(i);

                dest_cache(v) = NULL_PART; // reset the cache

                u32 r_beg = bc_row(v);
                u32 size = bc_sizes(v);

                u32 idx = new_u_id % size;

                // first pass look for new_u_id
                bool success = false;
                for (u32 j = 0; j < size; j++) {
                    idx = (new_u_id + j) % size;
                    partition_t id = Kokkos::atomic_load(&bc_ids(r_beg + idx));

                    if (id == new_u_id) {
                        success = true;
                        break;
                    }
                    if (id == NULL_PART) { break; }
                }

                if (!success) {
                    for (u32 j = 0; j < size; j++) {
                        idx = (new_u_id + j) % size;
                        partition_t id = Kokkos::atomic_load(&bc_ids(r_beg + idx));

                        if (id == new_u_id) {
                            success = true;
                            break;
                        }

                        if (id == NULL_PART || id == HASH_RECLAIM) {
                            partition_t found_id = Kokkos::atomic_compare_exchange(&bc_ids(r_beg + idx), id, new_u_id);
                            if (found_id == new_u_id || found_id == NULL_PART || found_id == HASH_RECLAIM) {
                                success = true;
                                break;
                            }
                        }
                    }
                }

                if (!success) {
                    idx = size;
                    while (true) {
                        partition_t id = Kokkos::atomic_load(&bc_ids(r_beg + idx));

                        if (id == new_u_id) {
                            success = true;
                            break;
                        }

                        if (id == NULL_PART || id == HASH_RECLAIM) {
                            partition_t found_id = Kokkos::atomic_compare_exchange(&bc_ids(r_beg + idx), id, new_u_id);
                            if (found_id == id) {
                                Kokkos::atomic_add(&bc_sizes(v), 1);
                                break;
                            }
                            if (found_id == new_u_id) { break; }
                        }

                        idx++;
                    }
                }
                Kokkos::atomic_add(&bc_weights(r_beg + idx), w);
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);
    }

    KOKKOS_INLINE_FUNCTION
    static weight_t lookup(const partition_t *keys, const weight_t *vals, const partition_t target, const u32 size) {
        for (u32 i = 0; i < size; i++) {
            u32 idx = ((u32) target + i) % size;

            if (keys[idx] == target) { return vals[idx]; }
            if (keys[idx] == NULL_PART) { return 0; }
        }
        return 0;
    }

    struct HostBlockConn {
        vertex_t n = 0;
        u32 size = 0;

        HostU32 row;
        HostU32 sizes;

        HostPartition ids;
        HostWeight weights;
    };

    inline HostBlockConn to_host_block_conn(const BlockConn &device_bc, DeviceExecutionSpace &exec_space) {
        HostBlockConn host_bc;

        host_bc.n = device_bc.n;
        host_bc.size = device_bc.size;

        host_bc.row = HostU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "row"), device_bc.n + 1);
        host_bc.sizes = HostU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sizes"), device_bc.n);
        host_bc.ids = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ids"), device_bc.size);
        host_bc.weights = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weights"), device_bc.size);

        Kokkos::deep_copy(exec_space, host_bc.row, device_bc.row);
        Kokkos::deep_copy(exec_space, host_bc.sizes, device_bc.sizes);
        Kokkos::deep_copy(exec_space, host_bc.ids, device_bc.ids);
        Kokkos::deep_copy(exec_space, host_bc.weights, device_bc.weights);
        exec_space.fence();

        return host_bc;
    }
}

#endif //GPU_HEIPA_BLOCK_CONN_H
