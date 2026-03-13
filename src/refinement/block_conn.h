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

#include "../utility/definitions.h"
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

    inline BlockConn init_BlockConn(const Graph &g,
                                    const Partition &partition,
                                    KokkosMemoryStack &mem_stack,
                                    Kokkos::Cuda &exec_space
                                ) {
        BlockConn bc;
        //
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "allocate_rows");

            bc.n = g.n;
            bc.row = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
            bc.sizes = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * g.n), g.n);

            KOKKOS_PROFILE_FENCE();
        }

        // set rows
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "set_rows");

            Kokkos::parallel_scan("set_rows", 
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, g.n + 1),
                KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (i == 0) {
                    // first slot is 0
                    if (final) bc.row(0) = 0;
                    return;
                }

                const vertex_t u = i - 1;
                const u32 len = g.neighborhood(u + 1) - g.neighborhood(u);
                const u32 c = len < partition.k ? len : partition.k;

                // write inclusive row[i] = running + c
                if (final) {
                    bc.row(i) = running + c;
                    bc.sizes(u) = 0; // c;
                }

                running += c;
            });

            Kokkos::deep_copy(exec_space, bc.size, Kokkos::subview(bc.row, g.n));
            exec_space.fence();
            KOKKOS_PROFILE_FENCE();
        }

        // allocate rest
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "allocate");

            bc.ids = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * bc.size), bc.size);
            bc.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * bc.size), bc.size);
            Kokkos::deep_copy(exec_space, bc.ids, NULL_PART);
            Kokkos::deep_copy(exec_space, bc.weights, 0);
            exec_space.fence();
            KOKKOS_PROFILE_FENCE();
        }

        // first fill of the structure
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "fill");

            Kokkos::parallel_for("fill", 
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, g.m), 
                KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = g.edges_u(i);
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                u32 r_beg = bc.row(u);
                u32 r_end = bc.row(u + 1);
                u32 r_len = r_end - r_beg;

                partition_t v_id = partition.map(v);

                for (u32 j = 0; j < r_len; j++) {
                    u32 idx = r_beg + j; // ((v_id + j) % r_len);
                    partition_t val = bc.ids(idx);
                    if (val == v_id) {
                        Kokkos::atomic_add(&bc.weights(idx), w);
                        return;
                    }
                    if (val == NULL_PART) {
                        val = Kokkos::atomic_compare_exchange(&bc.ids(idx), NULL_PART, v_id);
                        if (val == NULL_PART) {
                            Kokkos::atomic_add(&bc.weights(idx), w);
                            Kokkos::atomic_inc(&bc.sizes(u));
                            return;
                        }
                        if (val == v_id) {
                            Kokkos::atomic_add(&bc.weights(idx), w);
                            return;
                        }
                    }
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        return bc;
    }

    inline void free_BlockConn(BlockConn &bc,
                               KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refinement", "BlockConnectivity", "free");

        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline void update_large(const Graph &g,
                             Partition &partition,
                             UnmanagedDeviceU32 &zeros,
                             UnmanagedDevicePartition &dest_cache,
                             BlockConn &bc,
                             const DeviceVertex &moves,
                             Kokkos::Cuda &exec_space
                            ) {
        u32 total_moves = (u32) moves.extent(0);

        Kokkos::parallel_for("mark", 
            Kokkos::TeamPolicy(exec_space, (int) total_moves, Kokkos::AUTO), 
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
            u32 i = (u32) t.league_rank();
            vertex_t u = moves(i);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 j) {
                vertex_t v = g.edges_v(j);
                zeros(v) = 1;
            });
        });

        //recompute conn tables for each vertex adjacent to a moved vertex
        Kokkos::parallel_for("rebuild", 
            Kokkos::TeamPolicy<>(exec_space, (int) g.n, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(partition.k * sizeof(weight_t) + partition.k * sizeof(partition_t) + 4 * sizeof(partition_t))), 
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
            vertex_t u = (vertex_t) t.league_rank();

            if (zeros(u) == 1) {
                u32 r_beg = bc.row(u);
                u32 r_end = bc.row(u + 1);
                u32 r_len = r_end - r_beg;

                // reset global memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const u32 i) {
                    bc.weights(r_beg + i) = 0;
                    bc.ids(r_beg + i) = NULL_PART;
                });

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
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [&](const u32 &i) {
                    vertex_t v = g.edges_v(i);
                    weight_t w = g.edges_w(i);
                    partition_t v_id = partition.map(v);
                    u32 idx = (u32) v_id % r_len;

                    if (r_len == (u32) partition.k) {
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
                            idx = (idx + 1) % r_len;
                        }
                    }
                    Kokkos::atomic_add(s_weights + idx, w);
                });
                t.team_barrier();

                u32 old_size = r_len;
                u32 new_size = *n_needed_slots + ((*n_needed_slots / 4) < 3 ? 3 : (*n_needed_slots / 4));

                if (new_size < old_size) {
                    bc.sizes(u) = new_size;
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, old_size), [&](const u32 &i) {
                        partition_t id = s_ids[i];
                        if (id != NULL_PART) {
                            u32 idx = (u32) id % new_size;

                            while (true) {
                                partition_t found_id = Kokkos::atomic_compare_exchange(&bc.ids(r_beg + idx), NULL_PART, id);
                                if (found_id == NULL_PART || found_id == id) { break; }
                                idx = (idx + 1) % new_size;
                            }

                            bc.weights(r_beg + idx) = s_weights[i];
                        }
                    });
                } else {
                    bc.sizes(u) = old_size;
                    //copy conn table into global memory
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, old_size), [&](const u32 i) {
                        bc.weights(r_beg + i) = s_weights[i];
                        bc.ids(r_beg + i) = s_ids[i];
                    });
                }

                // reset cache and memory
                Kokkos::single(Kokkos::PerTeam(t), [=]() {
                    zeros(u) = 0;
                    dest_cache(u) = NULL_PART;
                });
            }
        });
    }

    inline void update_small(const Graph &g,
                             Partition &partition,
                             UnmanagedDevicePartition &dest_part,
                             UnmanagedDevicePartition &dest_cache,
                             BlockConn &bc,
                             const DeviceVertex &moves,
                             Kokkos::Cuda &exec_space
                            ) {
        u32 total_moves = (u32) moves.extent(0);

        //
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "update_small_remove_weight");

            Kokkos::parallel_for("remove_weight", 
                Kokkos::TeamPolicy<>(exec_space, (int) total_moves, Kokkos::AUTO), 
                KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
                vertex_t u = moves((u32) t.league_rank());
                partition_t old_u_id = dest_part(u);

                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 i) {
                    vertex_t v = g.edges_v(i);
                    weight_t w = g.edges_w(i);

                    u32 r_beg = bc.row(v);
                    u32 r_len = bc.sizes(v);

                    // find correct idx
                    partition_t idx = old_u_id % r_len;
                    while (bc.ids(r_beg + idx) != old_u_id) { idx = (idx + 1) % r_len; }

                    // remove weight
                    weight_t id_w = Kokkos::atomic_fetch_add(&bc.weights(r_beg + idx), -w);

                    if (r_len != partition.k && id_w == w) { bc.ids(r_beg + idx) = HASH_RECLAIM; }
                });
            });

            KOKKOS_PROFILE_FENCE();
        }

        //
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "update_small_add_weight");

            Kokkos::parallel_for("add_weight", 
                Kokkos::TeamPolicy<>(exec_space, (int) total_moves, Kokkos::AUTO), 
                KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
                vertex_t u = moves((u32) t.league_rank());
                partition_t new_u_id = partition.map(u);

                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 i) {
                    vertex_t v = g.edges_v(i);
                    weight_t w = g.edges_w(i);

                    dest_cache(v) = NULL_PART; // reset the cache

                    u32 r_beg = bc.row(v);
                    u32 r_len = bc.sizes(v);

                    u32 idx = new_u_id % r_len;

                    // first pass look for new_u_id
                    bool success = false;
                    for (u32 j = 0; j < r_len; j++) {
                        idx = (new_u_id + j) % r_len;
                        partition_t id = bc.ids(r_beg + idx);

                        if (id == new_u_id) {
                            success = true;
                            break;
                        }
                        if (id == NULL_PART) { break; }
                    }

                    if (!success) {
                        for (u32 j = 0; j < r_len; j++) {
                            idx = (new_u_id + j) % r_len;
                            partition_t id = bc.ids(r_beg + idx);

                            if (id == new_u_id) {
                                success = true;
                                break;
                            }

                            if (id == NULL_PART || id == HASH_RECLAIM) {
                                partition_t found_id = Kokkos::atomic_compare_exchange(&bc.ids(r_beg + idx), id, new_u_id);
                                if (found_id == new_u_id || found_id == NULL_PART || found_id == HASH_RECLAIM) {
                                    success = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (!success) {
                        idx = r_len;
                        while (true) {
                            partition_t id = bc.ids(r_beg + idx);

                            if (id == new_u_id) {
                                success = true;
                                break;
                            }

                            if (id == NULL_PART || id == HASH_RECLAIM) {
                                partition_t found_id = Kokkos::atomic_compare_exchange(&bc.ids(r_beg + idx), id, new_u_id);
                                if (found_id == id) {
                                    Kokkos::atomic_add(&bc.sizes(v), 1);
                                    break;
                                }
                                if (found_id == new_u_id) { break; }
                            }

                            idx++;
                        }
                    }
                    Kokkos::atomic_add(&bc.weights(r_beg + idx), w);
                });
            });

            KOKKOS_PROFILE_FENCE();
        }
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
}

#endif //GPU_HEIPA_BLOCK_CONN_H
