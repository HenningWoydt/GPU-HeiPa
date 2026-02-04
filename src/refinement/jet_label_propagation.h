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

#ifndef GPU_HEIPA_JET_LABEL_PROPAGATION_H
#define GPU_HEIPA_JET_LABEL_PROPAGATION_H

#include <limits>
#include <Kokkos_Core.hpp>

#include "../utility/definitions.h"

namespace GPU_HeiPa {
    constexpr u32 N_MAX_ITERATIONS = 12;
    constexpr u32 N_MAX_WEAK_ITERATIONS = 2;
    constexpr f64 PHI = 0.999;
    constexpr f64 HEAVY_ALPHA = 1.5;

    constexpr vertex_t MAX_SECTIONS = 128;
    constexpr int MAX_BUCKETS = 50;
    constexpr int MID_BUCKETS = 25;

    struct LabelPropagation {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;
        vertex_t min_size = 0;

        Partition partition{};

        UnmanagedDeviceWeight gain1, temp_gain, gain_cache, evict_start, evict_adjust;
        UnmanagedDeviceVertex vtx1, vtx2;
        UnmanagedDevicePartition dest_part, underloaded_blocks;
        UnmanagedDeviceU32 zeros;

        DeviceScalarU32 idx;

        HostScalarPinnedVertex scan_host;
        HostScalarPinnedWeight cut_change1, cut_change2, max_part;
        HostPinnedWeight reduce_locs;
        DeviceScalarPartition n_underloaded_blocks;
        DeviceScalarWeight max_vwgt;
    };

    inline LabelPropagation initialize_label_propagation(const vertex_t t_n,
                                                         const vertex_t t_m,
                                                         const partition_t t_k,
                                                         const weight_t t_lmax,
                                                         KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "allocate");

        LabelPropagation lp;

        lp.n = t_n;
        lp.m = t_m;
        lp.k = t_k;
        lp.lmax = t_lmax;
        lp.min_size = t_k * MAX_SECTIONS * MAX_BUCKETS;

        lp.partition = initialize_partition(t_n, t_k, t_lmax, mem_stack);

        lp.gain1 = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * std::max(lp.n, lp.min_size)), std::max(lp.n, lp.min_size));
        lp.temp_gain = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * lp.n), lp.n);
        lp.gain_cache = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * lp.n), lp.n);
        lp.evict_start = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * (lp.k + 1)), (lp.k + 1));
        lp.evict_adjust = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * lp.k), lp.k);

        lp.vtx1 = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * lp.n), lp.n);
        lp.vtx2 = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * std::max(lp.n, lp.min_size)), std::max(lp.n, lp.min_size));

        lp.dest_part = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * lp.n), lp.n);
        lp.underloaded_blocks = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * lp.k), lp.k);

        lp.zeros = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * lp.n), lp.n);
        Kokkos::deep_copy(lp.zeros, 0);

        lp.idx = DeviceScalarU32("idx");

        lp.scan_host = HostScalarPinnedVertex("scan host");
        lp.n_underloaded_blocks = DeviceScalarPartition("total undersized");
        lp.max_vwgt = DeviceScalarWeight("max vwgt allowed");
        lp.reduce_locs = HostPinnedWeight("reduce to here", 3);
        lp.cut_change1 = Kokkos::subview(lp.reduce_locs, 0);
        lp.cut_change2 = Kokkos::subview(lp.reduce_locs, 1);
        lp.max_part = Kokkos::subview(lp.reduce_locs, 2);

        return lp;
    }

    inline void free_LabelPropagation(const LabelPropagation &lp,
                                      KokkosMemoryStack &mem_stack) {
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

        free_partition(lp.partition, mem_stack);
    }

    inline weight_t get_max_weight(const UnmanagedDeviceWeight &ps,
                                   partition_t k,
                                   HostScalarPinnedWeight &result) {
        Kokkos::parallel_reduce("get max part size (store in view)", k, KOKKOS_LAMBDA(const s32 i, weight_t &update) {
            if (ps(i) > update) {
                update = ps(i);
            }
        }, Kokkos::Max(result));
        Kokkos::fence();

        return result();
    }

    struct BlockConn {
        vertex_t n = 0;
        u32 size = 0;

        UnmanagedDeviceU32 row;
        UnmanagedDeviceU32 sizes;

        UnmanagedDevicePartition ids;
        UnmanagedDeviceWeight weights;

        UnmanagedDeviceU32 lock;
        UnmanagedDevicePartition dest_cache;
    };

    inline BlockConn init_BlockConn(const LabelPropagation &lp,
                                    const Graph &g,
                                    KokkosMemoryStack &mem_stack) {
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

            Kokkos::parallel_scan("set_rows", g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (i == 0) {
                    // first slot is 0
                    if (final) bc.row(0) = 0;
                    return;
                }

                const vertex_t u = i - 1;
                const u32 len = g.neighborhood(u + 1) - g.neighborhood(u);
                const u32 c = len < lp.partition.k ? len : lp.partition.k;

                // write inclusive row[i] = running + c
                if (final) {
                    bc.row(i) = running + c;
                    bc.sizes(u) = 0; // c;
                }

                running += c;
            });
            Kokkos::deep_copy(bc.size, Kokkos::subview(bc.row, g.n));

            KOKKOS_PROFILE_FENCE();
        }

        // allocate rest
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "allocate");

            bc.ids = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * bc.size), bc.size);
            bc.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * bc.size), bc.size);
            Kokkos::deep_copy(DeviceExecutionSpace(), bc.ids, NULL_PART);
            Kokkos::deep_copy(DeviceExecutionSpace(), bc.weights, 0);

            bc.lock = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * bc.n), bc.n);
            bc.dest_cache = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * bc.n), bc.n);
            Kokkos::deep_copy(DeviceExecutionSpace(), bc.lock, 0);
            Kokkos::deep_copy(DeviceExecutionSpace(), bc.dest_cache, NULL_PART);

            KOKKOS_PROFILE_FENCE();
        }

        // first fill of the structure - around 50% faster than Jets
        ScopedTimer _t("refinement", "BlockConnectivity_fs", "fill");

        Kokkos::parallel_for("fill", g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = g.edges_u(i);
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);

            u32 r_beg = bc.row(u);
            u32 r_end = bc.row(u + 1);

            partition_t v_id = lp.partition.map(v);

            for (u32 j = r_beg; j < r_end; j++) {
                partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), NULL_PART, v_id);
                if (val == NULL_PART) {
                    Kokkos::atomic_add(&bc.weights(j), w);
                    Kokkos::atomic_inc(&bc.sizes(u));
                    return;
                }
                if (val == v_id) {
                    Kokkos::atomic_add(&bc.weights(j), w);
                    return;
                }
            }
        });
        KOKKOS_PROFILE_FENCE();

        return bc;
    }

    inline void free_BlockConn(BlockConn &bc,
                               KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refinement", "BlockConnectivity", "free");

        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline void update_large(const LabelPropagation &lp, const Graph &g, const BlockConn &bc, const DeviceVertex &moves) {
        u32 total_moves = (u32) moves.extent(0);

        Kokkos::parallel_for("mark", Kokkos::TeamPolicy<DeviceExecutionSpace>((int) total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            u32 i = (u32) t.league_rank();
            vertex_t u = moves(i);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 j) {
                vertex_t v = g.edges_v(j);
                lp.zeros(v) = 1;
            });
        });

        //recompute conn tables for each vertex adjacent to a moved vertex
        Kokkos::parallel_for("rebuild", Kokkos::TeamPolicy<DeviceExecutionSpace>((int) g.n, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(lp.k * sizeof(weight_t) + lp.k * sizeof(partition_t) + 4 * sizeof(partition_t))), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            vertex_t u = (vertex_t) t.league_rank();

            if (lp.zeros(u) == 1) {
                u32 r_beg = bc.row(u);
                u32 r_end = bc.row(u + 1);
                u32 r_len = r_end - r_beg;

                // reset global memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const u32 i) { bc.weights(r_beg + i) = 0; });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const u32 i) { bc.ids(r_beg + i) = NULL_PART; });

                // build the row
                weight_t *s_weights = (weight_t *) t.team_shmem().get_shmem(sizeof(weight_t) * r_len);
                partition_t *s_ids = (partition_t *) t.team_shmem().get_shmem(sizeof(partition_t) * r_len);
                u32 *n_needed_slots = (u32 *) t.team_shmem().get_shmem(sizeof(u32));

                // reset weights and ids
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const vertex_t j) { s_weights[j] = 0; });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, r_len), [&](const vertex_t j) { s_ids[j] = NULL_PART; });
                *n_needed_slots = 0;
                t.team_barrier();

                // construct conn table from scratch in shared memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [&](const u32 &i) {
                    vertex_t v = g.edges_v(i);
                    weight_t w = g.edges_w(i);
                    partition_t v_id = lp.partition.map(v);
                    u32 idx = (u32) v_id % r_len;

                    if (r_len == (u32) lp.k) {
                        if (NULL_PART == Kokkos::atomic_compare_exchange(s_ids + idx, NULL_PART, v_id)) { Kokkos::atomic_add(n_needed_slots, 1); }
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
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, old_size), [&](const u32 i) { bc.weights(r_beg + i) = s_weights[i]; });
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, old_size), [&](const u32 i) { bc.ids(r_beg + i) = s_ids[i]; });
                }

                // reset cache and memory
                Kokkos::single(Kokkos::PerTeam(t), [=]() {
                    lp.zeros(u) = 0;
                    bc.dest_cache(u) = NULL_PART;
                });
            }
        });
    }

    inline void update_small(const LabelPropagation &lp, const Graph &g, const BlockConn &bc, const DeviceVertex &moves) {
        u32 total_moves = (u32) moves.extent(0);

        Kokkos::parallel_for("remove_weight", Kokkos::TeamPolicy<DeviceExecutionSpace>((int) total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            vertex_t u = moves((u32) t.league_rank());
            partition_t old_u_id = lp.dest_part(u);

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

                if (r_len != lp.k && id_w == w) { bc.ids(r_beg + idx) = HASH_RECLAIM; }
            });
        });

        Kokkos::parallel_for("add_weight", Kokkos::TeamPolicy<DeviceExecutionSpace>((int) total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            vertex_t u = moves((u32) t.league_rank());
            partition_t new_u_id = lp.partition.map(u);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 i) {
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                bc.dest_cache(v) = NULL_PART; // reset the cache

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
    }

    KOKKOS_INLINE_FUNCTION
    vertex_t gain_bucket(const weight_t &gx, const weight_t &vwgt) {
        //cast to float so we can approximate log_1.5
        f64 gain = (f64) gx / (f64) vwgt;
        vertex_t gain_type = 0;
        if (gain > 0.0) {
            gain_type = 0;
        } else if (gain == 0.0) {
            gain_type = 1;
        } else {
            gain_type = MID_BUCKETS;
            gain = abs(gain);
            if (gain < 1.0) {
                while (gain < 1.0) {
                    gain *= 1.5;
                    gain_type -= 1;
                }
                if (gain_type < 2) {
                    gain_type = 2;
                }
            } else {
                while (gain > 1.0) {
                    gain /= 1.5;
                    gain_type += 1;
                }
                if (gain_type > MAX_BUCKETS) {
                    gain_type = MAX_BUCKETS - 1;
                }
            }
        }
        return gain_type;
    }

    inline DeviceVertex jet_lp(LabelPropagation &lp,
                               const Graph &g,
                               const BlockConn &bc,
                               f64 conn_c) {
        vertex_t num_pos = 0;
        //
        {
            ScopedTimer _t("refinement", "jetlp", "best_block");
            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("best_block", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t temp_u_dest_part;
                weight_t temp_u_gain_cache;

                if (bc.dest_cache(u) != NULL_PART) {
                    // cached for this vertex
                    lp.dest_part(u) = bc.dest_cache(u);

                    temp_u_dest_part = bc.dest_cache(u);
                    temp_u_gain_cache = lp.gain_cache(u);
                } else {
                    partition_t u_id = lp.partition.map(u);
                    partition_t best_id = NO_MOVE;
                    weight_t best_conn = 0;
                    weight_t own_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        if (id == NULL_PART || id == HASH_RECLAIM) { continue; } // no valid entry

                        weight_t w = bc.weights(i);

                        if (id == u_id) {
                            own_conn = w;
                            continue;
                        }

                        if (w > best_conn) {
                            best_id = id;
                            best_conn = w;
                        }
                    }

                    weight_t gain = 0;

                    if (best_id != NO_MOVE) {
                        if (best_conn >= own_conn || ((own_conn - best_conn) < floor(conn_c * own_conn))) {
                            gain = best_conn - own_conn;
                        } else {
                            best_id = NO_MOVE;
                        }
                    }
                    lp.gain_cache(u) = gain;
                    bc.dest_cache(u) = best_id;
                    lp.dest_part(u) = best_id;

                    temp_u_dest_part = best_id;
                    temp_u_gain_cache = gain;
                }

                if (temp_u_dest_part != NO_MOVE && bc.lock(u) == 0) {
                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx1(idx) = u;
                    lp.gain1(u) = temp_u_gain_cache;
                } else {
                    lp.gain1(u) = GAIN_MIN;
                    bc.lock(u) = 0;
                }
            });

            Kokkos::fence();
            u32 t;
            Kokkos::deep_copy(t, lp.idx);
            num_pos = (vertex_t) t;

            KOKKOS_PROFILE_FENCE();
        }

        // use afterburner
        {
            ScopedTimer _t("refinement", "jetlp", "afterburner");
            Kokkos::deep_copy(lp.idx, 0);
            Kokkos::parallel_for("afterburner heuristic", num_pos, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.vtx1(i);
                weight_t u_gain = lp.gain1(u);
                partition_t old_u_id = lp.partition.map(u);
                partition_t new_u_id = lp.dest_part(u);

                weight_t change = 0;
                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                    vertex_t v = g.edges_v(j);
                    weight_t v_gain = lp.gain1(v);

                    if (v_gain > u_gain || (v_gain == u_gain && v < u)) {
                        partition_t v_new_id = lp.dest_part(v);
                        partition_t v_old_id = lp.partition.map(v);
                        weight_t w = g.edges_w(j);

                        if (v_new_id == old_u_id) { change -= w; } else if (v_new_id == new_u_id) { change += w; }
                        if (v_old_id == old_u_id) { change += w; } else if (v_old_id == new_u_id) { change -= w; }
                    }
                }
                if (u_gain + change >= 0) {
                    bc.lock(u) = 1;
                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx2(idx) = u;
                }
            });
            Kokkos::fence();
            u32 t;
            Kokkos::deep_copy(t, lp.idx);
            num_pos = (vertex_t) t;

            KOKKOS_PROFILE_FENCE();
        }
        return Kokkos::subview(lp.vtx2, std::make_pair((vertex_t) 0, num_pos));
    }

    inline DeviceVertex rebalance_strong(LabelPropagation &lp, const Graph &g, const BlockConn &bc) {
        weight_t opt_weight = (g.g_weight + (weight_t) (lp.k - 1)) / (weight_t) lp.k;
        weight_t max_b_w = std::max(opt_weight + 1, (weight_t) ((f64) lp.lmax * 0.99));

        vertex_t sections = MAX_SECTIONS;
        vertex_t section_size = (g.n + sections * lp.k) / (sections * lp.k);
        if (section_size < 4096) {
            section_size = 4096;
            sections = (g.n + section_size * lp.k) / (section_size * lp.k);
        }
        vertex_t t_minibuckets = MAX_BUCKETS * lp.k * sections;
        vertex_t width = MAX_BUCKETS * sections;

        //use minibuckets within each gain bucket to reduce atomic contention
        //because the number of gain buckets is small
        Kokkos::deep_copy(Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);

        // Determine maximum allowed vertex weight
        {
            ScopedTimer _t("refinement", "jetrs", "find_max_vwgt");

            Kokkos::parallel_reduce("find max size", lp.k, KOKKOS_LAMBDA(const partition_t id, weight_t &update) {
                weight_t size = lp.partition.bweights(id);
                if (size < max_b_w) {
                    weight_t cap = max_b_w - size;
                    if (cap > update) {
                        update = cap;
                    }
                }
            }, Kokkos::Max(lp.max_vwgt));

            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("refinement", "jetrs", "score_candidates");

            Kokkos::parallel_for("assign move scores part1", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                lp.vtx2(u) = NO_BLOCK_ID;

                if (lp.partition.bweights(u_id) > lp.lmax && g.weights(u) <= 2 * lp.max_vwgt() && g.weights(u) < 2 * (lp.partition.bweights(u_id) - opt_weight)) {
                    weight_t own_conn = 0;
                    weight_t count = 0;
                    weight_t sum_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;
                    for (vertex_t i = r_beg; i < r_end; i++) {
                        partition_t id = bc.ids(i);
                        if (id == u_id) {
                            own_conn = bc.weights(i);
                            continue;
                        }
                        if (id != NULL_PART && id != HASH_RECLAIM && lp.partition.bweights(id) < max_b_w) {
                            sum_conn += bc.weights(i);
                            count += 1;
                        }
                    }

                    if (count == 0) count = 1;
                    weight_t gain = (sum_conn / count) - own_conn;
                    vertex_t gain_type = gain_bucket(gain, Kokkos::min(g.weights(u), lp.partition.bweights(u_id) - lp.lmax));

                    //add to count of appropriate bucket
                    if (gain_type < MAX_BUCKETS) {
                        vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections) + 1;
                        lp.vtx2(u) = g_id;
                        lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.gain1(g_id), g.weights(u));
                    }
                }
            });

            KOKKOS_PROFILE_FENCE();
        }

        //
        {
            ScopedTimer _t("refinement", "jetrs", "scan_score_buckets");

            if (t_minibuckets < 10000) {
                Kokkos::parallel_for("scan score buckets", Kokkos::TeamPolicy<DeviceExecutionSpace>(1, 1024), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
                    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t &u, weight_t &update, const bool final) {
                        weight_t gain = lp.gain1(u);
                        if (final) {
                            lp.gain1(u) = update;
                        }
                        update += gain;
                    });
                });
            } else {
                Kokkos::parallel_scan("scan score buckets", Policy(0, t_minibuckets + 1), KOKKOS_LAMBDA(const vertex_t &u, weight_t &update, const bool final) {
                    weight_t gain = lp.gain1(u);
                    if (final) {
                        lp.gain1(u) = update;
                    }
                    update += gain;
                });
            }
        }

        u32 num_moves = 0;
        //
        {
            ScopedTimer _t("refinement", "jetrs", "filter_scores");

            Kokkos::deep_copy(lp.evict_adjust, 0);
            Kokkos::parallel_scan("filter_scores", g.n, KOKKOS_LAMBDA(const u32 u, vertex_t &update, const bool final) {
                vertex_t b_id = lp.vtx2(u);
                if (b_id != NO_BLOCK_ID) {
                    partition_t u_id = lp.partition.map(u);
                    vertex_t begin_bucket = u_id * width;
                    weight_t score = lp.temp_gain(u) + lp.gain1(b_id) - lp.gain1(begin_bucket);
                    weight_t limit = lp.partition.bweights(u_id) - lp.lmax;

                    if (score < limit) {
                        if (final) {
                            if (score + g.weights(u) >= limit) {
                                lp.evict_adjust(u_id) = score + g.weights(u);
                            }
                            lp.vtx1(update) = u;
                        }
                        update++;
                    }
                }
            }, lp.scan_host);
            Kokkos::fence();

            num_moves = (u32) lp.scan_host();
        }

        // the rest of this method determines the destination part for each evicted vtx
        //assign consecutive chunks of vertices to undersized parts using scan result
        //
        {
            ScopedTimer _t("refinement", "jetrs", "cookie_cutter");

            Kokkos::parallel_for("cookie cutter", Kokkos::TeamPolicy<DeviceExecutionSpace>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp.k), [&](const partition_t p, weight_t &update, const bool final) {
                    weight_t add = lp.evict_adjust(p);
                    vertex_t begin_bucket = MAX_BUCKETS * p * sections;
                    if (add == 0) {
                        // evict_adjust(p) isn't set if there aren't enough evictions to balance part p
                        add = lp.gain1(begin_bucket + MAX_BUCKETS * sections) - lp.gain1(begin_bucket);
                    }
                    if (final) {
                        lp.evict_adjust(p) = lp.gain1(begin_bucket) - update;
                    }
                    update += add;
                    if (final && p + 1 == lp.k) {
                        lp.max_vwgt() = update;
                    }
                });
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp.k), [&](const partition_t p, weight_t &update, const bool final) {
                    if (final && p == 0) {
                        lp.evict_start(0) = 0;
                    }
                    if (max_b_w > lp.partition.bweights(p)) {
                        update += max_b_w - lp.partition.bweights(p);
                    }
                    if (final) {
                        lp.evict_start(p + 1) = update;
                    }
                });
            });

            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("refinement", "jetrs", "adjust_scores");

            Kokkos::parallel_for("adjust_scores", num_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.vtx1(i);
                partition_t u_id = lp.partition.map(u);
                vertex_t b_id = lp.vtx2(u);
                weight_t score = lp.temp_gain(u) + lp.gain1(b_id) - lp.evict_adjust(u_id);

                lp.temp_gain(u) = score;
                s32 id = 0;
                while (id < (s32) lp.k) {
                    //find chunk that contains i
                    while (id <= (s32) lp.k && lp.evict_start(id) <= lp.temp_gain(u)) {
                        id++;
                    }
                    id--;
                    if (id < (s32) lp.k && g.weights(u) / 2 <= lp.evict_start(id + 1) - lp.temp_gain(u)) {
                        // at least half of vtx weight lies in chunk p
                        lp.dest_part(u) = (partition_t) id;
                        return;
                    }
                    if (id < (s32) lp.k) {
                        lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.max_vwgt(), g.weights(u));
                    }
                }
                lp.dest_part(u) = lp.partition.map(u);
            });

            KOKKOS_PROFILE_FENCE();
        }

        return Kokkos::subview(lp.vtx1, std::make_pair((u32) 0, num_moves));
    }

    inline DeviceVertex rebalance_weak(LabelPropagation &lp, Graph &g, const BlockConn &bc) {
        weight_t opt_weight = (g.g_weight + (weight_t) (lp.k - 1)) / (weight_t) lp.k;
        weight_t max_b_w = (weight_t) ((f64) lp.lmax * 0.99);
        if (max_b_w < lp.lmax - 100) { max_b_w = lp.lmax - 100; }

        vertex_t sections = MAX_SECTIONS;
        vertex_t section_size = (g.n + sections * lp.k) / (sections * lp.k);
        if (section_size < 4096) {
            section_size = 4096;
            sections = (g.n + section_size * lp.k) / (section_size * lp.k);
        }
        vertex_t t_minibuckets = MAX_BUCKETS * lp.k * sections;
        vertex_t width = MAX_BUCKETS * sections;

        // determine underloaded blocks
        {
            ScopedTimer _t("refinement", "jetrw", "underloaded_blocks");

            Kokkos::parallel_for("init undersized parts list", Kokkos::TeamPolicy<DeviceExecutionSpace>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
                //this scan is small so do it within a team instead of an entire grid to save kernel launch time
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp.k), [&](const partition_t i, partition_t &update, const bool final) {
                    if (lp.partition.bweights(i) < max_b_w) {
                        if (final) {
                            lp.underloaded_blocks(update) = i;
                        }
                        update++;
                    }
                    if (final && i + 1 == lp.k) {
                        lp.n_underloaded_blocks() = update;
                    }
                });
            });

            KOKKOS_PROFILE_FENCE();
        }

        // determine best block
        {
            ScopedTimer _t("refinement", "jetrw", "best_block");

            Kokkos::parallel_for("select destination parts (rw)", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);

                if (lp.partition.bweights(u_id) > lp.lmax && g.weights(u) < 1.5 * (lp.partition.bweights(u_id) - opt_weight)) {
                    partition_t best_id = u_id;
                    weight_t best_id_w = 0;
                    weight_t own_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;
                    for (vertex_t j = r_beg; j < r_end; j++) {
                        partition_t id = bc.ids(j);
                        weight_t w = bc.weights(j);

                        if (id != NULL_PART && id != HASH_RECLAIM && lp.partition.bweights(id) < max_b_w) {
                            if (w > best_id_w) {
                                best_id = id;
                                best_id_w = w;
                            }
                        }
                        if (id == u_id) {
                            own_conn = bc.weights(j);
                        }
                    }
                    if (best_id_w > 0) {
                        lp.dest_part(u) = best_id;
                        lp.temp_gain(u) = best_id_w - own_conn;
                    } else {
                        //choose arbitrary undersized part
                        best_id = lp.underloaded_blocks(u % lp.n_underloaded_blocks());
                        lp.dest_part(u) = best_id;
                        lp.temp_gain(u) = -own_conn;
                    }
                } else {
                    lp.dest_part(u) = u_id;
                }
            });

            KOKKOS_PROFILE_FENCE();
        }

        //
        {
            ScopedTimer _t("refinement", "jetrw", "assign_move_scores");

            Kokkos::deep_copy(Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);

            Kokkos::parallel_for("assign move scores", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                partition_t best_id = lp.dest_part(u);
                lp.vtx2(u) = NO_BLOCK_ID;
                if (u_id != best_id) {
                    weight_t gain = lp.temp_gain(u);
                    vertex_t gain_type = gain_bucket(gain, g.weights(u));
                    vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections);
                    lp.vtx2(u) = g_id;
                    lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.gain1(g_id), g.weights(u));
                }
            });

            KOKKOS_PROFILE_FENCE();
        }

        //
        {
            ScopedTimer _t("refinement", "jetrw", "scan_score_buckets");

            if (t_minibuckets < 10000) {
                Kokkos::parallel_for("scan score buckets", Kokkos::TeamPolicy<DeviceExecutionSpace>(1, 1024), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
                    //this scan is small so do it within a team instead of an entire grid to save kernel launch time
                    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t i, weight_t &update, const bool final) {
                        weight_t x = lp.gain1(i);
                        if (final) {
                            lp.gain1(i) = update;
                        }
                        update += x;
                    });
                });
            } else {
                Kokkos::parallel_scan("scan score buckets", t_minibuckets + 1, KOKKOS_LAMBDA(const vertex_t &i, weight_t &update, const bool final) {
                    weight_t x = lp.gain1(i);
                    if (final) {
                        lp.gain1(i) = update;
                    }
                    update += x;
                });
            }

            KOKKOS_PROFILE_FENCE();
        }

        //
        {
            ScopedTimer _t("refinement", "jetrw", "filter_scores");

            Kokkos::parallel_scan("filter scores below cutoff", g.n, KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
                vertex_t b = lp.vtx2(i);
                if (b != NO_BLOCK_ID) {
                    partition_t p = lp.partition.map(i);
                    vertex_t begin_bucket = p * width;
                    weight_t score = lp.temp_gain(i) + lp.gain1(b) - lp.gain1(begin_bucket);
                    weight_t limit = lp.partition.bweights(p) - lp.lmax;
                    if (score < limit) {
                        if (final) {
                            lp.vtx1(update) = i;
                        }
                        update++;
                    }
                }
            }, lp.scan_host);
            Kokkos::fence();
            KOKKOS_PROFILE_FENCE();
        }
        vertex_t num_moves = lp.scan_host();
        DeviceVertex only_moves = Kokkos::subview(lp.vtx1, std::make_pair((vertex_t) 0, num_moves));
        return only_moves;
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

    inline void perform_moves(LabelPropagation &lp,
                              const Graph &g,
                              const BlockConn &bc,
                              const DeviceVertex &moves,
                              weight_t &curr_max_weight,
                              weight_t &curr_cut) {
        u32 n_moves = (u32) moves.extent(0);

        // first change in cut
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "cut_change_1");

            Kokkos::parallel_reduce("cut_change_1", n_moves, KOKKOS_LAMBDA(const u32 &i, weight_t &gain_update) {
                vertex_t u = moves(i);
                partition_t old_id = lp.partition.map(u);
                partition_t new_id = lp.dest_part(u);

                u32 beg = bc.row(u);
                u32 len = bc.sizes(u);
                weight_t old_conn = lookup(bc.ids.data() + beg, bc.weights.data() + beg, old_id, len);
                weight_t new_conn = lookup(bc.ids.data() + beg, bc.weights.data() + beg, new_id, len);
                gain_update += new_conn - old_conn;
            }, lp.cut_change1);

            KOKKOS_PROFILE_FENCE();
        }

        // update mapping
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "update_mapping");

            Kokkos::parallel_for("update_mapping", n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = moves(i);
                partition_t old_id = lp.partition.map(u);
                partition_t new_id = lp.dest_part(u);

                bc.dest_cache(u) = NULL_PART;
                Kokkos::atomic_add(&lp.partition.bweights(old_id), -g.weights(u));
                Kokkos::atomic_add(&lp.partition.bweights(new_id), g.weights(u));

                lp.partition.map(u) = new_id;
                lp.dest_part(u) = old_id;
            });

            KOKKOS_PROFILE_FENCE();
        }

        // update block conn
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "update_block_conn");

            if (n_moves > (u32) g.n / 10) {
                update_large(lp, g, bc, moves);
            } else {
                update_small(lp, g, bc, moves);
            }

            KOKKOS_PROFILE_FENCE();
        }

        // second change in cut
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "cut_change_2");

            Kokkos::parallel_reduce("cut_change_2", n_moves, KOKKOS_LAMBDA(const vertex_t &i, weight_t &gain_update) {
                vertex_t u = moves(i);
                partition_t old_id = lp.dest_part(u);
                partition_t new_id = lp.partition.map(u);

                u32 beg = bc.row(u);
                u32 len = bc.sizes(u);
                weight_t p_con = lookup(bc.ids.data() + beg, bc.weights.data() + beg, old_id, len);
                weight_t b_con = lookup(bc.ids.data() + beg, bc.weights.data() + beg, new_id, len);
                gain_update += b_con - p_con;
            }, lp.cut_change2);

            KOKKOS_PROFILE_FENCE();
        }

        // update max weight
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "update_max_weight");

            curr_max_weight = get_max_weight(lp.partition.bweights, lp.k, lp.max_part);

            KOKKOS_PROFILE_FENCE();
        }

        // update cut
        curr_cut -= (lp.cut_change2() + lp.cut_change1()) / 2;
    }


    inline std::pair<weight_t, weight_t> jet_refine(Graph &g,
                                                    Partition &partition,
                                                    partition_t k,
                                                    weight_t lmax,
                                                    u32 level,
                                                    weight_t curr_edge_cut,
                                                    weight_t curr_max_weight,
                                                    KokkosMemoryStack &mem_stack) {
        LabelPropagation lp = initialize_label_propagation(g.n, g.m, k, lmax, mem_stack);

        // copy partition
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");
            copy_into(lp.partition, partition, g.n);
            KOKKOS_PROFILE_FENCE();
        }

        weight_t best_edge_cut = curr_edge_cut;
        weight_t best_max_weight = curr_max_weight;

        ASSERT(curr_edge_cut == get_total_cut(g, curr_partition.map) / 2);
        ASSERT(curr_weight == max_weight(curr_partition));

        BlockConn bc = init_BlockConn(lp, g, mem_stack);

        f64 filter_ratio = 0.75;
        if (level == 0) { filter_ratio = 0.25; }

        u32 balance_iteration = 0;
        u32 iteration = 0;
        while (iteration < N_MAX_ITERATIONS) {
            iteration += 1;

            DeviceVertex moves;
            if (curr_max_weight <= lmax) {
                moves = jet_lp(lp, g, bc, filter_ratio);
                balance_iteration = 0;
            } else {
                if (balance_iteration < N_MAX_WEAK_ITERATIONS) {
                    moves = rebalance_weak(lp, g, bc);
                } else {
                    moves = rebalance_strong(lp, g, bc);
                }
                balance_iteration++;
            }

            u32 n_moves = (u32) moves.extent(0);
            if (n_moves == 0) { continue; }

            perform_moves(lp, g, bc, moves, curr_max_weight, curr_edge_cut);

            if (best_max_weight > lmax && curr_max_weight < best_max_weight) {
                // copy the partition
                {
                    ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_edge_cut = curr_edge_cut;
                    best_max_weight = curr_max_weight;
                    iteration = 0;

                    KOKKOS_PROFILE_FENCE();
                }
            } else if (curr_edge_cut < best_edge_cut && (curr_max_weight <= lmax || curr_max_weight < best_max_weight)) {
                if ((f64) curr_edge_cut < PHI * (f64) best_edge_cut) { iteration = 0; }
                //
                {
                    ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_edge_cut = curr_edge_cut;
                    best_max_weight = curr_max_weight;

                    KOKKOS_PROFILE_FENCE();
                }
            }

            ASSERT(curr_edge_cut == get_total_cut(g, curr_partition.map) / 2);
            ASSERT(curr_weight == max_weight(curr_partition));
        }

        free_BlockConn(bc, mem_stack);
        free_LabelPropagation(lp, mem_stack);

        return std::make_pair(best_edge_cut, best_max_weight);
    }
}
#endif //GPU_HEIPA_JET_LABEL_PROPAGATION_H
