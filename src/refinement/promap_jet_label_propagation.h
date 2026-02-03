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

#ifndef GPU_HEIPA_PROMAP_JET_LABEL_PROPAGATION_H
#define GPU_HEIPA_PROMAP_JET_LABEL_PROPAGATION_H

#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/comm_cost.h"
#include "../utility/macros.h"
#include "../datastructures/partition.h"
#include "../datastructures/block_connectivity.h"
#include "jet_label_propagation.h"

namespace GPU_HeiPa {
    constexpr u32 TARGET_PER_SHARD = 4096;

    struct ProMapJetLabelPropagation {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        u32 round = 0;

        u32 n_max_iterations = 12;
        u32 max_weak_iterations = 2;
        f64 phi = 0.999;
        f64 heavy_alpha = 1.5; // smaller - fewer vertices moved, larger more vertices moved
        weight_t sigma = 10;
        f64 conn_c = 0.75;
        u32 sections = 1; // adaptive mini buckets per (part,bucket)
        Partition partition;

        UnmanagedDeviceWeight lp_gain;
        UnmanagedDevicePartition lp_id;
        UnmanagedDeviceU32 lp_to_check; // 0 means to check, 1 means put in list, 2 means checked but did not pass filter
        UnmanagedDeviceU32 to_move;

        UnmanagedDeviceWeight gain;
        UnmanagedDevicePartition id;

        UnmanagedDeviceU32 bid;
        UnmanagedDeviceWeight evict_start;
        UnmanagedDeviceWeight evict_adjust;
        Kokkos::View<weight_t, DeviceMemorySpace> max_vwgt;

        u32 temp_n_moves = 0;
        Kokkos::View<u32, DeviceMemorySpace> temp_moves_idx;
        UnmanagedDeviceVertex temp_moves;
        u32 n_moves = 0;
        Kokkos::View<u32, DeviceMemorySpace> moves_idx;
        UnmanagedDeviceVertex moves;

        UnmanagedDeviceWeight bucket_sizes;
        UnmanagedDeviceWeight bucket_offsets;

        Kokkos::View<u32, DeviceMemorySpace> n_underloaded_blocks;
        UnmanagedDevicePartition underloaded_blocks;

        UnmanagedDeviceWeight save_atomic;

        UnmanagedDevicePartition old_map;
    };

    inline ProMapJetLabelPropagation initialize_promap_lp(const vertex_t t_n,
                                                          const vertex_t t_m,
                                                          const partition_t t_k,
                                                          const weight_t t_lmax,
                                                          KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "allocate");

        ProMapJetLabelPropagation lp;

        lp.n = t_n;
        lp.m = t_m;
        lp.k = t_k;
        lp.lmax = t_lmax;
        lp.sigma = (weight_t) (0.99 * (f64) lp.lmax);
        if (lp.sigma < lp.lmax - 100) lp.sigma = lp.lmax - 100;

        lp.partition = initialize_partition(t_n, t_k, t_lmax, mem_stack);

        lp.lp_gain = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n), t_n);
        lp.lp_id = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * t_n), t_n);
        lp.lp_to_check = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n), t_n);
        lp.to_move = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n), t_n);

        lp.gain = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n), t_n);
        lp.id = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * t_n), t_n);

        lp.bid = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n), t_n);
        lp.evict_start = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * (t_k + 1)), t_k + 1);
        lp.evict_adjust = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_k), t_k);
        lp.max_vwgt = Kokkos::View<weight_t, DeviceMemorySpace>("max_vwgt");

        lp.temp_moves_idx = Kokkos::View<u32, DeviceMemorySpace>("temp_moves_idx");
        lp.temp_moves = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * t_n), t_n);
        lp.moves_idx = Kokkos::View<u32, DeviceMemorySpace>("moves_idx");
        lp.moves = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * t_n), t_n);

        // compute sections so that k * MAX_BUCKETS * sections ≈ n / TARGET_PER_SHARD
        u64 shards = (t_n + TARGET_PER_SHARD - 1) / TARGET_PER_SHARD;
        u64 per_part = MAX_BUCKETS; // buckets per part
        u64 s = (shards + (t_k * per_part - 1)) / (t_k * per_part);
        lp.sections = s == 0 ? 1 : (u32) s;

        size_t t_minibuckets = (size_t) t_k * MAX_BUCKETS * lp.sections;
        lp.bucket_sizes = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * (t_minibuckets + 1)), t_minibuckets + 1);
        lp.bucket_offsets = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * (t_minibuckets + 1)), t_minibuckets + 1);

        lp.n_underloaded_blocks = Kokkos::View<u32, DeviceMemorySpace>("n_underloaded_blocks");
        lp.underloaded_blocks = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * t_k), t_k);

        lp.save_atomic = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n), t_n);

        lp.old_map = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * t_n), t_n);

        return lp;
    }

    inline void free_promap_lp(ProMapJetLabelPropagation &lp,
                               KokkosMemoryStack &mem_stack) {
        free_partition(lp.partition, mem_stack);

        ScopedTimer _t("refinement", "JetLabelPropagation", "free");

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
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    template<typename d_oracle_t>
    inline void promap_jetlp(ProMapJetLabelPropagation &lp,
                             Graph &g,
                             BlockConnectivity &bc,
                             d_oracle_t &d_oracle) {
        lp.round += 1;

        // for each vertex determine best block
        {
            ScopedTimer _t("refinement", "jetlp", "best_block");
            Kokkos::deep_copy(lp.temp_moves_idx, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (lp.to_move(u) != 0) { return; } // already moved

                u32 status = lp.lp_to_check(u);
                if (status == 1) {
                    // have cached and is good, insert into list
                    u32 idx = Kokkos::atomic_fetch_add(&lp.temp_moves_idx(), 1);
                    lp.temp_moves(idx) = u;
                    return;
                }
                if (status == 2) { return; } // have cached, but bad

                // not cached need to find the best move

                partition_t u_id = lp.partition.map(u);

                partition_t best_id = u_id;
                weight_t best_delta = -max_sentinel<weight_t>();

                u32 r_beg = bc.row(u);
                u32 r_len = bc.sizes(u);
                u32 r_end = r_beg + r_len;

                for (u32 i = r_beg; i < r_end; ++i) {
                    partition_t id = bc.ids(i);
                    if (id == u_id) { continue; }      // dont move into self
                    if (id == NULL_PART) { continue; } // skip invalid

                    weight_t delta = 0;
                    for (u32 j = r_beg; j < r_end; ++j) {
                        partition_t temp_id = bc.ids(j);
                        weight_t w = bc.weights(j);

                        if (temp_id == NULL_PART) { continue; } // skip invalid

                        weight_t old_d = get(d_oracle, u_id, temp_id);
                        weight_t new_d = get(d_oracle, id, temp_id);

                        delta += w * (old_d - new_d);
                    }

                    if (delta > best_delta) {
                        best_delta = delta;
                        best_id = id;
                    }
                }

                status = 2; // first assume is bad
                if (best_delta >= 0) {
                    status = 1; // passed filter so is good

                    u32 idx = Kokkos::atomic_fetch_add(&lp.temp_moves_idx(), 1);
                    lp.temp_moves(idx) = u;
                }

                lp.lp_to_check(u) = status;
                lp.lp_id(u) = best_id;
                lp.lp_gain(u) = best_delta;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // use afterburner
        {
            ScopedTimer _t("refinement", "jetlp", "afterburner");
            Kokkos::deep_copy(lp.temp_n_moves, lp.temp_moves_idx);
            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("afterburner", lp.temp_n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.temp_moves(i);
                weight_t u_gain = lp.lp_gain(u);

                partition_t old_u_id = lp.partition.map(u);
                partition_t new_u_id = lp.lp_id(u);

                weight_t update = 0;
                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                    vertex_t v = g.edges_v(j);
                    weight_t w = g.edges_w(j);

                    weight_t v_gain = lp.lp_gain(v);

                    if (v_gain > u_gain || (v_gain == u_gain && v < u)) {
                        // Assume v moves first
                        partition_t v_old_id = lp.partition.map(v);
                        partition_t v_new_id = lp.lp_id(v);

                        // distances BEFORE v moves
                        weight_t d_unew_vold = get(d_oracle, new_u_id, v_old_id);
                        weight_t d_uold_vold = get(d_oracle, old_u_id, v_old_id);

                        // distances AFTER v moves
                        weight_t d_unew_vnew = get(d_oracle, new_u_id, v_new_id);
                        weight_t d_uold_vnew = get(d_oracle, old_u_id, v_new_id);

                        // gains are (cost_before - cost_after)
                        weight_t before_u = d_uold_vold - d_unew_vold; // gain if only u moves
                        weight_t after_u = d_uold_vnew - d_unew_vnew;  // gain if v moved first

                        update += w * (after_u - before_u);
                    }
                }

                if (u_gain + update >= 0) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.moves_idx(), 1);
                    lp.moves(idx) = u;
                    lp.to_move(u) = lp.round;
                    lp.id(u) = new_u_id;
                }
            });

            Kokkos::deep_copy(lp.n_moves, lp.moves_idx);

            KOKKOS_PROFILE_FENCE();
        }
    }

    template<typename d_oracle_t>
    inline void promap_jetrw(ProMapJetLabelPropagation &lp,
                             Graph &g,
                             BlockConnectivity &bc,
                             d_oracle_t &d_oracle) {
        lp.round += 1;

        weight_t opt_weight = (weight_t) ((f64) g.g_weight / (f64) lp.k);
        weight_t max_b_w = (weight_t) (lp.lmax * 0.99);
        if (max_b_w < lp.lmax - 100) { max_b_w = lp.lmax - 100; }
        u32 t_minibuckets = lp.k * MAX_BUCKETS * lp.sections;
        u32 width = MAX_BUCKETS * lp.sections;

        // determine underloaded blocks
        {
            ScopedTimer _t("refinement", "jetrw", "underloaded_blocks");
            Kokkos::deep_copy(lp.n_underloaded_blocks, 0);

            Kokkos::parallel_for("underloaded_blocks", lp.k, KOKKOS_LAMBDA(const partition_t id) {
                if (lp.partition.bweights(id) < max_b_w) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.n_underloaded_blocks(), 1);
                    lp.underloaded_blocks(idx) = id;
                }
            });

            Kokkos::deep_copy(lp.n_moves, lp.n_underloaded_blocks);
            KOKKOS_PROFILE_FENCE();
        }

        if (lp.n_moves == 0) { return; }

        // determine best block
        {
            ScopedTimer _t("refinement", "jetrw", "best_block");
            Kokkos::deep_copy(lp.temp_moves_idx, 0);
            Kokkos::deep_copy(lp.bucket_sizes, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                weight_t u_id_w = lp.partition.bweights(u_id);
                weight_t u_w = g.weights(u);

                partition_t best_id = u_id;
                weight_t best_delta = -max_sentinel<weight_t>();

                lp.id(u) = u_id; // default move to self
                lp.bid(u) = (u32) -1;

                if (u_id_w <= lp.lmax) { return; }                                         // u_id not overloaded
                if ((f64) u_w >= lp.heavy_alpha * ((f64) u_id_w - opt_weight)) { return; } // vertex is too heavy

                u32 r_beg = bc.row(u);
                u32 r_len = bc.sizes(u);
                u32 r_end = r_beg + r_len;

                for (u32 i = r_beg; i < r_end; ++i) {
                    partition_t id = bc.ids(i);

                    if (u_id == id) { continue; } // dont move to same partition
                    if (id == NULL_PART) { continue; }
                    if (lp.partition.bweights(id) > lp.sigma) { continue; } // dont move to overloaded block
                    if (lp.partition.bweights(id) >= max_b_w) { continue; } // dont move into non underloaded partition

                    weight_t delta = 0;
                    for (u32 j = r_beg; j < r_end; ++j) {
                        partition_t temp_id = bc.ids(j);
                        weight_t w = bc.weights(j);

                        if (temp_id == NULL_PART) { continue; }

                        weight_t old_d = get(d_oracle, u_id, temp_id);
                        weight_t new_d = get(d_oracle, id, temp_id);

                        delta += w * (old_d - new_d);
                    }

                    if (delta > best_delta) {
                        best_delta = delta;
                        best_id = id;
                    }
                }


                if (best_delta <= 0) {
                    best_id = lp.underloaded_blocks(u % lp.n_underloaded_blocks());
                    best_delta = 0;
                    for (u32 j = r_beg; j < r_end; ++j) {
                        partition_t temp_id = bc.ids(j);
                        weight_t w = bc.weights(j);

                        if (temp_id == NULL_PART) { continue; }

                        weight_t old_d = get(d_oracle, u_id, temp_id);
                        weight_t new_d = get(d_oracle, best_id, temp_id);

                        best_delta += w * (old_d - new_d);
                    }
                }

                lp.bid(u) = (u32) -1;
                if (best_id != u_id) {
                    u32 b = gain_bucket(best_delta, u_w);
                    u32 g_id = (MAX_BUCKETS * u_id + b) * lp.sections + (u % lp.sections);

                    lp.bid(u) = g_id;
                    lp.save_atomic(u) = Kokkos::atomic_fetch_add(&lp.bucket_sizes(g_id), u_w); // prefix inside this bucket
                    lp.id(u) = best_id;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        // determine bucket offsets
        {
            ScopedTimer _t("refinement", "jetrw", "bucket_offsets");

            // prefix sum for bucket offsets
            Kokkos::parallel_scan("bucket_offsets", t_minibuckets + 1, KOKKOS_LAMBDA(const u32 i, weight_t &upd, const bool final_pass) {
                weight_t val = lp.bucket_sizes(i);
                if (final_pass) lp.bucket_offsets(i) = upd; // write old prefix
                upd += val;
            });
            KOKKOS_PROFILE_FENCE();
        }
        // pick prefix
        {
            ScopedTimer _t("refinement", "jetrs", "pick_evictions");

            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("jetrs_pick_evictions", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 b_id = lp.bid(u);
                if (b_id == (u32) -1) return; // not a candidate

                partition_t u_id = lp.partition.map(u);

                // Per-part bucket range start
                u32 p_begin = u_id * width;

                // Jet's score computation
                weight_t score = lp.save_atomic(u) + lp.bucket_offsets(b_id) - lp.bucket_offsets(p_begin);

                // Limit = overweight of part u_id
                weight_t limit = lp.partition.bweights(u_id) - lp.lmax;
                if (score < limit) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.moves_idx(), 1);
                    lp.moves(idx) = u;
                }
            });
            KOKKOS_PROFILE_FENCE();

            Kokkos::deep_copy(lp.n_moves, lp.moves_idx);
        }
    }

    template<typename d_oracle_t>
    inline void promap_jetrs(ProMapJetLabelPropagation &lp,
                             Graph &g,
                             BlockConnectivity &bc,
                             d_oracle_t &d_oracle) {
        lp.round += 1;

        weight_t opt_weight = (weight_t) ((f64) g.g_weight / (f64) lp.k);
        weight_t max_b_w = std::max(opt_weight + 1, (weight_t) (lp.lmax * 0.99));
        u32 t_minibuckets = lp.k * MAX_BUCKETS * lp.sections;
        u32 width = MAX_BUCKETS * lp.sections;

        // Determine maximum allowed vertex weight
        {
            ScopedTimer _t("refinement", "jetrs", "find_max_vwgt");

            Kokkos::parallel_reduce("find_max_vwgt", lp.k, KOKKOS_LAMBDA(const partition_t id, weight_t &update) {
                weight_t size = lp.partition.bweights(id);
                if (size < max_b_w) {
                    weight_t cap = max_b_w - size;
                    if (cap > update) {
                        update = cap;
                    }
                }
            }, Kokkos::Max<weight_t, Kokkos::DefaultExecutionSpace>(lp.max_vwgt));
            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("refinement", "jetrs", "score_candidates");

            Kokkos::deep_copy(lp.bucket_sizes, 0);

            Kokkos::parallel_for("jetrs_score_candidates", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                weight_t u_w = g.weights(u);
                weight_t u_id_w = lp.partition.bweights(u_id);

                // Default: not a candidate
                lp.gain(u) = -max_sentinel<weight_t>();
                lp.save_atomic(u) = 0;
                lp.bid(u) = (u32) -1;

                if (u_id_w <= lp.lmax) return;               // block not overloaded
                if (u_w > 2 * (u_id_w - opt_weight)) return; // too heavy
                if (u_w > 2 * lp.max_vwgt()) return;         // too heavy

                weight_t own_conn = 0;
                weight_t sum_other = 0;
                weight_t dist_other = 0;
                u32 count_other = 0;

                u32 r_beg = bc.row(u);
                u32 r_len = bc.sizes(u);
                u32 r_end = r_beg + r_len;

                // Build average connectivity to "underloaded" blocks (< max_b_w)
                for (u32 i = r_beg; i < r_end; ++i) {
                    partition_t id = bc.ids(i);
                    weight_t w = bc.weights(i);

                    own_conn = (id == u_id) ? w : own_conn;

                    if (id == NULL_PART) continue;                      // invalid entry
                    if (id == u_id) continue;                           // don't move to self
                    if (lp.partition.bweights(id) >= max_b_w) continue; // don't count overloaded destinations

                    sum_other += w;
                    dist_other += get(d_oracle, u_id, id);
                    count_other += 1;
                }

                if (count_other == 0) count_other = 1;

                weight_t g_u = ((weight_t) ((f64) (dist_other * sum_other) / (f64) count_other)) - (own_conn * dist_other);

                // Map to gain bucket
                u32 b = gain_bucket(g_u, Kokkos::min(u_w, u_id_w - lp.lmax));
                if (b >= MAX_BUCKETS) return;

                // minibucket id
                u32 g_id = (MAX_BUCKETS * u_id + b) * lp.sections + (u % lp.sections) + 1;

                // atomic prefix inside minibucket
                lp.save_atomic(u) = Kokkos::atomic_fetch_add(&lp.bucket_sizes(g_id), u_w);
                lp.bid(u) = g_id;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // --- 3. Prefix-sum over buckets (Jet: bucket_offsets) --------------------
        {
            ScopedTimer _t("refinement", "jetrs", "bucket_offsets");

            Kokkos::parallel_scan("jetrs_bucket_offsets", t_minibuckets + 1, KOKKOS_LAMBDA(const u32 i, weight_t &upd, const bool final_pass) {
                weight_t val = lp.bucket_sizes(i);
                if (final_pass) lp.bucket_offsets(i) = upd;
                upd += val;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // --- 4. Select evicted vertices (Jet: get_evictions<true>) ---------------
        //
        {
            ScopedTimer _t("refinement", "jetrs", "pick_evictions");

            Kokkos::deep_copy(lp.evict_adjust, 0);
            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("jetrs_pick_evictions", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 b_id = lp.bid(u);
                if (b_id == (u32) -1) return; // not a candidate

                partition_t u_id = lp.partition.map(u);
                weight_t u_w = g.weights(u);

                // Per-part bucket range start
                u32 p_begin = u_id * width;

                // Jet's score computation
                weight_t score = lp.save_atomic(u) + lp.bucket_offsets(b_id) - lp.bucket_offsets(p_begin);

                // Limit = overweight of part u_id
                weight_t limit = lp.partition.bweights(u_id) - lp.lmax;
                if (score < limit) {
                    // Jet: if this vertex straddles the limit, record tighter adjust
                    if (score + u_w >= limit) {
                        lp.evict_adjust(u_id) = score + u_w;
                    }
                    u32 idx = Kokkos::atomic_fetch_add(&lp.moves_idx(), 1);
                    lp.moves(idx) = u;
                }
            });
            KOKKOS_PROFILE_FENCE();

            Kokkos::deep_copy(lp.n_moves, lp.moves_idx);
        }

        if (lp.n_moves == 0) {
            return;
        }

        // --- 5. Cookie-cutter destination assignment (Jet) -----------------------
        // evict_start: prefix over per-part capacity
        //
        {
            ScopedTimer _t("refinement", "jetrs", "cookie_cutter");

            // Team policy with a single team, like Jet
            Kokkos::parallel_for("jetrs_cookie_cutter", Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type &t) {
                // First scan: adjust evict_adjust and set max_vwgt
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp.k), [&](const partition_t p, weight_t &upd, const bool final_pass) {
                    weight_t add = lp.evict_adjust(p);
                    u32 begin_bucket = MAX_BUCKETS * p * lp.sections;
                    if (add == 0) {
                        add = lp.bucket_offsets(begin_bucket + width) - lp.bucket_offsets(begin_bucket);
                    }
                    if (final_pass) {
                        lp.evict_adjust(p) = lp.bucket_offsets(begin_bucket) - upd;
                    }
                    upd += add;
                    if (final_pass && p + 1 == lp.k) {
                        lp.max_vwgt() = upd;
                    }
                });

                // Second scan: construct evict_start (capacity prefix)
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp.k), [&](const partition_t p, weight_t &upd, const bool final_pass) {
                    if (final_pass && p == 0) {
                        lp.evict_start(0) = 0;
                    }
                    if (max_b_w > lp.partition.bweights(p)) {
                        upd += max_b_w - lp.partition.bweights(p);
                    }
                    if (final_pass) {
                        lp.evict_start(p + 1) = upd;
                    }
                });
            });
            KOKKOS_PROFILE_FENCE();
        }

        // Adjust scores with evict_adjust (Jet: "adjust scores")
        {
            ScopedTimer _t("refinement", "jetrs", "adjust_scores");

            Kokkos::parallel_for("jetrs_adjust_scores", lp.n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.moves(i);
                partition_t u_id = lp.partition.map(u);
                u32 b_id = lp.bid(u);

                weight_t score = lp.save_atomic(u) + lp.bucket_offsets(b_id) - lp.evict_adjust(u_id);
                lp.save_atomic(u) = score;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // Final destination selection (Jet's "select destination parts (rs)")
        {
            ScopedTimer _t("refinement", "jetrs", "select_destinations");

            Kokkos::parallel_for("jetrs_select_dest", lp.n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.moves(i);
                weight_t u_w = g.weights(u);
                partition_t old_p = lp.partition.map(u);

                int p = 0;

                while (p < (int) lp.k) {
                    // Find chunk p where this score lies
                    while (p <= (int) lp.k && lp.evict_start(p) <= lp.save_atomic(u)) { ++p; }
                    --p;
                    if (p < (int) lp.k && (u_w / 2) <= (lp.evict_start(p + 1) - lp.save_atomic(u))) {
                        lp.id(u) = (partition_t) p;
                        return;
                    }
                    if (p < (int) lp.k) {
                        // overflow: push to the end via max_vwgt
                        lp.save_atomic(u) = Kokkos::atomic_fetch_add(&lp.max_vwgt(), u_w);
                    }
                }
                // Fallback: stay in old part
                lp.id(u) = old_p;
            });
            KOKKOS_PROFILE_FENCE();
        }
    }

    template<typename d_oracle_t>
    inline std::pair<weight_t, weight_t> promap_refine(Graph &g,
                                                       Partition &partition,
                                                       d_oracle_t &d_oracle,
                                                       partition_t k,
                                                       weight_t lmax,
                                                       u32 level,
                                                       weight_t curr_comm_cost,
                                                       weight_t curr_weight,
                                                       KokkosMemoryStack &mem_stack) {
        ProMapJetLabelPropagation lp = initialize_promap_lp(g.n, g.m, k, lmax, mem_stack);

        if (level == 0) { lp.conn_c = 0.25; }

        // copy partition
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");
            copy_into(lp.partition, partition, g.n);
            KOKKOS_PROFILE_FENCE();
        }

        // initial build of block connectivity
        BlockConnectivity bc = from_scratch(g, lp.partition, mem_stack);
        // assert_bc(bc, g, lp.partition, lp.k);

        weight_t best_comm_cost = curr_comm_cost;
        weight_t best_weight = curr_weight;

        // init arrays
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "init_arrays");
            Kokkos::deep_copy(lp.to_move, 0);
            Kokkos::deep_copy(lp.lp_to_check, 0); // initialize all as to check
            KOKKOS_PROFILE_FENCE();
        }

        u32 balance_iterations = 0;
        u32 iteration = 0;
        u32 total_n_iteration = 0;
        while (iteration < lp.n_max_iterations) {
            iteration += 1;
            total_n_iteration += 1;

            if (curr_weight <= lp.lmax) {
                promap_jetlp(lp, g, bc, d_oracle);
                balance_iterations = 0;
            } else {
                if (balance_iterations < lp.max_weak_iterations) {
                    promap_jetrw(lp, g, bc, d_oracle);
                } else {
                    promap_jetrs(lp, g, bc, d_oracle);
                }
                balance_iterations++;
            }

            if (lp.n_moves == 0) { continue; } // no move found, we can quit now

            // move in block connectivity
            move_bc(bc, g, lp.partition, lp.id, lp.moves, lp.n_moves);

            // save old map
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "copy_old_map");

                Kokkos::deep_copy(lp.old_map, lp.partition.map);

                KOKKOS_PROFILE_FENCE();
            }
            // moves in partition
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "apply_moves");

                Kokkos::parallel_for("move", lp.n_moves, KOKKOS_LAMBDA(const u32 i) {
                    vertex_t u = lp.moves(i);
                    weight_t u_w = g.weights(u);
                    partition_t u_old_id = lp.partition.map(u);
                    partition_t u_new_id = lp.id(u);

                    lp.partition.map(u) = u_new_id;
                    Kokkos::atomic_add(&lp.partition.bweights(u_old_id), -u_w);
                    Kokkos::atomic_add(&lp.partition.bweights(u_new_id), u_w);
                });
                KOKKOS_PROFILE_FENCE();
            }

            // invalidate caches of neighbors
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "invalidate_caches");

                Kokkos::parallel_for("invalidate_caches", lp.n_moves, KOKKOS_LAMBDA(const u32 i) {
                    vertex_t u = lp.moves(i);
                    lp.lp_gain(u) = -max_sentinel<weight_t>();

                    for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                        vertex_t v = g.edges_v(j);

                        lp.lp_to_check(v) = 0; // all neighbors of moved vertices need to be checked
                    }
                });
                KOKKOS_PROFILE_FENCE();
            }

            // recalculate comm cost and max weight
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "get_comm_cost");

                curr_comm_cost = comm_cost_update(curr_comm_cost, g, lp.partition, lp.old_map, lp.moves, lp.n_moves, d_oracle);

                KOKKOS_PROFILE_FENCE();
            }
            // recalculate max weight
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "get_max_weight");
                curr_weight = max_weight(lp.partition);
                KOKKOS_PROFILE_FENCE();
            }

            if (best_weight > lp.lmax && curr_weight < best_weight) {
                // copy the partition
                {
                    ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_comm_cost = curr_comm_cost;
                    best_weight = curr_weight;
                    iteration = 0;

                    KOKKOS_PROFILE_FENCE();
                }
            } else if (curr_comm_cost < best_comm_cost && (curr_weight <= lp.lmax || curr_weight < best_weight)) {
                if ((f64) curr_comm_cost < lp.phi * (f64) best_comm_cost) { iteration = 0; } {
                    ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_comm_cost = curr_comm_cost;
                    best_weight = curr_weight;

                    KOKKOS_PROFILE_FENCE();
                }
            }
        }

        free_bc(bc, mem_stack);
        free_promap_lp(lp, mem_stack);

        assert_back_is_empty(mem_stack);

        return std::make_pair(best_comm_cost, best_weight);;
    }
}

#endif //GPU_HEIPA_PROMAP_JET_LABEL_PROPAGATION_H
