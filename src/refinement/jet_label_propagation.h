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

#include <filesystem>

#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/edge_cut.h"
#include "../utility/macros.h"
#include "../datastructures/partition.h"
#include "../datastructures/block_connectivity.h"

namespace GPU_HeiPa {
    constexpr u32 MAX_BUCKETS = 50;   // replace MAX_SLOTS in bucket math
    constexpr u32 MID_BUCKET = 25;    // bucket for zero gain
    constexpr f32 BUCKET_BASE = 1.5f; // geometric spacing for |gain|/weight
    constexpr u32 TARGET_PER_SHARD = 4096;

    KOKKOS_INLINE_FUNCTION
    u32 gain_bucket_old(weight_t gain, weight_t vwgt) {
        #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        f32 w = (f32) (vwgt > 0 ? vwgt : 1);
        f32 r = (f32) gain / w;
        if (r > 0.0f) return 0;
        if (r == 0.0f) return 1;
        r = -r;

        // b = MID + round(log_base(r)), with base = 1.5
        f32 t = logf(r) / logf(BUCKET_BASE);
        int b = (int) MID_BUCKET + (int) floorf(t + 0.5f);
        if (b < 2) b = 2;
        if (b > (int) MAX_BUCKETS - 1) b = (int) MAX_BUCKETS - 1;
        return (u32) b;
        #else
        // Normalize by vertex weight (avoid div-by-zero)
        f32 g = static_cast<f32>(gain);
        f32 w = static_cast<f32>(vwgt > 0 ? vwgt : 1);
        f32 r = g / w;

        // Positive gain: best
        if (r > 0.0f) return 0;
        // Exact tie: middle left (keep 1 == no-loss "tie" like your current mapping?)
        if (r == 0.0f) return 1;

        // Negative normalized gain -> spread around MID_BUCKET via base 1.5
        r = -r;

        // Start from the mid bucket (loss ≈ weight)
        u32 b = MID_BUCKET;

        // Move left for small losses, right for large losses.
        // We avoid logf for portability; the loop is tiny (≤ ~10 steps typically).
        if (r < 1.0f) {
            while (r < 1.0f && b > 2) {
                // keep 0=pos, 1=tie reserved
                r *= BUCKET_BASE;
                --b;
            }
            if (b < 2) b = 2;
        } else {
            while (r > 1.0f && b < (MAX_BUCKETS - 1)) {
                r /= BUCKET_BASE;
                ++b;
            }
            if (b > MAX_BUCKETS - 1) b = MAX_BUCKETS - 1;
        }
        return b;
        #endif
    }

    KOKKOS_INLINE_FUNCTION
    u32 gain_bucket(weight_t g, weight_t vwgt) {
        if (g > 0) { return 0; }
        if (g == 0) { return 1; }

        //cast to float so we can approximate log_1.5
        f64 gain = (f64) -g / (f64) vwgt;
        u32 gain_type = MID_BUCKET;

        if (gain < 1.0) {
            while (gain < 1.0) {
                gain *= 1.5;
                gain_type--;

                if (gain_type < 2) {
                    return 2;
                }
            }
        } else {
            while (gain > 1.0) {
                gain /= 1.5;
                gain_type++;

                if (gain_type >= MAX_BUCKETS) {
                    return MAX_BUCKETS - 1;
                }
            }
        }
        return gain_type;
    }

    struct JetLabelPropagation {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        u32 n_max_iterations = 12;
        u32 max_weak_iterations = 2;
        f64 phi = 0.999;
        f64 heavy_alpha = 1.5; // smaller - fewer vertices moved, larger more vertices moved
        f64 conn_c = 0.75;
        u32 sections = 1; // adaptive mini buckets per (part,bucket)
        Partition partition;

        UnmanagedDeviceWeight save_gains;
        UnmanagedDeviceWeight pre_gain;
        UnmanagedDevicePartition dest_part;

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

    inline JetLabelPropagation initialize_lp(const vertex_t t_n,
                                             const vertex_t t_m,
                                             const partition_t t_k,
                                             const weight_t t_lmax,
                                             KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "allocate");

        JetLabelPropagation lp;

        lp.n = t_n;
        lp.m = t_m;
        lp.k = t_k;
        lp.lmax = t_lmax;

        lp.partition = initialize_partition(t_n, t_k, t_lmax, mem_stack);

        lp.save_gains = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n), t_n);
        lp.pre_gain = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n), t_n);
        lp.dest_part = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * t_n), t_n);

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

    inline void free_lp(JetLabelPropagation &lp,
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
    }

    inline void jetlp(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        // for each vertex determine best block
        {
            ScopedTimer _t("refinement", "jetlp", "best_block");
            Kokkos::deep_copy(lp.temp_moves_idx, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (bc.dest_cache(u) != NULL_PART) {
                    lp.dest_part(u) = bc.dest_cache(u);
                } else {
                    partition_t u_id = lp.partition.map(u);
                    partition_t best_id = NO_MOVE;

                    weight_t own_conn = 0;
                    weight_t best_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        weight_t w = bc.weights(i);

                        own_conn = id == u_id ? w : own_conn;

                        if (id == NULL_PART) { continue; } // no valid entry
                        if (id == u_id) { continue; }      // do not move to self

                        if (w > best_conn || (w == best_conn && id < best_id)) {
                            best_id = id;
                            best_conn = w;
                        }
                    }

                    weight_t gain = 0;

                    if (best_id != NO_MOVE) {
                        if ((best_conn >= own_conn) || (own_conn - best_conn) < Kokkos::floor(lp.conn_c * (f64) own_conn)) {
                            gain = best_conn - own_conn;
                        } else {
                            best_id = NO_MOVE;
                        }
                    }

                    bc.dest_cache(u) = best_id;
                    lp.dest_part(u) = best_id;
                    lp.save_gains(u) = gain;
                }

                if (lp.dest_part(u) != NO_MOVE && bc.lock(u) == 0) {
                    lp.pre_gain(u) = lp.save_gains(u);
                    u32 idx = Kokkos::atomic_fetch_inc(&lp.temp_moves_idx());
                    lp.temp_moves(idx) = u;
                } else {
                    lp.pre_gain(u) = min_sentinel<weight_t>();
                    bc.lock(u) = 0;
                }
            });

            Kokkos::deep_copy(lp.temp_n_moves, lp.temp_moves_idx);

            KOKKOS_PROFILE_FENCE();
        }

        // use afterburner
        {
            ScopedTimer _t("refinement", "jetlp", "afterburner");

            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("afterburner", lp.temp_n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.temp_moves(i);
                weight_t u_gain = lp.pre_gain(u);
                partition_t old_u_id = lp.partition.map(u);
                partition_t new_u_id = lp.dest_part(u);

                weight_t update = 0;
                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                    vertex_t v = g.edges_v(j);

                    weight_t v_gain = lp.pre_gain(v);

                    if (v_gain > u_gain || (v_gain == u_gain && v < u)) {
                        partition_t v_old_id = lp.partition.map(v);
                        partition_t v_new_id = lp.dest_part(v);
                        weight_t w = g.edges_w(j);

                        if (v_new_id == old_u_id) { update -= w; }
                        if (v_new_id == new_u_id) { update += w; }

                        if (v_old_id == old_u_id) { update += w; }
                        if (v_old_id == new_u_id) { update -= w; }
                    }
                }

                if (u_gain + update >= 0) {
                    bc.lock(u) = 1;
                    lp.id(u) = new_u_id;

                    u32 idx = Kokkos::atomic_fetch_inc(&lp.moves_idx());
                    lp.moves(idx) = u;
                }
            });

            Kokkos::deep_copy(lp.n_moves, lp.moves_idx);

            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void jetrw(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        weight_t opt_weight = (weight_t) ((f64) g.g_weight / (f64) lp.k);
        weight_t max_b_w = (weight_t) (lp.lmax * 0.99);
        if (max_b_w < lp.lmax - 100) { max_b_w = lp.lmax - 100; }
        u32 t_minibuckets = lp.k * MAX_BUCKETS * lp.sections;
        u32 width = MAX_BUCKETS * lp.sections;

        // u32 n_underloaded_blocks;
        // determine underloaded blocks
        {
            ScopedTimer _t("refinement", "jetrw", "underloaded_blocks");
            Kokkos::deep_copy(lp.n_underloaded_blocks, 0);

            Kokkos::parallel_for("underloaded_blocks", lp.k, KOKKOS_LAMBDA(const partition_t id) {
                if (lp.partition.bweights(id) < max_b_w) {
                    u32 idx = Kokkos::atomic_fetch_inc(&lp.n_underloaded_blocks());
                    lp.underloaded_blocks(idx) = id;
                }
            });

            // Kokkos::deep_copy(n_underloaded_blocks, lp.n_underloaded_blocks);

            KOKKOS_PROFILE_FENCE();
        }
        // if (n_underloaded_blocks == 0) { return; }

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
                weight_t best_id_w = -max_sentinel<weight_t>();

                lp.id(u) = u_id; // default move to self
                lp.bid(u) = (u32) -1;

                if (u_id_w <= lp.lmax) { return; }                                         // u_id not overloaded
                if ((f64) u_w >= lp.heavy_alpha * ((f64) u_id_w - opt_weight)) { return; } // vertex is too heavy

                u32 r_beg = bc.row(u);
                u32 r_len = bc.sizes(u);
                u32 r_end = r_beg + r_len;

                weight_t own_conn = 0;
                for (u32 i = r_beg; i < r_end; ++i) {
                    partition_t id = bc.ids(i);
                    weight_t w = bc.weights(i);

                    own_conn = id == u_id ? w : own_conn;

                    if (id == NULL_PART) { continue; } // no valid entry
                    // if (id == u_id) { continue; }                           // dont move to self
                    if (lp.partition.bweights(id) >= max_b_w) { continue; } // dont move into non underloaded partition

                    if (w > best_id_w || (w == best_id_w && id < best_id)) {
                        best_id = id;
                        best_id_w = w;
                    }
                }

                weight_t gain = best_id_w - own_conn;

                if (gain <= 0) {
                    if (lp.n_underloaded_blocks() == 0) {
                        best_id = u_id;
                    } else {
                        best_id = lp.underloaded_blocks(u % lp.n_underloaded_blocks());
                        gain = -own_conn;
                    }
                }

                u32 g_id = (u32) -1;
                if (best_id != u_id) {
                    u32 b = gain_bucket(gain, u_w);
                    g_id = (MAX_BUCKETS * u_id + b) * lp.sections + (u % lp.sections);

                    lp.save_atomic(u) = Kokkos::atomic_fetch_add(&lp.bucket_sizes(g_id), u_w); // prefix inside this bucket
                    lp.id(u) = best_id;

                    u32 idx = Kokkos::atomic_fetch_inc(&lp.temp_moves_idx());
                    lp.temp_moves(idx) = u;
                }

                lp.bid(u) = g_id;
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
            ScopedTimer _t("refinement", "jetrw", "pick_evictions");

            Kokkos::deep_copy(lp.temp_n_moves, lp.temp_moves_idx);
            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("jetrw_pick_evictions", lp.temp_n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.temp_moves(i);
                u32 b_id = lp.bid(u);
                // if (b_id == (u32) -1) return; // not a candidate

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

    inline void jetrs(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
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
                    count_other += 1;
                }

                if (count_other == 0) count_other = 1;

                weight_t g_u = ((weight_t) ((f64) sum_other / (f64) count_other)) - own_conn;

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

    inline weight_t refine(Graph &g,
                           Partition &partition,
                           partition_t k,
                           weight_t lmax,
                           u32 level,
                           weight_t curr_edge_cut,
                           KokkosMemoryStack &mem_stack) {
        JetLabelPropagation lp = initialize_lp(g.n, g.m, k, lmax, mem_stack);

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

        // analyze_block_connectivity(bc, g, lp.partition, k);

        weight_t best_edge_cut = curr_edge_cut;
        weight_t best_weight = 0;
        // initial maximum weight
        {
            ScopedTimer _t_edge_cut("refinement", "JetLabelPropagation", "get_max_weight");
            best_weight = max_weight(lp.partition);
            KOKKOS_PROFILE_FENCE();
        }
        weight_t curr_weight = best_weight;

        u32 balance_iterations = 0;
        u32 iteration = 0;
        while (iteration < lp.n_max_iterations) {
            iteration += 1;

            if (curr_weight <= lp.lmax) {
                jetlp(lp, g, bc);
                // assert_bc(bc, g, lp.partition, lp.k);
                balance_iterations = 0;
            } else {
                if (balance_iterations < lp.max_weak_iterations) {
                    jetrw(lp, g, bc);
                    // assert_bc(bc, g, lp.partition, lp.k);
                } else {
                    jetrs(lp, g, bc);
                    // assert_bc(bc, g, lp.partition, lp.k);
                }
                balance_iterations++;
            }

            weight_t cut_change1 = 0;
            // 1) compute cut change from "old" connectivity (Jet: cut_change1)
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "cut_change1");

                Kokkos::parallel_reduce("cut_change1", lp.n_moves, KOKKOS_LAMBDA(const u32 i, weight_t &local) {
                    vertex_t u = lp.moves(i);
                    partition_t old_b = lp.partition.map(u); // current block
                    partition_t new_b = lp.id(u);            // selected destination

                    if (old_b == new_b) { return; }

                    // connectivity of u to its old / new blocks BEFORE moving
                    weight_t old_con = 0;
                    weight_t new_con = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;
                    for (u32 j = r_beg; j < r_end; ++j) {
                        partition_t id = bc.ids(j);
                        weight_t w = bc.weights(j);

                        old_con = id == old_b ? w : old_con;
                        new_con = id == new_b ? w : new_con;
                    }

                    // Jet uses (b_con - p_con). Sign/factor must match your cut definition.
                    local += (new_con - old_con);
                }, cut_change1);

                KOKKOS_PROFILE_FENCE();
            }

            // move in block connectivity
            move_bc(bc, g, lp.partition, lp.id, lp.moves, lp.n_moves);

            // moves in partition
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "apply_moves");

                Kokkos::parallel_for("move", lp.n_moves, KOKKOS_LAMBDA(const u32 i) {
                    vertex_t u = lp.moves(i);
                    weight_t u_w = g.weights(u);
                    partition_t u_old_id = lp.partition.map(u);
                    partition_t u_new_id = lp.id(u);

                    if (u_old_id == u_new_id) { return; }

                    MY_KOKKOS_ASSERT(u_old_id < lp.partition.k);
                    MY_KOKKOS_ASSERT(u_new_id < lp.partition.k);

                    bc.dest_cache(u) = NULL_PART;

                    lp.old_map(u) = u_old_id;
                    lp.partition.map(u) = u_new_id;
                    Kokkos::atomic_add(&lp.partition.bweights(u_old_id), -u_w);
                    Kokkos::atomic_add(&lp.partition.bweights(u_new_id), u_w);
                });
                KOKKOS_PROFILE_FENCE();
            }

            weight_t cut_change2 = 0;
            // 5) compute cut change from "new" connectivity (Jet: cut_change2)
            {
                ScopedTimer _t("refinement", "JetLabelPropagation", "cut_change2");

                Kokkos::parallel_reduce("cut_change2", lp.n_moves, KOKKOS_LAMBDA(const u32 i, weight_t &local) {
                    vertex_t u = lp.moves(i);

                    partition_t old_b = lp.old_map(u);       // BEFORE moves
                    partition_t new_b = lp.partition.map(u); // AFTER moves

                    if (old_b == new_b) { return; }

                    // connectivity AFTER connectivity update
                    weight_t old_con = 0;
                    weight_t new_con = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;
                    for (u32 j = r_beg; j < r_end; ++j) {
                        partition_t id = bc.ids(j);
                        weight_t w = bc.weights(j);

                        old_con = id == old_b ? w : old_con;
                        new_con = id == new_b ? w : new_con;
                    }

                    local += (new_con - old_con);
                }, cut_change2);

                curr_edge_cut -= (cut_change1 + cut_change2) / 2;

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
                    best_edge_cut = curr_edge_cut;
                    best_weight = curr_weight;
                    iteration = 0;

                    KOKKOS_PROFILE_FENCE();
                }
            } else if (curr_edge_cut < best_edge_cut && (curr_weight <= lp.lmax || curr_weight < best_weight)) {
                if ((f64) curr_edge_cut < lp.phi * (f64) best_edge_cut) { iteration = 0; } {
                    ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_edge_cut = curr_edge_cut;
                    best_weight = curr_weight;

                    KOKKOS_PROFILE_FENCE();
                }
            }
        }

        // analyze_block_connectivity(bc, g, lp.partition, k);
        // assert_bc(bc, g, lp.partition, lp.k);

        free_bc(bc, mem_stack);
        free_lp(lp, mem_stack);

        assert_back_is_empty(mem_stack);

        return best_edge_cut;
    }
}

#endif //GPU_HEIPA_JET_LABEL_PROPAGATION_H
