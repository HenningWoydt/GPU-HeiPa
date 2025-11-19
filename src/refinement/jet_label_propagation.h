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

#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/edge_cut.h"
#include "../utility/macros.h"
#include "../datastructures/partition.h"
#include "../datastructures/block_connectivity.h"

namespace GPU_HeiPa {
    constexpr u32 MAX_BUCKETS = 50;     // replace MAX_SLOTS in bucket math
    constexpr u32 MID_BUCKET = 25;      // bucket for zero gain
    constexpr float BUCKET_BASE = 1.5f; // geometric spacing for |gain|/weight
    constexpr u32 TARGET_PER_SHARD = 4096;

    KOKKOS_INLINE_FUNCTION
    u32 gain_bucket(weight_t gain, weight_t vwgt) {
        #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        float w = (float) (vwgt > 0 ? vwgt : 1);
        float r = (float) gain / w;
        if (r > 0.0f) return 0;
        if (r == 0.0f) return 1;
        r = -r;

        // b = MID + round(log_base(r)), with base = 1.5
        float t = logf(r) / logf(BUCKET_BASE);
        int b = (int) MID_BUCKET + (int) floorf(t + 0.5f);
        if (b < 2) b = 2;
        if (b > (int) MAX_BUCKETS - 1) b = (int) MAX_BUCKETS - 1;
        return (u32) b;
        #else
        // Normalize by vertex weight (avoid div-by-zero)
        float g = static_cast<float>(gain);
        float w = static_cast<float>(vwgt > 0 ? vwgt : 1);
        float r = g / w;

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
    u32 get_idx(partition_t p, u32 s, u32 m, u32 sections) {
        return (p * MAX_BUCKETS + s) * sections + m;
    }

    KOKKOS_INLINE_FUNCTION
    bool ord_smaller(vertex_t u, vertex_t v, weight_t gain_u, weight_t gain_v) {
        if (gain_u > gain_v) { return true; }
        if (gain_u == gain_v && u < v) { return true; }
        return false;
    }

    struct JetLabelPropagation {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        u32 round = 0;

        u32 n_max_iterations = 12;
        u32 max_weak_iterations = 2;
        f64 phi = 0.999;
        f64 heavy_alpha = 1.5; // smaller - less vertices moved, larger more vertices moved
        f64 sigma_percent = 0.03;
        f64 sigma_percent_min = 0.005;
        weight_t sigma = 10;
        f64 conn_c = 0.75;
        u32 sections = 1; // adaptive mini buckets per (part,bucket)
        Partition partition;

        UnmanagedDeviceWeight gain;
        UnmanagedDevicePartition id;
        UnmanagedDeviceU32 in_X;
        UnmanagedDeviceU32 to_move;

        Kokkos::View<u32, DeviceMemorySpace> to_move_idx;
        UnmanagedDeviceMove to_move_list;
        Kokkos::View<u32, DeviceMemorySpace> to_move_idx_2;
        u32 list_size_2 = 0;
        UnmanagedDeviceMove to_move_list_2;

        UnmanagedDeviceWeight bucket_counts;
        UnmanagedDeviceWeight bucket_offsets;
        UnmanagedDeviceU32 bucket_cursor;
        UnmanagedDeviceVertex flat_buckets;

        Kokkos::View<u32, DeviceMemorySpace> n_underloaded_blocks;
        UnmanagedDevicePartition underloaded_blocks;

        UnmanagedDeviceWeight save_atomic;
    };

    inline JetLabelPropagation initialize_lp(const vertex_t t_n,
                                             const vertex_t t_m,
                                             const partition_t t_k,
                                             const weight_t t_lmax,
                                             KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refine", "JetLabelPropagation", "allocate");

        JetLabelPropagation lp;

        lp.n = t_n;
        lp.m = t_m;
        lp.k = t_k;
        lp.lmax = t_lmax;
        lp.sigma = (weight_t) (0.99 * (f64) lp.lmax);
        if (lp.sigma < lp.lmax - 100) lp.sigma = lp.lmax - 100;

        lp.partition = initialize_partition(t_n, t_k, t_lmax, mem_stack);

        auto *gain_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n);
        auto *id_ptr = (partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * t_n);
        auto *in_X_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n);
        auto *to_move_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n);
        lp.gain = UnmanagedDeviceWeight(gain_ptr, t_n);
        lp.id = UnmanagedDevicePartition(id_ptr, t_n);
        lp.in_X = UnmanagedDeviceU32(in_X_ptr, t_n);
        lp.to_move = UnmanagedDeviceU32(to_move_ptr, t_n);

        auto *to_move_list_ptr = (Move *) get_chunk_back(mem_stack, sizeof(Move) * t_n);
        auto *to_move_list_2_ptr = (Move *) get_chunk_back(mem_stack, sizeof(Move) * t_n);
        lp.to_move_idx = Kokkos::View<u32, DeviceMemorySpace>("to_move_idx");
        lp.to_move_list = UnmanagedDeviceMove(to_move_list_ptr, t_n);
        lp.to_move_idx_2 = Kokkos::View<u32, DeviceMemorySpace>("to_move_idx_2");
        lp.to_move_list_2 = UnmanagedDeviceMove(to_move_list_2_ptr, t_n);

        // compute sections so that k * MAX_BUCKETS * sections ≈ n / TARGET_PER_SHARD
        u64 shards = (t_n + TARGET_PER_SHARD - 1) / TARGET_PER_SHARD;
        u64 per_part = MAX_BUCKETS; // buckets per part
        u64 s = (shards + (t_k * per_part - 1)) / (t_k * per_part);
        lp.sections = s == 0 ? 1 : (u32) s;

        auto total_shards = (size_t) t_k * MAX_BUCKETS * lp.sections;

        auto *bucket_counts_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * total_shards);
        auto *bucket_offsets_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * total_shards);
        auto *bucket_cursor_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * total_shards);
        auto *flat_buckets_ptr = (vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * lp.n);
        lp.bucket_counts = UnmanagedDeviceWeight(bucket_counts_ptr, total_shards);
        lp.bucket_offsets = UnmanagedDeviceWeight(bucket_offsets_ptr, total_shards);
        lp.bucket_cursor = UnmanagedDeviceU32(bucket_cursor_ptr, total_shards);
        lp.flat_buckets = UnmanagedDeviceVertex(flat_buckets_ptr, lp.n);

        auto *underloaded_blocks_ptr = (partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * t_k);
        lp.n_underloaded_blocks = Kokkos::View<u32, DeviceMemorySpace>("n_underloaded_blocks");
        lp.underloaded_blocks = UnmanagedDevicePartition(underloaded_blocks_ptr, t_k);

        auto *save_atomic_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n);
        lp.save_atomic = UnmanagedDeviceWeight(save_atomic_ptr, t_n);

        return lp;
    }

    inline void free_lp(JetLabelPropagation &lp,
                        KokkosMemoryStack &mem_stack) {
        free_partition(lp.partition, mem_stack);

        ScopedTimer _t("refine", "JetLabelPropagation", "free");

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
        lp.round += 1;

        // for each vertex determine best block
        {
            ScopedTimer _t("refine", "jetlp", "best_block");
            Kokkos::deep_copy(lp.to_move_idx, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (lp.to_move(u) != 0) { return; } // already moved

                partition_t u_id = lp.partition.map(u);
                weight_t u_w = g.weights(u);
                weight_t own_conn = 0;

                partition_t best_id = u_id;
                weight_t best_id_w = -max_sentinel<weight_t>();

                bool found = false;
                for (u32 i = bc.row(u); i < bc.row(u + 1); ++i) {
                    partition_t id = bc.ids(i);
                    weight_t w = bc.weights(i);

                    own_conn = id == u_id ? w : own_conn;

                    if (id == lp.k) { continue; } // no valid entry
                    if (id == u_id) { continue; } // do not move to self

                    if (w > best_id_w || w == best_id_w && id < best_id) {
                        best_id = id;
                        best_id_w = w;
                        found = true;
                    }
                }

                if (found) {
                    weight_t gain = best_id_w - own_conn;

                    // apply first filter
                    bool non_neg = gain >= 0;
                    bool conn_filter = -gain < Kokkos::floor(lp.conn_c * (f64) own_conn);

                    if (non_neg || conn_filter) {
                        u32 idx = Kokkos::atomic_fetch_add(&lp.to_move_idx(), 1);
                        lp.to_move_list(idx) = {u, u_w, u_id, best_id, gain};
                        lp.id(u) = best_id;
                        lp.gain(u) = gain;
                        lp.in_X(u) = lp.round;
                    }
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        u32 list_size;
        // use afterburner
        {
            ScopedTimer _t("refine", "jetlp", "afterburner");
            Kokkos::deep_copy(list_size, lp.to_move_idx);
            Kokkos::deep_copy(lp.to_move_idx_2, 0);

            Kokkos::parallel_for("afterburner", list_size, KOKKOS_LAMBDA(const u32 i) {
                Move m = lp.to_move_list(i);
                vertex_t u = m.u;
                weight_t u_w = m.w;
                weight_t u_gain = m.gain;
                partition_t old_u_id = m.old_id;
                partition_t new_u_id = m.new_id;

                weight_t gain = 0;
                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                    vertex_t v = g.edges_v(j);
                    weight_t w = g.edges_w(j);

                    bool high_priority = lp.in_X(v) == lp.round && ord_smaller(u, v, u_gain, lp.gain(v));

                    partition_t v_id = high_priority ? lp.id(v) : lp.partition.map(v);

                    gain += w * ((old_u_id != v_id) - (new_u_id != v_id));
                }

                if (gain >= 0) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.to_move_idx_2(), 1);
                    lp.to_move_list_2(idx) = {u, u_w, old_u_id, new_u_id, gain};
                    lp.to_move(u) = lp.round;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        Kokkos::deep_copy(lp.list_size_2, lp.to_move_idx_2);
        // move in block connectivity
        move_bc(bc, g, lp.partition, lp.to_move_list_2, lp.list_size_2);
        // moves in partition
        {
            ScopedTimer _t("refine", "jetlp", "apply_moves");

            Kokkos::parallel_for("move", lp.list_size_2, KOKKOS_LAMBDA(const u32 i) {
                                     Move m = lp.to_move_list_2(i);
                                     vertex_t u = m.u;
                                     weight_t u_w = m.w;
                                     partition_t u_old_id = m.old_id;
                                     partition_t u_new_id = m.new_id;

                                     lp.partition.map(u) = u_new_id;
                                     Kokkos::atomic_add(&lp.partition.bweights(u_old_id), -u_w);
                                     Kokkos::atomic_add(&lp.partition.bweights(u_new_id), u_w);
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void jetrw(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        lp.round += 1;

        // determine underloaded blocks
        {
            ScopedTimer _t("rebalance", "jetrw", "underloaded_blocks");
            Kokkos::deep_copy(lp.n_underloaded_blocks, 0);

            Kokkos::parallel_for("underloaded_blocks", lp.k, KOKKOS_LAMBDA(const partition_t id) {
                if (lp.partition.bweights(id) < lp.sigma) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.n_underloaded_blocks(), 1);
                    lp.underloaded_blocks(idx) = id;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        u32 n_blocks;
        Kokkos::deep_copy(n_blocks, lp.n_underloaded_blocks);
        if (n_blocks == 0) { return; }

        f64 opt_weight = (f64) g.g_weight / (f64) lp.k;
        // determine best block
        {
            ScopedTimer _t("rebalance", "jetrw", "best_block");
            Kokkos::deep_copy(lp.to_move_idx, 0);
            Kokkos::deep_copy(lp.bucket_counts, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                bool found = false;
                partition_t old_u_id = lp.partition.map(u);
                weight_t u_w = g.weights(u);

                partition_t best_id = old_u_id;
                weight_t best_id_w = -max_sentinel<weight_t>();

                if (lp.partition.bweights(old_u_id) <= lp.lmax) { return; }                                        // u_id not overloaded
                if ((f64) u_w > lp.heavy_alpha * ((f64) lp.partition.bweights(old_u_id) - opt_weight)) { return; } // vertex is too heavy

                weight_t own_conn = 0;
                for (u32 i = bc.row(u); i < bc.row(u + 1); ++i) {
                    partition_t id = bc.ids(i);
                    weight_t w = bc.weights(i);

                    own_conn = id == old_u_id ? w : own_conn;

                    if (id == lp.k) { continue; }                            // no valid entry
                    if (id == old_u_id) { continue; }                        // dont move to self
                    if (lp.partition.bweights(id) >= lp.sigma) { continue; } // dont move into non underloaded partition

                    if (w > best_id_w || (w == best_id_w && id < best_id)) {
                        best_id = id;
                        best_id_w = w;
                        found = true;
                    }
                }

                weight_t gain = best_id_w - own_conn;

                if (!found || gain < 0) {
                    best_id = lp.underloaded_blocks(u % lp.n_underloaded_blocks());
                    gain = -own_conn;
                }

                u32 b = gain_bucket(gain, u_w);
                u32 mini = u % lp.sections;
                u32 g_id = get_idx(old_u_id, b, mini, lp.sections);

                weight_t old = Kokkos::atomic_fetch_add(&lp.bucket_counts(g_id), u_w);
                lp.save_atomic(u) = old; // prefix inside this bucket

                u32 idx = Kokkos::atomic_fetch_add(&lp.to_move_idx(), 1);
                lp.to_move_list(idx) = {u, u_w, old_u_id, best_id, gain};
                lp.id(u) = best_id;
            });
            KOKKOS_PROFILE_FENCE();
        }
        // determine bucket offsets
        {
            ScopedTimer _t("rebalance", "jetrw", "bucket_offsets");

            // prefix sum for bucket offsets
            Kokkos::parallel_scan("bucket_offsets", lp.k * MAX_BUCKETS * lp.sections, KOKKOS_LAMBDA(const u32 i, weight_t &upd, const bool final_pass) {
                                      weight_t val = lp.bucket_counts(i);
                                      if (final_pass) lp.bucket_offsets(i) = upd; // write old prefix
                                      upd += val;
                                  }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // pick prefix
        {
            ScopedTimer _t("rebalance", "jetrw", "pick_prefix");
            u32 list_size;
            Kokkos::deep_copy(list_size, lp.to_move_idx);
            Kokkos::deep_copy(lp.to_move_idx_2, 0);

            Kokkos::parallel_for("pick_prefix", list_size, KOKKOS_LAMBDA(const u32 i) {
                Move m = lp.to_move_list(i);
                vertex_t u = m.u;
                weight_t u_w = m.w;
                partition_t old_u_id = m.old_id;
                partition_t new_u_id = m.new_id;
                weight_t gain = m.gain;

                weight_t size_p = lp.partition.bweights(old_u_id);

                if (size_p <= lp.sigma) return; // not overloaded enough

                weight_t limit = size_p - lp.sigma;

                u32 b = gain_bucket(gain, u_w);
                u32 mini = u % lp.sections;
                u32 g_id = get_idx(old_u_id, b, mini, lp.sections);

                u32 width = MAX_BUCKETS * lp.sections;
                u32 p_begin_shard = old_u_id * width;

                weight_t begin_bucket = lp.bucket_offsets(p_begin_shard);
                weight_t score = lp.save_atomic(u) + lp.bucket_offsets(g_id) - begin_bucket;

                if (score < limit) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.to_move_idx_2(), 1);
                    lp.to_move_list_2(idx) = {u, u_w, old_u_id, new_u_id, 0};
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        // move in block connectivity
        Kokkos::deep_copy(lp.list_size_2, lp.to_move_idx_2);
        move_bc(bc, g, lp.partition, lp.to_move_list_2, lp.list_size_2);
        // moves in partition
        {
            ScopedTimer _t("rebalance", "jetrw", "apply_moves");

            Kokkos::parallel_for("move", lp.list_size_2, KOKKOS_LAMBDA(const u32 i) {
                                     Move m = lp.to_move_list_2(i);
                                     vertex_t u = m.u;
                                     weight_t u_w = m.w;
                                     partition_t old_u_id = m.old_id;
                                     partition_t new_u_id = m.new_id;

                                     lp.partition.map(u) = new_u_id;
                                     Kokkos::atomic_add(&lp.partition.bweights(old_u_id), -u_w);
                                     Kokkos::atomic_add(&lp.partition.bweights(new_u_id), u_w);
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void jetrs(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        lp.round += 1;

        // determine best block
        {
            ScopedTimer _t("rebalance", "jetrs", "best_block");
            Kokkos::deep_copy(lp.to_move_idx, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     bool found = false;
                                     partition_t best_id = lp.partition.map(u);
                                     weight_t best_id_w = -max_sentinel<weight_t>();

                                     partition_t old_u_id = lp.partition.map(u);
                                     weight_t u_w = g.weights(u);

                                     if (lp.partition.bweights(old_u_id) <= lp.lmax) { return; }                                                                 // u_id not overloaded
                                     if ((f64) g.weights(u) > lp.heavy_alpha * (lp.partition.bweights(old_u_id) - ((f64) g.g_weight / (f64) lp.k))) { return;; } // vertex is too heavy

                                     weight_t own_conn = 0;
                                     for (u32 i = bc.row(u); i < bc.row(u + 1); ++i) {
                                         partition_t id = bc.ids(i);
                                         weight_t w = bc.weights(i);

                                         own_conn = id == old_u_id ? w : own_conn;

                                         if (id == lp.k) { continue; }                                // no valid entry
                                         if (lp.partition.bweights(id) + u_w > lp.lmax) { continue; } // dont move into overloaded partition
                                         if (old_u_id == id) { continue; }                            // dont move to self

                                         if (w > best_id_w || w == best_id_w && id < best_id) {
                                             best_id = id;
                                             best_id_w = w;
                                             found = true;
                                         }
                                     }

                                     weight_t gain = best_id_w - own_conn;

                                     if (found) {
                                         u32 idx = Kokkos::atomic_fetch_add(&lp.to_move_idx(), 1);
                                         lp.to_move_list(idx) = {u, u_w, old_u_id, best_id, gain};
                                         lp.id(u) = best_id;
                                     }
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        u32 list_size;
        // count number of elements in each bucket
        {
            ScopedTimer _t("rebalance", "jetrs", "count_buckets");

            Kokkos::deep_copy(list_size, lp.to_move_idx);
            Kokkos::deep_copy(lp.bucket_counts, 0);

            // 2) For each vertex u in overloaded sources: pick best destination d with cap>0
            Kokkos::parallel_for("count_buckets", list_size, KOKKOS_LAMBDA(const u32 i) {
                                     Move m = lp.to_move_list(i);
                                     vertex_t u = m.u;
                                     weight_t u_w = m.w;
                                     partition_t new_u_id = m.new_id;
                                     weight_t gain = m.gain;

                                     u32 b = gain_bucket(gain, u_w);
                                     u32 mini = u % lp.sections;
                                     Kokkos::atomic_inc(&lp.bucket_counts(get_idx(new_u_id, b, mini, lp.sections)));
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // determine bucket offsets
        {
            ScopedTimer _t("rebalance", "jetrs", "bucket_offsets");

            // prefix sum
            Kokkos::parallel_scan("bucket_offsets", lp.k * MAX_BUCKETS * lp.sections, KOKKOS_LAMBDA(const u32 i, u32 &upd, const bool final_pass) {
                                      u32 val = lp.bucket_counts(i);
                                      if (final_pass) lp.bucket_offsets(i) = upd;
                                      upd += val;
                                  }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // copy cursors
        {
            ScopedTimer _t_fill_buckets("rebalance", "jetrs", "copy_cursor");
            Kokkos::deep_copy(lp.bucket_cursor, lp.bucket_offsets);
            KOKKOS_PROFILE_FENCE();
        }
        // fill buckets
        {
            ScopedTimer _t_fill_buckets("rebalance", "jetrs", "fill_buckets");

            Kokkos::parallel_for("fill_buckets", list_size, KOKKOS_LAMBDA(const u32 i) {
                                     Move m = lp.to_move_list(i);
                                     vertex_t u = m.u;
                                     weight_t u_w = m.w;
                                     partition_t new_u_id = m.new_id;
                                     weight_t gain = m.gain;

                                     u32 b = gain_bucket(gain, u_w);
                                     u32 mini = u % lp.sections;
                                     u32 pos = Kokkos::atomic_fetch_inc(&lp.bucket_cursor(get_idx(new_u_id, b, mini, lp.sections)));
                                     lp.flat_buckets(pos) = u;
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // pick prefix
        {
            ScopedTimer _t("rebalance", "jetrs", "pick_prefix");
            Kokkos::deep_copy(lp.to_move_idx_2, 0);

            // 4) Per-destination prefix selection: honor cap[d]
            Kokkos::parallel_for("pick_prefix", lp.k, KOKKOS_LAMBDA(const partition_t d) {
                                     weight_t w = lp.partition.bweights(d);
                                     weight_t cap_d = (w < lp.lmax) ? (lp.lmax - w) : 0;
                                     if (cap_d <= 0) return;

                                     weight_t acquired = 0;
                                     for (u32 s = 0; s < MAX_BUCKETS && acquired < cap_d; ++s) {
                                         for (u32 mini = 0; mini < lp.sections && acquired < cap_d; ++mini) {
                                             u32 ridx = get_idx(d, s, mini, lp.sections);
                                             u32 beg = lp.bucket_offsets(ridx);
                                             u32 end = lp.bucket_offsets(ridx) + lp.bucket_counts(ridx);

                                             for (u32 pos = beg; pos < end && acquired < cap_d; ++pos) {
                                                 vertex_t u = lp.flat_buckets(pos);
                                                 partition_t old_u_id = lp.partition.map(u);
                                                 weight_t u_w = g.weights(u);

                                                 if (acquired + u_w <= cap_d) {
                                                     u32 idx = Kokkos::atomic_fetch_add(&lp.to_move_idx_2(), 1);
                                                     lp.to_move_list_2(idx) = {u, u_w, old_u_id, lp.id(u), 0};
                                                     acquired += u_w;
                                                 } else {
                                                     break;
                                                 }
                                             }
                                         }
                                     }
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // move in block connectivity
        Kokkos::deep_copy(lp.list_size_2, lp.to_move_idx_2);
        move_bc(bc, g, lp.partition, lp.to_move_list_2, lp.list_size_2);
        // moves in partition
        {
            ScopedTimer _t("rebalance", "jetrs", "apply_moves");

            Kokkos::parallel_for("move", lp.list_size_2, KOKKOS_LAMBDA(const u32 i) {
                                     Move m = lp.to_move_list_2(i);
                                     vertex_t u = m.u;
                                     weight_t u_w = m.w;
                                     partition_t old_u_id = m.old_id;
                                     partition_t new_u_id = m.new_id;

                                     lp.partition.map(u) = new_u_id;
                                     Kokkos::atomic_add(&lp.partition.bweights(old_u_id), -u_w);
                                     Kokkos::atomic_add(&lp.partition.bweights(new_u_id), u_w);
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void refine(Graph &g,
                       Partition &partition,
                       partition_t k,
                       weight_t lmax,
                       u32 max_level,
                       u32 level,
                       KokkosMemoryStack &mem_stack) {
        JetLabelPropagation lp = initialize_lp(g.n, g.m, k, lmax, mem_stack);

        if (level == 0) { lp.conn_c = 0.25; }

        lp.sigma_percent = lp.sigma_percent_min + (level == 0 ? 0.0 : lp.sigma_percent * ((f64) level / (f64) max_level));
        lp.sigma = lp.lmax - (weight_t) ((f64) lp.lmax * lp.sigma_percent);
        // copy partition
        {
            ScopedTimer _t("refine", "JetLabelPropagation", "copy_partition");
            copy_into(lp.partition, partition, g.n);
            KOKKOS_PROFILE_FENCE();
        }

        // initial build of block connectivity
        BlockConnectivity bc = from_scratch(g, lp.partition, mem_stack);
        // assert_bc(bc, g, lp.partition, lp.k);

        weight_t best_edge_cut = 0;
        // initial edge cut
        {
            ScopedTimer _t_edge_cut("refine", "JetLabelPropagation", "get_edge_cut");
            best_edge_cut = edge_cut(bc, lp.partition);
            KOKKOS_PROFILE_FENCE();
        }
        weight_t best_weight = 0;
        // initial maximum weight
        {
            ScopedTimer _t_edge_cut("refine", "JetLabelPropagation", "get_max_weight");
            best_weight = max_weight(lp.partition);
            KOKKOS_PROFILE_FENCE();
        }

        weight_t curr_edge_cut = best_edge_cut;
        weight_t curr_weight = best_weight;

        // reset to move array
        {
            ScopedTimer _t("refine", "JetLabelPropagation", "reset_to_move");
            Kokkos::deep_copy(lp.to_move, 0);
            KOKKOS_PROFILE_FENCE();
        }

        u32 weak_iterations = 0;
        u32 iteration = 0;
        while (iteration < lp.n_max_iterations) {
            if (curr_weight <= lp.lmax) {
                jetlp(lp, g, bc);
                // assert_bc(bc, g, lp.partition, lp.k);
                weak_iterations = 0;
            } else {
                if (weak_iterations < lp.max_weak_iterations) {
                    jetrw(lp, g, bc);
                    // assert_bc(bc, g, lp.partition, lp.k);
                    weak_iterations++;
                } else {
                    jetrs(lp, g, bc);
                    // assert_bc(bc, g, lp.partition, lp.k);
                    weak_iterations = 0;
                }
            }

            // recalculate edge cut and max weight
            {
                ScopedTimer _t("refine", "JetLabelPropagation", "get_edge_cut");
                // curr_edge_cut = edge_cut(bc, lp.partition);
                curr_edge_cut = edge_cut_update(curr_edge_cut, g, lp.partition, lp.to_move_list_2, lp.list_size_2, mem_stack);

                KOKKOS_PROFILE_FENCE();
            }
            // recalculate max weight
            {
                ScopedTimer _t("refine", "JetLabelPropagation", "get_max_weight");
                curr_weight = max_weight(lp.partition);
                KOKKOS_PROFILE_FENCE();
            }

            if (curr_weight <= lp.lmax) {
                if (curr_edge_cut < best_edge_cut) {
                    if ((f64) curr_edge_cut < lp.phi * (f64) best_edge_cut) { iteration = 0; }
                    // copy the partition
                    {
                        ScopedTimer _t("refine", "JetLabelPropagation", "copy_partition");

                        copy_into(partition, lp.partition, g.n);
                        best_edge_cut = curr_edge_cut;
                        best_weight = curr_weight;

                        KOKKOS_PROFILE_FENCE();
                    }
                }
            } else if (curr_weight < best_weight) {
                // copy the partition
                {
                    ScopedTimer _t("refine", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_edge_cut = curr_edge_cut;
                    best_weight = curr_weight;

                    KOKKOS_PROFILE_FENCE();
                }
                iteration = 0;
            }

            iteration += 1;
        }

        free_bc(bc, mem_stack);
        free_lp(lp, mem_stack);

        assert_back_is_empty(mem_stack);
    }
}

#endif //GPU_HEIPA_JET_LABEL_PROPAGATION_H
