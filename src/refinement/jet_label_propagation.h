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

    struct JetLabelPropagation {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;

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

        UnmanagedDeviceWeight conn;
        UnmanagedDeviceU64 weight_id;
        UnmanagedDeviceWeight gain2;
        UnmanagedDeviceU32 locked;
        UnmanagedDeviceU32 in_X;
        UnmanagedDeviceU32 to_move;

        UnmanagedDeviceVertex to_move_list;

        UnmanagedDeviceU32 bucket_counts;
        UnmanagedDeviceU32 bucket_offsets;
        UnmanagedDeviceU32 bucket_cursor;
        UnmanagedDeviceVertex flat_buckets;
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
        lp.sigma = lp.lmax - (weight_t) ((f64) lp.lmax * lp.sigma_percent);

        lp.partition = initialize_partition(t_n, t_k, t_lmax, mem_stack);

        auto *conn_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n);
        auto *weight_id_ptr = (u64 *) get_chunk_back(mem_stack, sizeof(u64) * t_n);
        auto *gain2_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * t_n);
        auto *locked_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n);
        auto *in_X_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n);
        auto *to_move_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n);
        auto *to_move_list_ptr = (vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * t_n);
        lp.conn = UnmanagedDeviceWeight(conn_ptr, t_n);
        lp.weight_id = UnmanagedDeviceU64(weight_id_ptr, t_n);
        lp.gain2 = UnmanagedDeviceWeight(gain2_ptr, t_n);
        lp.locked = UnmanagedDeviceU32(locked_ptr, t_n);
        lp.in_X = UnmanagedDeviceU32(in_X_ptr, t_n);
        lp.to_move = UnmanagedDeviceU32(to_move_ptr, t_n);
        lp.to_move_list = UnmanagedDeviceVertex(to_move_list_ptr, t_n);

        // compute sections so that k * MAX_BUCKETS * sections ≈ n / TARGET_PER_SHARD
        u64 shards = (t_n + TARGET_PER_SHARD - 1) / TARGET_PER_SHARD;
        u64 per_part = MAX_BUCKETS; // buckets per part
        u64 s = (shards + (t_k * per_part - 1)) / (t_k * per_part);
        lp.sections = s == 0 ? 1 : (u32) s;

        auto total_shards = (size_t) t_k * MAX_BUCKETS * lp.sections;

        auto *bucket_counts_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * total_shards);
        auto *bucket_offsets_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * total_shards);
        auto *bucket_cursor_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * total_shards);
        auto *flat_buckets_ptr = (vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * lp.n);
        lp.bucket_counts = UnmanagedDeviceU32(bucket_counts_ptr, total_shards);
        lp.bucket_offsets = UnmanagedDeviceU32(bucket_offsets_ptr, total_shards);
        lp.bucket_cursor = UnmanagedDeviceU32(bucket_cursor_ptr, total_shards);
        lp.flat_buckets = UnmanagedDeviceVertex(flat_buckets_ptr, lp.n);

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
    }

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
    u32 idx_psm(partition_t p, u32 s, u32 m, const JetLabelPropagation &lp) {
        return (p * MAX_BUCKETS + s) * lp.sections + m;
    }

    KOKKOS_INLINE_FUNCTION
    u32 idx_dsm(partition_t d, u32 s, u32 m, const JetLabelPropagation &lp) {
        return (d * MAX_BUCKETS + s) * lp.sections + m;
    }

    KOKKOS_INLINE_FUNCTION
    bool ord_smaller(const JetLabelPropagation &lp, vertex_t u, vertex_t v) {
        weight_t gain_u = unpack_score(lp.weight_id(u));
        weight_t gain_v = unpack_score(lp.weight_id(v));

        if (gain_u > gain_v) { return true; }
        if (gain_u == gain_v && u < v) { return true; }
        return false;
    }

    inline void jetlp(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc,
                      KokkosMemoryStack &mem_stack) {
        // reset all arrays
        {
            ScopedTimer _t("refine", "jetlp", "reset");

            Kokkos::deep_copy(lp.to_move, 0);
            Kokkos::deep_copy(lp.gain2, -max_sentinel<weight_t>());
            Kokkos::deep_copy(lp.conn, 0);
            Kokkos::deep_copy(lp.in_X, 0);
            Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     lp.weight_id(u) = pack_s32_partition(-max_sentinel<weight_t>(), lp.k);
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // for each vertex determine best block
        {
            ScopedTimer _t("refine", "jetlp", "best_block");

            Kokkos::parallel_for("best_block", bc.size, KOKKOS_LAMBDA(const u32 i) {
                                     vertex_t u = bc.us(i);
                                     partition_t id = bc.ids(i);
                                     weight_t w = bc.weights(i);

                                     partition_t u_id = lp.partition.map(u);
                                     weight_t u_w = g.weights(u);

                                     if (lp.locked(u) == 1) { return; }

                                     if (id == lp.k) { return; }
                                     if (id == u_id) {
                                         lp.conn(u) = w;
                                         return;
                                     }
                                     if (lp.partition.bweights(id) + u_w > lp.lmax) { return; }

                                     Kokkos::atomic_max(&lp.weight_id(u), pack_s32_partition(w, id));
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // apply first filter
        {
            ScopedTimer _t("refine", "jetlp", "first_filter");

            Kokkos::parallel_for("first_filter", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     partition_t new_id = unpack_partition(lp.weight_id(u));
                                     weight_t F = unpack_score(lp.weight_id(u)) - lp.conn(u);
                                     if (new_id == lp.k) { return; }

                                     bool non_neg = F >= 0;
                                     bool conn_filter = -F < Kokkos::floor(lp.conn_c * (f64) lp.conn(u));

                                     if (non_neg || conn_filter) {
                                         lp.in_X(u) = 1;
                                         lp.gain2(u) = 0;
                                     }
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // use afterburner
        {
            ScopedTimer _t("refine", "jetlp", "afterburner");

            Kokkos::parallel_for("afterburner", g.m, KOKKOS_LAMBDA(const u32 i) {
                                     vertex_t u = g.edges_u(i);
                                     vertex_t v = g.edges_v(i);
                                     weight_t w = g.edges_w(i);

                                     if (lp.in_X(u) == 0) { return; }

                                     partition_t old_id = lp.partition.map(u);
                                     partition_t new_id = unpack_partition(lp.weight_id(u));

                                     // decide assumed part of neighbor v
                                     partition_t v_id = lp.partition.map(v);
                                     if (lp.in_X(v) == 1 && ord_smaller(lp, u, v)) {
                                         v_id = unpack_partition(lp.weight_id(v)); // assume v was moved
                                     }

                                     weight_t edge_gain = w * ((old_id != v_id) - (new_id != v_id)); // get_diff(d_oracle, old_id, new_id, v_id);
                                     Kokkos::atomic_add(&lp.gain2(u), edge_gain);
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // nonnegative filter
        {
            ScopedTimer _t("refine", "jetlp", "nonnegative_filter");

            Kokkos::parallel_for("nonnegative_filter", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     lp.to_move(u) = lp.gain2(u) >= 0;
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // lock for next iteration
        {
            ScopedTimer _t("refine", "jetlp", "copy_locked");

            Kokkos::deep_copy(lp.locked, lp.to_move);
            KOKKOS_PROFILE_FENCE();
        }
        // move in block connectivity
        move(bc, g, lp.partition, lp.to_move, lp.weight_id, mem_stack);
        // moves in partition
        {
            ScopedTimer _t("refine", "jetlp", "apply_moves");

            Kokkos::parallel_for("move", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     if (lp.to_move(u) == 1) {
                                         weight_t u_w = g.weights(u);
                                         partition_t u_id = lp.partition.map(u);
                                         partition_t v_id = unpack_partition(lp.weight_id(u));

                                         lp.partition.map(u) = v_id;
                                         Kokkos::atomic_add(&lp.partition.bweights(u_id), -u_w);
                                         Kokkos::atomic_add(&lp.partition.bweights(v_id), u_w);
                                     }
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void jetrw(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc,
                      KokkosMemoryStack &mem_stack) {
        // reset arrays
        {
            ScopedTimer _t("rebalance", "jetrw", "reset");

            Kokkos::deep_copy(lp.bucket_counts, 0);
            Kokkos::deep_copy(lp.to_move, 0);
            Kokkos::deep_copy(lp.conn, 0);
            Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     lp.weight_id(u) = pack_s32_partition(-max_sentinel<weight_t>(), lp.k);
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // determine best block
        {
            ScopedTimer _t("rebalance", "jetrw", "best_block");

            Kokkos::parallel_for("best_block", bc.size, KOKKOS_LAMBDA(const u32 i) {
                                     vertex_t u = bc.us(i);
                                     partition_t id = bc.ids(i);
                                     weight_t w = bc.weights(i);

                                     partition_t u_id = lp.partition.map(u);
                                     weight_t u_w = g.weights(u);

                                     if (id == lp.k) { return; }                                                                                            // no valid entry
                                     if (lp.partition.bweights(u_id) <= lp.lmax) { return; }                                                                // u_id not overloaded
                                     if ((f64) g.weights(u) > lp.heavy_alpha * (lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex is too heavy
                                     if (lp.partition.bweights(id) + u_w > lp.lmax) { return; }                                                             // dont move into overloaded partition

                                     if (u_id == id) {
                                         lp.conn(u) = w;
                                         return;
                                     }

                                     Kokkos::atomic_max(&lp.weight_id(u), pack_s32_partition(w, id));
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // count number of elements in buckets
        {
            ScopedTimer _t("rebalance", "jetrw", "count_buckets");

            Kokkos::parallel_for("count_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     partition_t u_id = lp.partition.map(u);
                                     partition_t new_id = unpack_partition(lp.weight_id(u));
                                     weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

                                     if (new_id == lp.k) { return; }                                                                                              // no block found
                                     if (lp.partition.bweights(u_id) <= lp.lmax) { return; }                                                                      // u_id not overloaded
                                     if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

                                     u32 b = gain_bucket(gain, g.weights(u));
                                     u32 mini = u % lp.sections;
                                     Kokkos::atomic_inc(&lp.bucket_counts(idx_dsm(new_id, b, mini, lp)));
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // determine bucket offsets
        {
            ScopedTimer _t("rebalance", "jetrw", "bucket_offsets");

            // prefix sum for bucket offsets
            Kokkos::parallel_scan("bucket_offsets", lp.k * MAX_BUCKETS * lp.sections, KOKKOS_LAMBDA(const u32 i, u32 &upd, const bool final_pass) {
                                      u32 val = lp.bucket_counts(i);
                                      if (final_pass) lp.bucket_offsets(i) = upd; // write old prefix
                                      upd += val;
                                  }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // copy cursor
        {
            ScopedTimer _t("rebalance", "jetrw", "copy_cursor");

            Kokkos::deep_copy(lp.bucket_cursor, lp.bucket_offsets);
        }
        // fill buckets
        {
            ScopedTimer _t("rebalance", "jetrw", "fill_buckets");

            Kokkos::parallel_for("fill_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     partition_t u_id = lp.partition.map(u);
                                     partition_t new_id = unpack_partition(lp.weight_id(u));
                                     weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

                                     if (new_id == lp.k) { return; }                                                                                              // not a candidate
                                     if (lp.partition.bweights(u_id) <= lp.lmax) { return; }                                                                      // u_id not overloaded
                                     if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

                                     u32 b = gain_bucket(gain, g.weights(u));
                                     u32 mini = u % lp.sections;
                                     u32 pos = Kokkos::atomic_fetch_inc(&lp.bucket_cursor(idx_dsm(new_id, b, mini, lp)));
                                     lp.flat_buckets(pos) = u;
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // pick prefix
        {
            ScopedTimer _t("rebalance", "jetrw", "pick_prefix");

            Kokkos::parallel_for("pick_prefix", lp.k, KOKKOS_LAMBDA(const partition_t id) {
                                     if (lp.partition.bweights(id) <= lp.sigma) { return; }

                                     weight_t min_to_lose = lp.partition.bweights(id) - lp.sigma;
                                     weight_t moved = 0;

                                     // slots in increasing loss order
                                     for (u32 b = 0; b < MAX_BUCKETS && moved < min_to_lose; ++b) {
                                         for (u32 r = 0; r < lp.sections && moved < min_to_lose; ++r) {
                                             u32 b_idx = idx_psm(id, b, r, lp);
                                             u32 beg = lp.bucket_offsets(b_idx);
                                             u32 end = lp.bucket_offsets(b_idx) + lp.bucket_counts(b_idx);

                                             for (u32 pos = beg; pos < end && moved < min_to_lose; ++pos) {
                                                 vertex_t u = lp.flat_buckets(pos);
                                                 weight_t wu = g.weights(u);

                                                 if (moved < min_to_lose) {
                                                     lp.to_move(u) = 1; // mark selection
                                                     moved += wu;
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
        move(bc, g, lp.partition, lp.to_move, lp.weight_id, mem_stack);
        // moves in partition
        {
            ScopedTimer _t("rebalance", "jetrw", "apply_moves");

            Kokkos::parallel_for("move", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     if (lp.to_move(u) == 1) {
                                         weight_t u_w = g.weights(u);
                                         partition_t u_id = lp.partition.map(u);
                                         partition_t v_id = unpack_partition(lp.weight_id(u));

                                         lp.partition.map(u) = v_id;
                                         Kokkos::atomic_add(&lp.partition.bweights(u_id), -u_w);
                                         Kokkos::atomic_add(&lp.partition.bweights(v_id), u_w);
                                     }
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void jetrs(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc,
                      KokkosMemoryStack &mem_stack) {
        // reset arrays
        {
            ScopedTimer _t("rebalance", "jetrs", "reset");

            Kokkos::deep_copy(lp.bucket_counts, 0);
            Kokkos::deep_copy(lp.to_move, 0);
            Kokkos::deep_copy(lp.conn, 0);
            Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     lp.weight_id(u) = pack_s32_partition(-max_sentinel<weight_t>(), lp.k);
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // determine best block
        {
            ScopedTimer _t("rebalance", "jetrs", "best_block");

            Kokkos::parallel_for("best_block", bc.size, KOKKOS_LAMBDA(const u32 i) {
                                     vertex_t u = bc.us(i);
                                     partition_t id = bc.ids(i);
                                     weight_t w = bc.weights(i);

                                     partition_t u_id = lp.partition.map(u);
                                     weight_t u_w = g.weights(u);

                                     if (id == lp.k) { return; }                                                                                            // no valid entry
                                     if (lp.partition.bweights(u_id) <= lp.lmax) { return; }                                                                // u_id not overloaded
                                     if ((f64) g.weights(u) > lp.heavy_alpha * (lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex is too heavy
                                     if (lp.partition.bweights(id) + u_w > lp.lmax) { return; }                                                             // dont move into overloaded partition

                                     if (u_id == id) {
                                         lp.conn(u) = w;
                                         return;
                                     }

                                     Kokkos::atomic_max(&lp.weight_id(u), pack_s32_partition(w, id));
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // counter number of elements in each bucket
        {
            ScopedTimer _t("rebalance", "jetrs", "count_buckets");

            // 2) For each vertex u in overloaded sources: pick best destination d with cap>0
            Kokkos::parallel_for("count_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     partition_t u_id = lp.partition.map(u);
                                     partition_t new_id = unpack_partition(lp.weight_id(u));
                                     weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

                                     if (new_id == lp.k) { return; }                                                                                              // no block found
                                     if (lp.partition.bweights(u_id) <= lp.lmax) { return; }                                                                      // u_id not overloaded
                                     if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

                                     u32 b = gain_bucket(gain, g.weights(u));
                                     u32 mini = u % lp.sections;
                                     Kokkos::atomic_inc(&lp.bucket_counts(idx_dsm(new_id, b, mini, lp)));
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

            Kokkos::parallel_for("fill_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     partition_t u_id = lp.partition.map(u);
                                     partition_t new_id = unpack_partition(lp.weight_id(u));
                                     weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

                                     if (new_id == lp.k) { return; }                                                                                              // not a candidate
                                     if (lp.partition.bweights(u_id) <= lp.lmax) { return; }                                                                      // u_id not overloaded
                                     if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

                                     u32 b = gain_bucket(gain, g.weights(u));
                                     u32 mini = u % lp.sections;
                                     u32 pos = Kokkos::atomic_fetch_inc(&lp.bucket_cursor(idx_dsm(new_id, b, mini, lp)));
                                     lp.flat_buckets(pos) = u;
                                 }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // pick prefix
        {
            ScopedTimer _t("rebalance", "jetrs", "pick_prefix");

            // 4) Per-destination prefix selection: honor cap[d]
            Kokkos::parallel_for("pick_prefix", lp.k, KOKKOS_LAMBDA(const partition_t d) {
                                     weight_t w = lp.partition.bweights(d);
                                     weight_t cap_d = (w < lp.lmax) ? (lp.lmax - w) : 0;
                                     if (cap_d <= 0) return;

                                     weight_t acquired = 0;
                                     for (u32 s = 0; s < MAX_BUCKETS && acquired < cap_d; ++s) {
                                         for (u32 mini = 0; mini < lp.sections && acquired < cap_d; ++mini) {
                                             u32 ridx = idx_dsm(d, s, mini, lp);
                                             u32 beg = lp.bucket_offsets(ridx);
                                             u32 end = lp.bucket_offsets(ridx) + lp.bucket_counts(ridx);

                                             for (u32 pos = beg; pos < end && acquired < cap_d; ++pos) {
                                                 vertex_t u = lp.flat_buckets(pos);

                                                 // partition_t pu = lp.p_manager.partition(u);
                                                 // if (lp.p_manager.bweights(pu) <= lp.lmax) continue;

                                                 weight_t wu = g.weights(u);
                                                 if (acquired + wu <= cap_d) {
                                                     lp.to_move(u) = 1;
                                                     acquired += wu;
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
        move(bc, g, lp.partition, lp.to_move, lp.weight_id, mem_stack);
        // moves in partition
        {
            ScopedTimer _t("rebalance", "jetrs", "apply_moves");

            Kokkos::parallel_for("move", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                     if (lp.to_move(u) == 1) {
                                         weight_t u_w = g.weights(u);
                                         partition_t u_id = lp.partition.map(u);
                                         partition_t v_id = unpack_partition(lp.weight_id(u));

                                         lp.partition.map(u) = v_id;
                                         Kokkos::atomic_add(&lp.partition.bweights(u_id), -u_w);
                                         Kokkos::atomic_add(&lp.partition.bweights(v_id), u_w);
                                     }
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

        bool executed_rw = false;
        weight_t last_rw_edge_cut = max_sentinel<weight_t>();
        weight_t last_rw_weight = max_sentinel<weight_t>();

        // open lock for all vertices
        {
            ScopedTimer _t("refine", "JetLabelPropagation", "reset_lock");
            Kokkos::deep_copy(lp.locked, 0);
            KOKKOS_PROFILE_FENCE();
        }

        u32 weak_iterations = 0;
        u32 iteration = 0;
        while (iteration < lp.n_max_iterations) {
            executed_rw = false;
            if (curr_weight <= lp.lmax) {
                jetlp(lp, g, bc, mem_stack);
                weak_iterations = 0;
            } else {
                // open locks
                {
                    ScopedTimer _t("refine", "JetLabelPropagation", "reset_lock");
                    Kokkos::deep_copy(lp.locked, 0);
                    KOKKOS_PROFILE_FENCE();
                }
                if (weak_iterations < lp.max_weak_iterations) {
                    jetrw(lp, g, bc, mem_stack);
                    weak_iterations++;
                    executed_rw = true;
                } else {
                    jetrs(lp, g, bc, mem_stack);
                    weak_iterations = 0;
                }
            }

            // recalculate edge cut and max weight
            {
                ScopedTimer _t("refine", "JetLabelPropagation", "get_edge_cut");
                curr_edge_cut = edge_cut(bc, lp.partition);
                KOKKOS_PROFILE_FENCE();
            }
            // recalculate max weight
            {
                ScopedTimer _t("refine", "JetLabelPropagation", "get_max_weight");
                curr_weight = max_weight(lp.partition);
                KOKKOS_PROFILE_FENCE();
            }

            if (executed_rw) {
                if (curr_edge_cut == last_rw_edge_cut && last_rw_weight == curr_weight) {
                    // we executed rw, but it stayed the same, another rw call will change nothing
                    weak_iterations = lp.max_weak_iterations;
                } else {
                    last_rw_edge_cut = curr_edge_cut;
                    last_rw_weight = curr_weight;
                }
            } else {
                last_rw_edge_cut = max_sentinel<weight_t>();
                last_rw_weight = max_sentinel<weight_t>();
            }

            if (curr_weight <= lp.lmax) {
                if (curr_edge_cut < best_edge_cut) {
                    if ((f64) curr_edge_cut < lp.phi * (f64) best_edge_cut) { iteration = 0; }
                    // copy the partition
                    {
                        ScopedTimer _t("refine", "JetLabelPropagation", "copy_partition");
                        copy_into(partition, lp.partition, g.n);
                        KOKKOS_PROFILE_FENCE();
                    }
                    best_edge_cut = curr_edge_cut;
                    best_weight = curr_weight;
                }
            } else if (curr_weight < best_weight) {
                // copy the partition
                {
                    ScopedTimer _t("refine", "JetLabelPropagation", "copy_partition");
                    copy_into(partition, lp.partition, g.n);
                    KOKKOS_PROFILE_FENCE();
                }
                best_edge_cut = curr_edge_cut;
                best_weight = curr_weight;
                iteration = 0;
            }

            iteration += 1;
        }

        free_bc(bc, mem_stack);
        free_lp(lp, mem_stack);

        std::cout << level << " " << mem_stack.n_bytes_in_use << " " << mem_stack.n_bytes_in_use_back << std::endl;
    }
}

#endif //GPU_HEIPA_JET_LABEL_PROPAGATION_H
