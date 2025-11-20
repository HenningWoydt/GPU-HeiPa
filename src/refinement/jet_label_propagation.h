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
        //cast to float so we can approximate log_1.5
        f64 gain = static_cast<f64>(g) / static_cast<f64>(vwgt);
        u32 gain_type = 0;
        if (gain > 0.0) {
            gain_type = 0;
        } else if (gain == 0.0) {
            gain_type = 1;
        } else {
            gain_type = MID_BUCKET;
            gain = Kokkos::abs(gain);
            if (gain < 1.0) {
                while (gain < 1.0) {
                    gain *= 1.5;
                    gain_type--;
                }
                if (gain_type < 2) {
                    gain_type = 2;
                }
            } else {
                while (gain > 1.0) {
                    gain /= 1.5;
                    gain_type++;
                }
                if (gain_type > MAX_BUCKETS) {
                    gain_type = MAX_BUCKETS - 1;
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

        u32 round = 0;

        u32 n_max_iterations = 12;
        u32 max_weak_iterations = 2;
        f64 phi = 0.999;
        f64 heavy_alpha = 1.5; // smaller - less vertices moved, larger more vertices moved
        weight_t sigma = 10;
        f64 conn_c = 0.75;
        u32 sections = 1; // adaptive mini buckets per (part,bucket)
        Partition partition;

        UnmanagedDeviceWeight gain;
        UnmanagedDevicePartition id;
        UnmanagedDeviceU32 to_move;

        u32 temp_n_moves = 0;
        Kokkos::View<u32, DeviceMemorySpace> temp_moves_idx;
        UnmanagedDeviceMove temp_moves;
        u32 n_moves = 0;
        Kokkos::View<u32, DeviceMemorySpace> moves_idx;
        UnmanagedDeviceMove moves;

        UnmanagedDeviceWeight bucket_sizes;
        UnmanagedDeviceWeight bucket_offsets;

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
        auto *to_move_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * t_n);
        lp.gain = UnmanagedDeviceWeight(gain_ptr, t_n);
        lp.id = UnmanagedDevicePartition(id_ptr, t_n);
        lp.to_move = UnmanagedDeviceU32(to_move_ptr, t_n);

        auto *temp_moves_ptr = (Move *) get_chunk_back(mem_stack, sizeof(Move) * t_n);
        auto *moves_ptr = (Move *) get_chunk_back(mem_stack, sizeof(Move) * t_n);
        lp.temp_moves_idx = Kokkos::View<u32, DeviceMemorySpace>("temp_moves_idx");
        lp.temp_moves = UnmanagedDeviceMove(temp_moves_ptr, t_n);
        lp.moves_idx = Kokkos::View<u32, DeviceMemorySpace>("moves_idx");
        lp.moves = UnmanagedDeviceMove(moves_ptr, t_n);

        // compute sections so that k * MAX_BUCKETS * sections ≈ n / TARGET_PER_SHARD
        u64 shards = (t_n + TARGET_PER_SHARD - 1) / TARGET_PER_SHARD;
        u64 per_part = MAX_BUCKETS; // buckets per part
        u64 s = (shards + (t_k * per_part - 1)) / (t_k * per_part);
        lp.sections = s == 0 ? 1 : (u32) s;

        auto total_shards = (size_t) t_k * MAX_BUCKETS * lp.sections;

        auto *bucket_sizes_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * total_shards);
        auto *bucket_offsets_ptr = (weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * total_shards);
        lp.bucket_sizes = UnmanagedDeviceWeight(bucket_sizes_ptr, total_shards);
        lp.bucket_offsets = UnmanagedDeviceWeight(bucket_offsets_ptr, total_shards);

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
    }

    inline void jetlp(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        lp.round += 1;

        // for each vertex determine best block
        {
            ScopedTimer _t("refine", "jetlp", "best_block");
            Kokkos::deep_copy(lp.temp_moves_idx, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (lp.to_move(u) != 0) { return; } // already moved

                partition_t u_id = lp.partition.map(u);
                weight_t u_w = g.weights(u);

                weight_t own_conn = 0;
                partition_t best_id = lp.k;
                weight_t best_id_w = -max_sentinel<weight_t>();
                weight_t gain = -max_sentinel<weight_t>();

                bool found = false;
                for (u32 i = bc.row(u); i < bc.row(u + 1); ++i) {
                    partition_t id = bc.ids(i);
                    weight_t w = bc.weights(i);

                    own_conn = id == u_id ? w : own_conn;

                    if (id == lp.k) { continue; } // no valid entry
                    if (id == u_id) { continue; } // do not move to self

                    if (w > best_id_w || (w == best_id_w && id < best_id)) {
                        best_id = id;
                        best_id_w = w;
                        found = true;
                    }
                }

                if (found) {
                    weight_t F = best_id_w - own_conn;

                    // apply first filter
                    bool non_neg = F >= 0;
                    bool conn_filter = -F < Kokkos::floor(lp.conn_c * (f64) own_conn) && false;

                    if (non_neg || conn_filter) {
                        gain = F;
                        u32 idx = Kokkos::atomic_fetch_add(&lp.temp_moves_idx(), 1);
                        lp.temp_moves(idx) = {u, u_w, u_id, best_id, gain};
                    }
                }

                lp.id(u) = best_id;
                lp.gain(u) = gain;
            });
            KOKKOS_PROFILE_FENCE();
        }
        // use afterburner
        {
            ScopedTimer _t("refine", "jetlp", "afterburner");
            Kokkos::deep_copy(lp.temp_n_moves, lp.temp_moves_idx);
            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("afterburner", lp.temp_n_moves, KOKKOS_LAMBDA(const u32 i) {
                Move m = lp.temp_moves(i);
                vertex_t u = m.u;
                weight_t u_w = m.w;
                weight_t u_gain = m.gain;
                partition_t old_u_id = m.old_id;
                partition_t new_u_id = m.new_id;

                weight_t update = 0;
                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                    vertex_t v = g.edges_v(j);

                    weight_t v_gain = lp.gain(v);

                    if (v_gain > u_gain || (v_gain == u_gain && v < u)) {
                        partition_t v_old_id = lp.partition.map(v);
                        partition_t v_new_id = lp.id(v);
                        weight_t w = g.edges_w(j);

                        if (v_new_id == old_u_id) { update -= w; }
                        if (v_new_id == new_u_id) { update += w; }

                        if (v_old_id == old_u_id) { update += w; }
                        if (v_old_id == new_u_id) { update -= w; }
                    }
                }

                if (u_gain + update >= 0) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.moves_idx(), 1);
                    lp.moves(idx) = {u, u_w, old_u_id, new_u_id, u_gain + update};
                    lp.to_move(u) = lp.round;
                }
            });

            Kokkos::deep_copy(lp.n_moves, lp.moves_idx);

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

        weight_t max_b_w = (weight_t) ((f64) lp.lmax * 0.99);
        if (max_b_w < lp.lmax - 100) {
            max_b_w = lp.lmax - 100;
        }

        // determine best block
        {
            ScopedTimer _t("rebalance", "jetrw", "best_block");
            Kokkos::deep_copy(lp.temp_moves_idx, 0);
            Kokkos::deep_copy(lp.bucket_sizes, 0);

            Kokkos::parallel_for("best_block", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                weight_t u_id_w = lp.partition.bweights(u_id);
                weight_t u_w = g.weights(u);

                partition_t best_id = u_id;
                weight_t best_id_w = -1;

                if (u_id_w <= lp.lmax) { return; }                                        // u_id not overloaded
                if ((f64) u_w > lp.heavy_alpha * ((f64) u_id_w - opt_weight)) { return; } // vertex is too heavy

                weight_t own_conn = 0;
                for (u32 i = bc.row(u); i < bc.row(u + 1); ++i) {
                    partition_t id = bc.ids(i);
                    weight_t w = bc.weights(i);

                    own_conn = id == u_id ? w : own_conn;

                    if (id == lp.k) { continue; }                           // no valid entry
                    if (id == u_id) { continue; }                           // dont move to self
                    if (lp.partition.bweights(id) >= max_b_w) { continue; } // dont move into non underloaded partition

                    if (w > best_id_w || (w == best_id_w && id < best_id)) {
                        best_id = id;
                        best_id_w = w;
                    }
                }

                weight_t gain = best_id_w - own_conn;

                if (gain <= 0) {
                    best_id = lp.underloaded_blocks(u % lp.n_underloaded_blocks());
                    gain = -own_conn;
                }

                if (best_id != u_id) {
                    u32 b = gain_bucket(gain, u_w);
                    u32 g_id = (MAX_BUCKETS * u_id + b) * lp.sections + (u % lp.sections);

                    weight_t old = Kokkos::atomic_fetch_add(&lp.bucket_sizes(g_id), u_w);
                    lp.save_atomic(u) = old; // prefix inside this bucket

                    u32 idx = Kokkos::atomic_fetch_add(&lp.temp_moves_idx(), 1);
                    lp.temp_moves(idx) = {u, u_w, u_id, best_id, gain};
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        // determine bucket offsets
        {
            ScopedTimer _t("rebalance", "jetrw", "bucket_offsets");

            // prefix sum for bucket offsets
            Kokkos::parallel_scan("bucket_offsets", lp.k * MAX_BUCKETS * lp.sections, KOKKOS_LAMBDA(const u32 i, weight_t &upd, const bool final_pass) {
                weight_t val = lp.bucket_sizes(i);
                if (final_pass) lp.bucket_offsets(i) = upd; // write old prefix
                upd += val;
            });
            KOKKOS_PROFILE_FENCE();
        }
        // pick prefix
        {
            ScopedTimer _t("rebalance", "jetrw", "pick_prefix");
            Kokkos::deep_copy(lp.temp_n_moves, lp.temp_moves_idx);
            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("pick_prefix", lp.temp_n_moves, KOKKOS_LAMBDA(const u32 i) {
                Move m = lp.temp_moves(i);
                vertex_t u = m.u;
                weight_t u_w = m.w;
                partition_t u_id = m.old_id;
                partition_t new_u_id = m.new_id;
                weight_t gain = m.gain;

                weight_t limit = lp.partition.bweights(u_id) - lp.sigma;

                u32 b = gain_bucket(gain, u_w);
                u32 g_id = (MAX_BUCKETS * u_id + b) * lp.sections + (u % lp.sections);

                u32 width = MAX_BUCKETS * lp.sections;
                u32 p_begin_shard = u_id * width;

                weight_t begin_bucket = lp.bucket_offsets(p_begin_shard);
                weight_t score = lp.save_atomic(u) + lp.bucket_offsets(g_id) - begin_bucket;

                if (score < limit) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.moves_idx(), 1);
                    lp.moves(idx) = {u, u_w, u_id, new_u_id, 0};
                }
            });

            Kokkos::deep_copy(lp.n_moves, lp.moves_idx);

            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void jetrs(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        lp.round += 1;

        weight_t opt_weight = (weight_t) ((f64) g.g_weight / (f64) lp.k);
        weight_t max_b_w = std::max(opt_weight + 1, (weight_t) (lp.lmax * 0.99));
        weight_t max_v_w = 0;
        // determine maximum allowed vertex weight
        {
            Kokkos::parallel_reduce("find max size", lp.k, KOKKOS_LAMBDA(const partition_t id, weight_t &update) {
                weight_t size = lp.partition.bweights(id);
                if (size < max_b_w) {
                    weight_t cap = max_b_w - size;
                    if (cap > update) {
                        update = cap;
                    }
                }
            }, Kokkos::Max<weight_t>(max_v_w));
            KOKKOS_PROFILE_FENCE();
        }
        // eviction candidates
        {
            ScopedTimer _t("rebalance", "jetrs", "score_candidates");

            Kokkos::deep_copy(lp.bucket_sizes, 0);

            Kokkos::parallel_for("score_candidates", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                weight_t u_w = g.weights(u);
                weight_t u_id_w = lp.partition.bweights(u_id);

                // Default: not a candidate
                lp.gain(u) = -max_sentinel<weight_t>();
                lp.save_atomic(u) = 0;

                if (u_id_w <= lp.lmax) { return; }               // u_id not overloaded
                if (u_w > 2 * (u_id_w - opt_weight)) { return; } // vertex is too heavy
                if (u_w > 2 * max_v_w) { return; }               // vertex is too heavy

                weight_t own_conn = 0;
                weight_t sum_other = 0;
                u32 count_other = 0;

                for (u32 i = bc.row(u); i < bc.row(u + 1); ++i) {
                    partition_t id = bc.ids(i);
                    weight_t w = bc.weights(i);

                    own_conn = id == u_id ? w : own_conn;

                    if (id == lp.k) { continue; }                            // no valid entry
                    if (id == u_id) { continue; }                            // dont move to self
                    if (lp.partition.bweights(id) >= lp.sigma) { continue; } // dont count oload blocks

                    sum_other += w;
                    count_other += 1;
                }

                if (count_other == 0) { count_other = 1; }

                weight_t gain = (weight_t) ((f64) sum_other / (f64) count_other) - own_conn;

                // Map to gain bucket
                u32 b = gain_bucket(gain, Kokkos::min(u_w, u_id_w - lp.lmax));
                if (b >= MAX_BUCKETS) { return; }

                u32 g_id = (MAX_BUCKETS * u_id + b) * lp.sections + (u % lp.sections) + 1;

                // Atomic prefix inside this minibucket
                lp.save_atomic(u) = Kokkos::atomic_fetch_add(&lp.bucket_sizes(g_id), u_w);
                lp.gain(u) = gain; // mark as candidate & keep score
            });
            KOKKOS_PROFILE_FENCE();
        }
        // determine bucket offsets
        {
            ScopedTimer _t("rebalance", "jetrs", "bucket_offsets");

            // prefix sum
            Kokkos::parallel_scan("bucket_offsets", lp.k * MAX_BUCKETS * lp.sections, KOKKOS_LAMBDA(const u32 i, weight_t &upd, const bool final_pass) {
                weight_t val = lp.bucket_sizes(i);
                if (final_pass) lp.bucket_offsets(i) = upd;
                upd += val;
            });
            KOKKOS_PROFILE_FENCE();
        }
        // select evicted
        {
            ScopedTimer _t("rebalance", "jetrs", "pick_evictions");

            Kokkos::deep_copy(lp.temp_moves_idx, 0);

            const u32 width = MAX_BUCKETS * lp.sections;

            Kokkos::parallel_for("jetrs_pick_evictions", bc.n, KOKKOS_LAMBDA(const vertex_t u) {
                weight_t gain = lp.gain(u);
                if (gain == -max_sentinel<weight_t>()) { return; } // not a candidate

                partition_t u_id = lp.partition.map(u);
                weight_t u_id_w = lp.partition.bweights(u_id);
                weight_t u_w = g.weights(u);

                weight_t over = u_id_w - lp.lmax;

                u32 b = gain_bucket(gain, Kokkos::min(u_w, u_id_w - lp.lmax));
                u32 g_id = (MAX_BUCKETS * u_id + b) * lp.sections + (u % lp.sections) + 1;
                if (b >= MAX_BUCKETS) { return; }

                // Starting offset for this part's shard range
                u32 p_begin_shard = u_id * width;
                weight_t begin_bucket = lp.bucket_offsets(p_begin_shard);
                weight_t score = lp.save_atomic(u) + lp.bucket_offsets(g_id) - begin_bucket;

                // Evict until we remove "over" weight from part p
                if (score < over) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.temp_moves_idx(), 1);
                    // new_id will be chosen later
                    lp.temp_moves(idx) = {u, u_w, u_id, lp.k, 0};
                }
            });
            KOKKOS_PROFILE_FENCE();

            Kokkos::deep_copy(lp.temp_n_moves, lp.moves_idx);
        }

        if (lp.temp_n_moves == 0) {
            lp.n_moves = 0;
            return;
        }

        // Per-part capacity up to lmax
        Kokkos::View<weight_t *, DeviceMemorySpace> capacity("jetrs_capacity", lp.k);
        Kokkos::View<weight_t *, DeviceMemorySpace> acquired("jetrs_acquired", lp.k);

        // assign destinations
        {
            ScopedTimer _t("rebalance", "jetrs", "compute_capacity");

            Kokkos::deep_copy(capacity, 0);
            Kokkos::deep_copy(acquired, 0);

            // Compute remaining capacity per part
            Kokkos::parallel_for("compute_capacity", lp.k, KOKKOS_LAMBDA(const partition_t p) {
                weight_t size_p = lp.partition.bweights(p);
                capacity(p) = (size_p < lp.lmax) ? (lp.lmax - size_p) : 0;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // Choose best destination per evicted vertex, but respect capacity via atomics
        {
            ScopedTimer _t("rebalance", "jetrs", "assign_destinations");

            Kokkos::parallel_for("choose_dest", lp.temp_n_moves, KOKKOS_LAMBDA(const u32 i) {
                Move m = lp.temp_moves(i);
                vertex_t u = m.u;
                weight_t u_w = m.w;
                partition_t old_u_id = m.old_id;

                // Find best-connected destination block with some remaining capacity
                partition_t best_id = old_u_id;
                weight_t best_w = -max_sentinel<weight_t>();

                for (u32 j = bc.row(u); j < bc.row(u + 1); ++j) {
                    partition_t id = bc.ids(j);
                    weight_t w = bc.weights(j);

                    if (id == lp.k) continue;        // invalid
                    if (id == old_u_id) continue;    // don't "move" to self
                    if (capacity(id) <= 0) continue; // no capacity at all (fast filter)

                    if (w > best_w || (w == best_w && id < best_id)) {
                        best_w = w;
                        best_id = id;
                    }
                }

                // Try to reserve capacity at best_id
                partition_t final_id = old_u_id;

                if (best_id != old_u_id) {
                    // Reserve capacity atomically
                    weight_t old_acc = Kokkos::atomic_fetch_add(&acquired(best_id), u_w);
                    if (old_acc + u_w <= capacity(best_id)) {
                        // We fit → accept move
                        final_id = best_id;
                    } else {
                        // We overfilled → roll back and keep vertex in old part
                        Kokkos::atomic_add(&acquired(best_id), -u_w);
                    }
                }

                lp.temp_moves(i) = {u, u_w, old_u_id, final_id, 0};
            });
            KOKKOS_PROFILE_FENCE();
        }
        // compact moves
        {
            ScopedTimer _t("rebalance", "jetrs", "compact_moves");

            Kokkos::deep_copy(lp.moves_idx, 0);

            Kokkos::parallel_for("jetrs_compact_moves", lp.temp_n_moves, KOKKOS_LAMBDA(const u32 i) {
                Move m = lp.temp_moves(i);
                if (m.old_id != m.new_id && m.new_id != lp.k) {
                    u32 idx = Kokkos::atomic_fetch_add(&lp.moves_idx(), 1);
                    lp.moves(idx) = m;
                }
            });

            Kokkos::deep_copy(lp.n_moves, lp.moves_idx);

            KOKKOS_PROFILE_FENCE();
        }
    }

    inline void refine(Graph &g,
                       Partition &partition,
                       partition_t k,
                       weight_t lmax,
                       u32 level,
                       KokkosMemoryStack &mem_stack) {
        JetLabelPropagation lp = initialize_lp(g.n, g.m, k, lmax, mem_stack);

        if (level == 0) { lp.conn_c = 0.25; }

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

        // init arrays
        {
            ScopedTimer _t("refine", "JetLabelPropagation", "init_arrays");
            Kokkos::deep_copy(lp.to_move, 0);
            KOKKOS_PROFILE_FENCE();
        }

        u32 balance_iterations = 0;
        u32 iteration = 0;
        u32 total_n_iteration = 0;
        while (iteration < lp.n_max_iterations) {
            iteration += 1;
            total_n_iteration += 1;

            if (curr_weight <= lp.lmax) {
                jetlp(lp, g, bc);
                std::cout << level << " jetlp " << lp.n_moves << " ";
                // assert_bc(bc, g, lp.partition, lp.k);
                balance_iterations = 0;
            } else {
                if (balance_iterations < lp.max_weak_iterations) {
                    jetrw(lp, g, bc);
                    std::cout << level << " jetrw " << lp.n_moves << " ";
                    // assert_bc(bc, g, lp.partition, lp.k);
                } else {
                    jetrs(lp, g, bc);
                    std::cout << level << " jetrs " << lp.n_moves << " ";
                    // assert_bc(bc, g, lp.partition, lp.k);
                }
                balance_iterations++;
            }

            // move in block connectivity
            move_bc(bc, g, lp.partition, lp.moves, lp.n_moves);
            // moves in partition
            {
                ScopedTimer _t("refine", "JetLabelPropagation", "apply_moves");

                Kokkos::parallel_for("move", lp.n_moves, KOKKOS_LAMBDA(const u32 i) {
                    Move m = lp.moves(i);
                    vertex_t u = m.u;
                    weight_t u_w = m.w;
                    partition_t u_old_id = m.old_id;
                    partition_t u_new_id = m.new_id;

                    lp.partition.map(u) = u_new_id;
                    Kokkos::atomic_add(&lp.partition.bweights(u_old_id), -u_w);
                    Kokkos::atomic_add(&lp.partition.bweights(u_new_id), u_w);
                });
                KOKKOS_PROFILE_FENCE();
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

            std::cout << total_n_iteration << " " << curr_edge_cut << " " << curr_weight << std::endl;

            if (best_weight > lp.lmax && curr_weight < best_weight) {
                // copy the partition
                {
                    ScopedTimer _t("refine", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_edge_cut = curr_edge_cut;
                    best_weight = curr_weight;
                    iteration = 0;

                    KOKKOS_PROFILE_FENCE();
                }
            } else if (curr_edge_cut < best_edge_cut && (curr_weight <= lp.lmax || curr_weight < best_weight)) {
                if ((f64) curr_edge_cut < lp.phi * (f64) best_edge_cut) { iteration = 0; } {
                    ScopedTimer _t("refine", "JetLabelPropagation", "copy_partition");

                    copy_into(partition, lp.partition, g.n);
                    best_edge_cut = curr_edge_cut;
                    best_weight = curr_weight;

                    KOKKOS_PROFILE_FENCE();
                }
            }
        }

        free_bc(bc, mem_stack);
        free_lp(lp, mem_stack);

        assert_back_is_empty(mem_stack);
    }
}

#endif //GPU_HEIPA_JET_LABEL_PROPAGATION_H
