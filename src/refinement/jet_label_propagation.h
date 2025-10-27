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
#include "../datastructures/partition.h"
#include "../datastructures/block_connectivity.h"

namespace GPU_HeiPa {
    constexpr u32 MAX_SLOTS = 10;
    constexpr u32 RHO = 4;

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
        Partition partition;

        DeviceWeight conn;
        DeviceU64 weight_id;
        DeviceWeight gain2;
        DeviceU32 locked;
        DeviceU32 in_X;
        DeviceU32 to_move;

        DeviceU32 bucket_counts;
        DeviceU32 bucket_offsets;
        DeviceU32 bucket_cursor;
        DeviceVertex flat_buckets;
    };

    inline JetLabelPropagation initialize_lp(const vertex_t t_n,
                                             const vertex_t t_m,
                                             const partition_t t_k,
                                             const weight_t t_lmax) {
        ScopedTimer _t("io", "JetLabelPropagation", "allocate");

        JetLabelPropagation lp;

        lp.n = t_n;
        lp.m = t_m;
        lp.k = t_k;
        lp.lmax = t_lmax;
        lp.sigma = lp.lmax - (weight_t) ((f64) lp.lmax * lp.sigma_percent);

        lp.partition = initialize_partition(t_n, t_k, t_lmax);
        lp.conn = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "conn"), lp.n);
        lp.weight_id = DeviceU64(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weight_id"), lp.n);
        lp.gain2 = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gain2"), lp.n);
        lp.locked = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "locked"), lp.n);
        lp.in_X = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "in_X"), lp.n);
        lp.to_move = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "to_move"), lp.n);

        lp.bucket_counts = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "bucket_counts"), lp.k * MAX_SLOTS * RHO);
        lp.bucket_offsets = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "bucket_offsets"), lp.k * MAX_SLOTS * RHO);
        lp.bucket_cursor = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "bucket_cursor"), lp.k * MAX_SLOTS * RHO);
        lp.flat_buckets = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "flat_buckets"), lp.n);

        return lp;
    }

    KOKKOS_INLINE_FUNCTION
    u32 loss_slot(const weight_t gain, const JetLabelPropagation &lp) {
        if (gain > 0) return 0; // best: positive gain
        if (gain == 0) return 1; // tie

        u32 s = 2 + floor_log2_u64((u64) -gain); // negative gain: log2 buckets
        return s < MAX_SLOTS ? s : MAX_SLOTS - 1;
    }

    KOKKOS_INLINE_FUNCTION
    u32 idx_psm(partition_t p, u32 s, u32 m, const JetLabelPropagation &lp) {
        return (p * MAX_SLOTS + s) * RHO + m;
    }

    KOKKOS_INLINE_FUNCTION
    u32 idx_dsm(partition_t d, u32 s, u32 m, const JetLabelPropagation &lp) {
        return (d * MAX_SLOTS + s) * RHO + m;
    }

    KOKKOS_INLINE_FUNCTION
    bool ord_smaller(const JetLabelPropagation &lp, vertex_t u, vertex_t v) {
        weight_t gain_u = unpack_score(lp.weight_id(u));
        weight_t gain_v = unpack_score(lp.weight_id(v));

        if (gain_u > gain_v) { return true; }
        if (gain_u == gain_v && u < v) { return true; }
        return false;
    }

    KOKKOS_INLINE_FUNCTION
    partition_t random_partition(vertex_t u, u32 seed, u32 prime, u32 xor_const, partition_t k) {
        // Mix in the seed with a 32-bit hash style formula
        u32 key = (u * prime) ^ (xor_const + seed * 0x9e3779b9u); // 0x9e3779b9u is 32-bit golden ratio
        key ^= key >> 16;
        key *= 0x85ebca6bu; // Murmur3 finalizer constants
        key ^= key >> 13;
        key *= 0xc2b2ae35u;
        key ^= key >> 16;

        return key % k;
    }

    inline void apply_moves(JetLabelPropagation &lp,
                            Graph &g,
                            BlockConnectivity &bc) {
        move(bc, g, lp.partition, lp.to_move, lp.weight_id);

        Kokkos::parallel_for("move", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            if (lp.to_move(u) == 1) {
                weight_t u_w = g.weights(u);
                partition_t u_id = lp.partition.map(u);
                partition_t v_id = unpack_partition(lp.weight_id(u));

                lp.partition.map(u) = v_id;
                Kokkos::atomic_add(&lp.partition.bweights(u_id), -u_w);
                Kokkos::atomic_add(&lp.partition.bweights(v_id), u_w);
            }
        });
        Kokkos::fence();
    }

    inline void jetlp(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        ScopedTimer _t_allocate("refine", "jetlp", "allocate");

        Kokkos::deep_copy(lp.to_move, 0);
        Kokkos::deep_copy(lp.gain2, -max_sentinel<weight_t>());
        Kokkos::deep_copy(lp.conn, 0);
        Kokkos::deep_copy(lp.in_X, 0);
        Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            lp.weight_id(u) = pack_s32_partition(-max_sentinel<weight_t>(), lp.k);
        });
        Kokkos::fence();

        _t_allocate.stop();
        ScopedTimer _t_determine_max_delta("refine", "jetlp", "best_block");

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
        Kokkos::fence();

        _t_determine_max_delta.stop();
        ScopedTimer _t_first_filter("refine", "jetlp", "first_filter");

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
        Kokkos::fence();

        _t_first_filter.stop();
        ScopedTimer _t_afterburner("refine", "jetlp", "afterburner");

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
        });
        Kokkos::fence();

        _t_afterburner.stop();
        ScopedTimer _t_nonnegative_filter("refine", "jetlp", "nonnegative_filter");

        Kokkos::parallel_for("nonnegative_filter", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            lp.to_move(u) = lp.gain2(u) >= 0;
        });
        Kokkos::fence();

        _t_nonnegative_filter.stop();
        ScopedTimer _t_copy_locked("refine", "jetlp", "copy_locked");

        Kokkos::deep_copy(lp.locked, lp.to_move);
        Kokkos::fence();

        _t_copy_locked.stop();
        ScopedTimer _t_apply_moves("refine", "jetlp", "apply_moves");

        apply_moves(lp, g, bc);
    }

    inline void jetrw(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        ScopedTimer _t_allocate("rebalance", "jetrw", "allocate");

        Kokkos::deep_copy(lp.bucket_counts, 0);
        Kokkos::deep_copy(lp.to_move, 0);
        Kokkos::deep_copy(lp.conn, 0);
        Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            lp.weight_id(u) = pack_s32_partition(-max_sentinel<weight_t>(), lp.k);
        });
        Kokkos::fence();

        _t_allocate.stop();
        ScopedTimer _t_best_block("rebalance", "jetrw", "best_block");

        Kokkos::parallel_for("best_block", bc.size, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = bc.us(i);
            partition_t id = bc.ids(i);
            weight_t w = bc.weights(i);

            partition_t u_id = lp.partition.map(u);
            weight_t u_w = g.weights(u);

            if (id == lp.k) { return; } // no valid entry
            if (lp.partition.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) g.weights(u) > lp.heavy_alpha * (lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex is too heavy
            if (lp.partition.bweights(id) + u_w > lp.lmax) { return; } // dont move into overloaded partition

            if (u_id == id) {
                lp.conn(u) = w;
                return;
            }

            Kokkos::atomic_max(&lp.weight_id(u), pack_s32_partition(w, id));
        });
        Kokkos::fence();

        _t_best_block.stop();
        ScopedTimer _t_count_buckets("rebalance", "jetrw", "count_buckets");

        Kokkos::parallel_for("count_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = lp.partition.map(u);
            partition_t new_id = unpack_partition(lp.weight_id(u));
            weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

            if (new_id == lp.k) { return; } // no block found
            if (lp.partition.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

            u32 s = loss_slot(gain, lp);
            u32 mini = u % RHO;
            Kokkos::atomic_inc(&lp.bucket_counts(idx_dsm(new_id, s, mini, lp)));
        });
        Kokkos::fence();

        _t_count_buckets.stop();
        ScopedTimer _t_bucket_offsets("rebalance", "jetrw", "bucket_offsets");

        // prefix sum for bucket offsets
        Kokkos::parallel_scan("bucket_offsets", lp.k * MAX_SLOTS * RHO,KOKKOS_LAMBDA(const u32 i, u32 &upd, const bool final_pass) {
            u32 val = lp.bucket_counts(i);
            if (final_pass) lp.bucket_offsets(i) = upd; // write old prefix
            upd += val;
        });
        Kokkos::fence();

        _t_bucket_offsets.stop();
        ScopedTimer _t_fill_buckets("rebalance", "jetrw", "fill_buckets");

        // fill
        Kokkos::deep_copy(lp.bucket_cursor, lp.bucket_offsets);
        Kokkos::fence();

        Kokkos::parallel_for("fill_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = lp.partition.map(u);
            partition_t new_id = unpack_partition(lp.weight_id(u));
            weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

            if (new_id == lp.k) { return; } // not a candidate
            if (lp.partition.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

            u32 s = loss_slot(gain, lp);
            u32 mini = u % RHO;
            u32 pos = Kokkos::atomic_fetch_inc(&lp.bucket_cursor(idx_dsm(new_id, s, mini, lp)));
            lp.flat_buckets(pos) = u;
        });
        Kokkos::fence();

        _t_fill_buckets.stop();
        ScopedTimer _t_pick_prefix("rebalance", "jetrw", "pick_prefix");

        Kokkos::parallel_for("pick_prefix", lp.k,KOKKOS_LAMBDA(const partition_t id) {
            if (lp.partition.bweights(id) <= lp.sigma) { return; }

            weight_t min_to_lose = lp.partition.bweights(id) - lp.sigma;
            weight_t moved = 0;

            // slots in increasing loss order
            for (u32 s_idx = 0; s_idx < MAX_SLOTS && moved < min_to_lose; ++s_idx) {
                for (u32 r_idx = 0; r_idx < RHO && moved < min_to_lose; ++r_idx) {
                    u32 b_idx = idx_psm(id, s_idx, r_idx, lp);
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
        });
        Kokkos::fence();

        _t_pick_prefix.stop();
        ScopedTimer _t_apply_moves("rebalance", "jetrw", "apply_moves");

        apply_moves(lp, g, bc);
    }

    inline void jetrs(JetLabelPropagation &lp,
                      Graph &g,
                      BlockConnectivity &bc) {
        ScopedTimer _t_allocate("rebalance", "jetrs", "allocate");

        Kokkos::deep_copy(lp.bucket_counts, 0);
        Kokkos::deep_copy(lp.to_move, 0);
        Kokkos::deep_copy(lp.conn, 0);
        Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            lp.weight_id(u) = pack_s32_partition(-max_sentinel<weight_t>(), lp.k);
        });
        Kokkos::fence();

        _t_allocate.stop();
        ScopedTimer _t_best_block("rebalance", "jetrs", "best_block");

        Kokkos::parallel_for("best_block", bc.size, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = bc.us(i);
            partition_t id = bc.ids(i);
            weight_t w = bc.weights(i);

            partition_t u_id = lp.partition.map(u);
            weight_t u_w = g.weights(u);

            if (id == lp.k) { return; } // no valid entry
            if (lp.partition.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) g.weights(u) > lp.heavy_alpha * (lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex is too heavy
            if (lp.partition.bweights(id) + u_w > lp.lmax) { return; } // dont move into overloaded partition

            if (u_id == id) {
                lp.conn(u) = w;
                return;
            }

            Kokkos::atomic_max(&lp.weight_id(u), pack_s32_partition(w, id));
        });
        Kokkos::fence();

        _t_best_block.stop();
        ScopedTimer _t_count_buckets("rebalance", "jetrs", "count_buckets");

        // 2) For each vertex u in overloaded sources: pick best destination d with cap>0
        Kokkos::parallel_for("count_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = lp.partition.map(u);
            partition_t new_id = unpack_partition(lp.weight_id(u));
            weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

            if (new_id == lp.k) { return; } // no block found
            if (lp.partition.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

            u32 s = loss_slot(gain, lp);
            u32 mini = u % RHO;
            Kokkos::atomic_inc(&lp.bucket_counts(idx_dsm(new_id, s, mini, lp)));
        });
        Kokkos::fence();

        _t_count_buckets.stop();
        ScopedTimer _t_bucket_offsets("rebalance", "jetrs", "bucket_offsets");

        // prefix sum
        Kokkos::parallel_scan("bucket_offsets", lp.k * MAX_SLOTS * RHO, KOKKOS_LAMBDA(const u32 i, u32 &upd, const bool final_pass) {
            u32 val = lp.bucket_counts(i);
            if (final_pass) lp.bucket_offsets(i) = upd;
            upd += val;
        });
        Kokkos::fence();

        _t_bucket_offsets.stop();
        ScopedTimer _t_fill_buckets("rebalance", "jetrs", "fill_buckets");

        // fill
        Kokkos::deep_copy(lp.bucket_cursor, lp.bucket_offsets);
        Kokkos::fence();

        Kokkos::parallel_for("fill_buckets", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = lp.partition.map(u);
            partition_t new_id = unpack_partition(lp.weight_id(u));
            weight_t gain = unpack_score(lp.weight_id(u)) - lp.conn(u);

            if (new_id == lp.k) { return; } // not a candidate
            if (lp.partition.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) g.weights(u) > lp.heavy_alpha * ((f64) lp.partition.bweights(u_id) - ((f64) g.g_weight / (f64) lp.k))) { return; } // vertex too heavy

            u32 s = loss_slot(gain, lp);
            u32 mini = u % RHO;
            u32 pos = Kokkos::atomic_fetch_inc(&lp.bucket_cursor(idx_dsm(new_id, s, mini, lp)));
            lp.flat_buckets(pos) = u;
        });
        Kokkos::fence();

        _t_fill_buckets.stop();
        ScopedTimer _t_pick_prefix("rebalance", "jetrs", "pick_prefix");

        // 4) Per-destination prefix selection: honor cap[d]
        Kokkos::parallel_for("pick_prefix", lp.k, KOKKOS_LAMBDA(const partition_t d) {
            weight_t w = lp.partition.bweights(d);
            weight_t cap_d = (w < lp.lmax) ? (lp.lmax - w) : 0;
            if (cap_d <= 0) return;

            weight_t acquired = 0;
            for (u32 s = 0; s < MAX_SLOTS && acquired < cap_d; ++s) {
                for (u32 mini = 0; mini < RHO && acquired < cap_d; ++mini) {
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
        });
        Kokkos::fence();

        _t_pick_prefix.stop();
        ScopedTimer _t_apply_moves("rebalance", "jetrs", "apply_moves");

        apply_moves(lp, g, bc);
    }

    inline void refine(Graph &g,
                       Partition &partition,
                       partition_t k,
                       weight_t lmax,
                       u32 max_level,
                       u32 level) {
        JetLabelPropagation lp = initialize_lp(g.n, g.m, k, lmax);

        if (level == 0) { lp.conn_c = 0.25; }

        lp.sigma_percent = lp.sigma_percent_min + (level == 0 ? 0.0 : lp.sigma_percent * ((f64) level / (f64) max_level));
        lp.sigma = lp.lmax - (weight_t) ((f64) lp.lmax * lp.sigma_percent);

        copy_into(lp.partition, partition, g.n);

        ScopedTimer _t("refine", "BlockConnectivity", "build_scratch");
        BlockConnectivity bc = rebuild_scratch(g, lp.partition);
        _t.stop();

        ScopedTimer _t_edge_cut("refine", "JetLabelPropagation", "get_edge_cut");
        weight_t best_edge_cut = edge_cut(bc, lp.partition);
        weight_t best_weight = max_weight(lp.partition);
        _t_edge_cut.stop();

        weight_t curr_edge_cut = best_edge_cut;
        weight_t curr_weight = best_weight;

        bool executed_rw = false;
        weight_t last_rw_edge_cut = max_sentinel<weight_t>();
        weight_t last_rw_weight = max_sentinel<weight_t>();

        ScopedTimer _t_reset_lock("refine", "JetLabelPropagation", "reset_lock");
        Kokkos::deep_copy(lp.locked, 0);
        Kokkos::fence();
        _t_reset_lock.stop();

        u32 weak_iterations = 0;
        u32 iteration = 0;
        while (iteration < lp.n_max_iterations) {
            executed_rw = false;
            if (curr_weight <= lp.lmax) {
                jetlp(lp, g, bc);
                weak_iterations = 0;
            } else {
                ScopedTimer _t_reset_lock2("refine", "JetLabelPropagation", "reset_lock");
                Kokkos::deep_copy(lp.locked, 0);
                Kokkos::fence();
                _t_reset_lock2.stop();
                if (weak_iterations < lp.max_weak_iterations) {
                    jetrw(lp, g, bc);
                    weak_iterations++;
                    executed_rw = true;
                } else {
                    jetrs(lp, g, bc);
                    weak_iterations = 0;
                }
            }

            ScopedTimer _t_edge_cut2("refine", "JetLabelPropagation", "get_edge_cut");
            curr_edge_cut = edge_cut(bc, lp.partition);
            curr_weight = max_weight(lp.partition);
            _t_edge_cut2.stop();

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
                    copy_into(partition, lp.partition, g.n);
                    best_edge_cut = curr_edge_cut;
                    best_weight = curr_weight;
                }
            } else if (curr_weight < best_weight) {
                copy_into(partition, lp.partition, g.n);
                best_edge_cut = curr_edge_cut;
                best_weight = curr_weight;
                iteration = 0;
            }

            iteration += 1;
        }
    }
}

#endif //GPU_HEIPA_JET_LABEL_PROPAGATION_H
