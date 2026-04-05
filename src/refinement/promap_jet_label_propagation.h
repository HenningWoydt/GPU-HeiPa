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

#ifndef GPU_HEIPA_NEW_PROMAP_JET_LABEL_PROPAGATION_H
#define GPU_HEIPA_NEW_PROMAP_JET_LABEL_PROPAGATION_H

#include <Kokkos_Core.hpp>

#include "../utility/definitions.h"
#include "jet_label_propagation.h"
#include "block_conn.h"

namespace GPU_HeiPa {
    struct ProMapLabelPropagation {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;
        vertex_t min_size = 0;

        Partition partition{};

        UnmanagedDeviceWeight gain1, temp_gain, gain_cache, evict_start, evict_adjust;
        UnmanagedDeviceVertex vtx1, vtx2, vtx3;
        UnmanagedDevicePartition dest_part, underloaded_blocks;
        UnmanagedDeviceU32 zeros;

        DeviceScalarU32 idx;

        UnmanagedDeviceU32 lock;
        UnmanagedDevicePartition dest_cache;
        UnmanagedDevicePartition old_map;

        HostScalarPinnedU32 host_pinned_u32;
        HostScalarPinnedVertex scan_host;
        HostScalarPinnedWeight cut_change1, cut_change2, max_part;
        HostPinnedWeight reduce_locs;
        DeviceScalarPartition n_underloaded_blocks;
        DeviceScalarWeight max_vwgt;

        UnmanagedDeviceU64 histogram;
    };

    inline ProMapLabelPropagation initialize_ProMap_label_propagation(const vertex_t t_n,
                                                                      const vertex_t t_m,
                                                                      const partition_t t_k,
                                                                      const weight_t t_lmax,
                                                                      KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "allocate");

        ProMapLabelPropagation lp;

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
        lp.vtx3 = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * lp.n), lp.n);

        lp.dest_part = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * lp.n), lp.n);
        lp.underloaded_blocks = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * lp.k), lp.k);

        lp.zeros = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * lp.n), lp.n);
        Kokkos::deep_copy(lp.zeros, 0);

        lp.idx = DeviceScalarU32("idx");

        lp.lock = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * lp.n), lp.n);
        lp.dest_cache = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * lp.n), lp.n);
        lp.old_map = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * lp.n), lp.n);
        Kokkos::deep_copy(lp.lock, 0);
        Kokkos::deep_copy(lp.dest_cache, NULL_PART);

        lp.host_pinned_u32 = HostScalarPinnedU32("pinned_u32");
        lp.scan_host = HostScalarPinnedVertex("scan host");
        lp.n_underloaded_blocks = DeviceScalarPartition("total undersized");
        lp.max_vwgt = DeviceScalarWeight("max vwgt allowed");
        lp.reduce_locs = HostPinnedWeight("reduce to here", 3);
        lp.cut_change1 = Kokkos::subview(lp.reduce_locs, 0);
        lp.cut_change2 = Kokkos::subview(lp.reduce_locs, 1);
        lp.max_part = Kokkos::subview(lp.reduce_locs, 2);

        lp.histogram = UnmanagedDeviceU64((u64 *) get_chunk_back(mem_stack, sizeof(u64) * MAX_BUCKETS), MAX_BUCKETS);
        Kokkos::deep_copy(lp.histogram, 0);

        return lp;
    }

    inline void free_ProMap_LabelPropagation(const ProMapLabelPropagation &lp,
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
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);

        free_partition(lp.partition, mem_stack);
    }

    KOKKOS_INLINE_FUNCTION
    vertex_t ProMap_gain_bucket(const weight_t &gx, const weight_t &vwgt) {
        // keep the cheap cases cheap (no div/log)
        if (gx > (weight_t) 0) return (vertex_t) 0;
        if (gx == (weight_t) 0) return (vertex_t) 1;

        // gx < 0: bucket by magnitude of |gx/vwgt| around 1.0
        // Use float for speed on GPU (usually plenty for bucketing)
        const float g = (float) (-gx) / (float) vwgt; // positive magnitude

        // MID_BUCKETS is the "around 1.0" bucket
        int b = (int) MID_BUCKETS;

        // If g == 1 -> k=0 -> b stays MID_BUCKETS (matches loop behavior)
        // k = ceil(log_{1.5}(g)) for g>1, and k = ceil(log_{1.5}(1/g)) for g<1
        constexpr float INV_LOG2_1P5 = 1.0f / 0.5849625007211562f; // 1 / log2(1.5)
        constexpr float EPS = 1e-6f; // avoid rounding up at exact powers

        if (g < 1.0f) {
            // k = ceil(-log_{1.5}(g))
            const float x = (-Kokkos::log2(g)) * INV_LOG2_1P5;
            const int k = (int) Kokkos::ceil(x - EPS);
            b -= k;
            if (b < 2) b = 2;
        } else {
            // k = ceil(log_{1.5}(g))
            const float x = (Kokkos::log2(g)) * INV_LOG2_1P5;
            const int k = (int) Kokkos::ceil(x - EPS);
            b += k;
            if (b > (int) MAX_BUCKETS - 1) b = (int) MAX_BUCKETS - 1;
        }

        return (vertex_t) b;
    }

    template<typename d_oracle_t>
    KOKKOS_INLINE_FUNCTION
    weight_t gain_delta(const BlockConn &bc,
                        d_oracle_t &d_oracle,
                        vertex_t u,
                        partition_t old_id,
                        partition_t new_id) {
        u32 r_beg = bc.row(u);
        u32 r_end = bc.row(u + 1);

        weight_t delta = 0;
        for (u32 j = r_beg; j < r_end; j++) {
            partition_t id = bc.ids(j);
            weight_t w = bc.weights(j);

            if (id == NULL_PART || id == HASH_RECLAIM) { continue; }

            weight_t old_d = get(d_oracle, old_id, id);
            weight_t new_d = get(d_oracle, new_id, id);

            delta += w * (old_d - new_d);
        }

        return delta;
    }

    template<bool uniform_v_weights, bool uniform_e_weights, typename d_oracle_t>
    inline UnmanagedDeviceVertex ProMap_jet_lp(ProMapLabelPropagation &lp,
                                               const Graph &g,
                                               const BlockConn &bc,
                                               d_oracle_t &d_oracle,
                                               f64 conn_c) {
        vertex_t num_pos = 0;
        //
        {
            ScopedTimer _t("refinement", "jetlp", "best_block");
            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("best_block", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t temp_u_dest_part;
                weight_t temp_u_gain_cache;

                if (lp.dest_cache(u) != NULL_PART) {
                    // cached for this vertex
                    lp.dest_part(u) = lp.dest_cache(u);

                    temp_u_dest_part = lp.dest_cache(u);
                    temp_u_gain_cache = lp.gain_cache(u);
                } else {
                    partition_t u_id = lp.partition.map(u);
                    partition_t best_id = NO_MOVE;
                    weight_t best_delta = -1;

                    u32 r_beg = bc.row(u);
                    u32 r_end = bc.row(u + 1);

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);

                        if (id == u_id || id == NULL_PART || id == HASH_RECLAIM) { continue; } // dont move into self or invalid

                        weight_t delta = gain_delta(bc, d_oracle, u, u_id, id);
                        if (delta > best_delta) {
                            best_delta = delta;
                            best_id = id;
                        }
                    }

                    weight_t gain = 0;
                    if (best_id != NO_MOVE) {
                        gain = best_delta;
                    }

                    lp.gain_cache(u) = gain;
                    lp.dest_cache(u) = best_id;
                    lp.dest_part(u) = best_id;

                    temp_u_dest_part = best_id;
                    temp_u_gain_cache = gain;
                }

                if (temp_u_dest_part != NO_MOVE && lp.lock(u) == 0) {
                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx1(idx) = u;
                    lp.gain1(u) = temp_u_gain_cache;
                } else {
                    lp.gain1(u) = GAIN_MIN;
                    lp.lock(u) = 0;
                }
            });

            Kokkos::fence();
            Kokkos::deep_copy(lp.host_pinned_u32, lp.idx);
            num_pos = (vertex_t) lp.host_pinned_u32();

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
                    weight_t w = (uniform_e_weights ? 1 : g.edges_w(j));

                    weight_t v_gain = lp.gain1(v);

                    if (v_gain > u_gain || (v_gain == u_gain && v < u)) {
                        partition_t v_new_id = lp.dest_part(v);
                        partition_t v_old_id = lp.partition.map(v);

                        // distances BEFORE v moves
                        weight_t d_unew_vold = get(d_oracle, new_u_id, v_old_id);
                        weight_t d_uold_vold = get(d_oracle, old_u_id, v_old_id);

                        // distances AFTER v moves
                        weight_t d_unew_vnew = get(d_oracle, new_u_id, v_new_id);
                        weight_t d_uold_vnew = get(d_oracle, old_u_id, v_new_id);

                        // gains are (cost_before - cost_after)
                        weight_t before_u = d_uold_vold - d_unew_vold; // gain if only u moves
                        weight_t after_u = d_uold_vnew - d_unew_vnew; // gain if v moved first

                        change += w * (after_u - before_u);
                    }
                }

                if (u_gain + change >= 0) {
                    lp.lock(u) = 1;
                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx2(idx) = u;
                }
            });
            Kokkos::fence();
            Kokkos::deep_copy(lp.host_pinned_u32, lp.idx);
            num_pos = (vertex_t) lp.host_pinned_u32();

            KOKKOS_PROFILE_FENCE();
        }
        return Kokkos::subview(lp.vtx2, std::make_pair((vertex_t) 0, num_pos));
    }

    template<bool uniform_v_weights, typename d_oracle_t>
    inline UnmanagedDeviceVertex ProMap_rebalance_strong(ProMapLabelPropagation &lp,
                                                         const Graph &g,
                                                         const BlockConn &bc,
                                                         d_oracle_t &d_oracle) {
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

        u32 num_moves = 0;
        //
        {
            ScopedTimer _t("refinement", "jetrs", "score_candidates");

            Kokkos::deep_copy(Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);
            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("score_candidates", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                weight_t u_id_w = lp.partition.bweights(u_id);

                lp.vtx2(u) = NO_BLOCK_ID;

                if (u_id_w > lp.lmax && (uniform_v_weights ? 1 : g.weights(u)) <= 2 * lp.max_vwgt() && (uniform_v_weights ? 1 : g.weights(u)) < 2 * (u_id_w - opt_weight)) {
                    weight_t own_conn = 0;
                    weight_t count = 0;
                    weight_t sum_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;
                    for (vertex_t i = r_beg; i < r_end; i++) {
                        partition_t id = bc.ids(i);
                        weight_t conn = bc.weights(i);
                        if (id == u_id) {
                            own_conn = conn;
                            continue;
                        }
                        if (id != NULL_PART && id != HASH_RECLAIM && lp.partition.bweights(id) < max_b_w) {
                            sum_conn += conn;
                            count += 1;
                        }
                    }

                    if (count == 0) count = 1;
                    weight_t gain = (sum_conn / count) - own_conn;
                    vertex_t gain_type = ProMap_gain_bucket(gain, Kokkos::min((uniform_v_weights ? 1 : g.weights(u)), u_id_w - lp.lmax));

                    //add to count of appropriate bucket
                    if (gain_type < MAX_BUCKETS) {
                        vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections) + 1;
                        lp.vtx2(u) = g_id;
                        lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.gain1(g_id), (uniform_v_weights ? 1 : g.weights(u)));

                        u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                        lp.vtx3(idx) = u;
                    }
                }
            });

            Kokkos::deep_copy(num_moves, lp.idx);

            KOKKOS_PROFILE_FENCE();
        }

        //
        {
            ScopedTimer _t("refinement", "jetrs", "prefix_sum_score_buckets");

            if (t_minibuckets < 10000) {
                Kokkos::parallel_for("prefix_sum_score_buckets", Kokkos::TeamPolicy<>(1, 1024), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
                    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t &u, weight_t &update, const bool final) {
                        weight_t gain = lp.gain1(u);
                        if (final) {
                            lp.gain1(u) = update;
                        }
                        update += gain;
                    });
                });
            } else {
                Kokkos::parallel_scan("prefix_sum_score_buckets", Policy(0, t_minibuckets + 1), KOKKOS_LAMBDA(const vertex_t &u, weight_t &update, const bool final) {
                    weight_t gain = lp.gain1(u);
                    if (final) {
                        lp.gain1(u) = update;
                    }
                    update += gain;
                });
            }
        }

        //
        {
            ScopedTimer _t("refinement", "jetrs", "filter_scores");

            Kokkos::deep_copy(lp.evict_adjust, 0);
            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("filter_scores", num_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.vtx3(i);
                vertex_t b_id = lp.vtx2(u);
                partition_t u_id = lp.partition.map(u);
                vertex_t begin_bucket = u_id * width;
                weight_t score = lp.temp_gain(u) + lp.gain1(b_id) - lp.gain1(begin_bucket);
                weight_t limit = lp.partition.bweights(u_id) - lp.lmax;

                if (score < limit) {
                    if (score + (uniform_v_weights ? 1 : g.weights(u)) >= limit) {
                        lp.evict_adjust(u_id) = score + (uniform_v_weights ? 1 : g.weights(u));
                    }

                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx1(idx) = u;
                }
            });
            Kokkos::fence();

            Kokkos::deep_copy(num_moves, lp.idx);

            KOKKOS_PROFILE_FENCE();
        }

        // the rest of this method determines the destination part for each evicted vtx
        //assign consecutive chunks of vertices to undersized parts using scan result
        //
        {
            ScopedTimer _t("refinement", "jetrs", "cookie_cutter");

            Kokkos::parallel_for("cookie cutter", Kokkos::TeamPolicy<>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
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
                    if (id < (s32) lp.k && (uniform_v_weights ? 1 : g.weights(u)) / 2 <= lp.evict_start(id + 1) - lp.temp_gain(u)) {
                        // at least half of vtx weight lies in chunk p
                        lp.dest_part(u) = (partition_t) id;
                        return;
                    }
                    if (id < (s32) lp.k) {
                        lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.max_vwgt(), (uniform_v_weights ? 1 : g.weights(u)));
                    }
                }
                lp.dest_part(u) = lp.partition.map(u);
            });

            KOKKOS_PROFILE_FENCE();
        }

        return Kokkos::subview(lp.vtx1, std::make_pair((u32) 0, num_moves));
    }

    template<bool uniform_v_weights, typename d_oracle_t>
    inline UnmanagedDeviceVertex ProMap_rebalance_weak(ProMapLabelPropagation &lp,
                                                       Graph &g,
                                                       const BlockConn &bc,
                                                       d_oracle_t &d_oracle) {
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

            Kokkos::parallel_for("underloaded_blocks", Kokkos::TeamPolicy<DeviceExecutionSpace>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
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

        //
        {
            ScopedTimer _t("refinement", "jetrw", "reset_minibuckets");

            Kokkos::deep_copy(Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);

            KOKKOS_PROFILE_FENCE();
        }

        // determine best block
        {
            ScopedTimer _t("refinement", "jetrw", "best_block");

            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("best_block", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                partition_t best_id = u_id;

                weight_t gain = 0;
                if (lp.partition.bweights(u_id) > lp.lmax && (uniform_v_weights ? 1 : g.weights(u)) < 1.5 * (lp.partition.bweights(u_id) - opt_weight)) {
                    weight_t best_id_w = 0;
                    weight_t own_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;
                    for (u32 j = r_beg; j < r_end; j++) {
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

                    gain = best_id_w - own_conn;

                    if (best_id_w <= 0) {
                        best_id = lp.underloaded_blocks(u % lp.n_underloaded_blocks());
                        gain = -own_conn;
                    }
                }
                lp.dest_part(u) = best_id;

                if (u_id != best_id) {
                    vertex_t gain_type = ProMap_gain_bucket(gain, (uniform_v_weights ? 1 : g.weights(u)));
                    vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections);
                    lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.gain1(g_id), (uniform_v_weights ? 1 : g.weights(u)));
                    lp.vtx2(u) = g_id;

                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx3(idx) = u;
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
                    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t u, weight_t &update, const bool final) {
                        weight_t x = lp.gain1(u);
                        if (final) {
                            lp.gain1(u) = update;
                        }
                        update += x;
                    });
                });
            } else {
                Kokkos::parallel_scan("scan score buckets", t_minibuckets + 1, KOKKOS_LAMBDA(const vertex_t &u, weight_t &update, const bool final) {
                    weight_t x = lp.gain1(u);
                    if (final) {
                        lp.gain1(u) = update;
                    }
                    update += x;
                });
            }

            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("refinement", "jetrw", "filter_scores");

            u32 num_moves;
            Kokkos::deep_copy(num_moves, lp.idx);

            Kokkos::parallel_scan("filter_scores", num_moves, KOKKOS_LAMBDA(const u32 i, vertex_t &update, const bool final) {
                vertex_t u = lp.vtx3(i);
                vertex_t b_id = lp.vtx2(u);
                partition_t u_id = lp.partition.map(u);
                vertex_t begin_bucket = u_id * width;
                weight_t score = lp.temp_gain(u) + lp.gain1(b_id) - lp.gain1(begin_bucket);
                weight_t limit = lp.partition.bweights(u_id) - lp.lmax;

                if (score < limit) {
                    if (final) {
                        lp.vtx1(update) = u;
                    }
                    update++;
                }
            }, lp.scan_host);

            KOKKOS_PROFILE_FENCE();
        }

        Kokkos::fence();
        u32 num_moves = lp.scan_host();

        return Kokkos::subview(lp.vtx1, std::make_pair((vertex_t) 0, num_moves));
    }

    template<bool uniform_v_weights, typename d_oracle_t>
    inline UnmanagedDeviceVertex ProMap_rebalance_strong_new(ProMapLabelPropagation &lp,
                                                             const Graph &g,
                                                             const BlockConn &bc,
                                                             d_oracle_t &d_oracle) {
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

        u32 num_moves = 0;
        //
        {
            ScopedTimer _t("refinement", "jetrs", "score_candidates");

            Kokkos::deep_copy(Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);
            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("score_candidates", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                weight_t u_id_w = lp.partition.bweights(u_id);

                lp.vtx2(u) = NO_BLOCK_ID;

                if (u_id_w > lp.lmax && (uniform_v_weights ? 1 : g.weights(u)) <= 2 * lp.max_vwgt() && (uniform_v_weights ? 1 : g.weights(u)) < 2 * (u_id_w - opt_weight)) {
                    weight_t own_conn = 0;
                    weight_t count = 0;
                    weight_t sum_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_end = bc.row(u + 1);
                    for (u32 i = r_beg; i < r_end; i++) {
                        partition_t id = bc.ids(i);

                        weight_t conn = bc.weights(i);
                        if (id == u_id) {
                            own_conn = conn;
                            continue;
                        }
                        if (id != NULL_PART && id != HASH_RECLAIM && lp.partition.bweights(id) < max_b_w) {
                            sum_conn += conn * get(d_oracle, u_id, id);
                            count += 1;
                        }
                    }

                    if (count == 0) count = 1;
                    weight_t gain = (sum_conn / count) - own_conn;
                    vertex_t gain_type = ProMap_gain_bucket(gain, Kokkos::min((uniform_v_weights ? 1 : g.weights(u)), u_id_w - lp.lmax));

                    //add to count of appropriate bucket
                    if (gain_type < MAX_BUCKETS) {
                        vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections) + 1;
                        lp.vtx2(u) = g_id;
                        lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.gain1(g_id), (uniform_v_weights ? 1 : g.weights(u)));

                        u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                        lp.vtx3(idx) = u;
                    }
                }
            });

            Kokkos::deep_copy(num_moves, lp.idx);

            KOKKOS_PROFILE_FENCE();
        }

        //
        {
            ScopedTimer _t("refinement", "jetrs", "prefix_sum_score_buckets");

            if (t_minibuckets < 10000) {
                Kokkos::parallel_for("prefix_sum_score_buckets", Kokkos::TeamPolicy<>(1, 1024), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
                    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t &u, weight_t &update, const bool final) {
                        weight_t gain = lp.gain1(u);
                        if (final) {
                            lp.gain1(u) = update;
                        }
                        update += gain;
                    });
                });
            } else {
                Kokkos::parallel_scan("prefix_sum_score_buckets", Policy(0, t_minibuckets + 1), KOKKOS_LAMBDA(const vertex_t &u, weight_t &update, const bool final) {
                    weight_t gain = lp.gain1(u);
                    if (final) {
                        lp.gain1(u) = update;
                    }
                    update += gain;
                });
            }
        }

        //
        {
            ScopedTimer _t("refinement", "jetrs", "filter_scores");

            Kokkos::deep_copy(lp.evict_adjust, 0);
            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("filter_scores", num_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = lp.vtx3(i);
                vertex_t b_id = lp.vtx2(u);
                partition_t u_id = lp.partition.map(u);
                vertex_t begin_bucket = u_id * width;
                weight_t score = lp.temp_gain(u) + lp.gain1(b_id) - lp.gain1(begin_bucket);
                weight_t limit = lp.partition.bweights(u_id) - lp.lmax;

                if (score < limit) {
                    if (score + (uniform_v_weights ? 1 : g.weights(u)) >= limit) {
                        lp.evict_adjust(u_id) = score + (uniform_v_weights ? 1 : g.weights(u));
                    }

                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx1(idx) = u;
                }
            });
            Kokkos::fence();

            Kokkos::deep_copy(num_moves, lp.idx);

            KOKKOS_PROFILE_FENCE();
        }

        // the rest of this method determines the destination part for each evicted vtx
        //assign consecutive chunks of vertices to undersized parts using scan result
        //
        {
            ScopedTimer _t("refinement", "jetrs", "cookie_cutter");

            Kokkos::parallel_for("cookie cutter", Kokkos::TeamPolicy<>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
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
                    if (id < (s32) lp.k && (uniform_v_weights ? 1 : g.weights(u)) / 2 <= lp.evict_start(id + 1) - lp.temp_gain(u)) {
                        // at least half of vtx weight lies in chunk p
                        lp.dest_part(u) = (partition_t) id;
                        return;
                    }
                    if (id < (s32) lp.k) {
                        lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.max_vwgt(), (uniform_v_weights ? 1 : g.weights(u)));
                    }
                }
                lp.dest_part(u) = lp.partition.map(u);
            });

            KOKKOS_PROFILE_FENCE();
        }

        return Kokkos::subview(lp.vtx1, std::make_pair((u32) 0, num_moves));
    }

    template<bool uniform_v_weights, typename d_oracle_t>
    inline UnmanagedDeviceVertex ProMap_rebalance_weak_new(ProMapLabelPropagation &lp,
                                                           Graph &g,
                                                           const BlockConn &bc,
                                                           d_oracle_t &d_oracle) {
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

            Kokkos::parallel_for("underloaded_blocks", Kokkos::TeamPolicy<DeviceExecutionSpace>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
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

        //
        {
            ScopedTimer _t("refinement", "jetrw", "reset_minibuckets");

            Kokkos::deep_copy(Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);

            KOKKOS_PROFILE_FENCE();
        }

        // determine best block
        {
            ScopedTimer _t("refinement", "jetrw", "best_block");

            Kokkos::deep_copy(lp.idx, 0);

            Kokkos::parallel_for("best_block", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = lp.partition.map(u);
                partition_t best_id = u_id;

                weight_t gain = 0;
                if (lp.partition.bweights(u_id) > lp.lmax && (uniform_v_weights ? 1 : g.weights(u)) < 1.5 * (lp.partition.bweights(u_id) - opt_weight)) {
                    weight_t best_delta = -max_sentinel<weight_t>();;

                    u32 r_beg = bc.row(u);
                    u32 r_end = bc.row(u + 1);
                    for (u32 i = r_beg; i < r_end; i++) {
                        partition_t id = bc.ids(i);

                        // dont move to invalid or overloaded block
                        if (id == NULL_PART || id == HASH_RECLAIM || lp.partition.bweights(id) >= max_b_w) { continue; }

                        weight_t delta = gain_delta(bc, d_oracle, u, u_id, id);
                        if (delta > best_delta) {
                            best_delta = delta;
                            best_id = id;
                        }
                    }

                    if (best_delta <= 0) {
                        best_id = lp.underloaded_blocks(u % lp.n_underloaded_blocks());
                        best_delta = gain_delta(bc, d_oracle, u, u_id, best_id);
                    }
                    gain = best_delta;
                }
                lp.dest_part(u) = best_id;

                if (u_id != best_id) {
                    vertex_t gain_type = ProMap_gain_bucket(gain, (uniform_v_weights ? 1 : g.weights(u)));
                    Kokkos::atomic_add(&lp.histogram(gain_type), 1);

                    vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections);
                    lp.temp_gain(u) = Kokkos::atomic_fetch_add(&lp.gain1(g_id), (uniform_v_weights ? 1 : g.weights(u)));
                    lp.vtx2(u) = g_id;

                    u32 idx = Kokkos::atomic_fetch_inc(&lp.idx());
                    lp.vtx3(idx) = u;
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
                    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t u, weight_t &update, const bool final) {
                        weight_t x = lp.gain1(u);
                        if (final) {
                            lp.gain1(u) = update;
                        }
                        update += x;
                    });
                });
            } else {
                Kokkos::parallel_scan("scan score buckets", t_minibuckets + 1, KOKKOS_LAMBDA(const vertex_t &u, weight_t &update, const bool final) {
                    weight_t x = lp.gain1(u);
                    if (final) {
                        lp.gain1(u) = update;
                    }
                    update += x;
                });
            }

            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("refinement", "jetrw", "filter_scores");

            u32 num_moves;
            Kokkos::deep_copy(num_moves, lp.idx);

            Kokkos::parallel_scan("filter_scores", num_moves, KOKKOS_LAMBDA(const u32 i, vertex_t &update, const bool final) {
                vertex_t u = lp.vtx3(i);
                vertex_t b_id = lp.vtx2(u);
                partition_t u_id = lp.partition.map(u);
                vertex_t begin_bucket = u_id * width;
                weight_t score = lp.temp_gain(u) + lp.gain1(b_id) - lp.gain1(begin_bucket);
                weight_t limit = lp.partition.bweights(u_id) - lp.lmax;

                if (score < limit) {
                    if (final) {
                        lp.vtx1(update) = u;
                    }
                    update++;
                }
            }, lp.scan_host);

            KOKKOS_PROFILE_FENCE();
        }

        Kokkos::fence();
        u32 num_moves = lp.scan_host();

        return Kokkos::subview(lp.vtx1, std::make_pair((vertex_t) 0, num_moves));
    }

    template<bool uniform_v_weights, bool uniform_e_weights, typename d_oracle_t>
    inline void ProMap_perform_moves(ProMapLabelPropagation &lp,
                                     const Graph &g,
                                     BlockConn &bc,
                                     d_oracle_t &d_oracle,
                                     const UnmanagedDeviceVertex &moves,
                                     weight_t &curr_max_weight,
                                     weight_t &curr_comm_cost) {
        u32 n_moves = (u32) moves.extent(0);

        // copy old partition
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "copy_old");

            Kokkos::deep_copy(lp.old_map, lp.partition.map);

            KOKKOS_PROFILE_FENCE();
        }

        // first change in cut
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "cut_change_1");

            Kokkos::parallel_for("cut_change_1", n_moves, KOKKOS_LAMBDA(const u32 &i) {
                vertex_t u = moves(i);
                weight_t u_w = (uniform_v_weights ? 1 : g.weights(u));
                partition_t old_id = lp.partition.map(u);
                partition_t new_id = lp.dest_part(u);

                lp.dest_cache(u) = NULL_PART;
                Kokkos::atomic_add(&lp.partition.bweights(old_id), -u_w);
                Kokkos::atomic_add(&lp.partition.bweights(new_id), u_w);

                lp.partition.map(u) = new_id;
                lp.dest_part(u) = old_id;
            });

            KOKKOS_PROFILE_FENCE();
        }

        // update max weight
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "update_max_weight");

            Kokkos::parallel_reduce("update_max_weight", Kokkos::RangePolicy<>(DeviceExecutionSpace(), 0, lp.k), KOKKOS_LAMBDA(const s32 i, weight_t &update) {
                if (lp.partition.bweights(i) > update) {
                    update = lp.partition.bweights(i);
                }
            }, Kokkos::Max(lp.max_part));

            KOKKOS_PROFILE_FENCE();
        }

        // update block conn
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "update_block_conn");

            if (n_moves > (u32) g.n / 10) {
                update_large<uniform_e_weights>(g, lp.partition, lp.zeros, lp.dest_cache, bc, moves);
            } else {
                update_small<uniform_e_weights>(g, lp.partition, lp.dest_part, lp.dest_cache, bc, moves);
            }

            KOKKOS_PROFILE_FENCE();
        }

        // change in comm cost
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "comm_cost_delta");

            Kokkos::parallel_reduce("comm_cost", n_moves, KOKKOS_LAMBDA(const u32 i, weight_t &local_delta) {
                vertex_t u = moves(i);
                partition_t old_u_id = lp.old_map(u);
                partition_t new_u_id = lp.partition.map(u);

                // NEW: skip fake moves
                if (old_u_id == new_u_id) {
                    return;
                }

                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                    vertex_t v = g.edges_v(j);
                    weight_t w = (uniform_e_weights ? 1 : g.edges_w(j));

                    partition_t old_v_id = lp.old_map(v);
                    partition_t new_v_id = lp.partition.map(v);

                    weight_t old_d = get(d_oracle, old_u_id, old_v_id);
                    weight_t new_d = get(d_oracle, new_u_id, new_v_id);

                    weight_t contrib = w * (new_d - old_d);

                    // If the global cost counts each undirected edge twice (symmetric CSR):
                    // - edges with exactly one moved endpoint are only visited once here,
                    //   so we need to double their contribution.
                    bool u_moved = (old_u_id != new_u_id);
                    bool v_moved = (old_v_id != new_v_id);
                    if (u_moved && !v_moved) {
                        contrib *= 2;
                    }

                    local_delta += contrib;
                }
            }, lp.cut_change1);
        }
        Kokkos::fence();

        // update cut
        curr_max_weight = lp.max_part();
        curr_comm_cost += lp.cut_change1();
    }

    template<bool uniform_v_weights, bool uniform_e_weights, typename d_oracle_t>
    inline std::pair<weight_t, weight_t> ProMap_jet_refine(Graph &g,
                                                           Partition &partition,
                                                           d_oracle_t &d_oracle,
                                                           partition_t k,
                                                           weight_t lmax,
                                                           u32 level,
                                                           weight_t curr_comm_cost,
                                                           weight_t curr_max_weight,
                                                           KokkosMemoryStack &mem_stack) {
        ProMapLabelPropagation lp = initialize_ProMap_label_propagation(g.n, g.m, k, lmax, mem_stack);

        // copy partition
        {
            ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");
            copy_into(lp.partition, partition, g.n);
            KOKKOS_PROFILE_FENCE();
        }

        weight_t best_comm_cost = curr_comm_cost;
        weight_t best_max_weight = curr_max_weight;

        BlockConn bc;
        bc = init_BlockConn<uniform_e_weights>(g, lp.partition, mem_stack);

        std::vector<f64> filter_ratios;

        if (level == 0) {
            filter_ratios.push_back(0.25);
        } else {
            filter_ratios.push_back(0.75);
        }

        for (auto filter_ratio: filter_ratios) {
            u32 balance_iteration = 0;
            u32 iteration = 0;
            while (iteration < N_MAX_ITERATIONS) {
                iteration += 1;

                UnmanagedDeviceVertex moves;
                if (curr_max_weight <= lmax) {
                    moves = ProMap_jet_lp<uniform_v_weights, uniform_e_weights>(lp, g, bc, d_oracle, filter_ratio);
                    balance_iteration = 0;
                } else {
                    balance_iteration++;

                    if (balance_iteration < N_MAX_WEAK_ITERATIONS) {
                        moves = ProMap_rebalance_weak<uniform_v_weights>(lp, g, bc, d_oracle);
                    } else {
                        moves = ProMap_rebalance_strong<uniform_v_weights>(lp, g, bc, d_oracle);
                    }
                }

                u32 n_moves = (u32) moves.extent(0);
                if (n_moves == 0) { continue; }

                ProMap_perform_moves<uniform_v_weights, uniform_e_weights>(lp, g, bc, d_oracle, moves, curr_max_weight, curr_comm_cost);

                if (best_max_weight > lmax && curr_max_weight < best_max_weight) {
                    // copy the partition
                    {
                        ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");

                        copy_into(partition, lp.partition, g.n);
                        best_comm_cost = curr_comm_cost;
                        best_max_weight = curr_max_weight;
                        iteration = 0;

                        KOKKOS_PROFILE_FENCE();
                    }
                } else if (curr_comm_cost < best_comm_cost && (curr_max_weight <= lmax || curr_max_weight < best_max_weight)) {
                    if ((f64) curr_comm_cost < PHI * (f64) best_comm_cost) { iteration = 0; }
                    //
                    {
                        ScopedTimer _t("refinement", "JetLabelPropagation", "copy_partition");

                        copy_into(partition, lp.partition, g.n);
                        best_comm_cost = curr_comm_cost;
                        best_max_weight = curr_max_weight;

                        KOKKOS_PROFILE_FENCE();
                    }
                }
            }
        }

        free_BlockConn(bc, mem_stack);
        free_ProMap_LabelPropagation(lp, mem_stack);

        return std::make_pair(best_comm_cost, best_max_weight);
    }
}

#endif //GPU_HEIPA_NEW_PROMAP_JET_LABEL_PROPAGATION_H
