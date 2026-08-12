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

#include <Kokkos_Core.hpp>

#include "../definitions.h"
#include "block_conn.h"

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
        UnmanagedDeviceVertex vtx1, vtx2, vtx3;
        UnmanagedDevicePartition dest_part, underloaded_blocks;
        UnmanagedDeviceU32 moved_round;
        u32 round = 0;

        DeviceScalarU32 idx;

        UnmanagedDeviceU32 lock;
        UnmanagedDevicePartition dest_cache;

        HostScalarPinnedU32 host_pinned_u32;
        HostScalarPinnedVertex scan_host;
        HostScalarPinnedWeight cut_change1, cut_change2, host_max_part;
        HostPinnedWeight reduce_locs;
        DeviceScalarPartition n_underloaded_blocks;
        DeviceScalarWeight max_vwgt, dev_max_part;
    };

    inline LabelPropagation initialize_label_propagation(const vertex_t t_n,
                                                         const vertex_t t_m,
                                                         const partition_t t_k,
                                                         const weight_t t_lmax,
                                                         KokkosMemoryStack &mem_stack,
                                                         DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "allocate");

        LabelPropagation lp;

        lp.n = t_n;
        lp.m = t_m;
        lp.k = t_k;
        lp.lmax = t_lmax;
        lp.min_size = t_k * MAX_SECTIONS * MAX_BUCKETS;

        lp.partition = initialize_partition(t_n, t_k, t_lmax, mem_stack, exec_space);

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

        lp.moved_round = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * lp.n), lp.n);
        Kokkos::deep_copy(exec_space, lp.moved_round, 0);

        lp.idx = DeviceScalarU32("idx");

        lp.lock = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * lp.n), lp.n);
        lp.dest_cache = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * lp.n), lp.n);
        Kokkos::deep_copy(exec_space, lp.lock, 0);
        Kokkos::deep_copy(exec_space, lp.dest_cache, NULL_PART);

        lp.host_pinned_u32 = HostScalarPinnedU32("pinned_u32");
        lp.scan_host = HostScalarPinnedVertex("scan host");
        lp.n_underloaded_blocks = DeviceScalarPartition("total undersized");
        lp.max_vwgt = DeviceScalarWeight("max vwgt allowed");
        lp.dev_max_part = DeviceScalarWeight("max_part");
        lp.reduce_locs = HostPinnedWeight("reduce to here", 3);
        lp.cut_change1 = Kokkos::subview(lp.reduce_locs, 0);
        lp.cut_change2 = Kokkos::subview(lp.reduce_locs, 1);
        lp.host_max_part = Kokkos::subview(lp.reduce_locs, 2);

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
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);

        free_partition(lp.partition, mem_stack);
    }

    KOKKOS_INLINE_FUNCTION
    vertex_t gain_bucket(const weight_t &gx, const weight_t &vwgt) {
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

    template<bool uniform_v_weights, bool uniform_e_weights>
    inline UnmanagedDeviceVertex jet_lp(LabelPropagation &lp,
                                        const Graph &g,
                                        const BlockConn &bc,
                                        f64 conn_c,
                                        DeviceExecutionSpace &exec_space) {
        vertex_t num_pos = 0;

        HEIPA_PROFILE_SCOPE("refinement", "jetlp", "best_block");
        auto dest_cache = lp.dest_cache;
        auto dest_part = lp.dest_part;
        auto p_map = lp.partition.map;
        auto gain_cache = lp.gain_cache;
        auto bc_row = bc.row;
        auto bc_sizes = bc.sizes;
        auto bc_ids = bc.ids;
        auto bc_weights = bc.weights;

        Kokkos::parallel_for("best_block", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            if (dest_cache(u) != NULL_PART) {
                // cached for this vertex
                dest_part(u) = dest_cache(u);
            } else {
                partition_t u_id = p_map(u);
                partition_t best_id = NO_MOVE;
                weight_t best_conn = 0;
                weight_t own_conn = 0;

                u32 r_beg = bc_row(u);
                u32 r_len = bc_sizes(u);
                u32 r_end = r_beg + r_len;

                for (u32 i = r_beg; i < r_end; ++i) {
                    partition_t id = bc_ids(i);
                    weight_t w = bc_weights(i);

                    bool valid = (id != NULL_PART) & (id != HASH_RECLAIM); // single mask
                    bool is_own = valid & (id == u_id);
                    bool is_cand = valid & !is_own;

                    // Update own_conn if this is our id
                    own_conn = is_own ? w : own_conn;

                    // Update best if it's a candidate and better
                    bool better = is_cand & (w > best_conn);
                    best_conn = better ? w : best_conn;
                    best_id = better ? id : best_id;
                }

                weight_t gain = 0;

                if (best_id != NO_MOVE) {
                    if (best_conn >= own_conn || ((f64) own_conn - (f64) best_conn) < floor(conn_c * (f64) own_conn)) {
                        gain = best_conn - own_conn;
                    } else {
                        best_id = NO_MOVE;
                    }
                }

                gain_cache(u) = gain;
                dest_cache(u) = best_id;
                dest_part(u) = best_id;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetlp", "best_block_filter");
        auto vtx1 = lp.vtx1;
        auto gain1 = lp.gain1;
        auto lock = lp.lock;

        Kokkos::parallel_scan("filter potentially viable moves", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const u32 u, u32 &update, const bool final) {
            if (dest_part(u) != NO_MOVE && lock(u) == 0) {
                if (final) {
                    vtx1(update) = u;
                    gain1(u) = gain_cache(u);
                }
                update++;
            } else if (final) {
                gain1(u) = GAIN_MIN;
                lock(u) = 0;
            }
        }, lp.scan_host);
        exec_space.fence();
        num_pos = lp.scan_host();
        KOKKOS_PROFILE_FENCE(exec_space);

        // use afterburner
        auto g_neighborhood = g.neighborhood;
        auto g_edges_v = g.edges_v;
        auto g_edges_w = g.edges_w;

        if ((f64) g.m / (f64) g.n < 8) {
            HEIPA_PROFILE_SCOPE("refinement", "jetlp", "afterburner");
            Kokkos::parallel_for("afterburner heuristic", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_pos), KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = vtx1(i);
                weight_t u_gain = gain1(u);
                partition_t old_u_id = p_map(u);
                partition_t new_u_id = dest_part(u);

                weight_t change = 0;
                for (u32 j = g_neighborhood(u); j < g_neighborhood(u + 1); ++j) {
                    vertex_t v = g_edges_v(j);
                    weight_t v_gain = gain1(v);

                    bool move_first = v_gain > u_gain || (v_gain == u_gain && v < u);
                    partition_t v_new_id = dest_part(v);
                    partition_t v_old_id = p_map(v);
                    weight_t w = uniform_e_weights ? move_first : g_edges_w(j) * move_first;
                    change += w * ((v_new_id == new_u_id) - (v_new_id == old_u_id) + (v_old_id == old_u_id) - (v_old_id == new_u_id));
                }

                if (u_gain + change >= 0) {
                    lock(u) = 1;
                }
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        } else {
            using team_policy = Kokkos::TeamPolicy<DeviceExecutionSpace>;
            using member_type = team_policy::member_type;

            HEIPA_PROFILE_SCOPE("refinement", "jetlp", "afterburner_team");
            Kokkos::parallel_for("afterburner heuristic", team_policy(exec_space, num_pos, Kokkos::AUTO()), KOKKOS_LAMBDA(const member_type &team) {
                u32 i = team.league_rank();

                vertex_t u = vtx1(i);
                weight_t u_gain = gain1(u);
                partition_t old_u_id = p_map(u);
                partition_t new_u_id = dest_part(u);

                weight_t change = 0;

                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, g_neighborhood(u), g_neighborhood(u + 1)), [&](const u32 j, weight_t &local_change) {
                    vertex_t v = g_edges_v(j);
                    weight_t v_gain = gain1(v);

                    partition_t v_new_id = dest_part(v);
                    partition_t v_old_id = p_map(v);

                    bool move_first = (v_gain > u_gain) || ((v_gain == u_gain) && (v < u));

                    weight_t w = uniform_e_weights ? (weight_t) move_first : g_edges_w(j) * (weight_t) move_first;

                    local_change += w * ((v_new_id == new_u_id) - (v_new_id == old_u_id) + (v_old_id == old_u_id) - (v_old_id == new_u_id));
                }, change);

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    if (u_gain + change >= 0) {
                        lock(u) = 1;
                    }
                });
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        HEIPA_PROFILE_SCOPE("refinement", "jetlp", "afterburner_filter");
        auto vtx2 = lp.vtx2;
        Kokkos::parallel_scan("filter beneficial moves", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_pos), KOKKOS_LAMBDA(const u32 i, u32 &update, const bool final) {
            vertex_t v = vtx1(i);
            if (final && lock(v)) {
                vtx2(update) = v;
            }
            update += lock(v);
        }, lp.scan_host);
        exec_space.fence();

        num_pos = lp.scan_host();
        KOKKOS_PROFILE_FENCE(exec_space);

        return Kokkos::subview(vtx2, std::make_pair((vertex_t) 0, num_pos));
    }

    template<bool uniform_v_weights>
    inline UnmanagedDeviceVertex rebalance_strong(LabelPropagation &lp,
                                                  const Graph &g,
                                                  const BlockConn &bc,
                                                  DeviceExecutionSpace &exec_space) {
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
        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "find_max_vwgt");
        auto p_bweights = lp.partition.bweights;
        auto max_vwgt = lp.max_vwgt;
        Kokkos::parallel_reduce("find max size", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, lp.k), KOKKOS_LAMBDA(const partition_t id, weight_t &update) {
            weight_t size = p_bweights(id);
            if (size < max_b_w) {
                weight_t cap = max_b_w - size;
                if (cap > update) {
                    update = cap;
                }
            }
        }, Kokkos::Max(max_vwgt));
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "reset_mini_buckets");
        Kokkos::deep_copy(exec_space, Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "score_candidates");
        auto p_map = lp.partition.map;
        auto vtx2 = lp.vtx2;
        auto g_weights = g.weights;
        auto bc_row = bc.row;
        auto bc_sizes = bc.sizes;
        auto bc_ids = bc.ids;
        auto bc_weights = bc.weights;
        auto lp_lmax = lp.lmax;
        auto temp_gain = lp.temp_gain;
        auto gain1 = lp.gain1;
        
        Kokkos::parallel_for("score_candidates", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = p_map(u);
            weight_t u_id_w = p_bweights(u_id);

            vtx2(u) = NO_BLOCK_ID;

            if (u_id_w > lp_lmax && (uniform_v_weights ? 1 : g_weights(u)) <= 2 * max_vwgt() && (uniform_v_weights ? 1 : g_weights(u)) < 2 * (u_id_w - opt_weight)) {
                weight_t own_conn = 0;
                weight_t count = 0;
                weight_t sum_conn = 0;

                u32 r_beg = bc_row(u);
                u32 r_len = bc_sizes(u);
                u32 r_end = r_beg + r_len;
                for (vertex_t i = r_beg; i < r_end; i++) {
                    partition_t id = bc_ids(i);
                    weight_t conn = bc_weights(i);
                    if (id == u_id) {
                        own_conn = conn;
                        continue;
                    }
                    if (id != NULL_PART && id != HASH_RECLAIM && p_bweights(id) < max_b_w) {
                        sum_conn += conn;
                        count += 1;
                    }
                }

                if (count == 0) count = 1;
                weight_t gain = (sum_conn / count) - own_conn;
                vertex_t gain_type = gain_bucket(gain, Kokkos::min((uniform_v_weights ? 1 : g_weights(u)), u_id_w - lp_lmax));

                //add to count of appropriate bucket
                if (gain_type < MAX_BUCKETS) {
                    vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections) + 1;
                    vtx2(u) = g_id;
                    temp_gain(u) = Kokkos::atomic_fetch_add(&gain1(g_id), (uniform_v_weights ? 1 : g_weights(u)));
                }
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "prefix_sum_score_buckets");
        auto gain1_p = lp.gain1;
        if (t_minibuckets < 10000) {
            Kokkos::parallel_for("prefix_sum_score_buckets", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, 1024), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t &u, weight_t &update, const bool final) {
                    weight_t gain = gain1_p(u);
                    if (final) {
                        gain1_p(u) = update;
                    }
                    update += gain;
                });
            });
        } else {
            Kokkos::parallel_scan("prefix_sum_score_buckets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, t_minibuckets + 1), KOKKOS_LAMBDA(const vertex_t &u, weight_t &update, const bool final) {
                weight_t gain = gain1_p(u);
                if (final) {
                    gain1_p(u) = update;
                }
                update += gain;
            });
        }
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "reset_evict_adjust");
        Kokkos::deep_copy(exec_space, lp.evict_adjust, 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "filter_scores");
        auto vtx2_f = lp.vtx2;
        auto p_map_f = lp.partition.map;
        auto temp_gain_f = lp.temp_gain;
        auto gain1_f = lp.gain1;
        auto p_bweights_f = lp.partition.bweights;
        auto lp_lmax_f = lp.lmax;
        auto g_weights_f = g.weights;
        auto evict_adjust_f = lp.evict_adjust;
        auto vtx1_f = lp.vtx1;
        
        Kokkos::parallel_scan("filter_scores", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
            vertex_t b_id = vtx2_f(u);
            if (b_id != NO_BLOCK_ID) {
                partition_t u_id = p_map_f(u);
                vertex_t begin_bucket = u_id * width;
                weight_t score = temp_gain_f(u) + gain1_f(b_id) - gain1_f(begin_bucket);
                weight_t limit = p_bweights_f(u_id) - lp_lmax_f;

                if (score < limit) {
                    if (final) {
                        if (score + (uniform_v_weights ? 1 : g_weights_f(u)) >= limit) {
                            evict_adjust_f(u_id) = score + (uniform_v_weights ? 1 : g_weights_f(u));
                        }

                        vtx1_f(update) = u;
                    }
                    update++;
                }
            }
        }, lp.scan_host);
        exec_space.fence();

        u32 num_moves = lp.scan_host();
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "cookie_cutter");
        auto evict_adjust_c = lp.evict_adjust;
        auto gain1_c = lp.gain1;
        auto max_vwgt_c = lp.max_vwgt;
        auto evict_start_c = lp.evict_start;
        auto p_bweights_c = lp.partition.bweights;
        auto lp_k = lp.k;
        
        Kokkos::parallel_for("cookie cutter", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp_k), [&](const partition_t p, weight_t &update, const bool final) {
                weight_t add = evict_adjust_c(p);
                vertex_t begin_bucket = MAX_BUCKETS * p * sections;
                if (add == 0) {
                    // evict_adjust_c(p) isn't set if there aren't enough evictions to balance part p
                    add = gain1_c(begin_bucket + MAX_BUCKETS * sections) - gain1_c(begin_bucket);
                }
                if (final) {
                    evict_adjust_c(p) = gain1_c(begin_bucket) - update;
                }
                update += add;
                if (final && p + 1 == lp_k) {
                    max_vwgt_c() = update;
                }
            });
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp_k), [&](const partition_t p, weight_t &update, const bool final) {
                if (final && p == 0) {
                    evict_start_c(0) = 0;
                }
                if (max_b_w > p_bweights_c(p)) {
                    update += max_b_w - p_bweights_c(p);
                }
                if (final) {
                    evict_start_c(p + 1) = update;
                }
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrs", "adjust_scores");
        auto vtx1_a = lp.vtx1;
        auto p_map_a = lp.partition.map;
        auto vtx2_a = lp.vtx2;
        auto temp_gain_a = lp.temp_gain;
        auto gain1_a = lp.gain1;
        auto evict_adjust_a = lp.evict_adjust;
        auto evict_start_a = lp.evict_start;
        auto g_weights_a = g.weights;
        auto dest_part_a = lp.dest_part;
        auto max_vwgt_a = lp.max_vwgt;
        auto lp_k_a = lp.k;
        
        Kokkos::parallel_for("adjust_scores", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_moves), KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = vtx1_a(i);
            partition_t u_id = p_map_a(u);
            vertex_t b_id = vtx2_a(u);
            weight_t score = temp_gain_a(u) + gain1_a(b_id) - evict_adjust_a(u_id);

            temp_gain_a(u) = score;
            s32 id = 0;
            while (id < (s32) lp_k_a) {
                //find chunk that contains i
                while (id <= (s32) lp_k_a && evict_start_a(id) <= temp_gain_a(u)) {
                    id++;
                }
                id--;
                if (id < (s32) lp_k_a && (uniform_v_weights ? 1 : g_weights_a(u)) / 2 <= evict_start_a(id + 1) - temp_gain_a(u)) {
                    // at least half of vtx weight lies in chunk p
                    dest_part_a(u) = (partition_t) id;
                    return;
                }
                if (id < (s32) lp_k_a) {
                    temp_gain_a(u) = Kokkos::atomic_fetch_add(&max_vwgt_a(), (uniform_v_weights ? 1 : g_weights_a(u)));
                }
            }
            dest_part_a(u) = p_map_a(u);
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        return Kokkos::subview(lp.vtx1, std::make_pair((u32) 0, num_moves));
    }

    template<bool uniform_v_weights>
    inline UnmanagedDeviceVertex rebalance_weak(LabelPropagation &lp,
                                                Graph &g,
                                                const BlockConn &bc,
                                                DeviceExecutionSpace &exec_space) {
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
        HEIPA_PROFILE_SCOPE("refinement", "jetrw", "underloaded_blocks");
        auto p_bweights_u = lp.partition.bweights;
        auto underloaded_blocks = lp.underloaded_blocks;
        auto n_underloaded_blocks = lp.n_underloaded_blocks;
        auto lp_k_u = lp.k;

        Kokkos::parallel_for("underloaded_blocks", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
            //this scan is small so do it within a team instead of an entire grid to save kernel launch time
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, lp_k_u), [&](const partition_t i, partition_t &update, const bool final) {
                if (p_bweights_u(i) < max_b_w) {
                    if (final) {
                        underloaded_blocks(update) = i;
                    }
                    update++;
                }
                if (final && i + 1 == lp_k_u) {
                    n_underloaded_blocks() = update;
                }
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrw", "reset_minibuckets");
        Kokkos::deep_copy(exec_space, Kokkos::subview(lp.gain1, std::make_pair((vertex_t) 0, t_minibuckets + 1)), 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        // determine best block
        HEIPA_PROFILE_SCOPE("refinement", "jetrw", "best_block");
        auto p_map_b = lp.partition.map;
        auto p_bweights_b = lp.partition.bweights;
        auto dest_part_b = lp.dest_part;
        auto vtx2_b = lp.vtx2;
        auto temp_gain_b = lp.temp_gain;
        auto gain1_b = lp.gain1;
        auto lp_lmax_b = lp.lmax;
        auto underloaded_blocks_b = lp.underloaded_blocks;
        auto n_underloaded_blocks_b = lp.n_underloaded_blocks;
        auto g_weights_b = g.weights;
        auto bc_row_b = bc.row;
        auto bc_sizes_b = bc.sizes;
        auto bc_ids_b = bc.ids;
        auto bc_weights_b = bc.weights;

        Kokkos::parallel_for("best_block", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = p_map_b(u);
            partition_t best_id = u_id;

            weight_t gain = 0;
            if (p_bweights_b(u_id) > lp_lmax_b && (uniform_v_weights ? 1 : g_weights_b(u)) < 1.5 * (p_bweights_b(u_id) - opt_weight)) {
                weight_t best_id_w = 0;
                weight_t own_conn = 0;

                u32 r_beg = bc_row_b(u);
                u32 r_len = bc_sizes_b(u);
                u32 r_end = r_beg + r_len;
                for (u32 j = r_beg; j < r_end; j++) {
                    partition_t id = bc_ids_b(j);
                    weight_t w = bc_weights_b(j);

                    if (id != NULL_PART && id != HASH_RECLAIM && p_bweights_b(id) < max_b_w) {
                        if (w > best_id_w) {
                            best_id = id;
                            best_id_w = w;
                        }
                    }
                    if (id == u_id) {
                        own_conn = bc_weights_b(j);
                    }
                }

                gain = best_id_w - own_conn;

                if (best_id_w <= 0) {
                    u32 n_under = n_underloaded_blocks_b();
                    if (n_under > 0) {
                        best_id = underloaded_blocks_b(u % n_under);
                    } else {
                        best_id = u_id;
                    }
                    gain = -own_conn;
                }
            }
            dest_part_b(u) = best_id;

            if (u_id != best_id) {
                vertex_t gain_type = gain_bucket(gain, (uniform_v_weights ? 1 : g_weights_b(u)));
                vertex_t g_id = (MAX_BUCKETS * u_id + gain_type) * sections + (u % sections);
                temp_gain_b(u) = Kokkos::atomic_fetch_add(&gain1_b(g_id), (uniform_v_weights ? 1 : g_weights_b(u)));
                vtx2_b(u) = g_id;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrw", "scan_score_buckets");
        auto gain1_s = lp.gain1;
        if (t_minibuckets < 10000) {
            Kokkos::parallel_for("scan score buckets", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, 1024), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &t) {
                //this scan is small so do it within a team instead of an entire grid to save kernel launch time
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 1), [&](const vertex_t u, weight_t &update, const bool final) {
                    weight_t x = gain1_s(u);
                    if (final) {
                        gain1_s(u) = update;
                    }
                    update += x;
                });
            });
        } else {
            Kokkos::parallel_scan("scan score buckets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, t_minibuckets + 1), KOKKOS_LAMBDA(const vertex_t &u, weight_t &update, const bool final) {
                weight_t x = gain1_s(u);
                if (final) {
                    gain1_s(u) = update;
                }
                update += x;
            });
        }
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("refinement", "jetrw", "filter_scores");
        auto p_map_f = lp.partition.map;
        auto dest_part_f = lp.dest_part;
        auto vtx2_f = lp.vtx2;
        auto temp_gain_f = lp.temp_gain;
        auto gain1_f = lp.gain1;
        auto p_bweights_f = lp.partition.bweights;
        auto lp_lmax_f = lp.lmax;
        auto vtx1_f = lp.vtx1;

        Kokkos::parallel_scan("filter_scores", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
            partition_t u_id = p_map_f(u);
            partition_t best_id = dest_part_f(u);

            if (u_id != best_id) {
                vertex_t g_id = vtx2_f(u);

                vertex_t begin_bucket = u_id * width;
                weight_t temp_weight = temp_gain_f(u) + (gain1_f(g_id) - gain1_f(begin_bucket));
                weight_t limit = p_bweights_f(u_id) - lp_lmax_f;

                const vertex_t take = (vertex_t) (temp_weight < limit); // 0 or 1
                if (final && take) {
                    vtx1_f(update) = u;
                }
                update += take;
            }
        }, lp.scan_host);
        exec_space.fence();

        u32 num_moves = lp.scan_host();
        KOKKOS_PROFILE_FENCE(exec_space);

        return Kokkos::subview(lp.vtx1, std::make_pair((vertex_t) 0, num_moves));
    }

    template<bool uniform_v_weights, bool uniform_e_weights>
    inline void perform_moves(LabelPropagation &lp,
                              const Graph &g,
                              BlockConn &bc,
                              const UnmanagedDeviceVertex &moves,
                              weight_t &curr_max_weight,
                              weight_t &curr_edge_cut,
                              KokkosMemoryStack &mem_stack,
                              DeviceExecutionSpace &exec_space) {
        u32 n_moves = (u32) moves.extent(0);

        auto p_map = lp.partition.map;
        auto p_bweights = lp.partition.bweights;
        auto dest_part = lp.dest_part;
        auto dest_cache = lp.dest_cache;
        auto g_weights = g.weights;
        auto bc_row = bc.row;
        auto bc_sizes = bc.sizes;
        auto bc_ids = bc.ids;
        auto bc_weights = bc.weights;
        auto cut_change1 = lp.cut_change1;

        // first change in cut
        if (n_moves < 32) {
            using TeamPol = Kokkos::TeamPolicy<DeviceExecutionSpace>;
            using Member = TeamPol::member_type;

            HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "cut_change_1_<32");
            Kokkos::parallel_for("cut_change_1_small", TeamPol(exec_space, 1, 32), KOKKOS_LAMBDA(const Member &team) {
                weight_t sum = 0;

                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, (int) n_moves), [&](const int i, weight_t &gain_update) {
                    vertex_t u = moves((u32) i);
                    weight_t u_w = (uniform_v_weights ? 1 : g_weights(u));
                    partition_t old_id = p_map(u);
                    partition_t new_id = dest_part(u);

                    u32 beg = bc_row(u);
                    u32 len = bc_sizes(u);

                    // KEEP THESE LINES (unchanged)
                    weight_t old_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, old_id, len);
                    weight_t new_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, new_id, len);
                    gain_update += new_conn - old_conn;

                    // side effects (same as your original)
                    dest_cache(u) = NULL_PART;
                    Kokkos::atomic_add(&p_bweights(old_id), -u_w);
                    Kokkos::atomic_add(&p_bweights(new_id), u_w);

                    p_map(u) = new_id;
                    dest_part(u) = old_id;
                }, sum);

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    cut_change1() = sum; // device scalar view
                });
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        } else {
            HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "cut_change_1");
            Kokkos::parallel_reduce("cut_change_1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n_moves), KOKKOS_LAMBDA(const u32 &i, weight_t &gain_update) {
                vertex_t u = moves(i);
                weight_t u_w = (uniform_v_weights ? 1 : g_weights(u));
                partition_t old_id = p_map(u);
                partition_t new_id = dest_part(u);

                u32 beg = bc_row(u);
                u32 len = bc_sizes(u);
                weight_t old_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, old_id, len);
                weight_t new_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, new_id, len);
                gain_update += new_conn - old_conn;

                dest_cache(u) = NULL_PART;
                Kokkos::atomic_add(&p_bweights(old_id), -u_w);
                Kokkos::atomic_add(&p_bweights(new_id), u_w);

                p_map(u) = new_id;
                dest_part(u) = old_id;
            }, cut_change1);
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // update max weight
        HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "update_max_weight");
        auto host_max_part = lp.host_max_part;
        auto lp_k = lp.k;
        
        Kokkos::parallel_for("max_weight", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, 32), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            weight_t local_max = 0;

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, (int) lp_k), [&](const int i, weight_t &m) {
                const weight_t w = p_bweights(i); // bweights on device
                if (w > m) m = w;
            }, Kokkos::Max<weight_t>(local_max));

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                host_max_part() = local_max; // lp.max_part is a device scalar view
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // update block conn
        if (n_moves > (u32) g.n / 10) {
            update_large<uniform_e_weights>(g, lp.partition, lp.moved_round, lp.round, lp.dest_cache, bc, moves, exec_space);
        } else {
            update_small<uniform_e_weights>(g, lp.partition, lp.dest_part, lp.dest_cache, bc, moves, exec_space);
        }

        // second change in cut
        auto cut_change2 = lp.cut_change2;
        if (n_moves < 32) {
            using Exec = DeviceExecutionSpace;
            using TeamPolicy = Kokkos::TeamPolicy<Exec>;
            using Member = TeamPolicy::member_type;

            constexpr int TEAM_SIZE = 32;

            HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "cut_change_2_<32");
            Kokkos::parallel_for("cut_change_2_small", TeamPolicy(exec_space, 1, TEAM_SIZE), KOKKOS_LAMBDA(const Member &team) {
                weight_t sum = 0;

                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, (int) n_moves), [&](const int ii, weight_t &lsum) {
                    const vertex_t u = moves((vertex_t) ii);
                    const partition_t old_id = dest_part(u);
                    const partition_t new_id = p_map(u);

                    const u32 beg = bc_row(u);
                    const u32 len = bc_sizes(u);

                    weight_t old_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, old_id, len);
                    weight_t new_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, new_id, len);

                    lsum += (new_conn - old_conn);
                }, sum);

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    cut_change2() = sum; // device scalar
                });
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        } else {
            HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "cut_change_2");
            Kokkos::parallel_reduce("cut_change_2", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n_moves), KOKKOS_LAMBDA(const vertex_t &i, weight_t &gain_update) {
                vertex_t u = moves(i);
                partition_t old_id = dest_part(u);
                partition_t new_id = p_map(u);

                u32 beg = bc_row(u);
                u32 len = bc_sizes(u);
                weight_t old_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, old_id, len);
                weight_t new_conn = lookup(bc_ids.data() + beg, bc_weights.data() + beg, new_id, len);

                gain_update += new_conn - old_conn;
            }, cut_change2);
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        exec_space.fence();

        // update cut
        curr_max_weight = lp.host_max_part();
        curr_edge_cut -= (lp.cut_change2() + lp.cut_change1()) / 2;
    }

    template<bool uniform_v_weights, bool uniform_e_weights>
    inline std::pair<weight_t, weight_t> jet_refine(Graph &g,
                                                    Partition &partition,
                                                    partition_t k,
                                                    weight_t lmax,
                                                    bool use_ultra,
                                                    u32 level,
                                                    weight_t curr_edge_cut,
                                                    weight_t curr_max_weight,
                                                    KokkosMemoryStack &mem_stack,
                                                    DeviceExecutionSpace &exec_space) {
        LabelPropagation lp = initialize_label_propagation(g.n, g.m, k, lmax, mem_stack, exec_space);

        // copy partition
        HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "copy_partition");
        copy_into(lp.partition, partition, g.n, exec_space);
        KOKKOS_PROFILE_FENCE(exec_space);

        weight_t best_edge_cut = curr_edge_cut;
        weight_t best_max_weight = curr_max_weight;

        BlockConn bc;
        bc = init_BlockConn<uniform_e_weights>(g, lp.partition, mem_stack, exec_space);

        std::vector<f64> filter_ratios;

        if (use_ultra) {
            filter_ratios = {0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05};
        } else {
            if (level == 0) {
                filter_ratios.push_back(0.25);
            } else {
                filter_ratios.push_back(0.75);
            }
        }

        for (auto filter_ratio: filter_ratios) {
            u32 balance_iteration = 0;
            u32 iteration = 0;
            while (iteration < N_MAX_ITERATIONS) {
                iteration += 1;

                UnmanagedDeviceVertex moves;
                if (curr_max_weight <= lmax) {
                    moves = jet_lp<uniform_v_weights, uniform_e_weights>(lp, g, bc, filter_ratio, exec_space);
                    balance_iteration = 0;

                    // if lp found 0 moves, it will find 0 moves in the next iteration so skip
                    if (moves.extent(0) == 0) { break; }
                } else {
                    if (balance_iteration < N_MAX_WEAK_ITERATIONS) {
                        moves = rebalance_weak<uniform_v_weights>(lp, g, bc, exec_space);

                        // if weak reb found 0 moves, it will find 0 moves in the next iteration so skip to strong rebalance
                        if (moves.extent(0) == 0) {
                            balance_iteration = N_MAX_WEAK_ITERATIONS;
                            continue;
                        }
                    } else {
                        moves = rebalance_strong<uniform_v_weights>(lp, g, bc, exec_space);

                        // if strong reb found 0 moves, it will find 0 moves in the next iteration so skip
                        if (moves.extent(0) == 0) { break; }
                    }
                    balance_iteration++;
                }

                perform_moves<uniform_v_weights, uniform_e_weights>(lp, g, bc, moves, curr_max_weight, curr_edge_cut, mem_stack, exec_space);

                if (best_max_weight > lmax && curr_max_weight < best_max_weight) {
                    // copy the partition
                    HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "copy_partition");
                    copy_into(partition, lp.partition, g.n, exec_space);

                    best_edge_cut = curr_edge_cut;
                    best_max_weight = curr_max_weight;
                    iteration = 0;
                    KOKKOS_PROFILE_FENCE(exec_space);
                } else if (curr_edge_cut < best_edge_cut && (curr_max_weight <= lmax || curr_max_weight <= best_max_weight)) {
                    if ((f64) curr_edge_cut < PHI * (f64) best_edge_cut) { iteration = 0; }

                    // copy the partition
                    HEIPA_PROFILE_SCOPE("refinement", "JetLabelPropagation", "copy_partition");
                    copy_into(partition, lp.partition, g.n, exec_space);

                    best_edge_cut = curr_edge_cut;
                    best_max_weight = curr_max_weight;
                    KOKKOS_PROFILE_FENCE(exec_space);
                }
            }
        }

        free_BlockConn(bc, mem_stack);

        free_LabelPropagation(lp, mem_stack);

        return std::make_pair(best_edge_cut, best_max_weight);
    }
}

#endif //GPU_HEIPA_JET_LABEL_PROPAGATION_H
