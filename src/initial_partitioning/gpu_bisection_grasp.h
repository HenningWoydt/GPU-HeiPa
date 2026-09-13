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

#ifndef GPU_HEIPA_GPU_BISECTION_GRASP_H
#define GPU_HEIPA_GPU_BISECTION_GRASP_H

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/small_graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"

namespace GPU_HeiPa {

    template<bool uvw, bool uew>
    inline void batched_grasp_bisect(const GraphBatch &batch,
                                     const DeviceU8 &active_mask,
                                     const DeviceU32 &current_targets_dev,
                                     weight_t lmax_global,
                                     KokkosMemoryStack &mem_stack,
                                     DeviceExecutionSpace &exec_space,
                                     u64 seed = 0) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "batched_grasp_bisect");

        // Hyperparameters for GPU saturation: 16 teams per graph * 128 threads = 2048 trials per graph
        constexpr u32 TEAMS_PER_GRAPH = 16;
        constexpr int TEAM_SIZE = 128;
        const u32 k = batch.k;
        const u32 total_teams = k * TEAMS_PER_GRAPH;

        Kokkos::View<BestBisectConfig *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > team_results(
            (BestBisectConfig *) get_chunk_back(mem_stack, sizeof(BestBisectConfig) * total_teams), total_teams);

        auto d_actual_n = batch.batch_ns;
        auto d_results = batch.bisection_results;
        auto g_mem = batch.graph_memory;
        auto d_actual_g_weight = batch.batch_weights;
        vertex_t b_n = batch.n;
        vertex_t b_m = batch.m;
        u64 n_bytes_weights = round_up_64(b_n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(b_n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(b_m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, total_teams, TEAM_SIZE);

        Kokkos::parallel_for("batched_grasp_trials", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const u32 global_team_id = team.league_rank();
            const partition_t graph_id = global_team_id / TEAMS_PER_GRAPH;
            const u32 team_in_graph = global_team_id % TEAMS_PER_GRAPH;

            if (!active_mask(graph_id)) return;
            const vertex_t gn = d_actual_n(graph_id);
            if (gn <= 1) {
                if (gn == 1 && team_in_graph == 0) {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        partition_t *part = batch.get_partition_ptr(graph_id);
                        part[0] = 0;
                        d_results(graph_id).penalty = 0;
                        d_results(graph_id).cut = 0;
                        d_results(graph_id).config = 0;
                    });
                }
                return;
            }

            const partition_t tk = current_targets_dev(graph_id);
            const partition_t left_tk = tk & 0xFFFF;
            const partition_t right_tk = tk >> 16;
            const weight_t lmax_left = lmax_global * left_tk;
            const weight_t lmax_right = lmax_global * right_tk;
            const weight_t g_weight = d_actual_g_weight(graph_id);
            const partition_t total_tk = left_tk + right_tk;
            const weight_t target_w1 = total_tk > 0 ? (weight_t) (((u64) g_weight * right_tk) / total_tk) : (g_weight / 2);

            u8 *base = g_mem.data() + (u64) graph_id * n_bytes_one_graph;
            weight_t *g_w = (weight_t *) base;
            base += n_bytes_weights;
            u32 *g_n = (u32 *) base;
            base += n_bytes_neighborhood;
            base += n_bytes_edges_u;
            vertex_t *g_ev = (vertex_t *) base;
            base += n_bytes_edges_v;
            weight_t *g_ew = (weight_t *) base;

            BestBisectConfig team_best;
            BestBisectReducer reducer(team_best);
            reducer.init(team_best);

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](const int thread_idx, BestBisectConfig &trial_best) {
                const u32 trial_id = team_in_graph * TEAM_SIZE + thread_idx;
                const vertex_t seed_v = trial_id % gn;
                const u32 rollout = trial_id / gn;

                // Thread-local PRNG: fast 32-bit xorshift seeded uniquely per trial and global seed
                u32 rng = ((trial_id + 1) * 2654435761U) ^ ((graph_id + 1) * 2246822519U) ^ (u32) (seed * 2147483647ULL);
                auto xorshift32 = [&]() {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    return rng;
                };

                int gains[64];
                for (vertex_t v = 0; v < gn; ++v) {
                    int initial_gain = 0;
                    for (u32 e = g_n[v]; e < g_n[v + 1]; ++e) {
                        initial_gain -= (uew ? 1 : g_ew[e]);
                    }
                    gains[v] = initial_gain;
                }

                u64 part_mask = 0;
                weight_t w1 = 0;
                weight_t cut = 0;

                auto add_to_part = [&](vertex_t v) {
                    w1 += (uvw ? 1 : g_w[v]);
                    for (u32 e = g_n[v]; e < g_n[v + 1]; ++e) {
                        vertex_t u = g_ev[e];
                        weight_t ew = (uew ? 1 : g_ew[e]);
                        gains[u] += 2 * ew;
                        if (part_mask & (1ULL << u)) cut -= ew;
                        else cut += ew;
                    }
                    part_mask |= (1ULL << v);
                };

                add_to_part(seed_v);

                // Phase 1: Construction (GRASP randomized greedy with Restricted Candidate List)
                // If rollout == 0, strictly deterministic greedy (guarantees baseline heuristic quality)
                // If rollout > 0, randomize selection among RCL candidates within delta of best gain
                const int delta = (rollout == 0) ? -1 : (int) (rollout % 3);

                while (true) {
                    vertex_t best_v = gn;
                    if (delta < 0) {
                        int max_gain = -99999999;
                        for (vertex_t v = 0; v < gn; ++v) {
                            if (part_mask & (1ULL << v)) continue;
                            weight_t vw = uvw ? 1 : g_w[v];
                            if (w1 + vw <= target_w1 && gains[v] > max_gain) {
                                max_gain = gains[v];
                                best_v = v;
                            }
                        }
                    } else {
                        int max_gain = -99999999;
                        for (vertex_t v = 0; v < gn; ++v) {
                            if (part_mask & (1ULL << v)) continue;
                            weight_t vw = uvw ? 1 : g_w[v];
                            if (w1 + vw <= target_w1 && gains[v] > max_gain) {
                                max_gain = gains[v];
                            }
                        }

                        if (max_gain != -99999999) {
                            const int threshold = max_gain - delta;
                            vertex_t rcl[8];
                            int rcl_count = 0;
                            for (vertex_t v = 0; v < gn; ++v) {
                                if (part_mask & (1ULL << v)) continue;
                                weight_t vw = uvw ? 1 : g_w[v];
                                if (w1 + vw <= target_w1 && gains[v] >= threshold) {
                                    if (rcl_count < 8) {
                                        rcl[rcl_count++] = v;
                                    } else {
                                        u32 r = xorshift32() % (rcl_count + 1);
                                        if (r < 8) rcl[r] = v;
                                        rcl_count++;
                                    }
                                }
                            }
                            if (rcl_count > 0) {
                                int chosen = xorshift32() % (rcl_count < 8 ? rcl_count : 8);
                                best_v = rcl[chosen];
                            }
                        }
                    }

                    if (best_v != gn) {
                        add_to_part(best_v);
                        continue;
                    }

                    // Overshoot check
                    if (w1 < target_w1) {
                        vertex_t overshoot_v = gn;
                        int max_overshoot_gain = -99999999;
                        for (vertex_t v = 0; v < gn; ++v) {
                            if (part_mask & (1ULL << v)) continue;
                            weight_t vw = uvw ? 1 : g_w[v];
                            if (w1 + vw <= lmax_right && (w1 < g_weight - lmax_left || (w1 + vw - target_w1) < (target_w1 - w1))) {
                                if (gains[v] > max_overshoot_gain) {
                                    max_overshoot_gain = gains[v];
                                    overshoot_v = v;
                                }
                            }
                        }
                        if (overshoot_v != gn) {
                            add_to_part(overshoot_v);
                        }
                    }
                    break;
                }

                // Penalty scoring
                const weight_t wl = g_weight - w1;
                const u64 p_l = wl > lmax_left ? (u64) (wl - lmax_left) : 0;
                const u64 p_r = w1 > lmax_right ? (u64) (w1 - lmax_right) : 0;

                u64 penalty = p_l * p_l + p_r * p_r;
                if (p_l > 0 || p_r > 0) penalty += OVERLOAD_PENALTY;
                if (wl == 0 || w1 == 0) penalty += EMPTY_BLOCK_PENALTY;

                if (penalty < trial_best.penalty || (penalty == trial_best.penalty && cut < trial_best.cut)) {
                    trial_best.penalty = penalty;
                    trial_best.cut = cut;
                    trial_best.config = part_mask;
                }
            }, reducer);

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                team_results(global_team_id) = team_best;
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        Kokkos::parallel_for("final_reduce_grasp", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t graph_id) {
            if (!active_mask(graph_id)) return;
            const vertex_t gn = d_actual_n(graph_id);
            if (gn <= 1) return;

            BestBisectConfig best_overall;
            BestBisectReducer reducer(best_overall);
            reducer.init(best_overall);

            const u32 start_team = graph_id * TEAMS_PER_GRAPH;
            for (u32 t = 0; t < TEAMS_PER_GRAPH; ++t) {
                reducer.join(best_overall, team_results(start_team + t));
            }

            d_results(graph_id) = best_overall;
            const u64 config = best_overall.config;
            partition_t *part = batch.get_partition_ptr(graph_id);
            for (vertex_t u = 0; u < gn; ++u) {
                part[u] = (partition_t) ((config >> u) & 1ULL);
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        pop_back(mem_stack); // team_results
    }

} // namespace GPU_HeiPa

#endif // GPU_HEIPA_GPU_BISECTION_GRASP_H
