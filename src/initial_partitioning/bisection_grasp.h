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

#ifndef GPU_HEIPA_BISECTION_GRASP_H
#define GPU_HEIPA_BISECTION_GRASP_H

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/small_graph.h"
#include "../datastructures/mtx_graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"

namespace GPU_HeiPa {

    template<int N>
    struct BisectTopN {
        int gains[N];
        vertex_t v[N];
        int count = 0;

        KOKKOS_INLINE_FUNCTION void init() {
            count = 0;
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                gains[i] = -99999999;
            }
        }

        KOKKOS_INLINE_FUNCTION void insert(int gain, vertex_t vert) {
            if (count == N && gain <= gains[N - 1]) return;

            int pos = count < N ? count++ : (N - 1);
            while (pos > 0 && gains[pos - 1] < gain) {
                gains[pos] = gains[pos - 1];
                v[pos] = v[pos - 1];
                --pos;
            }
            gains[pos] = gain;
            v[pos] = vert;
        }
    };

    struct BestMove {
        u64 penalty = 0xFFFFFFFFFFFFFFFFULL;
        int gain = -99999999;
        vertex_t u = 0xFF;
        vertex_t v = 0xFF;
        weight_t delta_w1 = 0;

        KOKKOS_INLINE_FUNCTION BestMove() = default;
    };

    struct BestMoveReducer {
        using reducer = BestMoveReducer;
        using value_type = BestMove;
        using result_view_type = Kokkos::View<value_type, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

        KOKKOS_INLINE_FUNCTION void join(value_type &dest, const value_type &src) const {
            if (src.penalty < dest.penalty || (src.penalty == dest.penalty && src.gain > dest.gain)) {
                dest = src;
            }
        }

        KOKKOS_INLINE_FUNCTION void init(value_type &val) const {
            val.penalty = 0xFFFFFFFFFFFFFFFFULL;
            val.gain = -99999999;
            val.u = 0xFF;
            val.v = 0xFF;
            val.delta_w1 = 0;
        }

        value_type *value;

        KOKKOS_INLINE_FUNCTION BestMoveReducer(value_type &val) : value(&val) {}

        KOKKOS_INLINE_FUNCTION BestMoveReducer(const result_view_type &view) : value(view.data()) {}

        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }

        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }

        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    struct RefineTeamState {
        int single_gains[64];
        u64 part_mask;
        weight_t w1;
        weight_t wl;
        weight_t cut;
        u64 cur_penalty;
        int improved;
    };

    using BisectTop4 = BisectTopN<4>;

    template<vertex_t N, bool uvw, bool uew>
    inline void convert_GraphBatch_to_BatchMtxGraph(const GraphBatch &batch,
                                                   BatchMtxGraph<N> &mtx_batch,
                                                   const UnmanagedDevicePartition &active_graph_ids,
                                                   u32 num_active_graphs,
                                                   DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "convert_to_mtx", "convert_GraphBatch_to_BatchMtxGraph");

        auto d_actual_n = batch.batch_ns;
        auto d_actual_m = batch.batch_ms;
        auto d_actual_g_weight = batch.batch_weights;
        auto g_mem = batch.graph_memory;

        vertex_t b_n = batch.n;
        vertex_t b_m = batch.m;
        u64 n_bytes_weights = round_up_64(b_n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(b_n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(b_m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;

        auto graphs = mtx_batch.graphs;

        Kokkos::parallel_for("convert_batch_to_mtx", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, num_active_graphs, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const partition_t graph_id = active_graph_ids(team.league_rank());
            const vertex_t gn = d_actual_n(graph_id);
            const vertex_t gm = d_actual_m(graph_id);
            const weight_t gw = d_actual_g_weight(graph_id);

            u8 *base = g_mem.data() + (u64) graph_id * n_bytes_one_graph;
            weight_t *src_w = (weight_t *) base;
            base += n_bytes_weights;
            u32 *src_n = (u32 *) base;
            base += n_bytes_neighborhood;
            vertex_t *src_eu = (vertex_t *) base;
            base += n_bytes_edges_u;
            vertex_t *src_ev = (vertex_t *) base;
            base += n_bytes_edges_v;
            weight_t *src_ew = (weight_t *) base;

            MtxGraph<N> &g = graphs(graph_id);

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                g.n = gn;
                g.m = gm;
                g.g_weight = gw;
                g.uniform_vertex_weights = uvw;
                g.uniform_edge_weights = uew;
            });

            // Initialize adjacency matrix to 0
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N * N), [&](const int idx) {
                g.adj_matrix[idx] = 0;
            });

            // Initialize vertex weights
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int idx) {
                if ((vertex_t) idx < gn) {
                    g.weights[idx] = uvw ? 1 : src_w[idx];
                } else {
                    g.weights[idx] = 0;
                }
            });
            team.team_barrier();

            // Populate adjacency matrix from CSR edges
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn), [&](const vertex_t u) {
                const u32 start = src_n[u];
                const u32 end = src_n[u + 1];
                for (u32 e = start; e < end; ++e) {
                    vertex_t v = src_ev[e];
                    if (v == SENTINEL || v >= N) break;
                    weight_t w = uew ? 1 : src_ew[e];
                    g.adj_matrix[u * N + v] = w;
                }
            });

            // Copy global IDs
            vertex_t *src_gids = batch.get_global_ids_ptr(graph_id);
            vertex_t *dst_gids = mtx_batch.get_global_ids_ptr(graph_id);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn), [&](const vertex_t u) {
                dst_gids[u] = src_gids[u];
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);
    }

    template<vertex_t N, bool uvw, bool uew, int TOP_N = 4>
    inline void batched_grasp_bisect_mtx(const GraphBatch &batch,
                                         BatchMtxGraph<N> &mtx_batch,
                                         const DeviceU8 &active_mask,
                                         const UnmanagedDevicePartition &active_graph_ids,
                                         u32 num_active_graphs,
                                         const DeviceU32 &current_targets_dev,
                                         weight_t lmax_global,
                                         KokkosMemoryStack &mem_stack,
                                         DeviceExecutionSpace &exec_space,
                                         u64 seed = 0) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "batched_grasp_bisect_mtx");

        constexpr u32 TEAMS_PER_GRAPH = 4;
        constexpr int TEAM_SIZE = 128;
        const u32 total_teams = num_active_graphs * TEAMS_PER_GRAPH;

        Kokkos::View<BestBisectConfig *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > team_results(
            (BestBisectConfig *) get_chunk_back(mem_stack, sizeof(BestBisectConfig) * total_teams), total_teams);

        auto d_actual_n = batch.batch_ns;
        auto d_results = batch.bisection_results;
        auto graphs = mtx_batch.graphs;

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, total_teams, TEAM_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(sizeof(int) * N));

        Kokkos::parallel_for("batched_grasp_trials_mtx", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const u32 global_team_id = team.league_rank();
            const u32 active_idx = global_team_id / TEAMS_PER_GRAPH;
            const partition_t graph_id = active_graph_ids(active_idx);
            const u32 team_in_graph = global_team_id % TEAMS_PER_GRAPH;

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
            const auto &g = graphs(graph_id);
            const weight_t g_weight = g.g_weight;
            const partition_t total_tk = left_tk + right_tk;
            const weight_t target_w1 = total_tk > 0 ? (weight_t) (((u64) g_weight * right_tk) / total_tk) : (g_weight / 2);

            typedef Kokkos::View<int *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchInt;
            ScratchInt s_initial_gains(team.team_scratch(0), N);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](const int thread_idx) {
                if ((vertex_t) thread_idx < gn) {
                    int initial_gain = 0;
                    #pragma unroll
                    for (vertex_t v = 0; v < N; ++v) {
                        if (v < gn) initial_gain -= g.edge_weight(thread_idx, v);
                    }
                    s_initial_gains(thread_idx) = initial_gain;
                }
            });
            team.team_barrier();

            BestBisectConfig team_best;
            BestBisectReducer reducer(team_best);
            reducer.init(team_best);

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](const int thread_idx, BestBisectConfig &trial_best) {
                const u32 total_trials = TEAMS_PER_GRAPH * TEAM_SIZE;
                const u32 trial_id = team_in_graph * TEAM_SIZE + thread_idx;
                const vertex_t seed_v = (vertex_t) (((u64) trial_id * gn) / total_trials);
                const bool is_greedy = (trial_id == 0);

                u32 rng = ((trial_id + 1) * 2654435761U) ^ ((graph_id + 1) * 2246822519U) ^ (u32) (seed * 2147483647ULL);
                auto xorshift32 = [&]() {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    return rng;
                };

                int gains[N];
                #pragma unroll
                for (vertex_t v = 0; v < N; ++v) {
                    gains[v] = (v < gn) ? s_initial_gains(v) : -99999999;
                }

                u64 part_mask = 0;
                weight_t w1 = 0;
                weight_t cut = 0;

                auto add_to_part = [&](vertex_t v) {
                    w1 += (uvw ? 1 : g.weights[v]);
                    #pragma unroll
                    for (vertex_t u = 0; u < N; ++u) {
                        if (u < gn) {
                            weight_t ew = g.edge_weight(v, u);
                            gains[u] += 2 * ew;
                            if (part_mask & (1ULL << u)) cut -= ew;
                            else cut += ew;
                        }
                    }
                    part_mask |= (1ULL << v);
                };

                add_to_part(seed_v);

                while (true) {
                    vertex_t best_v = gn;
                    BisectTopN<TOP_N> top_n;
                    top_n.init();

                    for (vertex_t v = 0; v < gn; ++v) {
                        if (part_mask & (1ULL << v)) continue;
                        weight_t vw = uvw ? 1 : g.weights[v];
                        if (w1 + vw <= target_w1) {
                            top_n.insert(gains[v], v);
                        }
                    }

                    if (top_n.count > 0) {
                        const int chosen = is_greedy ? 0 : (int) (xorshift32() % (u32) top_n.count);
                        best_v = top_n.v[chosen];
                    }

                    if (best_v != gn) {
                        add_to_part(best_v);
                        continue;
                    }

                    if (w1 < target_w1) {
                        vertex_t overshoot_v = gn;
                        int max_overshoot_gain = -99999999;
                        for (vertex_t v = 0; v < gn; ++v) {
                            if (part_mask & (1ULL << v)) continue;
                            weight_t vw = uvw ? 1 : g.weights[v];
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

        // Phase 2: Refinement on MtxGraph
        constexpr int REFINE_TEAM_SIZE = 128;
        auto refine_policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, total_teams, REFINE_TEAM_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(sizeof(RefineTeamState)));

        Kokkos::parallel_for("grasp_refinement_mtx", refine_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const u32 global_team_id = team.league_rank();
            const u32 active_idx = global_team_id / TEAMS_PER_GRAPH;
            const partition_t graph_id = active_graph_ids(active_idx);
            const vertex_t gn = d_actual_n(graph_id);
            if (gn <= 1) return;

            typedef Kokkos::View<RefineTeamState, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchRefineState;
            ScratchRefineState s_state(team.team_scratch(0));

            const partition_t tk = current_targets_dev(graph_id);
            const partition_t left_tk = tk & 0xFFFF;
            const partition_t right_tk = tk >> 16;
            const weight_t lmax_left = lmax_global * left_tk;
            const weight_t lmax_right = lmax_global * right_tk;
            const auto &g = graphs(graph_id);
            const weight_t g_weight = g.g_weight;

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                BestBisectConfig cfg = team_results(global_team_id);
                s_state().part_mask = cfg.config;
                s_state().cut = cfg.cut;
                s_state().cur_penalty = cfg.penalty;

                weight_t w1 = 0;
                for (vertex_t v = 0; v < gn; ++v) {
                    if ((cfg.config >> v) & 1ULL) {
                        w1 += (uvw ? 1 : g.weights[v]);
                    }
                }
                s_state().w1 = w1;
                s_state().wl = g_weight - w1;
                s_state().improved = 0;
            });
            team.team_barrier();

            int passes = 0;
            while (passes < 3) {
                ++passes;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, REFINE_TEAM_SIZE), [&](const int thread_idx) {
                    if ((vertex_t) thread_idx < gn) {
                        const vertex_t u = (vertex_t) thread_idx;
                        const bool in_right = (s_state().part_mask >> u) & 1ULL;
                        int int_w = 0;
                        int ext_w = 0;
                        #pragma unroll
                        for (vertex_t w = 0; w < N; ++w) {
                            if (w < gn) {
                                weight_t ew = g.edge_weight(u, w);
                                bool w_in_right = (s_state().part_mask >> w) & 1ULL;
                                if (w_in_right == in_right) int_w += ew;
                                else ext_w += ew;
                            }
                        }
                        s_state().single_gains[u] = ext_w - int_w;
                    }
                });
                team.team_barrier();

                const u64 cur_mask = s_state().part_mask;
                const weight_t cur_w1 = s_state().w1;
                const weight_t cur_wl = s_state().wl;
                const u64 cur_penalty = s_state().cur_penalty;

                BestMove best_move;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, REFINE_TEAM_SIZE), [&](const int thread_idx, BestMove &my_best) {
                    for (vertex_t u = (vertex_t) thread_idx; u < gn; u += REFINE_TEAM_SIZE) {
                        const int gain = s_state().single_gains[u];
                        const weight_t wu = uvw ? 1 : g.weights[u];
                        const bool in_right = (cur_mask >> u) & 1ULL;
                        const weight_t new_w1 = in_right ? (cur_w1 - wu) : (cur_w1 + wu);
                        const weight_t new_wl = g_weight - new_w1;
                        if (new_w1 <= 0 || new_wl <= 0) continue;

                        const u64 new_p_l = new_wl > lmax_left ? (u64) (new_wl - lmax_left) : 0;
                        const u64 new_p_r = new_w1 > lmax_right ? (u64) (new_w1 - lmax_right) : 0;
                        u64 new_penalty = new_p_l * new_p_l + new_p_r * new_p_r;
                        if (new_p_l > 0 || new_p_r > 0) new_penalty += OVERLOAD_PENALTY;

                        if (new_penalty > cur_penalty) continue;
                        if (new_penalty == cur_penalty && gain <= 0) continue;

                        if (new_penalty < my_best.penalty || (new_penalty == my_best.penalty && gain > my_best.gain)) {
                            my_best.penalty = new_penalty;
                            my_best.gain = gain;
                            my_best.u = u;
                            my_best.v = 0xFF;
                            my_best.delta_w1 = in_right ? -wu : wu;
                        }
                    }

                    const int total_pairs = (int) (gn * gn);
                    for (int idx = thread_idx; idx < total_pairs; idx += REFINE_TEAM_SIZE) {
                        const vertex_t u = (vertex_t) (idx / gn);
                        const vertex_t v = (vertex_t) (idx % gn);
                        if (u >= v) continue;

                        const int g_u = s_state().single_gains[u];
                        const int g_v = s_state().single_gains[v];
                        const weight_t ew = g.edge_weight(u, v);

                        const bool u_in_right = (cur_mask >> u) & 1ULL;
                        const bool v_in_right = (cur_mask >> v) & 1ULL;
                        const weight_t wu = uvw ? 1 : g.weights[u];
                        const weight_t wv = uvw ? 1 : g.weights[v];

                        int pair_gain = 0;
                        weight_t delta_w = 0;

                        if (u_in_right != v_in_right) {
                            pair_gain = g_u + g_v + 2 * ew;
                            delta_w = u_in_right ? (wv - wu) : (wu - wv);
                        } else {
                            pair_gain = g_u + g_v - 2 * ew;
                            delta_w = u_in_right ? -(wu + wv) : (wu + wv);
                        }

                        const weight_t new_w1 = cur_w1 + delta_w;
                        const weight_t new_wl = g_weight - new_w1;
                        if (new_w1 <= 0 || new_wl <= 0) continue;

                        const u64 new_p_l = new_wl > lmax_left ? (u64) (new_wl - lmax_left) : 0;
                        const u64 new_p_r = new_w1 > lmax_right ? (u64) (new_w1 - lmax_right) : 0;
                        u64 new_penalty = new_p_l * new_p_l + new_p_r * new_p_r;
                        if (new_p_l > 0 || new_p_r > 0) new_penalty += OVERLOAD_PENALTY;

                        if (new_penalty > cur_penalty) continue;
                        if (new_penalty == cur_penalty && pair_gain <= 0) continue;

                        if (new_penalty < my_best.penalty || (new_penalty == my_best.penalty && pair_gain > my_best.gain)) {
                            my_best.penalty = new_penalty;
                            my_best.gain = pair_gain;
                            my_best.u = u;
                            my_best.v = v;
                            my_best.delta_w1 = delta_w;
                        }
                    }
                }, BestMoveReducer(best_move));

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    if (best_move.penalty < cur_penalty || (best_move.penalty == cur_penalty && best_move.gain > 0)) {
                        s_state().part_mask ^= (1ULL << best_move.u);
                        if (best_move.v != 0xFF) {
                            s_state().part_mask ^= (1ULL << best_move.v);
                        }
                        s_state().w1 += best_move.delta_w1;
                        s_state().wl = g_weight - s_state().w1;
                        s_state().cut -= best_move.gain;
                        s_state().cur_penalty = best_move.penalty;
                        s_state().improved = 1;
                    } else {
                        s_state().improved = 0;
                    }
                });
                team.team_barrier();

                if (!s_state().improved) break;
            }

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                BestBisectConfig cfg;
                cfg.config = s_state().part_mask;
                cfg.cut = s_state().cut;
                cfg.penalty = s_state().cur_penalty;
                team_results(global_team_id) = cfg;
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // Final reduce and write partition back
        Kokkos::parallel_for("final_reduce_grasp_mtx", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_active_graphs), KOKKOS_LAMBDA(const u32 active_idx) {
            const partition_t graph_id = active_graph_ids(active_idx);
            const vertex_t gn = d_actual_n(graph_id);
            if (gn <= 1) return;

            BestBisectConfig best_overall;
            BestBisectReducer reducer(best_overall);
            reducer.init(best_overall);

            const u32 start_team = active_idx * TEAMS_PER_GRAPH;
            for (u32 t = 0; t < TEAMS_PER_GRAPH; ++t) {
                reducer.join(best_overall, team_results(start_team + t));
            }

            d_results(graph_id) = best_overall;
            const u64 config = best_overall.config;
            partition_t *part = batch.get_partition_ptr(graph_id);
            partition_t *mtx_part = mtx_batch.get_partition_ptr(graph_id);
            for (vertex_t u = 0; u < gn; ++u) {
                partition_t p = (partition_t) ((config >> u) & 1ULL);
                part[u] = p;
                mtx_part[u] = p;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        pop_back(mem_stack); // team_results
    }

    template<bool uvw, bool uew, int TOP_N = 4>
    inline void batched_grasp_bisect(const GraphBatch &batch,
                                     const DeviceU8 &active_mask,
                                     const UnmanagedDevicePartition &active_graph_ids,
                                     u32 num_active_graphs,
                                     const DeviceU32 &current_targets_dev,
                                     weight_t lmax_global,
                                     KokkosMemoryStack &mem_stack,
                                     DeviceExecutionSpace &exec_space,
                                     u64 seed = 0) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "batched_grasp_bisect");

        if (num_active_graphs == 0) return;

        // Specialized path for small graphs (n <= 16) using dense adjacency matrix
        if (batch.n <= 16) {
            BatchMtxGraph<16> mtx_batch;
            init_BatchMtxGraph(mtx_batch, batch.k, mem_stack);
            convert_GraphBatch_to_BatchMtxGraph<16, uvw, uew>(batch, mtx_batch, active_graph_ids, num_active_graphs, exec_space);
            batched_grasp_bisect_mtx<16, uvw, uew, TOP_N>(batch, mtx_batch, active_mask, active_graph_ids, num_active_graphs, current_targets_dev, lmax_global, mem_stack, exec_space, seed);
            free_BatchMtxGraph(mtx_batch, mem_stack);
            return;
        }

        // Hyperparameters for GPU saturation: 4 teams per graph * 128 threads = 512 trials per graph
        constexpr u32 TEAMS_PER_GRAPH = 4;
        constexpr int TEAM_SIZE = 128;
        const u32 total_teams = num_active_graphs * TEAMS_PER_GRAPH;

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

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, total_teams, TEAM_SIZE).set_scratch_size(0, Kokkos::PerTeam(sizeof(int) * 64));

        Kokkos::parallel_for("batched_grasp_trials", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const u32 global_team_id = team.league_rank();
            const u32 active_idx = global_team_id / TEAMS_PER_GRAPH;
            const partition_t graph_id = active_graph_ids(active_idx);
            const u32 team_in_graph = global_team_id % TEAMS_PER_GRAPH;

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

            typedef Kokkos::View<int *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchInt;
            ScratchInt s_initial_gains(team.team_scratch(0), 64);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](const int thread_idx) {
                if ((vertex_t) thread_idx < gn) {
                    int initial_gain = 0;
                    for (u32 e = g_n[thread_idx]; e < g_n[thread_idx + 1]; ++e) {
                        if (g_ev[e] == SENTINEL) break;
                        initial_gain -= (uew ? 1 : g_ew[e]);
                    }
                    s_initial_gains(thread_idx) = initial_gain;
                }
            });
            team.team_barrier();

            BestBisectConfig team_best;
            BestBisectReducer reducer(team_best);
            reducer.init(team_best);

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](const int thread_idx, BestBisectConfig &trial_best) {
                const u32 total_trials = TEAMS_PER_GRAPH * TEAM_SIZE;
                const u32 trial_id = team_in_graph * TEAM_SIZE + thread_idx;
                // Warp alignment: consecutive threads in the same warp share the same seed vertex
                const vertex_t seed_v = (vertex_t) (((u64) trial_id * gn) / total_trials);
                // Exactly one thread per graph executes the deterministic greedy rollout
                const bool is_greedy = (trial_id == 0);

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
                    gains[v] = s_initial_gains(v);
                }

                u64 part_mask = 0;
                weight_t w1 = 0;
                weight_t cut = 0;

                auto add_to_part = [&](vertex_t v) {
                    w1 += (uvw ? 1 : g_w[v]);
                    for (u32 e = g_n[v]; e < g_n[v + 1]; ++e) {
                        vertex_t u = g_ev[e];
                        if (u == SENTINEL || u >= gn) break;
                        weight_t ew = (uew ? 1 : g_ew[e]);
                        gains[u] += 2 * ew;
                        if (part_mask & (1ULL << u)) cut -= ew;
                        else cut += ew;
                    }
                    part_mask |= (1ULL << v);
                };

                add_to_part(seed_v);

                // Phase 1: Construction (GRASP randomized greedy with Top-N candidates)
                while (true) {
                    vertex_t best_v = gn;
                    BisectTopN<TOP_N> top_n;
                    top_n.init();

                    for (vertex_t v = 0; v < gn; ++v) {
                        if (part_mask & (1ULL << v)) continue;
                        weight_t vw = uvw ? 1 : g_w[v];
                        if (w1 + vw <= target_w1) {
                            top_n.insert(gains[v], v);
                        }
                    }

                    if (top_n.count > 0) {
                        const int chosen = is_greedy ? 0 : (int) (xorshift32() % (u32) top_n.count);
                        best_v = top_n.v[chosen];
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

        // Phase 2: Parallel cooperative 1-flip and 2-swap refinement
        constexpr int REFINE_TEAM_SIZE = 256;
        auto refine_policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, total_teams, REFINE_TEAM_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(sizeof(RefineTeamState)));

        Kokkos::parallel_for("grasp_refinement_2swap", refine_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const u32 global_team_id = team.league_rank();
            const u32 active_idx = global_team_id / TEAMS_PER_GRAPH;
            const partition_t graph_id = active_graph_ids(active_idx);
            const vertex_t gn = d_actual_n(graph_id);
            if (gn <= 1) return;

            typedef Kokkos::View<RefineTeamState, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchRefineState;
            ScratchRefineState s_state(team.team_scratch(0));

            const partition_t tk = current_targets_dev(graph_id);
            const partition_t left_tk = tk & 0xFFFF;
            const partition_t right_tk = tk >> 16;
            const weight_t lmax_left = lmax_global * left_tk;
            const weight_t lmax_right = lmax_global * right_tk;
            const weight_t g_weight = d_actual_g_weight(graph_id);

            u8 *base = g_mem.data() + (u64) graph_id * n_bytes_one_graph;
            weight_t *g_w = (weight_t *) base;
            base += n_bytes_weights;
            u32 *g_n = (u32 *) base;
            base += n_bytes_neighborhood;
            base += n_bytes_edges_u;
            vertex_t *g_ev = (vertex_t *) base;
            base += n_bytes_edges_v;
            weight_t *g_ew = (weight_t *) base;

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                BestBisectConfig cfg = team_results(global_team_id);
                s_state().part_mask = cfg.config;
                s_state().cut = cfg.cut;
                s_state().cur_penalty = cfg.penalty;

                weight_t w1 = 0;
                for (vertex_t v = 0; v < gn; ++v) {
                    if ((cfg.config >> v) & 1ULL) {
                        w1 += (uvw ? 1 : g_w[v]);
                    }
                }
                s_state().w1 = w1;
                s_state().wl = g_weight - w1;
                s_state().improved = 0;
            });
            team.team_barrier();

            int passes = 0;
            while (passes < 3) {
                ++passes;

                // Step 1: Precalculate all single-vertex gains in parallel across the team
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, REFINE_TEAM_SIZE), [&](const int thread_idx) {
                    if ((vertex_t) thread_idx < gn) {
                        const vertex_t u = (vertex_t) thread_idx;
                        const bool in_right = (s_state().part_mask >> u) & 1ULL;
                        int int_w = 0;
                        int ext_w = 0;
                        for (u32 e = g_n[u]; e < g_n[u + 1]; ++e) {
                            vertex_t w = g_ev[e];
                            weight_t ew = (uew ? 1 : g_ew[e]);
                            bool w_in_right = (s_state().part_mask >> w) & 1ULL;
                            if (w_in_right == in_right) int_w += ew;
                            else ext_w += ew;
                        }
                        s_state().single_gains[u] = ext_w - int_w;
                    }
                });
                team.team_barrier();

                // Step 2: Concurrently evaluate all 1-flips and 2-swaps, reducing to the best move
                const u64 cur_mask = s_state().part_mask;
                const weight_t cur_w1 = s_state().w1;
                const weight_t cur_wl = s_state().wl;
                const u64 cur_penalty = s_state().cur_penalty;

                BestMove best_move;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, REFINE_TEAM_SIZE), [&](const int thread_idx, BestMove &my_best) {
                    // 1-flips:
                    for (vertex_t u = (vertex_t) thread_idx; u < gn; u += REFINE_TEAM_SIZE) {
                        const int gain = s_state().single_gains[u];
                        const weight_t wu = uvw ? 1 : g_w[u];
                        const bool in_right = (cur_mask >> u) & 1ULL;
                        const weight_t new_w1 = in_right ? (cur_w1 - wu) : (cur_w1 + wu);
                        const weight_t new_wl = g_weight - new_w1;
                        if (new_w1 <= 0 || new_wl <= 0) continue;

                        const u64 new_p_l = new_wl > lmax_left ? (u64) (new_wl - lmax_left) : 0;
                        const u64 new_p_r = new_w1 > lmax_right ? (u64) (new_w1 - lmax_right) : 0;
                        u64 new_penalty = new_p_l * new_p_l + new_p_r * new_p_r;
                        if (new_p_l > 0 || new_p_r > 0) new_penalty += OVERLOAD_PENALTY;

                        if (new_penalty > cur_penalty) continue;
                        if (new_penalty == cur_penalty && gain <= 0) continue;

                        if (new_penalty < my_best.penalty || (new_penalty == my_best.penalty && gain > my_best.gain)) {
                            my_best.penalty = new_penalty;
                            my_best.gain = gain;
                            my_best.u = u;
                            my_best.v = 0xFF;
                            my_best.delta_w1 = in_right ? -wu : wu;
                        }
                    }

                    // 2-flips (any pair u < v):
                    const int total_pairs = (int) (gn * gn);
                    for (int idx = thread_idx; idx < total_pairs; idx += REFINE_TEAM_SIZE) {
                        const vertex_t u = (vertex_t) (idx / gn);
                        const vertex_t v = (vertex_t) (idx % gn);
                        if (u >= v) continue;

                        const int g_u = s_state().single_gains[u];
                        const int g_v = s_state().single_gains[v];

                        weight_t ew = 0;
                        for (u32 e = g_n[u]; e < g_n[u + 1]; ++e) {
                            if (g_ev[e] == v) {
                                ew = (uew ? 1 : g_ew[e]);
                                break;
                            }
                        }

                        const bool u_in_right = (cur_mask >> u) & 1ULL;
                        const bool v_in_right = (cur_mask >> v) & 1ULL;
                        const bool same_part = (u_in_right == v_in_right);
                        const int gain = g_u + g_v + (same_part ? (2 * ew) : (-2 * ew));

                        const weight_t wu = uvw ? 1 : g_w[u];
                        const weight_t wv = uvw ? 1 : g_w[v];
                        const weight_t delta_w1 = (u_in_right ? -wu : wu) + (v_in_right ? -wv : wv);
                        const weight_t new_w1 = cur_w1 + delta_w1;
                        const weight_t new_wl = g_weight - new_w1;
                        if (new_w1 <= 0 || new_wl <= 0) continue;

                        const u64 new_p_l = new_wl > lmax_left ? (u64) (new_wl - lmax_left) : 0;
                        const u64 new_p_r = new_w1 > lmax_right ? (u64) (new_w1 - lmax_right) : 0;
                        u64 new_penalty = new_p_l * new_p_l + new_p_r * new_p_r;
                        if (new_p_l > 0 || new_p_r > 0) new_penalty += OVERLOAD_PENALTY;

                        if (new_penalty > cur_penalty) continue;
                        if (new_penalty == cur_penalty && gain <= 0) continue;

                        if (new_penalty < my_best.penalty || (new_penalty == my_best.penalty && gain > my_best.gain)) {
                            my_best.penalty = new_penalty;
                            my_best.gain = gain;
                            my_best.u = u;
                            my_best.v = v;
                            my_best.delta_w1 = delta_w1;
                        }
                    }
                }, BestMoveReducer(best_move));

                // Step 3: Apply the best move if improving
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    if (best_move.penalty < cur_penalty || (best_move.penalty == cur_penalty && best_move.gain > 0)) {
                        if (best_move.v == 0xFF) {
                            s_state().part_mask ^= (1ULL << best_move.u);
                        } else {
                            s_state().part_mask ^= (1ULL << best_move.u) ^ (1ULL << best_move.v);
                        }
                        s_state().w1 += best_move.delta_w1;
                        s_state().wl = g_weight - s_state().w1;
                        s_state().cut -= best_move.gain;
                        s_state().cur_penalty = best_move.penalty;
                        s_state().improved = 1;
                    } else {
                        s_state().improved = 0;
                    }
                });
                team.team_barrier();

                if (!s_state().improved) {
                    break;
                }
            }

            // Write back refined result to team_results
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                BestBisectConfig cfg;
                cfg.config = s_state().part_mask;
                cfg.cut = s_state().cut;
                cfg.penalty = s_state().cur_penalty;
                team_results(global_team_id) = cfg;
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        Kokkos::parallel_for("final_reduce_grasp", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_active_graphs), KOKKOS_LAMBDA(const u32 active_idx) {
            const partition_t graph_id = active_graph_ids(active_idx);
            const vertex_t gn = d_actual_n(graph_id);
            if (gn <= 1) return;

            BestBisectConfig best_overall;
            BestBisectReducer reducer(best_overall);
            reducer.init(best_overall);

            const u32 start_team = active_idx * TEAMS_PER_GRAPH;
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

#endif // GPU_HEIPA_BISECTION_GRASP_H
