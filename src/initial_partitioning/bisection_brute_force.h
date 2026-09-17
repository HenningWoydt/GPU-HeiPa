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

#ifndef GPU_HEIPA_BISECTION_BRUTE_FORCE_H
#define GPU_HEIPA_BISECTION_BRUTE_FORCE_H

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

    template<bool uvw, bool uew, int CHUNK>
    inline void brute_force_bisect_async(const Graph &g,
                                         const UnmanagedDeviceWeight& left_lmax,
                                         const UnmanagedDeviceWeight& right_lmax,
                                         partition_t id,
                                         UnmanagedDevicePartition &partition_map,
                                         const Kokkos::View<BestBisectConfig, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >& result_view,
                                         DeviceExecutionSpace &exec_space) {
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
                partition_map(0) = 0;
            });
            return;
        }

        // Initialize the result view on device
        Kokkos::parallel_for("init_result", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
            result_view().penalty = 0xFFFFFFFFFFFFFFFFULL;
            result_view().cut = 0x7FFFFFFF;
            result_view().config = 0;
        });

        const vertex_t gn = g.n;
        const u32 gm = g.m;
        const vertex_t last = gn - 1;
        const u32 shift = (last >= 28) ? 28 : last;
        const u64 num_configs = 1ULL << shift;

        const int team_size = 256;
        const u64 configs_per_team = (u64) team_size * CHUNK;
        const u32 num_teams = (u32) ((num_configs + configs_per_team - 1) / configs_per_team);

        size_t shmem_size = (gn + 1) * sizeof(u32); // neighborhood
        shmem_size += gm * sizeof(vertex_t); // edges_u
        shmem_size += gm * sizeof(vertex_t); // edges_v
        if (!uvw) shmem_size += gn * sizeof(weight_t); // weights
        if (!uew) shmem_size += gm * sizeof(weight_t); // edges_w

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, num_teams, team_size).set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_reduce("brute_force_bisect_reduction", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team, BestBisectConfig &team_best) {
            const weight_t lmax_left = left_lmax(id);
            const weight_t lmax_right = right_lmax(id);
            typedef Kokkos::View<u32 *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchU32;
            typedef Kokkos::View<vertex_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchVertex;
            typedef Kokkos::View<weight_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchWeight;

            ScratchU32 s_neigh(team.team_scratch(0), gn + 1);
            ScratchVertex s_edges_u(team.team_scratch(0), gm);
            ScratchVertex s_edges_v(team.team_scratch(0), gm);
            ScratchWeight s_weights;
            if (!uvw) s_weights = ScratchWeight(team.team_scratch(0), gn);
            ScratchWeight s_edges_w;
            if (!uew) s_edges_w = ScratchWeight(team.team_scratch(0), gm);

            // Load graph data into shared memory
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn + 1), [&](const u32 i) {
                s_neigh(i) = g.neighborhood(i);
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gm), [&](const u32 i) {
                s_edges_u(i) = g.edges_u(i);
                s_edges_v(i) = g.edges_v(i);
            });
            if (!uvw) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn), [&](const vertex_t i) {
                    s_weights(i) = g.weights(i);
                });
            }
            if (!uew) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gm), [&](const u32 i) {
                    s_edges_w(i) = g.edges_w(i);
                });
            }
            team.team_barrier();

            BestBisectConfig best_in_team;
            BestBisectReducer reducer(best_in_team);
            reducer.init(best_in_team);

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, team_size), [&](const int tid, BestBisectConfig &local_best) {
                const u64 chunk_id = (u64) team.league_rank() * team_size + tid;
                const u64 begin = chunk_id * CHUNK;
                if (begin >= num_configs) return;
                const u64 end = begin + CHUNK < num_configs ? begin + CHUNK : num_configs;

                u64 gray = begin ^ (begin >> 1);
                weight_t wr = 0;
                for (vertex_t u = 0; u < last; ++u) {
                    if ((gray >> u) & 1ULL) {
                        wr += uvw ? 1 : s_weights(u);
                    }
                }

                weight_t cut = 0;
                for (u32 e = 0; e < gm; ++e) {
                    const vertex_t u = s_edges_u(e);
                    const vertex_t v = s_edges_v(e);
                    if (u < v) {
                        const u64 pu = (gray >> u) & 1ULL;
                        const u64 pv = (gray >> v) & 1ULL;
                        if (pu != pv) {
                            cut += uew ? 1 : s_edges_w(e);
                        }
                    }
                }

                auto evaluate_current = [&](const u64 config, const weight_t wr_cur, const weight_t cut_cur, BestBisectConfig &best_cur) {
                    const weight_t wl = g.g_weight - wr_cur;
                    const u64 p_l = wl > lmax_left ? (u64) (wl - lmax_left) : 0;
                    const u64 p_r = wr_cur > lmax_right ? (u64) (wr_cur - lmax_right) : 0;
                    u64 penalty = p_l * p_l + p_r * p_r;
                    if (p_l > 0 || p_r > 0) penalty += OVERLOAD_PENALTY;

                    if (wl == 0 || wr_cur == 0) {
                        penalty += EMPTY_BLOCK_PENALTY;
                    }

                    if (penalty < best_cur.penalty || (penalty == best_cur.penalty && cut_cur < best_cur.cut)) {
                        best_cur.penalty = penalty;
                        best_cur.cut = cut_cur;
                        best_cur.config = config;
                    }
                };

                evaluate_current(gray, wr, cut, local_best);

                // #pragma unroll
                for (u64 i = begin + 1; i < end; i++) {
                    #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                    const vertex_t flip_u = (vertex_t) __ffsll(i) - 1;
                    #else
                    const vertex_t flip_u = (vertex_t) __builtin_ctzll(i);
                    #endif

                    gray ^= (1ULL << flip_u);
                    const u64 new_part_u = (gray >> flip_u) & 1ULL;
                    const u64 old_part_u = new_part_u ^ 1ULL;
                    const weight_t wu = uvw ? 1 : s_weights(flip_u);

                    if (new_part_u) wr += wu;
                    else wr -= wu;

                    for (u32 e = s_neigh(flip_u); e < s_neigh(flip_u + 1); ++e) {
                        const vertex_t v = s_edges_v(e);
                        const u64 part_v = (gray >> v) & 1ULL;
                        const bool was_cut = old_part_u != part_v;
                        const bool now_cut = new_part_u != part_v;
                        const weight_t ew = uew ? 1 : s_edges_w(e);
                        if (was_cut && !now_cut) cut -= ew;
                        else if (!was_cut && now_cut) cut += ew;
                    }
                    evaluate_current(gray, wr, cut, local_best);
                }
            }, reducer);

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                BestBisectReducer(team_best).join(team_best, best_in_team);
            });
        }, BestBisectReducer(result_view));

        Kokkos::parallel_for("apply_best_config", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            partition_map(u) = (partition_t) ((result_view().config >> u) & 1ULL);
        });
    }

    template<bool uvw, bool uew, bool DO_HEURISTIC = true>
    inline void batched_brute_force_bisect(const GraphBatch &batch,
                                           const DeviceU8 &active_mask,
                                           const DeviceU32 &current_targets_dev,
                                           weight_t lmax_global,
                                           KokkosMemoryStack &mem_stack,
                                           DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "batched_brute_force_bisect");

        // --- Hyperparameters ---
        // Number of threads per Kokkos team on the GPU. Affects parallelism and register/shared memory usage.
        constexpr int TEAM_SIZE = 256;
        // The number of partition configurations (Gray code steps) each thread evaluates sequentially.
        constexpr int CHUNK = 128 * 4;
        // -----------------------
        const u32 k = batch.k;

        UnmanagedDeviceU32 teams_per_graph((u32 *) get_chunk_back(mem_stack, sizeof(u32) * k), k);
        UnmanagedDeviceU32 teams_offset((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (k + 1)), k + 1);
        UnmanagedDeviceU32 max_sizes((u32 *) get_chunk_back(mem_stack, sizeof(u32) * 2), 2);
        Kokkos::deep_copy(exec_space, max_sizes, 0);

        auto d_actual_n = batch.batch_ns;
        auto d_actual_m = batch.batch_ms;

        u32 total_teams = 0;
        u32 max_n = 0;
        u32 max_m = 0;
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "calc_teams");
        Kokkos::parallel_for("calc_teams", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t id) {
            if (!active_mask(id)) {
                teams_per_graph(id) = 0;
                return;
            }
            vertex_t gn = d_actual_n(id);
            if (gn <= 1) {
                teams_per_graph(id) = 0;
                return;
            }
            const u32 shift = (gn - 1 >= 28) ? 28 : (gn - 1);
            const u64 num_configs = 1ULL << shift;
            const u64 configs_per_team = (u64) TEAM_SIZE * CHUNK;
            const u32 num_teams = (u32) ((num_configs + configs_per_team - 1) / configs_per_team);
            teams_per_graph(id) = num_teams;
            Kokkos::atomic_max(&max_sizes(0), (u32) gn);
            Kokkos::atomic_max(&max_sizes(1), (u32) d_actual_m(id));
        });

        Kokkos::parallel_scan("prefix_sum_teams", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t id, u32 &update, const bool final) {
            if (final) teams_offset(id) = update;
            update += teams_per_graph(id);
            if (final && id == k - 1) teams_offset(k) = update;
        });

        Kokkos::deep_copy(exec_space, total_teams, Kokkos::subview(teams_offset, k));
        Kokkos::deep_copy(exec_space, max_n, Kokkos::subview(max_sizes, 0));
        Kokkos::deep_copy(exec_space, max_m, Kokkos::subview(max_sizes, 1));
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "bisect1_batched");
        Kokkos::parallel_for("bisect1_batched", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t id) {
            if (active_mask(id) && d_actual_n(id) == 1) {
                partition_t *part = batch.get_partition_ptr(id);
                part[0] = 0;
                batch.bisection_results(id).penalty = 0;
                batch.bisection_results(id).cut = 0;
                batch.bisection_results(id).config = 0;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        if (total_teams == 0) {
            pop_back(mem_stack); // max_sizes
            pop_back(mem_stack); // teams_offset
            pop_back(mem_stack); // teams_per_graph
            return;
        }

        Kokkos::View<BestBisectConfig *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > team_results((BestBisectConfig *) get_chunk_back(mem_stack, sizeof(BestBisectConfig) * total_teams), total_teams);
        Kokkos::View<BestBisectConfig *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > heuristic_results;
        
        if constexpr (DO_HEURISTIC) {
            heuristic_results = Kokkos::View<BestBisectConfig *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >((BestBisectConfig *) get_chunk_back(mem_stack, sizeof(BestBisectConfig) * k), k);
        }

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

        if constexpr (DO_HEURISTIC) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "batched_brute_force_bisect", "heuristic_prepass");
            Kokkos::parallel_for("heuristic_prepass", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, k, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
                const partition_t graph_id = team.league_rank();
                if (!active_mask(graph_id)) return;
                
                const vertex_t gn = d_actual_n(graph_id);
                if (gn <= 1) return;

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
                // vertex_t *g_eu = (vertex_t *) base;
                base += n_bytes_edges_u;
                vertex_t *g_ev = (vertex_t *) base;
                base += n_bytes_edges_v;
                weight_t *g_ew = (weight_t *) base;

                BestBisectConfig best_in_team;
                BestBisectReducer reducer(best_in_team);
                reducer.init(best_in_team);

                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, gn), [&](const vertex_t seed, BestBisectConfig &heuristic_best) {
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

                    add_to_part(seed);

                    while (true) {
                        vertex_t best_v = gn;
                        int max_gain = -99999999;
                        for (vertex_t v = 0; v < gn; ++v) {
                            if (part_mask & (1ULL << v)) continue;
                            weight_t vw = uvw ? 1 : g_w[v];
                            if (w1 + vw <= target_w1 && gains[v] > max_gain) {
                                max_gain = gains[v];
                                best_v = v;
                            }
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

                    const weight_t wl = g_weight - w1;
                    const u64 p_l = wl > lmax_left ? (u64) (wl - lmax_left) : 0;
                    const u64 p_r = w1 > lmax_right ? (u64) (w1 - lmax_right) : 0;
                    u64 penalty = p_l * p_l + p_r * p_r;
                    if (p_l > 0 || p_r > 0) penalty += OVERLOAD_PENALTY;

                    if (wl == 0 || w1 == 0) penalty += EMPTY_BLOCK_PENALTY;

                    if (penalty < heuristic_best.penalty || (penalty == heuristic_best.penalty && cut < heuristic_best.cut)) {
                        heuristic_best.penalty = penalty;
                        heuristic_best.cut = cut;
                        heuristic_best.config = part_mask;
                    }
                }, reducer);

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    heuristic_results(graph_id) = best_in_team;
                });
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "bisect_kernel");
        typedef Kokkos::View<u32 *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchU32;
        typedef Kokkos::View<vertex_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchVertex;
        typedef Kokkos::View<weight_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchWeight;

        size_t shmem_size = ScratchU32::shmem_size(max_n + 1);
        shmem_size += ScratchVertex::shmem_size(max_m);
        if (!uvw) shmem_size += ScratchWeight::shmem_size(max_n);
        if (!uew) shmem_size += ScratchWeight::shmem_size(max_m);

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, total_teams, TEAM_SIZE).set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for("init_team_results", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, total_teams), KOKKOS_LAMBDA(const u32 i) {
            team_results(i).penalty = 0xFFFFFFFFFFFFFFFFULL;
            team_results(i).cut = 0x7FFFFFFF;
            team_results(i).config = 0;
        });

        Kokkos::parallel_for("batched_brute_force_bisect", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const u32 global_rank = team.league_rank();

            partition_t left = 0;
            partition_t right = k - 1;
            partition_t graph_id = 0;
            while (left <= right) {
                partition_t mid = left + (right - left) / 2;
                if (teams_offset(mid) <= global_rank && teams_offset(mid + 1) > global_rank) {
                    graph_id = mid;
                    break;
                } else if (teams_offset(mid) > global_rank) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            const u32 local_team_rank = global_rank - teams_offset(graph_id);
            const vertex_t gn = d_actual_n(graph_id);
            const u32 gm = d_actual_m(graph_id);
            const partition_t tk = current_targets_dev(graph_id);
            const partition_t left_tk = tk & 0xFFFF;
            const partition_t right_tk = tk >> 16;
            const weight_t lmax_left = lmax_global * left_tk;
            const weight_t lmax_right = lmax_global * right_tk;
            const weight_t g_weight = d_actual_g_weight(graph_id);
            const vertex_t last = gn - 1;
            const u64 num_configs = 1ULL << last;

            typedef Kokkos::View<u32 *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchU32;
            typedef Kokkos::View<vertex_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchVertex;
            typedef Kokkos::View<weight_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchWeight;

            ScratchU32 s_neigh(team.team_scratch(0), gn + 1);
            ScratchVertex s_edges_u(team.team_scratch(0), gm);
            ScratchVertex s_edges_v(team.team_scratch(0), gm);
            ScratchWeight s_weights;
            if (!uvw) s_weights = ScratchWeight(team.team_scratch(0), gn);
            ScratchWeight s_edges_w;
            if (!uew) s_edges_w = ScratchWeight(team.team_scratch(0), gm);

            u8 *base = g_mem.data() + (u64) graph_id * n_bytes_one_graph;
            weight_t *g_w = (weight_t *) base;
            base += n_bytes_weights;
            u32 *g_n = (u32 *) base;
            base += n_bytes_neighborhood;
            vertex_t *g_eu = (vertex_t *) base;
            base += n_bytes_edges_u;
            vertex_t *g_ev = (vertex_t *) base;
            base += n_bytes_edges_v;
            weight_t *g_ew = (weight_t *) base;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn + 1), [&](const u32 i) {
                s_neigh(i) = g_n[i];
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gm), [&](const u32 i) {
                s_edges_u(i) = g_eu[i];
                s_edges_v(i) = g_ev[i];
            });
            if (!uvw) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn), [&](const vertex_t i) {
                    s_weights(i) = g_w[i];
                });
            }
            if (!uew) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gm), [&](const u32 i) {
                    s_edges_w(i) = g_ew[i];
                });
            }
            team.team_barrier();

            BestBisectConfig best_in_team;
            BestBisectReducer reducer(best_in_team);
            reducer.init(best_in_team);

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](const int tid, BestBisectConfig &local_best) {
                (void)heuristic_results; if constexpr (DO_HEURISTIC) {
                    BestBisectConfig h_best = heuristic_results(graph_id);
                    if (h_best.penalty < local_best.penalty || (h_best.penalty == local_best.penalty && h_best.cut < local_best.cut)) {
                        local_best = h_best;
                    }
                }

                const u64 chunk_id = (u64) local_team_rank * TEAM_SIZE + tid;
                const u64 begin = chunk_id * CHUNK;
                if (begin >= num_configs) return;
                const u64 end = begin + CHUNK < num_configs ? begin + CHUNK : num_configs;
                
                (void)heuristic_results; if constexpr (DO_HEURISTIC) {
                    const u64 chunk_gray = begin ^ (begin >> 1);
                    weight_t min_chunk_cut = 0;
                    const vertex_t K = 9; // Since CHUNK = 512 = 2^9
                    const vertex_t num_var = gn < K ? gn : K;
                    
                    for (vertex_t u = num_var; u < gn; ++u) {
                        const bool u_part = (chunk_gray >> u) & 1ULL;
                        for (u32 e = s_neigh(u); e < s_neigh(u + 1); ++e) {
                            vertex_t v = s_edges_v(e);
                            if (v >= num_var && v > u) {
                                const bool v_part = (chunk_gray >> v) & 1ULL;
                                if (u_part != v_part) min_chunk_cut += uew ? 1 : s_edges_w(e);
                            }
                        }
                    }
                    
                    for (vertex_t u = 0; u < num_var; ++u) {
                        weight_t cut_if_0 = 0;
                        weight_t cut_if_1 = 0;
                        for (u32 e = s_neigh(u); e < s_neigh(u + 1); ++e) {
                            vertex_t v = s_edges_v(e);
                            if (v >= num_var) {
                                const bool v_part = (chunk_gray >> v) & 1ULL;
                                const weight_t ew = uew ? 1 : s_edges_w(e);
                                if (v_part == 0) cut_if_1 += ew;
                                else cut_if_0 += ew;
                            }
                        }
                        min_chunk_cut += (cut_if_0 < cut_if_1) ? cut_if_0 : cut_if_1;
                    }

                    if (local_best.penalty == 0 && min_chunk_cut >= local_best.cut) {
                        return;
                    }
                }

                u64 gray = begin ^ (begin >> 1);
                weight_t wr = 0;
                for (vertex_t u = 0; u < last; ++u) {
                    if ((gray >> u) & 1ULL) {
                        wr += uvw ? 1 : s_weights(u);
                    }
                }

                weight_t cut = 0;
                for (u32 e = 0; e < gm; ++e) {
                    const vertex_t u = s_edges_u(e);
                    const vertex_t v = s_edges_v(e);
                    if (u < v) {
                        const u64 pu = (gray >> u) & 1ULL;
                        const u64 pv = (gray >> v) & 1ULL;
                        if (pu != pv) {
                            cut += uew ? 1 : s_edges_w(e);
                        }
                    }
                }

                auto evaluate_current = [&](const u64 config, const weight_t wr_cur, const weight_t cut_cur, BestBisectConfig &best_cur) {
                    const weight_t wl = g_weight - wr_cur;
                    const u64 p_l = wl > lmax_left ? (u64) (wl - lmax_left) : 0;
                    const u64 p_r = wr_cur > lmax_right ? (u64) (wr_cur - lmax_right) : 0;
                    u64 penalty = p_l * p_l + p_r * p_r;
                    if (p_l > 0 || p_r > 0) penalty += OVERLOAD_PENALTY;

                    if (wl == 0 || wr_cur == 0) {
                        penalty += EMPTY_BLOCK_PENALTY;
                    }

                    if (penalty < best_cur.penalty || (penalty == best_cur.penalty && cut_cur < best_cur.cut)) {
                        best_cur.penalty = penalty;
                        best_cur.cut = cut_cur;
                        best_cur.config = config;
                    }
                };

                evaluate_current(gray, wr, cut, local_best);

                // #pragma unroll
                for (u64 i = begin + 1; i < end; i++) {
                    #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                    const vertex_t flip_u = (vertex_t) __ffsll(i) - 1;
                    #else
                    const vertex_t flip_u = (vertex_t) __builtin_ctzll(i);
                    #endif

                    if (flip_u >= gn) Kokkos::abort("flip_u out of bounds");

                    gray ^= (1ULL << flip_u);
                    const u64 new_part_u = (gray >> flip_u) & 1ULL;
                    const u64 old_part_u = new_part_u ^ 1ULL;
                    const weight_t wu = uvw ? 1 : s_weights(flip_u);

                    if (new_part_u) wr += wu;
                    else wr -= wu;

                    for (u32 e = s_neigh(flip_u); e < s_neigh(flip_u + 1); ++e) {
                        if (e >= gm) Kokkos::abort("e out of bounds");
                        const vertex_t v = s_edges_v(e);
                        if (v >= gn) Kokkos::abort("v out of bounds");
                        const u64 part_v = (gray >> v) & 1ULL;
                        const bool was_cut = old_part_u != part_v;
                        const bool now_cut = new_part_u != part_v;
                        const weight_t ew = uew ? 1 : s_edges_w(e);
                        if (was_cut && !now_cut) cut -= ew;
                        else if (!was_cut && now_cut) cut += ew;
                    }
                    evaluate_current(gray, wr, cut, local_best);
                }
            }, reducer);

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                team_results(global_rank) = best_in_team;
            });
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "reduce_and_apply");
        auto d_results = batch.bisection_results;
        Kokkos::parallel_for("reduce_team_results", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t id) {
            if (!active_mask(id) || d_actual_n(id) <= 1) return;
            BestBisectConfig best;
            best.penalty = 0xFFFFFFFFFFFFFFFFULL;
            best.cut = 0x7FFFFFFF;
            best.config = 0;
            const u32 start = teams_offset(id);
            const u32 end = teams_offset(id + 1);
            for (u32 i = start; i < end; ++i) {
                if (team_results(i).penalty < best.penalty) {
                    best = team_results(i);
                } else if (team_results(i).penalty == best.penalty) {
                    if (team_results(i).cut < best.cut) {
                        best = team_results(i);
                    }
                }
            }
            d_results(id) = best;
        });

        Kokkos::parallel_for("apply_batched_best_config", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t id) {
            if (active_mask(id) && d_actual_n(id) > 1) {
                const u64 config = d_results(id).config;
                partition_t *part = batch.get_partition_ptr(id);
                const vertex_t gn = d_actual_n(id);
                for (vertex_t u = 0; u < gn; ++u) {
                    part[u] = (partition_t) ((config >> u) & 1ULL);
                }
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        if constexpr (DO_HEURISTIC) {
            pop_back(mem_stack); // heuristic_results
        }
        pop_back(mem_stack); // team_results
        pop_back(mem_stack); // max_sizes
        pop_back(mem_stack); // teams_offset
        pop_back(mem_stack); // teams_per_graph
    }

} // namespace GPU_HeiPa

#endif // GPU_HEIPA_GPU_BISECTION_BRUTE_FORCE_H
