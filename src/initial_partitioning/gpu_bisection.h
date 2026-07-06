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

#ifndef GPU_HEIPA_GPU_BISECTION_H
#define GPU_HEIPA_GPU_BISECTION_H

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"

namespace GPU_HeiPa {
    struct BestBisectConfig {
        u64 penalty = 0xFFFFFFFFFFFFFFFFULL;
        weight_t cut = 0x7FFFFFFF;
        u64 config = 0;

        KOKKOS_INLINE_FUNCTION BestBisectConfig() = default;
    };

    struct BestBisectReducer {
        using reducer = BestBisectReducer;
        using value_type = BestBisectConfig;
        using result_view_type = Kokkos::View<value_type, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

        KOKKOS_INLINE_FUNCTION void join(value_type &dst, const value_type &src) const {
            if (src.penalty < dst.penalty) {
                dst = src;
            } else if (src.penalty == dst.penalty) {
                if (src.cut < dst.cut) {
                    dst = src;
                }
            }
        }

        KOKKOS_INLINE_FUNCTION void init(value_type &dst) const {
            dst.penalty = 0xFFFFFFFFFFFFFFFFULL;
            dst.cut = 0x7FFFFFFF;
            dst.config = 0;
        }

        value_type *value;

        KOKKOS_INLINE_FUNCTION BestBisectReducer(value_type &val) : value(&val) {
        }

        KOKKOS_INLINE_FUNCTION BestBisectReducer(result_view_type view) : value(view.data()) {
        }

        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }

        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }

        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    template<bool uvw, bool uew, int CHUNK>
    inline void brute_force_bisect_async(const Graph &g,
                                         UnmanagedDeviceWeight left_lmax,
                                         UnmanagedDeviceWeight right_lmax,
                                         partition_t id,
                                         UnmanagedDevicePartition &partition_map,
                                         Kokkos::View<BestBisectConfig, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > result_view,
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
        const u64 num_configs = 1ULL << last;

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

                    if (wl == 0 || wr_cur == 0) {
                        penalty += 1000000000000ULL;
                    }

                    if (penalty < best_cur.penalty || (penalty == best_cur.penalty && cut_cur < best_cur.cut)) {
                        best_cur.penalty = penalty;
                        best_cur.cut = cut_cur;
                        best_cur.config = config;
                    }
                };

                evaluate_current(gray, wr, cut, local_best);

                #pragma unroll
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

    struct GraphBatch {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;

        UnmanagedDeviceU8 graph_memory;
        UnmanagedDeviceU8 partition_memory;
        UnmanagedDeviceU8 global_ids_memory;


        UnmanagedDeviceVertex d_actual_n;
        UnmanagedDeviceVertex d_actual_m;
        UnmanagedDeviceWeight d_actual_g_weight;


        Kokkos::View<BestBisectConfig *, DeviceMemorySpace> d_bisection_results;
        KOKKOS_INLINE_FUNCTION
        partition_t *get_partition_ptr(partition_t id) const {
            u64 n_bytes_partition = round_up_64(n) * sizeof(partition_t);
            u64 memory_offset = (u64) id * n_bytes_partition;
            return (partition_t *) (partition_memory.data() + memory_offset);
        }

        KOKKOS_INLINE_FUNCTION
        vertex_t *get_global_ids_ptr(partition_t id) const {
            u64 n_bytes_global_ids = round_up_64(n) * sizeof(vertex_t);
            u64 memory_offset = (u64) id * n_bytes_global_ids;
            return (vertex_t *) (global_ids_memory.data() + memory_offset);
        }
    };

    inline void init_GraphBatch(GraphBatch &batch,
                                Graph &g,
                                partition_t k,
                                KokkosMemoryStack &mem_stack) {
        batch.n = g.n;
        batch.m = g.m;
        batch.k = k;
        

        u64 n_bytes_weights = round_up_64(g.n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(g.n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(g.m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(g.m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(g.m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;
        u64 n_bytes_graph_total = (u64) batch.k * n_bytes_one_graph;
        batch.graph_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_graph_total), n_bytes_graph_total);

        u64 n_bytes_partition = round_up_64(g.n) * sizeof(partition_t);
        u64 n_bytes_partition_total = (u64) batch.k * n_bytes_partition;
        batch.partition_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_partition_total), n_bytes_partition_total);

        u64 n_bytes_global_ids = round_up_64(g.n) * sizeof(vertex_t);
        u64 n_bytes_global_ids_total = (u64) batch.k * n_bytes_global_ids;
        batch.global_ids_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_global_ids_total), n_bytes_global_ids_total);


        batch.d_actual_n = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        batch.d_actual_m = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        batch.d_actual_g_weight = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * batch.k), batch.k);

        batch.d_bisection_results = Kokkos::View<BestBisectConfig *, DeviceMemorySpace>("d_bisection_results", batch.k);

    }

    inline void free_GraphBatch(GraphBatch &batch,
                                KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
    }





    inline void extract_all_subgraphs(const Graph &g,
                                      GraphBatch &batch,
                                      Partition &partition,
                                      const DeviceU8 &active_mask,
                                      UnmanagedDeviceVertex &local_ids,
                                      UnmanagedDeviceVertex &local_degree,
                                      DeviceExecutionSpace &exec_space) {
        auto map = partition.map;
        partition_t k = (partition_t) batch.k;
        bool use_mask = active_mask.extent(0) > 0;
        auto d_actual_n = batch.d_actual_n;
        auto d_actual_m = batch.d_actual_m;
        auto d_actual_g_weight = batch.d_actual_g_weight;
        Kokkos::deep_copy(exec_space, d_actual_n, 0);
        Kokkos::deep_copy(exec_space, d_actual_m, 0);
        Kokkos::deep_copy(exec_space, d_actual_g_weight, 0);
        auto g_weights = g.weights;
        bool g_uvw = g.uniform_vertex_weights;
        auto graph_memory = batch.graph_memory;
        auto global_ids_memory = batch.global_ids_memory;
        vertex_t b_n = batch.n;
        vertex_t b_m = batch.m;
        u64 n_bytes_weights = round_up_64(b_n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(b_n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(b_m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;
        u64 n_bytes_global_ids = round_up_64(b_n) * sizeof(vertex_t);

        Kokkos::parallel_for("batched_vertex_assignment", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            if (use_mask && !active_mask(id)) return;
            vertex_t local_u = Kokkos::atomic_fetch_add(&d_actual_n(id), 1);
            local_ids(u) = local_u;
            vertex_t *g_ids_ptr = (vertex_t *) (global_ids_memory.data() + (u64) id * n_bytes_global_ids);
            g_ids_ptr[local_u] = u;
            weight_t *weights_ptr = (weight_t *) (graph_memory.data() + (u64) id * n_bytes_one_graph);
            weights_ptr[local_u] = g_uvw ? 1 : g_weights(u);
        });
        auto g_neighborhood = g.neighborhood;
        auto g_edges_v = g.edges_v;
        Kokkos::parallel_for("batched_edge_counting", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            if (use_mask && !active_mask(id)) {
                local_degree(u) = 0;
                return;
            }
            u32 count = 0;
            for (u32 e = g_neighborhood(u); e < g_neighborhood(u + 1); ++e) {
                if (map(g_edges_v(e)) == id) count++;
            }
            local_degree(u) = count;
        });

        // 1. Batched block neighborhood scan
        typedef Kokkos::TeamPolicy<DeviceExecutionSpace> TeamPolicy;
        typedef TeamPolicy::member_type TeamMember;
        
        Kokkos::parallel_for("batched_block_neighborhood_scan", TeamPolicy(exec_space, k, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
            partition_t id = team.league_rank();
            vertex_t sub_n = d_actual_n(id);
            if (sub_n == 0) return;
            
            u32 *sub_g_neighborhood = (u32 *) (graph_memory.data() + (u64) id * n_bytes_one_graph + n_bytes_weights);
            vertex_t *g_ids_ptr = (vertex_t *) (global_ids_memory.data() + (u64) id * n_bytes_global_ids);
            
            u32 total_m = 0;
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, sub_n), [&](const vertex_t local_u, u32 &running, bool final) {
                u32 deg = local_degree(g_ids_ptr[local_u]);
                if (final) sub_g_neighborhood[local_u] = running;
                running += deg;
            }, total_m);
            
            if (team.team_rank() == 0) {
                sub_g_neighborhood[sub_n] = total_m;
                d_actual_m(id) = total_m;
            }
        });
        exec_space.fence();

        auto g_edges_w = g.edges_w;
        bool g_uew = g.uniform_edge_weights;
        Kokkos::parallel_for("batched_edge_population", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            if (use_mask && !active_mask(id)) return;
            u32 *sub_g_neighborhood = (u32 *) (graph_memory.data() + (u64) id * n_bytes_one_graph + n_bytes_weights);
            vertex_t *sub_g_edges_u = (vertex_t *) ((u8 *) sub_g_neighborhood + n_bytes_neighborhood);
            vertex_t *sub_g_edges_v = (vertex_t *) ((u8 *) sub_g_edges_u + n_bytes_edges_u);
            weight_t *sub_g_edges_w = (weight_t *) ((u8 *) sub_g_edges_v + n_bytes_edges_v);
            u32 edge_idx = sub_g_neighborhood[local_ids(u)];
            for (u32 e = g_neighborhood(u); e < g_neighborhood(u + 1); ++e) {
                vertex_t v = g_edges_v(e);
                if (map(v) == id) {
                    sub_g_edges_u[edge_idx] = local_ids(u);
                    sub_g_edges_v[edge_idx] = local_ids(v);
                    if (!g_uew) sub_g_edges_w[edge_idx] = g_edges_w(e);
                    else sub_g_edges_w[edge_idx] = 1;
                    edge_idx++;
                }
            }
        });
        
        // 2. Batched subgraph weight sum
        Kokkos::parallel_for("batched_sum_subgraph_weight", TeamPolicy(exec_space, k, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
            partition_t id = team.league_rank();
            vertex_t sub_n = d_actual_n(id);
            if (sub_n == 0) return;
            
            weight_t *sub_g_weights = (weight_t *) (graph_memory.data() + (u64) id * n_bytes_one_graph);
            
            weight_t local_sum = 0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, sub_n), [&](const vertex_t local_u, weight_t &lsum) {
                lsum += sub_g_weights[local_u];
            }, local_sum);
            
            if (team.team_rank() == 0) {
                d_actual_g_weight(id) = local_sum;
            }
        });
        exec_space.fence();

        exec_space.fence();
    }

    template<bool uvw, bool uew>
    inline void batched_brute_force_bisect(const GraphBatch &batch,
                                           const DeviceU8 &active_mask,
                                           const DeviceWeight &left_lmax,
                                           const DeviceWeight &right_lmax,
                                           KokkosMemoryStack &mem_stack,
                                           DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "batched_brute_force_bisect");
        
        // --- Hyperparameters ---
        // Number of threads per Kokkos team on the GPU. Affects parallelism and register/shared memory usage.
        const int TEAM_SIZE = 256;
        // High penalty added to configurations that result in an empty partition block (0 weight on one side).
        const u64 EMPTY_BLOCK_PENALTY = 1000000000000ULL;
        // The number of partition configurations (Gray code steps) each thread evaluates sequentially.
        const int CHUNK = 128 * 4;
        // -----------------------
        const u32 k = batch.k;

        UnmanagedDeviceU32 teams_per_graph((u32*) get_chunk_back(mem_stack, sizeof(u32) * k), k);
        UnmanagedDeviceU32 teams_offset((u32*) get_chunk_back(mem_stack, sizeof(u32) * (k + 1)), k + 1);
        UnmanagedDeviceU32 max_sizes((u32*) get_chunk_back(mem_stack, sizeof(u32) * 2), 2);
        Kokkos::deep_copy(exec_space, max_sizes, 0);

        auto d_actual_n = batch.d_actual_n;
        auto d_actual_m = batch.d_actual_m;

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
            const u64 num_configs = 1ULL << (gn - 1);
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
                batch.d_bisection_results(id).penalty = 0;
                batch.d_bisection_results(id).cut = 0;
                batch.d_bisection_results(id).config = 0;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        if (total_teams == 0) {
            pop_back(mem_stack); // max_sizes
            pop_back(mem_stack); // teams_offset
            pop_back(mem_stack); // teams_per_graph
            return;
        }

        Kokkos::View<BestBisectConfig*, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> team_results((BestBisectConfig*) get_chunk_back(mem_stack, sizeof(BestBisectConfig) * total_teams), total_teams);

        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "bisect_kernel");
        typedef Kokkos::View<u32 *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchU32;
        typedef Kokkos::View<vertex_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchVertex;
        typedef Kokkos::View<weight_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchWeight;

        size_t shmem_size = ScratchU32::shmem_size(max_n + 1);
        shmem_size += ScratchVertex::shmem_size(max_m);
        shmem_size += ScratchVertex::shmem_size(max_m);
        if (!uvw) shmem_size += ScratchWeight::shmem_size(max_n);
        if (!uew) shmem_size += ScratchWeight::shmem_size(max_m);

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, total_teams, TEAM_SIZE).set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for("init_team_results", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, total_teams), KOKKOS_LAMBDA(const u32 i) {
            team_results(i).penalty = 0xFFFFFFFFFFFFFFFFULL;
            team_results(i).cut = 0x7FFFFFFF;
            team_results(i).config = 0;
        });

        auto g_mem = batch.graph_memory;
        auto d_actual_g_weight = batch.d_actual_g_weight;
        vertex_t b_n = batch.n;
        vertex_t b_m = batch.m;
        u64 n_bytes_weights = round_up_64(b_n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(b_n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(b_m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;
        KOKKOS_PROFILE_FENCE(exec_space);

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
            const weight_t lmax_left = left_lmax(graph_id);
            const weight_t lmax_right = right_lmax(graph_id);
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
                const u64 chunk_id = (u64) local_team_rank * TEAM_SIZE + tid;
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
                    const weight_t wl = g_weight - wr_cur;
                    const u64 p_l = wl > lmax_left ? (u64) (wl - lmax_left) : 0;
                    const u64 p_r = wr_cur > lmax_right ? (u64) (wr_cur - lmax_right) : 0;
                    u64 penalty = p_l * p_l + p_r * p_r;

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

                #pragma unroll
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
        auto d_results = batch.d_bisection_results;
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
        exec_space.fence();
        KOKKOS_PROFILE_FENCE(exec_space);

        pop_back(mem_stack); // team_results
        pop_back(mem_stack); // max_sizes
        pop_back(mem_stack); // teams_offset
        pop_back(mem_stack); // teams_per_graph
    }

    inline void bisect(Graph &g, weight_t lmax_1, weight_t lmax_2, UnmanagedDevicePartition &partition, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection", "bisect");
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) { partition(0) = 0; });
            exec_space.fence();
            return;
        }
        Kokkos::View<BestBisectConfig, DeviceMemorySpace> device_result("result_view");
        BestBisectReducer::result_view_type result_view(device_result.data());
        // if (g.uniform_vertex_weights && g.uniform_edge_weights) brute_force_bisect_async<true, true, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        // else if (g.uniform_vertex_weights) brute_force_bisect_async<true, false, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        // else if (g.uniform_edge_weights) brute_force_bisect_async<false, true, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        // else brute_force_bisect_async<false, false, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        exec_space.fence();
    }

    inline void recalculate_block_weights(const Graph &g, const UnmanagedDevicePartition &map, UnmanagedDeviceWeight &bweights, DeviceExecutionSpace &exec_space) {
        Kokkos::deep_copy(exec_space, bweights, 0);
        Kokkos::parallel_for("recalculate_block_weights", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            Kokkos::atomic_add(&bweights(id), g.uniform_vertex_weights ? 1 : g.weights(u));
        });
        exec_space.fence();
    }

    inline void calculate_block_sizes(const Graph &g, const Mapping *mapping, const UnmanagedDevicePartition &map, UnmanagedDeviceVertex &sizes, DeviceExecutionSpace &exec_space) {
        bool has_mapping = (mapping != nullptr);
        vertex_t old_n = has_mapping ? mapping->old_n : 0;
        UnmanagedDeviceVertex mapping_view = has_mapping ? mapping->mapping : UnmanagedDeviceVertex();
        u32 k = sizes.extent(0);

        Kokkos::parallel_for("calculate_block_sizes_fused", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k), [&](const int i) {
                sizes(i) = 0;
            });

            team.team_barrier();

            if (!has_mapping) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                    partition_t id = map(u);
                    Kokkos::atomic_add(&sizes(id), 1);
                });
            } else {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, old_n), [&](const vertex_t u) {
                    vertex_t new_v = mapping_view(u);
                    partition_t id = map(new_v);
                    Kokkos::atomic_add(&sizes(id), 1);
                });
            }
        });
        exec_space.fence();
    }
} // namespace GPU_HeiPa

#endif //GPU_HEIPA_GPU_BISECTION_H
