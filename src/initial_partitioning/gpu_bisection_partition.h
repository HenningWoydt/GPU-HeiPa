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

#ifndef GPU_HEIPA_GPU_BISECTION_PARTITION_H
#define GPU_HEIPA_GPU_BISECTION_PARTITION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/two_hop_matching.h"
#include "../coarsening/independent_edge_set.h"
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/custom_reductions.h"

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
        using result_view_type = Kokkos::View<value_type, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

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
                                         weight_t lmax_left,
                                         weight_t lmax_right,
                                         UnmanagedDevicePartition &partition_map,
                                         Kokkos::View<BestBisectConfig, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> > result_view,
                                         DeviceExecutionSpace &exec_space) {
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
                partition_map(0) = 0;
            });
            return;
        }

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

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, num_teams, team_size)
                .set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_reduce("brute_force_bisect_reduction", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team, BestBisectConfig &team_best) {
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

                for (u64 i = begin + 1; i < end; i++) {
                    const u64 next_gray = i ^ (i >> 1);
                    const u64 diff = gray ^ next_gray;

                    #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                    const vertex_t flip_u = (vertex_t) __ffsll(diff) - 1;
                    #else
                    const vertex_t flip_u = (vertex_t) __builtin_ctzll(diff);
                    #endif

                    const u64 old_part_u = (gray >> flip_u) & 1ULL;
                    const u64 new_part_u = old_part_u ^ 1ULL;
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
                    gray = next_gray;
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
        size_t max_blocks = 0;

        UnmanagedDeviceU8 graph_memory;
        UnmanagedDeviceU8 partition_memory;
        UnmanagedDeviceU8 global_ids_memory;

        HostPinnedVertex actual_n;
        HostPinnedVertex actual_m;
        HostPinnedWeight actual_g_weight;

        DeviceVertex d_actual_n;
        DeviceVertex d_actual_m;
        DeviceWeight d_actual_g_weight;

        HostPinnedU32 m_scan_results;
        HostPinnedWeight g_weight_results;

        Kokkos::View<BestBisectConfig *, HostMemory> h_bisection_results;

        HostVertex h_bsizes;
        HostVertex h_projected_bsizes;
        HostU8 h_active_mask;
        HostWeight h_lmax_l;
        HostWeight h_lmax_r;
        HostPartition h_left_strides;
        HostPartition h_right_strides;
    };

    inline void init_GraphBatch(GraphBatch &batch,
                                Graph &g,
                                partition_t k,
                                KokkosMemoryStack &mem_stack) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "init_GraphBatch");
        batch.n = g.n;
        batch.m = g.m;
        batch.k = k;
        batch.max_blocks = 2 * k;

        u64 n_bytes_weights = round_up_64(g.n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(g.n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(g.m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(g.m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(g.m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;
        u64 n_bytes_graph_total = (u64) batch.max_blocks * n_bytes_one_graph;
        batch.graph_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_graph_total), n_bytes_graph_total);

        u64 n_bytes_partition = round_up_64(g.n) * sizeof(partition_t);
        u64 n_bytes_partition_total = (u64) batch.max_blocks * n_bytes_partition;
        batch.partition_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_partition_total), n_bytes_partition_total);

        u64 n_bytes_global_ids = round_up_64(g.n) * sizeof(vertex_t);
        u64 n_bytes_global_ids_total = (u64) batch.max_blocks * n_bytes_global_ids;
        batch.global_ids_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_global_ids_total), n_bytes_global_ids_total);

        batch.actual_n = HostPinnedVertex("actual_n", batch.max_blocks);
        batch.actual_m = HostPinnedVertex("actual_m", batch.max_blocks);
        batch.actual_g_weight = HostPinnedWeight("actual_g_weight", batch.max_blocks);

        batch.d_actual_n = DeviceVertex("d_actual_n", batch.max_blocks);
        batch.d_actual_m = DeviceVertex("d_actual_m", batch.max_blocks);
        batch.d_actual_g_weight = DeviceWeight("d_actual_g_weight", batch.max_blocks);

        batch.m_scan_results = HostPinnedU32("m_scan_results", batch.max_blocks);
        batch.g_weight_results = HostPinnedWeight("g_weight_results", batch.max_blocks);
        batch.h_bisection_results = Kokkos::View<BestBisectConfig *, HostMemory>("h_bisection_results", batch.max_blocks);

        batch.h_bsizes = HostVertex("h_bsizes", batch.max_blocks);
        batch.h_projected_bsizes = HostVertex("h_projected_bsizes", batch.max_blocks);
        batch.h_active_mask = HostU8("h_active_mask", batch.max_blocks);
        batch.h_lmax_l = HostWeight("h_lmax_l", batch.max_blocks);
        batch.h_lmax_r = HostWeight("h_lmax_r", batch.max_blocks);
        batch.h_left_strides = HostPartition("h_left_strides", batch.max_blocks);
        batch.h_right_strides = HostPartition("h_right_strides", batch.max_blocks);
    }

    inline void free_GraphBatch(GraphBatch &batch,
                                KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
    }

    inline Graph get_Graph(const GraphBatch &batch,
                           partition_t id) {
        Graph graph;
        graph.n = batch.actual_n(id);
        graph.m = batch.actual_m(id);
        graph.g_weight = batch.actual_g_weight(id);
        graph.uniform_edge_weights = false;
        graph.uniform_vertex_weights = false;
        graph.n_pops = 0;

        u64 n_bytes_weights = round_up_64(batch.n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(batch.n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(batch.m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(batch.m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(batch.m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;

        u64 memory_offset = (u64) id * n_bytes_one_graph;
        u8 *base = batch.graph_memory.data() + memory_offset;

        vertex_t n_view = graph.n > 0 ? graph.n : batch.n;
        vertex_t m_view = graph.m > 0 ? graph.m : batch.m;

        graph.weights = UnmanagedDeviceWeight((weight_t *) base, n_view);
        base += n_bytes_weights;
        graph.neighborhood = UnmanagedDeviceU32((u32 *) base, n_view + 1);
        base += n_bytes_neighborhood;
        graph.edges_u = UnmanagedDeviceVertex((vertex_t *) base, m_view);
        base += n_bytes_edges_u;
        graph.edges_v = UnmanagedDeviceVertex((vertex_t *) base, m_view);
        base += n_bytes_edges_v;
        graph.edges_w = UnmanagedDeviceWeight((weight_t *) base, m_view);

        return graph;
    }

    inline UnmanagedDeviceVertex get_global_ids(const GraphBatch &batch,
                                                partition_t id) {
        u64 n_bytes_global_ids = round_up_64(batch.n) * sizeof(vertex_t);
        u64 memory_offset = (u64) id * n_bytes_global_ids;
        u8 *base = batch.global_ids_memory.data() + memory_offset;
        vertex_t n_view = batch.actual_n(id) > 0 ? batch.actual_n(id) : batch.n;
        return UnmanagedDeviceVertex((vertex_t *) base, n_view);
    }

    inline UnmanagedDevicePartition get_partition(const GraphBatch &batch,
                                                  partition_t id) {
        u64 n_bytes_partition = round_up_64(batch.n) * sizeof(partition_t);

        u64 memory_offset = (u64) id * n_bytes_partition;
        u8 *base = batch.partition_memory.data() + memory_offset;
        vertex_t n_view = batch.actual_n(id) > 0 ? batch.actual_n(id) : batch.n;

        return UnmanagedDevicePartition((partition_t *) base, n_view);
    }

    inline void extract_subgraph(const Graph &g,
                                 Graph &sub_g,
                                 UnmanagedDeviceVertex &global_ids,
                                 partition_t id,
                                 Partition &partition,
                                 KokkosMemoryStack &mem_stack,
                                 DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "extract_subgraph");
        auto map = partition.map;
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
        vertex_t sub_n = 0;
        Kokkos::parallel_scan("local_id_assignment", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &running, bool final) {
            bool in_block = (map(u) == id);
            if (final && in_block) {
                local_ids(u) = running;
                global_ids(running) = u;
                if (!sub_g.uniform_vertex_weights) {
                    sub_g.weights(running) = g.uniform_vertex_weights ? 1 : g.weights(u);
                }
            }
            running += in_block;
        }, sub_n);
        exec_space.fence();
        sub_g.n = sub_n;
        UnmanagedDeviceVertex degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * sub_n), sub_n);
        Kokkos::parallel_for("count_local_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local) {
            vertex_t u_global = global_ids(u_local);
            u32 count = 0;
            for (u32 e = g.neighborhood(u_global); e < g.neighborhood(u_global + 1); ++e) {
                if (map(g.edges_v(e)) == id) count++;
            }
            degree(u_local) = count;
        });
        u32 sub_m = 0;
        Kokkos::parallel_scan("prefix_sum_neighborhood", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local, u32 &running, bool final) {
            u32 deg = degree(u_local);
            if (final) sub_g.neighborhood(u_local) = running;
            running += deg;
        }, sub_m);
        Kokkos::parallel_for("set_last_neighborhood", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
            sub_g.neighborhood(sub_n) = sub_m;
        });
        exec_space.fence();
        sub_g.m = sub_m;
        auto g_neigh = g.neighborhood;
        auto g_ev = g.edges_v;
        auto g_ew = g.edges_w;
        bool g_uew = g.uniform_edge_weights;
        Kokkos::parallel_for("populate_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local) {
            vertex_t u_global = global_ids(u_local);
            u32 edge_idx = sub_g.neighborhood(u_local);
            for (u32 e = g_neigh(u_global); e < g_neigh(u_global + 1); ++e) {
                vertex_t v_global = g_ev(e);
                if (map(v_global) == id) {
                    sub_g.edges_u(edge_idx) = u_local;
                    sub_g.edges_v(edge_idx) = local_ids(v_global);
                    if (!sub_g.uniform_edge_weights) {
                        sub_g.edges_w(edge_idx) = g_uew ? 1 : g_ew(e);
                    }
                    edge_idx++;
                }
            }
        });
        exec_space.fence();
        weight_t sub_weight = 0;
        Kokkos::parallel_reduce("sum_subgraph_weight", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local, weight_t &local_sum) {
            local_sum += sub_g.uniform_vertex_weights ? 1 : sub_g.weights(u_local);
        }, sub_weight);
        sub_g.g_weight = sub_weight;
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline void extract_all_subgraphs(const Graph &g,
                                      GraphBatch &batch,
                                      Partition &partition,
                                      const DeviceU8 &active_mask,
                                      UnmanagedDeviceVertex &local_ids,
                                      UnmanagedDeviceVertex &local_degree,
                                      DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "extract_all_subgraphs");
        auto map = partition.map;
        partition_t k = (partition_t) batch.max_blocks;
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

        // Single deep_copy of counts to host
        Kokkos::deep_copy(exec_space, batch.actual_n, d_actual_n);
        exec_space.fence();

        for (partition_t id = 0; id < k; ++id) {
            vertex_t sub_n_host = batch.actual_n(id);
            if (sub_n_host == 0) continue;
            u32 *sub_g_neighborhood = (u32 *) (graph_memory.data() + (u64) id * n_bytes_one_graph + n_bytes_weights);
            vertex_t *g_ids_ptr = (vertex_t *) (global_ids_memory.data() + (u64) id * n_bytes_global_ids);
            auto m_sum_view = Kokkos::subview(batch.m_scan_results, id);
            Kokkos::parallel_scan("block_neighborhood_scan", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n_host), KOKKOS_LAMBDA(const vertex_t local_u, u32 &running, bool final) {
                u32 deg = local_degree(g_ids_ptr[local_u]);
                if (final) sub_g_neighborhood[local_u] = running;
                running += deg;
            }, m_sum_view);
            Kokkos::parallel_for("set_last_neighborhood", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
                sub_g_neighborhood[sub_n_host] = m_sum_view();
                d_actual_m(id) = m_sum_view();
            });
        }
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

        for (partition_t id = 0; id < k; ++id) {
            vertex_t sub_n_host = batch.actual_n(id);
            if (sub_n_host == 0) continue;
            weight_t *sub_g_weights = (weight_t *) (graph_memory.data() + (u64) id * n_bytes_one_graph);
            auto sub_weight_view = Kokkos::subview(batch.g_weight_results, id);
            Kokkos::parallel_reduce("sum_subgraph_weight", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n_host), KOKKOS_LAMBDA(const vertex_t local_u, weight_t &local_sum) {
                local_sum += sub_g_weights[local_u];
            }, sub_weight_view);
            Kokkos::parallel_for("update_actual_g_weight", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
                d_actual_g_weight(id) = sub_weight_view();
            });
        }
        exec_space.fence();
        Kokkos::deep_copy(exec_space, batch.actual_m, d_actual_m);
        Kokkos::deep_copy(exec_space, batch.actual_g_weight, d_actual_g_weight);
        exec_space.fence();
    }

    inline void batched_bisect(const GraphBatch &batch,
                               const DeviceU8 &active_mask,
                               const DeviceWeight &lmax_l,
                               const DeviceWeight &lmax_r,
                               DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "batched_bisect");
        std::vector<int> active_ids;
        HostU8 h_active_mask(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_active_mask"), batch.max_blocks);
        Kokkos::deep_copy(exec_space, h_active_mask, active_mask);
        exec_space.fence();
        for (partition_t i = 0; i < (partition_t) batch.max_blocks; ++i) {
            if (h_active_mask(i) && batch.actual_n(i) > 0) active_ids.push_back((int) i);
        }
        if (active_ids.empty()) return;
        u32 n_instances = (u32) active_ids.size();
        std::vector<DeviceExecutionSpace> instances = Kokkos::Experimental::partition_space(exec_space, std::vector<int>(n_instances, 1));
        auto h_results = batch.h_bisection_results;
        auto h_lmax_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lmax_l);
        auto h_lmax_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lmax_r);
        for (u32 s = 0; s < n_instances; ++s) {
            partition_t id = (partition_t) active_ids[s];
            Graph sub_g = get_Graph(batch, id);
            UnmanagedDevicePartition sub_part = get_partition(batch, id);
            weight_t l_l = h_lmax_l(id);
            weight_t l_r = h_lmax_r(id);
            auto result_view_managed = Kokkos::subview(h_results, (u32) id);
            BestBisectReducer::result_view_type result_view(result_view_managed.data());
            if (sub_g.n == 1) {
                Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(instances[s], 0, 1), KOKKOS_LAMBDA(int) { sub_part(0) = 0; });
            } else {
                if (sub_g.uniform_vertex_weights && sub_g.uniform_edge_weights) brute_force_bisect_async<true, true, 64>(sub_g, l_l, l_r, sub_part, result_view, instances[s]);
                else if (sub_g.uniform_vertex_weights) brute_force_bisect_async<true, false, 64>(sub_g, l_l, l_r, sub_part, result_view, instances[s]);
                else if (sub_g.uniform_edge_weights) brute_force_bisect_async<false, true, 64>(sub_g, l_l, l_r, sub_part, result_view, instances[s]);
                else brute_force_bisect_async<false, false, 64>(sub_g, l_l, l_r, sub_part, result_view, instances[s]);
            }
        }
        for (auto &inst: instances) inst.fence();
    }

    inline void bisect(Graph &g, weight_t lmax_1, weight_t lmax_2, UnmanagedDevicePartition &partition, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "bisect");
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) { partition(0) = 0; });
            exec_space.fence();
            return;
        }
        Kokkos::View<BestBisectConfig, DeviceMemorySpace> device_result("result_view");
        auto result_view = Kokkos::View<BestBisectConfig, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(device_result.data());
        if (g.uniform_vertex_weights && g.uniform_edge_weights) brute_force_bisect_async<true, true, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        else if (g.uniform_vertex_weights) brute_force_bisect_async<true, false, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        else if (g.uniform_edge_weights) brute_force_bisect_async<false, true, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        else brute_force_bisect_async<false, false, 64>(g, lmax_1, lmax_2, partition, result_view, exec_space);
        exec_space.fence();
    }

    struct HierarchyManager {
        std::vector<partition_t> hierarchy;
        partition_t total_k;
        std::vector<partition_t> unit_sizes;
        std::vector<u8> active;
        std::vector<u32> curr_level;
        std::vector<u32> curr_load;
        size_t max_blocks;
    };

    inline void init_HierarchyManager(HierarchyManager &manager, const std::vector<partition_t> &t_hierarchy, size_t t_k) {
        manager.hierarchy = t_hierarchy;
        manager.max_blocks = t_k;
        size_t num_levels = manager.hierarchy.size();
        manager.unit_sizes.assign(num_levels, 1);
        size_t current = 1;
        for (size_t i = 0; i < num_levels; ++i) {
            manager.unit_sizes[i] = (partition_t) current;
            current *= manager.hierarchy[i];
        }
        manager.total_k = (partition_t) current;
        manager.active.assign(manager.max_blocks, 0);
        manager.curr_level.assign(manager.max_blocks, 0);
        manager.curr_load.assign(manager.max_blocks, 0);
        manager.active[0] = 1;
        manager.curr_level[0] = (u32) num_levels - 1;
        manager.curr_load[0] = manager.hierarchy.back();
    }

    inline void split_into(const HierarchyManager &manager, partition_t id, partition_t &left_k, partition_t &right_k) {
        u32 level = manager.curr_level[id];
        u32 load = manager.curr_load[id];
        partition_t p = 1;
        while (p * 2 < load) p *= 2;
        partition_t left_load = p;
        partition_t right_load = load - p;
        left_k = left_load * manager.unit_sizes[level];
        right_k = right_load * manager.unit_sizes[level];
    }

    inline void split(HierarchyManager &manager, partition_t id, partition_t left_k, partition_t right_k) {
        u32 level = manager.curr_level[id];
        partition_t left_id = id;
        partition_t right_id = id + left_k;
        if (right_id >= manager.max_blocks) throw std::runtime_error("max_blocks exceeded");
        manager.active[left_id] = 1;
        manager.curr_level[left_id] = level;
        manager.curr_load[left_id] = left_k / manager.unit_sizes[level];
        manager.active[right_id] = 1;
        manager.curr_level[right_id] = level;
        manager.curr_load[right_id] = right_k / manager.unit_sizes[level];
    }

    inline bool descend(HierarchyManager &manager, partition_t id) {
        if (manager.curr_level[id] > 0) {
            manager.curr_level[id]--;
            manager.curr_load[id] = manager.hierarchy[manager.curr_level[id]];
            return true;
        }
        return false;
    }

    inline void recalculate_block_weights(const Graph &g, const UnmanagedDevicePartition &map, UnmanagedDeviceWeight &bweights, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "recalculate_block_weights");
        Kokkos::deep_copy(exec_space, bweights, 0);
        Kokkos::parallel_for("recalculate_block_weights", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            Kokkos::atomic_add(&bweights(id), g.uniform_vertex_weights ? 1 : g.weights(u));
        });
        exec_space.fence();
    }

    inline void calculate_block_sizes(const Graph &g, const Mapping *mapping, const UnmanagedDevicePartition &map, UnmanagedDeviceVertex &bsizes, UnmanagedDeviceVertex &projected_sizes, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "calculate_block_sizes");
        Kokkos::deep_copy(exec_space, bsizes, 0);
        Kokkos::parallel_for("calculate_block_sizes", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            Kokkos::atomic_add(&bsizes(id), 1);
        });
        if (mapping != nullptr) {
            Kokkos::deep_copy(exec_space, projected_sizes, 0);
            Kokkos::parallel_for("calculate_projected_block_sizes", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, mapping->old_n), KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t new_v = mapping->mapping(u);
                partition_t id = map(new_v);
                Kokkos::atomic_add(&projected_sizes(id), 1);
            });
        }
        exec_space.fence();
    }

    inline void gpu_bisect_partition(Graph &g, const std::vector<partition_t> &hierarchy, partition_t k, f64 imbalance, u64 seed, u32 threshold, Partition &partition, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "gpu_bisect_partition");
        GraphBatch batch;
        init_GraphBatch(batch, g, k, mem_stack);
        HierarchyManager manager;
        init_HierarchyManager(manager, hierarchy, batch.max_blocks);
        std::vector<Graph> graphs = {g};
        std::vector<Mapping> mappings;
        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);
        while (graphs.back().n > threshold) {
            assert_state_pre_partition(graphs.back(), exec_space);
            mappings.push_back(two_hop_matcher_get_mapping<false, false>(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            graphs.push_back(from_Graph_Mapping<false, false>(graphs.back(), mappings.back(), mem_stack, exec_space));
            contract(partition, mappings.back(), exec_space);
            assert_coarsening(graphs[graphs.size() - 2], graphs.back(), mappings.back(), exec_space);
            assert_state_pre_partition(graphs.back(), exec_space);
        }
        {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "initial_partitioning_phase");
            partition_t l_k, r_k;
            split_into(manager, 0, l_k, r_k);
            UnmanagedDevicePartition temp_partition = get_partition(batch, 0);
            bisect(graphs.back(), l_k * lmax_global, r_k * lmax_global, temp_partition, exec_space);
            partition_t left_id = 0;
            partition_t right_id = l_k;
            split(manager, 0, l_k, r_k);
            auto map = partition.map;
            Kokkos::parallel_for("update_partition_initial", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graphs.back().n), KOKKOS_LAMBDA(const vertex_t u) {
                map(u) = (temp_partition(u) == 0) ? left_id : right_id;
            });
            exec_space.fence();
            recalculate_block_weights(graphs.back(), map, partition.bweights, exec_space);
        }
        DeviceU8 active_mask("active_mask", batch.max_blocks);
        DeviceWeight lmax_l("lmax_l", batch.max_blocks);
        DeviceWeight lmax_r("lmax_r", batch.max_blocks);
        DevicePartition left_strides("left_strides", batch.max_blocks);
        DevicePartition right_strides("right_strides", batch.max_blocks);
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
        UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
        UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.max_blocks), batch.max_blocks);
        UnmanagedDeviceVertex projected_bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.max_blocks), batch.max_blocks);
        std::vector<u8> iteration_active(batch.max_blocks);
        while (true) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "uncontraction_extraction_loop");
            bool do_extract = true;
            while (do_extract) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "extraction_inner_loop");
                do_extract = false;
                const Mapping *mapping_ptr = mappings.empty() ? nullptr : &mappings.back();
                calculate_block_sizes(graphs.back(), mapping_ptr, partition.map, bsizes, projected_bsizes, exec_space);
                Kokkos::deep_copy(exec_space, batch.h_bsizes, bsizes);
                if (!mappings.empty()) Kokkos::deep_copy(exec_space, batch.h_projected_bsizes, projected_bsizes);
                exec_space.fence();
                Kokkos::deep_copy(exec_space, batch.h_active_mask, (u8) 0);
                bool any_active = false;
                std::copy(manager.active.begin(), manager.active.end(), iteration_active.begin());
                for (partition_t id = 0; id < (partition_t) batch.max_blocks; id++) {
                    if (iteration_active[id]) {
                        if (manager.curr_load[id] > 1) {
                            vertex_t projected_n = mappings.empty() ? batch.h_bsizes(id) : batch.h_projected_bsizes(id);
                            if (mappings.empty() || projected_n > threshold) {
                                partition_t l_k, r_k;
                                split_into(manager, id, l_k, r_k);
                                batch.h_active_mask(id) = 1;
                                batch.h_lmax_l(id) = l_k * lmax_global;
                                batch.h_lmax_r(id) = r_k * lmax_global;
                                batch.h_left_strides(id) = id;
                                batch.h_right_strides(id) = id + l_k;
                                split(manager, id, l_k, r_k);
                                any_active = true;
                                do_extract = true;
                            }
                        } else if (manager.curr_level[id] > 0) {
                            descend(manager, id);
                            do_extract = true;
                        }
                    }
                }
                if (any_active) {
                    Kokkos::deep_copy(exec_space, active_mask, batch.h_active_mask);
                    Kokkos::deep_copy(exec_space, lmax_l, batch.h_lmax_l);
                    Kokkos::deep_copy(exec_space, lmax_r, batch.h_lmax_r);
                    Kokkos::deep_copy(exec_space, left_strides, batch.h_left_strides);
                    Kokkos::deep_copy(exec_space, right_strides, batch.h_right_strides);
                    exec_space.fence();
                    extract_all_subgraphs(graphs.back(), batch, partition, active_mask, local_ids, local_degree, exec_space);
                    batched_bisect(batch, active_mask, lmax_l, lmax_r, exec_space);
                    auto map = partition.map;
                    auto g_n = graphs.back().n;
                    u64 n_bytes_partition = round_up_64(batch.n) * sizeof(partition_t);
                    u8 *partition_base = batch.partition_memory.data();
                    Kokkos::parallel_for("batched_map_update", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g_n), KOKKOS_LAMBDA(const vertex_t u) {
                        partition_t id = map(u);
                        if (active_mask(id)) {
                            vertex_t l_u = local_ids(u);
                            partition_t *sub_part = (partition_t *) (partition_base + (u64) id * n_bytes_partition);
                            partition_t b_res = sub_part[l_u];
                            map(u) = (b_res == 0) ? left_strides(id) : right_strides(id);
                        }
                    });
                    exec_space.fence();
                    recalculate_block_weights(graphs.back(), map, partition.bweights, exec_space);
                }
                assert_state_after_partition(graphs.back(), partition, k, exec_space);
            }
            if (mappings.empty()) break;
            uncontract(partition, mappings.back(), exec_space);
            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();
            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();
            recalculate_block_weights(graphs.back(), partition.map, partition.bweights, exec_space);
        }
        recalculate_block_weights(graphs.back(), partition.map, partition.bweights, exec_space);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        free_GraphBatch(batch, mem_stack);
    }
}

#endif //GPU_HEIPA_GPU_BISECTION_PARTITION_H
