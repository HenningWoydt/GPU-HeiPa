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

#ifndef GPU_HEIPA_GPU_RB_PARTITION_H
#define GPU_HEIPA_GPU_RB_PARTITION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/two_hop_matching.h"
#include "../coarsening/dumb_matching.h"
#include "../coarsening/heavy_edge_matching.h"
#include "../coarsening/independent_edge_set.h"
#include "../refinement/greedy_refinement.h"
#include "../refinement/jet_label_propagation.h"
#include "../datastructures/small_graph.h"
#include "../utility/edge_cut.h"
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "gpu_bisection.h"

namespace GPU_HeiPa {
    inline void refine_small_graph(Graph &g, Partition &partition, int small_graph_refinement, weight_t lmax_global, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        if (g.n <= 2000 && small_graph_refinement > 0) {
            if (small_graph_refinement == 1) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "greedy_refinement");
                bool uvw = g.uniform_vertex_weights;
                bool uew = g.uniform_edge_weights;

                weight_t cut_before = uew ? edge_cut<true>(g, partition, exec_space) : edge_cut<false>(g, partition, exec_space);

                if (uvw && uew) greedy_kway_refinement_small<true, true>(g, partition.map, partition.bweights, partition.k, lmax_global, 10, exec_space);
                else if (uvw) greedy_kway_refinement_small<true, false>(g, partition.map, partition.bweights, partition.k, lmax_global, 10, exec_space);
                else if (uew) greedy_kway_refinement_small<false, true>(g, partition.map, partition.bweights, partition.k, lmax_global, 10, exec_space);
                else greedy_kway_refinement_small<false, false>(g, partition.map, partition.bweights, partition.k, lmax_global, 10, exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);

                weight_t cut_after = uew ? edge_cut<true>(g, partition, exec_space) : edge_cut<false>(g, partition, exec_space);
                std::cout << "Greedy Refinement (Uncoarsen) N=" << g.n << " Cut: " << cut_before << " -> " << cut_after << std::endl;
            } else if (small_graph_refinement == 2) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "jet_label_propagation");
                bool uvw = g.uniform_vertex_weights;
                bool uew = g.uniform_edge_weights;
                weight_t cut_before = uew ? edge_cut<true>(g, partition, exec_space) : edge_cut<false>(g, partition, exec_space);
                weight_t curr_max_weight = max_weight(partition, exec_space);

                if (uvw && uew) jet_refine<true, true>(g, partition, partition.k, lmax_global, false, 0, cut_before, curr_max_weight, mem_stack, exec_space);
                else if (uvw) jet_refine<true, false>(g, partition, partition.k, lmax_global, false, 0, cut_before, curr_max_weight, mem_stack, exec_space);
                else if (uew) jet_refine<false, true>(g, partition, partition.k, lmax_global, false, 0, cut_before, curr_max_weight, mem_stack, exec_space);
                else jet_refine<false, false>(g, partition, partition.k, lmax_global, false, 0, cut_before, curr_max_weight, mem_stack, exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);

                weight_t cut_after = uew ? edge_cut<true>(g, partition, exec_space) : edge_cut<false>(g, partition, exec_space);
                std::cout << "Jet Refinement (Uncoarsen) N=" << g.n << " Cut: " << cut_before << " -> " << cut_after << std::endl;
            }
        }
    }

    inline void gpu_rb_partition(Graph &g,
                                 partition_t k,
                                 f64 imbalance,
                                 u64 seed,
                                 u32 threshold,
                                 Partition &partition,
                                 KokkosMemoryStack &mem_stack,
                                 DeviceExecutionSpace &exec_space,
                                 BisectionMethod bisection_method = BisectionMethod::HEURISTIC_ONLY) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "gpu_rb_partition");

        // --- Phase 1: Coarsening ---
        std::vector<SmallGraph> graphs = {from_Graph_to_SmallGraph(g, mem_stack, exec_space)};
        std::vector<Mapping> mappings;
        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > threshold) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "coarsening");

            mappings.push_back(heavy_edge_matching_small_get_mapping<false, false>(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);

            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "contraction");
            graphs.push_back(from_Graph_Mapping_small<false, false, true>(graphs.back(), mappings.back(), mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // --- Phase 2: Interleaved Recursive Bisection and Uncontraction ---
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "partition_phase");

        GraphBatch batch;
        init_GraphBatch(batch, graphs[0], k, mem_stack);

        // Scratch for extraction
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * graphs[0].n), graphs[0].n);
        UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * graphs[0].n), graphs[0].n);
        UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        UnmanagedDeviceU8 active_mask((u8 *) get_chunk_back(mem_stack, sizeof(u8) * batch.k), batch.k);
        UnmanagedDeviceU32 current_targets_dev((u32 *) get_chunk_back(mem_stack, sizeof(u32) * batch.k), batch.k);

        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "init_targets");
        Kokkos::parallel_for("init_targets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const int id) {
            if (id == 0) {
                partition.bweights(id) = g.g_weight;
                current_targets_dev(id) = k;
            } else {
                partition.bweights(id) = 0;
                current_targets_dev(id) = 0;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        while (true) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "calculate_block_sizes_fused");
            u32 split_needed_int = 0;
            bool has_mapping = !mappings.empty();
            vertex_t old_n = has_mapping ? mappings.back().old_n : 0;
            UnmanagedDeviceVertex mapping_view = has_mapping ? mappings.back().mapping : UnmanagedDeviceVertex();
            auto map = partition.map;
            auto offset_n = batch.offset_n;
            vertex_t g_n = graphs.back().n;

            Kokkos::parallel_reduce("calculate_block_sizes_fused", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team, u32 &local_split) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k), [&](const int i) {
                    bsizes(i) = 0;
                });

                team.team_barrier();

                if (!has_mapping) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                        partition_t id = map(u);
                        Kokkos::atomic_add(&bsizes(id), 1);
                    });
                } else {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, old_n), [&](const vertex_t u) {
                        vertex_t new_v = mapping_view(u);
                        partition_t id = map(new_v);
                        Kokkos::atomic_add(&bsizes(id), 1);
                    });
                }

                team.team_barrier();

                if (team.team_rank() == 0) {
                    u32 running = 0;
                    u32 s = 0;
                    for (u32 id = 0; id < k; ++id) {
                        bool active = false;
                        if (current_targets_dev(id) > 1) {
                            if (!has_mapping) {
                                if (bsizes(id) >= 2 && bsizes(id) <= threshold) active = true;
                            } else {
                                if (bsizes(id) > threshold) active = true;
                            }
                        }
                        active_mask(id) = active ? 1 : 0;
                        if (active) s = 1;
                        offset_n(id) = running;
                        running += bsizes(id);
                    }
                    local_split = s;
                }
            }, Kokkos::Max<u32>(split_needed_int));
            
            bool split_needed = split_needed_int != 0;
            KOKKOS_PROFILE_FENCE(exec_space);

            if (split_needed) {

                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "extract_graphs");
                extract_all_subgraphs(graphs.back(), batch, partition, active_mask, local_ids, local_degree, exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);

                // split all graphs
                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "split_graphs");
                bool g_uvw = graphs.back().uniform_vertex_weights;
                bool g_uew = graphs.back().uniform_edge_weights;

                dispatch_batched_bisection(bisection_method, g_uvw, g_uew, batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);

                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "insert_solution_and_weights");
                // insert all solutions and update block weights
                auto p_map = partition.map;
                auto p_bweights = partition.bweights;
                auto d_actual_n = batch.d_actual_n;
                auto g_weights = graphs.back().weights;

                Kokkos::parallel_for("insert_all_solutions_and_weights", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, k, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
                    partition_t id = team.league_rank();

                    if (active_mask(id)) {
                        partition_t tk = current_targets_dev(id);
                        partition_t lk = tk / 2;
                        partition_t rk = tk - lk;

                        vertex_t sub_n = d_actual_n(id);
                        partition_t *sub_part_ptr = batch.get_partition_ptr(id);
                        vertex_t *sub_global_ids_ptr = batch.get_global_ids_ptr(id);

                        Kokkos::single(Kokkos::PerTeam(team), [&]() {
                            current_targets_dev(id) = lk;
                            current_targets_dev(id + lk) = rk;
                        });
                        team.team_barrier();

                        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, sub_n), [&](const vertex_t u) {
                            partition_t u_id = sub_part_ptr[u];
                            if (u_id != 0) {
                                vertex_t g_u = sub_global_ids_ptr[u];
                                weight_t w = g_uvw ? 1 : g_weights(g_u);
                                p_map(g_u) = id + lk;
                                Kokkos::atomic_sub(&p_bweights(id), w);
                                Kokkos::atomic_add(&p_bweights(id + lk), w);
                            }
                        });
                    }
                });
                KOKKOS_PROFILE_FENCE(exec_space);

                continue;
            }

            if (!mappings.empty()) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "uncoarsening");
                uncontract(partition, mappings.back(), exec_space);

                free_graph(graphs.back(), mem_stack);
                graphs.pop_back();

                free_mapping(mappings.back(), mem_stack);
                mappings.pop_back();

                KOKKOS_PROFILE_FENCE(exec_space);

                continue;
            }

            break;
        }

        // Cleanup
        pop_back(mem_stack); // current_targets_dev
        pop_back(mem_stack); // active_mask
        pop_back(mem_stack); // bsizes
        pop_back(mem_stack); // local_degree
        pop_back(mem_stack); // local_ids
        free_GraphBatch(batch, mem_stack);
        free_graph(graphs[0], mem_stack);
        KOKKOS_PROFILE_FENCE(exec_space);
    }
} // namespace GPU_HeiPa

#endif //GPU_HEIPA_GPU_RB_PARTITION_H
