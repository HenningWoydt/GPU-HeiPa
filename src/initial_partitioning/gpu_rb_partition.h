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
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "gpu_bisection.h"

namespace GPU_HeiPa {
    inline void gpu_rb_partition(Graph &g,
                                 partition_t k,
                                 f64 imbalance,
                                 u64 seed,
                                 u32 threshold,
                                 Partition &partition,
                                 KokkosMemoryStack &mem_stack,
                                 DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "gpu_rb_partition");

        // --- Phase 1: Coarsening ---
        std::vector<Graph> graphs = {g};
        std::vector<Mapping> mappings;
        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > threshold) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "coarsening");
            assert_state_pre_partition(graphs.back(), exec_space);

            mappings.push_back(two_hop_matcher_get_mapping<false, false>(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            graphs.push_back(from_Graph_Mapping<false, false>(graphs.back(), mappings.back(), mem_stack, exec_space));
            contract(partition, mappings.back(), exec_space);
            exec_space.fence();

            assert_coarsening(graphs[graphs.size() - 2], graphs.back(), mappings.back(), exec_space);
            assert_state_pre_partition(graphs.back(), exec_space);
        }

        recalculate_block_weights(graphs.back(), partition.map, partition.bweights, exec_space);

        // --- Phase 2: Interleaved Recursive Bisection and Uncontraction ---
        {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "partition_phase");

            GraphBatch batch;
            init_GraphBatch(batch, g, k, mem_stack);

            DeviceU32 current_targets_dev("current_targets_dev", k);
            Kokkos::deep_copy(exec_space, current_targets_dev, 0);
            Kokkos::parallel_for("init_targets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
                current_targets_dev(0) = k;
            });
            exec_space.fence();

            // Initialize partition map on coarsest graph to all 0
            auto map = partition.map;
            Kokkos::deep_copy(exec_space, map, 0);
            exec_space.fence();

            // Scratch for extraction
            UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
            UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
            UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.max_blocks), batch.max_blocks);
            UnmanagedDeviceVertex projected_bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.max_blocks), batch.max_blocks);

            UnmanagedDeviceU8 active_mask((u8 *) get_chunk_back(mem_stack, sizeof(u8) * batch.max_blocks), batch.max_blocks);
            UnmanagedDevicePartition left_k((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * batch.max_blocks), batch.max_blocks);
            UnmanagedDevicePartition right_k((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * batch.max_blocks), batch.max_blocks);
            UnmanagedDeviceWeight left_lmax((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * batch.max_blocks), batch.max_blocks);
            UnmanagedDeviceWeight right_lmax((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * batch.max_blocks), batch.max_blocks);

            // std::cout << "k = " << k << std::endl;
            // print(current_targets_dev, "target_dev ");

            while (true) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "calculate_block_sizes");
                calculate_block_sizes(graphs.back(), mappings.empty() ? nullptr : &mappings.back(), partition.map, bsizes, projected_bsizes, exec_space);

                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "get_active");
                u32 split_needed_int = 0;
                bool is_mapping_empty = mappings.empty();
                Kokkos::parallel_reduce("check_split_needed", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t id, u32 &local_split) {
                    bool active = false;
                    if (current_targets_dev(id) > 1) {
                        if (is_mapping_empty) {
                            if (bsizes(id) >= 2 && bsizes(id) <= threshold) {
                                active = true;
                            }
                        } else {
                            if (projected_bsizes(id) > threshold) {
                                active = true;
                            }
                        }
                    }
                    active_mask(id) = active ? 1 : 0;

                    if (active) {
                        partition_t tk = current_targets_dev(id);
                        left_k(id) = tk / 2;
                        right_k(id) = tk - left_k(id);
                        left_lmax(id) = lmax_global * left_k(id);
                        right_lmax(id) = lmax_global * right_k(id);
                    }

                    if (active) local_split = 1;
                }, Kokkos::Max<u32>(split_needed_int));
                exec_space.fence();
                bool split_needed = split_needed_int != 0;

                // std::cout << "k = " << k << std::endl;
                // print(current_targets_dev, "target_dev ");
                // print(bsizes, "bsizes");
                // print(projected_bsizes, "bsizes proj");
                // std::cout << "n_splits " << split_needed_int << std::endl;
                // print(active_mask, "active");

                if (split_needed) {
                    HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "extract_graphs");
                    extract_all_subgraphs(graphs.back(), batch, partition, active_mask, local_ids, local_degree, exec_space);
                    exec_space.fence();

                    // split all graphs
                    HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "split_graphs");
                    auto d_results = batch.d_bisection_results;
                    std::vector<DeviceExecutionSpace> instances = Kokkos::Experimental::partition_space(exec_space, std::vector<int>(k, 1));
                    
                    for (partition_t id = 0; id < k; ++id) {
                        Graph sub_g = get_Graph(batch, id);
                        if (sub_g.n == 0) continue;

                        UnmanagedDevicePartition sub_part = get_partition(batch, id);

                        auto result_view_managed = Kokkos::subview(d_results, id);
                        BestBisectReducer::result_view_type result_view(result_view_managed.data());

                        if (sub_g.uniform_vertex_weights && sub_g.uniform_edge_weights) brute_force_bisect_async<true, true, 128>(sub_g, left_lmax, right_lmax, id, sub_part, result_view, instances[id]);
                        else if (sub_g.uniform_vertex_weights) brute_force_bisect_async<true, false, 128>(sub_g, left_lmax, right_lmax, id, sub_part, result_view, instances[id]);
                        else if (sub_g.uniform_edge_weights) brute_force_bisect_async<false, true, 128>(sub_g, left_lmax, right_lmax, id, sub_part, result_view, instances[id]);
                        else brute_force_bisect_async<false, false, 128>(sub_g, left_lmax, right_lmax, id, sub_part, result_view, instances[id]);
                    }
                    for (auto &inst: instances) inst.fence();

                    HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "insert_solution");
                    // insert all solutions
                    auto p_map = partition.map;
                    auto d_actual_n = batch.d_actual_n;

                    Kokkos::parallel_for("insert_all_solutions", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, k, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
                        partition_t id = team.league_rank();

                        if (active_mask(id)) {
                            partition_t tk = current_targets_dev(id);
                            partition_t lk = tk / 2;
                            partition_t rk = tk - lk;

                            vertex_t sub_n = d_actual_n(id) > 0 ? d_actual_n(id) : batch.n;
                            partition_t* sub_part_ptr = batch.get_partition_ptr(id);
                            vertex_t* sub_global_ids_ptr = batch.get_global_ids_ptr(id);

                            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, sub_n), [&](const vertex_t u) {
                                partition_t u_id = sub_part_ptr[u];
                                vertex_t g_u = sub_global_ids_ptr[u];
                                if (u_id == 0) {
                                    p_map(g_u) = id;
                                } else {
                                    p_map(g_u) = id + lk;
                                }
                            });

                            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                                current_targets_dev(id) = lk;
                                current_targets_dev(id + lk) = rk;
                            });
                        }
                    });
                    exec_space.fence();

                    HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "recalculate_block_weights");
                    recalculate_block_weights(graphs.back(), partition.map, partition.bweights, exec_space);
                    continue;
                }

                if (!mappings.empty()) {
                    HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "uncoarsening");
                    uncontract(partition, mappings.back(), exec_space);
                    free_graph(graphs.back(), mem_stack);
                    graphs.pop_back();

                    free_mapping(mappings.back(), mem_stack);
                    mappings.pop_back();

                    recalculate_block_weights(graphs.back(), partition.map, partition.bweights, exec_space);
                    continue;
                }

                break;
            }

            // Cleanup
            pop_back(mem_stack); // right_lmax
            pop_back(mem_stack); // left_lmax
            pop_back(mem_stack); // right_k
            pop_back(mem_stack); // left_k
            pop_back(mem_stack); // active_mask
            pop_back(mem_stack); // projected_bsizes
            pop_back(mem_stack); // bsizes
            pop_back(mem_stack); // local_degree
            pop_back(mem_stack); // local_ids
            free_GraphBatch(batch, mem_stack);
        }

        // recalculate_block_weights(g, partition.map, partition.bweights, exec_space);
    }
} // namespace GPU_HeiPa

#endif //GPU_HEIPA_GPU_RB_PARTITION_H
