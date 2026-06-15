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

            std::vector<partition_t> current_targets(k, 0);
            current_targets[0] = k;

            // Initialize partition map on coarsest graph to all 0
            auto map = partition.map;
            Kokkos::deep_copy(exec_space, map, 0);
            exec_space.fence();

            // Scratch for extraction
            UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
            UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
            UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.max_blocks), batch.max_blocks);
            UnmanagedDeviceVertex projected_bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.max_blocks), batch.max_blocks);

            DeviceU8 active_mask("active_mask", batch.max_blocks);
            std::vector<u8> h_batch_active(batch.max_blocks, 0);

            while (true) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "get_active");
                calculate_block_sizes(graphs.back(), mappings.empty() ? nullptr : &mappings.back(), partition.map, bsizes, projected_bsizes, exec_space);
                auto h_bsizes = Kokkos::create_mirror_view_and_copy(HostMemory(), bsizes);
                auto h_proj_bsizes = Kokkos::create_mirror_view_and_copy(HostMemory(), projected_bsizes);

                bool split_needed = false;
                std::fill(h_batch_active.begin(), h_batch_active.end(), 0);
                for (partition_t id = 0; id < k; ++id) {
                    if (current_targets[id] > 1) {
                        // We split now if n_sub <= threshold AND n_sub >= 2
                        if (mappings.empty()) {
                            if (h_bsizes(id) >= 2 && h_bsizes(id) <= threshold) {
                                h_batch_active[id] = 1;
                                split_needed = true;
                            }
                        } else {
                            if (h_proj_bsizes(id) > threshold) {
                                h_batch_active[id] = 1;
                                split_needed = true;
                            }
                        }
                    }
                }

                // print(h_bsizes, "pre:");
                // print(h_proj_bsizes, "aft:");
                // print(h_batch_active, "act:");

                if (split_needed) {
                    HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "extract_graphs");
                    Kokkos::deep_copy(exec_space, active_mask, Kokkos::View<u8 *, HostMemory>(h_batch_active.data(), batch.max_blocks));
                    extract_all_subgraphs(graphs.back(), batch, partition, active_mask, local_ids, local_degree, exec_space);
                    exec_space.fence();

                    // split all graphs
                    HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "split_graphs");
                    auto d_results = batch.d_bisection_results;
                    std::vector<DeviceExecutionSpace> instances = Kokkos::Experimental::partition_space(exec_space, std::vector<int>(k, 1));
                    for (partition_t id = 0; id < k; ++id) {
                        if (h_batch_active[id]) {
                            partition_t tk = current_targets[id];
                            partition_t lk = tk / 2;
                            partition_t rk = tk - lk;

                            Graph sub_g = get_Graph(batch, id);
                            UnmanagedDevicePartition sub_part = get_partition(batch, id);

                            auto result_view_managed = Kokkos::subview(d_results, id);
                            BestBisectReducer::result_view_type result_view(result_view_managed.data());

                            weight_t lmax_l = lmax_global * lk;
                            weight_t lmax_r = lmax_global * rk;

                            // std::cout << "splitting " << id << " into " << lk << " " << rk << std::endl;

                            if (sub_g.uniform_vertex_weights && sub_g.uniform_edge_weights) brute_force_bisect_async<true, true, 64>(sub_g, lmax_l, lmax_r, sub_part, result_view, instances[id]);
                            else if (sub_g.uniform_vertex_weights) brute_force_bisect_async<true, false, 64>(sub_g, lmax_l, lmax_r, sub_part, result_view, instances[id]);
                            else if (sub_g.uniform_edge_weights) brute_force_bisect_async<false, true, 64>(sub_g, lmax_l, lmax_r, sub_part, result_view, instances[id]);
                            else brute_force_bisect_async<false, false, 64>(sub_g, lmax_l, lmax_r, sub_part, result_view, instances[id]);
                        }
                    }
                    for (auto &inst: instances) inst.fence();

                    // insert all solutions
                    for (partition_t id = 0; id < k; ++id) {
                        if (h_batch_active[id]) {
                            partition_t tk = current_targets[id];
                            partition_t lk = tk / 2;
                            partition_t rk = tk - lk;

                            Graph sub_g = get_Graph(batch, id);
                            UnmanagedDevicePartition sub_part = get_partition(batch, id);

                            // std::cout << "saving " << id << " split " << lk << " " << rk << " to " << id << " and " << id + lk << std::endl;

                            UnmanagedDeviceVertex sub_global_ids = get_global_ids(batch, id);
                            Kokkos::parallel_for("rb_map_update", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                                partition_t u_id = sub_part(u);
                                vertex_t g_u = sub_global_ids(u);
                                if (u_id == 0) {
                                    partition.map(g_u) = id;
                                } else {
                                    partition.map(g_u) = id + lk;
                                }
                            });
                            exec_space.fence();

                            current_targets[id] = lk;
                            current_targets[id + lk] = rk;
                        }
                    }

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
            pop_back(mem_stack); // projected_bsizes
            pop_back(mem_stack); // bsizes
            pop_back(mem_stack); // local_degree
            pop_back(mem_stack); // local_ids
            free_GraphBatch(batch, mem_stack);
        }

        recalculate_block_weights(g, partition.map, partition.bweights, exec_space);
    }
} // namespace GPU_HeiPa

#endif //GPU_HEIPA_GPU_RB_PARTITION_H
