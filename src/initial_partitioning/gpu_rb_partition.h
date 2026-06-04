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
        // We MUST coarsen until the graph is small enough for brute-force (max 24)
        const u32 max_bf_n = 24;
        std::vector<Graph> graphs = {g};
        std::vector<Mapping> mappings;
        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > max_bf_n) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "coarsening");
            mappings.push_back(two_hop_matcher_get_mapping<false, false>(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            graphs.push_back(from_Graph_Mapping<false, false>(graphs.back(), mappings.back(), mem_stack, exec_space));
            contract(partition, mappings.back(), exec_space);
            exec_space.fence();
        }

        recalculate_block_weights(graphs.back(), partition.map, partition.bweights, exec_space);

        // --- Phase 2: Interleaved Recursive Bisection and Uncontraction ---
        {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "interleaved_rb_phase");

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
            DeviceWeight lmax_l("lmax_l", batch.max_blocks);
            DeviceWeight lmax_r("lmax_r", batch.max_blocks);
            DevicePartition left_target("left_target", batch.max_blocks);
            DevicePartition right_target("right_target", batch.max_blocks);

            std::vector<u8> h_batch_active(batch.max_blocks, 0);
            std::vector<u8> h_retry_active(batch.max_blocks, 0);
            std::vector<weight_t> h_lmax_l(batch.max_blocks, 0);
            std::vector<weight_t> h_lmax_r(batch.max_blocks, 0);
            std::vector<partition_t> h_left_target(batch.max_blocks, 0);
            std::vector<partition_t> h_right_target(batch.max_blocks, 0);

            while (true) {
                bool split_needed = false;
                std::fill(h_batch_active.begin(), h_batch_active.end(), 0);

                calculate_block_sizes(graphs.back(), nullptr, partition.map, bsizes, projected_bsizes, exec_space);
                auto h_bsizes = Kokkos::create_mirror_view_and_copy(HostMemory(), bsizes);

                bool any_target_gt_1 = false;
                for (partition_t id = 0; id < k; ++id) {
                    if (current_targets[id] > 1) {
                        any_target_gt_1 = true;
                        // We split now if n_sub <= 24 AND n_sub >= 2
                        if (h_bsizes(id) >= 2 && h_bsizes(id) <= max_bf_n) {
                            h_batch_active[id] = 1;
                            split_needed = true;
                        }
                    }
                }

                if (split_needed) {
                    Kokkos::deep_copy(exec_space, active_mask, Kokkos::View<u8 *, HostMemory>(h_batch_active.data(), batch.max_blocks));
                    extract_all_subgraphs(graphs.back(), batch, partition, active_mask, local_ids, local_degree, exec_space);
                    exec_space.fence();

                    for (partition_t id = 0; id < k; ++id) {
                        if (h_batch_active[id]) {
                            weight_t sub_w = batch.actual_g_weight(id);
                            partition_t tk = current_targets[id];
                            partition_t lk = tk / 2;
                            partition_t rk = tk - lk;

                            h_lmax_l[id] = std::max((weight_t)std::ceil((f64) sub_w * lk / (f64) tk * 1.10), (weight_t)1);
                            h_lmax_r[id] = std::max((weight_t)std::ceil((f64) sub_w * rk / (f64) tk * 1.10), (weight_t)1);
                            h_lmax_l[id] = std::min(h_lmax_l[id], (weight_t)(lk * lmax_global));
                            h_lmax_r[id] = std::min(h_lmax_r[id], (weight_t)(rk * lmax_global));

                            h_left_target[id] = lk;
                            h_right_target[id] = rk;
                        }
                    }

                    std::copy(h_batch_active.begin(), h_batch_active.end(), h_retry_active.begin());
                    int retries = 0;
                    while (retries < 6) {
                        Kokkos::deep_copy(exec_space, active_mask, Kokkos::View<u8 *, HostMemory>(h_retry_active.data(), batch.max_blocks));
                        Kokkos::deep_copy(exec_space, lmax_l, Kokkos::View<weight_t *, HostMemory>(h_lmax_l.data(), batch.max_blocks));
                        Kokkos::deep_copy(exec_space, lmax_r, Kokkos::View<weight_t *, HostMemory>(h_lmax_r.data(), batch.max_blocks));
                        exec_space.fence();

                        batched_bisect(batch, active_mask, lmax_l, lmax_r, exec_space);
                        exec_space.fence();

                        bool any_still_failing = false;
                        auto h_results = Kokkos::create_mirror_view_and_copy(HostMemory(), batch.d_bisection_results);
                        for (partition_t id = 0; id < k; ++id) {
                            if (h_retry_active[id]) {
                                if (h_results(id).penalty >= 1000000000000ULL) {
                                    h_lmax_l[id] = (weight_t)(h_lmax_l[id] * 1.2 + 1);
                                    h_lmax_r[id] = (weight_t)(h_lmax_r[id] * 1.2 + 1);
                                    any_still_failing = true;
                                } else {
                                    h_retry_active[id] = 0;
                                }
                            }
                        }
                        if (!any_still_failing) break;
                        retries++;
                    }

                    Kokkos::deep_copy(exec_space, active_mask, Kokkos::View<u8 *, HostMemory>(h_batch_active.data(), batch.max_blocks));
                    Kokkos::deep_copy(exec_space, left_target, Kokkos::View<partition_t *, HostMemory>(h_left_target.data(), batch.max_blocks));
                    u64 n_bytes_partition = round_up_64(batch.n) * sizeof(partition_t);
                    u8 *partition_base = batch.partition_memory.data();
                    
                    Kokkos::parallel_for("rb_map_update", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graphs.back().n), KOKKOS_LAMBDA(const vertex_t u) {
                        partition_t id = map(u);
                        if (active_mask(id)) {
                            vertex_t l_u = local_ids(u);
                            partition_t *sub_part = (partition_t *) (partition_base + (u64) id * n_bytes_partition);
                            if (sub_part[l_u] == 1) map(u) = id + left_target(id);
                        }
                    });
                    exec_space.fence();

                    for (partition_t id = 0; id < k; ++id) {
                        if (h_batch_active[id]) {
                            partition_t lk = h_left_target[id];
                            partition_t rk = h_right_target[id];
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

                if (!any_target_gt_1) break;
                break; // Finest level and no split needed or possible
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
