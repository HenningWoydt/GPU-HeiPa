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
#include "gpu_bisection.h"

namespace GPU_HeiPa {
    struct HierarchyManager {
        std::vector<partition_t> hierarchy;
        partition_t total_k;
        std::vector<partition_t> unit_sizes;
        std::vector<u8> active;
        std::vector<u32> curr_level;
        std::vector<u32> curr_load;
        
    };

    inline void init_HierarchyManager(HierarchyManager &manager, const std::vector<partition_t> &t_hierarchy, size_t t_k) {
        manager.hierarchy = t_hierarchy;
        
        size_t num_levels = manager.hierarchy.size();
        manager.unit_sizes.assign(num_levels, 1);
        size_t current = 1;
        for (size_t i = 0; i < num_levels; ++i) {
            manager.unit_sizes[i] = (partition_t) current;
            current *= manager.hierarchy[i];
        }
        manager.total_k = (partition_t) current;
        manager.active.assign(manager.total_k, 0);
        manager.curr_level.assign(manager.total_k, 0);
        manager.curr_load.assign(manager.total_k, 0);
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
        if (right_id >= manager.total_k) throw std::runtime_error("max_blocks exceeded");
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

    inline void gpu_bisect_partition(Graph &g, const std::vector<partition_t> &hierarchy, partition_t k, f64 imbalance, u64 seed, u32 threshold, Partition &partition, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "gpu_bisect_partition");
        GraphBatch batch;
        init_GraphBatch(batch, g, k, mem_stack);
        HierarchyManager manager;
        init_HierarchyManager(manager, hierarchy, batch.k);
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
        DeviceU8 active_mask("active_mask", batch.k);
        DeviceWeight lmax_l("lmax_l", batch.k);
        DeviceWeight lmax_r("lmax_r", batch.k);
        DevicePartition left_strides("left_strides", batch.k);
        DevicePartition right_strides("right_strides", batch.k);
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
        UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
        UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        UnmanagedDeviceVertex projected_bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        std::vector<u8> iteration_active(batch.k);
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
                for (partition_t id = 0; id < (partition_t) batch.k; id++) {
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
                    // batched_bisect(batch, active_mask, lmax_l, lmax_r, exec_space);
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
