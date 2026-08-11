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
#include "../coarsening/heavy_edge_matching.h"
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
        std::vector<u32> curr_load;
    };

    inline void init_HierarchyManager(HierarchyManager &manager, const std::vector<partition_t> &t_hierarchy, size_t t_k) {
        manager.hierarchy = t_hierarchy;
        manager.total_k = t_k;
        manager.active.assign(t_k, 0);
        manager.active[0] = 1;
        manager.curr_load.assign(t_k, 0);
        manager.curr_load[0] = t_k;
        
        manager.unit_sizes.clear();
        partition_t current_unit = 1;
        manager.unit_sizes.push_back(current_unit);
        for (size_t i = 0; i < t_hierarchy.size(); i++) {
            current_unit *= t_hierarchy[i];
            if (current_unit < t_k) {
                manager.unit_sizes.push_back(current_unit);
            }
        }
        std::reverse(manager.unit_sizes.begin(), manager.unit_sizes.end());
    }

    inline void split_into(HierarchyManager &manager, partition_t id, partition_t &l_k, partition_t &r_k) {
        partition_t k = manager.curr_load[id];
        if (k <= 1) {
            l_k = 1; r_k = 0;
            return;
        }
        
        partition_t best_unit = 1;
        for (partition_t u : manager.unit_sizes) {
            if (u < k && k % u == 0) {
                best_unit = u;
                break;
            }
        }
        
        partition_t num_chunks = k / best_unit;
        partition_t left_chunks = num_chunks / 2;
        partition_t right_chunks = num_chunks - left_chunks;
        
        l_k = left_chunks * best_unit;
        r_k = right_chunks * best_unit;
    }

    inline void split(HierarchyManager &manager, partition_t id, partition_t l_k, partition_t r_k) {
        manager.active[id] = 1;
        manager.active[id + l_k] = 1;
        manager.curr_load[id] = l_k;
        manager.curr_load[id + l_k] = r_k;
    }

    inline void gpu_bisect_partition(Graph &g, const std::vector<partition_t> &hierarchy, partition_t k, f64 imbalance, u64 seed, u32 threshold, Partition &partition, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space, BisectionMethod bisection_method = BisectionMethod::BRUTE_FORCE) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "gpu_bisect_partition");

        // --- Phase 1: Coarsening ---
        std::vector<SmallGraph> graphs = {from_Graph_to_SmallGraph(g, mem_stack, exec_space)};
        std::vector<Mapping> mappings;
        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > threshold) {
            vertex_t old_n = graphs.back().n;
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "coarsening");
            mappings.push_back(dispatch_heavy_edge_matching_small_get_mapping(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);

            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "contraction");
            graphs.push_back(dispatch_from_Graph_Mapping_small<true>(graphs.back(), mappings.back(), mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // --- Phase 2: Interleaved Recursive Bisection and Uncontraction ---
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "partition_phase");

        GraphBatch batch;
        init_GraphBatch(batch, graphs[0], k, mem_stack);

        HierarchyManager manager;
        init_HierarchyManager(manager, hierarchy, batch.k);

        // Scratch memory
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * graphs[0].n), graphs[0].n);
        UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * graphs[0].n), graphs[0].n);
        UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        UnmanagedDeviceU8 active_mask((u8 *) get_chunk_back(mem_stack, sizeof(u8) * batch.k), batch.k);
        UnmanagedDeviceU32 current_targets_dev((u32 *) get_chunk_back(mem_stack, sizeof(u32) * batch.k), batch.k);

        HostU8 h_active_mask("h_active_mask", batch.k);
        HostPartition h_left_strides("h_left_strides", batch.k);
        HostPartition h_right_strides("h_right_strides", batch.k);
        HostU32 h_current_targets("h_current_targets", batch.k);
        DevicePartition left_strides("left_strides", batch.k);
        DevicePartition right_strides("right_strides", batch.k);

        Kokkos::parallel_for("init_partition", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graphs.back().n), KOKKOS_LAMBDA(const vertex_t u) {
            partition.map(u) = 0;
        });
        KOKKOS_PROFILE_FENCE(exec_space);
        
        while (true) {
            bool split_needed = false;
            
            Kokkos::deep_copy(exec_space, bsizes, 0);
            Kokkos::parallel_for("calc_sizes", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graphs.back().n), KOKKOS_LAMBDA(const vertex_t u) {
                partition_t target = partition.map(u);
                Kokkos::atomic_add(&bsizes(target), (vertex_t) 1);
            });
            KOKKOS_PROFILE_FENCE(exec_space);
            
            HostVertex h_bsizes("h_bsizes", batch.k);
            Kokkos::deep_copy(exec_space, h_bsizes, bsizes);
            exec_space.fence();
            
            Kokkos::parallel_scan("prefix_sum_batch_offsets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, batch.k), KOKKOS_LAMBDA(const partition_t id, vertex_t &update, const bool final) {
                if (final) batch.batch_offsets(id) = update;
                update += bsizes(id);
            });
            KOKKOS_PROFILE_FENCE(exec_space);
            
            Kokkos::deep_copy(h_active_mask, (u8) 0);
            
            for (partition_t id = 0; id < (partition_t) batch.k; id++) {
                if (manager.active[id] && manager.curr_load[id] > 1) {
                    vertex_t projected_n = h_bsizes(id);
                    if (mappings.empty() || projected_n > threshold) {
                        partition_t l_k, r_k;
                        split_into(manager, id, l_k, r_k);
                        h_active_mask(id) = 1;
                        h_left_strides(id) = id;
                        h_right_strides(id) = id + l_k;
                        h_current_targets(id) = (l_k & 0xFFFF) | (r_k << 16);
                        
                        split(manager, id, l_k, r_k);
                        split_needed = true;
                    }
                }
            }
            
            if (split_needed) {
                Kokkos::deep_copy(exec_space, active_mask, h_active_mask);
                Kokkos::deep_copy(exec_space, current_targets_dev, h_current_targets);
                Kokkos::deep_copy(exec_space, left_strides, h_left_strides);
                Kokkos::deep_copy(exec_space, right_strides, h_right_strides);
                exec_space.fence();
                
                dispatch_extract_all_subgraphs(graphs.back(), batch, partition, active_mask, local_ids, local_degree, exec_space);
                
                bool g_uvw = graphs.back().uniform_vertex_weights;
                bool g_uew = graphs.back().uniform_edge_weights;
                dispatch_batched_bisection(bisection_method, g_uvw, g_uew, batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
                
                auto d_results = batch.bisection_results;
                auto h_results = Kokkos::create_mirror_view(d_results);
                Kokkos::deep_copy(exec_space, h_results, d_results);
                exec_space.fence();
                
                auto map = partition.map;

                Kokkos::parallel_for("insert_all_solutions", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, batch.k, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
                    partition_t id = team.league_rank();
                    if (!active_mask(id)) return;
                    
                    vertex_t sub_n = batch.batch_ns(id);
                    partition_t *sub_part_ptr = batch.get_partition_ptr(id);
                    vertex_t *sub_global_ids_ptr = batch.get_global_ids_ptr(id);
                    partition_t left_id = left_strides(id);
                    partition_t right_id = right_strides(id);
                    
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, sub_n), [&](const vertex_t u) {
                        partition_t u_id = sub_part_ptr[u];
                        vertex_t g_u = sub_global_ids_ptr[u];
                        if (u_id != 0) {
                            map(g_u) = right_id;
                        } else {
                            map(g_u) = left_id;
                        }
                    });
                });
                KOKKOS_PROFILE_FENCE(exec_space);
                
                continue;
            }
            
            if (mappings.empty()) break;
            
            HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_bisection_partition", "uncoarsening");
            uncontract(partition, mappings.back(), exec_space);
            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();
            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();
            KOKKOS_PROFILE_FENCE(exec_space);
        }
        recalculate_block_weights(graphs.back(), partition.map, partition.bweights, exec_space);
        
        pop_back(mem_stack); // current_targets_dev
        pop_back(mem_stack); // active_mask
        pop_back(mem_stack); // bsizes
        pop_back(mem_stack); // local_degree
        pop_back(mem_stack); // local_ids
        free_GraphBatch(batch, mem_stack);
        free_graph(graphs[0], mem_stack);
    }
}

#endif //GPU_HEIPA_GPU_BISECTION_PARTITION_H
