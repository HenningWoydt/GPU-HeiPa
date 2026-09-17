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

#ifndef GPU_HEIPA_BIML_BISECTION_H
#define GPU_HEIPA_BIML_BISECTION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/small_graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/two_hop_matching.h"
#include "../coarsening/dumb_matching.h"
#include "../coarsening/heavy_edge_matching.h"
#include "../coarsening/independent_edge_set.h"
#include "../refinement/greedy_refinement.h"
#include "../refinement/jet_label_propagation.h"
#include "../utility/edge_cut.h"
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"

namespace GPU_HeiPa {
    enum class BisectionMethod {
        BRUTE_FORCE,
        BRUTE_FORCE_WITH_HEURISTIC,
        HEURISTIC_ONLY,
        GRASP
    };

    constexpr u64 OVERLOAD_PENALTY = 10000000000ULL;
    constexpr u64 EMPTY_BLOCK_PENALTY = 1000000000000ULL;

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

        KOKKOS_INLINE_FUNCTION BestBisectReducer(const result_view_type &view) : value(view.data()) {
        }

        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }

        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }

        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    struct GraphBatch {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;

        UnmanagedDeviceU8 graph_memory;
        UnmanagedDeviceU8 partition_memory;
        UnmanagedDeviceU8 global_ids_memory;

        UnmanagedDeviceVertex batch_ns;
        UnmanagedDeviceVertex batch_offsets;
        UnmanagedDeviceVertex batch_ms;
        UnmanagedDeviceWeight batch_weights;

        Kokkos::View<BestBisectConfig *, DeviceMemorySpace> bisection_results;
        KOKKOS_INLINE_FUNCTION
        partition_t *get_partition_ptr(partition_t id) const {
            u64 memory_offset = (u64) batch_offsets(id) * sizeof(partition_t);
            return (partition_t *) (partition_memory.data() + memory_offset);
        }

        KOKKOS_INLINE_FUNCTION
        vertex_t *get_global_ids_ptr(partition_t id) const {
            u64 memory_offset = (u64) batch_offsets(id) * sizeof(vertex_t);
            return (vertex_t *) (global_ids_memory.data() + memory_offset);
        }
    };

    inline void init_GraphBatch(GraphBatch &batch,
                                const SmallGraph &g,
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

        u64 n_bytes_partition_total = round_up_64(g.n) * sizeof(partition_t);
        batch.partition_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_partition_total), n_bytes_partition_total);

        u64 n_bytes_global_ids_total = round_up_64(g.n) * sizeof(vertex_t);
        batch.global_ids_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_global_ids_total), n_bytes_global_ids_total);

        batch.batch_ns = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        batch.batch_offsets = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        batch.batch_ms = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        batch.batch_weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * batch.k), batch.k);

        batch.bisection_results = Kokkos::View<BestBisectConfig *, DeviceMemorySpace>("d_bisection_results", batch.k);
    }

    inline void free_GraphBatch(GraphBatch &batch,
                                KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
    }


    template<bool uvw, bool uew>
    inline void extract_all_subgraphs(const SmallGraph &g,
                                      GraphBatch &batch,
                                      Partition &partition,
                                      const DeviceU8 &active_mask,
                                      UnmanagedDeviceVertex &local_ids,
                                      UnmanagedDeviceVertex &local_degree,
                                      DeviceExecutionSpace &exec_space) {
        // --- 1. Extract variables for Kokkos device lambdas ---
        auto map = partition.map;
        bool use_mask = active_mask.extent(0) > 0;

        // Graph properties
        auto g_weights = g.weights;
        auto g_begin = g.edge_begin;
        auto g_end = g.edge_end;
        auto g_edges_v = g.edges_v;

        // Batch properties
        partition_t k = batch.k;
        auto batch_ns = batch.batch_ns;
        auto batch_ms = batch.batch_ms;
        auto batch_weights = batch.batch_weights;
        auto batch_graph_memory = batch.graph_memory;

        // --- 2. Reset batched graph counters ---
        HEIPA_PROFILE_SCOPE("initial_partitioning", "extract_all_subgraphs", "reset");
        Kokkos::deep_copy(exec_space, batch_ns, 0);
        Kokkos::deep_copy(exec_space, batch_ms, 0);
        Kokkos::deep_copy(exec_space, batch_weights, 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        // --- 3. Compute memory byte offsets for a single graph in the batch ---
        vertex_t b_n = batch.n;
        vertex_t b_m = batch.m;

        u64 n_bytes_weights = round_up_64(b_n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(b_n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(b_m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(b_m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;

        HEIPA_PROFILE_SCOPE("initial_partitioning", "extract_all_subgraphs", "batched_vertex_assignment_and_edge_counting");
        Kokkos::parallel_for("batched_vertex_assignment_and_edge_counting", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);

            // 1. Skip inactive subgraphs if masking is used
            if (use_mask && !active_mask(id)) {
                local_degree(u) = 0;
                return;
            }

            // 2. Assign a new local ID to vertex `u` in its respective subgraph
            vertex_t local_u = Kokkos::atomic_fetch_add(&batch_ns(id), 1);
            local_ids(u) = local_u;

            // Map the subgraph's local ID back to the global graph's ID
            vertex_t *g_ids_ptr = batch.get_global_ids_ptr(id);
            g_ids_ptr[local_u] = u;

            // 3. Copy vertex weight to the subgraph memory
            weight_t w = uvw ? 1 : g_weights(u);
            weight_t *weights_ptr = (weight_t *) (batch_graph_memory.data() + (u64) id * n_bytes_one_graph);
            weights_ptr[local_u] = w;
            Kokkos::atomic_add(&batch_weights(id), w);

            // 4. Count the internal edges (edges connecting vertices in the same subgraph)
            u32 count = 0;
            u32 start = g_begin(u);
            u32 limit = g_end(u);
            for (u32 e = start; e < limit; ++e) {
                vertex_t v = g_edges_v(e);
                if (v == SENTINEL) break;
                if (map(v) == id) count++;
            }
            local_degree(u) = count;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // 1. Batched block neighborhood scan
        typedef Kokkos::TeamPolicy<DeviceExecutionSpace> TeamPolicy;
        typedef TeamPolicy::member_type TeamMember;

        HEIPA_PROFILE_SCOPE("initial_partitioning", "extract_all_subgraphs", "batched_block_neighborhood_scan");
        Kokkos::parallel_for("batched_block_neighborhood_scan", TeamPolicy(exec_space, k, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
            partition_t id = team.league_rank();
            vertex_t sub_n = batch_ns(id);
            if (sub_n == 0) return;

            u32 *sub_g_neighborhood = (u32 *) (batch_graph_memory.data() + (u64) id * n_bytes_one_graph + n_bytes_weights);
            vertex_t *g_ids_ptr = batch.get_global_ids_ptr(id);

            u32 total_m = 0;
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, sub_n), [&](const vertex_t local_u, u32 &running, bool final) {
                u32 deg = local_degree(g_ids_ptr[local_u]);
                if (final) sub_g_neighborhood[local_u] = running;
                running += deg;
            }, total_m);

            if (team.team_rank() == 0) {
                sub_g_neighborhood[sub_n] = total_m;
                batch_ms(id) = total_m;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        auto g_edges_w = g.edges_w;
        HEIPA_PROFILE_SCOPE("initial_partitioning", "extract_all_subgraphs", "batched_edge_population");
        Kokkos::parallel_for("batched_edge_population", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            if (use_mask && !active_mask(id)) return;

            // Reconstruct pointers to the subgraph's data arrays inside the packed memory
            u8 *base_ptr = batch_graph_memory.data() + (u64) id * n_bytes_one_graph;
            u32 *sub_g_neighborhood = (u32 *) (base_ptr + n_bytes_weights);
            vertex_t *sub_g_edges_u = (vertex_t *) ((u8 *) sub_g_neighborhood + n_bytes_neighborhood);
            vertex_t *sub_g_edges_v = (vertex_t *) ((u8 *) sub_g_edges_u + n_bytes_edges_u);
            weight_t *sub_g_edges_w = (weight_t *) ((u8 *) sub_g_edges_v + n_bytes_edges_v);

            // Find where this vertex's edges should start in the subgraph
            u32 edge_idx = sub_g_neighborhood[local_ids(u)];

            // Iterate over all edges, keeping only those entirely within the same subgraph
            u32 start = g_begin(u);
            u32 limit = g_end(u);
            for (u32 e = start; e < limit; ++e) {
                vertex_t v = g_edges_v(e);
                if (v == SENTINEL) break;

                if (map(v) == id) {
                    sub_g_edges_u[edge_idx] = local_ids(u);
                    sub_g_edges_v[edge_idx] = local_ids(v);
                    sub_g_edges_w[edge_idx] = !uew ? g_edges_w(e) : 1;
                    edge_idx++;
                }
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);
    }

    inline void dispatch_extract_all_subgraphs(const SmallGraph &g,
                                               GraphBatch &batch,
                                               Partition &partition,
                                               const DeviceU8 &active_mask,
                                               UnmanagedDeviceVertex &local_ids,
                                               UnmanagedDeviceVertex &local_degree,
                                               DeviceExecutionSpace &exec_space) {
        bool uvw = g.uniform_vertex_weights;
        bool uew = g.uniform_edge_weights;
        if (uvw && uew) {
            extract_all_subgraphs<true, true>(g, batch, partition, active_mask, local_ids, local_degree, exec_space);
        } else if (uvw) {
            extract_all_subgraphs<true, false>(g, batch, partition, active_mask, local_ids, local_degree, exec_space);
        } else if (uew) {
            extract_all_subgraphs<false, true>(g, batch, partition, active_mask, local_ids, local_degree, exec_space);
        } else {
            extract_all_subgraphs<false, false>(g, batch, partition, active_mask, local_ids, local_degree, exec_space);
        }
    }
} // namespace GPU_HeiPa

#include "bisection_brute_force.h"
#include "bisection_heuristic.h"
#include "bisection_grasp.h"

namespace GPU_HeiPa {
    inline void dispatch_batched_bisection(BisectionMethod method,
                                           bool uvw,
                                           bool uew,
                                           const GraphBatch &batch,
                                           const DeviceU8 &active_mask,
                                           const DeviceU32 &current_targets_dev,
                                           weight_t lmax_global,
                                           KokkosMemoryStack &mem_stack,
                                           DeviceExecutionSpace &exec_space,
                                           u64 seed = 0) {
        if (method == BisectionMethod::HEURISTIC_ONLY) {
            if (uvw && uew) batched_heuristic_bisect<true, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else if (uvw) batched_heuristic_bisect<true, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else if (uew) batched_heuristic_bisect<false, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else batched_heuristic_bisect<false, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
        } else if (method == BisectionMethod::BRUTE_FORCE_WITH_HEURISTIC) {
            if (uvw && uew) batched_brute_force_bisect<true, true, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else if (uvw) batched_brute_force_bisect<true, false, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else if (uew) batched_brute_force_bisect<false, true, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else batched_brute_force_bisect<false, false, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
        } else if (method == BisectionMethod::BRUTE_FORCE) {
            if (uvw && uew) batched_brute_force_bisect<true, true, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else if (uvw) batched_brute_force_bisect<true, false, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else if (uew) batched_brute_force_bisect<false, true, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
            else batched_brute_force_bisect<false, false, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space);
        } else if (method == BisectionMethod::GRASP) {
            if (uvw && uew) batched_grasp_bisect<true, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space, seed);
            else if (uvw) batched_grasp_bisect<true, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space, seed);
            else if (uew) batched_grasp_bisect<false, true>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space, seed);
            else batched_grasp_bisect<false, false>(batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space, seed);
        }
    }

    inline void biml_bisection(Graph &g,
                               partition_t k,
                               f64 imbalance,
                               u64 seed,
                               u32 threshold,
                               Partition &partition,
                               KokkosMemoryStack &mem_stack,
                               DeviceExecutionSpace &exec_space,
                               BisectionMethod bisection_method = BisectionMethod::HEURISTIC_ONLY) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "biml_bisection");

        // --- Phase 1: Coarsening ---
        std::vector<SmallGraph> graphs = {from_Graph_to_SmallGraph(g, mem_stack, exec_space)};
        std::vector<Mapping> mappings;
        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > threshold) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "coarsening");

            mappings.push_back(dispatch_heavy_edge_matching_small_get_mapping(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);

            HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "contraction");
            graphs.push_back(dispatch_from_Graph_Mapping_small<true>(graphs.back(), mappings.back(), mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // --- Phase 2: Interleaved Recursive Bisection and Uncontraction ---
        HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "partition_phase");

        GraphBatch batch;
        init_GraphBatch(batch, graphs[0], k, mem_stack);

        // Scratch for extraction
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * graphs[0].n), graphs[0].n);
        UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * graphs[0].n), graphs[0].n);
        UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * batch.k), batch.k);
        UnmanagedDeviceU8 active_mask((u8 *) get_chunk_back(mem_stack, sizeof(u8) * batch.k), batch.k);
        UnmanagedDeviceU32 current_targets_dev((u32 *) get_chunk_back(mem_stack, sizeof(u32) * batch.k), batch.k);

        HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "init_targets");
        Kokkos::parallel_for("init_targets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const int id) {
            partition.bweights(id) = (id == 0) ? g.g_weight : 0;
            partition_t left_k = k / 2;
            partition_t right_k = k - left_k;
            current_targets_dev(id) = (id == 0) ? ((left_k & 0xFFFF) | (right_k << 16)) : 0;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        while (true) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "calculate_block_sizes_fused");
            u32 split_needed_int = 0;
            
            // Extract necessary variables for the Kokkos kernel
            bool has_mapping = !mappings.empty();
            vertex_t old_n = has_mapping ? mappings.back().old_n : 0;
            UnmanagedDeviceVertex mapping_view = has_mapping ? mappings.back().mapping : UnmanagedDeviceVertex();
            auto map = partition.map;
            auto offset_n = batch.batch_offsets;
            vertex_t g_n = graphs.back().n;

            Kokkos::parallel_reduce("calculate_block_sizes_fused", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team, u32 &local_split) {
                // 1. Initialize block sizes to 0
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k), [&](const int i) {
                    bsizes(i) = 0;
                });
                team.team_barrier();

                // 2. Count the number of vertices in each partition block
                if (has_mapping) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, old_n), [&](const vertex_t u) {
                        vertex_t new_v = mapping_view(u);
                        partition_t id = map(new_v);
                        Kokkos::atomic_add(&bsizes(id), 1);
                    });
                } else {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                        partition_t id = map(u);
                        Kokkos::atomic_add(&bsizes(id), 1);
                    });
                }
                team.team_barrier();

                // 3. Compute running offsets and identify active blocks that need splitting
                if (team.team_rank() == 0) {
                    u32 running = 0;
                    u32 s = 0;
                    
                    for (u32 id = 0; id < k; ++id) {
                        bool active = false;
                        
                        partition_t packed_targets = current_targets_dev(id);
                        partition_t left_targets = packed_targets & 0xFFFF;
                        partition_t right_targets = packed_targets >> 16;
                        
                        if (left_targets + right_targets > 1) {
                            if (has_mapping) {
                                active = (bsizes(id) > threshold);
                            } else {
                                active = (bsizes(id) >= 2 && bsizes(id) <= threshold);
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

            bool split_needed = (split_needed_int != 0);
            KOKKOS_PROFILE_FENCE(exec_space);

            if (split_needed) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "extract_graphs");
                dispatch_extract_all_subgraphs(graphs.back(), batch, partition, active_mask, local_ids, local_degree, exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);

                // split all graphs
                HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "split_graphs");
                bool g_uvw = graphs.back().uniform_vertex_weights;
                bool g_uew = graphs.back().uniform_edge_weights;
                dispatch_batched_bisection(bisection_method, g_uvw, g_uew, batch, active_mask, current_targets_dev, lmax_global, mem_stack, exec_space, seed);
                KOKKOS_PROFILE_FENCE(exec_space);

                HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "insert_solution_and_weights");
                // insert all solutions and update block weights
                auto p_map = partition.map;
                auto p_bweights = partition.bweights;
                auto d_actual_n = batch.batch_ns;
                auto g_weights = graphs.back().weights;

                Kokkos::parallel_for("insert_all_solutions_and_weights", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, k, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
                    partition_t id = team.league_rank();

                    // 1. Skip inactive partition blocks
                    if (!active_mask(id)) return;

                    // 2. Calculate targets for left and right bisection splits
                    partition_t packed_targets = current_targets_dev(id);
                    partition_t left_targets = packed_targets & 0xFFFF;
                    partition_t right_targets = packed_targets >> 16;
                    // partition_t total_targets = left_targets + right_targets;

                    vertex_t sub_n = d_actual_n(id);
                    partition_t *sub_part_ptr = batch.get_partition_ptr(id);
                    vertex_t *sub_global_ids_ptr = batch.get_global_ids_ptr(id);

                    // 3. Update the global targets array for the next level of bisection
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        partition_t left_l_k = left_targets / 2;
                        partition_t left_r_k = left_targets - left_l_k;
                        partition_t right_l_k = right_targets / 2;
                        partition_t right_r_k = right_targets - right_l_k;
                        current_targets_dev(id) = (left_l_k & 0xFFFF) | (left_r_k << 16);
                        current_targets_dev(id + left_targets) = (right_l_k & 0xFFFF) | (right_r_k << 16);
                    });
                    team.team_barrier();

                    // 4. Update the global partition map and block weights
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, sub_n), [&](const vertex_t u) {
                        partition_t u_id = sub_part_ptr[u];
                        
                        // If the vertex was moved to the right partition (u_id != 0)
                        if (u_id != 0) {
                            vertex_t g_u = sub_global_ids_ptr[u];
                            weight_t w = g_uvw ? 1 : g_weights(g_u);
                            
                            // Assign to the new right-side partition ID
                            p_map(g_u) = id + left_targets;
                            
                            // Transfer the weight from the left partition to the right partition
                            Kokkos::atomic_sub(&p_bweights(id), w);
                            Kokkos::atomic_add(&p_bweights(id + left_targets), w);
                        }
                    });
                });
                KOKKOS_PROFILE_FENCE(exec_space);

                continue;
            }

            if (!mappings.empty()) {
                HEIPA_PROFILE_SCOPE("initial_partitioning", "biml_bisection", "uncoarsening");
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

    inline void gpu_rb_partition(Graph &g,
                                 partition_t k,
                                 f64 imbalance,
                                 u64 seed,
                                 u32 threshold,
                                 Partition &partition,
                                 KokkosMemoryStack &mem_stack,
                                 DeviceExecutionSpace &exec_space,
                                 BisectionMethod bisection_method = BisectionMethod::HEURISTIC_ONLY) {
        biml_bisection(g, k, imbalance, seed, threshold, partition, mem_stack, exec_space, bisection_method);
    }
} // namespace GPU_HeiPa

#endif // GPU_HEIPA_BIML_BISECTION_H
