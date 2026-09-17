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

#ifndef GPU_HEIPA_RECURSIVE_BISECTION_H
#define GPU_HEIPA_RECURSIVE_BISECTION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/small_graph.h"
#include "../datastructures/mtx_graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/heavy_edge_matching.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "cpu_fm_refinement.h"
#include "biml_bisection.h"

namespace GPU_HeiPa {
    /**
     * Extracts an induced subgraph from `src` corresponding to partition block `part_id`.
     * Allocations are placed on `mem_stack` (front), while temporary scratch is placed on (back).
     */
    inline void extract_induced_subgraph(const SmallGraph &src,
                                         const UnmanagedDeviceVertex &src_global_ids,
                                         const UnmanagedDevicePartition &part_map,
                                         partition_t part_id,
                                         SmallGraph &dst,
                                         UnmanagedDeviceVertex &dst_global_ids,
                                         KokkosMemoryStack &mem_stack,
                                         DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "recursive_bisection", "extract_induced_subgraph");

        // Temporary scratch on back of mem_stack
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * src.n), src.n);
        UnmanagedDeviceVertex local_deg((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * src.n), src.n);

        // Step 1: Filter vertices belonging to part_id and compute local dense IDs
        vertex_t n_dst = 0;
        Kokkos::parallel_scan("filter_vertices", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, src.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &running, bool final) {
            bool in_part = (part_map(u) == part_id);
            if (final) {
                local_ids(u) = in_part ? running : SENTINEL;
            }
            if (in_part) running++;
        }, n_dst);

        if (n_dst == 0) {
            dst.n = 0;
            dst.m = 0;
            dst.g_weight = 0;
            dst.n_pops = 0;
            pop_back(mem_stack); // local_deg
            pop_back(mem_stack); // local_ids
            return;
        }

        // Step 2: Compute local incident degrees
        auto s_begin = src.edge_begin;
        auto s_end = src.edge_end;
        auto s_ev = src.edges_v;
        Kokkos::parallel_for("count_local_degrees", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, src.n), KOKKOS_LAMBDA(const vertex_t u) {
            if (part_map(u) != part_id) {
                local_deg(u) = 0;
                return;
            }
            u32 count = 0;
            u32 start = s_begin(u);
            u32 limit = s_end(u);
            for (u32 e = start; e < limit; ++e) {
                vertex_t v = s_ev(e);
                if (v == SENTINEL) break;
                if (part_map(v) == part_id) {
                    count++;
                }
            }
            local_deg(u) = count;
        });

        // Step 3: Allocate dst vertex arrays on mem_stack (front)
        dst.n = n_dst;
        dst.uniform_vertex_weights = src.uniform_vertex_weights;
        dst.uniform_edge_weights = src.uniform_edge_weights;
        dst.n_pops = 6;

        dst.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * dst.n), dst.n);
        dst.edge_begin = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * dst.n), dst.n);
        dst.edge_end = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * dst.n), dst.n);

        // Step 4: Map vertex weights, degrees, and global IDs
        auto d_w = dst.weights;
        auto d_beg = dst.edge_begin;
        auto s_w = src.weights;
        bool uvw = src.uniform_vertex_weights;

        // Allocate dst_global_ids temporarily on stack later, or now?
        // To preserve LIFO: we allocate dst.edges_u, edges_v, edges_w first, then dst_global_ids!
        // First, temporarily store degree in dst.edge_begin
        Kokkos::parallel_for("copy_vw_and_deg", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, src.n), KOKKOS_LAMBDA(const vertex_t u) {
            if (part_map(u) == part_id) {
                vertex_t lu = local_ids(u);
                d_w(lu) = uvw ? 1 : s_w(u);
                d_beg(lu) = local_deg(u);
            }
        });

        // Prefix sum on dst.edge_begin to get edge offsets and total edges
        vertex_t m_dst = 0;
        auto d_end = dst.edge_end;
        Kokkos::parallel_scan("scan_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, dst.n), KOKKOS_LAMBDA(const vertex_t lu, vertex_t &running, bool final) {
            u32 deg = d_beg(lu);
            if (final) {
                d_beg(lu) = running;
                d_end(lu) = running + deg;
            }
            running += deg;
        }, m_dst);
        dst.m = m_dst;

        // Step 5: Allocate dst edge arrays (front)
        dst.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * dst.m), dst.m);
        dst.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * dst.m), dst.m);
        dst.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * dst.m), dst.m);

        // Step 6: Allocate dst_global_ids on front (popped before free_graph)
        dst_global_ids = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * dst.n), dst.n);

        // Copy global IDs
        auto d_gids = dst_global_ids;
        Kokkos::parallel_for("copy_gids", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, src.n), KOKKOS_LAMBDA(const vertex_t u) {
            if (part_map(u) == part_id) {
                d_gids(local_ids(u)) = src_global_ids(u);
            }
        });

        // Step 7: Populate induced edges
        auto d_eu = dst.edges_u;
        auto d_ev = dst.edges_v;
        auto d_ew = dst.edges_w;
        auto s_ew = src.edges_w;
        bool uew = src.uniform_edge_weights;

        Kokkos::parallel_for("populate_induced_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, src.n), KOKKOS_LAMBDA(const vertex_t u) {
            if (part_map(u) != part_id) return;

            vertex_t lu = local_ids(u);
            u32 e_idx = d_beg(lu);
            u32 start = s_begin(u);
            u32 limit = s_end(u);

            for (u32 e = start; e < limit; ++e) {
                vertex_t v = s_ev(e);
                if (v == SENTINEL) break;
                if (part_map(v) == part_id) {
                    d_eu(e_idx) = lu;
                    d_ev(e_idx) = local_ids(v);
                    d_ew(e_idx) = uew ? 1 : s_ew(e);
                    e_idx++;
                }
            }
        });

        // Step 8: Calculate total graph weight
        weight_t g_weight = 0;
        Kokkos::parallel_reduce("calc_g_weight", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, dst.n), KOKKOS_LAMBDA(const vertex_t lu, weight_t &sum) {
            sum += d_w(lu);
        }, g_weight);
        dst.g_weight = g_weight;

        // Cleanup scratch from back
        pop_back(mem_stack); // local_deg
        pop_back(mem_stack); // local_ids
        KOKKOS_PROFILE_FENCE(exec_space);
    }

    inline void free_induced_subgraph(SmallGraph &sg,
                                      KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack); // dst_global_ids
        free_graph(sg, mem_stack); // 6 chunks of SmallGraph
    }

    /**
     * Bisects a single coarse SmallGraph into 2 parts using dispatch_batched_bisection.
     */
    inline void bisect_single_coarse_graph(const SmallGraph &g,
                                           partition_t left_k,
                                           partition_t right_k,
                                           weight_t lmax_global,
                                           UnmanagedDevicePartition &part_out,
                                           BisectionMethod method,
                                           KokkosMemoryStack &mem_stack,
                                           DeviceExecutionSpace &exec_space,
                                           u64 seed) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "recursive_bisection", "bisect_single_coarse_graph");

        if (g.n <= 1) {
            Kokkos::deep_copy(exec_space, part_out, 0);
            return;
        }

        GraphBatch batch;
        init_GraphBatch(batch, g, 1, mem_stack);

        UnmanagedDeviceU8 active_mask((u8 *) get_chunk_back(mem_stack, sizeof(u8) * 1), 1);
        UnmanagedDevicePartition active_graph_ids((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * 1), 1);
        UnmanagedDeviceU32 current_targets_dev((u32 *) get_chunk_back(mem_stack, sizeof(u32) * 1), 1);

        Kokkos::parallel_for("init_batch_single", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
            batch.batch_ns(0) = g.n;
            batch.batch_offsets(0) = 0;
            batch.batch_ms(0) = g.m;
            batch.batch_weights(0) = g.g_weight;
            active_mask(0) = 1;
            active_graph_ids(0) = 0;
            current_targets_dev(0) = (left_k & 0xFFFF) | (right_k << 16);
        });

        u64 n_bytes_weights = round_up_64(g.n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(g.n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(g.m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(g.m) * sizeof(vertex_t);
        u8 *base_ptr = batch.graph_memory.data();
        weight_t *sub_g_weights = (weight_t *) base_ptr;
        u32 *sub_g_neighborhood = (u32 *) (base_ptr + n_bytes_weights);
        vertex_t *sub_g_edges_u = (vertex_t *) ((u8 *) sub_g_neighborhood + n_bytes_neighborhood);
        vertex_t *sub_g_edges_v = (vertex_t *) ((u8 *) sub_g_edges_u + n_bytes_edges_u);
        weight_t *sub_g_edges_w = (weight_t *) ((u8 *) sub_g_edges_v + n_bytes_edges_v);

        auto gw = g.weights;
        auto g_beg = g.edge_begin;
        auto g_end = g.edge_end;
        auto g_eu = g.edges_u;
        auto g_ev = g.edges_v;
        auto g_ew = g.edges_w;
        vertex_t gn = g.n;
        vertex_t gm = g.m;
        bool uvw = g.uniform_vertex_weights;
        bool uew = g.uniform_edge_weights;

        auto d_gids_ptr = batch.get_global_ids_ptr(0);
        vertex_t total_coarse_m = 0;
        Kokkos::parallel_scan("scan_coarse_csr", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u, u32 &running, bool final) {
            u32 deg = g_end(u) - g_beg(u);
            if (final) {
                sub_g_neighborhood[u] = running;
                sub_g_weights[u] = uvw ? 1 : gw(u);
                d_gids_ptr[u] = u;
            }
            running += deg;
        }, total_coarse_m);

        auto d_b_ms = batch.batch_ms;
        Kokkos::parallel_for("set_last_offset", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
            sub_g_neighborhood[gn] = total_coarse_m;
            d_b_ms(0) = total_coarse_m;
        });

        // Copy edges compactly without any padded SENTINELs
        Kokkos::parallel_for("copy_edges_compact", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            u32 dst_e = sub_g_neighborhood[u];
            u32 start = g_beg(u);
            u32 limit = g_end(u);
            for (u32 e = start; e < limit; ++e) {
                vertex_t v = g_ev(e);
                if (v == SENTINEL) break;
                sub_g_edges_u[dst_e] = u;
                sub_g_edges_v[dst_e] = v;
                sub_g_edges_w[dst_e] = uew ? 1 : g_ew(e);
                dst_e++;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        dispatch_batched_bisection(method, uvw, uew, batch, active_mask, active_graph_ids, 1, current_targets_dev, lmax_global, mem_stack, exec_space, seed);
        KOKKOS_PROFILE_FENCE(exec_space);

        Kokkos::parallel_for("copy_part_out", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t *part_ptr = batch.get_partition_ptr(0);
            part_out(u) = part_ptr[u];
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        pop_back(mem_stack); // current_targets_dev
        pop_back(mem_stack); // active_graph_ids
        pop_back(mem_stack); // active_mask
        free_GraphBatch(batch, mem_stack);
        KOKKOS_PROFILE_FENCE(exec_space);
    }

    /**
     * Recursive multilevel bisection step on a subgraph `g_curr`.
     */
    inline void recursive_bisection_internal(SmallGraph &g_curr,
                                             UnmanagedDeviceVertex &curr_global_ids,
                                             partition_t k_sub,
                                             partition_t b_offset,
                                             weight_t lmax_global,
                                             f64 imbalance,
                                             weight_t total_g_weight,
                                             partition_t total_k,
                                             u32 threshold,
                                             u64 seed,
                                             Partition &global_partition,
                                             BisectionMethod bisection_method,
                                             KokkosMemoryStack &mem_stack,
                                             DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "recursive_bisection", "internal_step");

        if (g_curr.n == 0) return;

        // Base case: 1 block targeted or small enough that vertices can simply be assigned
        if (k_sub <= 1) {
            auto p_map = global_partition.map;
            auto p_bweights = global_partition.bweights;
            auto weights = g_curr.weights;
            bool uvw = g_curr.uniform_vertex_weights;
            vertex_t n = g_curr.n;

            Kokkos::parallel_for("assign_leaf_block", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t g_u = curr_global_ids(u);
                p_map(g_u) = b_offset;
                weight_t w = uvw ? 1 : weights(u);
                Kokkos::atomic_add(&p_bweights(b_offset), w);
            });
            KOKKOS_PROFILE_FENCE(exec_space);
            return;
        }

        if (g_curr.n <= k_sub) {
            auto p_map = global_partition.map;
            auto p_bweights = global_partition.bweights;
            auto weights = g_curr.weights;
            bool uvw = g_curr.uniform_vertex_weights;
            vertex_t n = g_curr.n;

            Kokkos::parallel_for("assign_few_vertices", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t g_u = curr_global_ids(u);
                partition_t target_b = b_offset + u;
                p_map(g_u) = target_b;
                weight_t w = uvw ? 1 : weights(u);
                Kokkos::atomic_add(&p_bweights(target_b), w);
            });
            KOKKOS_PROFILE_FENCE(exec_space);
            return;
        }

        partition_t k1 = k_sub / 2;
        partition_t k2 = k_sub - k1;

        // Allocate 2-way partition for g_curr on front of mem_stack
        Partition bisect_part = initialize_partition(g_curr.n, 2, lmax_global, mem_stack, exec_space);

        // --- Phase 1: Coarsening g_curr ---
        std::vector<SmallGraph> graphs = {g_curr};
        std::vector<Mapping> mappings;

        while (graphs.back().n > threshold) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "recursive_bisection", "coarsening");
            mappings.push_back(dispatch_heavy_edge_matching_small_get_mapping(graphs.back(), bisect_part, lmax_global, mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);

            HEIPA_PROFILE_SCOPE("initial_partitioning", "recursive_bisection", "contraction");
            graphs.push_back(dispatch_from_Graph_Mapping_small<true>(graphs.back(), mappings.back(), mem_stack, exec_space));
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // --- Phase 2: Initial Bisection of coarsest graph ---
        bisect_single_coarse_graph(graphs.back(), k1, k2, lmax_global, bisect_part.map, bisection_method, mem_stack, exec_space, seed);

        // Calculate initial block weights for coarsest partition
        Kokkos::deep_copy(exec_space, bisect_part.bweights, 0);
        auto bp_map = bisect_part.map;
        auto bp_bw = bisect_part.bweights;
        auto c_w = graphs.back().weights;
        bool c_uvw = graphs.back().uniform_vertex_weights;
        vertex_t cn = graphs.back().n;
        Kokkos::parallel_for("calc_bweights_coarse", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, cn), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t p = bp_map(u);
            weight_t w = c_uvw ? 1 : c_w(u);
            Kokkos::atomic_add(&bp_bw(p), w);
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // Allocate 2-way target scratch for FM refinement
        UnmanagedDeviceU32 targets_dev_2way((u32 *) get_chunk_back(mem_stack, sizeof(u32) * 2), 2);
        Kokkos::parallel_for("init_targets_2way", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 2), KOKKOS_LAMBDA(const int i) {
            if (i == 0) targets_dev_2way(0) = (k1 & 0xFFFF);
            else targets_dev_2way(1) = (k2 & 0xFFFF);
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // Refine initial bisection on coarsest graph
        cpu_fm_refine(graphs.back(), 2, bisect_part, targets_dev_2way, g_curr.g_weight, imbalance, exec_space);

        // --- Phase 3: Uncoarsening and FM refinement back to g_curr ---
        while (!mappings.empty()) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "recursive_bisection", "uncoarsening");
            uncontract(bisect_part, mappings.back(), exec_space);

            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();

            // Recalculate block weights at this uncontracted level
            Kokkos::deep_copy(exec_space, bisect_part.bweights, 0);
            auto cur_bp_map = bisect_part.map;
            auto cur_bp_bw = bisect_part.bweights;
            auto cur_w = graphs.back().weights;
            bool cur_uvw = graphs.back().uniform_vertex_weights;
            vertex_t cur_n = graphs.back().n;
            Kokkos::parallel_for("calc_bweights_uncoarse", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, cur_n), KOKKOS_LAMBDA(const vertex_t u) {
                partition_t p = cur_bp_map(u);
                weight_t w = cur_uvw ? 1 : cur_w(u);
                Kokkos::atomic_add(&cur_bp_bw(p), w);
            });
            KOKKOS_PROFILE_FENCE(exec_space);

            // Refine 2-way partition at this uncontracted level
            cpu_fm_refine(graphs.back(), 2, bisect_part, targets_dev_2way, g_curr.g_weight, imbalance, exec_space);
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        pop_back(mem_stack); // targets_dev_2way
        KOKKOS_PROFILE_FENCE(exec_space);

        // --- Phase 4: Extract induced subgraphs and recurse ---
        // Extract G1 (part 0)
        SmallGraph G1;
        UnmanagedDeviceVertex G1_global_ids;
        extract_induced_subgraph(g_curr, curr_global_ids, bisect_part.map, 0, G1, G1_global_ids, mem_stack, exec_space);

        // Recurse on G1 with k1 blocks starting at b_offset
        recursive_bisection_internal(G1, G1_global_ids, k1, b_offset,
                                     lmax_global, imbalance, total_g_weight, total_k,
                                     threshold, seed + 1, global_partition, bisection_method,
                                     mem_stack, exec_space);

        // Free G1 from mem_stack
        if (G1.n > 0) {
            free_induced_subgraph(G1, mem_stack);
        }

        // Extract G2 (part 1)
        SmallGraph G2;
        UnmanagedDeviceVertex G2_global_ids;
        extract_induced_subgraph(g_curr, curr_global_ids, bisect_part.map, 1, G2, G2_global_ids, mem_stack, exec_space);

        // Recurse on G2 with k2 blocks starting at b_offset + k1
        recursive_bisection_internal(G2, G2_global_ids, k2, b_offset + k1,
                                     lmax_global, imbalance, total_g_weight, total_k,
                                     threshold, seed + 2, global_partition, bisection_method,
                                     mem_stack, exec_space);

        // Free G2 from mem_stack
        if (G2.n > 0) {
            free_induced_subgraph(G2, mem_stack);
        }

        // Free bisect_part from mem_stack
        free_partition(bisect_part, mem_stack);
        KOKKOS_PROFILE_FENCE(exec_space);
    }

    /**
     * Top-level entry point for multilevel recursive bisection.
     */
    inline void recursive_bisection(Graph &g,
                                    partition_t k,
                                    f64 imbalance,
                                    u64 seed,
                                    u32 threshold,
                                    Partition &partition,
                                    KokkosMemoryStack &mem_stack,
                                    DeviceExecutionSpace &exec_space,
                                    BisectionMethod bisection_method = BisectionMethod::GRASP) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "recursive_bisection", "recursive_bisection");

        if (g.n == 0 || k == 0) return;

        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        // Convert root Graph to SmallGraph
        SmallGraph root_g = from_Graph_to_SmallGraph(g, mem_stack, exec_space);

        // Allocate root global vertex IDs
        UnmanagedDeviceVertex root_global_ids((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * root_g.n), root_g.n);
        Kokkos::parallel_for("init_root_gids", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, root_g.n), KOKKOS_LAMBDA(const vertex_t u) {
            root_global_ids(u) = u;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // Reset global partition block weights
        Kokkos::deep_copy(exec_space, partition.bweights, 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        recursive_bisection_internal(root_g,
                                     root_global_ids,
                                     k,
                                     0,
                                     lmax_global,
                                     imbalance,
                                     g.g_weight,
                                     k,
                                     threshold,
                                     seed,
                                     partition,
                                     bisection_method,
                                     mem_stack,
                                     exec_space);

        // Free root global IDs and root graph
        pop_front(mem_stack); // root_global_ids
        free_graph(root_g, mem_stack);
        KOKKOS_PROFILE_FENCE(exec_space);
    }
} // namespace GPU_HeiPa

#endif // GPU_HEIPA_RECURSIVE_BISECTION_H
