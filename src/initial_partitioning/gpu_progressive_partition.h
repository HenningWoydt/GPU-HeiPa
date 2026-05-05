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

#ifndef GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H
#define GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H

#include <vector>
#include <cmath>
#include <iostream>

#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/two_hop_matching.h"
#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "gpu_initial_partition.h"

namespace GPU_HeiPa {

    /**
     * Extracts a single block from the current graph into a new CSR sub-graph.
     */
    inline void extract_single_block_gpu(const Graph &g,
                                         const UnmanagedDevicePartition &part,
                                         partition_t target_b,
                                         KokkosMemoryStack &mem_stack,
                                         Graph &sub_g,
                                         UnmanagedDeviceVertex &sub_n2o,
                                         DeviceExecutionSpace &exec_space) {
        vertex_t n = g.n;
        DeviceVertex rename("rename", n);
        vertex_t sub_n = 0;
        
        // 1. Scan to find nodes in block and their new IDs
        Kokkos::parallel_scan("RenameSub", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &pref, bool final) {
            if (part(u) == target_b) {
                if (final) rename(u) = pref;
                pref++;
            }
        }, sub_n);

        if (sub_n == 0) {
            sub_g.n = 0; return;
        }

        // 2. Build local-to-original mapping
        sub_n2o = UnmanagedDeviceVertex((vertex_t*)get_chunk_front(mem_stack, sizeof(vertex_t) * sub_n), sub_n);
        Kokkos::parallel_for("MapN2O", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            if (part(u) == target_b) sub_n2o(rename(u)) = u;
        });

        // 3. Count internal edges
        UnmanagedDeviceU32 local_degrees((u32 *) get_chunk_front(mem_stack, sizeof(u32) * sub_n), sub_n);
        Kokkos::parallel_for("CountEdges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_sub) {
            vertex_t u = sub_n2o(u_sub);
            u32 deg = 0;
            for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                if (part(g.edges_v(i)) == target_b) deg++;
            }
            local_degrees(u_sub) = deg;
        });

        vertex_t sub_m = 0;
        Kokkos::parallel_reduce("SumM", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t i, vertex_t &update) {
            update += local_degrees(i);
        }, sub_m);

        weight_t sub_w = 0;
        if (g.uniform_vertex_weights) sub_w = sub_n;
        else {
            Kokkos::parallel_reduce("SumW", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t i, weight_t &update) {
                update += g.weights(sub_n2o(i));
            }, sub_w);
        }

        // 4. Allocate sub-graph CSR
        sub_g = make_graph(sub_n, sub_m, sub_w, g.uniform_vertex_weights, g.uniform_edge_weights, mem_stack);
        
        // 5. Build offsets and fill CSR
        Kokkos::parallel_scan("SubOffsets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n + 1), KOKKOS_LAMBDA(const u32 i, u32 &running, bool final) {
            if (final) sub_g.neighborhood(i) = running;
            if (i < sub_n) running += local_degrees(i);
        });

        Kokkos::parallel_for("FillEdges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_sub) {
            vertex_t u = sub_n2o(u_sub);
            u32 curr = sub_g.neighborhood(u_sub);
            for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                vertex_t v = g.edges_v(i);
                if (part(v) == target_b) {
                    sub_g.edges_v(curr) = rename(v);
                    sub_g.edges_u(curr) = u_sub;
                    if (!g.uniform_edge_weights) sub_g.edges_w(curr) = g.edges_w(i);
                    curr++;
                }
            }
            if (!g.uniform_vertex_weights) sub_g.weights(u_sub) = g.weights(u);
        });

        pop_front(mem_stack); // local_degrees
    }

    /**
     * Performs progressive partitioning by interleaving uncoarsening and bisection.
     */
    inline void gpu_progressive_partition(const Graph &g,
                                          const std::vector<partition_t> &hierarchy,
                                          partition_t k,
                                          f64 imbalance,
                                          u64 seed,
                                          u32 threshold,
                                          Partition &partition,
                                          KokkosMemoryStack &mem_stack,
                                          DeviceExecutionSpace &exec_space) {
        ScopedTimer _t_total("initial_partitioning", "gpu_progressive_partition", "total");

        // --- 1. COARSENING ---
        std::vector<Graph> graphs;
        std::vector<Mapping> mappings;
        
        // Upload/Copy initial graph to local stack management
        Graph dev_g = make_graph(g.n, g.m, g.g_weight, g.uniform_vertex_weights, g.uniform_edge_weights, mem_stack);
        Kokkos::deep_copy(exec_space, dev_g.neighborhood, g.neighborhood);
        Kokkos::deep_copy(exec_space, dev_g.edges_v, g.edges_v);
        if (!g.uniform_edge_weights) Kokkos::deep_copy(exec_space, dev_g.edges_w, g.edges_w);
        Kokkos::deep_copy(exec_space, dev_g.edges_u, g.edges_u);
        if (!g.uniform_vertex_weights) Kokkos::deep_copy(exec_space, dev_g.weights, g.weights);
        graphs.push_back(dev_g);

        weight_t lmax_global = (weight_t)std::ceil((1.0 + imbalance) * (f64)g.g_weight / (f64)k);

        while (graphs.back().n > threshold * 2) {
            Partition p_dummy = initialize_partition(graphs.back().n, 2, lmax_global, mem_stack, exec_space);
            Mapping cmap = two_hop_matcher_get_mapping<false, false>(graphs.back(), p_dummy, lmax_global, mem_stack, exec_space);
            free_partition(p_dummy, mem_stack);

            if (cmap.coarse_n >= graphs.back().n * 0.95) {
                free_mapping(cmap, mem_stack);
                break;
            }

            Graph next_g = from_Graph_Mapping<false, false>(graphs.back(), cmap, mem_stack, exec_space);
            mappings.push_back(cmap);
            graphs.push_back(next_g);
        }

        // --- 2. SETUP HIERARCHY STRIDES ---
        u32 num_lvls = hierarchy.size();
        std::vector<partition_t> strides(num_lvls);
        for (u32 l = 0; l < num_lvls; ++l) {
            partition_t s = 1;
            for (u32 j = l + 1; j < num_lvls; ++j) s *= hierarchy[j];
            strides[l] = s;
        }

        // --- 3. PROGRESSIVE BISECTION LOOP ---
        u32 cl = graphs.size() - 1;
        UnmanagedDevicePartition part = partition.map;
        Kokkos::deep_copy(exec_space, part, 0);

        // Pre-calculate fine node counts for lookahead postponement
        std::vector<UnmanagedDeviceVertex> fnc_list;
        for (u32 i = 0; i < mappings.size(); ++i) {
            vertex_t fine_n = (i == 0) ? g.n : mappings[i-1].coarse_n;
            UnmanagedDeviceVertex fnc((vertex_t*)get_chunk_front(mem_stack, sizeof(vertex_t) * mappings[i].coarse_n), mappings[i].coarse_n);
            Kokkos::deep_copy(exec_space, fnc, 0);
            auto mapping_view = mappings[i].mapping;
            Kokkos::parallel_for("FNC", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, fine_n), KOKKOS_LAMBDA(const vertex_t u) {
                Kokkos::atomic_increment(&fnc(mapping_view(u)));
            });
            fnc_list.push_back(fnc);
        }

        // Metadata on Device
        DeviceWeight b_weights("b_weights", k);
        DeviceVertex b_counts("b_counts", k);
        DeviceU32 b_h_lvl("b_h_lvl", k);
        DeviceU32 b_h_fact("b_h_fact", k);
        Kokkos::deep_copy(exec_space, b_weights, 0);
        Kokkos::deep_copy(exec_space, b_counts, 0);
        Kokkos::deep_copy(exec_space, b_h_lvl, 0);
        Kokkos::deep_copy(exec_space, b_h_fact, 0);

        // Initialize block 0 at coarsest level
        Kokkos::parallel_for("InitBlocks", 1, KOKKOS_LAMBDA(int) {
            b_weights(0) = graphs[cl].g_weight;
            b_counts(0) = graphs[cl].n;
            b_h_lvl(0) = 0;
            b_h_fact(0) = (partition_t)hierarchy[0];
        });

        DeviceVertex lookahead_counts("lookahead_counts", k);

        while (true) {
            bool split_any = false;
            Graph &curr_g = graphs[cl];

            // 1. Calculate lookahead counts for all blocks
            if (cl > 0) {
                Kokkos::deep_copy(exec_space, lookahead_counts, 0);
                auto fnc_view = fnc_list[cl-1];
                Kokkos::parallel_for("CalcLookahead", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, curr_g.n), KOKKOS_LAMBDA(const vertex_t i) {
                    Kokkos::atomic_add(&lookahead_counts(part(i)), (vertex_t)fnc_view(i));
                });
            }

            // Sync mirrors for host-driven control loop
            auto h_b_counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_counts);
            auto h_b_h_lvl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_h_lvl);
            auto h_b_h_fact = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_h_fact);
            auto h_lookahead = (cl > 0) ? Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lookahead_counts) : h_b_counts;

            for (partition_t b = 0; b < k; ++b) {
                if (h_b_counts(b) == 0) continue;
                u32 lvl = h_b_h_lvl(b);
                if (lvl >= num_lvls) continue;

                partition_t f = h_b_h_fact(b);
                if (f > 1) {
                    bool can_partition = false;
                    if (cl == 0) can_partition = true;
                    else if (h_lookahead(b) > (vertex_t)threshold || h_b_counts(b) > (vertex_t)threshold) can_partition = true;

                    if (can_partition && h_b_counts(b) >= (vertex_t)(threshold / 2)) {
                        Graph sub_g;
                        UnmanagedDeviceVertex sub_n2o;
                        extract_single_block_gpu(curr_g, part, b, mem_stack, sub_g, sub_n2o, exec_space);

                        if (sub_g.n > 0) {
                            partition_t rp = (1U << (u32)std::floor(std::log2((f64)f - 1.0)));
                            partition_t lp = f - rp;
                            partition_t stride = strides[lvl];
                            
                            weight_t lmax_l = (weight_t)std::ceil((1.0 + imbalance) * (f64)g.g_weight * (f64)(lp * stride) / (f64)k);
                            weight_t lmax_r = (weight_t)std::ceil((1.0 + imbalance) * (f64)g.g_weight * (f64)(rp * stride) / (f64)k);

                            UnmanagedDevicePartition sub_p((partition_t*)get_chunk_front(mem_stack, sizeof(partition_t) * sub_g.n), sub_g.n);
                            UnmanagedDeviceWeight sub_bw((weight_t*)get_chunk_front(mem_stack, sizeof(weight_t) * 2), 2);
                            
                            brute_force_bisect_gpu(sub_g, lmax_l, lmax_r, sub_p, sub_bw, exec_space);
                            
                            partition_t rid = b + lp * stride;
                            
                            Kokkos::parallel_for("UpdatePart", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_g.n), KOKKOS_LAMBDA(const vertex_t i_sub) {
                                part(sub_n2o(i_sub)) = (sub_p(i_sub) == 0) ? b : rid;
                            });

                            auto h_sub_bw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sub_bw);
                            vertex_t sub_n_r = 0;
                            Kokkos::parallel_reduce("CountR", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_g.n), KOKKOS_LAMBDA(const vertex_t i_sub, vertex_t &update) {
                                if (sub_p(i_sub) == 1) update++;
                            }, sub_n_r);

                            Kokkos::parallel_for("UpdateMeta", 1, KOKKOS_LAMBDA(int) {
                                b_weights(b) = h_sub_bw(0);
                                b_weights(rid) = h_sub_bw(1);
                                b_counts(rid) = sub_n_r;
                                b_counts(b) = sub_g.n - sub_n_r;
                                b_h_lvl(rid) = lvl;
                                b_h_fact(b) = lp;
                                b_h_fact(rid) = rp;
                            });
                            exec_space.fence();

                            pop_front(mem_stack); // sub_bw
                            pop_front(mem_stack); // sub_p
                            free_graph(sub_g, mem_stack);
                            pop_front(mem_stack); // sub_n2o
                            split_any = true;
                            
                            // Re-sync host mirrors for next block in loop
                            Kokkos::deep_copy(h_b_counts, b_counts);
                            Kokkos::deep_copy(h_b_h_lvl, b_h_lvl);
                            Kokkos::deep_copy(h_b_h_fact, b_h_fact);
                        }
                    }
                } else if (f == 1 && lvl + 1 < num_lvls) {
                    Kokkos::parallel_for("AdvLvl", 1, KOKKOS_LAMBDA(int) {
                        b_h_lvl(b)++;
                        b_h_fact(b) = (partition_t)hierarchy[b_h_lvl(b)];
                    });
                    split_any = true;
                    Kokkos::deep_copy(h_b_h_lvl, b_h_lvl);
                    Kokkos::deep_copy(h_b_h_fact, b_h_fact);
                }
            }

            if (!split_any) {
                if (cl > 0) {
                    cl--;
                    auto mapping = mappings[cl].mapping;
                    auto cur_n = graphs[cl].n;
                    Kokkos::parallel_for("Project", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, cur_n), KOKKOS_LAMBDA(const vertex_t u) {
                        part(u) = part(mapping(u));
                    });
                    
                    Kokkos::deep_copy(exec_space, b_counts, 0);
                    Kokkos::deep_copy(exec_space, b_weights, 0);
                    auto vw_fine = graphs[cl].weights;
                    bool uniform_vw = graphs[cl].uniform_vertex_weights;
                    Kokkos::parallel_for("RecalcMeta", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, cur_n), KOKKOS_LAMBDA(const vertex_t u) {
                        Kokkos::atomic_increment(&b_counts(part(u)));
                        weight_t uw = uniform_vw ? (weight_t)1 : vw_fine(u);
                        Kokkos::atomic_add(&b_weights(part(u)), uw);
                    });
                    exec_space.fence();
                } else {
                    break;
                }
            }
        }

        // Cleanup
        for (int i = (int)fnc_list.size() - 1; i >= 0; --i) pop_front(mem_stack);
        while (graphs.size() > 1) {
            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();
            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();
        }
        free_graph(graphs[0], mem_stack);
    }
}

#endif //GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H
