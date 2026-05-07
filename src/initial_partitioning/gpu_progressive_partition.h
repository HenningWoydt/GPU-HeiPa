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
#include <limits>

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/two_hop_matching.h"
#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/custom_reductions.h"

namespace GPU_HeiPa {
    /**
     * Counts the number of vertices assigned to each block.
     * @param partition The current partition.
     * @param g The graph.
     * @param d_counts Pre-allocated device view to store counts (size k).
     * @param exec_space Execution space.
     */
    inline void count_block_vertices(const Partition &partition,
                                     const Graph &g,
                                     UnmanagedDeviceVertex &d_counts,
                                     DeviceExecutionSpace &exec_space) {
        Kokkos::deep_copy(exec_space, d_counts, 0);
        auto part_map = partition.map;
        Kokkos::parallel_for("count_block_vertices", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = part_map(u);
            Kokkos::atomic_add(&d_counts(b), 1);
        });
    }

    /**
     * Performs a simple bisection by assigning vertices to blocks based on vertex ID mod 2.
     * @param g The graph.
     * @param partition_map The device partition map (size g.n).
     * @param exec_space Execution space.
     */
    inline void bisection_mod2(const Graph &g,
                               UnmanagedDevicePartition &partition_map,
                               DeviceExecutionSpace &exec_space) {
        Kokkos::parallel_for("bisection_mod2", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_map(u) = u % 2;
        });
    }

    struct BestBisectConfig {
        u64 penalty = 0xFFFFFFFFFFFFFFFFULL;
        weight_t cut = 0x7FFFFFFF;
        u64 config = 0;

        KOKKOS_INLINE_FUNCTION BestBisectConfig() = default;
    };

    struct BestBisectReducer {
        using reducer = BestBisectReducer;
        using value_type = BestBisectConfig;
        using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

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

        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }

        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }

        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    /**
     * Brute-force bisection for small graphs.
     * Maps vertex 0 to block 0, and evaluates all 2^(n-1) configurations.
     */
    inline void brute_force_bisect(const Graph &g,
                                   weight_t lmax_left,
                                   weight_t lmax_right,
                                   UnmanagedDevicePartition &partition_map,
                                   DeviceExecutionSpace &exec_space) {
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
                partition_map(0) = 0;
            });
            return;
        }

        if (g.n > 31) {
            std::cerr << "Error: brute_force_bisect called with n=" << g.n << " (max 31)" << std::endl;
            return;
        }

        BestBisectConfig best;
        u64 num_configs = 1ULL << (g.n - 1);

        auto weights = g.weights;
        auto edges_u = g.edges_u;
        auto edges_v = g.edges_v;
        auto edges_w = g.edges_w;
        auto gn = g.n;
        auto gm = g.m;
        bool uvw = g.uniform_vertex_weights;
        bool uew = g.uniform_edge_weights;

        Kokkos::parallel_reduce("brute_force_bisect", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_configs), KOKKOS_LAMBDA(const u64 i, BestBisectConfig &local_best) {
            weight_t wl = uvw ? 1 : weights(0);
            weight_t wr = 0;
            for (vertex_t u = 1; u < gn; ++u) {
                if ((i >> (u - 1)) & 1)
                    wr += uvw ? 1 : weights(u);
                else
                    wl += uvw ? 1 : weights(u);
            }

            u64 p_l = (wl > lmax_left) ? (u64) (wl - lmax_left) : 0;
            u64 p_r = (wr > lmax_right) ? (u64) (wr - lmax_right) : 0;
            u64 penalty = p_l * p_l + p_r * p_r;

            weight_t cut = 0;
            for (u32 j = 0; j < gm; ++j) {
                vertex_t u = edges_u(j);
                vertex_t v = edges_v(j);
                if (u < v) {
                    int part_u = (u == 0) ? 0 : ((i >> (u - 1)) & 1);
                    int part_v = (v == 0) ? 0 : ((i >> (v - 1)) & 1);
                    if (part_u != part_v) cut += uew ? 1 : edges_w(j);
                }
            }

            if (penalty < local_best.penalty) {
                local_best.penalty = penalty;
                local_best.cut = cut;
                local_best.config = i;
            } else if (penalty == local_best.penalty) {
                if (cut < local_best.cut) {
                    local_best.cut = cut;
                    local_best.config = i;
                }
            }
        }, BestBisectReducer(best));

        Kokkos::parallel_for("apply_best_config", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            if (u == 0)
                partition_map(u) = 0;
            else
                partition_map(u) = (best.config >> (u - 1)) & 1;
        });
    }

    /**
     * Extracts a subgraph for a given block.
     * @param g The original graph.
     * @param partition_map The device partition map.
     * @param b The block ID to extract.
     * @param sub_g Output subgraph.
     * @param sub_n_to_o_buffer Pre-allocated device buffer for sub_n_to_o.
     * @param o_to_sub_n_buffer Pre-allocated device buffer for o_to_sub_n.
     * @param sub_n_to_o Output subview of sub_n_to_o_buffer.
     * @param exec_space Execution space.
     */
    inline void extract_block_subgraph(const Graph &g,
                                       const UnmanagedDevicePartition &partition_map,
                                       partition_t b,
                                       Graph &sub_g,
                                       UnmanagedDeviceVertex &sub_n_to_o_buffer,
                                       UnmanagedDeviceVertex &o_to_sub_n_buffer,
                                       UnmanagedDeviceVertex &sub_n_to_o,
                                       DeviceExecutionSpace &exec_space) {
        vertex_t sub_n = 0;
        Kokkos::parallel_reduce("count_sub_n", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
            if (partition_map(u) == b) lsum++;
        }, sub_n);

        if (sub_n == 0) return;

        bigAccumulator acc;
        auto g_weights = g.weights;
        auto g_neigh = g.neighborhood;
        auto g_edges_v = g.edges_v;
        bool uniform_vw = g.uniform_vertex_weights;

        Kokkos::parallel_reduce("count_sub_m_w", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, bigAccumulator &lacc) {
            if (partition_map(u) == b) {
                lacc.weight_0s += uniform_vw ? 1 : g_weights(u);
                for (u32 i = g_neigh(u); i < g_neigh(u + 1); ++i) {
                    vertex_t v = g_edges_v(i);
                    if (partition_map(v) == b) lacc.num_edges_0s++;
                }
            }
        }, acc);

        // We assume sub_g has already been allocated with enough capacity
        sub_g.n = sub_n;
        sub_g.m = acc.num_edges_0s;
        sub_g.g_weight = acc.weight_0s;

        sub_n_to_o = Kokkos::subview(sub_n_to_o_buffer, std::make_pair((vertex_t) 0, sub_n));
        UnmanagedDeviceVertex o_to_sub_n = Kokkos::subview(o_to_sub_n_buffer, std::make_pair((vertex_t) 0, g.n));

        auto sub_weights = sub_g.weights;
        Kokkos::parallel_scan("fill_sub_mapping", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &prefix, bool final) {
            if (partition_map(u) == b) {
                if (final) {
                    sub_n_to_o(prefix) = u;
                    o_to_sub_n(u) = prefix;
                    if (!uniform_vw) sub_weights(prefix) = g_weights(u);
                }
                prefix++;
            }
        });

        auto sub_neigh = sub_g.neighborhood;
        auto sub_edges_v = sub_g.edges_v;
        auto sub_edges_w = sub_g.edges_w;
        auto g_edges_w = g.edges_w;
        bool uniform_ew = g.uniform_edge_weights;

        Kokkos::parallel_for("init_sub_neigh", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) { sub_neigh(0) = 0; });

        Kokkos::parallel_scan("fill_sub_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t sub_u, u32 &prefix, bool final) {
            vertex_t u = sub_n_to_o(sub_u);
            u32 cnt = 0;
            for (u32 i = g_neigh(u); i < g_neigh(u + 1); ++i) {
                vertex_t v = g_edges_v(i);
                if (partition_map(v) == b) {
                    if (final) {
                        sub_edges_v(prefix + cnt) = o_to_sub_n(v);
                        if (!uniform_ew) sub_edges_w(prefix + cnt) = g_edges_w(i);
                    }
                    cnt++;
                }
            }
            if (final) {
                sub_neigh(sub_u + 1) = prefix + cnt;
            }
            prefix += cnt;
        });

        auto sub_edges_u = sub_g.edges_u;
        Kokkos::parallel_for("fill_sub_edges_u", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t sub_u) {
            for (u32 i = sub_neigh(sub_u); i < sub_neigh(sub_u + 1); ++i) {
                sub_edges_u(i) = sub_u;
            }
        });
    }

    /**
     * Performs progressive partitioning by interleaving uncoarsening and bisection.
     */
    inline void gpu_progressive_partition(Graph &g,
                                          const std::vector<partition_t> &hierarchy,
                                          partition_t k,
                                          f64 imbalance,
                                          u64 seed,
                                          u32 threshold,
                                          Partition &partition,
                                          KokkosMemoryStack &mem_stack,
                                          DeviceExecutionSpace &exec_space) {
        // --- 1. COARSENING ---
        std::vector<Graph> graphs = {g};
        std::vector<Mapping> mappings;

        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > threshold) {
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "coarsening");

                mappings.push_back(two_hop_matcher_get_mapping<false, false>(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            }
            //
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "graph_contraction");

                graphs.push_back(from_Graph_Mapping<false, false>(graphs.back(), mappings.back(), mem_stack, exec_space));
            }
            //
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "partition_contraction");

                contract(partition, mappings.back(), exec_space);
            }

            std::cout << "  [Progressive Coarsening] Level " << graphs.size() - 1 << ": n=" << graphs.back().n << std::endl;
        }

        std::cout << "  [Progressive Bisection] Starting loop at level " << graphs.size() - 1 << " (n=" << graphs.back().n << ")" << std::endl;

        UnmanagedDeviceVertex vertex_count = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);
        HostVertex h_counts(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_counts"), k);
        HostWeight h_weights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_weights"), k);

        // Pre-allocate buffers for subgraph extraction and bisection
        Graph sub_g_buf;
        sub_g_buf.uniform_vertex_weights = g.uniform_vertex_weights;
        sub_g_buf.uniform_edge_weights = g.uniform_edge_weights;
        sub_g_buf.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
        sub_g_buf.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        sub_g_buf.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        if (!g.uniform_vertex_weights) sub_g_buf.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.n), g.n);
        if (!g.uniform_edge_weights) sub_g_buf.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.m), g.m);

        UnmanagedDevicePartition sub_part_buf = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * g.n), g.n);
        UnmanagedDeviceVertex sub_n_to_o_buf = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.n), g.n);
        UnmanagedDeviceVertex o_to_sub_n_buf = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);

        // --- Hierarchy and Metadata Setup ---
        u32 num_levels = (u32) hierarchy.size();
        std::vector<partition_t> strides(num_levels);
        strides[0] = 1;
        for (u32 i = 1; i < num_levels; ++i) strides[i] = strides[i - 1] * hierarchy[i - 1];

        struct BlockMeta {
            u32 lvl = 0;
            partition_t fact = 0;
            vertex_t n = 0;
            weight_t weight = 0;
        };
        std::vector<BlockMeta> b_meta(k);
        b_meta[0].lvl = num_levels - 1;
        b_meta[0].fact = hierarchy[num_levels - 1];
        b_meta[0].n = graphs.back().n;
        b_meta[0].weight = graphs.back().g_weight;

        f64 avg_core_weight = (f64) g.g_weight / (f64) k;
        int cl = (int) graphs.size() - 1;

        // Initialize partition to 0
        Kokkos::deep_copy(exec_space, partition.map, 0);

        while (true) {
            bool split_occurred = false;
            Graph &curr_g = graphs[cl];

            for (partition_t b = 0; b < k; ++b) {
                if (b_meta[b].n == 0) continue;
                u32 lvl = b_meta[b].lvl;
                partition_t f = b_meta[b].fact;

                if (f > 1) {
                    bool can_partition = (cl == 0) || (b_meta[b].n > threshold);

                    if (can_partition && b_meta[b].n >= (threshold / 2)) {
                        split_occurred = true;

                        // Bisect factor f into lp, rp
                        partition_t rp = 1 << (u32) std::log2(f - 1);
                        partition_t lp = f - rp;
                        partition_t stride = strides[lvl];
                        partition_t rid = b + lp * stride;

                        weight_t lmax_l = (weight_t) std::ceil((1.0 + imbalance) * avg_core_weight * (lp * stride));
                        weight_t lmax_r = (weight_t) std::ceil((1.0 + imbalance) * avg_core_weight * (rp * stride));

                        std::cout << "      [Lvl " << lvl << "][Block " << b << "] n=" << b_meta[b].n << " f=" << f << " -> Splitting into " << lp << " and " << rp << " parts" << std::endl;

                        UnmanagedDeviceVertex sub_n_to_o;
                        //
                        {
                            ScopedTimer _t_sub("initial_partitioning", "gpu_progressive_partition", "extract_block_subgraph");

                            extract_block_subgraph(curr_g, partition.map, b, sub_g_buf, sub_n_to_o_buf, o_to_sub_n_buf, sub_n_to_o, exec_space);
                        }
                        UnmanagedDevicePartition sub_part = Kokkos::subview(sub_part_buf, std::make_pair((vertex_t) 0, sub_g_buf.n));

                        //
                        {
                            ScopedTimer _t_sub("initial_partitioning", "gpu_progressive_partition", "brute_force_bisect");

                            brute_force_bisect(sub_g_buf, lmax_l, lmax_r, sub_part, exec_space);
                        }
                        //
                        {
                            ScopedTimer _t_sub("initial_partitioning", "gpu_progressive_partition", "update_partition");

                            Kokkos::parallel_for("update_partition", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_g_buf.n), KOKKOS_LAMBDA(const vertex_t i) {
                                if (sub_part(i) == 1) {
                                    partition.map(sub_n_to_o(i)) = rid;
                                }
                            });
                            exec_space.fence();
                        }
                        //
                        {
                            ScopedTimer _t_sub("initial_partitioning", "gpu_progressive_partition", "update_weighs_and_count");

                            recalculate_weights<false>(partition, curr_g, exec_space);
                            count_block_vertices(partition, curr_g, vertex_count, exec_space);
                            Kokkos::deep_copy(exec_space, h_counts, vertex_count);
                            Kokkos::deep_copy(exec_space, h_weights, partition.bweights);
                            exec_space.fence();
                        }

                        // Update metadata for both blocks
                        b_meta[b].fact = lp;
                        b_meta[b].n = h_counts(b);
                        b_meta[b].weight = h_weights(b);

                        b_meta[rid].lvl = lvl;
                        b_meta[rid].fact = rp;
                        b_meta[rid].n = h_counts(rid);
                        b_meta[rid].weight = h_weights(rid);
                    }
                } else {
                    // fact == 1: done with this level, advance if possible
                    if (lvl > 0) {
                        b_meta[b].lvl = lvl - 1;
                        b_meta[b].fact = hierarchy[lvl - 1];
                        split_occurred = true;
                    }
                }
            }

            if (!split_occurred) {
                if (cl > 0) {
                    cl--;
                    std::cout << "  [Progressive Bisection] uncontract " << cl << " (n=" << graphs[cl].n << ")" << std::endl;
                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "uncontraction");
                        uncontract(partition, mappings.back(), exec_space);
                    }
                    {
                        recalculate_weights<false>(partition, graphs[cl], exec_space);
                        count_block_vertices(partition, graphs[cl], vertex_count, exec_space);
                        Kokkos::deep_copy(exec_space, h_counts, vertex_count);
                        Kokkos::deep_copy(exec_space, h_weights, partition.bweights);
                        exec_space.fence();

                        for (partition_t i = 0; i < k; ++i) {
                            b_meta[i].n = h_counts(i);
                            b_meta[i].weight = h_weights(i);
                        }
                    }
                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "free_memory");
                        free_graph(graphs.back(), mem_stack);
                        graphs.pop_back();
                        free_mapping(mappings.back(), mem_stack);
                        mappings.pop_back();
                        exec_space.fence();
                    }
                } else {
                    break;
                }
            }
        }

        // Deallocate buffers
        pop_back(mem_stack); // o_to_sub_n_buf
        pop_front(mem_stack); // sub_n_to_o_buf
        pop_front(mem_stack); // sub_part_buf
        if (!g.uniform_edge_weights) pop_front(mem_stack);
        if (!g.uniform_vertex_weights) pop_front(mem_stack);
        pop_front(mem_stack); // sub_g_buf.edges_u
        pop_front(mem_stack); // sub_g_buf.edges_v
        pop_front(mem_stack); // sub_g_buf.neighborhood
        pop_front(mem_stack); // vertex_count
    }
}

#endif //GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H
