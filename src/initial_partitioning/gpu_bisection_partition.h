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
#include "gpu_progressive_partition.h"
#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/custom_reductions.h"

namespace GPU_HeiPa {
    struct GraphBatch {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;

        UnmanagedDeviceU8 graph_memory;
        UnmanagedDeviceU8 partition_memory;
        UnmanagedDeviceU8 global_ids_memory;
    };

    inline void init_GraphBatch(GraphBatch &batch,
                                Graph &g,
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
        u64 n_bytes_graph_total = k * n_bytes_one_graph;
        batch.graph_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_graph_total), n_bytes_graph_total);

        u64 n_bytes_partition = round_up_64(g.n) * sizeof(partition_t);
        u64 n_bytes_partition_total = k * n_bytes_partition;
        batch.partition_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_partition_total), n_bytes_partition_total);

        u64 n_bytes_global_ids = round_up_64(g.n) * sizeof(vertex_t);
        u64 n_bytes_global_ids_total = k * n_bytes_global_ids;
        batch.global_ids_memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * n_bytes_global_ids_total), n_bytes_global_ids_total);
    }

    inline void free_GraphBatch(GraphBatch &batch,
                                KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
    }

    inline Graph get_Graph(GraphBatch &batch,
                           partition_t id) {
        Graph graph;
        graph.uniform_edge_weights = false;
        graph.uniform_vertex_weights = false;
        graph.n_pops = 0; // Not used as it's unmanaged from batch

        u64 n_bytes_weights = round_up_64(batch.n) * sizeof(weight_t);
        u64 n_bytes_neighborhood = round_up_64(batch.n + 1) * sizeof(u32);
        u64 n_bytes_edges_u = round_up_64(batch.m) * sizeof(vertex_t);
        u64 n_bytes_edges_v = round_up_64(batch.m) * sizeof(vertex_t);
        u64 n_bytes_edges_w = round_up_64(batch.m) * sizeof(weight_t);
        u64 n_bytes_one_graph = n_bytes_weights + n_bytes_neighborhood + n_bytes_edges_u + n_bytes_edges_v + n_bytes_edges_w;

        u64 memory_offset = id * n_bytes_one_graph;
        u8 *base = batch.graph_memory.data() + memory_offset;

        graph.weights = UnmanagedDeviceWeight((weight_t *) base, batch.n);
        base += n_bytes_weights;
        graph.neighborhood = UnmanagedDeviceU32((u32 *) base, batch.n + 1);
        base += n_bytes_neighborhood;
        graph.edges_u = UnmanagedDeviceVertex((vertex_t *) base, batch.m);
        base += n_bytes_edges_u;
        graph.edges_v = UnmanagedDeviceVertex((vertex_t *) base, batch.m);
        base += n_bytes_edges_v;
        graph.edges_w = UnmanagedDeviceWeight((weight_t *) base, batch.m);

        return graph;
    }

    inline UnmanagedDeviceVertex get_global_ids(GraphBatch &batch,
                                                partition_t id) {
        u64 n_bytes_global_ids = round_up_64(batch.n) * sizeof(vertex_t);
        u64 memory_offset = id * n_bytes_global_ids;
        u8 *base = batch.global_ids_memory.data() + memory_offset;
        return UnmanagedDeviceVertex((vertex_t *) base, batch.n);
    }

    inline UnmanagedDevicePartition get_partition(GraphBatch &batch,
                                                  partition_t id) {
        u64 n_bytes_partition = round_up_64(batch.n) * sizeof(partition_t);

        u64 memory_offset = id * n_bytes_partition;
        u8 *base = batch.partition_memory.data() + memory_offset;

        return UnmanagedDevicePartition((partition_t *) base, batch.n);
    }

    inline void extract_subgraph(const Graph &g,
                                 Graph &sub_g,
                                 UnmanagedDeviceVertex &global_ids,
                                 partition_t id,
                                 Partition &partition,
                                 KokkosMemoryStack &mem_stack,
                                 DeviceExecutionSpace &exec_space) {
        ScopedTimer _t("initial_partitioning", "gpu_bisection_partition", "extract_subgraph");

        auto map = partition.map;
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);

        // 1. Assign local IDs to vertices in the block and copy weights
        vertex_t sub_n = 0;
        Kokkos::parallel_scan("local_id_assignment", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &running, bool final) {
            bool in_block = (map(u) == id);
            if (final && in_block) {
                local_ids(u) = running;
                global_ids(running) = u;
                sub_g.weights(running) = g.uniform_vertex_weights ? 1 : g.weights(u);
            }
            running += in_block;
        }, sub_n);
        sub_g.n = sub_n;

        // 2. Count local edges
        UnmanagedDeviceVertex degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * sub_n), sub_n);

        Kokkos::parallel_for("count_local_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local) {
            vertex_t u_global = global_ids(u_local);
            u32 count = 0;
            for (u32 e = g.neighborhood(u_global); e < g.neighborhood(u_global + 1); ++e) {
                vertex_t v = g.edges_v(e);
                if (map(v) == id) {
                    count++;
                }
            }
            degree(u_local) = count;
        });

        // 3. Prefix sum to form neighborhood
        u32 sub_m = 0;
        Kokkos::parallel_scan("prefix_sum_neighborhood", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local, u32 &running, bool final) {
            u32 deg = degree(u_local);
            if (final) {
                sub_g.neighborhood(u_local) = running;
            }
            running += deg;
        }, sub_m);

        Kokkos::parallel_for("set_last_neighborhood", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
            sub_g.neighborhood(sub_n) = sub_m;
        });
        sub_g.m = sub_m;

        // 4. Populate edges
        Kokkos::parallel_for("populate_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local) {
            vertex_t u_global = global_ids(u_local);
            u32 edge_idx = sub_g.neighborhood(u_local);
            for (u32 e = g.neighborhood(u_global); e < g.neighborhood(u_global + 1); ++e) {
                vertex_t v_global = g.edges_v(e);
                if (map(v_global) == id) {
                    sub_g.edges_u(edge_idx) = u_local;
                    sub_g.edges_v(edge_idx) = local_ids(v_global);
                    sub_g.edges_w(edge_idx) = g.uniform_edge_weights ? 1 : g.edges_w(e);
                    edge_idx++;
                }
            }
        });
        exec_space.fence();

        weight_t sub_weight = 0;

        Kokkos::parallel_reduce(
            "sum_subgraph_weight",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n),
            KOKKOS_LAMBDA(const vertex_t u_local, weight_t &local_sum) {
                local_sum += sub_g.weights(u_local);
            },
            sub_weight
        );

        sub_g.g_weight = sub_weight;

        pop_back(mem_stack); // degree
        pop_back(mem_stack); // local_ids
    }

    struct HierarchyManager {
        std::vector<partition_t> hierarchy;
        partition_t total_k;
        std::vector<partition_t> unit_sizes;

        std::vector<u8> active;
        std::vector<u32> curr_level;
        std::vector<u32> curr_load;

        size_t max_blocks;
    };

    inline void init_HierarchyManager(HierarchyManager &manager,
                                      const std::vector<partition_t> &t_hierarchy,
                                      size_t t_k) {
        manager.hierarchy = t_hierarchy;
        manager.max_blocks = t_k;
        size_t num_levels = manager.hierarchy.size();

        manager.unit_sizes.assign(num_levels, 1);
        size_t current = 1;
        for (size_t i = 0; i < num_levels; ++i) {
            manager.unit_sizes[i] = current;
            current *= manager.hierarchy[i];
        }
        manager.total_k = current;

        manager.active.assign(manager.max_blocks, 0);
        manager.curr_level.assign(manager.max_blocks, 0);
        manager.curr_load.assign(manager.max_blocks, 0);

        // Initial state: start at the top of the hierarchy (last index)
        manager.active[0] = 1;
        manager.curr_level[0] = num_levels - 1;
        manager.curr_load[0] = manager.hierarchy.back();
    }

    inline void split_into(const HierarchyManager &manager,
                           partition_t id,
                           partition_t &left_k,
                           partition_t &right_k) {
        u32 level = manager.curr_level[id];
        u32 load = manager.curr_load[id];

        partition_t p = 1;
        while (p * 2 < load) p *= 2;

        partition_t left_load = p;
        partition_t right_load = load - p;

        left_k = left_load * manager.unit_sizes[level];
        right_k = right_load * manager.unit_sizes[level];
    }

    inline void split(HierarchyManager &manager,
                      partition_t id,
                      partition_t left_k,
                      partition_t right_k) {
        u32 level = manager.curr_level[id];

        partition_t left_id = id;
        partition_t right_id = id + left_k;

        if (right_id >= manager.max_blocks) {
            throw std::runtime_error("max_blocks exceeded");
        }

        // Left child
        manager.active[left_id] = 1;
        manager.curr_level[left_id] = level;
        manager.curr_load[left_id] = left_k / manager.unit_sizes[level];

        // Right child
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

    inline void recalculate_block_weights(const Graph &g,
                                          const UnmanagedDevicePartition &map,
                                          UnmanagedDeviceWeight &bweights,
                                          DeviceExecutionSpace &exec_space) {
        Kokkos::deep_copy(exec_space, bweights, 0);
        Kokkos::parallel_for("recalculate_block_weights", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            Kokkos::atomic_add(&bweights(id), g.uniform_vertex_weights ? 1 : g.weights(u));
        });
        exec_space.fence();
    }

    inline void calculate_block_sizes(const Graph &g,
                                      const Mapping *mapping,
                                      const UnmanagedDevicePartition &map,
                                      UnmanagedDeviceVertex &bsizes,
                                      UnmanagedDeviceVertex &projected_sizes,
                                      DeviceExecutionSpace &exec_space) {
        Kokkos::deep_copy(exec_space, bsizes, 0);
        Kokkos::parallel_for("calculate_block_sizes", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t id = map(u);
            Kokkos::atomic_add(&bsizes(id), 1);
        });

        if (mapping != nullptr) {
            Kokkos::deep_copy(exec_space, projected_sizes, 0);
            Kokkos::parallel_for("calculate_projected_block_sizes", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, mapping->old_n), KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t new_v = mapping->mapping(u);
                partition_t id = map(new_v);
                Kokkos::atomic_add(&projected_sizes(id), 1);
            });
        }
        exec_space.fence();
    }

    inline void bisect(Graph &g,
                       weight_t lmax_1,
                       weight_t lmax_2,
                       UnmanagedDevicePartition &partition,
                       DeviceExecutionSpace &exec_space) {
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
                partition(0) = 0;
            });
            exec_space.fence();
            return;
        }

        const vertex_t last = g.n - 1;
        const u64 num_configs = 1ULL << last;

        BestBisectConfig best_config;
        BestBisectReducer reducer(best_config);

        Kokkos::parallel_reduce("brute_force_bisect", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_configs), KOKKOS_LAMBDA(const u64 gray_config, BestBisectConfig &local_best) {
            weight_t wr = 0;
            for (vertex_t u = 0; u < last; ++u) {
                if ((gray_config >> u) & 1ULL) {
                    wr += g.uniform_vertex_weights ? 1 : g.weights(u);
                }
            }

            weight_t cut = 0;
            for (u32 e = 0; e < g.m; ++e) {
                const vertex_t u = g.edges_u(e);
                const vertex_t v = g.edges_v(e);
                if (u < v) {
                    // Only count each undirected edge once
                    const u64 pu = (gray_config >> u) & 1ULL;
                    const u64 pv = (gray_config >> v) & 1ULL;
                    if (pu != pv) {
                        cut += g.uniform_edge_weights ? 1 : g.edges_w(e);
                    }
                }
            }

            const weight_t wl = g.g_weight - wr;
            const u64 p_l = wl > lmax_1 ? (u64) (wl - lmax_1) : 0;
            const u64 p_r = wr > lmax_2 ? (u64) (wr - lmax_2) : 0;
            u64 penalty = p_l * p_l + p_r * p_r;

            if (wl == 0 || wr == 0) {
                // Massive penalty to strictly prohibit empty partitions
                penalty += 1000000000000ULL;
            }

            if (penalty < local_best.penalty || (penalty == local_best.penalty && cut < local_best.cut)) {
                local_best.penalty = penalty;
                local_best.cut = cut;
                local_best.config = gray_config;
            }
        }, reducer);
        exec_space.fence();

        Kokkos::parallel_for("apply_best_config", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition(u) = (partition_t) ((best_config.config >> u) & 1ULL);
        });
        exec_space.fence();
    }

    inline void gpu_bisect_partition(Graph &g,
                                     const std::vector<partition_t> &hierarchy,
                                     partition_t k,
                                     f64 imbalance,
                                     u64 seed,
                                     u32 threshold,
                                     Partition &partition,
                                     KokkosMemoryStack &mem_stack,
                                     DeviceExecutionSpace &exec_space) {
        // allocate all memory up front
        GraphBatch batch;
        init_GraphBatch(batch, g, k, mem_stack);

        HierarchyManager manager;
        init_HierarchyManager(manager, hierarchy, 2 * k);

        // --- coarsening phase ---
        std::vector<Graph> graphs = {g};
        std::vector<Mapping> mappings;

        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > threshold) {
            assert_state_pre_partition(graphs.back(), exec_space);

            // coarsen
            mappings.push_back(two_hop_matcher_get_mapping<false, false>(graphs.back(), partition, lmax_global, mem_stack, exec_space));

            // contract graph
            graphs.push_back(from_Graph_Mapping<false, false>(graphs.back(), mappings.back(), mem_stack, exec_space));

            // contract mapping
            contract(partition, mappings.back(), exec_space);

            assert_coarsening(graphs[graphs.size() - 2], graphs.back(), mappings.back(), exec_space);
            assert_state_pre_partition(graphs.back(), exec_space);
        }

        // --- initial partitioning phase ---
        {
            partition_t l_k, r_k;
            split_into(manager, 0, l_k, r_k);

            UnmanagedDevicePartition temp_partition = get_partition(batch, 0);

            bisect(graphs.back(), l_k * lmax_global, r_k * lmax_global, temp_partition, exec_space);

            partition_t left_id = 0;
            partition_t right_id = l_k;
            split(manager, 0, l_k, r_k);

            // update the partition using the hierarchy manager's stride
            partition_t l_stride = left_id;
            partition_t r_stride = right_id;
            auto map = partition.map;
            Kokkos::parallel_for("update_partition_initial", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graphs.back().n), KOKKOS_LAMBDA(const vertex_t u) {
                map(u) = (temp_partition(u) == 0) ? l_stride : r_stride;
            });
            exec_space.fence();

            recalculate_block_weights(graphs.back(), map, partition.bweights, exec_space);
        }

        // --- uncontraction & extraction phase ---
        while (true) {
            bool do_extract = true;
            while (do_extract) {
                do_extract = false;

                UnmanagedDeviceVertex bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * manager.max_blocks), manager.max_blocks);
                UnmanagedDeviceVertex projected_bsizes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * manager.max_blocks), manager.max_blocks);

                const Mapping *mapping_ptr = mappings.empty() ? nullptr : &mappings.back();
                calculate_block_sizes(graphs.back(), mapping_ptr, partition.map, bsizes, projected_bsizes, exec_space);

                HostVertex h_bsizes("h_bsizes", manager.max_blocks);
                HostVertex h_projected_bsizes("h_projected_bsizes", manager.max_blocks);
                Kokkos::deep_copy(exec_space, h_bsizes, bsizes);
                if (!mappings.empty()) {
                    Kokkos::deep_copy(exec_space, h_projected_bsizes, projected_bsizes);
                }
                exec_space.fence();

                for (partition_t id = 0; id < manager.max_blocks; id++) {
                    if (manager.active[id]) {
                        if (manager.curr_load[id] > 1) {
                            partition_t l_k, r_k;
                            split_into(manager, id, l_k, r_k);

                            Graph sub_g = get_Graph(batch, id);
                            UnmanagedDeviceVertex g_ids = get_global_ids(batch, id);

                            vertex_t current_n = h_bsizes(id);
                            if (current_n > 0) {
                                extract_subgraph(graphs.back(), sub_g, g_ids, id, partition, mem_stack, exec_space);
                                assert_state_pre_partition(sub_g, exec_space);
                            } else {
                                sub_g.n = 0;
                                sub_g.m = 0;
                            }

                            vertex_t projected_n = mappings.empty() ? h_bsizes(id) : h_projected_bsizes(id);

                            if ((mappings.empty() || projected_n > threshold) && sub_g.n > 0) {
                                UnmanagedDevicePartition t_partition = get_partition(batch, id);
                                bisect(sub_g, l_k * lmax_global, r_k * lmax_global, t_partition, exec_space);

                                partition_t left_id = id;
                                partition_t right_id = id + l_k;
                                split(manager, id, l_k, r_k);

                                // update the partition
                                partition_t l_stride = left_id;
                                partition_t r_stride = right_id;
                                auto map = partition.map;
                                vertex_t sub_n = sub_g.n;
                                Kokkos::parallel_for("update_partition_loop", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub_n), KOKKOS_LAMBDA(const vertex_t u_local) {
                                    vertex_t u_global = g_ids(u_local);
                                    map(u_global) = (t_partition(u_local) == 0) ? l_stride : r_stride;
                                });
                                exec_space.fence();

                                recalculate_block_weights(graphs.back(), map, partition.bweights, exec_space);
                                do_extract = true;
                            }
                        } else if (manager.curr_level[id] > 0) {
                            descend(manager, id);
                            do_extract = true;
                        }
                    }
                }

                pop_back(mem_stack); // projected_bsizes
                pop_back(mem_stack); // bsizes

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

        // deallocate batch
        free_GraphBatch(batch, mem_stack);
    }
}

#endif //GPU_HEIPA_GPU_BISECTION_PARTITION_H
