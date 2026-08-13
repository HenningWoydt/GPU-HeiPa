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

#ifndef GPU_HEIPA_HIERARCHICAL_MULTISECTION_H
#define GPU_HEIPA_HIERARCHICAL_MULTISECTION_H

#include <vector>
#include <algorithm>

#include "../definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/GPU_HeiPa_solver.h"
#include "../GPU_HeiProMap_configuration.h"

namespace GPU_HeiPa {

    struct HMStats {
        f64 coarsening_ms = 0;
        f64 contraction_ms = 0;
        f64 initial_partitioning_ms = 0;
        f64 uncontraction_ms = 0;
        f64 refinement_ms = 0;
        f64 down_up_load_ms = 0;
        f64 misc_ms = 0;
        f64 subgraph_generation_ms = 0;

        void print(f64 total_duration) const {
            std::cout << "------- Time -------" << std::endl;
            std::cout << "Total solve time  : " << total_duration << std::endl;
            std::cout << "Coarsening        : " << coarsening_ms << std::endl;
            std::cout << "Contraction       : " << contraction_ms << std::endl;
            std::cout << "Init. Part.       : " << initial_partitioning_ms << std::endl;
            std::cout << "Uncontraction     : " << uncontraction_ms << std::endl;
            std::cout << "Refinement        : " << refinement_ms << std::endl;
            std::cout << "Down/Upload       : " << down_up_load_ms << std::endl;
            std::cout << "Misc              : " << misc_ms << std::endl;
            std::cout << "Subgraph Gen.     : " << subgraph_generation_ms << std::endl;
            std::cout << "ALL               : " << coarsening_ms + contraction_ms + initial_partitioning_ms + uncontraction_ms + refinement_ms + down_up_load_ms + misc_ms + subgraph_generation_ms << std::endl;
        }
    };

    inline void gpu_heipa_partition(Graph &device_g,
                                    const Configuration &heipa_config,
                                    HMStats &stats,
                                    UnmanagedDevicePartition &partition,
                                    KokkosMemoryStack &mem_stack,
                                    DeviceExecutionSpace &exec_space) {
        if (heipa_config.k == 1) {
            HEIPA_PROFILE_SCOPE("hm", "recursive", "partition k=1");
            Kokkos::deep_copy(exec_space, partition, 0);
            return;
        }

        Solver solver(device_g, heipa_config, partition, mem_stack, exec_space);
        solver.solve_device_graph(mem_stack);

        stats.coarsening_ms += solver.coarsening_ms;
        stats.contraction_ms += solver.contraction_ms;
        stats.initial_partitioning_ms += solver.initial_partitioning_ms;
        stats.uncontraction_ms += solver.uncontraction_ms;
        stats.refinement_ms += solver.refinement_ms;
        stats.down_up_load_ms += solver.down_up_load_ms;
        stats.misc_ms += solver.misc_ms;
    }

    template<bool uniform_vw, bool uniform_ew>
    inline void recursive_multisection_device(Graph &device_g,
                                              const UnmanagedDeviceVertex &n_to_o, // local->original mapping for this node
                                              const std::vector<partition_t> &hierarchy, // e.g. {k0,k1,k2,...}
                                              u64 level, // start with hierarchy.size()-1 and count down
                                              f64 global_imbalance,
                                              weight_t global_g_weight,
                                              partition_t global_k,
                                              vertex_t global_n,
                                              u64 seed,
                                              bool use_ultra,
                                              const std::vector<partition_t> &index_vec, // as in your host code
                                              const std::vector<partition_t> &k_rem,
                                              std::vector<partition_t> &identifier, // path of ids
                                              HMStats &stats,
                                              UnmanagedDevicePartition &global_partition, // size global_n
                                              KokkosMemoryStack &mem_stack,
                                              DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("hm", "recursive", "allocate");
        const partition_t l = (partition_t) hierarchy.size();
        const partition_t k = hierarchy[level];

        // Allocate temp partition for *this* node
        UnmanagedDevicePartition tmp_part = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * device_g.n), device_g.n);
        KOKKOS_PROFILE_FENCE(exec_space);

        const f64 imb = determine_adaptive_imbalance(global_imbalance, global_g_weight, global_k, device_g.g_weight, k_rem[l - 1 - identifier.size()], l - identifier.size());

        Configuration heipa_config;
        heipa_config.k = k;
        heipa_config.imbalance = imb;
        heipa_config.seed = seed;
        heipa_config.config = use_ultra ? "ultra" : "default";
        heipa_config.verbose_level = 0;
        heipa_config.initial_partitioning = "kway";

        // 1) Partition current device graph into k blocks
        gpu_heipa_partition(device_g, heipa_config, stats, tmp_part, mem_stack, exec_space);

        // 2) Leaf: last split -> write into global_partition
        if (identifier.size() == (size_t) (l - 1)) {
            HEIPA_PROFILE_SCOPE("hm", "recursive", "write_to_global");
            // offset = sum_{i=0..l-2} identifier[i] * index_vec[last-i]
            partition_t offset = 0;
            for (partition_t i = 0; i < l - 1; ++i) { offset += identifier[i] * index_vec[index_vec.size() - 1 - i]; }

            Kokkos::parallel_for("WriteLeafPartition", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, device_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                const vertex_t orig_u = n_to_o(u);
                global_partition(orig_u) = offset + tmp_part(u);
            });
            KOKKOS_PROFILE_FENCE(exec_space);

            pop_front(mem_stack); // tmp_part
            return;
        }

        // 3) Non-leaf: build each child and recurse immediately
        for (partition_t id = 0; id < k; ++id) {
            auto sp_subgraph = get_time_point();
            HEIPA_PROFILE_SCOPE("hm", "recursive", "generate_subgraph");

            // --- First pass: compute sub_n, sub_m, sub_weight for this id
            vertex_t sub_n = 0;
            vertex_t sub_m = 0;
            weight_t sub_weight = 0;

            Kokkos::parallel_reduce("SubN", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, device_g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                if (tmp_part(u) == id) lsum += 1;
            }, sub_n);

            Kokkos::parallel_reduce("SubWeight", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, device_g.n), KOKKOS_LAMBDA(const vertex_t u, weight_t &lsum) {
                if (tmp_part(u) == id) lsum += uniform_vw ? 1 : device_g.weights(u);
            }, sub_weight);

            Kokkos::parallel_reduce("SubM", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, device_g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                if (tmp_part(u) == id) {
                    vertex_t cnt = 0;
                    for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                        const vertex_t v = device_g.edges_v(i);
                        if (tmp_part(v) == id) ++cnt;
                    }
                    lsum += cnt;
                }
            }, sub_m);

            // Empty block => skip
            if (sub_n == 0) {
                continue;
            }

            // --- Allocate child graph + mappings
            Graph child_g = make_graph(sub_n, sub_m, sub_weight, uniform_vw, uniform_ew, mem_stack);
            UnmanagedDeviceVertex child_n_to_o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * sub_n), sub_n);
            UnmanagedDeviceVertex child_o_to_n = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * global_n), global_n);

            // --- Fill translation tables and weights
            Kokkos::parallel_scan("AssignLocalIndex", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, device_g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &prefix, const bool final) {
                if (tmp_part(u) == id) {
                    const vertex_t my_idx = prefix;
                    if (final) {
                        const vertex_t old_u = n_to_o(u);
                        child_o_to_n(old_u) = my_idx;
                        child_n_to_o(my_idx) = old_u;
                        if (!uniform_vw) {
                            child_g.weights(my_idx) = device_g.weights(u);
                        }
                    }
                    prefix += 1;
                }
            });

            // init neighborhood(0)
            Kokkos::parallel_for("InitNeighborhood0", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) { child_g.neighborhood(0) = 0; });

            // --- Fill edges + neighborhood offsets
            Kokkos::parallel_scan("FillEdges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, device_g.n), KOKKOS_LAMBDA(const vertex_t u, u32 &edge_prefix, const bool final) {
                if (tmp_part(u) == id) {
                    u32 start = edge_prefix;
                    u32 cnt = 0;

                    for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                        const vertex_t v = device_g.edges_v(i);
                        if (tmp_part(v) == id) {
                            if (final) {
                                const vertex_t sub_v = child_o_to_n(n_to_o(v));
                                child_g.edges_v(start) = sub_v;
                                if (!uniform_ew) {
                                    child_g.edges_w(start) = device_g.edges_w(i);
                                }
                            }
                            ++start;
                            ++cnt;
                        }
                    }

                    if (final) {
                        const vertex_t sub_u = child_o_to_n(n_to_o(u));
                        child_g.neighborhood(sub_u + 1) = edge_prefix + cnt;
                    }

                    edge_prefix += cnt;
                }
            });

            // fill the u array
            Kokkos::parallel_for("fill_edges_u", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, child_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = child_g.neighborhood(u);
                u32 end = child_g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    child_g.edges_u(i) = u;
                }
            });

            // We no longer need child_o_to_n after edges built
            pop_back(mem_stack);

            stats.subgraph_generation_ms += get_milli_seconds(sp_subgraph, get_time_point());

            // --- Recurse into this child
            const size_t bytes_in_use_before = mem_stack.n_bytes_in_use;
            const size_t bytes_in_use_back_before = mem_stack.n_bytes_in_use_back;
            
            identifier.push_back(id);
            recursive_multisection_device<uniform_vw, uniform_ew>(
                child_g,
                child_n_to_o,
                hierarchy,
                level - 1,
                global_imbalance,
                global_g_weight,
                global_k,
                global_n,
                seed,
                use_ultra,
                index_vec,
                k_rem,
                identifier,
                stats,
                global_partition,
                mem_stack,
                exec_space
            );
            identifier.pop_back();
            
            if (bytes_in_use_before != mem_stack.n_bytes_in_use || bytes_in_use_back_before != mem_stack.n_bytes_in_use_back) {
                std::cerr << "ERROR: Memory leak detected during recursion in hierarchical_multisection!" << std::endl;
                std::cerr << "       Front before: " << bytes_in_use_before << "  after: " << mem_stack.n_bytes_in_use << std::endl;
                std::cerr << "       Back before:  " << bytes_in_use_back_before << "  after: " << mem_stack.n_bytes_in_use_back << std::endl;
                abort();
            }

            // --- Free child allocations (reverse of allocations for this child)
            pop_front(mem_stack); // child_n_to_o
            free_graph(child_g, mem_stack); // whatever make_graph allocated
        }

        // Done at this node
        pop_front(mem_stack); // tmp_part
    }

    inline HostPartition hierarchical_multisection(const HostGraph &g,
                                                   const ProMapConfiguration &config) {
        auto sp = get_time_point();
        HEIPA_PROFILE_SCOPE("hm", "initialize", "allocate");

        DeviceExecutionSpace exec_space = DeviceExecutionSpace();
        HMStats stats;

        KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(config.n_bytes_requested, "Stack");

        f64 time = 0.0;
        Graph dev_g = from_HostGraph(g, mem_stack, time, exec_space);
        UnmanagedDevicePartition dev_global_part = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * g.n), g.n);
        UnmanagedDeviceVertex dev_n_to_o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.n), g.n);
        KOKKOS_PROFILE_FENCE(exec_space);

        Kokkos::parallel_for("InitIdMap", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) { dev_n_to_o(u) = u; });

        partition_t l = (partition_t) config.hierarchy.size();

        // index_vec as in your iterative version
        std::vector<partition_t> index_vec = {1};
        for (partition_t i = 0; i < l - 1; ++i) { index_vec.push_back(index_vec[i] * config.hierarchy[i]); }

        // k_rem as in your iterative version
        std::vector<partition_t> k_rem(l);
        u32 p = 1;
        for (partition_t i = 0; i < l; ++i) {
            k_rem[i] = p * config.hierarchy[i];
            p *= config.hierarchy[i];
        }

        std::vector<partition_t> identifier;
        identifier.reserve(l);

        if (dev_g.uniform_vertex_weights && dev_g.uniform_edge_weights) {
            recursive_multisection_device<true, true>(dev_g, dev_n_to_o, config.hierarchy, (u64) (l - 1), config.imbalance, g.g_weight, config.k, g.n, config.seed, config.use_ultra, index_vec, k_rem, identifier, stats, dev_global_part, mem_stack, exec_space);
        } else if (dev_g.uniform_vertex_weights) {
            recursive_multisection_device<true, false>(dev_g, dev_n_to_o, config.hierarchy, (u64) (l - 1), config.imbalance, g.g_weight, config.k, g.n, config.seed, config.use_ultra, index_vec, k_rem, identifier, stats, dev_global_part, mem_stack, exec_space);
        } else if (dev_g.uniform_edge_weights) {
            recursive_multisection_device<false, true>(dev_g, dev_n_to_o, config.hierarchy, (u64) (l - 1), config.imbalance, g.g_weight, config.k, g.n, config.seed, config.use_ultra, index_vec, k_rem, identifier, stats, dev_global_part, mem_stack, exec_space);
        } else {
            recursive_multisection_device<false, false>(dev_g, dev_n_to_o, config.hierarchy, (u64) (l - 1), config.imbalance, g.g_weight, config.k, g.n, config.seed, config.use_ultra, index_vec, k_rem, identifier, stats, dev_global_part, mem_stack, exec_space);
        }

        HEIPA_PROFILE_SCOPE("hm", "io", "copy_to_host");

        // copy back to host
        HostPartition host_part = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), g.n);;
        Kokkos::deep_copy(exec_space, host_part, dev_global_part);
        KOKKOS_PROFILE_FENCE(exec_space);

        // cleanup (reverse order)
        pop_front(mem_stack); // dev_n_to_o
        pop_front(mem_stack); // dev_global_part
        free_graph(dev_g, mem_stack);

        stats.print(get_milli_seconds(sp, get_time_point()));

        return host_part;
    }
}

#endif //GPU_HEIPA_HIERARCHICAL_MULTISECTION_H
