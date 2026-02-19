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

#include "../utility/definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/solver.h"

namespace GPU_HeiPa {
    inline void gpu_heipa_partition(Graph &device_g,
                                    partition_t k,
                                    f64 imbalance,
                                    u64 seed,
                                    bool use_ultra,
                                    UnmanagedDevicePartition &partition,
                                    KokkosMemoryStack &mem_stack) {
        if (k == 1) {
            ScopedTimer t{"hm", "recursive", "partition k=1"};
            Kokkos::deep_copy(partition, 0);
            Kokkos::fence();
            return;
        }

        Solver solver(device_g, k, imbalance, seed, use_ultra, partition, mem_stack);
    }

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
                                              std::vector<partition_t> &identifier, // path of ids
                                              UnmanagedDevicePartition &global_partition, // size global_n
                                              KokkosMemoryStack &mem_stack) {
        ScopedTimer t{"hm", "recursive", "allocate"};
        const partition_t l = (partition_t) hierarchy.size();
        const partition_t k = hierarchy[level];

        // Allocate temp partition for *this* node
        UnmanagedDevicePartition tmp_part = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * device_g.n), device_g.n);

        // Compute adaptive imbalance if you want (you used determine_adaptive_imbalance in host version)
        // Here: just use global_imbalance directly (plug your adaptive formula if desired).
        const f64 imb = global_imbalance;

        t.stop();

        // 1) Partition current device graph into k blocks
        gpu_heipa_partition(device_g, k, imb, seed, use_ultra, tmp_part, mem_stack);

        // 2) Leaf: last split -> write into global_partition
        if (identifier.size() == (size_t) (l - 1)) {
            ScopedTimer t_write{"hm", "recursive", "write_to_global"};
            // offset = sum_{i=0..l-2} identifier[i] * index_vec[last-i]
            partition_t offset = 0;
            for (partition_t i = 0; i < l - 1; ++i) { offset += identifier[i] * index_vec[index_vec.size() - 1 - i]; }

            Kokkos::parallel_for("WriteLeafPartition", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                const vertex_t orig_u = n_to_o(u);
                global_partition(orig_u) = offset + tmp_part(u);
            });
            Kokkos::fence();

            pop_front(mem_stack); // tmp_part
            return;
        }

        // 3) Non-leaf: build each child and recurse immediately
        for (partition_t id = 0; id < k; ++id) {
            ScopedTimer t_subgraph{"hm", "recursive", "generate_subgraph"};

            // --- First pass: compute sub_n, sub_m, sub_weight for this id
            vertex_t sub_n = 0;
            vertex_t sub_m = 0;
            weight_t sub_weight = 0;

            Kokkos::parallel_reduce("SubN", device_g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                if (tmp_part(u) == id) lsum += 1;
            }, sub_n);

            Kokkos::parallel_reduce("SubWeight", device_g.n, KOKKOS_LAMBDA(const vertex_t u, weight_t &lsum) {
                if (tmp_part(u) == id) lsum += device_g.weights(u);
            }, sub_weight);

            Kokkos::parallel_reduce("SubM", device_g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                if (tmp_part(u) == id) {
                    vertex_t cnt = 0;
                    for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                        const vertex_t v = device_g.edges_v(i);
                        if (tmp_part(v) == id) ++cnt;
                    }
                    lsum += cnt;
                }
            }, sub_m);

            Kokkos::fence();

            // Empty block => skip
            if (sub_n == 0) {
                continue;
            }

            // --- Allocate child graph + mappings
            Graph child_g = make_graph(sub_n, sub_m, sub_weight, mem_stack);
            UnmanagedDeviceVertex child_n_to_o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * sub_n), sub_n);
            UnmanagedDeviceVertex child_o_to_n = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * global_n), global_n);

            // --- Fill translation tables and weights
            Kokkos::parallel_scan("AssignLocalIndex", device_g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &prefix, const bool final) {
                if (tmp_part(u) == id) {
                    const vertex_t my_idx = prefix;
                    if (final) {
                        const vertex_t old_u = n_to_o(u);
                        child_o_to_n(old_u) = my_idx;
                        child_n_to_o(my_idx) = old_u;
                        child_g.weights(my_idx) = device_g.weights(u);
                    }
                    prefix += 1;
                }
            });
            Kokkos::fence();

            // init neighborhood(0)
            Kokkos::parallel_for("InitNeighborhood0", 1, KOKKOS_LAMBDA(const int) { child_g.neighborhood(0) = 0; });
            Kokkos::fence();

            // --- Fill edges + neighborhood offsets
            Kokkos::parallel_scan("FillEdges", device_g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &edge_prefix, const bool final) {
                if (tmp_part(u) == id) {
                    u32 start = edge_prefix;
                    u32 cnt = 0;

                    for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                        const vertex_t v = device_g.edges_v(i);
                        if (tmp_part(v) == id) {
                            if (final) {
                                const vertex_t sub_v = child_o_to_n(n_to_o(v));
                                child_g.edges_v(start) = sub_v;
                                child_g.edges_w(start) = device_g.edges_w(i);
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
            Kokkos::fence();

            // fill the u array
            Kokkos::parallel_for("fill_edges_u", child_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = child_g.neighborhood(u);
                u32 end = child_g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    child_g.edges_u(i) = u;
                }
            });
            Kokkos::fence();

            // We no longer need child_o_to_n after edges built
            pop_back(mem_stack);

            t_subgraph.stop();

            // --- Recurse into this child
            identifier.push_back(id);
            recursive_multisection_device(
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
                identifier,
                global_partition,
                mem_stack
            );
            identifier.pop_back();

            // --- Free child allocations (reverse of allocations for this child)
            pop_front(mem_stack); // child_n_to_o
            free_graph(child_g, mem_stack); // whatever make_graph allocated
        }

        // Done at this node
        pop_front(mem_stack); // tmp_part
    }

    inline HostPartition hierarchical_multisection(const HostGraph &g,
                                                   const std::vector<partition_t> &hierarchy,
                                                   partition_t global_k,
                                                   f64 imbalance,
                                                   u64 seed,
                                                   bool use_ultra) {
        ScopedTimer t{"hm", "initialize", "allocate"};

        KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(30 * (size_t) g.n * sizeof(vertex_t) + 10 * (size_t) g.m * sizeof(vertex_t), "Stack");

        f64 time = 0.0;
        Graph dev_g = from_HostGraph(g, mem_stack, time);
        UnmanagedDevicePartition dev_global_part = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * g.n), g.n);
        UnmanagedDeviceVertex dev_n_to_o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.n), g.n);
        Kokkos::fence();

        Kokkos::parallel_for("InitIdMap", g.n, KOKKOS_LAMBDA(const vertex_t u) { dev_n_to_o(u) = u; });
        Kokkos::fence();

        // index_vec (same as your host version idea)
        const partition_t l = (partition_t) hierarchy.size();
        std::vector<partition_t> index_vec = {1};
        for (partition_t i = 0; i < l - 1; ++i) index_vec.push_back(index_vec[i] * hierarchy[i]);

        std::vector<partition_t> identifier;
        identifier.reserve(l);

        t.stop();

        recursive_multisection_device(dev_g,
                                      dev_n_to_o,
                                      hierarchy,
                                      (u64) (l - 1),
                                      imbalance,
                                      g.g_weight,
                                      global_k,
                                      g.n,
                                      seed,
                                      use_ultra,
                                      index_vec,
                                      identifier,
                                      dev_global_part,
                                      mem_stack);

        ScopedTimer t_copy{"hm", "io", "copy_to_host"};

        // copy back to host
        HostPartition host_part = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), g.n);;
        Kokkos::deep_copy(host_part, dev_global_part);
        Kokkos::fence();

        // cleanup (reverse order)
        pop_front(mem_stack); // dev_n_to_o
        pop_front(mem_stack); // dev_global_part
        free_graph(dev_g, mem_stack);

        return host_part;
    }
}

#endif //GPU_HEIPA_HIERARCHICAL_MULTISECTION_H
