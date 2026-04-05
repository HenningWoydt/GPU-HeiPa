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

#ifndef GPU_HEIPA_GLOBAL_MULTISECTION_H
#define GPU_HEIPA_GLOBAL_MULTISECTION_H

#include <vector>
#include <algorithm>

#include "../utility/definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "kway_partitioner/kway_core.h"

namespace GPU_HeiPa {
    inline f64 determine_adaptive_imbalance(const f64 global_imbalance,
                                            const weight_t global_g_weight,
                                            const partition_t global_k,
                                            const weight_t local_g_weight,
                                            const partition_t local_k_rem,
                                            const u64 depth) {
        f64 local_imbalance = (1.0 + global_imbalance) * ((f64) (local_k_rem * (u64) global_g_weight) / (f64) (global_k * (u64) local_g_weight));
        local_imbalance = std::pow(local_imbalance, (f64) 1 / (f64) depth) - 1.0;
        return local_imbalance;
    }

    inline void build_subgraphs(const HostGraph &g,
                                const HostPartition &partition,
                                const partition_t k,
                                const std::vector<vertex_t> &parent_n_to_o,
                                const vertex_t max_n,
                                std::vector<HostGraph> &subgraphs,
                                std::vector<std::vector<vertex_t> > &n_to_os,
                                std::vector<std::vector<vertex_t> > &o_to_ns) {
        // reset
        subgraphs.resize(k);
        n_to_os.resize(k);
        o_to_ns.resize(k);
        for (partition_t id = 0; id < k; ++id) {
            subgraphs[id].n = 0;
            subgraphs[id].m = 0;
            subgraphs[id].g_weight = 0;
        }

        // count vertices, edges and weight
        for (vertex_t u = 0; u < g.n; ++u) {
            partition_t u_id = partition(u);
            subgraphs[u_id].n += 1;
            subgraphs[u_id].g_weight += g.weights(u);

            for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                vertex_t v = g.edges_v(i);
                partition_t v_id = partition(v);
                if (u_id == v_id) { subgraphs[u_id].m += 1; }
            }
        }

        // allocate memory
        for (partition_t id = 0; id < k; ++id) {
            allocate_memory(subgraphs[id], subgraphs[id].n, subgraphs[id].m, subgraphs[id].g_weight);

            n_to_os[id].resize(max_n);
            o_to_ns[id].resize(max_n);
        }

        // fill the translation tables
        std::vector<vertex_t> new_us(k, 0);
        for (vertex_t old_u = 0; old_u < g.n; ++old_u) {
            partition_t u_id = partition(old_u);

            vertex_t original_u = parent_n_to_o[old_u];
            vertex_t new_u = new_us[u_id];

            o_to_ns[u_id][original_u] = new_u;
            n_to_os[u_id][new_u] = original_u;
            new_us[u_id] += 1;
        }

        // create the graphs
        for (vertex_t old_u = 0; old_u < g.n; ++old_u) {
            partition_t u_id = partition(old_u);

            vertex_t original_u = parent_n_to_o[old_u];
            vertex_t new_u = o_to_ns[u_id][original_u]; // vertex in new graph

            // set the weight
            subgraphs[u_id].weights(new_u) = g.weights(old_u);
            subgraphs[u_id].neighborhood(new_u + 1) = subgraphs[u_id].neighborhood(new_u);

            // set the edges
            for (vertex_t i = g.neighborhood(old_u); i < g.neighborhood(old_u + 1); ++i) {
                vertex_t old_v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                if (u_id == partition(old_v)) {
                    // add the edge
                    vertex_t original_v = parent_n_to_o[old_v];
                    vertex_t new_v = o_to_ns[u_id][original_v]; // vertex in new graph
                    vertex_t edge_idx = subgraphs[u_id].neighborhood(new_u + 1);

                    subgraphs[u_id].edges_v(edge_idx) = new_v;
                    subgraphs[u_id].edges_w(edge_idx) = w;
                    subgraphs[u_id].neighborhood(new_u + 1) += 1;
                }
            }
        }
    }

    inline void recursive_partition(HostGraph &g,
                                    const std::vector<partition_t> &hierarchy,
                                    const u64 curr_l, // start with hierarchy.size()-1 and count down
                                    const std::vector<partition_t> &k_rem,
                                    const std::vector<partition_t> &index_vec,
                                    const std::vector<vertex_t> &n_to_o,
                                    std::vector<partition_t> &identifier,
                                    f64 global_imbalance,
                                    weight_t global_g_weight,
                                    partition_t global_k,
                                    vertex_t global_n,
                                    u64 seed,
                                    HostPartition &temp_partition,
                                    HostPartition &global_partition) {
        const partition_t l = (partition_t) hierarchy.size();
        const partition_t k = hierarchy[curr_l];
        const f64 imb = determine_adaptive_imbalance(global_imbalance, global_g_weight, global_k, g.g_weight, k_rem[l - 1 - identifier.size()], l - identifier.size());

        #if ASSERT_ENABLED
        for (vertex_t u = 0; u < g.n; ++u) { temp_partition[u] = k; }
        #endif

        // partition current graph into k_here blocks (writes temp_partition for vertices 0..g.n-1)
        {
            // kaffpa_partition(g, (int) k, imb, seed, temp_partition, FAST);
            // metis_partition(g, (int) k, imb, seed, temp_partition, METIS_KWAY);
            kway_partition(g, (int) k, imb, seed, temp_partition);
            // recursive_bisec_partition(g, k, imb, seed, temp_partition);
        }

        #if ASSERT_ENABLED
        for (vertex_t u = 0; u < g.n; ++u) { ASSERT(temp_partition[u] < k); }
        #endif

        // leaf: last split (identifier already contains all previous split-ids)
        if (identifier.size() == l - 1) {
            ScopedTimer _t("initial_partitioning", "global_multisection", "insert_solution");

            partition_t offset = 0;
            for (partition_t i = 0; i < l - 1; ++i) { offset += identifier[i] * index_vec[index_vec.size() - 1 - i]; }
            for (vertex_t u = 0; u < g.n; ++u) { global_partition(n_to_o[u]) = offset + temp_partition(u); }
            return;
        }

        std::vector<HostGraph> sub_gs;
        std::vector<std::vector<vertex_t> > n_to_os;
        std::vector<std::vector<vertex_t> > o_to_ns;
        //
        {
            ScopedTimer _t("initial_partitioning", "global_multisection", "build subgraphs");
            build_subgraphs(g, temp_partition, k, n_to_o, global_n, sub_gs, n_to_os, o_to_ns);
        }

        for (partition_t id = 0; id < k; ++id) {
            identifier.push_back(id);
            recursive_partition(sub_gs[id],
                                hierarchy,
                                curr_l - 1,
                                k_rem,
                                index_vec,
                                n_to_os[id],
                                identifier,
                                global_imbalance,
                                global_g_weight,
                                global_k,
                                global_n,
                                seed,
                                temp_partition,
                                global_partition);
            identifier.pop_back();
        }
    }

    inline void global_multisection_host(HostGraph &g,
                                         const std::vector<partition_t> &hierarchy,
                                         partition_t k,
                                         f64 imbalance,
                                         u64 seed,
                                         HostPartition &partition) {
        ScopedTimer _t("initial_partitioning", "global_multisection", "global_multisection_host");

        const f64 global_imbalance = imbalance;
        const weight_t global_g_weight = g.g_weight;
        const partition_t global_k = k;
        const partition_t l = (partition_t) hierarchy.size();

        // index_vec as in your iterative version
        std::vector<partition_t> index_vec = {1};
        for (partition_t i = 0; i < l - 1; ++i) { index_vec.push_back(index_vec[i] * hierarchy[i]); }

        // k_rem as in your iterative version
        std::vector<partition_t> k_rem_vec(l);
        u32 p = 1;
        for (partition_t i = 0; i < l; ++i) {
            k_rem_vec[i] = p * hierarchy[i];
            p *= hierarchy[i];
        }

        // identity mapping to original vertices
        std::vector<vertex_t> n_to_o(g.n);
        for (vertex_t u = 0; u < g.n; ++u) n_to_o[u] = u;

        // temp partition buffer (will be resized inside recursion if needed)
        HostPartition temp_partition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "temp_partition"), g.n);

        std::vector<partition_t> identifier;
        identifier.reserve(l);

        _t.stop();

        // IMPORTANT: start from hierarchy.back() like your stack version
        recursive_partition(g,
                            hierarchy,
                            (u64) (l - 1),
                            k_rem_vec,
                            index_vec,
                            n_to_o,
                            identifier,
                            global_imbalance,
                            global_g_weight,
                            global_k,
                            g.n,
                            seed,
                            temp_partition,
                            partition);
    }

    inline void global_multisection(const Graph &g,
                                    const std::vector<partition_t> &hierarchy,
                                    partition_t k,
                                    f64 imbalance,
                                    u64 seed,
                                    Partition &partition,
                                    DeviceExecutionSpace &exec_space) {
        HostGraph host_g;
        HostPartition host_partition;
        // initialize the host
        {
            ScopedTimer _t("initial_partitioning", "global_multisection", "init_host");

            // Convert device graph to simple CSR arrays on host
            host_g = to_host_graph(g, exec_space);
            host_partition = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), g.n);
        }

        global_multisection_host(host_g, hierarchy, k, imbalance, seed, host_partition);

        // upload the partition
        {
            ScopedTimer _t("initial_partitioning", "global_multisection", "upload_partition");

            auto device_subview = Kokkos::subview(partition.map, std::pair<size_t, size_t>(0, host_partition.extent(0)));
            Kokkos::deep_copy(exec_space, device_subview, host_partition);
            exec_space.fence();
        }
    }
}

#endif //GPU_HEIPA_GLOBAL_MULTISECTION_H
