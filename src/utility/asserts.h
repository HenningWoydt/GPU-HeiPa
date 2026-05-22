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

#ifndef GPU_HEIPA_ASSERTS_H
#define GPU_HEIPA_ASSERTS_H

#include <unordered_set>
#include <map>
#include <vector>
#include <iostream>

#include "macros.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    inline void assert_all_vertices_small(const HostGraph &host_g) {
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood[u];
            u32 end = host_g.neighborhood[u + 1];

            for (u32 i = begin; i < end; ++i) {
                vertex_t v = host_g.edges_v[i];
                ASSERT(v < host_g.n);
            }
        }
    }

    inline void assert_no_loops(const HostGraph &host_g) {
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood[u];
            u32 end = host_g.neighborhood[u + 1];

            for (u32 i = begin; i < end; ++i) {
                vertex_t v = host_g.edges_v[i];
                ASSERT(v != u);
            }
        }
    }

    inline void assert_no_double_edges(const HostGraph &host_g) {
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood[u];
            u32 end = host_g.neighborhood[u + 1];

            std::unordered_set<vertex_t> seen;

            for (u32 i = begin; i < end; ++i) {
                vertex_t v = host_g.edges_v[i];
                if (!seen.insert(v).second) {
                    std::cerr << "Vertex " << v << " already in use." << std::endl;
                }
            }
        }
    }

    inline void assert_positive_edges(const HostGraph &host_g) {
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood(u);
            u32 end = host_g.neighborhood(u + 1);

            for (u32 i = begin; i < end; ++i) {
                weight_t w = host_g.uniform_edge_weights ? 1 : host_g.edges_w(i);

                if (w <= 0) {
                    std::cerr << "INVALID EDGE WEIGHT! u: " << u << " v: " << host_g.edges_v(i) << " weight: " << w << std::endl;
                    ASSERT(w > 0);
                }
            }
        }
    }

    inline void assert_edges_u(const HostGraph &host_g,
                               const UnmanagedHostVertex &host_edges_u) {
        u32 i = 0;
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood[u];
            u32 end = host_g.neighborhood[u + 1];

            for (u32 j = begin; j < end; ++j) {
                ASSERT(host_edges_u(i) == u);
                i += 1;
            }
        }
    }

    inline void assert_partition(const HostGraph &g,
                                 const PartitionHost &partition,
                                 const partition_t k) {
        for (vertex_t u = 0; u < g.n; ++u) {
            partition_t u_id = partition.map(u);

            ASSERT(u_id < k);
        }
    }

    inline void assert_bweights(const HostGraph &g,
                                const PartitionHost &partition,
                                const partition_t k) {
        std::vector<weight_t> weights(k, 0);

        for (vertex_t u = 0; u < g.n; ++u) {
            partition_t u_id = partition.map(u);

            weights[u_id] += g.uniform_vertex_weights ? 1 : g.weights(u);
        }

        for (partition_t id = 0; id < k; ++id) {
            ASSERT(weights[id] == partition.bweights(id));
        }
    }

    inline void assert_hierarchy(const partition_t k,
                                 const DeviceU32 &d_block_lvl,
                                 const DeviceU32 &d_block_fact,
                                 const DeviceU32 &d_hierarchy,
                                 DeviceExecutionSpace &exec_space) {
        #if !ASSERT_ENABLED
        return;
        #endif

        HostU32 h_block_lvl("h_block_lvl", k);
        HostU32 h_block_fact("h_block_fact", k);
        HostU32 h_hierarchy("h_hierarchy", d_hierarchy.extent(0));

        Kokkos::deep_copy(exec_space, h_block_lvl, d_block_lvl);
        Kokkos::deep_copy(exec_space, h_block_fact, d_block_fact);
        Kokkos::deep_copy(exec_space, h_hierarchy, d_hierarchy);
        exec_space.fence();

        for (partition_t i = 0; i < k; ++i) {
            u32 lvl = h_block_lvl(i);
            u32 fact = h_block_fact(i);
            ASSERT(lvl < h_hierarchy.extent(0));
            if (fact > 0) {
                // If fact > 1, it should be <= the hierarchy value at that level (or equal if just normalized)
                // This is a loose check as fact changes during splitting.
                ASSERT(fact <= h_hierarchy(lvl));
            }
        }
    }


    inline void assert_coarsening(const Graph &old_g,
                                 const Graph &coarse_g,
                                 const Mapping &mapping,
                                 DeviceExecutionSpace &exec_space) {
        #if !ASSERT_ENABLED
        return;
        #endif

        HostGraph h_old = to_host_graph(old_g, exec_space);
        HostGraph h_coarse = to_host_graph(coarse_g, exec_space);

        HostVertex h_mapping("h_mapping", mapping.old_n);
        Kokkos::deep_copy(exec_space, h_mapping, mapping.mapping);
        exec_space.fence();

        // 1. Check vertex weights
        std::vector<weight_t> expected_v_weights(h_coarse.n, 0);
        weight_t total_old_v_weight = 0;
        for (vertex_t u = 0; u < h_old.n; ++u) {
            vertex_t u_new = h_mapping(u);
            ASSERT(u_new < h_coarse.n);
            weight_t uw = h_old.uniform_vertex_weights ? 1 : h_old.weights(u);
            expected_v_weights[u_new] += uw;
            total_old_v_weight += uw;
        }

        weight_t total_coarse_v_weight = 0;
        for (vertex_t U = 0; U < h_coarse.n; ++U) {
            weight_t Uw = h_coarse.uniform_vertex_weights ? 1 : h_coarse.weights(U);
            ASSERT(expected_v_weights[U] == Uw);
            total_coarse_v_weight += Uw;
        }
        ASSERT(total_old_v_weight == total_coarse_v_weight);

        // 2. Check edges
        // We use a map to store expected coarse edges: (U, V) -> weight
        // Since the graph is undirected and stored symmetrically, we check all directed edges.
        struct Edge {
            vertex_t u, v;
            bool operator<(const Edge &other) const {
                if (u != other.u) return u < other.u;
                return v < other.v;
            }
        };
        std::map<Edge, weight_t> expected_e_weights;

        for (vertex_t u = 0; u < h_old.n; ++u) {
            vertex_t U = h_mapping(u);
            for (u32 i = h_old.neighborhood(u); i < h_old.neighborhood(u + 1); ++i) {
                vertex_t v = h_old.edges_v(i);
                vertex_t V = h_mapping(v);
                if (U == V) continue; // Skip self-loops in coarse graph

                weight_t ew = h_old.uniform_edge_weights ? 1 : h_old.edges_w(i);
                expected_e_weights[{U, V}] += ew;
            }
        }

        u32 coarse_edges_found = 0;
        for (vertex_t U = 0; U < h_coarse.n; ++U) {
            for (u32 i = h_coarse.neighborhood(U); i < h_coarse.neighborhood(U + 1); ++i) {
                vertex_t V = h_coarse.edges_v(i);
                weight_t ew = h_coarse.uniform_edge_weights ? 1 : h_coarse.edges_w(i);

                auto it = expected_e_weights.find({U, V});
                ASSERT(it != expected_e_weights.end());
                ASSERT(it->second == ew);
                coarse_edges_found++;
            }
        }

        ASSERT(coarse_edges_found == expected_e_weights.size());
    }

    inline void assert_state_pre_partition(const Graph &device_g,
                                           DeviceExecutionSpace &exec_space) {
        #if !ASSERT_ENABLED
        return;
        #endif
        HostGraph host_g = to_host_graph(device_g, exec_space);
        HostVertex host_edges_u = HostVertex("edges_u", host_g.m);
        Kokkos::deep_copy(exec_space, host_edges_u, Kokkos::subview(device_g.edges_u, std::make_pair(0U, device_g.m)));
        exec_space.fence();

        assert_all_vertices_small(host_g);
        assert_no_loops(host_g);
        assert_no_double_edges(host_g);
        assert_positive_edges(host_g);
        assert_edges_u(host_g, host_edges_u);
    }

    inline void assert_state_after_partition(const Graph &device_g,
                                             const Partition &partition,
                                             const partition_t k,
                                             DeviceExecutionSpace &exec_space) {
        #if !ASSERT_ENABLED
        return;
        #endif
        HostGraph host_g = to_host_graph(device_g, exec_space);
        HostVertex host_edges_u = HostVertex("edges_u", host_g.m);
        Kokkos::deep_copy(exec_space, host_edges_u, Kokkos::subview(device_g.edges_u, std::make_pair(0U, device_g.m)));
        PartitionHost host_p_manager = to_host_partition(partition, exec_space);

        assert_all_vertices_small(host_g);
        assert_no_loops(host_g);
        assert_no_double_edges(host_g);
        assert_positive_edges(host_g);
        assert_edges_u(host_g, host_edges_u);

        assert_partition(host_g, host_p_manager, k);
        assert_bweights(host_g, host_p_manager, k);
    }
}

#endif //GPU_HEIPA_ASSERTS_H
