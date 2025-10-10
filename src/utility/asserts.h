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

#include "../datastructures/graph.h"

namespace GPU_HeiPa {
#ifndef ASSERT_ENABLED
#define ASSERT_ENABLED false
#endif

#if (ASSERT_ENABLED)
#define ASSERT(condition) if(!(condition)) {std::cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " at line " << __LINE__ << "!" << std::endl; abort(); } ((void)0)
#else
#define ASSERT(condition) if(!(condition)) {((void)0); } ((void)0)
#endif


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
                weight_t w = host_g.edges_w(i);

                ASSERT(w > 0);
            }
        }
    }

    inline void assert_edges_u(const HostGraph &host_g,
                               const HostVertex &host_edges_u) {
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

            weights[u_id] += g.weights(u);
        }

        for (partition_t id = 0; id < k; ++id) {
            ASSERT(weights[id] == partition.bweights(id));
        }
    }


    inline void assert_state_pre_partition(const Graph &device_g) {
#if !ASSERT_ENABLED
        return;
#endif
        HostGraph host_g = to_host_graph(device_g);
        HostVertex host_edges_u = HostVertex("edges_u", host_g.m);
        Kokkos::deep_copy(host_edges_u, device_g.edges_u);
        Kokkos::fence();

        assert_no_loops(host_g);
        assert_no_double_edges(host_g);
        assert_positive_edges(host_g);
        assert_edges_u(host_g, host_edges_u);
    }

    inline void assert_state_after_partition(const Graph &device_g,
                                             const Partition &partition,
                                             const partition_t k) {
#if !ASSERT_ENABLED
        return;
#endif
        HostGraph host_g = to_host_graph(device_g);
        HostVertex host_edges_u = HostVertex("edges_u", host_g.m);
        Kokkos::deep_copy(host_edges_u, device_g.edges_u);
        PartitionHost host_p_manager = to_host_p_manager(partition);

        assert_no_loops(host_g);
        assert_no_double_edges(host_g);
        assert_positive_edges(host_g);
        assert_edges_u(host_g, host_edges_u);

        assert_partition(host_g, host_p_manager, k);
        assert_bweights(host_g, host_p_manager, k);
    }
}

#endif //GPU_HEIPA_ASSERTS_H
