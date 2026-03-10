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

#ifndef GPU_HEIPA_KAFFPA_INITIAL_PARTITIONING_H
#define GPU_HEIPA_KAFFPA_INITIAL_PARTITIONING_H

/*
#include "../../extern/local/kahip/include/kaHIP_interface.h"

#include "../utility/definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include <vector>
#include <algorithm>

namespace GPU_HeiPa {
    inline void kaffpa_partition(HostGraph &g,
                                 int k,
                                 f64 imbalance,
                                 u64 seed,
                                 HostPartition &partition,
                                 int mode) {
        ScopedTimer _t("initial_partitioning", "global_multisection", "kaffpa_partition");

        if (k == 1) {
            for (vertex_t u = 0; u < g.n; ++u) {partition[u] = 0; }
            return;
        }

        int n = (int) g.n;
        vertex_t m = g.m;
        int *vwgt = (int *) malloc((u64) n * sizeof(int));
        int *xadj = (int *) malloc((u64) (n + 1) * sizeof(int));
        xadj[0] = 0;
        int *adjcwgt = (int *) malloc(m * sizeof(int));
        int *adjncy = (int *) malloc(m * sizeof(int));
        bool suppress_output = true;
        int edge_cut = std::numeric_limits<int>::max();
        int *part = (int *) malloc((u64) n * sizeof(int));

        std::vector<std::pair<vertex_t, weight_t> > edges;
        edges.reserve(m);
        for (vertex_t old_u = 0; old_u < g.n; ++old_u) {
            vwgt[old_u] = (int) g.weights(old_u);
            xadj[old_u + 1] = xadj[old_u];

            // Collect edges for this vertex
            edges.clear();
            for (u32 i = g.neighborhood(old_u); i < g.neighborhood(old_u + 1); ++i) {
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);
                edges.emplace_back(v, w);
            }

            // Sort edges by vertex ID for deterministic ordering
            std::sort(edges.begin(), edges.end());

            // Add sorted edges to arrays
            for (const auto &edge: edges) {
                adjncy[xadj[old_u + 1]] = (int) edge.first;
                adjcwgt[xadj[old_u + 1]] = (int) edge.second;
                xadj[old_u + 1] += 1;
            }
        }

        kaffpa(&n, vwgt, xadj, adjcwgt, adjncy, &k, &imbalance, suppress_output, (int) seed, mode, &edge_cut, part);

        for (int i = 0; i < n; ++i) {
            partition(i) = (partition_t) part[i];
        }

        free(vwgt);
        free(xadj);
        free(adjcwgt);
        free(adjncy);
        free(part);
    }

    inline void kaffpa_partition(Graph &g,
                                 int k,
                                 f64 imbalance,
                                 u64 seed,
                                 Partition &partition,
                                 int mode) {
        ScopedTimer _t_copy("initial_partitioning", "KaFFPa", "copy");

        // Convert device graph to simple CSR arrays on host
        HostGraph host_g = to_host_graph(g);
        HostPartition host_partition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), g.n);

        kaffpa_partition(host_g, k, imbalance, seed, host_partition, mode);

        auto device_subview = Kokkos::subview(partition.map, std::pair<size_t, size_t>(0, host_partition.extent(0)));
        Kokkos::deep_copy(device_subview, host_partition);
        Kokkos::fence();
    }
}
*/

#endif //GPU_HEIPA_KAFFPA_INITIAL_PARTITIONING_H
