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

#ifndef GPU_HEIPA_METIS_INITIAL_PARTITIONING_H
#define GPU_HEIPA_METIS_INITIAL_PARTITIONING_H

#include "../../extern/local/METIS/include/metis.h"

#include "../utility/definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    inline void metis_initial_partition(Graph &g,
                                        int k,
                                        f64 imbalance,
                                        u64 seed,
                                        Partition &partition) {
        ScopedTimer _t_copy("initial_partitioning", "METIS", "copy");

        // Convert device graph to simple CSR arrays on host
        HostGraph host_g = to_host_graph(g);
        HostPartition host_partition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), g.n);

        idx_t n = (idx_t) g.n;
        idx_t ncon = 1;  // Number of constraints (typically 1 for vertex weights)
        vertex_t m = g.m;
        idx_t *vwgt = (idx_t *) malloc((u64) n * sizeof(idx_t));
        idx_t *xadj = (idx_t *) malloc((u64) (n + 1) * sizeof(idx_t));
        xadj[0] = 0;
        idx_t *adjwgt = (idx_t *) malloc(m * sizeof(idx_t));
        idx_t *adjncy = (idx_t *) malloc(m * sizeof(idx_t));
        idx_t nparts = (idx_t) k;
        idx_t edgecut = 0;
        idx_t *part = (idx_t *) malloc((u64) n * sizeof(idx_t));
        idx_t options[METIS_NOPTIONS];
        real_t *ubvec = (real_t *) malloc((u64) ncon * sizeof(real_t));

        // Set default options and configure
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_SEED] = (idx_t) seed;
        options[METIS_OPTION_NCUTS] = 1;      // Single cut for determinism
        options[METIS_OPTION_NUMBERING] = 0;  // C-style numbering
        
        // Set imbalance constraint
        for (int i = 0; i < ncon; i++) {
            ubvec[i] = (real_t) (1.0 + imbalance);
        }

        std::vector<std::pair<vertex_t, weight_t>> edges;
        edges.reserve(m);
        for (vertex_t old_u = 0; old_u < host_g.n; ++old_u) {
            vwgt[old_u] = (idx_t) host_g.weights(old_u);
            xadj[old_u + 1] = xadj[old_u];

            // Collect edges for this vertex
            edges.clear();
            for (u32 i = host_g.neighborhood(old_u); i < host_g.neighborhood(old_u + 1); ++i) {
                vertex_t v = host_g.edges_v(i);
                weight_t w = host_g.edges_w(i);
                edges.emplace_back(v, w);
            }
            
            // Sort edges by vertex ID for deterministic ordering
            std::sort(edges.begin(), edges.end());
            
            // Add sorted edges to arrays
            for (const auto& edge : edges) {
                adjncy[xadj[old_u + 1]] = (idx_t) edge.first;
                adjwgt[xadj[old_u + 1]] = (idx_t) edge.second;
                xadj[old_u + 1] += 1;
            }
        }

        _t_copy.stop();
        ScopedTimer _t_partition("initial_partitioning", "METIS", "partition");

        METIS_PartGraphKway(&n, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, NULL, ubvec, options, &edgecut, part);

        _t_partition.stop();
        ScopedTimer _t_upload("initial_partitioning", "METIS", "upload");

        for (int i = 0; i < n; ++i) {
            host_partition(i) = (partition_t) part[i];
        }

        free(vwgt);
        free(xadj);
        free(adjwgt);
        free(adjncy);
        free(part);
        free(ubvec);

        auto device_subview = Kokkos::subview(partition.map, std::pair<size_t, size_t>(0, host_partition.extent(0)));
        Kokkos::deep_copy(device_subview, host_partition);
        Kokkos::fence();
    }
}

#endif //GPU_HEIPA_METIS_INITIAL_PARTITIONING_H
