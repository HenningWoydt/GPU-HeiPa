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

#include <stack>

#include "../../extern/local/kahip/include/kaHIP_interface.h"

#include "../utility/definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    inline void kaffpa_initial_partition(Graph &device_g,
                                         int k,
                                         f64 imbalance,
                                         u32 seed,
                                         Partition &partition) {
        ScopedTimer _t_copy("initial_partitioning", "KaFFPa", "copy");

        // Convert device graph to simple CSR arrays on host
        HostGraph host_g = to_host_graph(device_g);
        HostPartition host_partition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), device_g.n);

        int n = (int) device_g.n;
        vertex_t m = device_g.m;
        int *vwgt = (int *) malloc(n * sizeof(int));
        int *xadj = (int *) malloc((n + 1) * sizeof(int));
        xadj[0] = 0;
        int *adjcwgt = (int *) malloc(m * sizeof(int));
        int *adjncy = (int *) malloc(m * sizeof(int));
        bool suppress_output = true;
        int mode = FAST;
        int edge_cut_temp = 0;
        int *part_temp = (int *) malloc(n * sizeof(int));
        int edge_cut = std::numeric_limits<int>::max();
        int *part = (int *) malloc(n * sizeof(int));

        for (vertex_t old_u = 0; old_u < host_g.n; ++old_u) {
            vwgt[old_u] = (int) host_g.weights(old_u);
            xadj[old_u + 1] = xadj[old_u];

            for (u32 i = host_g.neighborhood(old_u); i < host_g.neighborhood(old_u + 1); ++i) {
                vertex_t v = host_g.edges_v(i);
                weight_t w = host_g.edges_w(i);

                adjcwgt[xadj[old_u + 1]] = (int) w;
                adjncy[xadj[old_u + 1]] = (int) v;
                xadj[old_u + 1] += 1;
            }
        }

        _t_copy.stop();
        ScopedTimer _t_partition("initial_partitioning", "KaFFPa", "partition");

        kaffpa(&n, vwgt, xadj, adjcwgt, adjncy, &k, &imbalance, suppress_output, (int) seed, mode, &edge_cut_temp, part_temp);

        _t_partition.stop();
        ScopedTimer _t_upload("initial_partitioning", "KaFFPa", "upload");

        if (edge_cut_temp < edge_cut) {
            edge_cut = edge_cut_temp;
            std::swap(part_temp, part);
        }

        for (int i = 0; i < n; ++i) {
            host_partition(i) = (partition_t) part[i];
        }

        free(vwgt);
        free(xadj);
        free(adjcwgt);
        free(adjncy);
        free(part_temp);
        free(part);

        auto device_subview = Kokkos::subview(partition.map, std::pair<size_t, size_t>(0, host_partition.extent(0)));
        Kokkos::deep_copy(device_subview, host_partition);
        Kokkos::fence();
    }
}

#endif //GPU_HEIPA_KAFFPA_INITIAL_PARTITIONING_H
