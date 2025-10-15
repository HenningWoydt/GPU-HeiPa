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

#ifndef GPU_HEIPA_EDGE_CUT_H
#define GPU_HEIPA_EDGE_CUT_H

#include "definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/block_connectivity.h"

namespace GPU_HeiPa {
    inline weight_t edge_cut(const Graph &g,
                             const Partition &partition) {
        weight_t sum = 0;

        Kokkos::parallel_reduce("edge_cut", g.m, KOKKOS_LAMBDA(const u32 i, weight_t &local_sum) {
                                    vertex_t u = g.edges_u(i);
                                    vertex_t v = g.edges_v(i);
                                    weight_t w = g.edges_w(i);

                                    partition_t u_id = partition.map[u];
                                    partition_t v_id = partition.map[v];

                                    local_sum += w * (u_id != v_id);
                                },
                                sum);
        Kokkos::fence();

        return sum;
    }

    inline weight_t edge_cut(const Graph &device_g,
                             const Partition &partition,
                             const BlockConnectivity &bc) {
        weight_t total_comm_cost = 0;

        Kokkos::parallel_reduce("edge_cut", device_g.n, KOKKOS_LAMBDA(const vertex_t u, weight_t &local_comm_cost) {
                                    partition_t u_id = partition.map(u);
                                    weight_t sum = 0;

                                    for (u32 i = bc.row(u); i < bc.row(u + 1); ++i) {
                                        partition_t id = bc.ids(i);
                                        weight_t w = bc.weights(i);

                                        if (id == u_id) { continue; }
                                        if (id == partition.k) { continue; }

                                        sum += w;
                                    }

                                    local_comm_cost += sum;
                                },
                                total_comm_cost);
        Kokkos::fence();

        /*
        Kokkos::parallel_reduce("edge_cut", bc.size, KOKKOS_LAMBDA(const u32 j, weight_t &local_comm_cost) {
                                    vertex_t u = bc.us(j);
                                    partition_t u_id = partition.map(u);

                                    partition_t id = bc.ids(j);
                                    weight_t w = bc.weights(j);

                                    if (id != u_id && id != partition.k) {
                                        local_comm_cost += w;
                                    }
                                },
                                total_comm_cost);
        Kokkos::fence();
        */

        return total_comm_cost;
    }
}

#endif //GPU_HEIPA_EDGE_CUT_H
