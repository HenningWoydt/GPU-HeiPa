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

            partition_t u_id = partition.map(u);
            partition_t v_id = partition.map(v);

            local_sum += w * (u_id != v_id);
        }, sum);
        Kokkos::fence();

        return sum / 2;
    }

    inline weight_t edge_cut_update(weight_t old_edge_cut,
                                    const Graph &g,
                                    const Partition &partition,
                                    const UnmanagedDevicePartition &old_map,
                                    const UnmanagedDeviceVertex &moves,
                                    const u32 n_moves) {
        weight_t delta = 0;
        Kokkos::parallel_reduce("edge_cut", n_moves, KOKKOS_LAMBDA(const u32 i, weight_t &local_delta) {
            vertex_t u = moves(i);
            partition_t old_u_id = old_map(u);
            partition_t new_u_id = partition.map(u);

            for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                vertex_t v = g.edges_v(j);
                weight_t w = g.edges_w(j);

                partition_t old_v_id = old_map(v);
                partition_t new_v_id = partition.map(v);

                bool old_cut = (old_u_id != old_v_id);
                bool new_cut = (new_u_id != new_v_id);

                local_delta += w * (new_cut - old_cut);
            }
        }, delta);

        return old_edge_cut + delta / 2;
    }
}

#endif //GPU_HEIPA_EDGE_CUT_H
