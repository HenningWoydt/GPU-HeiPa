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

#ifndef GPU_HEIPA_COMM_COST_H
#define GPU_HEIPA_COMM_COST_H

#include "definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../distance_oracles/distance_oracle_helpers.h"

namespace GPU_HeiPa {
    template<typename d_oracle_t>
    inline weight_t comm_cost(const Graph &g,
                              const Partition &partition,
                              d_oracle_t &d_oracle) {
        weight_t sum = 0;

        Kokkos::parallel_reduce("comm_cost", g.m, KOKKOS_LAMBDA(const u32 i, weight_t &local_sum) {
            vertex_t u = g.edges_u(i);
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);

            partition_t u_id = partition.map(u);
            partition_t v_id = partition.map(v);

            local_sum += w * get(d_oracle, u_id, v_id);
        }, sum);
        Kokkos::fence();

        return sum;
    }

    template<typename d_oracle_t>
    inline weight_t comm_cost_update(weight_t old_comm_cost,
                                     const Graph &g,
                                     const Partition &partition,
                                     const UnmanagedDevicePartition &old_map,
                                     const UnmanagedDeviceVertex &moves,
                                     const u32 n_moves,
                                     d_oracle_t &d_oracle) {
        weight_t delta = 0;
        Kokkos::parallel_reduce("comm_cost", n_moves, KOKKOS_LAMBDA(const u32 i, weight_t &local_delta) {
            vertex_t u = moves(i);
            partition_t old_u_id = old_map(u);
            partition_t new_u_id = partition.map(u);

            // NEW: skip fake moves
            if (old_u_id == new_u_id) {
                return;
            }

            for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                vertex_t v = g.edges_v(j);
                weight_t w = g.edges_w(j);

                partition_t old_v_id = old_map(v);
                partition_t new_v_id = partition.map(v);

                weight_t old_d = get(d_oracle, old_u_id, old_v_id);
                weight_t new_d = get(d_oracle, new_u_id, new_v_id);

                weight_t contrib = w * (new_d - old_d);

                // If the global cost counts each undirected edge twice (symmetric CSR):
                // - edges with exactly one moved endpoint are only visited once here,
                //   so we need to double their contribution.
                bool u_moved = (old_u_id != new_u_id);
                bool v_moved = (old_v_id != new_v_id);
                if (u_moved && !v_moved) {
                    contrib *= 2;
                }

                local_delta += contrib;
            }
        }, delta);

        return old_comm_cost + delta;
    }
}

#endif //GPU_HEIPA_COMM_COST_H
