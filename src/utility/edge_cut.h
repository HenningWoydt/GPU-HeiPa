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
                                },
                                sum);
        Kokkos::fence();

        return sum;
    }

    inline weight_t edge_cut(const BlockConnectivity &bc,
                             const Partition &partition) {
        weight_t total_edge_cut = 0;
        Kokkos::parallel_reduce("edge_cut", bc.size, KOKKOS_LAMBDA(const u32 i, weight_t &local_edge_cut) {
                                    vertex_t u = bc.us(i);
                                    partition_t u_id = partition.map(u);

                                    partition_t id = bc.ids(i);
                                    weight_t w = bc.weights(i);

                                    bool not_self = u_id != id;
                                    bool not_sentinel = id != partition.k;

                                    local_edge_cut += (not_self * not_sentinel) * w;
                                },
                                total_edge_cut);

        return total_edge_cut;
    }

    inline weight_t edge_cut_update(weight_t old_edge_cut,
                                    const Graph &g,
                                    const Partition &partition,
                                    const UnmanagedDeviceMove &to_move_list,
                                    const u32 list_size,
                                    KokkosMemoryStack &mem_stack) {
        auto *old_map_ptr = (partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * g.n);
        UnmanagedDevicePartition old_map = UnmanagedDevicePartition(old_map_ptr, g.n);
        Kokkos::deep_copy(old_map, partition.map);

        Kokkos::parallel_for("edge_cut", list_size, KOKKOS_LAMBDA(const u32 i) {
            Move m = to_move_list(i);
            vertex_t u = m.u;
            partition_t old_u_id = m.old_id;

            old_map(u) = old_u_id;
        });

        weight_t delta = 0;
        Kokkos::parallel_reduce("edge_cut", list_size, KOKKOS_LAMBDA(const u32 i, weight_t &local_delta) {
                                    Move m = to_move_list(i);
                                    vertex_t u = m.u;
                                    partition_t old_u_id = m.old_id;
                                    partition_t new_u_id = m.new_id;

                                    for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                                        vertex_t v = g.edges_v(j);
                                        weight_t w = g.edges_w(j);

                                        partition_t old_v_id = old_map(v);
                                        partition_t new_v_id = partition.map(v);

                                        local_delta += w * ((old_u_id != old_v_id) - (new_u_id != new_v_id));
                                    }
                                },
                                delta);

        pop_back(mem_stack);

        return old_edge_cut - 2*delta;
    }
}

#endif //GPU_HEIPA_EDGE_CUT_H
