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

#ifndef GPU_HEIPA_BLOCK_CONNECTIVITY_H
#define GPU_HEIPA_BLOCK_CONNECTIVITY_H

#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    struct BlockConnectivity {
        DeviceU32 row;
        DeviceVertex us;
        DevicePartition ids;
        DeviceWeight weights;
        u32 size = 0;
    };

    inline BlockConnectivity rebuild_scratch(const Graph &g,
                                             const Partition &partition) {
        BlockConnectivity bc;
        bc.row = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "row"), g.n + 1);
        Kokkos::deep_copy(Kokkos::subview(bc.row, 0), 0);
        Kokkos::fence();

        Kokkos::parallel_scan("set_rows", g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &carry, const bool final) {
            u32 len = g.neighborhood(u + 1) - g.neighborhood(u);
            const u32 c = len < partition.k ? len : partition.k;
            if (final) bc.row(u + 1) = carry + c; // write inclusive, row[0] already 0
            carry += c;
        });
        Kokkos::fence();

        Kokkos::deep_copy(bc.size, Kokkos::subview(bc.row, g.n));
        Kokkos::fence();

        bc.us = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "us"), bc.size);
        bc.ids = DevicePartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ids"), bc.size);
        bc.weights = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weights"), bc.size);
        Kokkos::deep_copy(bc.ids, partition.k);
        Kokkos::deep_copy(bc.weights, 0);
        Kokkos::fence();

        // set u values
        Kokkos::parallel_for("fill", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            for (size_t j = bc.row(u); j < bc.row(u + 1); j++) {
                bc.us(j) = u;
            }
        });
        Kokkos::fence();

        Kokkos::parallel_for("fill", g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = g.edges_u(i);
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);

            u32 r_beg = bc.row(u);
            u32 r_end = bc.row(u + 1);
            u32 r_len = r_end - r_beg;

            partition_t v_id = partition.map(v);

            u32 j = r_beg + hash32(v_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), partition.k, v_id);
                if (val == partition.k || val == v_id) {
                    Kokkos::atomic_add(&bc.weights(j), w);
                    break;
                }
                j += 1;
            }
        });
        Kokkos::fence();

        return bc;
    }

    inline BlockConnectivity rebuild(BlockConnectivity &old_bc,
                                     DeviceU32 &needs_more,
                                     const Graph &g,
                                     const Partition &partition,
                                     const DeviceU32 &to_move,
                                     const DeviceU64 &weight_id) {
        BlockConnectivity bc;
        bc.row = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "row"), g.n + 1);
        Kokkos::deep_copy(Kokkos::subview(bc.row, 0), 0);
        Kokkos::fence();

        Kokkos::parallel_scan("set_rows", g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &carry, const bool final) {
            u32 deg = g.neighborhood(u + 1) - g.neighborhood(u);
            u32 old_c = old_bc.row(u + 1) - old_bc.row(u);
            u32 add = needs_more(u) == 0 ? 0 : needs_more(u) + 2; // additional slots requested
            u32 max_c = deg < partition.k ? deg : partition.k;

            // New capacity = clamp(old + add, 0..min(deg, k))
            u32 target = old_c + add;
            u32 c = target < max_c ? target : max_c;

            if (final) bc.row(u + 1) = carry + c;
            carry += c;
        });
        Kokkos::fence();

        Kokkos::deep_copy(bc.size, Kokkos::subview(bc.row, g.n));
        Kokkos::fence();

        bc.us = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "us"), bc.size);
        bc.ids = DevicePartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ids"), bc.size);
        bc.weights = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weights"), bc.size);
        Kokkos::deep_copy(bc.ids, partition.k);
        Kokkos::deep_copy(bc.weights, 0);
        Kokkos::fence();

        // set u values
        Kokkos::parallel_for("fill", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            for (size_t j = bc.row(u); j < bc.row(u + 1); j++) {
                bc.us(j) = u;
            }
        });
        Kokkos::fence();

        // copy old values
        Kokkos::parallel_for("fill", old_bc.size, KOKKOS_LAMBDA(const u32 old_j) {
            vertex_t u = old_bc.us(old_j);
            if (needs_more(u) >= 1) { return; }

            u32 old_r_beg = old_bc.row(u);
            u32 new_r_beg = bc.row(u);
            u32 new_j = old_j - old_r_beg;

            bc.ids(new_r_beg + new_j) = old_bc.ids(old_j);
            bc.weights(new_r_beg + new_j) = old_bc.weights(old_j);
        });

        // insert new values
        Kokkos::parallel_for("fill", g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = g.edges_u(i);
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);

            if (needs_more(u) == 0) { return; }

            partition_t v_id = partition.map(v);
            if (to_move(v) == 1) { v_id = unpack_partition(weight_id(v)); }

            u32 r_beg = bc.row(u);
            u32 r_end = bc.row(u + 1);
            u32 r_len = r_end - r_beg;

            u32 j = r_beg + hash32(v_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), partition.k, v_id);
                if (val == partition.k || val == v_id) {
                    Kokkos::atomic_add(&bc.weights(j), w);
                    break;
                }
                j += 1;
            }
        });
        Kokkos::fence();

        return bc;
    }

    inline void move(BlockConnectivity &bc,
                     const Graph &g,
                     const Partition &partition,
                     const DeviceU32 &to_move,
                     const DeviceU64 &weight_id) {
        Kokkos::parallel_for("remove_weight", g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = g.edges_u(i);
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);
            if (to_move(u) == 0) { return; }

            partition_t u_id = partition.map(u);

            // search in v's neighborhood for u_id
            u32 r_beg = bc.row(v);
            u32 r_end = bc.row(v + 1);
            u32 r_len = r_end - r_beg;

            u32 j = r_beg + hash32(u_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                if (bc.ids(j) == u_id) {
                    weight_t old_w = Kokkos::atomic_fetch_sub(&bc.weights(j), w);
                    if (old_w == w) {
                        bc.ids(j) = partition.k;
                        return;
                    }
                }
                j += 1;
            }
        });
        Kokkos::fence();

        DeviceU32 needs_more(Kokkos::view_alloc(Kokkos::WithoutInitializing, "needs_more"), g.n);
        Kokkos::deep_copy(needs_more, 0);
        Kokkos::fence();

        Kokkos::parallel_for("add_conn", g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = g.edges_u(i);
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);
            if (to_move(u) == 0) { return; }

            partition_t new_u_id = unpack_partition(weight_id(u));

            // search in v's neighborhood for new_u_id
            u32 r_beg = bc.row(v);
            u32 r_end = bc.row(v + 1);
            u32 r_len = r_end - r_beg;

            // first pass check if new_u_id exists anywhere
            u32 j = r_beg + hash32(new_u_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                if (bc.ids(j) == new_u_id) {
                    // found the spot, add the weight
                    Kokkos::atomic_add(&bc.weights(j), w);
                    return;
                }
                j += 1;
            }

            // new_u_id does not exist, now search for an empty spot
            j = r_beg + hash32(new_u_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), partition.k, new_u_id);
                if (val == partition.k || val == new_u_id) {
                    // found empty spot or good spot, add the weight
                    Kokkos::atomic_add(&bc.weights(j), w);
                    return;
                }
                j += 1;
            }

            // could not insert in v, it needs more space
            Kokkos::atomic_inc(&needs_more(v));
        });
        Kokkos::fence();

        u32 sum = 0;
        Kokkos::parallel_reduce("count_and_sum", g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &lsum) {
                                    lsum += needs_more(u);
                                },
                                Kokkos::Sum<u32>(sum)
        );
        Kokkos::fence();

        if (sum > 0) {
            // not enough space, we need to rebuild
            BlockConnectivity new_bc = rebuild(bc, needs_more, g, partition, to_move, weight_id);
            std::swap(bc, new_bc);
        }
    }
}

#endif //GPU_HEIPA_BLOCK_CONNECTIVITY_H
