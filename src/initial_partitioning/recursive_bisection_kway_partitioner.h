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

#ifndef GPU_HEIPA_RECURSIVE_BISECTION_KWAY_PARTITIONER_H
#define GPU_HEIPA_RECURSIVE_BISECTION_KWAY_PARTITIONER_H

#include <string>
#include <numeric>

#include "../datastructures/graph.h"
#include "../datastructures/host_graph.h"
#include "../utility/definitions.h"
#include "recursive_bisection_2way_partitioner.h"

namespace GPU_HeiPa {
    inline void internal_partition(HostGraph &g,
                                   partition_t k,
                                   f64 imbalance,
                                   u64 seed,
                                   HostPartition &partition,
                                   partition_t k_min,
                                   partition_t k_max,
                                   std::vector<vertex_t> &n_to_o,
                                   ScratchMemory &scratch_memory,
                                   vertex_t max_n) {
        if (g.n == 0) { return; }

        if (k == 1) {
            for (vertex_t u = 0; u < g.n; u++) {
                vertex_t old_u = n_to_o[u];
                partition[old_u] = k_min;
            }
            return;
        }

        partition_t k_left = k / 2;
        partition_t k_right = k - k_left;

        double max_per_block = (1.0 + imbalance) * ((double) g.g_weight / (double) k);
        weight_t left_lmax = (weight_t) std::ceil(max_per_block * (double) k_left);
        weight_t right_lmax = (weight_t) std::ceil(max_per_block * (double) k_right);

        std::vector<vertex_t> &temp_partition = scratch_memory.partition;
        temp_partition.resize(g.n);

        partition_2way(g, left_lmax, right_lmax, k_left, k_right, temp_partition, scratch_memory);

        // HostPartition t_partition = HostPartition("partition", max_n);
        // metis_partition(g, (int) 2, imbalance, seed, t_partition, METIS_KWAY);
        // for (vertex_t u = 0; u < g.n; u++) {
        //     temp_partition[u] = t_partition(u);
        // }

        // std::cout << "top: " << g.n << " " << g.m << " " << g.g_weight << " " << k << " " << edge_cut(g, temp_partition) << std::endl;

        if (k == 2) {
            for (vertex_t u = 0; u < g.n; u++) {
                vertex_t old_u = n_to_o[u];
                partition[old_u] = k_min + temp_partition[u];
            }
            return;
        }
        HostGraph left_g;
        std::vector<vertex_t> left_n_to_o;
        std::vector<vertex_t> left_o_to_n;
        HostGraph right_g;
        std::vector<vertex_t> right_n_to_o(max_n);
        std::vector<vertex_t> right_o_to_n(max_n);
        //
        {
            ScopedTimer _t("initial_partitioning", "internal_partition", "determine_left_right_stats");

            left_g.n = 0;
            left_g.m = 0;
            left_g.g_weight = 0;
            left_n_to_o.resize(max_n);
            left_o_to_n.resize(max_n);

            right_g.n = 0;
            right_g.m = 0;
            right_g.g_weight = 0;
            right_n_to_o.resize(max_n);
            right_o_to_n.resize(max_n);

            for (vertex_t u = 0; u < g.n; u++) {
                ASSERT(temp_partition[u] == 0 || temp_partition[u] == 1);

                if (temp_partition[u] == 0) {
                    left_n_to_o[left_g.n] = n_to_o[u];
                    left_o_to_n[n_to_o[u]] = left_g.n;
                    left_g.n += 1;
                    left_g.g_weight += g.weights(u);
                    for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); i++) {
                        vertex_t v = g.edges_v(i);
                        if (temp_partition[v] == 0) { left_g.m += 1; }
                    }
                }
                if (temp_partition[u] == 1) {
                    right_n_to_o[right_g.n] = n_to_o[u];
                    right_o_to_n[n_to_o[u]] = right_g.n;
                    right_g.n += 1;
                    right_g.g_weight += g.weights(u);
                    for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); i++) {
                        vertex_t v = g.edges_v(i);
                        if (temp_partition[v] == 1) { right_g.m += 1; }
                    }
                }
            }
        }

        // std::cout << "left: " << left_g.n << " " << left_g.m << " " << left_g.g_weight << std::endl;
        // std::cout << "right: " << right_g.n << " " << right_g.m << " " << right_g.g_weight << std::endl;
        //
        {
            ScopedTimer _t("initial_partitioning", "internal_partition", "left_right_allocate");
            allocate_memory(left_g, left_g.n, left_g.m, left_g.g_weight);
            allocate_memory(right_g, right_g.n, right_g.m, right_g.g_weight);
        }
        //
        {
            ScopedTimer _t("initial_partitioning", "internal_partition", "left_right_build");

            vertex_t idx_left = 0;
            vertex_t idx_right = 0;
            for (vertex_t u = 0; u < g.n; u++) {
                if (temp_partition[u] == 0) {
                    vertex_t sub_u = left_o_to_n[n_to_o[u]];

                    left_g.weights(sub_u) = g.weights(u);

                    for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); i++) {
                        vertex_t v = g.edges_v(i);
                        weight_t w = g.edges_w(i);

                        if (temp_partition[v] == 0) {
                            vertex_t sub_v = left_o_to_n[n_to_o[v]];

                            left_g.edges_v(idx_left) = sub_v;
                            left_g.edges_w(idx_left) = w;
                            idx_left += 1;
                        }
                    }
                    left_g.neighborhood(sub_u + 1) = idx_left;
                }
                if (temp_partition[u] == 1) {
                    vertex_t sub_u = right_o_to_n[n_to_o[u]];

                    right_g.weights(sub_u) = g.weights(u);

                    for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); i++) {
                        vertex_t v = g.edges_v(i);
                        weight_t w = g.edges_w(i);

                        if (temp_partition[v] == 1) {
                            vertex_t sub_v = right_o_to_n[n_to_o[v]];

                            right_g.edges_v(idx_right) = sub_v;
                            right_g.edges_w(idx_right) = w;
                            idx_right += 1;
                        }
                    }
                    right_g.neighborhood(sub_u + 1) = idx_right;
                }
            }
        }

        internal_partition(left_g, k_left, imbalance, seed, partition, k_min, k_min + k_left, left_n_to_o, scratch_memory, max_n);
        internal_partition(right_g, k_right, imbalance, seed, partition, k_min + k_left, k_max, right_n_to_o, scratch_memory, max_n);
    }

    inline void recursive_bisec_partition(HostGraph &g,
                                          partition_t k,
                                          f64 imbalance,
                                          u64 seed,
                                          HostPartition &partition) {
        if (k == 1) {
            for (vertex_t u = 0; u < g.n; u++) { partition[u] = 0; }
            return;
        }

        ScopedTimer _t("initial_partitioning", "recursive_bisec_partition", "init_memory");

        ScratchMemory scratch_memory;

        std::vector<vertex_t> n_to_o(g.n);
        std::iota(n_to_o.begin(), n_to_o.end(), 0);
        _t.stop();

        internal_partition(g, k, imbalance, seed, partition, 0, k, n_to_o, scratch_memory, g.n);
    }
}

#endif //GPU_HEIPA_RECURSIVE_BISECTION_KWAY_PARTITIONER_H
