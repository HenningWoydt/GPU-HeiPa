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

#ifndef GPU_HEIPA_INDEPENDENT_EDGE_SET_H
#define GPU_HEIPA_INDEPENDENT_EDGE_SET_H

#include <Kokkos_Core.hpp>

#include "../definitions.h"
#include "../utility/util.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "../datastructures/mapping.h"

#include <numeric>
#include <algorithm>
#include <random>

namespace GPU_HeiPa {
    /**
     * @brief Computes a coarsening mapping based on an independent edge set (matching) exclusively on the GPU.
     */
    template<bool uniform_v_weights, bool uniform_e_weights>
    inline Mapping independent_edge_set_get_mapping_gpu(const Graph &g,
                                                        const Partition &partition,
                                                        const weight_t &lmax,
                                                        KokkosMemoryStack &mem_stack,
                                                        DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("coarsening", "IndependentEdgeSetGPU", "total");

        const vertex_t n = g.n;
        if (n == 0) return initialize_mapping(0, 0, mem_stack);

        // 1. Allocate matching and hn arrays on GPU
        UnmanagedDeviceVertex matching((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n), n);
        UnmanagedDeviceVertex hn((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n), n);

        Kokkos::deep_copy(exec_space, matching, SENTINEL);
        Kokkos::deep_copy(exec_space, hn, SENTINEL);

        // 2. Pick phase: Each vertex picks its best neighbor in the same partition
        u32 seed = 12345u;
        Kokkos::parallel_for("IES_GPU_Pick", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_part = partition.map(u);
            vertex_t best_v = SENTINEL;
            f64 best_score = -1.0;
            u32 r = xs32(u ^ seed);

            for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                vertex_t v = g.edges_v(i);
                if (u == v || partition.map(v) != u_part) continue;

                weight_t uw = uniform_v_weights ? 1 : g.weights(u);
                weight_t vw = uniform_v_weights ? 1 : g.weights(v);
                weight_t ew = uniform_e_weights ? 1 : g.edges_w(i);

                if (uw + vw > lmax) continue;

                f64 rating = ((f64) ew * (f64) ew) / ((f64) uw * (f64) vw);
                u32 tb = xs32(v ^ r);
                f64 score = rating + (f64) tb / 4294967296.0 * 1e-9;

                if (score > best_score) {
                    best_score = score;
                    best_v = v;
                }
            }
            hn(u) = best_v;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // 3. Commit phase: Check for mutual picks and identify heads/tails
        Kokkos::parallel_for("IES_GPU_Commit", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t v = hn(u);
            if (v != SENTINEL && hn(v) == u) {
                matching(u) = (u < v) ? u : v; // Store head index
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // 3.5. 2-hop phase for unmatched vertices
        Kokkos::parallel_for("IES_GPU_2Hop", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            if (matching(u) != SENTINEL) return;
            partition_t u_part = partition.map(u);
            vertex_t best_v = SENTINEL;
            weight_t uw = uniform_v_weights ? 1 : g.weights(u);

            for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                vertex_t v = g.edges_v(j);
                for (u32 k = g.neighborhood(v); k < g.neighborhood(v + 1); k++) {
                    vertex_t w = g.edges_v(k);
                    if (w != u && matching(w) == SENTINEL && partition.map(w) == u_part) {
                        weight_t ww = uniform_v_weights ? 1 : g.weights(w);
                        if (uw + ww > lmax) continue;

                        best_v = w;
                        break;
                    }
                }
                if (best_v != SENTINEL) break;
            }

            if (best_v != SENTINEL) {
                vertex_t min_v = (u < best_v) ? u : best_v;
                vertex_t max_v = (u > best_v) ? u : best_v;
                
                if (Kokkos::atomic_compare_exchange(&matching(max_v), SENTINEL, min_v) == SENTINEL) {
                    if (Kokkos::atomic_compare_exchange(&matching(min_v), SENTINEL, min_v) != SENTINEL) {
                        Kokkos::atomic_exchange(&matching(max_v), SENTINEL); // rollback
                    }
                }
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        Kokkos::parallel_for("IES_GPU_Singletons", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            if (matching(u) == SENTINEL) matching(u) = u;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // 4. Scan phase: Assign coarse IDs to heads and singletons
        vertex_t nc = 0;
        Kokkos::parallel_scan("IES_GPU_Scan", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
            if (matching(u) == u) {
                if (final) hn(u) = update; // Overwrite hn with coarse ID for heads/singletons
                update++;
            }
        }, nc);
        KOKKOS_PROFILE_FENCE(exec_space);

        // 5. Build Mapping: Final assignment
        Mapping mapping = initialize_mapping(n, nc, mem_stack);
        Kokkos::parallel_for("IES_GPU_FillMapping", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            if (matching(u) == u) {
                mapping.mapping(u) = hn(u);
            } else {
                mapping.mapping(u) = hn(matching(u));
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);
        exec_space.fence();

        pop_back(mem_stack); // hn
        pop_back(mem_stack); // matching

        return mapping;
    }

    /**
     * @brief Computes a coarsening mapping based on an independent edge set (matching) on the CPU.
     * 
     * This version copies the graph to the host, performs a simple greedy matching,
     * and then uploads the result back to the device.
     */
    template<bool uniform_v_weights, bool uniform_e_weights>
    inline Mapping independent_edge_set_get_mapping(const Graph &g,
                                                    const Partition &partition,
                                                    const weight_t &lmax,
                                                    KokkosMemoryStack &mem_stack,
                                                    DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("coarsening", "IndependentEdgeSetCPU", "total");

        return independent_edge_set_get_mapping_gpu<uniform_v_weights, uniform_e_weights>(g, partition, lmax, mem_stack, exec_space);

        const vertex_t n = g.n;
        if (n == 0) return initialize_mapping(0, 0, mem_stack);

        // 1. Transfer graph and partition to host
        HostGraph h_g = to_host_graph(g, exec_space);
        PartitionHost h_p = to_host_partition(partition, exec_space);

        // 2. Perform greedy matching on CPU
        std::vector<vertex_t> matched(n, SENTINEL);
        std::vector<vertex_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);

        std::mt19937 rng(12345);
        std::shuffle(perm.begin(), perm.end(), rng);

        for (vertex_t u: perm) {
            if (matched[u] != SENTINEL) continue;

            vertex_t best_v = SENTINEL;
            f32 best_rating = -1.0f;

            partition_t u_part = h_p.map(u);

            for (u32 i = h_g.neighborhood(u); i < h_g.neighborhood(u + 1); ++i) {
                vertex_t v = h_g.edges_v(i);
                if (u == v || matched[v] != SENTINEL) continue;

                // Respect partition boundaries
                if (h_p.map(v) != u_part) continue;

                weight_t uw = uniform_v_weights ? 1 : h_g.weights(u);
                weight_t vw = uniform_v_weights ? 1 : h_g.weights(v);
                weight_t ew = uniform_e_weights ? 1 : h_g.edges_w(i);

                // Heavy-edge rating
                f32 rating = ((f32) ew * (f32) ew) / ((f32) uw * (f32) vw);
                if (rating > best_rating) {
                    best_rating = rating;
                    best_v = v;
                }
            }

            if (best_v != SENTINEL) {
                matched[u] = best_v;
                matched[best_v] = u;
            }
        }

        // 3. Compute coarse IDs
        std::vector<vertex_t> h_mapping(n);
        vertex_t nc = 0;
        for (vertex_t u = 0; u < n; ++u) {
            if (matched[u] == SENTINEL || u < matched[u]) {
                h_mapping[u] = nc++;
            }
        }
        for (vertex_t u = 0; u < n; ++u) {
            if (matched[u] != SENTINEL && u > matched[u]) {
                h_mapping[u] = h_mapping[matched[u]];
            }
        }

        // 4. Initialize device mapping and upload
        Mapping mapping = initialize_mapping(n, nc, mem_stack);

        HostVertex host_mapping_view("h_mapping_view", n);
        for (vertex_t u = 0; u < n; ++u) {
            host_mapping_view(u) = h_mapping[u];
        }

        Kokkos::deep_copy(exec_space, mapping.mapping, host_mapping_view);
        exec_space.fence();

        std::cout << "CPU Coarsening finished. Coarse vertices: " << nc << " (reduction factor: " << (f64) n / nc << ")" << std::endl;

        return mapping;
    }
} // namespace GPU_HeiPa

#endif // GPU_HEIPA_INDEPENDENT_EDGE_SET_H
