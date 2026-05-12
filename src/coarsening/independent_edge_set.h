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

#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "../datastructures/mapping.h"

namespace GPU_HeiPa {

    /**
     * @brief Computes a coarsening mapping based on an independent edge set (matching).
     * 
     * This scheme finds a maximal matching in the graph and contracts each matched edge
     * into a single vertex in the coarse graph. Unmatched vertices are kept as singletons.
     * The implementation is edge-parallel: edges are rated, and an edge is selected if
     * its rating is the best among its "neighborhood" (incident edges).
     */
    template <bool uniform_v_weights, bool uniform_e_weights>
    inline Mapping independent_edge_set_get_mapping(const Graph &g,
                                                   const Partition &partition,
                                                   const weight_t &lmax,
                                                   KokkosMemoryStack &mem_stack,
                                                   DeviceExecutionSpace &exec_space) {
        ScopedTimer _t_total("coarsening", "IndependentEdgeSet", "total");

        const vertex_t n = g.n;
        const u32 m = g.m;
        if (n == 0) return initialize_mapping(0, 0, mem_stack);

        std::cout << "Starting Independent Edge Set Coarsening (n=" << n << ", m=" << m << ")..." << std::endl;

        constexpr vertex_t SENTINEL = std::numeric_limits<vertex_t>::max();

        // Temporary buffers
        // matched[u] stores the representative vertex of the match, or SENTINEL
        UnmanagedDeviceVertex matched = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n), n);
        // edge_ratings stores the rating for each directed edge entry
        UnmanagedDeviceU64 edge_ratings = UnmanagedDeviceU64((u64 *) get_chunk_back(mem_stack, sizeof(u64) * m), m);
        // vertex_max_rating stores the maximum rating of any edge incident to a vertex
        UnmanagedDeviceU64 vertex_max_rating = UnmanagedDeviceU64((u64 *) get_chunk_back(mem_stack, sizeof(u64) * n), n);

        Kokkos::deep_copy(exec_space, matched, SENTINEL);

        // Simple hash function for randomized tie-breaking
        auto simple_hash = KOKKOS_LAMBDA(u32 key) -> u32 {
            key ^= key << 13;
            key ^= key >> 17;
            key ^= key << 5;
            return key;
        };

        const u32 seed = 1234567u;
        const int n_rounds = 2; // Perform multiple iterations to increase matching size

        for (int round = 0; round < n_rounds; ++round) {
            // Phase 1: Rate each directed edge entry in parallel
            // Ratings are stable for undirected edges: rating(u,v) == rating(v,u)
            Kokkos::parallel_for("IES_rate_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, m), KOKKOS_LAMBDA(const u32 i) {
                const vertex_t u = g.edges_u(i);
                const vertex_t v = g.edges_v(i);
                
                // Only rate edges between currently unmatched vertices
                if (u != v && matched(u) == SENTINEL && matched(v) == SENTINEL) {
                    const weight_t ew = uniform_e_weights ? 1 : g.edges_w(i);
                    const vertex_t min_uv = u < v ? u : v;
                    const vertex_t max_uv = u < v ? v : u;
                    
                    // Combine edge weight (high bits) and a hash (low bits) for stable tie-breaking
                    edge_ratings(i) = (static_cast<u64>(ew) << 32) | simple_hash(min_uv ^ max_uv ^ seed ^ (u32)round);
                } else {
                    edge_ratings(i) = 0;
                }
            });

            // Phase 2: Each vertex identifies the maximum rating among its incident edges
            Kokkos::deep_copy(exec_space, vertex_max_rating, 0);
            Kokkos::parallel_for("IES_vertex_max_rating", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, m), KOKKOS_LAMBDA(const u32 i) {
                const vertex_t u = g.edges_u(i);
                const u64 r = edge_ratings(i);
                if (r > 0) {
                    Kokkos::atomic_max(&vertex_max_rating(u), r);
                }
            });

            // Phase 3: Selection - an edge is selected if its rating is the best for both endpoints
            Kokkos::parallel_for("IES_select_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, m), KOKKOS_LAMBDA(const u32 i) {
                const vertex_t u = g.edges_u(i);
                const vertex_t v = g.edges_v(i);
                
                // Only process each undirected edge once
                if (u < v) {
                    const u64 r = edge_ratings(i);
                    if (r > 0 && r == vertex_max_rating(u) && r == vertex_max_rating(v)) {
                        // Edge is selected for the independent set
                        matched(u) = u; 
                        matched(v) = u;
                    }
                }
            });
            exec_space.fence();

            vertex_t matched_count = 0;
            Kokkos::parallel_reduce("IES_count_matches", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &update) {
                if (matched(u) != SENTINEL) update++;
            }, matched_count);
            exec_space.fence();
        }

        // Finalize: unmatched vertices become singletons (map to themselves)
        Kokkos::parallel_for("IES_finalize_matches", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            if (matched(u) == SENTINEL) {
                matched(u) = u;
            }
        });

        // Step 4: Enumerate coarse vertices
        vertex_t nc = 0;
        // Reuse vertex_max_rating memory for coarse IDs to save space
        UnmanagedDeviceVertex coarse_ids = UnmanagedDeviceVertex((vertex_t*)vertex_max_rating.data(), n);
        
        Kokkos::parallel_scan("IES_enumerate_coarse", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
            // A vertex is a representative if matched[u] == u
            if (matched(u) == u) {
                if (final) coarse_ids(u) = update;
                update++;
            }
        }, nc);

        // Step 5: Build final Mapping
        Mapping mapping = initialize_mapping(n, nc, mem_stack);
        Kokkos::parallel_for("IES_build_mapping", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            const vertex_t rep = matched(u);
            mapping.mapping(u) = coarse_ids(rep);
        });

        // Ensure all kernels using temporary buffers are finished before popping them
        exec_space.fence();
        std::cout << "Coarsening finished. Coarse vertices: " << nc << " (reduction factor: " << (f64)n/nc << ")" << std::endl;

        // Cleanup temporary buffers from the stack
        pop_back(mem_stack); // vertex_max_rating
        pop_back(mem_stack); // edge_ratings
        pop_back(mem_stack); // matched

        return mapping;
    }

} // namespace GPU_HeiPa

#endif // GPU_HEIPA_INDEPENDENT_EDGE_SET_H
