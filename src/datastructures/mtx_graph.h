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

#ifndef GPU_HEIPA_MTX_GRAPH_H
#define GPU_HEIPA_MTX_GRAPH_H

#include <Kokkos_Core.hpp>
#include "../definitions.h"
#include "kokkos_memory_stack.h"

namespace GPU_HeiPa {

    template<vertex_t N>
    struct MtxGraph {
        static constexpr vertex_t MAX_N = N;

        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;
        bool uniform_vertex_weights = false;
        bool uniform_edge_weights = false;

        weight_t weights[N];
        weight_t adj_matrix[N * N];

        KOKKOS_INLINE_FUNCTION
        MtxGraph() {
            reset();
        }

        KOKKOS_INLINE_FUNCTION
        void reset() {
            n = 0;
            m = 0;
            g_weight = 0;
            uniform_vertex_weights = false;
            uniform_edge_weights = false;
            for (vertex_t i = 0; i < N; ++i) {
                weights[i] = 0;
            }
            for (vertex_t i = 0; i < N * N; ++i) {
                adj_matrix[i] = 0;
            }
        }

        KOKKOS_INLINE_FUNCTION
        weight_t edge_weight(vertex_t u, vertex_t v) const {
            return adj_matrix[u * N + v];
        }

        KOKKOS_INLINE_FUNCTION
        weight_t &edge_weight(vertex_t u, vertex_t v) {
            return adj_matrix[u * N + v];
        }

        KOKKOS_INLINE_FUNCTION
        bool has_edge(vertex_t u, vertex_t v) const {
            return adj_matrix[u * N + v] != 0;
        }

        KOKKOS_INLINE_FUNCTION
        void set_edge(vertex_t u, vertex_t v, weight_t w) {
            adj_matrix[u * N + v] = w;
        }

        KOKKOS_INLINE_FUNCTION
        void add_edge_undirected(vertex_t u, vertex_t v, weight_t w) {
            adj_matrix[u * N + v] += w;
            adj_matrix[v * N + u] += w;
        }

        KOKKOS_INLINE_FUNCTION
        weight_t vertex_weight(vertex_t u) const {
            return uniform_vertex_weights ? 1 : weights[u];
        }

        KOKKOS_INLINE_FUNCTION
        weight_t degree(vertex_t u) const {
            weight_t deg = 0;
            for (vertex_t v = 0; v < n; ++v) {
                deg += adj_matrix[u * N + v];
            }
            return deg;
        }
    };

    template<vertex_t N>
    struct BatchMtxGraph {
        static constexpr vertex_t MAX_N = N;

        partition_t k = 0;

        // Flattened array of MtxGraph<N> for all k graphs in device memory
        Kokkos::View<MtxGraph<N> *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > graphs;

        // Partition assignments (size: total max vertices across batch, or k * N)
        UnmanagedDevicePartition partitions;

        // Mapping from local vertex id (0..N-1) to original global graph vertex id
        UnmanagedDeviceVertex global_ids;

        KOKKOS_INLINE_FUNCTION
        MtxGraph<N> &get_graph(partition_t id) const {
            return graphs(id);
        }

        KOKKOS_INLINE_FUNCTION
        partition_t *get_partition_ptr(partition_t id) const {
            return partitions.data() + (u64) id * N;
        }

        KOKKOS_INLINE_FUNCTION
        vertex_t *get_global_ids_ptr(partition_t id) const {
            return global_ids.data() + (u64) id * N;
        }
    };

    template<vertex_t N>
    inline void init_BatchMtxGraph(BatchMtxGraph<N> &batch,
                                  partition_t k,
                                  KokkosMemoryStack &mem_stack) {
        batch.k = k;

        u64 graphs_bytes = sizeof(MtxGraph<N>) * k;
        batch.graphs = Kokkos::View<MtxGraph<N> *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
            (MtxGraph<N> *) get_chunk_front(mem_stack, graphs_bytes), k);

        u64 part_bytes = sizeof(partition_t) * k * N;
        batch.partitions = UnmanagedDevicePartition(
            (partition_t *) get_chunk_front(mem_stack, part_bytes), k * N);

        u64 gids_bytes = sizeof(vertex_t) * k * N;
        batch.global_ids = UnmanagedDeviceVertex(
            (vertex_t *) get_chunk_front(mem_stack, gids_bytes), k * N);
    }

    template<vertex_t N>
    inline void free_BatchMtxGraph(BatchMtxGraph<N> &batch,
                                  KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack); // global_ids
        pop_front(mem_stack); // partitions
        pop_front(mem_stack); // graphs
    }

} // namespace GPU_HeiPa

#endif // GPU_HEIPA_MTX_GRAPH_H

