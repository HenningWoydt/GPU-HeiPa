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

#ifndef GPU_HEIPA_GRAPH_H
#define GPU_HEIPA_GRAPH_H

#include "host_graph.h"
#include "kokkos_memory_stack.h"
#include "mapping.h"
#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    struct Graph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;

        UnmanagedDeviceU8 memory;
        u64 n_bytes = 0;
        u64 n_pops = 5;

        UnmanagedDeviceWeight weights;
        UnmanagedDeviceVertex neighborhood;
        UnmanagedDeviceVertex edges_u;
        UnmanagedDeviceVertex edges_v;
        UnmanagedDeviceWeight edges_w;
    };

    inline Graph from_HostGraph(const HostGraph &host_g,
                                KokkosMemoryStack &mem_stack) {
        Graph g;
        // initialize the graph
        {
            ScopedTimer _t{"misc", "from_HostGraph", "initialize"};

            g.n = host_g.n;
            g.m = host_g.m;
            g.g_weight = host_g.g_weight;

            // bytes needed per array
            size_t off_weights = 0;
            size_t bytes_weights = round_up_64(sizeof(weight_t) * (size_t) g.n);

            size_t off_neighborhood = off_weights + bytes_weights;
            size_t bytes_neighborhood = round_up_64(sizeof(vertex_t) * (size_t) (g.n + 1));

            size_t off_edges_v = off_neighborhood + bytes_neighborhood;
            size_t bytes_edges_v = round_up_64(sizeof(vertex_t) * (size_t) g.m);

            size_t off_edges_w = off_edges_v + bytes_edges_v;
            size_t bytes_edges_w = round_up_64(sizeof(weight_t) * (size_t) g.m);

            size_t off_edges_u = off_edges_w + bytes_edges_w;
            size_t bytes_edges_u = round_up_64(sizeof(vertex_t) * (size_t) g.m);

            // total n bytes
            g.n_bytes = off_edges_u + bytes_edges_u;

            // allocate one owning chunk
            g.memory = UnmanagedDeviceU8((u8 *) get_chunk_front(mem_stack, sizeof(u8) * g.n_bytes), g.n_bytes);
            uint8_t *base = g.memory.data();
            g.n_pops = 1;

            // create unmanaged views into the chunk
            g.weights = UnmanagedDeviceWeight((weight_t *) (base + off_weights), g.n);
            g.neighborhood = UnmanagedDeviceVertex((vertex_t *) (base + off_neighborhood), g.n + 1);
            g.edges_v = UnmanagedDeviceVertex((vertex_t *) (base + off_edges_v), g.m);
            g.edges_w = UnmanagedDeviceWeight((weight_t *) (base + off_edges_w), g.m);
            g.edges_u = UnmanagedDeviceVertex((vertex_t *) (base + off_edges_u), g.m);
        }
        // copy the structure to device
        {
            ScopedTimer _t{"misc", "from_HostGraph", "copy"};
            Kokkos::deep_copy(Kokkos::subview(g.memory, std::make_pair((size_t) 0, (size_t) host_g.n_bytes)), host_g.memory);
            KOKKOS_PROFILE_FENCE();
        }
        // create the third array
        {
            ScopedTimer _t{"misc", "from_HostGraph", "fill_edges_u"};
            Kokkos::parallel_for("fill_edges_u", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = g.neighborhood(u);
                u32 end = g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    g.edges_u(i) = u;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        return g;
    }

    inline Graph from_Graph_Mapping(const Graph &old_g,
                                    const Mapping &mapping,
                                    KokkosMemoryStack &mem_stack) {
        assert_back_is_empty(mem_stack);

        Graph coarse_g;
        // initialize graphs
        {
            ScopedTimer t_initialize{"contraction", "from_Graph_Mapping", "initialize_graph"};

            coarse_g.n = mapping.coarse_n;
            coarse_g.g_weight = old_g.g_weight;

            coarse_g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.n), coarse_g.n);
            coarse_g.neighborhood = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * (coarse_g.n + 1)), coarse_g.n + 1);
        }

        UnmanagedDeviceVertex degrees;
        UnmanagedDeviceVertex sum_degrees;

        UnmanagedDeviceU32 hash_offsets;
        UnmanagedDeviceVertex hash_keys;
        UnmanagedDeviceWeight hash_vals;
        // initialize helpers
        {
            ScopedTimer t_initialize{"contraction", "from_Graph_Mapping", "initialize_helpers"};

            degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * coarse_g.n), coarse_g.n);
            sum_degrees = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * coarse_g.n), coarse_g.n);

            hash_offsets = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            hash_keys = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * old_g.m), old_g.m);
            hash_vals = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * old_g.m), old_g.m);
        }

        // set memory to 0
        {
            ScopedTimer t_initialize_set_0{"contraction", "from_Graph_Mapping", "initialize_set_0"};

            Kokkos::deep_copy(coarse_g.weights, 0);
            Kokkos::deep_copy(degrees, 0);
            Kokkos::deep_copy(sum_degrees, 0);
            Kokkos::deep_copy(hash_offsets, 0);
            Kokkos::deep_copy(hash_keys, coarse_g.n);
            Kokkos::deep_copy(hash_vals, 0);
            KOKKOS_PROFILE_FENCE();
        }

        // determine weight of new vertices and maximum degree
        {
            ScopedTimer t_max_degrees{"contraction", "from_Graph_Mapping", "max_degrees"};

            Kokkos::parallel_for("max_degrees", old_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t deg = old_g.neighborhood(u + 1) - old_g.neighborhood(u);
                weight_t u_w = old_g.weights(u);
                vertex_t v = mapping.mapping(u);

                MY_KOKKOS_ASSERT(u_w > 0);
                MY_KOKKOS_ASSERT(v < coarse_g.n);

                Kokkos::atomic_add(&sum_degrees(v), deg);
                Kokkos::atomic_add(&coarse_g.weights(v), u_w);
            });
            KOKKOS_PROFILE_FENCE();
        }

        // prefix sum over all degrees
        {
            ScopedTimer t_max_degrees_prefix_sum{"contraction", "from_Graph_Mapping", "max_degrees_prefix_sum"};

            Kokkos::parallel_scan("prefix_sum_offsets", coarse_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                u32 cnt = i < coarse_g.n ? sum_degrees(i) : 0;
                if (final) hash_offsets(i) = running;
                running += cnt;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // hash edges - edge parallel
        {
            ScopedTimer t_hash_edges{"contraction", "from_Graph_Mapping", "hash_edges"};

            Kokkos::parallel_for("hash_edges", old_g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = old_g.edges_u(i);
                vertex_t v = old_g.edges_v(i);
                weight_t w = old_g.edges_w(i);

                MY_KOKKOS_ASSERT(u < old_g.n);
                MY_KOKKOS_ASSERT(v < old_g.n);
                MY_KOKKOS_ASSERT(w > 0);

                vertex_t u_new = mapping.mapping(u);
                vertex_t v_new = mapping.mapping(v);

                MY_KOKKOS_ASSERT(u_new < coarse_g.n);
                MY_KOKKOS_ASSERT(v_new < coarse_g.n);

                if (u_new == v_new) { return; } // this edge vanishes

                u32 beg = hash_offsets(u_new);
                u32 end = hash_offsets(u_new + 1);
                u32 len = end - beg;
                if (len == 0) { return; }

                u32 idx = beg + hash32(v_new) % len;
                for (u32 j = 0; j < len; ++j) {
                    if (idx == end) { idx = beg; }
                    vertex_t old = Kokkos::atomic_compare_exchange(&hash_keys(idx), coarse_g.n, v_new);
                    if (old == coarse_g.n || old == v_new) {
                        Kokkos::atomic_add(&hash_vals(idx), w);
                        if (old == coarse_g.n) { Kokkos::atomic_inc(&degrees(u_new)); }
                        break;
                    }
                    idx += 1;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        // hash edges - vertex parallel
        {
            /*
            ScopedTimer t_hash_edges{"contraction", "from_Graph_Mapping", "hash_edges"};

            using TeamPolicy = Kokkos::TeamPolicy<DeviceExecutionSpace, Kokkos::IndexType<u32> >;
            Kokkos::parallel_for("hash_edges", TeamPolicy((int) old_g.n, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamPolicy::member_type &t) {
                vertex_t u = t.league_rank();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, old_g.neighborhood(u), old_g.neighborhood(u + 1)), [=](const u32 i) {
                    vertex_t v = old_g.edges_v(i);
                    weight_t w = old_g.edges_w(i);

                    vertex_t u_new = mapping.mapping(u);
                    vertex_t v_new = mapping.mapping(v);

                    MY_KOKKOS_ASSERT(u_new < coarse_g.n);
                    MY_KOKKOS_ASSERT(v_new < coarse_g.n);

                    if (u_new == v_new) { return; } // this edge vanishes

                    u32 beg = hash_offsets(u_new);
                    u32 end = hash_offsets(u_new + 1);
                    u32 len = end - beg;
                    if (len == 0) { return; }

                    u32 idx = beg + hash32(v_new) % len;
                    for (u32 j = 0; j < len; ++j) {
                        if (idx == end) { idx = beg; }
                        vertex_t old = Kokkos::atomic_compare_exchange(&hash_keys(idx), coarse_g.n, v_new);
                        if (old == coarse_g.n || old == v_new) {
                            Kokkos::atomic_add(&hash_vals(idx), w);
                            if (old == coarse_g.n) { Kokkos::atomic_inc(&degrees(u_new)); }
                            break;
                        }
                        idx += 1;
                    }
                });
            });
            KOKKOS_PROFILE_FENCE();
            */
        }

        // build offsets
        {
            ScopedTimer t_build_offsets{"contraction", "from_Graph_Mapping", "build_offsets"};

            Kokkos::parallel_scan("build_offsets", coarse_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                const u32 cnt = i < coarse_g.n ? degrees(i) : 0;
                if (final) coarse_g.neighborhood(i) = running;
                running += cnt;
            });
            Kokkos::deep_copy(coarse_g.m, Kokkos::subview(coarse_g.neighborhood, coarse_g.n)); // copy final number of edges m

            KOKKOS_PROFILE_FENCE();
        }

        // allocate edges for coarse graph
        {
            ScopedTimer t_allocate_edges{"contraction", "from_Graph_Mapping", "allocate_edges"};

            coarse_g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
            coarse_g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
            coarse_g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.m), coarse_g.m);

            KOKKOS_PROFILE_FENCE();
        }

        // fill the coarse graph
        {
            ScopedTimer t_fill_coarse_graph{"contraction", "from_Graph_Mapping", "fill_coarse_graph"};

            Kokkos::parallel_for("materialize_edges", coarse_g.n, KOKKOS_LAMBDA(const vertex_t u_new) {
                u32 out = coarse_g.neighborhood(u_new);
                u32 end = coarse_g.neighborhood(u_new + 1);

                u32 off = hash_offsets(u_new);
                u32 size = hash_offsets(u_new + 1) - off;

                for (u32 idx = 0; idx < size && out < end; ++idx) {
                    u32 pos = off + idx;
                    vertex_t v = hash_keys(pos);
                    weight_t w = hash_vals(pos);
                    if (v != coarse_g.n) {
                        coarse_g.edges_v(out) = v;
                        coarse_g.edges_w(out) = w;
                        ++out;

                        MY_KOKKOS_ASSERT(v < coarse_g.n);
                        MY_KOKKOS_ASSERT(w > 0);
                    }
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        // fill the u array
        {
            ScopedTimer t_fill_coarse_u{"contraction", "from_Graph_Mapping", "fill_coarse_u"};

            Kokkos::parallel_for("fill_edges_u", coarse_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 begin = coarse_g.neighborhood(u);
                u32 end = coarse_g.neighborhood(u + 1);
                for (u32 i = begin; i < end; ++i) {
                    coarse_g.edges_u(i) = u;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        // deallocate mem
        {
            ScopedTimer t_fill_coarse_u{"contraction", "from_Graph_Mapping", "deallocate"};

            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);
            pop_back(mem_stack);

            KOKKOS_PROFILE_FENCE();
        }
        assert_back_is_empty(mem_stack);

        return coarse_g;
    }

    inline HostGraph to_host_graph(const Graph &device_g) {
        HostGraph host_g;

        allocate_memory(host_g, device_g.n, device_g.m, device_g.g_weight);

        Kokkos::deep_copy(host_g.weights, device_g.weights);
        Kokkos::deep_copy(host_g.neighborhood, device_g.neighborhood);
        Kokkos::deep_copy(host_g.edges_v, device_g.edges_v);
        Kokkos::deep_copy(host_g.edges_w, device_g.edges_w);
        Kokkos::fence();

        return host_g;
    }

    inline void free_graph(Graph &g,
                           KokkosMemoryStack &mem_stack) {
        for (u64 i = 0; i < g.n_pops; ++i) {
            pop_front(mem_stack);
        }
    }

    inline void print_graph_stats(const HostGraph &g, std::ostream &os = std::cout) {
        using std::numeric_limits;

        os << "========================================\n";
        os << " Graph statistics\n";
        os << "========================================\n";

        os << "  vertices (n): " << g.n << "\n";
        os << "  edges   (m): " << g.m << "\n";
        os << "  g_weight   : " << g.g_weight << "\n";

        if (g.n == 0) {
            os << "  (empty graph)\n";
            os << "========================================\n";
            return;
        }

        // Degree stats
        vertex_t min_deg = numeric_limits<vertex_t>::max();
        vertex_t max_deg = 0;
        std::uint64_t sum_deg = 0;
        std::uint64_t num_isolated = 0;

        for (vertex_t u = 0; u < g.n; ++u) {
            vertex_t begin = g.neighborhood(u);
            vertex_t end = g.neighborhood(u + 1);
            vertex_t deg = end - begin;

            sum_deg += deg;
            if (deg == 0) { ++num_isolated; }

            if (deg < min_deg) min_deg = deg;
            if (deg > max_deg) max_deg = deg;
        }

        double avg_deg = static_cast<double>(sum_deg) / static_cast<double>(g.n);

        // Try to infer whether edges are stored once or twice (undirected)
        std::uint64_t csr_sum = sum_deg;

        os << "  degree:\n";
        os << "    min       = " << min_deg << "\n";
        os << "    max       = " << max_deg << "\n";
        os << "    avg       = " << avg_deg << "\n";
        os << "    isolated  = " << num_isolated
                << " (" << 100.0 * static_cast<double>(num_isolated) / static_cast<double>(g.n) << " %)\n";
        os << "    CSR sum   = " << csr_sum << "\n";

        if (csr_sum == g.m) {
            os << "    note      = CSR degree sum == m (edges stored once)\n";
        } else if (csr_sum == 2ull * g.m) {
            os << "    note      = CSR degree sum == 2*m (undirected edges stored twice)\n";
        } else {
            os << "    note      = CSR degree sum != m and != 2*m (check consistency / multigraph?)\n";
        }

        // Density estimate (assuming simple undirected graph if possible)
        double density = 0.0;
        if (g.n > 1) {
            std::uint64_t undirected_m =
                    (csr_sum == 2ull * g.m) ? (csr_sum / 2ull) : static_cast<std::uint64_t>(g.m);
            double max_undirected = static_cast<double>(g.n) *
                                    static_cast<double>(g.n - 1) / 2.0;
            density = max_undirected > 0.0
                          ? static_cast<double>(undirected_m) / max_undirected
                          : 0.0;
        }
        os << "  density (approx, undirected) ~ " << density << "\n";

        // Vertex weight stats
        weight_t min_vw = numeric_limits<weight_t>::max();
        weight_t max_vw = 0;
        std::uint64_t sum_vw = 0;

        for (vertex_t u = 0; u < g.n; ++u) {
            weight_t w = g.weights(u);
            sum_vw += static_cast<std::uint64_t>(w);
            if (w < min_vw) min_vw = w;
            if (w > max_vw) max_vw = w;
        }

        os << "  vertex weights:\n";
        os << "    total     = " << sum_vw << "\n";
        os << "    min       = " << min_vw << "\n";
        os << "    max       = " << max_vw << "\n";

        if (static_cast<std::uint64_t>(g.g_weight) != sum_vw) {
            os << "    warning   = sum(vertex_weights) != g_weight\n";
        }

        // Edge weight stats
        weight_t min_ew = numeric_limits<weight_t>::max();
        weight_t max_ew = 0;
        std::uint64_t sum_ew = 0;

        for (vertex_t e = 0; e < g.m; ++e) {
            weight_t w = g.edges_w(e);
            sum_ew += static_cast<std::uint64_t>(w);
            if (w < min_ew) min_ew = w;
            if (w > max_ew) max_ew = w;
        }

        os << "  edge weights:\n";
        os << "    total     = " << sum_ew << "\n";
        os << "    min       = " << (g.m ? min_ew : 0) << "\n";
        os << "    max       = " << (g.m ? max_ew : 0) << "\n";

        os << "========================================\n";

        std::uint64_t cnt_deg_ge_10 = 0;
        std::uint64_t cnt_deg_ge_100 = 0;
        std::uint64_t cnt_deg_ge_1k = 0;
        std::uint64_t cnt_deg_ge_10k = 0;
        std::uint64_t cnt_deg_ge_100k = 0;
        std::uint64_t cnt_deg_ge_1M = 0;

        for (vertex_t u = 0; u < g.n; ++u) {
            vertex_t deg = g.neighborhood(u + 1) - g.neighborhood(u);

            if (deg >= 10) ++cnt_deg_ge_10;
            if (deg >= 100) ++cnt_deg_ge_100;
            if (deg >= 1000) ++cnt_deg_ge_1k;
            if (deg >= 10000) ++cnt_deg_ge_10k;
            if (deg >= 100000)++cnt_deg_ge_100k;
            if (deg >= 1000000)++cnt_deg_ge_1M;
        }

        os << "  heavy degree counts (deg >= X):\n";
        os << "    >= 10        : " << cnt_deg_ge_10 << "\n";
        os << "    >= 100       : " << cnt_deg_ge_100 << "\n";
        os << "    >= 1,000     : " << cnt_deg_ge_1k << "\n";
        os << "    >= 10,000    : " << cnt_deg_ge_10k << "\n";
        os << "    >= 100,000   : " << cnt_deg_ge_100k << "\n";
        os << "    >= 1,000,000 : " << cnt_deg_ge_1M << "\n";
    }

    inline void print_graph_stats(const Graph &device_g, std::ostream &os = std::cout) {
        HostGraph host_g = to_host_graph(device_g);
        print_graph_stats(host_g, os);
    }
}

#endif //GPU_HEIPA_GRAPH_H
