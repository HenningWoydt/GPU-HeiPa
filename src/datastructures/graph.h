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
#include "mapping.h"
#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    struct Graph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;

        DeviceWeight weights;
        DeviceVertex neighborhood;
        DeviceVertex edges_u;
        DeviceVertex edges_v;
        DeviceWeight edges_w;
    };

    inline Graph from_HostGraph(const HostGraph &host_g) {
        ScopedTimer t_init{"io", "from_HostGraph", "initialize"};
        Graph g;

        g.n = host_g.n;
        g.m = host_g.m;
        g.g_weight = host_g.g_weight;

        g.weights = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), host_g.n);
        g.neighborhood = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), host_g.n + 1);
        g.edges_u = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_u"), host_g.m);
        g.edges_v = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), host_g.m);
        g.edges_w = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), host_g.m);
        t_init.stop();

        ScopedTimer t_copy{"io", "from_HostGraph", "copy"};
        Kokkos::deep_copy(g.weights, host_g.weights);
        Kokkos::deep_copy(g.neighborhood, host_g.neighborhood);
        Kokkos::deep_copy(g.edges_v, host_g.edges_v);
        Kokkos::deep_copy(g.edges_w, host_g.edges_w);
        Kokkos::fence();
        t_copy.stop();

        ScopedTimer t_fill_edges_u{"io", "from_HostGraph", "fill_edges_u"};
        Kokkos::parallel_for("fill_edges_u", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            u32 begin = g.neighborhood(u);
            u32 end = g.neighborhood(u + 1);
            for (u32 i = begin; i < end; ++i) {
                g.edges_u(i) = u;
            }
        });
        Kokkos::fence();
        t_fill_edges_u.stop();

        return g;
    }

    inline Graph from_Graph_Mapping(const Graph &old_g,
                                    const Mapping &mapping) {
        // initialize vars and allocate memory
        ScopedTimer t_initialize{"contraction", "from_Graph_Mapping", "initialize"};
        Graph coarse_g;

        coarse_g.n = mapping.coarse_n;
        coarse_g.g_weight = old_g.g_weight;

        coarse_g.weights = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), mapping.coarse_n);
        coarse_g.neighborhood = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), mapping.coarse_n + 1);

        DeviceVertex degrees = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "degrees"), mapping.coarse_n);
        DeviceVertex max_degrees = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "max_degrees"), mapping.coarse_n);

        DeviceU32 hash_offsets = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hash_offsets"), coarse_g.n + 1);
        DeviceVertex hash_keys("hash_keys", old_g.m);
        DeviceWeight hash_vals("hash_vals", old_g.m);
        t_initialize.stop();

        // set memory to 0
        ScopedTimer t_initialize_set_0{"contraction", "from_Graph_Mapping", "initialize_set_0"};
        Kokkos::deep_copy(coarse_g.weights, 0);
        Kokkos::deep_copy(degrees, 0);
        Kokkos::deep_copy(max_degrees, 0);
        Kokkos::deep_copy(hash_offsets, 0);
        Kokkos::deep_copy(hash_keys, coarse_g.n);
        Kokkos::deep_copy(hash_vals, 0);
        t_initialize_set_0.stop();

        // determine weight of new vertices and maximum degree
        ScopedTimer t_max_degrees{"contraction", "from_Graph_Mapping", "max_degrees"};
        Kokkos::parallel_for("max_degrees", old_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t deg = old_g.neighborhood(u + 1) - old_g.neighborhood(u);
            weight_t u_w = old_g.weights(u);
            vertex_t v = mapping.mapping(u);

            Kokkos::atomic_add(&max_degrees(v), deg);
            Kokkos::atomic_add(&coarse_g.weights(v), u_w);
        });
        Kokkos::fence();
        t_max_degrees.stop();

        // prefix sum over all degrees
        ScopedTimer t_max_degrees_prefix_sum{"contraction", "from_Graph_Mapping", "max_degrees_prefix_sum"};
        Kokkos::parallel_scan("prefix_sum_offsets", coarse_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
            u32 cnt = i < coarse_g.n ? max_degrees(i) : 0;
            if (final) hash_offsets(i) = running;
            running += cnt;
        });
        Kokkos::fence();
        t_max_degrees_prefix_sum.stop();

        ScopedTimer t_hash_edges{"contraction", "from_Graph_Mapping", "hash_edges"};
        Kokkos::parallel_for("hash_edges", old_g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = old_g.edges_u(i);
            vertex_t v = old_g.edges_v(i);
            weight_t w = old_g.edges_w(i);

            vertex_t u_new = mapping.mapping(u);
            vertex_t v_new = mapping.mapping(v);

            if (u_new == v_new) { return; } // this edge vanishes

            u32 beg = hash_offsets(u_new);
            u32 end = hash_offsets(u_new + 1);
            u32 len = end - beg;
            if (len == 0) { return; }

            // 32-bit multiplicative mix on v_new; keep in range [0,size)
            u32 x = v_new;
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            x *= 0x846ca68bU;
            x ^= x >> 16;

            uint32_t idx = beg + x % len;
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
        Kokkos::fence();
        t_hash_edges.stop();

        ScopedTimer t_build_offsets{"contraction", "from_Graph_Mapping", "build_offsets"};
        Kokkos::parallel_scan("build_offsets", coarse_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
            const u32 cnt = i < coarse_g.n ? degrees(i) : 0;
            if (final) coarse_g.neighborhood(i) = running;
            running += cnt;
        });
        Kokkos::fence();
        Kokkos::deep_copy(coarse_g.m, Kokkos::subview(coarse_g.neighborhood, coarse_g.n)); // copy final number of edges m
        Kokkos::fence();

        t_build_offsets.stop();
        ScopedTimer t_allocate_edges{"contraction", "from_Graph_Mapping", "allocate_edges"};

        coarse_g.edges_u = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_u"), coarse_g.m);
        coarse_g.edges_v = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), coarse_g.m);
        coarse_g.edges_w = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), coarse_g.m);

        t_allocate_edges.stop();
        ScopedTimer t_fill_coarse_graph{"contraction", "from_Graph_Mapping", "fill_coarse_graph"};

        Kokkos::parallel_for("materialize_edges", coarse_g.n, KOKKOS_LAMBDA(const vertex_t u_new) {
            u32 out = coarse_g.neighborhood(u_new);
            u32 end = coarse_g.neighborhood(u_new + 1);

            u32 off = hash_offsets(u_new);
            u32 size = hash_offsets(u_new + 1) - off;

            for (u32 idx = 0; idx < size && out < end; ++idx) {
                u32 pos = off + idx;
                vertex_t v = hash_keys(pos);
                if (v != coarse_g.n) {
                    coarse_g.edges_v(out) = v;
                    coarse_g.edges_w(out) = hash_vals(pos);
                    ++out;
                }
            }
        });
        Kokkos::fence();

        t_fill_coarse_graph.stop();
        ScopedTimer t_fill_coarse_u{"contraction", "from_Graph_Mapping", "fill_coarse_u"};

        Kokkos::parallel_for("fill_edges_u", coarse_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            u32 begin = coarse_g.neighborhood(u);
            u32 end = coarse_g.neighborhood(u + 1);
            for (u32 i = begin; i < end; ++i) {
                coarse_g.edges_u(i) = u;
            }
        });
        Kokkos::fence();

        return coarse_g;
    }

    inline HostGraph to_host_graph(const Graph &device_g) {
        HostGraph host_g;

        host_g.n = device_g.n;
        host_g.m = device_g.m;
        host_g.g_weight = device_g.g_weight;

        host_g.weights = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weights_host"), device_g.n);
        host_g.neighborhood = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood_host"), device_g.n + 1);
        host_g.edges_v = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v_host"), device_g.m);
        host_g.edges_w = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w_host"), device_g.m);

        Kokkos::deep_copy(host_g.weights, device_g.weights);
        Kokkos::deep_copy(host_g.neighborhood, device_g.neighborhood);
        Kokkos::deep_copy(host_g.edges_v, device_g.edges_v);
        Kokkos::deep_copy(host_g.edges_w, device_g.edges_w);
        Kokkos::fence();

        return host_g;
    }
}

#endif //GPU_HEIPA_GRAPH_H
