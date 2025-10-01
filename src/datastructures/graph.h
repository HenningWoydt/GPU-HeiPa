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
#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    struct Graph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;

        DeviceWeight weights;
        DeviceU32 neighborhood;
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
        g.neighborhood = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), host_g.n + 1);
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
}

#endif //GPU_HEIPA_GRAPH_H
