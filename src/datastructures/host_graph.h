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

#ifndef GPU_HEIPA_HOST_GRAPH_H
#define GPU_HEIPA_HOST_GRAPH_H

#include <iostream>
#include <cstdlib>

#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    struct HostGraph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;
        bool uniform_edge_weights = false;
        bool uniform_vertex_weights = false;

        HostU8 memory;
        u64 n_bytes = 0;

        UnmanagedHostWeight weights;
        UnmanagedHostVertex neighborhood;
        UnmanagedHostVertex edges_v;
        UnmanagedHostWeight edges_w;
    };

    inline void allocate_memory(HostGraph &g, vertex_t n, vertex_t m, weight_t g_weight) {
        g.n = n;
        g.m = m;
        g.g_weight = g_weight;

        size_t total = 0;

        size_t off_weights = total;
        size_t bytes_weights = g.uniform_vertex_weights ? 0 : round_up_64(sizeof(weight_t) * (size_t) n);
        total += bytes_weights;

        size_t off_neighborhood = total;
        size_t bytes_neighborhood = round_up_64(sizeof(vertex_t) * (size_t) (n + 1));
        total += bytes_neighborhood;

        size_t off_edges_v = total;
        size_t bytes_edges_v = round_up_64(sizeof(vertex_t) * (size_t) m);
        total += bytes_edges_v;

        size_t off_edges_w = total;
        size_t bytes_edges_w = g.uniform_edge_weights ? 0 : round_up_64(sizeof(weight_t) * (size_t) m);
        total += bytes_edges_w;

        g.n_bytes = total;

        g.memory = HostU8(Kokkos::view_alloc(Kokkos::WithoutInitializing, "HostGraph::memory"), g.n_bytes);
        uint8_t *base = g.memory.data();

        if (!g.uniform_vertex_weights) {
            g.weights = UnmanagedHostWeight((weight_t *) (base + off_weights), n);
        }
        g.neighborhood = UnmanagedHostVertex((vertex_t *) (base + off_neighborhood), n + 1);
        g.neighborhood(0) = 0;
        g.edges_v = UnmanagedHostVertex((vertex_t *) (base + off_edges_v), m);
        if (!g.uniform_edge_weights) {
            g.edges_w = UnmanagedHostWeight((weight_t *) (base + off_edges_w), m);
        }
    }

    template<bool has_v_weights, bool has_e_weights>
    inline void read_edges(HostGraph &g, char *p, const char *end) {
        vertex_t *edges_v_ptr = g.edges_v.data();
        weight_t *edges_w_ptr = has_e_weights ? g.edges_w.data() : nullptr;
        weight_t *weights_ptr = has_v_weights ? g.weights.data() : nullptr;
        vertex_t *neighborhood_ptr = g.neighborhood.data();

        vertex_t u = 0;
        size_t curr_m = 0;

        while (p < end) {
            while (*p == '%') {
                while (*p != '\n') { ++p; }
                ++p;
            }
            while (*p == ' ') { ++p; }

            weight_t vw = 1;
            if constexpr (has_v_weights) {
                vw = 0;
                while (*p != ' ' && *p != '\n') {
                    vw = vw * 10 + (weight_t) (*p - '0');
                    ++p;
                }
                while (*p == ' ') { ++p; }
                weights_ptr[u] = vw;
            }
            g.g_weight += vw;

            while (*p != '\n' && p < end) {
                vertex_t v = 0;
                while (*p != ' ' && *p != '\n') {
                    v = v * 10 + (vertex_t) (*p - '0');
                    ++p;
                }
                while (*p == ' ') { ++p; }

                if constexpr (has_e_weights) {
                    weight_t w = 0;
                    while (*p != ' ' && *p != '\n') {
                        w = w * 10 + (weight_t) (*p - '0');
                        ++p;
                    }
                    while (*p == ' ') { ++p; }
                    edges_w_ptr[curr_m] = w;
                }

                edges_v_ptr[curr_m] = v - 1;
                ++curr_m;
            }
            neighborhood_ptr[u + 1] = (vertex_t) curr_m;
            ++u;
            ++p;
        }

        if (curr_m != g.m) {
            std::cerr << "Number of expected edges " << g.m << " not equal to number edges " << curr_m << " found!\n";
            exit(EXIT_FAILURE);
        }
    }

    inline HostGraph from_file(const std::string &file_path) {
        ScopedTimer _t_allocate("io", "CSRGraph", "allocate");
        if (!file_exists(file_path)) {
            std::cerr << "File " << file_path << " does not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // mmap the whole file
        MMap mm = mmap_file_ro(file_path);
        char *p = mm.data;
        const char *end = mm.data + mm.size;

        _t_allocate.stop();
        ScopedTimer _t_read_header("io", "CSRGraph", "read_header");

        // skip comment lines
        while (*p == '%') {
            while (*p != '\n') { ++p; }
            ++p;
        }

        // skip whitespace
        while (*p == ' ') { ++p; }

        // read number of vertices - optimized parsing
        HostGraph g;
        g.n = 0;
        while (*p != ' ' && *p != '\n') {
            g.n = g.n * 10 + (vertex_t) (*p - '0');
            ++p;
        }

        // skip whitespace
        while (*p == ' ') { ++p; }

        // read number of edges - optimized parsing  
        g.m = 0;
        while (*p != ' ' && *p != '\n') {
            g.m = g.m * 10 + (vertex_t) (*p - '0');
            ++p;
        }
        g.m *= 2;

        // search end of line or fmt
        std::string fmt = "000";
        bool has_v_weights = false;
        bool has_e_weights = false;
        while (*p == ' ') { ++p; }
        if (*p != '\n') {
            // found fmt
            fmt[0] = *p;
            ++p;
            if (*p != '\n') {
                // found fmt
                fmt[1] = *p;
                ++p;
                if (*p != '\n') {
                    // found fmt
                    fmt[2] = *p;
                    ++p;
                }
            }
            // skip whitespaces
            while (*p == ' ') { ++p; }
        }
        has_v_weights = fmt[1] == '1';
        has_e_weights = fmt[2] == '1';
        g.uniform_edge_weights = !has_e_weights;
        g.uniform_vertex_weights = !has_v_weights;

        g.g_weight = 0;
        allocate_memory(g, g.n, g.m, 0);

        _t_read_header.stop();
        ScopedTimer _t_read_edges("io", "CSRGraph", "read_edges");

        ++p;

        if (has_v_weights && has_e_weights) {
            read_edges<true, true>(g, p, end);
        } else if (has_v_weights) {
            read_edges<true, false>(g, p, end);
        } else if (has_e_weights) {
            read_edges<false, true>(g, p, end);
        } else {
            read_edges<false, false>(g, p, end);
        }

        _t_read_edges.stop();
        // done with the file
        munmap_file(mm);

        return g;
    }

    inline HostGraph from_edge_list_file(const std::string &edge_list_file_path,
                                         const std::string &vertex_weight_file_path) {
        std::ifstream ef(edge_list_file_path);
        if (!ef) {
            std::cerr << "Cannot open edge list file " << edge_list_file_path << "\n";
            std::exit(EXIT_FAILURE);
        }

        // --- read header ---
        vertex_t n_rows, n_cols;
        vertex_t m;
        ef >> n_rows >> n_cols >> m;

        if (n_rows != n_cols) {
            std::cerr << "Non-square graph (" << n_rows << " x " << n_cols << ")\n";
            std::exit(EXIT_FAILURE);
        }

        HostGraph g;
        allocate_memory(g, n_rows, m, 0);

        // --- read vertex weights ---
        {
            std::ifstream vf(vertex_weight_file_path);
            if (!vf) {
                std::cerr << "Cannot open vertex weight file "
                        << vertex_weight_file_path << "\n";
                std::exit(EXIT_FAILURE);
            }

            g.g_weight = 0;
            for (vertex_t u = 0; u < g.n; ++u) {
                vf >> g.weights(u);
                g.g_weight += g.weights(u);
            }
        }

        // --- temporary degree counter ---
        std::vector<vertex_t> degree(g.n, 0);

        // store edges temporarily
        std::vector<vertex_t> src(m), dst(m);
        std::vector<weight_t> wgt(m);

        for (vertex_t i = 0; i < m; ++i) {
            ef >> src[i] >> dst[i] >> wgt[i];
            ASSERT(src[i] < g.n);
            ASSERT(dst[i] < g.n);
            ++degree[src[i]];
        }

        // --- build CSR offsets ---
        g.neighborhood(0) = 0;
        for (vertex_t u = 0; u < g.n; ++u) {
            g.neighborhood(u + 1) = g.neighborhood(u) + degree[u];
        }

        // --- fill CSR ---
        std::vector<vertex_t> cursor(g.n);
        for (vertex_t u = 0; u < g.n; ++u)
            cursor[u] = g.neighborhood(u);

        for (vertex_t i = 0; i < m; ++i) {
            vertex_t u = src[i];
            vertex_t pos = cursor[u]++;
            g.edges_v(pos) = dst[i];
            g.edges_w(pos) = wgt[i];
        }

        return g;
    }
}

#endif //GPU_HEIPA_HOST_GRAPH_H
