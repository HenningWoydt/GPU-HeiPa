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

#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    struct HostGraph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;

        HostWeight weights;
        HostVertex neighborhood;
        HostVertex edges_v;
        HostWeight edges_w;
    };

    inline HostGraph from_file(const std::string &file_path) {
        ScopedTimer t_read_header{"io", "from_file", "read_header"};
        HostGraph g;

        if (!file_exists(file_path)) {
            std::cerr << "File " << file_path << " does not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Could not open file " << file_path << "!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line(64, ' ');
        bool has_v_weights = false;
        bool has_e_weights = false;

        // read in header
        while (std::getline(file, line)) {
            if (line[0] == '%') { continue; }

            // read in header
            std::vector<std::string> header = split_ws(line);
            g.n = (vertex_t) std::stoul(header[0]);
            g.m = (vertex_t) std::stoul(header[1]) * 2;

            // allocate space
            g.g_weight = 0;
            g.weights = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weights"), g.n);
            g.neighborhood = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), g.n + 1);
            g.neighborhood(0) = 0;
            g.edges_v = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), g.m);
            g.edges_w = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), g.m);

            // read in header
            std::string fmt = "000";
            if (header.size() == 3 && header[2].size() == 3) {
                fmt = header[2];
            }
            has_v_weights = fmt[1] == '1';
            has_e_weights = fmt[2] == '1';

            break;
        }
        t_read_header.stop();

        ScopedTimer t_read_edges{"io", "from_file", "read_edges"};
        // read in edges
        vertex_t u = 0;
        std::vector<vertex_t> ints(g.n);
        vertex_t curr_m = 0;

        while (std::getline(file, line)) {
            if (line[0] == '%') { continue; }
            // convert the lines into ints
            str_to_ints(line, ints);

            size_t i = 0;

            // check if vertex weights
            weight_t w = 1;
            if (has_v_weights) { w = (weight_t) ints[i++]; }
            g.weights(u) = w;
            g.g_weight += w;

            while (i < ints.size()) {
                vertex_t v = ints[i++] - 1;

                // check if edge weights
                w = 1;
                if (has_e_weights) { w = (weight_t) ints[i++]; }
                g.edges_v(curr_m) = v;
                g.edges_w(curr_m) = w;
                curr_m += 1;
            }
            g.neighborhood(u + 1) = curr_m;

            u += 1;
        }

        if (curr_m != g.m) {
            std::cerr << "Number of expected edges " << g.m << " not equal to number edges " << curr_m << " found!" << std::endl;
            exit(EXIT_FAILURE);
        }
        t_read_edges.stop();

        return g;
    }
}

#endif //GPU_HEIPA_HOST_GRAPH_H
