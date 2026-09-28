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

#ifndef GPU_HEIPA_KWAY_GRAPH_H
#define GPU_HEIPA_KWAY_GRAPH_H

#include <vector>

namespace GPU_HeiPa::ModifiedMetis {
    class Graph {
    public:
        int n = -1;
        int m = -1;
        int g_weight = -1;

        std::vector<int> v_weights;
        std::vector<int> rows;
        std::vector<int> edges_v;
        std::vector<int> edges_w;

        int mincut = -1;

        Graph() = default;

        Graph(int t_n, int t_m, int t_w = 0) : n(t_n), m(t_m), g_weight(t_w), v_weights(t_n), rows(t_n + 1), edges_v(t_m), edges_w(t_m), mincut(0) {
        }

        void resize(int t_n, int t_m, int t_w = -1) {
            n = t_n;
            m = t_m;
            g_weight = t_w;
            v_weights.resize(t_n);
            rows.resize(t_n + 1);
            edges_v.resize(t_m);
            edges_w.resize(t_m);
        }
    };
}


#endif //GPU_HEIPA_KWAY_GRAPH_H
