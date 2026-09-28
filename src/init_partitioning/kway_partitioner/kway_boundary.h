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

#ifndef GPU_HEIPA_KWAY_BOUNDARY_H
#define GPU_HEIPA_KWAY_BOUNDARY_H

#include <vector>

namespace GPU_HeiPa::ModifiedMetis {
    class Boundary {
    public:
        int curr_n = 0;
        int max_n = 0;
        std::vector<int> idx;
        std::vector<int> val;

        Boundary(int n = 0) : curr_n(0), max_n(n), idx(n, -1), val(n) {
        }

        void add(int v) {
            val[curr_n] = v;
            idx[v] = curr_n++;
        }

        void remove(int v) {
            int last = val[--curr_n];
            val[idx[v]] = last;
            idx[last] = idx[v];
            idx[v] = -1;
        }

        bool is_boundary(int v) const {
            return idx[v] != -1;
        }

        void reset() {
            curr_n = 0;
            std::fill(idx.begin(), idx.end(), -1);
        }
    };
}


#endif //GPU_HEIPA_KWAY_BOUNDARY_H
