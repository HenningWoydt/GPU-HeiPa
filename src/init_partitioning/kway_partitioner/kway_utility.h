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

#ifndef GPU_HEIPA_KWAY_UTILITY_H
#define GPU_HEIPA_KWAY_UTILITY_H

#include <chrono>
#include <cstddef>
#include <vector>


namespace GPU_HeiPa::ModifiedMetis {
    /* Types of boundaries */
    constexpr int BNDTYPE_REFINE = 1; /* Used for k-way refinement-purposes */
    constexpr int BNDTYPE_BALANCE = 2; /* Used for k-way balancing purposes */

    /* Mode of optimization */
    constexpr int OMODE_REFINE = 1; /* Optimize the objective function */
    constexpr int OMODE_BALANCE = 2; /* Balance the subdomains */

    constexpr int UNMATCHED = -1;

    class BisectInfo {
    public:
        std::vector<int> int_w; /*!< Internal degree for each vertex */
        std::vector<int> ext_w; /*!< External degree for each vertex */

        BisectInfo() = default;

        BisectInfo(int n) : int_w(n), ext_w(n) {
        }

        void resize(int n) {
            int_w.resize(n);
            ext_w.resize(n);
        }
    };

    inline void permute(int n, std::vector<int> &p, int nshuffles) {
        if (n < 10) {
            for (int i = 0; i < n; i++) {
                std::swap(p[rand() % n], p[rand() % n]);
            }
        } else {
            for (int i = 0; i < nshuffles; i++) {
                int v = rand() % (n - 3);
                int u = rand() % (n - 3);

                std::swap(p[v + 0], p[u + 2]);
                std::swap(p[v + 1], p[u + 3]);
                std::swap(p[v + 2], p[u + 0]);
                std::swap(p[v + 3], p[u + 1]);
            }
        }
    }

    inline void counting_sort(int n, int max, const std::vector<int> &keys, const std::vector<int> &tperm, std::vector<int> &perm) {
        std::vector<int> counts(max + 2, 0);

        for (int i = 0; i < n; ++i) {
            ++counts[keys[i]];
        }

        for (int i = 1; i < max + 1; ++i) counts[i] += counts[i - 1];
        for (int i = max + 1; i > 0; --i) counts[i] = counts[i - 1];
        counts[0] = 0;

        for (int ii = 0; ii < n; ++ii) {
            int i = tperm[ii];
            perm[counts[keys[i]]++] = i;
        }
    }

    inline auto get_time_point() {
        return std::chrono::high_resolution_clock::now();
    }

    inline double get_milli_seconds(std::chrono::high_resolution_clock::time_point sp,
                                    std::chrono::high_resolution_clock::time_point ep) {
        return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e6;
    }
}


#endif //GPU_HEIPA_KWAY_UTILITY_H
