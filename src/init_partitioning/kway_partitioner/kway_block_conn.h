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

#ifndef GPU_HEIPA_KWAY_BLOCK_CONN_H
#define GPU_HEIPA_KWAY_BLOCK_CONN_H

#include <vector>
#include <cassert>


namespace GPU_HeiPa::ModifiedMetis {
    class BlockConn {
        int n = -1;
        int k = -1;

        std::vector<int> n_conns;
        std::vector<int> n_conns_temp;
        std::vector<int> int_w;
        std::vector<int> int_w_temp;
        std::vector<int> ext_w;
        std::vector<int> ext_w_temp;

        std::vector<int> weights;
        std::vector<int> locator;
        std::vector<int> locator_list;

    public:
        BlockConn() = default;

        BlockConn(int t_n, int t_k)
            : n(t_n),
              k(t_k),
              n_conns(n, 0),
              n_conns_temp(n, 0),
              int_w(n, 0),
              int_w_temp(n, 0),
              ext_w(n, 0),
              ext_w_temp(n, 0),
              weights(n * k, 0),
              locator(n * k, -1),
              locator_list(n * k, -1) {
        }

        void update(int v, int id, int delta) {
            assert(v >= 0 && v < n);
            assert(id >= 0 && id < k);

            if (delta == 0) {
                return;
            }

            const int off = v * k + id;
            int old_w = weights[off];
            int new_w = old_w + delta;

            assert(new_w >= 0);

            weights[off] = new_w;

            // --- new connection appears ---
            if (old_w == 0 && new_w > 0) {
                int pos = n_conns[v]++;
                locator[off] = pos;
                locator_list[v * k + pos] = id;
                return;
            }

            // --- connection disappears ---
            if (old_w > 0 && new_w == 0) {
                int pos = locator[off];
                int last_pos = n_conns[v] - 1;

                if (pos != last_pos) {
                    int last_id = locator_list[v * k + last_pos];
                    locator_list[v * k + pos] = last_id;
                    locator[v * k + last_id] = pos;
                }

                locator_list[v * k + last_pos] = -1;
                locator[off] = -1;
                --n_conns[v];
            }
        }

        int get(int v, int id) const {
            assert(v >= 0 && v < n);
            assert(id >= 0 && id < k);
            return weights[v * k + id];
        }

        void reset() {
            std::fill(n_conns.begin(), n_conns.end(), 0);
            std::fill(n_conns_temp.begin(), n_conns_temp.end(), 0);
            std::fill(weights.begin(), weights.end(), 0);
            std::fill(locator.begin(), locator.end(), -1);
            std::fill(locator_list.begin(), locator_list.end(), -1);
            std::fill(int_w.begin(), int_w.end(), 0);
            std::fill(int_w_temp.begin(), int_w_temp.end(), 0);
            std::fill(ext_w.begin(), ext_w.end(), 0);
            std::fill(ext_w_temp.begin(), ext_w_temp.end(), 0);
        }

        void swap_infos() {
            n_conns.swap(n_conns_temp);
            int_w.swap(int_w_temp);
            ext_w.swap(ext_w_temp);
        }

        int get_temp_ext_w(int v) const { return ext_w_temp[v]; }

        int get_int_w(int v) const { return int_w[v]; }
        int get_ext_w(int v) const { return ext_w[v]; }
        int get_n_conns(int v) const { return n_conns[v]; }
        void add_int_w(int v, int x) { int_w[v] += x; }
        void add_ext_w(int v, int x) { ext_w[v] += x; }
        void set_int_w(int v, int x) { int_w[v] = x; }
        void set_ext_w(int v, int x) { ext_w[v] = x; }
        void set_n_conns(int v, int x) { n_conns[v] = x; }
        int get_conn_id(int v, int i) const { return locator_list[v * k + i]; }
        int get_conn_w(int v, int i) const { return get(v, get_conn_id(v, i)); }
    };
}


#endif // GPU_HEIPA_KWAY_BLOCK_CONN_H
