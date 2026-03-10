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

#ifndef GPU_HEIPA_KWAY_PRIO_QUEUE_H
#define GPU_HEIPA_KWAY_PRIO_QUEUE_H

#include <vector>


namespace GPU_HeiPa::ModifiedMetis {
    template<typename T>
    class PriorityQueue {
        struct Node {
            T key;
            int val;
        };

        size_t n;
        std::vector<Node> heap;
        std::vector<ssize_t> locator;

    private:
        void siftUp(ssize_t &i, const T &key) {
            while (i > 0) {
                ssize_t parent = (i - 1) >> 1;
                if (!(key > heap[parent].key))
                    break;

                heap[i] = heap[parent];
                locator[heap[i].val] = i;
                i = parent;
            }
        }

        void siftDown(ssize_t &i, const T &key) {
            while (true) {
                ssize_t left = (i << 1) + 1;
                if (left >= static_cast<ssize_t>(n))
                    break;

                ssize_t best = left;

                if (heap[left].key > key) {
                    if (left + 1 < static_cast<ssize_t>(n) &&
                        heap[left + 1].key > heap[left].key) {
                        best = left + 1;
                    }
                } else if (left + 1 < static_cast<ssize_t>(n) &&
                           heap[left + 1].key > key) {
                    best = left + 1;
                } else {
                    break;
                }

                heap[i] = heap[best];
                locator[heap[i].val] = i;
                i = best;
            }
        }

        void placeAt(ssize_t i, int node, const T &key) {
            heap[i].key = key;
            heap[i].val = node;
            locator[node] = i;
        }

    public:
        explicit PriorityQueue(size_t max_n)
            : n(0), heap(max_n), locator(max_n, -1) {
        }

        void reset() {
            for (ssize_t i = static_cast<ssize_t>(n) - 1; i >= 0; --i)
                locator[heap[i].val] = -1;
            n = 0;
        }

        bool empty() const {
            return n == 0;
        }

        size_t size() const {
            return n;
        }

        bool contains(int node) const {
            return node >= 0 && static_cast<size_t>(node) < locator.size() && locator[node] != -1;
        }

        int top() const {
            return heap[0].val;
        }

        T topKey() const {
            return heap[0].key;
        }

        void insert(int node, T key) {
            ssize_t i = static_cast<ssize_t>(n++);
            siftUp(i, key);
            placeAt(i, node, key);
        }

        void erase(int node) {
            ssize_t i = locator[node];
            locator[node] = -1;

            if (--n == 0 || heap[n].val == node)
                return;

            int lastNode = heap[n].val;
            T newKey = heap[n].key;
            T oldKey = heap[i].key;

            if (newKey > oldKey) {
                siftUp(i, newKey);
            } else {
                siftDown(i, newKey);
            }

            placeAt(i, lastNode, newKey);
        }

        void update(int node, T newkey) {
            ssize_t i = locator[node];
            T oldkey = heap[i].key;

            if (newkey == oldkey)
                return;

            if (newkey > oldkey) {
                siftUp(i, newkey);
            } else {
                siftDown(i, newkey);
            }

            placeAt(i, node, newkey);
        }

        int top_pop() {
            int top = heap[0].val;
            locator[top] = -1;

            if (--n == 0)
                return top;

            int lastNode = heap[n].val;
            T lastKey = heap[n].key;
            ssize_t i = 0;

            siftDown(i, lastKey);
            placeAt(i, lastNode, lastKey);

            return top;
        }
    };
}


#endif //GPU_HEIPA_KWAY_PRIO_QUEUE_H
