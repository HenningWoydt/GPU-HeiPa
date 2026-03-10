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

#ifndef GPU_HEIPA_KWAY_INDEXED_PRIO_QUEUE_H
#define GPU_HEIPA_KWAY_INDEXED_PRIO_QUEUE_H

#include <vector>


namespace GPU_HeiPa::ModifiedMetis {
    template<typename T>
    class IndexedPriorityQueue {
    public:
        struct Node {
            T key;
            int val;
        };

    private:
        size_t nnodes_ = 0;
        int ntouched_ = 0;

        std::vector<Node> heap_;
        std::vector<ssize_t> locator_;

        void siftUp(ssize_t &i, const T &key) {
            while (i > 0) {
                ssize_t parent = (i - 1) >> 1;
                if (!(key > heap_[parent].key))
                    break;

                heap_[i] = heap_[parent];
                locator_[heap_[i].val] = i;
                i = parent;
            }
        }

        void siftDown(ssize_t &i, const T &key) {
            while (true) {
                ssize_t left = (i << 1) + 1;
                if (left >= static_cast<ssize_t>(nnodes_))
                    break;

                ssize_t best = left;

                if (heap_[left].key > key) {
                    if (left + 1 < static_cast<ssize_t>(nnodes_) &&
                        heap_[left + 1].key > heap_[left].key) {
                        best = left + 1;
                    }
                } else if (left + 1 < static_cast<ssize_t>(nnodes_) &&
                           heap_[left + 1].key > key) {
                    best = left + 1;
                } else {
                    break;
                }

                heap_[i] = heap_[best];
                locator_[heap_[i].val] = i;
                i = best;
            }
        }

        void placeAt(ssize_t i, int node, const T &key) {
            heap_[i] = {key, node};
            locator_[node] = i;
        }

    public:
        explicit IndexedPriorityQueue(size_t maxnodes)
            : heap_(maxnodes),
              locator_(maxnodes, -1) {
        }

        bool empty() const {
            return nnodes_ == 0;
        }

        bool contains(int node) const {
            return node >= 0 && static_cast<size_t>(node) < locator_.size() && locator_[node] != -1;
        }

        void insert(int node, T key) {
            ssize_t i = static_cast<ssize_t>(nnodes_++);
            siftUp(i, key);
            placeAt(i, node, key);
        }

        void erase(int node) {
            ssize_t i = locator_[node];
            locator_[node] = -1;

            if (--nnodes_ != 0 && heap_[nnodes_].val != node) {
                int lastNode = heap_[nnodes_].val;
                T lastKey = heap_[nnodes_].key;
                T oldKey = heap_[i].key;

                if (lastKey > oldKey) {
                    siftUp(i, lastKey);
                } else {
                    siftDown(i, lastKey);
                }

                placeAt(i, lastNode, lastKey);
            }
        }

        void update(int node, T newkey) {
            ssize_t i = locator_[node];
            T oldkey = heap_[i].key;

            if (newkey == oldkey)
                return;

            if (newkey > oldkey) {
                siftUp(i, newkey);
            } else {
                siftDown(i, newkey);
            }

            placeAt(i, node, newkey);
        }

        void updateOrInsert(int node, T key) {
            ssize_t pos = locator_[node];
            if (pos != -1) {
                update(node, key);
            } else {
                insert(node, key);
            }
        }

        T top_key() {
            return heap_[0].key;
        }

        int top_pop() {
            int top = heap_[0].val;
            locator_[top] = -1;

            if (--nnodes_ == 0)
                return top;

            int lastNode = heap_[nnodes_].val;
            T lastKey = heap_[nnodes_].key;
            ssize_t i = 0;

            siftDown(i, lastKey);
            placeAt(i, lastNode, lastKey);

            return top;
        }
    };
}


#endif //GPU_HEIPA_KWAY_INDEXED_PRIO_QUEUE_H
