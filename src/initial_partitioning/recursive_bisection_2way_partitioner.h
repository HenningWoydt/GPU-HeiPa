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

#ifndef GPU_HEIPA_RECURSIVE_BISECTION_2WAY_PARTITIONER_H
#define GPU_HEIPA_RECURSIVE_BISECTION_2WAY_PARTITIONER_H

#include <vector>
#include <queue>

#include "../utility/definitions.h"
#include "../utility/macros.h"

namespace GPU_HeiPa {
    struct ComponentInfo {
        vertex_t id;
        vertex_t n_vertices;
        weight_t weight;
    };

    struct ScratchMemory {
        std::vector<vertex_t> vertices;
        std::vector<vertex_t> dist;
        std::vector<vertex_t> curr_stack;
        std::vector<vertex_t> next_stack;

        std::vector<vertex_t> curr_stack1;
        std::vector<vertex_t> next_stack1;
        std::vector<vertex_t> curr_stack2;
        std::vector<vertex_t> next_stack2;

        std::vector<vertex_t> missing_vertices;
        std::vector<vertex_t> next_missing_vertices;

        std::vector<vertex_t> component;
        std::vector<ComponentInfo> comp_weights;

        std::vector<vertex_t> part;
        std::vector<partition_t> global_part;

        std::vector<partition_t> partition;
    };

    inline vertex_t get_far_s(vertex_t s,
                              const HostGraph &g,
                              ScratchMemory &scratch_memory) {
        vertex_t INF = std::numeric_limits<vertex_t>::max();

        // do bfs to find vertex the furthest away
        std::vector<vertex_t> &dist = scratch_memory.dist;
        dist.resize(g.n);
        std::fill(dist.begin(), dist.end(), INF);

        std::vector<vertex_t> &curr_stack = scratch_memory.curr_stack;
        curr_stack.clear();
        curr_stack.push_back(s);
        std::vector<vertex_t> &next_stack = scratch_memory.next_stack;
        next_stack.clear();
        vertex_t distance = 0;
        vertex_t far_s = s;

        while (!curr_stack.empty()) {
            for (vertex_t u: curr_stack) {
                dist[u] = distance;
                far_s = u;
                for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    vertex_t v = g.edges_v(i);

                    if (dist[v] == INF) {
                        dist[v] = INF - 1;
                        next_stack.push_back(v);
                    }
                }
            }
            distance += 1;
            curr_stack.swap(next_stack);
            next_stack.clear();
        }

        return far_s;
    }

    inline void assign_missing_vertices(const HostGraph &g,
                                        const weight_t left_lmax,
                                        const weight_t right_lmax,
                                        weight_t &left_weight,
                                        weight_t &right_weight,
                                        const vertex_t left_min_n_vertices,
                                        const vertex_t right_min_n_vertices,
                                        vertex_t &left_n_vertices,
                                        vertex_t &right_n_vertices,
                                        std::vector<partition_t> &partition,
                                        std::vector<vertex_t> &vertices,
                                        ScratchMemory &scratch_memory) {
        std::vector<vertex_t> &missing_vertices = scratch_memory.missing_vertices;
        std::vector<vertex_t> &next_missing_vertices = scratch_memory.next_missing_vertices;
        missing_vertices.clear();
        next_missing_vertices.clear();

        for (vertex_t u: vertices) { if (partition[u] != 0 && partition[u] != 1) { missing_vertices.push_back(u); } }

        while (!missing_vertices.empty()) {
            for (vertex_t u: missing_vertices) {
                weight_t u_weight = g.weights(u);
                weight_t w1 = 0;
                weight_t w2 = 0;

                for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    vertex_t v = g.edges_v(i);
                    weight_t w = g.edges_w(i);

                    if (partition[v] == 0) { w1 += w; }
                    if (partition[v] == 1) { w2 += w; }
                }
                if (w1 == 0 && w2 == 0) {
                    next_missing_vertices.push_back(u);
                    continue;
                }

                if (w1 > w2) {
                    partition[u] = 0;
                    left_n_vertices += 1;
                    left_weight += u_weight;
                } else {
                    partition[u] = 1;
                    right_n_vertices += 1;
                    right_weight += u_weight;
                }
            }

            if (missing_vertices.size() == next_missing_vertices.size()) {
                vertex_t v = next_missing_vertices.back();
                next_missing_vertices.pop_back();
                weight_t v_weight = g.weights(v);

                if (left_lmax - left_weight > right_lmax - right_weight) {
                    partition[v] = 0;
                    left_n_vertices += 1;
                    left_weight += v_weight;
                } else {
                    partition[v] = 1;
                    right_n_vertices += 1;
                    right_weight += v_weight;
                }
            }

            missing_vertices.swap(next_missing_vertices);
            next_missing_vertices.clear();
        }

        for (vertex_t v: vertices) {
            ASSERT(partition[v] == 0 || partition[v] == 1);
        }
    }

    inline void edge_cut_not_assigned(const HostGraph &g,
                                      vertex_t u,
                                      std::vector<partition_t> &partition,
                                      weight_t &to0,
                                      weight_t &to1) {
        to0 = 0;
        to1 = 0;
        for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);

            if (partition[v] == 0) to0 += w;
            else if (partition[v] == 1) to1 += w;
        }
    }

    struct SmallMove {
        vertex_t u = 0;
        weight_t cut = 0;
        u64 state_id = 0;

        bool operator<(const SmallMove &m) const { return cut > m.cut; }
    };

    inline void partition_2way_one_component(const HostGraph &g,
                                             const weight_t left_lmax,
                                             const weight_t right_lmax,
                                             weight_t &left_weight,
                                             weight_t &right_weight,
                                             const vertex_t left_min_n_vertices,
                                             const vertex_t right_min_n_vertices,
                                             vertex_t &left_n_vertices,
                                             vertex_t &right_n_vertices,
                                             std::vector<partition_t> &partition,
                                             vertex_t component_id,
                                             std::vector<vertex_t> &component,
                                             ScratchMemory &scratch_memory) {
        std::vector<vertex_t> &vertices = scratch_memory.vertices;
        vertices.clear();
        vertices.reserve(g.n);

        for (vertex_t u = 0; u < g.n; ++u) { if (component[u] == component_id) { vertices.push_back(u); } }
        if (vertices.empty()) { return; }

        // only one vertex, put in larger block
        if (vertices.size() == 1) {
            vertex_t u = vertices[0];
            if (left_lmax - left_weight > right_lmax - right_weight) {
                partition[u] = 0;
                left_n_vertices += 1;
                left_weight += g.weights(u);
            } else {
                partition[u] = 1;
                right_n_vertices += 1;
                right_weight += g.weights(u);
            }
            return;
        }

        // only two vertices, put larger in large block and smaller in small block
        if (vertices.size() == 2) {
            vertex_t u = vertices[0];
            vertex_t v = vertices[1];
            if (g.weights(u) < g.weights(v)) { std::swap(u, v); }

            if (left_lmax - left_weight > right_lmax - right_weight) {
                partition[u] = 0;
                left_n_vertices += 1;
                left_weight += g.weights(u);
                partition[v] = 1;
                right_n_vertices += 1;
                right_weight += g.weights(v);
            } else {
                partition[u] = 1;
                right_n_vertices += 1;
                right_weight += g.weights(u);
                partition[v] = 0;
                left_n_vertices += 1;
                left_weight += g.weights(v);
            }
            return;
        }

        // choose a random starting vertex
        vertex_t s = vertices[(size_t) rand() % vertices.size()];
        s = get_far_s(s, g, scratch_memory);
        vertex_t far_s = get_far_s(s, g, scratch_memory);
        if (g.weights(s) < g.weights(far_s)) { std::swap(s, far_s); }

        // s is the larger vertex and far_s is the smaller vertex
        if (left_lmax - left_weight > right_lmax - right_weight) {
            partition[s] = 0;
            left_n_vertices += 1;
            left_weight += g.weights(s);
            partition[far_s] = 1;
            right_n_vertices += 1;
            right_weight += g.weights(far_s);
        } else {
            partition[s] = 1;
            right_n_vertices += 1;
            right_weight += g.weights(s);
            partition[far_s] = 0;
            left_n_vertices += 1;
            left_weight += g.weights(far_s);
        }

        std::vector<u64> states(g.n, 0);

        std::priority_queue<SmallMove> queue_0;
        std::priority_queue<SmallMove> queue_1;
        for (vertex_t i = g.neighborhood(s); i < g.neighborhood(s + 1); ++i) {
            vertex_t v = g.edges_v(i);
            if (partition[v] == 0 || partition[v] == 1) { continue; } // vertex already assigned

            weight_t w0, w1;
            edge_cut_not_assigned(g, v, partition, w0, w1);

            SmallMove move_0;
            move_0.u = v;
            move_0.cut = w0;
            move_0.state_id = states[v];

            SmallMove move_1;
            move_1.u = v;
            move_1.cut = w1;
            move_1.state_id = states[v];

            queue_0.push(move_0);
            queue_1.push(move_1);
        }

        for (vertex_t i = g.neighborhood(far_s); i < g.neighborhood(far_s + 1); ++i) {
            vertex_t v = g.edges_v(i);
            if (partition[v] == 0 || partition[v] == 1) { continue; } // vertex already assigned

            weight_t w0, w1;
            edge_cut_not_assigned(g, v, partition, w0, w1);

            SmallMove move_0;
            move_0.u = v;
            move_0.cut = w0;
            move_0.state_id = states[v];

            SmallMove move_1;
            move_1.u = v;
            move_1.cut = w1;
            move_1.state_id = states[v];

            queue_0.push(move_0);
            queue_1.push(move_1);
        }

        while (!queue_0.empty() || !queue_1.empty()) {
            bool draw_0 = true;
            if (queue_0.empty()) {
                // draw from queue_2
                draw_0 = false;
            } else if (queue_1.empty()) {
                // draw from queue_1
                draw_0 = true;
            } else if (left_lmax - left_weight > right_lmax - right_weight) {
                // draw from queue_1
                draw_0 = true;
            } else if (right_lmax - right_weight > left_lmax - left_weight) {
                // draw from queue_2
                draw_0 = false;
            } else if (left_lmax == right_lmax) {
                if (queue_1.top().cut < queue_0.top().cut) {
                    draw_0 = false;
                }
            }

            if (draw_0) {
                // draw from queue_0
                SmallMove move = queue_0.top();
                vertex_t u = move.u;
                u64 state_id = move.state_id;
                queue_0.pop();

                if (partition[u] == 0 || partition[u] == 1) { continue; } // vertex already assigned
                if (state_id != states[u]) { continue; }                  // this entry is outdated
                if (g.weights(u) > left_lmax - left_weight) { continue; } // vertex too heavy, we dont push it now

                partition[u] = 0;
                left_n_vertices += 1;
                left_weight += g.weights(u);
                for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    vertex_t v = g.edges_v(i);

                    if (partition[v] == 0 || partition[v] == 1) { continue; } // vertex already assigned

                    states[v] += 1;

                    weight_t w0, w1;
                    edge_cut_not_assigned(g, v, partition, w0, w1);

                    SmallMove move_0;
                    move_0.u = v;
                    move_0.cut = w0;
                    move_0.state_id = states[v];

                    SmallMove move_1;
                    move_1.u = v;
                    move_1.cut = w1;
                    move_1.state_id = states[v];

                    queue_0.push(move_0);
                    queue_1.push(move_1);
                }
            } else {
                // draw from queue_1
                SmallMove move = queue_1.top();
                vertex_t u = move.u;
                u64 state_id = move.state_id;
                queue_1.pop();

                if (partition[u] == 0 || partition[u] == 1) { continue; }   // vertex already assigned
                if (state_id != states[u]) { continue; }                    // this entry is outdated
                if (g.weights(u) > right_lmax - right_weight) { continue; } // vertex too heavy, we dont push it now

                partition[u] = 1;
                right_n_vertices += 1;
                right_weight += g.weights(u);
                for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    vertex_t v = g.edges_v(i);

                    if (partition[v] == 0 || partition[v] == 1) { continue; } // vertex already assigned

                    states[v] += 1;

                    weight_t w0, w1;
                    edge_cut_not_assigned(g, v, partition, w0, w1);

                    SmallMove move_0;
                    move_0.u = v;
                    move_0.cut = w0;
                    move_0.state_id = states[v];

                    SmallMove move_1;
                    move_1.u = v;
                    move_1.cut = w1;
                    move_1.state_id = states[v];

                    queue_0.push(move_0);
                    queue_1.push(move_1);
                }
            }
        }

        assign_missing_vertices(g, left_lmax, right_lmax, left_weight, right_weight, left_min_n_vertices, right_min_n_vertices, left_n_vertices, right_n_vertices, partition, vertices, scratch_memory);
    }

    inline void fix_min_n_vertices(const HostGraph &g,
                                   const weight_t left_lmax,
                                   const weight_t right_lmax,
                                   weight_t &left_weight,
                                   weight_t &right_weight,
                                   const vertex_t left_min_n_vertices,
                                   const vertex_t right_min_n_vertices,
                                   vertex_t &left_n_vertices,
                                   vertex_t &right_n_vertices,
                                   std::vector<partition_t> &partition,
                                   ScratchMemory &scratch_memory) {
    }

    inline void partition_2way_multiple_components(const HostGraph &g,
                                                   const weight_t left_lmax,
                                                   const weight_t right_lmax,
                                                   weight_t &left_weight,
                                                   weight_t &right_weight,
                                                   const vertex_t left_min_n_vertices,
                                                   const vertex_t right_min_n_vertices,
                                                   vertex_t &left_n_vertices,
                                                   vertex_t &right_n_vertices,
                                                   std::vector<partition_t> &partition,
                                                   ScratchMemory &scratch_memory) {
        vertex_t INF = std::numeric_limits<vertex_t>::max();

        // do a bfs to find all components
        std::vector<vertex_t> &component = scratch_memory.component;
        component.resize(g.n);
        std::fill(component.begin(), component.end(), INF);

        std::vector<vertex_t> &curr_stack = scratch_memory.curr_stack;
        std::vector<vertex_t> &next_stack = scratch_memory.next_stack;
        curr_stack.clear();
        next_stack.clear();
        vertex_t n_component = 0;

        for (vertex_t u = 0; u < g.n; ++u) {
            if (component[u] == INF) {
                // we have to do BFS from here
                curr_stack.clear();
                curr_stack.push_back(u);
                while (!curr_stack.empty()) {
                    for (vertex_t v: curr_stack) {
                        component[v] = n_component;
                        for (vertex_t i = g.neighborhood(v); i < g.neighborhood(v + 1); ++i) {
                            vertex_t uu = g.edges_v(i);
                            if (component[uu] == INF) {
                                component[uu] = INF - 1;
                                next_stack.push_back(uu);
                            }
                        }
                    }
                    curr_stack.swap(next_stack);
                    next_stack.clear();
                }
                n_component += 1;
            }
        }

        // get weight of all components
        std::vector<ComponentInfo> &components = scratch_memory.comp_weights;
        components.resize(n_component);
        for (vertex_t id = 0; id < n_component; ++id) {
            components[id].id = id;
            components[id].n_vertices = 0;
            components[id].weight = 0;
        }

        for (vertex_t u = 0; u < g.n; ++u) {
            components[component[u]].n_vertices += 1;
            components[component[u]].weight += g.weights(u);
        }

        // sort by weight decreasing
        std::sort(components.begin(), components.end(), [](const ComponentInfo &a, const ComponentInfo &b) { return a.weight > b.weight; });

        // assign components to the two blocks, one by one always choose block with more space first
        for (size_t i = 0; i < n_component; ++i) {
            vertex_t comp_id = components[i].id;
            weight_t comp_weight = components[i].weight;
            vertex_t comp_n_vertices = components[i].n_vertices;
            if (comp_weight <= std::max(left_lmax - left_weight, right_lmax - right_weight)) {
                // we can fit this component into the larger block
                if (left_lmax - left_weight > right_lmax - right_weight) {
                    left_n_vertices += comp_n_vertices;
                    left_weight += comp_weight;
                    for (vertex_t u = 0; u < g.n; ++u) { if (component[u] == comp_id) { partition[u] = 0; } }
                } else {
                    right_n_vertices += comp_n_vertices;
                    right_weight += comp_weight;
                    for (vertex_t u = 0; u < g.n; ++u) { if (component[u] == comp_id) { partition[u] = 1; } }
                }
            } else {
                // we can not fit the component into one block and have to split it
                partition_2way_one_component(g, left_lmax, right_lmax, left_weight, right_weight, left_min_n_vertices, right_min_n_vertices, left_n_vertices, right_n_vertices, partition, comp_id, component, scratch_memory);
            }
        }

        // fix min_n_vertices
        if (left_n_vertices < left_min_n_vertices || right_n_vertices < right_min_n_vertices) {
            fix_min_n_vertices(g, left_lmax, right_lmax, left_weight, right_weight, left_min_n_vertices, right_min_n_vertices, left_n_vertices, right_n_vertices, partition, scratch_memory);
        }
    }

    inline void make_swaps(const HostGraph &g,
                           const weight_t left_lmax,
                           const weight_t right_lmax,
                           weight_t &left_weight,
                           weight_t &right_weight,
                           const vertex_t left_min_n_vertices,
                           const vertex_t right_min_n_vertices,
                           vertex_t &left_n_vertices,
                           vertex_t &right_n_vertices,
                           std::vector<partition_t> &partition) {
        bool move_made = true;
        while ((left_weight > left_lmax || right_weight > right_lmax) && move_made) {
            move_made = false;
            // imbalanced, check if we can make swaps at the boundary

            weight_t curr_diff = 0;
            curr_diff += left_weight <= left_lmax ? 0 : left_weight - left_lmax;
            curr_diff += right_weight <= right_lmax ? 0 : right_weight - right_lmax;

            vertex_t best_v = g.n;
            f64 best_edge_cut_per_delta = -std::numeric_limits<f64>::max();

            for (vertex_t u = 0; u < g.n; ++u) {
                weight_t u_weight = g.weights(u);

                weight_t new_left_weight = partition[u] == 0 ? left_weight - u_weight : left_weight + u_weight;
                weight_t new_right_weight = partition[u] == 0 ? right_weight + u_weight : right_weight - u_weight;

                weight_t next_diff = 0;
                next_diff += new_left_weight <= left_lmax ? 0 : new_left_weight - left_lmax;
                next_diff += new_right_weight <= right_lmax ? 0 : new_right_weight - right_lmax;

                weight_t w_delta = curr_diff - next_diff;

                if (w_delta > 0) {
                    weight_t gain = 0;
                    for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                        vertex_t v = g.edges_v(i);
                        weight_t w = g.edges_w(i);

                        if (partition[u] != partition[v]) {
                            gain += w;
                        }
                        if (partition[u] == partition[v]) {
                            gain -= w;
                        }
                    }
                    f64 ratio = (f64) gain / (f64) w_delta;

                    if (ratio > best_edge_cut_per_delta) {
                        best_edge_cut_per_delta = ratio;
                        best_v = u;
                    }
                }
            }
            if (best_v != g.n) {
                move_made = true;
                weight_t best_v_weight = g.weights(best_v);
                if (partition[best_v] == 0) {
                    partition[best_v] = 1;
                    left_weight -= best_v_weight;
                    right_weight += best_v_weight;
                } else {
                    partition[best_v] = 0;
                    left_weight += best_v_weight;
                    right_weight -= best_v_weight;
                }
            }
        }
    }

    inline void partition_2way(const HostGraph &g,
                               const weight_t left_lmax,
                               const weight_t right_lmax,
                               const vertex_t left_min_n_vertices,
                               const vertex_t right_min_n_vertices,
                               std::vector<partition_t> &partition,
                               ScratchMemory &scratch_memory) {
        // each vertex into one partition
        if (g.n == 0) {
            return;
        }

        if (g.n == 1) {
            weight_t w = g.weights(0);

            // prefer feasible side
            if (w <= left_lmax && w <= right_lmax) {
                partition[0] = (left_lmax >= right_lmax) ? (partition_t) 0 : (partition_t) 1;
            } else if (w <= left_lmax) {
                partition[0] = (partition_t) 0;
            } else {
                partition[0] = (partition_t) 1;
            }
            return;
        }

        if (g.n == 2) {
            weight_t w0 = g.weights(0);
            weight_t w1 = g.weights(1);

            // try both assignments
            bool a0_ok = (w0 <= left_lmax && w1 <= right_lmax);
            bool a1_ok = (w1 <= left_lmax && w0 <= right_lmax);

            if (a0_ok && a1_ok) {
                // both feasible → heavier vertex goes to larger side
                if (left_lmax >= right_lmax) {
                    partition[0] = (w0 >= w1) ? 0 : 1;
                    partition[1] = 1 - partition[0];
                } else {
                    partition[0] = (w0 >= w1) ? 1 : 0;
                    partition[1] = 1 - partition[0];
                }
            } else if (a0_ok) {
                partition[0] = 0;
                partition[1] = 1;
            } else if (a1_ok) {
                partition[0] = 1;
                partition[1] = 0;
            } else {
                // neither feasible → minimize overload
                weight_t o0 = std::max<weight_t>(0, w0 - left_lmax) + std::max<weight_t>(0, w1 - right_lmax);
                weight_t o1 = std::max<weight_t>(0, w1 - left_lmax) + std::max<weight_t>(0, w0 - right_lmax);

                if (o0 <= o1) {
                    partition[0] = 0;
                    partition[1] = 1;
                } else {
                    partition[0] = 1;
                    partition[1] = 0;
                }
            }
            return;
        }

        for (vertex_t u = 0; u < g.n; ++u) { partition[u] = 2; }

        weight_t left_weight = 0;
        weight_t right_weight = 0;
        vertex_t left_n_vertices = 0;
        vertex_t right_n_vertices = 0;

        partition_2way_multiple_components(g, left_lmax, right_lmax, left_weight, right_weight, left_min_n_vertices, right_min_n_vertices, left_n_vertices, right_n_vertices, partition, scratch_memory);
        for (vertex_t u = 0; u < g.n; ++u) { ASSERT(partition[u] == 0 || partition[u] == 1); }
    }
}

#endif //GPU_HEIPA_RECURSIVE_BISECTION_2WAY_PARTITIONER_H
