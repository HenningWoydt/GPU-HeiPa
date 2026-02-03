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
#include <random>

#include "../utility/definitions.h"
#include "../utility/macros.h"
#include "../utility/edge_cut.h"

namespace GPU_HeiPa {
    struct ComponentInfo {
        vertex_t id;
        vertex_t n_vertices;
        weight_t weight;
    };

    struct SmallMove {
        vertex_t u = 0;
        partition_t id = 2;
        weight_t induced_cut = 0;
        u64 state_id = 0;

        bool operator<(const SmallMove &m) const { return induced_cut > m.induced_cut; }
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

        std::vector<u64> states;

        std::priority_queue<SmallMove> queue_0;
        std::priority_queue<SmallMove> queue_1;
    };

    inline void edge_cut_not_assigned(const HostGraph &g,
                                      const vertex_t u,
                                      const std::vector<partition_t> &partition,
                                      weight_t &cut_to_0,
                                      weight_t &cut_to_1) {
        cut_to_0 = 0;
        cut_to_1 = 0;
        for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
            vertex_t v = g.edges_v(i);
            weight_t w = g.edges_w(i);

            if (partition[v] == 0) cut_to_1 += w;
            else if (partition[v] == 1) cut_to_0 += w;
        }
    }

    inline vertex_t get_far_s(const vertex_t s,
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
                                        const std::vector<vertex_t> &vertices,
                                        ScratchMemory &scratch_memory) {
        std::vector<vertex_t> &missing_vertices = scratch_memory.missing_vertices;
        std::vector<vertex_t> &next_missing_vertices = scratch_memory.next_missing_vertices;
        missing_vertices.clear();
        next_missing_vertices.clear();

        for (vertex_t u: vertices) { if (partition[u] != 0 && partition[u] != 1) { missing_vertices.push_back(u); } }

        while (!missing_vertices.empty()) {
            for (vertex_t u: missing_vertices) {
                weight_t u_weight = g.weights(u);

                weight_t cut_to_0, cut_to_1;
                edge_cut_not_assigned(g, u, partition, cut_to_0, cut_to_1);

                if (cut_to_0 == 0 && cut_to_1 == 0) {
                    next_missing_vertices.push_back(u);
                    continue;
                }

                if (cut_to_0 < cut_to_1) {
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
        //
        {
            ScopedTimer _t("initial_partitioning", "2way_one_big", "determine_component");

            for (vertex_t u = 0; u < g.n; ++u) { if (component[u] == component_id) { vertices.push_back(u); } }
            if (vertices.empty()) { return; }
        }

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
        vertex_t s, far_s;
        // choose a random starting vertex
        {
            ScopedTimer _t("initial_partitioning", "2way_one_big", "get_seeds");

            s = vertices[(size_t) rand() % vertices.size()];
            s = get_far_s(s, g, scratch_memory);
            far_s = get_far_s(s, g, scratch_memory);
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
        }

        std::vector<u64> &states = scratch_memory.states;
        std::priority_queue<SmallMove> &queue = scratch_memory.queue_0;
        //
        {
            ScopedTimer _t("initial_partitioning", "2way_one_big", "init_queue_memory");

            states.resize(g.n);
            std::fill(states.begin(), states.end(), 0);

            queue = std::priority_queue<SmallMove>();
        }
        //

        std::vector<weight_t> cut_to_0(g.n, 0); // additional edge cut if we would assign u to 0 now
        std::vector<weight_t> cut_to_1(g.n, 0); // additional edge cut if we would assign u to 1 now

        {
            ScopedTimer _t("initial_partitioning", "2way_one_big", "init_queue");

            for (vertex_t i = g.neighborhood(s); i < g.neighborhood(s + 1); ++i) {
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                if (partition[v] == 0 || partition[v] == 1) { continue; } // vertex already assigned, no need to update

                if (partition[s] == 0) { cut_to_1[v] += w; }
                if (partition[s] == 1) { cut_to_0[v] += w; }
            }

            for (vertex_t i = g.neighborhood(far_s); i < g.neighborhood(far_s + 1); ++i) {
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                if (partition[v] == 0 || partition[v] == 1) { continue; } // vertex already assigned, no need to update

                if (partition[far_s] == 0) { cut_to_1[v] += w; }
                if (partition[far_s] == 1) { cut_to_0[v] += w; }
            }
        }
        //
        {
            ScopedTimer _t("initial_partitioning", "2way_one_big", "process_queue");
            while (true) {
                // linear search of best move for now
                vertex_t best_u = g.n;
                partition_t best_side = 0;
                weight_t best_add_cut = std::numeric_limits<weight_t>::max();
                weight_t best_strength = 0;

                for (vertex_t u = 0; u < g.n; ++u) {
                    if (partition[u] == 0 || partition[u] == 1) { continue; } // already assigned, skip

                    weight_t w = g.weights(u);

                    bool can0 = (left_weight + w <= left_lmax);
                    bool can1 = (right_weight + w <= right_lmax);
                    if (!can0 && !can1) { continue; } // no side can take it

                    partition_t side;
                    weight_t add_cut;
                    if (can0 && can1) {
                        if (cut_to_0[u] <= cut_to_1[u]) {
                            side = 0;
                            add_cut = cut_to_0[u];
                        } else {
                            side = 1;
                            add_cut = cut_to_1[u];
                        }
                    } else if (can0) {
                        side = 0;
                        add_cut = cut_to_0[u];
                    } else {
                        side = 1;
                        add_cut = cut_to_1[u];
                    }

                    // tie-breaks:
                    // 1) smaller added cut is better
                    // 2) if similar, take stronger preference to stabilize boundary early
                    weight_t strength = (cut_to_0[u] >= cut_to_1[u]) ? (cut_to_0[u] - cut_to_1[u]) : (cut_to_1[u] - cut_to_0[u]);

                    // optional: small bias toward the side with more remaining capacity
                    // (helps feasibility without dominating cut quality)
                    weight_t rem0 = left_lmax - left_weight;
                    weight_t rem1 = right_lmax - right_weight;
                    if (can0 && can1) {
                        if (rem0 > rem1 * (weight_t) 1.05 && side != 0 && cut_to_0[u] <= cut_to_1[u] + (weight_t) 0) {
                            side = 0;
                            add_cut = cut_to_0[u];
                        } else if (rem1 > rem0 * (weight_t) 1.05 && side != 1 && cut_to_1[u] <= cut_to_0[u] + (weight_t) 0) {
                            side = 1;
                            add_cut = cut_to_1[u];
                        }
                    }

                    if (add_cut < best_add_cut || (add_cut == best_add_cut && strength > best_strength)) {
                        best_add_cut = add_cut;
                        best_strength = strength;
                        best_u = u;
                        best_side = side;
                    }
                }

                // no move found
                if (best_u == g.n) { break; }

                // apply move
                if (best_side == 0) {
                    partition[best_u] = 0;
                    left_n_vertices += 1;
                    left_weight += g.weights(best_u);
                } else {
                    partition[best_u] = 1;
                    right_n_vertices += 1;
                    right_weight += g.weights(best_u);
                }
                for (vertex_t i = g.neighborhood(best_u); i < g.neighborhood(best_u + 1); ++i) {
                    vertex_t v = g.edges_v(i);
                    weight_t w = g.edges_w(i);

                    if (partition[v] == 0 || partition[v] == 1) { continue; } // vertex already assigned, no need to update

                    if (partition[best_u] == 0) { cut_to_1[v] += w; }
                    if (partition[best_u] == 1) { cut_to_0[v] += w; }
                }
            }
        }

        ScopedTimer _t("initial_partitioning", "2way_one_big", "assign_missing_vertices");
        assign_missing_vertices(g, left_lmax, right_lmax, left_weight, right_weight, left_min_n_vertices, right_min_n_vertices, left_n_vertices, right_n_vertices, partition, vertices, scratch_memory);
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
        std::vector<vertex_t> &component = scratch_memory.component;
        std::vector<ComponentInfo> &components = scratch_memory.comp_weights;
        vertex_t n_component = 0;
        // determine the components
        {
            ScopedTimer _t("initial_partitioning", "2way_partition", "determine_components");

            vertex_t INF = std::numeric_limits<vertex_t>::max();

            // do a bfs to find all components
            component.resize(g.n);
            std::fill(component.begin(), component.end(), INF);

            std::vector<vertex_t> &curr_stack = scratch_memory.curr_stack;
            std::vector<vertex_t> &next_stack = scratch_memory.next_stack;
            curr_stack.clear();
            next_stack.clear();

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
        }

        // sort by weight decreasing
        {
            ScopedTimer _t("initial_partitioning", "2way_partition", "sort_components");

            std::sort(components.begin(), components.end(), [](const ComponentInfo &a, const ComponentInfo &b) { return a.weight > b.weight; });
        }

        // assign components to the two blocks, one by one always choose block with more space first
        for (size_t i = 0; i < n_component; ++i) {
            vertex_t comp_id = components[i].id;
            weight_t comp_weight = components[i].weight;
            vertex_t comp_n_vertices = components[i].n_vertices;
            if (comp_weight <= std::max(left_lmax - left_weight, right_lmax - right_weight)) {
                ScopedTimer _t("initial_partitioning", "2way_partition", "insert_small_component");
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
    }

    struct FMMove {
        vertex_t u;
        weight_t gain; // higher is better
        u64 state_id;
        bool operator<(const FMMove &o) const { return gain < o.gain; } // max-heap
    };

    inline void fm_refinement(const HostGraph &g,
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
        ScopedTimer _t("initial_partitioning", "fm_refinement", "fm_refinement");

        const vertex_t n = g.n;
        if (n == 0) { return; }

        // -----------------------------
        // Build internal/external sums
        // -----------------------------
        std::vector<weight_t> internal(n, 0);
        std::vector<weight_t> external(n, 0);

        for (vertex_t u = 0; u < n; ++u) {
            partition_t pu = partition[u];
            for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);
                if (partition[v] == pu) internal[u] += w;
                else external[u] += w;
            }
        }

        // -----------------------------
        // Boundary vertices
        // -----------------------------
        std::vector<vertex_t> boundary_vertices;
        boundary_vertices.reserve(n);

        for (vertex_t u = 0; u < n; ++u) {
            partition_t pu = partition[u];
            bool boundary = false;
            for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                vertex_t v = g.edges_v(i);
                if (partition[v] != pu) {
                    boundary = true;
                    break;
                }
            }
            if (boundary) boundary_vertices.push_back(u);
        }

        // optional shuffle (kept from your structure; not required for PQ correctness)
        std::shuffle(boundary_vertices.begin(), boundary_vertices.end(), std::default_random_engine((unsigned) n));
        // boundary_vertices.reserve(boundary_vertices.size() / 2);

        // -----------------------------
        // Init scratch structures
        // -----------------------------
        std::vector<u64> &states = scratch_memory.states;
        states.resize(n);
        std::fill(states.begin(), states.end(), 0);

        std::priority_queue<SmallMove> &queue_0 = scratch_memory.queue_0; // candidates moving to 0 (from 1)
        std::priority_queue<SmallMove> &queue_1 = scratch_memory.queue_1; // candidates moving to 1 (from 0)
        queue_0 = std::priority_queue<SmallMove>();
        queue_1 = std::priority_queue<SmallMove>();

        std::vector<u8> moved(n, 0);
        std::vector<partition_t> old_side(n, 2);

        std::vector<vertex_t> moved_stack;
        moved_stack.reserve(boundary_vertices.size());

        // -----------------------------
        // Push initial moves (boundary only)
        // Store induced_cut = -gain so that a "min by induced_cut" queue behaves like max-gain.
        // If your SmallMove comparator is different, adjust accordingly.
        // -----------------------------
        for (vertex_t idx = 0; idx < (vertex_t) boundary_vertices.size(); ++idx) {
            vertex_t u = boundary_vertices[idx];
            if (partition[u] == 0) {
                // move 0 -> 1, target id = 1
                weight_t gain = external[u] - internal[u];
                SmallMove mv;
                mv.u = u;
                mv.id = 1;
                mv.induced_cut = -gain;
                mv.state_id = states[u];
                queue_1.push(mv);
            } else {
                // move 1 -> 0, target id = 0
                weight_t gain = external[u] - internal[u];
                SmallMove mv;
                mv.u = u;
                mv.id = 0;
                mv.induced_cut = -gain;
                mv.state_id = states[u];
                queue_0.push(mv);
            }
        }

        // -----------------------------
        // FM pass + best-prefix tracking
        // -----------------------------
        weight_t curr_gain_sum = 0;
        weight_t best_gain_sum = 0;
        size_t best_prefix = 0;

        while (!queue_0.empty() || !queue_1.empty()) {
            // -----------------------------------------
            // Get best feasible candidate from queue_0
            // (move 1 -> 0)
            // -----------------------------------------
            bool has0 = false;
            SmallMove best0;

            while (!queue_0.empty()) {
                SmallMove mv = queue_0.top();
                queue_0.pop();

                vertex_t u = mv.u;
                if (moved[u]) continue;
                if (mv.state_id != states[u]) continue;
                if (partition[u] != 1) continue; // must be in 1 to move to 0
                // feasibility
                weight_t wu = g.weights(u);
                if (left_weight + wu > left_lmax) continue;
                if (right_n_vertices <= right_min_n_vertices) continue;
                if (left_n_vertices + 1 < left_min_n_vertices) continue; // usually redundant
                // ok
                best0 = mv;
                has0 = true;
                break;
            }

            // -----------------------------------------
            // Get best feasible candidate from queue_1
            // (move 0 -> 1)
            // -----------------------------------------
            bool has1 = false;
            SmallMove best1;

            while (!queue_1.empty()) {
                SmallMove mv = queue_1.top();
                queue_1.pop();

                vertex_t u = mv.u;
                if (moved[u]) continue;
                if (mv.state_id != states[u]) continue;
                if (partition[u] != 0) continue; // must be in 0 to move to 1
                // feasibility
                weight_t wu = g.weights(u);
                if (right_weight + wu > right_lmax) continue;
                if (left_n_vertices <= left_min_n_vertices) continue;
                if (right_n_vertices + 1 < right_min_n_vertices) continue; // usually redundant
                // ok
                best1 = mv;
                has1 = true;
                break;
            }

            if (!has0 && !has1) { break; }

            // -----------------------------------------
            // Choose move (higher gain preferred)
            // gain = -induced_cut
            // -----------------------------------------
            bool choose_move_to_0 = false;
            if (has0 && !has1) {
                choose_move_to_0 = true;
            } else if (!has0 && has1) {
                choose_move_to_0 = false;
            } else {
                weight_t gain0 = -best0.induced_cut;
                weight_t gain1 = -best1.induced_cut;

                if (gain0 > gain1) choose_move_to_0 = true;
                else if (gain1 > gain0) choose_move_to_0 = false;
                else {
                    // tie-break: relieve overload / keep feasibility
                    if (left_weight > left_lmax && right_weight <= right_lmax) choose_move_to_0 = false;     // move 0->1
                    else if (right_weight > right_lmax && left_weight <= left_lmax) choose_move_to_0 = true; // move 1->0
                    else choose_move_to_0 = true;
                }
            }

            // -----------------------------------------
            // Apply chosen move
            // -----------------------------------------
            vertex_t u;
            partition_t from_side;
            partition_t to_side;

            if (choose_move_to_0) {
                // move 1 -> 0
                u = best0.u;
                from_side = 1;
                to_side = 0;
            } else {
                // move 0 -> 1
                u = best1.u;
                from_side = 0;
                to_side = 1;
            }

            // current gain for u (under current internal/external)
            weight_t gain_u = external[u] - internal[u];
            curr_gain_sum += gain_u;

            // record old side for rollback
            old_side[u] = from_side;

            // update balance
            weight_t wu = g.weights(u);
            if (from_side == 0 && to_side == 1) {
                left_weight -= wu;
                right_weight += wu;
                left_n_vertices -= 1;
                right_n_vertices += 1;
            } else {
                // 1 -> 0
                right_weight -= wu;
                left_weight += wu;
                right_n_vertices -= 1;
                left_n_vertices += 1;
            }

            // flip + lock
            partition[u] = to_side;
            moved[u] = 1;
            moved_stack.push_back(u);

            // update best prefix
            if (curr_gain_sum > best_gain_sum) {
                best_gain_sum = curr_gain_sum;
                best_prefix = moved_stack.size();
            }

            // -----------------------------------------
            // Incremental update of neighbors' internal/external
            // and push updated moves for neighbors
            // -----------------------------------------
            for (vertex_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                vertex_t v = g.edges_v(i);
                if (moved[v]) continue;

                weight_t w = g.edges_w(i);

                // v stayed where it is; u moved from_side -> to_side
                // If v is on from_side: edge was internal, becomes external
                if (partition[v] == from_side) {
                    internal[v] -= w;
                    external[v] += w;
                }
                // If v is on to_side: edge was external, becomes internal
                else if (partition[v] == to_side) {
                    external[v] -= w;
                    internal[v] += w;
                }

                // invalidate old entries for v and push new one
                states[v] += 1;

                // only boundary vertices need to be queued;
                // cheap boundary check (scan adjacency)
                bool boundary = false;
                partition_t pv = partition[v];
                for (vertex_t j = g.neighborhood(v); j < g.neighborhood(v + 1); ++j) {
                    vertex_t x = g.edges_v(j);
                    if (partition[x] != pv) {
                        boundary = true;
                        break;
                    }
                }
                if (!boundary) continue;

                weight_t gain_v = external[v] - internal[v];
                SmallMove mv;
                mv.u = v;
                mv.state_id = states[v];
                mv.induced_cut = -gain_v;

                if (partition[v] == 0) {
                    mv.id = 1; // 0 -> 1
                    queue_1.push(mv);
                } else {
                    mv.id = 0; // 1 -> 0
                    queue_0.push(mv);
                }
            }
        }

        // -----------------------------
        // Roll back to best prefix
        // -----------------------------
        for (size_t idx = moved_stack.size(); idx-- > best_prefix;) {
            vertex_t u = moved_stack[idx];
            partition_t from_side = old_side[u];
            partition_t to_side = partition[u]; // current side

            weight_t wu = g.weights(u);

            // revert balance
            if (from_side == 0 && to_side == 1) {
                // we had moved 0->1, revert 1->0
                right_weight -= wu;
                left_weight += wu;
                right_n_vertices -= 1;
                left_n_vertices += 1;
            } else {
                // we had moved 1->0, revert 0->1
                left_weight -= wu;
                right_weight += wu;
                left_n_vertices -= 1;
                right_n_vertices += 1;
            }

            partition[u] = from_side;
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
        if (g.n == 0) { return; }

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

        std::vector<partition_t> temp_partition = partition;
        weight_t best_edge_cut = 100000000;

        for (size_t i = 0; i < 10; ++i) {
            partition_2way_multiple_components(g, left_lmax, right_lmax, left_weight, right_weight, left_min_n_vertices, right_min_n_vertices, left_n_vertices, right_n_vertices, partition, scratch_memory);
            for (vertex_t u = 0; u < g.n; ++u) { ASSERT(partition[u] == 0 || partition[u] == 1); }

            fm_refinement(g, left_lmax, right_lmax, left_weight, right_weight, left_min_n_vertices, right_min_n_vertices, left_n_vertices, right_n_vertices, partition, scratch_memory);
            for (vertex_t u = 0; u < g.n; ++u) { ASSERT(partition[u] == 0 || partition[u] == 1); }

            weight_t temp_edge_cut = edge_cut(g, partition);
            if (temp_edge_cut < best_edge_cut) {
                best_edge_cut = temp_edge_cut;
                std::copy(partition.begin(), partition.end(), temp_partition.begin());
            }
        }
        std::copy(temp_partition.begin(), temp_partition.end(), partition.begin());

    }
}

#endif //GPU_HEIPA_RECURSIVE_BISECTION_2WAY_PARTITIONER_H
