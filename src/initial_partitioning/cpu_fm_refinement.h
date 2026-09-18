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

#ifndef GPU_HEIPA_CPU_FM_REFINEMENT_H
#define GPU_HEIPA_CPU_FM_REFINEMENT_H

#include <vector>
#include <queue>
#include <algorithm>
#include <utility>

#include <Kokkos_Core.hpp>

#include "../definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/small_graph.h"
#include "../datastructures/partition.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {

    inline void cpu_fm_refine_internal(vertex_t n,
                                       vertex_t m,
                                       const std::vector<u32> &h_edge_begin,
                                       const std::vector<u32> &h_edge_end,
                                       const std::vector<vertex_t> &h_edges_v,
                                       const std::vector<weight_t> &h_edges_w,
                                       const std::vector<weight_t> &h_v_weights,
                                       partition_t k,
                                       Partition &partition,
                                       const std::vector<weight_t> &block_lmax,
                                       DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "cpu_fm_refine", "cpu_fm_refine");

        if (n == 0 || k < 2) return;

        // Copy partition subview for the current graph level [0, n)
        auto map_sub = Kokkos::subview(partition.map, Kokkos::make_pair((vertex_t)0, n));
        std::vector<partition_t> h_map(n);
        std::vector<weight_t> h_bweights(k, 0);

        Kokkos::deep_copy(exec_space, Kokkos::View<partition_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_map.data(), n), map_sub);
        Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_bweights.data(), k), partition.bweights);
        exec_space.fence();

        bool overall_improved = false;

        // Priority queue item: {gain, vertex}
        using GainVertex = std::pair<weight_t, vertex_t>;

        // Perform pairwise 2-way FM passes between interacting blocks
        for (int pass = 0; pass < 4; ++pass) {
            bool pass_improved = false;

            for (partition_t u_id = 0; u_id < k; ++u_id) {
                for (partition_t v_id = u_id + 1; v_id < k; ++v_id) {
                    std::priority_queue<GainVertex> pq_u;
                    std::priority_queue<GainVertex> pq_v;

                    auto compute_gain = [&](vertex_t u, partition_t from, partition_t to) -> weight_t {
                        weight_t int_w = 0;
                        weight_t ext_w = 0;
                        for (u32 e = h_edge_begin[u]; e < h_edge_end[u]; ++e) {
                            vertex_t neighbor = h_edges_v[e];
                            partition_t n_part = h_map[neighbor];
                            weight_t ew = h_edges_w[e];
                            if (n_part == from) int_w += ew;
                            else if (n_part == to) ext_w += ew;
                        }
                        return ext_w - int_w;
                    };

                    for (vertex_t u = 0; u < n; ++u) {
                        if (h_map[u] == u_id) {
                            for (u32 e = h_edge_begin[u]; e < h_edge_end[u]; ++e) {
                                if (h_map[h_edges_v[e]] == v_id) {
                                    pq_u.push({compute_gain(u, u_id, v_id), u});
                                    break;
                                }
                            }
                        } else if (h_map[u] == v_id) {
                            for (u32 e = h_edge_begin[u]; e < h_edge_end[u]; ++e) {
                                if (h_map[h_edges_v[e]] == u_id) {
                                    pq_v.push({compute_gain(u, v_id, u_id), u});
                                    break;
                                }
                            }
                        }
                    }

                    if (pq_u.empty() && pq_v.empty()) continue;

                    // Rollout 2-way FM sequence
                    struct FMMove {
                        vertex_t u;
                        partition_t from;
                        partition_t to;
                        weight_t gain;
                        weight_t weight;
                    };

                    std::vector<FMMove> move_history;
                    std::vector<bool> moved(n, false);

                    weight_t u_bw = h_bweights[u_id];
                    weight_t v_bw = h_bweights[v_id];
                    weight_t u_lmax = block_lmax[u_id];
                    weight_t v_lmax = block_lmax[v_id];

                    auto compute_overload = [&](weight_t uw, weight_t vw) -> weight_t {
                        weight_t ol = 0;
                        if (uw > u_lmax) ol += (uw - u_lmax);
                        if (vw > v_lmax) ol += (vw - v_lmax);
                        return ol;
                    };

                    weight_t init_overload = compute_overload(u_bw, v_bw);
                    weight_t min_overload = init_overload;
                    weight_t curr_gain = 0;
                    weight_t best_gain = 0;
                    size_t best_step = 0;

                    while (!pq_u.empty() || !pq_v.empty()) {
                        while (!pq_u.empty() && moved[pq_u.top().second]) pq_u.pop();
                        while (!pq_v.empty() && moved[pq_v.top().second]) pq_v.pop();
                        if (pq_u.empty() && pq_v.empty()) break;

                        auto try_candidate = [&](bool is_u, vertex_t &cand_v, weight_t &cand_g, partition_t &cand_from, partition_t &cand_to, weight_t &cand_vw, weight_t &cand_to_nbw, weight_t &cand_from_nbw) -> bool {
                            auto &pq = is_u ? pq_u : pq_v;
                            if (pq.empty()) return false;
                            cand_v = pq.top().second;
                            cand_g = pq.top().first;
                            cand_from = is_u ? u_id : v_id;
                            cand_to = is_u ? v_id : u_id;
                            cand_vw = h_v_weights[cand_v];
                            cand_to_nbw = (cand_to == u_id ? u_bw : v_bw) + cand_vw;
                            cand_from_nbw = (cand_from == u_id ? u_bw : v_bw) - cand_vw;
                            weight_t to_lmax = (cand_to == u_id ? u_lmax : v_lmax);
                            if (to_lmax > 0 && cand_to_nbw > to_lmax) {
                                weight_t new_uw = (cand_from == u_id ? cand_from_nbw : cand_to_nbw);
                                weight_t new_vw = (cand_from == v_id ? cand_from_nbw : cand_to_nbw);
                                if (compute_overload(new_uw, new_vw) >= compute_overload(u_bw, v_bw)) {
                                    return false;
                                }
                            }
                            return true;
                        };

                        vertex_t vert = 0;
                        weight_t g_val = 0;
                        partition_t from = 0, to = 0;
                        weight_t vw = 0, to_new_bw = 0, from_new_bw = 0;

                        bool u_valid = try_candidate(true, vert, g_val, from, to, vw, to_new_bw, from_new_bw);
                        vertex_t v_vert = 0;
                        weight_t v_g_val = 0;
                        partition_t v_from = 0, v_to = 0;
                        weight_t v_vw = 0, v_to_new_bw = 0, v_from_new_bw = 0;
                        bool v_valid = try_candidate(false, v_vert, v_g_val, v_from, v_to, v_vw, v_to_new_bw, v_from_new_bw);

                        bool choose_u = true;
                        if (u_valid && v_valid) {
                            if (u_bw > u_lmax && v_bw <= v_lmax) {
                                choose_u = true;
                            } else if (v_bw > v_lmax && u_bw <= u_lmax) {
                                choose_u = false;
                            } else if (v_g_val > g_val) {
                                choose_u = false;
                            } else if (g_val == v_g_val) {
                                choose_u = (u_bw >= v_bw);
                            }
                        } else if (u_valid) {
                            choose_u = true;
                        } else if (v_valid) {
                            choose_u = false;
                        } else {
                            // Neither top candidate is feasible; pop both if they exist to allow other vertices a chance
                            if (!pq_u.empty()) pq_u.pop();
                            if (!pq_v.empty()) pq_v.pop();
                            continue;
                        }

                        if (choose_u) {
                            pq_u.pop();
                        } else {
                            pq_v.pop();
                            vert = v_vert;
                            g_val = v_g_val;
                            from = v_from;
                            to = v_to;
                            vw = v_vw;
                            to_new_bw = v_to_new_bw;
                            from_new_bw = v_from_new_bw;
                        }

                        moved[vert] = true;
                        curr_gain += g_val;
                        (from == u_id ? u_bw : v_bw) = from_new_bw;
                        (to == u_id ? u_bw : v_bw) = to_new_bw;
                        h_map[vert] = to;

                        move_history.push_back({vert, from, to, g_val, vw});

                        weight_t curr_overload = compute_overload(u_bw, v_bw);
                        bool update_best = false;
                        if (init_overload > 0) {
                            if (curr_overload < min_overload) {
                                update_best = true;
                            } else if (curr_overload == min_overload && curr_gain > best_gain) {
                                update_best = true;
                            }
                        } else {
                            if (curr_overload == 0 && curr_gain > best_gain) {
                                update_best = true;
                            }
                        }

                        if (update_best) {
                            min_overload = curr_overload;
                            best_gain = curr_gain;
                            best_step = move_history.size();
                        }

                        for (u32 e = h_edge_begin[vert]; e < h_edge_end[vert]; ++e) {
                            vertex_t neighbor = h_edges_v[e];
                            if (moved[neighbor]) continue;
                            partition_t n_part = h_map[neighbor];
                            if (n_part == u_id) {
                                pq_u.push({compute_gain(neighbor, u_id, v_id), neighbor});
                            } else if (n_part == v_id) {
                                pq_v.push({compute_gain(neighbor, v_id, u_id), neighbor});
                            }
                        }
                    }

                    // Revert moves past best_step
                    for (size_t i = move_history.size(); i > best_step; --i) {
                        auto &m = move_history[i - 1];
                        h_map[m.u] = m.from;
                    }

                    if (best_step > 0 && (min_overload < init_overload || (min_overload == 0 && best_gain > 0))) {
                        pass_improved = true;
                        overall_improved = true;
                        for (size_t i = 0; i < best_step; ++i) {
                            auto &m = move_history[i];
                            h_bweights[m.from] -= m.weight;
                            h_bweights[m.to] += m.weight;
                        }
                    }
                }
            }
            if (!pass_improved) break;
        }

        if (overall_improved) {
            Kokkos::deep_copy(exec_space, map_sub, Kokkos::View<const partition_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_map.data(), n));
            Kokkos::deep_copy(exec_space, partition.bweights, Kokkos::View<const weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_bweights.data(), k));
            KOKKOS_PROFILE_FENCE(exec_space);
        }
    }

    inline void cpu_fm_refine(const SmallGraph &g,
                              partition_t k,
                              Partition &partition,
                              const UnmanagedDeviceU32 &current_targets_dev,
                              weight_t g_weight,
                              f64 imbalance,
                              DeviceExecutionSpace &exec_space) {
        const vertex_t n = g.n;
        if (n == 0 || k < 2) return;

        std::vector<u32> h_edge_begin(n);
        std::vector<u32> h_edge_end(n);
        std::vector<vertex_t> h_edges_v(g.m);
        std::vector<weight_t> h_edges_w(g.m, 1);
        std::vector<weight_t> h_v_weights(n, 1);
        std::vector<u32> h_targets(k);

        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edge_begin.data(), n), g.edge_begin);
        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edge_end.data(), n), g.edge_end);
        Kokkos::deep_copy(exec_space, Kokkos::View<vertex_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_v.data(), g.m), g.edges_v);
        if (!g.uniform_edge_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_w.data(), g.m), g.edges_w);
        }
        if (!g.uniform_vertex_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_v_weights.data(), n), g.weights);
        }
        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_targets.data(), k), current_targets_dev);
        exec_space.fence();

        partition_t total_target_parts = 0;
        for (partition_t i = 0; i < k; ++i) {
            u32 packed_targets = h_targets[i];
            partition_t num_target_parts = (packed_targets & 0xFFFF) + (packed_targets >> 16);
            if (num_target_parts == 0) num_target_parts = 1;
            total_target_parts += num_target_parts;
        }
        if (total_target_parts == 0) total_target_parts = k;

        std::vector<weight_t> block_lmax(k);
        for (partition_t i = 0; i < k; ++i) {
            u32 packed_targets = h_targets[i];
            partition_t num_target_parts = (packed_targets & 0xFFFF) + (packed_targets >> 16);
            if (num_target_parts == 0) num_target_parts = 1;
            block_lmax[i] = (weight_t) ((((f64) num_target_parts * (1.0 + imbalance) * (f64) g_weight) / (f64) total_target_parts) + 0.999999);
        }

        cpu_fm_refine_internal(n, g.m, h_edge_begin, h_edge_end, h_edges_v, h_edges_w, h_v_weights, k, partition, block_lmax, exec_space);
    }

    inline void cpu_fm_refine(const SmallGraph &g,
                              partition_t k,
                              Partition &partition,
                              const std::vector<weight_t> &block_lmax,
                              DeviceExecutionSpace &exec_space) {
        const vertex_t n = g.n;
        if (n == 0 || k < 2) return;

        std::vector<u32> h_edge_begin(n);
        std::vector<u32> h_edge_end(n);
        std::vector<vertex_t> h_edges_v(g.m);
        std::vector<weight_t> h_edges_w(g.m, 1);
        std::vector<weight_t> h_v_weights(n, 1);

        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edge_begin.data(), n), g.edge_begin);
        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edge_end.data(), n), g.edge_end);
        Kokkos::deep_copy(exec_space, Kokkos::View<vertex_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_v.data(), g.m), g.edges_v);
        if (!g.uniform_edge_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_w.data(), g.m), g.edges_w);
        }
        if (!g.uniform_vertex_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_v_weights.data(), n), g.weights);
        }
        exec_space.fence();

        cpu_fm_refine_internal(n, g.m, h_edge_begin, h_edge_end, h_edges_v, h_edges_w, h_v_weights, k, partition, block_lmax, exec_space);
    }

    inline void cpu_fm_refine(const SmallGraph &g,
                              partition_t k,
                              Partition &partition,
                              DeviceExecutionSpace &exec_space) {
        const vertex_t n = g.n;
        if (n == 0 || k < 2) return;

        std::vector<u32> h_edge_begin(n);
        std::vector<u32> h_edge_end(n);
        std::vector<vertex_t> h_edges_v(g.m);
        std::vector<weight_t> h_edges_w(g.m, 1);
        std::vector<weight_t> h_v_weights(n, 1);

        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edge_begin.data(), n), g.edge_begin);
        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edge_end.data(), n), g.edge_end);
        Kokkos::deep_copy(exec_space, Kokkos::View<vertex_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_v.data(), g.m), g.edges_v);
        if (!g.uniform_edge_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_w.data(), g.m), g.edges_w);
        }
        if (!g.uniform_vertex_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_v_weights.data(), n), g.weights);
        }
        exec_space.fence();

        std::vector<weight_t> block_lmax(k, partition.lmax);
        cpu_fm_refine_internal(n, g.m, h_edge_begin, h_edge_end, h_edges_v, h_edges_w, h_v_weights, k, partition, block_lmax, exec_space);
    }

    inline void cpu_fm_refine(const Graph &g,
                              partition_t k,
                              Partition &partition,
                              DeviceExecutionSpace &exec_space) {
        const vertex_t n = g.n;
        if (n == 0 || k < 2) return;

        std::vector<u32> h_neighborhood(n + 1);
        std::vector<vertex_t> h_edges_v(g.m);
        std::vector<weight_t> h_edges_w(g.m, 1);
        std::vector<weight_t> h_v_weights(n, 1);

        Kokkos::deep_copy(exec_space, Kokkos::View<u32*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_neighborhood.data(), n + 1), g.neighborhood);
        Kokkos::deep_copy(exec_space, Kokkos::View<vertex_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_v.data(), g.m), g.edges_v);
        if (!g.uniform_edge_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_edges_w.data(), g.m), g.edges_w);
        }
        if (!g.uniform_vertex_weights) {
            Kokkos::deep_copy(exec_space, Kokkos::View<weight_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_v_weights.data(), n), g.weights);
        }
        exec_space.fence();

        std::vector<u32> h_edge_begin(n);
        std::vector<u32> h_edge_end(n);
        for (vertex_t u = 0; u < n; ++u) {
            h_edge_begin[u] = h_neighborhood[u];
            h_edge_end[u] = h_neighborhood[u + 1];
        }

        std::vector<weight_t> block_lmax(k, partition.lmax);
        cpu_fm_refine_internal(n, g.m, h_edge_begin, h_edge_end, h_edges_v, h_edges_w, h_v_weights, k, partition, block_lmax, exec_space);
    }

} // namespace GPU_HeiPa

#endif // GPU_HEIPA_CPU_FM_REFINEMENT_H
