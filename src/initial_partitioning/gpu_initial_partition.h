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

#ifndef GPU_HEIPA_GPU_INITIAL_MULTISECTION_H
#define GPU_HEIPA_GPU_INITIAL_MULTISECTION_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/two_hop_matching.h"
#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "global_multisection.h" // Assuming this is needed based on previous context

namespace GPU_HeiPa {
    struct BruteForceResult {
        f64 penalty;
        weight_t cut;
        u64 bitmask;

        KOKKOS_INLINE_FUNCTION BruteForceResult() : penalty(1e300), cut(std::numeric_limits<weight_t>::max()), bitmask(0) {
        }

        KOKKOS_INLINE_FUNCTION bool operator<(const BruteForceResult &other) const {
            if (penalty != other.penalty) return penalty < other.penalty;
            if (cut != other.cut) return cut < other.cut;
            return bitmask < other.bitmask;
        }
    };
} // end namespace GPU_HeiPa

namespace Kokkos {
    template<>
    struct reduction_identity<GPU_HeiPa::BruteForceResult> {
        KOKKOS_FORCEINLINE_FUNCTION static GPU_HeiPa::BruteForceResult min() {
            return GPU_HeiPa::BruteForceResult();
        }
    };
} // end namespace Kokkos

namespace GPU_HeiPa {
    inline void brute_force_bisect_gpu(const Graph &g,
                                       weight_t lmax_l,
                                       weight_t lmax_r,
                                       UnmanagedDevicePartition &map,
                                       UnmanagedDeviceWeight &bweights,
                                       DeviceExecutionSpace &exec_space) {
        ScopedTimer t{"recursive_bisection", "recursive_bisection_gpu", "brute_force_bisect_gpu"};

        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("BF_1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
                weight_t uw = g.uniform_vertex_weights ? 1 : g.weights(0);
                if (lmax_l >= lmax_r) {
                    map(0) = 0;
                    bweights(0) = uw;
                    bweights(1) = 0;
                } else {
                    map(0) = 1;
                    bweights(1) = uw;
                    bweights(0) = 0;
                }
            });
            exec_space.fence();
            return;
        }

        const vertex_t n = g.n;
        u64 num_combos = 1ULL << (n - 1); // fix vertex 0 to block 0

        BruteForceResult best; // Initialized by default constructor to max values

        Kokkos::parallel_reduce("BruteForce", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_combos), KOKKOS_LAMBDA(const u64 mask, BruteForceResult &local_best) {
            weight_t cut = 0;

            // Precompute partition for all vertices in the current mask for faster lookup
            // n is small (max 16), so a stack-allocated array is fine.
            int parts[16]; 
            parts[0] = 0; // Vertex 0 is fixed to block 0

            for (vertex_t u_idx = 1; u_idx < n; ++u_idx) {
                parts[u_idx] = (int) ((mask >> (u_idx - 1)) & 1ULL);
            }

            // Compute block weights.
            weight_t wl = g.uniform_vertex_weights ? 1 : g.weights(0); // Start with weight of vertex 0
            weight_t wr = 0;

            for (vertex_t u_idx = 1; u_idx < n; ++u_idx) {
                const weight_t uw = g.uniform_vertex_weights ? 1 : g.weights(u_idx);
                if (parts[u_idx] == 1) { // If u_idx is in partition 1
                    wr += uw;
                } else { // If u_idx is in partition 0
                    wl += uw;
                }
            }

            // Compute cut from edge list.
            for (u32 e = 0; e < g.m; ++e) {
                const vertex_t u_edge = g.edges_u(e);
                const vertex_t v_edge = g.edges_v(e);

                // If g.m stores both directions of an undirected graph,
                // count only one orientation.
                if (u_edge >= v_edge) continue;

                const int u_part = parts[u_edge];
                const int v_part = parts[v_edge];

                if (u_part != v_part) {
                    cut += g.uniform_edge_weights ? 1 : g.edges_w(e);
                }
            }

            const f64 dl = Kokkos::max(0.0, (f64) wl - (f64) lmax_l);
            const f64 dr = Kokkos::max(0.0, (f64) wr - (f64) lmax_r);
            const f64 penalty = dl * dl + dr * dr;

            BruteForceResult res;
            res.penalty = penalty;
            res.cut = cut;
            res.bitmask = mask;

            if (res < local_best) {
                local_best = res;
            }
        }, Kokkos::Min<BruteForceResult>(best));
        exec_space.fence();

        Kokkos::parallel_for("ApplyBF", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, n), KOKKOS_LAMBDA(const vertex_t u) {
            if (u == 0) map(u) = 0;
            else map(u) = (partition_t) ((best.bitmask >> (u - 1)) & 1);
        });
        exec_space.fence();

        Kokkos::parallel_for("UpdateBWeightsBF", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(const int) {
            weight_t wl = 0, wr = 0;
            for (vertex_t u = 0; u < n; ++u) {
                weight_t uw = g.uniform_vertex_weights ? 1 : g.weights(u);
                if (map(u) == 0) wl += uw;
                else wr += uw;
            }
            bweights(0) = wl;
            bweights(1) = wr;
        });
        exec_space.fence();
    }

    inline void extract_subgraphs_gpu(const Graph &g, const UnmanagedDevicePartition &b_part, KokkosMemoryStack &mem_stack, Graph &lg, Graph &rg, UnmanagedDeviceVertex &l_n2o, UnmanagedDeviceVertex &r_n2o, DeviceExecutionSpace &exec_space) {
        ScopedTimer t{"recursive_bisection", "recursive_bisection_gpu", "extract_subgraphs_gpu"};

        DeviceVertex rename("rename", g.n);
        vertex_t nl = 0, nr = 0;
        Kokkos::parallel_scan("RenameL", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &pref, bool final) {
            if (b_part(u) == 0) {
                if (final) rename(u) = pref;
                pref++;
            }
        }, nl);
        exec_space.fence();
        Kokkos::parallel_scan("RenameR", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u, vertex_t &pref, bool final) {
            if (b_part(u) == 1) {
                if (final) rename(u) = pref;
                pref++;
            }
        }, nr);
        exec_space.fence();

        UnmanagedDeviceU32 l_local_degrees((u32 *) get_chunk_front(mem_stack, sizeof(u32) * nl), nl);
        UnmanagedDeviceU32 r_local_degrees((u32 *) get_chunk_front(mem_stack, sizeof(u32) * nr), nr);
        Kokkos::deep_copy(exec_space, l_local_degrees, 0); // Initialize to 0
        Kokkos::deep_copy(exec_space, r_local_degrees, 0); // Initialize to 0
        exec_space.fence();

        // Calculate degrees for each vertex within its subgraph
        Kokkos::parallel_for("CountSubDegreesPass1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            if (b_part(u) == 0) {
                u32 current_sub_degree = 0;
                for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    if (b_part(g.edges_v(i)) == 0) {
                        current_sub_degree++;
                    }
                }
                l_local_degrees(rename(u)) = current_sub_degree;
            } else {
                u32 current_sub_degree = 0;
                for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    if (b_part(g.edges_v(i)) == 1) {
                        current_sub_degree++;
                    }
                }
                r_local_degrees(rename(u)) = current_sub_degree;
            }
        });
        exec_space.fence();

        weight_t wl = 0, wr = 0;
        vertex_t ml = 0, mr = 0;

        // Sum weights and degrees for left graph
        Kokkos::parallel_reduce("SumLeftGraphMetrics", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, nl), KOKKOS_LAMBDA(const vertex_t u, weight_t &wl_l, vertex_t &ml_l) {
            weight_t uw = g.uniform_vertex_weights ? 1 : g.weights(l_n2o(u)); // Need n2o_loc to map back to original u
            wl_l += uw;
            ml_l += l_local_degrees(u);
        }, wl, ml);
        exec_space.fence();

        // Sum weights and degrees for right graph
        Kokkos::parallel_reduce("SumRightGraphMetrics", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, nr), KOKKOS_LAMBDA(const vertex_t u, weight_t &wr_l, vertex_t &mr_l) {
            weight_t uw = g.uniform_vertex_weights ? 1 : g.weights(r_n2o(u)); // Need n2o_loc to map back to original u
            wr_l += uw;
            mr_l += r_local_degrees(u);
        }, wr, mr);
        exec_space.fence();

        // Safe allocation: ensure we always push 5 chunks to stack, even if n=0 or m=0
        lg = make_graph(std::max<vertex_t>(1, nl), std::max<vertex_t>(1, ml), wl, mem_stack);
        rg = make_graph(std::max<vertex_t>(1, nr), std::max<vertex_t>(1, mr), wr, mem_stack);
        lg.n = nl;
        lg.m = ml;
        rg.n = nr;
        rg.m = mr; // Restore correct logical sizes

        l_n2o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * std::max<vertex_t>(1, nl)), nl);
        r_n2o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * std::max<vertex_t>(1, nr)), nr);

        Kokkos::parallel_for("MapN2O", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            if (b_part(u) == 0) l_n2o(rename(u)) = u;
            else r_n2o(rename(u)) = u;
        });
        exec_space.fence();

        auto fill = [=](Graph &sub, int p, const UnmanagedDeviceU32 &local_degrees_view) {
            if (sub.n == 0) return;

            // Step 0: Ensure sub.neighborhood(0) is initialized
            Kokkos::parallel_for("Fill0", 1, KOKKOS_LAMBDA(int) { sub.neighborhood(0) = 0; });
            exec_space.fence();

            // Calculate prefix sum over local_degrees_view to get new neighborhood offsets
            Kokkos::parallel_scan("SubNeighborhoodScan", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sub.n + 1), KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (final) {
                    sub.neighborhood(i) = running;
                }
                if (i < sub.n) {
                    running += local_degrees_view(i);
                }
            });
            exec_space.fence();
            
            // Fill edges
            Kokkos::parallel_for("FillSubEdges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
                if (b_part(u) == p) {
                    u32 current_edge_idx = sub.neighborhood(rename(u));
                    for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                        vertex_t v_orig = g.edges_v(i);
                        if (b_part(v_orig) == p) {
                            sub.edges_v(current_edge_idx) = rename(v_orig);
                            sub.edges_w(current_edge_idx) = g.uniform_edge_weights ? 1 : g.edges_w(i);
                            sub.edges_u(current_edge_idx) = rename(u);
                            current_edge_idx++;
                        }
                    }
                }
            });
            exec_space.fence();

            Kokkos::parallel_for("Weights", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
                if (b_part(u) == p) sub.weights(rename(u)) = g.uniform_vertex_weights ? 1 : g.weights(u);
            });
            exec_space.fence();
        };
        fill(lg, 0, l_local_degrees);
        exec_space.fence();
        fill(rg, 1, r_local_degrees);
        exec_space.fence();

        pop_front(mem_stack); // r_local_degrees
        pop_front(mem_stack); // l_local_degrees
    }

    // --- Multilevel Bisection ---

    inline void multilevel_bisect_gpu(Graph &g, weight_t lmax_l, weight_t lmax_r, u32 seed, UnmanagedDevicePartition &final_p, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        std::vector<Graph> stack = {g};
        std::vector<Mapping> maps;
        Partition p = initialize_partition(stack.back().n, 2, lmax_l + lmax_r, mem_stack, exec_space);
        exec_space.fence();
        while (stack.back().n > 16) {
            {
                ScopedTimer t{"recursive_bisection", "recursive_bisection_gpu", "coarsening"};
                maps.emplace_back(two_hop_matcher_get_mapping<false, false>(stack.back(), p, lmax_l + lmax_r, mem_stack, exec_space));
                exec_space.fence();
            }
            if (maps.back().coarse_n >= stack.back().n * 0.99) {
                free_mapping(maps.back(), mem_stack);
                exec_space.fence();
                break;
            }
            {
                ScopedTimer t{"recursive_bisection", "recursive_bisection_gpu", "contraction"};
                stack.emplace_back(from_Graph_Mapping<false, false>(stack.back(), maps.back(), mem_stack, exec_space));
                exec_space.fence();
            }
        }

        brute_force_bisect_gpu(stack.back(), lmax_l, lmax_r, p.map, p.bweights, exec_space);
        exec_space.fence();

        while (!maps.empty()) {
            ScopedTimer t{"recursive_bisection", "recursive_bisection_gpu", "uncontraction"};

            uncontract(p, maps.back(), exec_space);
            exec_space.fence();

            free_graph(stack.back(), mem_stack);
            stack.pop_back();

            free_mapping(maps.back(), mem_stack);
            maps.pop_back();

            p.n = stack.back().n;
            exec_space.fence();
        }
        exec_space.fence();
        Kokkos::deep_copy(exec_space, final_p, p.map);
        exec_space.fence();
        free_partition(p, mem_stack);
    }

    // --- Recursive Driver ---

    inline void recursive_bisection_gpu(Graph &g, UnmanagedDeviceVertex &n2o, partition_t tk, std::vector<partition_t> hierarchy, weight_t global_w, partition_t global_k, f64 imbalance, u32 seed, UnmanagedDevicePartition &final_map, partition_t current_offset, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        if (g.n == 0) return;
        if (tk == 1) {
            Kokkos::parallel_for("FinalWrite", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) { final_map(n2o(u)) = current_offset; });
            exec_space.fence();
            return;
        }

        while (!hierarchy.empty() && hierarchy.back() == 1) {
            hierarchy.pop_back();
        }

        partition_t tk1, tk2;
        std::vector<partition_t> h1 = hierarchy;
        std::vector<partition_t> h2 = hierarchy;

        if (!hierarchy.empty()) {
            partition_t a_curr = hierarchy.back();
            partition_t tk1_h = (a_curr > 2) ? (1U << (u32) std::floor(std::log2((f64) a_curr - 1.0))) : 1;
            partition_t tk2_h = a_curr - tk1_h;
            partition_t unit = tk / a_curr;
            tk1 = tk1_h * unit;
            tk2 = tk2_h * unit;
            h1.back() = tk1_h;
            h2.back() = tk2_h;
        } else {
            tk1 = (tk > 2) ? (1U << (u32) std::floor(std::log2((f64) tk - 1.0))) : 1;
            tk2 = tk - tk1;
        }

        weight_t lmax_l = (weight_t) std::ceil((1.0 + imbalance) * (f64) global_w * (f64) tk1 / (f64) global_k);
        weight_t lmax_r = (weight_t) std::ceil((1.0 + imbalance) * (f64) global_w * (f64) tk2 / (f64) global_k);

        UnmanagedDevicePartition b_part((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * g.n), g.n);
        multilevel_bisect_gpu(g, lmax_l, lmax_r, seed, b_part, mem_stack, exec_space);

        Graph lg, rg;
        UnmanagedDeviceVertex ln2o_loc, rn2o_loc;
        extract_subgraphs_gpu(g, b_part, mem_stack, lg, rg, ln2o_loc, rn2o_loc, exec_space);

        UnmanagedDeviceVertex ln2o((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * std::max<vertex_t>(1, lg.n)), lg.n);
        UnmanagedDeviceVertex rn2o((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * std::max<vertex_t>(1, rg.n)), rg.n);
        {
            ScopedTimer t{"recursive_bisection", "recursive_bisection_gpu", "mapping"};
            if (lg.n > 0) {
                Kokkos::parallel_for("MapGlobalL", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, lg.n), KOKKOS_LAMBDA(const vertex_t u) { ln2o(u) = n2o(ln2o_loc(u)); });
            }
            if (rg.n > 0) {
                Kokkos::parallel_for("MapGlobalR", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, rg.n), KOKKOS_LAMBDA(const vertex_t u) { rn2o(u) = n2o(rn2o_loc(u)); });
            }
            exec_space.fence();
        }

        if (lg.n > 0) {
            recursive_bisection_gpu(lg, ln2o, tk1, h1, global_w, global_k, imbalance, seed, final_map, current_offset, mem_stack, exec_space);
        }
        if (rg.n > 0) {
            recursive_bisection_gpu(rg, rn2o, tk2, h2, global_w, global_k, imbalance, seed ^ 0xdeadbeef, final_map, current_offset + tk1, mem_stack, exec_space);
        }

        exec_space.fence(); 
        pop_front(mem_stack);
        pop_front(mem_stack); 
        pop_front(mem_stack);
        pop_front(mem_stack); 
        free_graph(rg, mem_stack);
        free_graph(lg, mem_stack);
        pop_front(mem_stack);
    }

    inline void gpu_initial_partition(const Graph &g, const std::vector<partition_t> &hierarchy, partition_t k, f64 imbalance, u64 seed, Partition &partition, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        ScopedTimer _t("initial_partitioning", "gpu_initial_partition", "total");

        Graph dev_g = make_graph(g.n, g.m, g.g_weight, mem_stack);
        Kokkos::deep_copy(exec_space, dev_g.neighborhood, g.neighborhood);
        Kokkos::deep_copy(exec_space, dev_g.edges_v, g.edges_v);
        Kokkos::deep_copy(exec_space, dev_g.edges_w, g.edges_w);
        Kokkos::deep_copy(exec_space, dev_g.edges_u, g.edges_u);
        Kokkos::deep_copy(exec_space, dev_g.weights, g.weights);
        dev_g.uniform_vertex_weights = g.uniform_vertex_weights;
        dev_g.uniform_edge_weights = g.uniform_edge_weights;

        UnmanagedDeviceVertex n2o((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.n), g.n);
        Kokkos::parallel_for("InitId", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) { n2o(u) = u; });
        exec_space.fence();

        recursive_bisection_gpu(dev_g, n2o, k, hierarchy, g.g_weight, k, imbalance, (u32) seed, partition.map, 0, mem_stack, exec_space);
        exec_space.fence();

        pop_front(mem_stack);
        free_graph(dev_g, mem_stack);
    }
}
#endif
