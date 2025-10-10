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

#ifndef GPU_HEIPA_TWO_HOP_MATCHING_H
#define GPU_HEIPA_TWO_HOP_MATCHING_H

#include <Kokkos_Sort.hpp>

#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"


namespace GPU_HeiPa {
    struct HashVertex {
        u32 hash;
        vertex_t v;

        KOKKOS_INLINE_FUNCTION
        bool operator<(const HashVertex &other) const {
            return (hash < other.hash) || (hash == other.hash && v < other.v);
        }
    };

    struct TwoHopMatcher {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t lmax = 0;
        f64 threshold = 0.75;

        DeviceF32 max_rating;
        DeviceVertex preferred_neighbor;

        DeviceU32 neighborhood_hash;
        Kokkos::View<HashVertex *> hash_vertex_array;
        DeviceU32 vertex_to_index; // index into hash_vertex_array
        DeviceU32 index_to_group_id; // for each vertex, which hash group it belongs
        DeviceU32 is_head;
        u32 n_hash_groups = 0; // number of hash groups
        DeviceU32 group_n_vertices; // sizes of each hash group
        DeviceU32 group_begin; // start index of each hash group

        u64 max_iterations_heavy = 3;
        u32 max_iterations_leaf = 3;
        u32 max_iterations_twins = 3;
        u32 max_iterations_relatives = 3;
        u32 max_iterations_zero_deg = 3;
    };

    inline TwoHopMatcher initialize_thm(const vertex_t t_n,
                                        const vertex_t t_m,
                                        const weight_t t_lmax) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "allocate");

        TwoHopMatcher thm;
        thm.n = t_n;
        thm.m = t_m;
        thm.lmax = t_lmax;

        thm.preferred_neighbor = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "preferred_neighbors"), t_n);
        thm.max_rating = DeviceF32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "scratch_rating"), t_n);

        thm.neighborhood_hash = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood_hash"), t_n);
        thm.hash_vertex_array = Kokkos::View<HashVertex *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hash_vertex_array"), t_n);
        thm.vertex_to_index = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_to_index"), t_n);
        thm.index_to_group_id = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "index_to_group_id"), t_n);
        thm.is_head = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "is_head"), t_n);
        thm.group_n_vertices = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "group_n_vertices"), t_n + 1);
        thm.group_begin = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "group_begin"), t_n + 2);

        return thm;
    }

    KOKKOS_INLINE_FUNCTION
    f32 edge_noise(vertex_t u, vertex_t v) {
        // make (u,v) order-independent
        uint32_t a = (u < v) ? u : v;
        uint32_t b = (u < v) ? v : u;

        // combine the pair (boost::hash_combine style)
        uint32_t x = a;
        x ^= b + 0x9e3779b9u + (x << 6) + (x >> 2);

        // Murmur3 32-bit finalizer (good avalanche, cheap)
        x ^= x >> 16;
        x *= 0x85ebca6bu;
        x ^= x >> 13;
        x *= 0xc2b2ae35u;
        x ^= x >> 16;

        // map to [0,1): use top 24 bits to match float mantissa width
        uint32_t mant = x >> 8; // top 24 bits
        f32 noise = (f32) mant * (1.0f / 16777216.0f);
        return noise * 0.000001f; // 1% amplitude
    }

    inline vertex_t n_matched_v(const DeviceVertex &matching,
                                const vertex_t n) {
        ScopedTimer _t("coarsening", "misc", "n_matched_v");

        vertex_t sum = 0;

        Kokkos::parallel_reduce("count_matches", n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &local_count) {
                                    vertex_t v = matching(u);
                                    vertex_t uu = matching(v);
                                    if (u != v && u == uu) { local_count += 1; }
                                },
                                sum);
        Kokkos::fence();

        return sum;
    }

    inline void build_hash_vertex_array(TwoHopMatcher &thm,
                                        const Graph &device_g) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "build_hash_vertex_array");

        Kokkos::deep_copy(thm.neighborhood_hash, 0);
        Kokkos::fence();

        Kokkos::parallel_for("hash", device_g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = device_g.edges_u(i);
            vertex_t v = device_g.edges_v(i);

            u32 h = v * 2654435761u; // 2654435761 = 2^32 / golden ratio

            Kokkos::atomic_fetch_add(&thm.neighborhood_hash(u), h);
        });
        Kokkos::fence();

        // build the hash array
        Kokkos::parallel_for("fill_array", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            thm.hash_vertex_array(u).hash = thm.neighborhood_hash(u);
            thm.hash_vertex_array(u).v = u;
        });
        Kokkos::fence();

        // sort the hash array
        Kokkos::sort(thm.hash_vertex_array);
        Kokkos::fence();

        // build index map
        Kokkos::parallel_for("index_map", device_g.n, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = thm.hash_vertex_array(i).v;
            thm.vertex_to_index(u) = i;
        });
        Kokkos::fence();

        // mark the start of each hash group
        Kokkos::parallel_scan("mark_and_map_groups", device_g.n, KOKKOS_LAMBDA(u32 i, u32 &running, bool final) {
                                  bool head = (i == 0 || thm.hash_vertex_array(i).hash != thm.hash_vertex_array(i - 1).hash);
                                  u32 inc = head ? 1 : 0;

                                  if (final) {
                                      thm.is_head(i) = inc;
                                      thm.index_to_group_id(i) = running;
                                  }

                                  running += inc;
                              },
                              thm.n_hash_groups
        );
        Kokkos::fence();
        thm.n_hash_groups += 1;

        // set the size of each group
        Kokkos::parallel_for("set_to_0", thm.n_hash_groups, KOKKOS_LAMBDA(u32 g) {
            thm.group_n_vertices(g) = 0;
        });
        Kokkos::fence();

        Kokkos::parallel_for("count", device_g.n, KOKKOS_LAMBDA(u32 i) {
            Kokkos::atomic_fetch_add(&thm.group_n_vertices(thm.index_to_group_id(i)), 1);
        });

        // set the start of each group
        Kokkos::parallel_scan("prefix_sum_groups", thm.n_hash_groups + 1, KOKKOS_LAMBDA(const u32 g, u32 &running, const bool final) {
                                  // for g < n_groups use the count, else (the last slot) use 0
                                  u32 cnt = (g < thm.n_hash_groups ? thm.group_n_vertices(g) : 0);
                                  if (final) {
                                      thm.group_begin(g) = running;
                                  }
                                  running += cnt;
                              }
        );
        Kokkos::fence();
    }

    inline Mapping determine_translation(const DeviceVertex &matching,
                                         const vertex_t n) {
        ScopedTimer _t("coarsening", "misc", "determine_mapping");

        Mapping mapping;
        mapping.old_n = n;
        mapping.mapping = DeviceVertex(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "mapping"), n);

        DeviceU32 needs_id(Kokkos::view_alloc(Kokkos::WithoutInitializing, "needs_id"), n);
        DeviceVertex assigned_id(Kokkos::view_alloc(Kokkos::WithoutInitializing, "assigned_id"), n);

        // Mark which vertices need IDs (one per unmatched vertex or per matched pair)
        Kokkos::parallel_for("mark_needs_id", n, KOKKOS_LAMBDA(const vertex_t u) {
            const vertex_t v = matching(u);
            // For a matched pair (u,v), only the smaller index assigns the ID.
            // For an unmatched vertex, v == u => it assigns an ID to itself.
            needs_id(u) = (v == u || u < v) ? 1u : 0u;
        });
        Kokkos::fence();

        // Count how many IDs will be assigned => coarse_n
        vertex_t coarse_n = 0;
        Kokkos::parallel_reduce("count_ids", n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                                    lsum += static_cast<vertex_t>(needs_id(u));
                                },
                                coarse_n
        );
        mapping.coarse_n = coarse_n;

        // Exclusive scan to assign compact IDs [0, coarse_n)
        Kokkos::parallel_scan("assign_ids", n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
                                  if (needs_id(u)) {
                                      if (final) assigned_id(u) = update; // write exclusive scan result
                                      update += 1;
                                  }
                              }
        );
        Kokkos::fence();

        // Build the translation mapping: old vertex -> new vertex
        Kokkos::parallel_for("assign_old_to_new", n, KOKKOS_LAMBDA(const vertex_t u) {
            const vertex_t v = matching(u);
            if (v == u || u < v) {
                // This thread is responsible for the pair (u,v) or the singleton u
                const vertex_t id = assigned_id(u);
                mapping.mapping(u) = id;
                mapping.mapping(v) = id; // v gets same new ID (safe: partner thread won't write)
            }
            // If u > v, its partner's thread (the smaller index) sets mapping.mapping(u)
        });
        Kokkos::fence();

        return mapping;
    }

    inline void heavy_edge_matching(TwoHopMatcher &thm,
                                    const Graph &device_g,
                                    DeviceVertex &matching,
                                    const Partition &partition) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "heavy_edge_matching");
        vertex_t n = device_g.n;

        for (u32 iteration = 0; iteration < thm.max_iterations_heavy; ++iteration) {
            if ((f64) n_matched_v(matching, n) >= thm.threshold * (f64) device_g.n) { return; }

            Kokkos::deep_copy(thm.max_rating, 0);
            Kokkos::parallel_for("set_matching", n, KOKKOS_LAMBDA(const vertex_t u) { thm.preferred_neighbor(u) = u; });
            Kokkos::fence();

            Kokkos::parallel_for("pick_neighbor", device_g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = device_g.edges_u(i);
                vertex_t v = device_g.edges_v(i);

                // if (partition.map(u) != partition.map(v)) { return; }
                if (matching(u) != u || matching(v) != v) { return; }
                if (device_g.weights(u) + device_g.weights(v) > thm.lmax) { return; }

                weight_t w = device_g.edges_w(i);
                // f32 rating = (f32) w;
                // f32 rating = (f32) (w) / (f32) (device_g.weights(u) + device_g.weights(v));
                f32 rating = (f32) (w * w) / (f32) (device_g.weights(u) * device_g.weights(v));
                // f32 rating = (f32) (1) / (f32) (device_g.weights(u) * device_g.weights(v));
                rating += edge_noise(u, v);

                Kokkos::atomic_max(&thm.max_rating(u), rating);
            });
            Kokkos::fence();

            Kokkos::parallel_for("pick_neighbor", device_g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = device_g.edges_u(i);
                vertex_t v = device_g.edges_v(i);

                // if (partition.map(u) != partition.map(v)) { return; }
                if (matching(u) != u || matching(v) != v) { return; }
                if (device_g.weights(u) + device_g.weights(v) > thm.lmax) { return; }

                weight_t w = device_g.edges_w(i);
                // f32 rating = (f32) w;
                // f32 rating = (f32) (w) / (f32) (device_g.weights(u) + device_g.weights(v));
                f32 rating = (f32) (w * w) / (f32) (device_g.weights(u) * device_g.weights(v));
                // f32 rating = (f32) (1) / (f32) (device_g.weights(u) * device_g.weights(v));
                rating += edge_noise(u, v);

                if (rating == thm.max_rating(u)) {
                    Kokkos::atomic_store(&thm.preferred_neighbor(u), v);
                }
            });
            Kokkos::fence();

            Kokkos::parallel_for("apply_matching", n, KOKKOS_LAMBDA(const vertex_t u) {
                if (matching(u) != u) { return; }

                vertex_t v = thm.preferred_neighbor(u);
                if (v == u || matching(v) != v) { return; }

                vertex_t pref_v = thm.preferred_neighbor(v);

                if (pref_v == u && u < v) {
                    matching(u) = v;
                    matching(v) = u;
                }
            });
            Kokkos::fence();
        }
    }

    inline void leaf_matching(TwoHopMatcher &thm,
                              const Graph &g,
                              DeviceVertex &matching,
                              const Partition &partition) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "leaf_matching");

        for (u32 iteration = 0; iteration < thm.max_iterations_leaf; ++iteration) {
            if ((f64) n_matched_v(matching, g.n) >= thm.threshold * (f64) g.n) { return; }

            Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                thm.preferred_neighbor(u) = u;
            });
            Kokkos::fence();

            Kokkos::parallel_for("pick_neighbor", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (matching(u) != u) { return; } // already matched
                if (g.neighborhood(u + 1) - g.neighborhood(u) != 1) { return; }

                // locate u’s bucket
                u32 idx = thm.vertex_to_index(u);
                u32 gid = thm.index_to_group_id(idx);
                u32 b = thm.group_begin(gid);
                u32 e = thm.group_begin(gid + 1);

                constexpr u32 LIMIT = 1024;
                if ((e - b) > LIMIT) {
                    const u32 half = LIMIT >> 1;
                    const u32 e_minus_LIMIT = e - LIMIT;
                    u32 start = idx > half ? (idx - half) : b;
                    if (start < b) start = b;
                    if (start > e_minus_LIMIT) start = e_minus_LIMIT;
                    b = start;
                    e = start + LIMIT;
                }

                f32 best_rating = -max_sentinel<f32>();
                vertex_t best_v = u;
                for (u32 i = b; i < e; ++i) {
                    vertex_t v = thm.hash_vertex_array(i).v;
                    if (v == u) { continue ; }
                    if (matching(v) != v) { continue ; }
                    // if (p_manager.partition(u) != p_manager.partition(v)) { continue ; }
                    if (g.weights(u) + g.weights(v) > thm.lmax) { continue ; }

                    f32 rating = (f32) (1) / (f32) (g.weights(u) * g.weights(v));
                    rating += edge_noise(u, v);
                    if (rating > best_rating) {
                        best_rating = rating;
                        best_v = v;
                    }
                }

                thm.preferred_neighbor(u) = best_v;
            });
            Kokkos::fence();

            Kokkos::parallel_for("apply_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (u == thm.preferred_neighbor(u)) { return; }

                vertex_t v = thm.preferred_neighbor(u);
                vertex_t preferred_v = thm.preferred_neighbor(v);

                if (matching(u) != u || matching(v) != v) { return; }

                if (u == preferred_v && u < v) {
                    matching(u) = v;
                    matching(v) = u;
                }
            });
            Kokkos::fence();
        }
    }

    inline void twin_matching(TwoHopMatcher &thm,
                              const Graph &g,
                              DeviceVertex &matching,
                              const Partition &partition) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "twin_matching");

        for (u32 iteration = 0; iteration < thm.max_iterations_twins; ++iteration) {
            if ((f64) n_matched_v(matching, g.n) >= thm.threshold * (f64) g.n) { return; }

            Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                thm.preferred_neighbor(u) = u;
            });
            Kokkos::fence();

            Kokkos::parallel_for("pick_neighbor", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (matching(u) != u) { return; } // already matched

                // locate u’s bucket
                u32 idx = thm.vertex_to_index(u);
                u32 gid = thm.index_to_group_id(idx);
                u32 b = thm.group_begin(gid);
                u32 e = thm.group_begin(gid + 1);

                constexpr u32 LIMIT = 1024;
                if ((e - b) > LIMIT) {
                    const u32 half = LIMIT >> 1;
                    const u32 e_minus_LIMIT = e - LIMIT;
                    u32 start = idx > half ? (idx - half) : b;
                    if (start < b) start = b;
                    if (start > e_minus_LIMIT) start = e_minus_LIMIT;
                    b = start;
                    e = start + LIMIT;
                }

                f32 best_rating = -max_sentinel<f32>();
                vertex_t best_v = u;
                for (u32 i = b; i < e; ++i) {
                    vertex_t v = thm.hash_vertex_array(i).v;
                    if (v == u) { continue ; }
                    if (matching(v) != v) { continue ; }
                    // if (p_manager.partition(u) != p_manager.partition(v)) { continue ; }
                    if (g.weights(u) + g.weights(v) > thm.lmax) { continue ; }

                    f32 rating = (f32) (1) / (f32) (g.weights(u) * g.weights(v));
                    rating += edge_noise(u, v);
                    if (rating > best_rating) {
                        best_rating = rating;
                        best_v = v;
                    }
                }

                thm.preferred_neighbor(u) = best_v;
            });
            Kokkos::fence();

            Kokkos::parallel_for("apply_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (u == thm.preferred_neighbor(u)) { return; }

                vertex_t v = thm.preferred_neighbor(u);
                vertex_t preferred_v = thm.preferred_neighbor(v);

                if (matching(u) != u || matching(v) != v) { return; }

                if (u == preferred_v && u < v) {
                    matching(u) = v;
                    matching(v) = u;
                }
            });
            Kokkos::fence();
        }
    }

    inline void relative_matching(TwoHopMatcher &thm,
                                  const Graph &g,
                                  DeviceVertex &matching,
                                  const Partition &partition) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "relative_matching");

        for (u32 iteration = 0; iteration < thm.max_iterations_relatives; ++iteration) {
            if ((f64) n_matched_v(matching, g.n) >= thm.threshold * (f64) g.n) { return; }

            Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                thm.preferred_neighbor(u) = u;
            });
            Kokkos::fence();

            Kokkos::parallel_for("pick_neighbor", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (matching(u) != u) { return; } // ignore already matched

                vertex_t best_v = u;
                f32 best_rating = 0;

                for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    vertex_t mid_v = g.edges_v(i);
                    weight_t mid_e_w = g.edges_w(i);
                    vertex_t mid_deg = g.neighborhood(mid_v + 1) - g.neighborhood(mid_v);
                    if (mid_deg > 10) { continue; } // avoid matchmaker with high degree

                    for (u32 j = g.neighborhood(mid_v); j < g.neighborhood(mid_v + 1); ++j) {
                        vertex_t v = g.edges_v(j);
                        weight_t v_e_w = g.edges_w(j);

                        if (u == v) { continue; }
                        if (matching(v) != v) { continue; } // ignore already matched
                        // if (p_manager.partition(u) != p_manager.partition(v)) { continue ; }
                        if (g.weights(u) + g.weights(v) > thm.lmax) { continue; } // resulting weight to large

                        f32 rating = (f32) ((mid_e_w + v_e_w) * (mid_e_w + v_e_w)) / (f32) (g.weights(u) * g.weights(v));
                        rating += edge_noise(u, v);

                        if (rating > best_rating) {
                            best_v = v;
                            best_rating = rating;
                        }
                    }
                }

                thm.preferred_neighbor(u) = best_v;
            });
            Kokkos::fence();

            Kokkos::parallel_for("apply_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t v = thm.preferred_neighbor(u);
                vertex_t preferred_v = thm.preferred_neighbor(v);

                if (matching(u) != u || matching(v) != v) { return; }

                if (u == preferred_v && u < v) {
                    matching(u) = v;
                    matching(v) = u;
                }
            });
            Kokkos::fence();
        }
    }

    inline void zero_deg_matching(TwoHopMatcher &thm,
                                  const Graph &g,
                                  DeviceVertex &matching,
                                  const Partition &partition) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "zero_deg_matching");

        for (u32 iteration = 0; iteration < thm.max_iterations_zero_deg; ++iteration) {
            if ((f64) n_matched_v(matching, g.n) >= thm.threshold * (f64) g.n) { return; }

            Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                thm.preferred_neighbor(u) = u;
            });
            Kokkos::fence();

            Kokkos::parallel_for("pick_neighbor", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (matching(u) != u) { return; } // ignore already matched

                vertex_t best_v = u;
                f32 best_rating = 0;

                for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    vertex_t mid_v = g.edges_v(i);
                    weight_t mid_e_w = g.edges_w(i);
                    vertex_t mid_deg = g.neighborhood(mid_v + 1) - g.neighborhood(mid_v);
                    if (mid_deg > 10) { continue; } // avoid matchmaker with high degree

                    for (u32 j = g.neighborhood(mid_v); j < g.neighborhood(mid_v + 1); ++j) {
                        vertex_t v = g.edges_v(j);
                        weight_t v_e_w = g.edges_w(j);

                        if (u == v) { continue; }
                        if (matching(v) != v) { continue; } // ignore already matched
                        // if (p_manager.partition(u) != p_manager.partition(v)) { continue ; }
                        if (g.weights(u) + g.weights(v) > thm.lmax) { continue; } // resulting weight to large

                        f32 rating = (f32) ((mid_e_w + v_e_w) * (mid_e_w + v_e_w)) / (f32) (g.weights(u) * g.weights(v));
                        rating += edge_noise(u, v);

                        if (rating > best_rating) {
                            best_v = v;
                            best_rating = rating;
                        }
                    }
                }

                thm.preferred_neighbor(u) = best_v;
            });
            Kokkos::fence();

            Kokkos::parallel_for("apply_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t v = thm.preferred_neighbor(u);
                vertex_t preferred_v = thm.preferred_neighbor(v);

                if (matching(u) != u || matching(v) != v) { return; }

                if (u == preferred_v && u < v) {
                    matching(u) = v;
                    matching(v) = u;
                }
            });
            Kokkos::fence();
        }
    }

    inline Mapping two_hop_matcher_get_mapping(TwoHopMatcher &thm,
                                               const Graph &g,
                                               const Partition &partition) {
        ScopedTimer _t_allocate("coarsening", "misc", "allocate_matching");

        DeviceVertex matching = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "preferred_neighbors"), g.n);
        Kokkos::parallel_for("set_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) { matching(u) = u; });
        Kokkos::fence();

        _t_allocate.stop();

        heavy_edge_matching(thm, g, matching, partition);
        vertex_t n_matched = n_matched_v(matching, g.n);

        if ((f64) n_matched < thm.threshold * (f64) g.n) {
            build_hash_vertex_array(thm, g);

            leaf_matching(thm, g, matching, partition);
            n_matched = n_matched_v(matching, g.n);
        }

        if ((f64) n_matched < thm.threshold * (f64) g.n) {
            twin_matching(thm, g, matching, partition);
            n_matched = n_matched_v(matching, g.n);
        }

        if ((f64) n_matched < thm.threshold * (f64) g.n) {
            relative_matching(thm, g, matching, partition);
        }

        if ((f64) n_matched < thm.threshold * (f64) g.n) {
            // zero_deg_matching(thm, g, matching, partition);
        }

        return determine_translation(matching, g.n);
    }
}

#endif //GPU_HEIPA_TWO_HOP_MATCHING_H
