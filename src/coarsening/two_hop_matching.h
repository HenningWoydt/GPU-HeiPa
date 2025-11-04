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

        UnmanagedDeviceF32 max_rating;
        UnmanagedDeviceVertex preferred_neighbor;
        UnmanagedDeviceU64 rating_vertex;

        Kokkos::View<HashVertex *, Kokkos::MemoryTraits<Kokkos::Unmanaged> > hash_vertex_array;
        UnmanagedDeviceU32 vertex_to_index;   // index into hash_vertex_array
        UnmanagedDeviceU32 index_to_group_id; // for each vertex, which hash group it belongs
        UnmanagedDeviceU32 is_head;
        UnmanagedDeviceU32 group_n_vertices; // sizes of each hash group
        UnmanagedDeviceU32 group_begin;      // start index of each hash group

        u64 max_iterations_heavy = 3;
        u32 max_iterations_leaf = 3;
        u32 max_iterations_twins = 3;
        u32 max_iterations_relatives = 3;
        u32 max_iterations_zero_deg = 3;
    };

    inline TwoHopMatcher initialize_thm(const vertex_t t_n,
                                        const vertex_t t_m,
                                        const weight_t t_lmax,
                                        KokkosMemoryStack &small_mem_stack) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "allocate");

        TwoHopMatcher thm;
        thm.n = t_n;
        thm.m = t_m;
        thm.lmax = t_lmax;

        auto *preferred_neighbor_ptr = (vertex_t *) get_chunk(small_mem_stack, sizeof(vertex_t) * t_n);
        auto *max_rating_ptr = (f32 *) get_chunk(small_mem_stack, sizeof(f32) * t_n);
        auto *rating_vertex_ptr = (u64 *) get_chunk(small_mem_stack, sizeof(u64) * t_n);
        thm.preferred_neighbor = UnmanagedDeviceVertex(preferred_neighbor_ptr, t_n);
        thm.max_rating = UnmanagedDeviceF32(max_rating_ptr, t_n);
        thm.rating_vertex = UnmanagedDeviceU64(rating_vertex_ptr, t_n);

        auto *neighborhood_hash_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * t_n);
        auto *hash_vertex_array_ptr = (HashVertex *) get_chunk(small_mem_stack, sizeof(HashVertex) * t_n);
        auto *vertex_to_index_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * t_n);
        auto *index_to_group_id_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * t_n);
        auto *is_head_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * t_n);
        auto *group_n_vertices_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * t_n);
        auto *group_begin_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * (t_n + 2));
        thm.hash_vertex_array = Kokkos::View<HashVertex *, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(hash_vertex_array_ptr, t_n);
        thm.vertex_to_index = UnmanagedDeviceU32(vertex_to_index_ptr, t_n);
        thm.index_to_group_id = UnmanagedDeviceU32(index_to_group_id_ptr, t_n);
        thm.is_head = UnmanagedDeviceU32(is_head_ptr, t_n);
        thm.group_n_vertices = UnmanagedDeviceU32(group_n_vertices_ptr, t_n);
        thm.group_begin = UnmanagedDeviceU32(group_begin_ptr, t_n);

        return thm;
    }

    inline void free_thm(TwoHopMatcher &thm,
                         KokkosMemoryStack &small_mem_stack) {
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
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
        // hash each neighborhood
        {
            ScopedTimer _t("coarsening", "build_hash_vertex_array", "hash");
            Kokkos::parallel_for("hash_optimized", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                u32 hash_sum = 0;
                for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                    vertex_t v = device_g.edges_v(i);
                    hash_sum ^= (v << 1) | (v >> 31);
                }
                thm.hash_vertex_array(u).hash = hash_sum;
                thm.hash_vertex_array(u).v = u;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // sort the hash array
        {
            ScopedTimer _t("coarsening", "build_hash_vertex_array", "sort");
            Kokkos::sort(thm.hash_vertex_array);
            KOKKOS_PROFILE_FENCE();
        }

        // 1) One scan to build index map, head flags, group ids, and group begins.
        u32 n_groups = 0; {
            ScopedTimer _t("coarsening", "build_hash_vertex_array", "mark_map_scan");
            Kokkos::parallel_scan("mark_map_scan", device_g.n, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                                      const auto cur = thm.hash_vertex_array(i);
                                      const bool head = (i == 0) || (cur.hash != thm.hash_vertex_array(i - 1).hash);
                                      const u32 inc = head ? 1u : 0u;

                                      if (final) {
                                          // index map
                                          thm.vertex_to_index(cur.v) = i;

                                          // head flag & group id
                                          thm.is_head(i) = inc;
                                          thm.index_to_group_id(i) = running;

                                          // write group begin at head positions
                                          if (head) thm.group_begin(running) = i;
                                      }

                                      running += inc;
                                  },
                                  n_groups // host scalar receives total number of groups
            );
            KOKKOS_PROFILE_FENCE(); // only if you time this phase
        }

        // 2) Set sentinel begin for the "end" (begin[n_groups] = N)
        {
            Kokkos::parallel_for("set_last_begin", 1u, KOKKOS_LAMBDA(const u32) {
                thm.group_begin(n_groups) = device_g.n;
            });
            KOKKOS_PROFILE_FENCE(); // only for timing separation
        }

        // 3) Compute group sizes from consecutive begins (no zeroing or atomics needed)
        {
            ScopedTimer _t("coarsening", "build_hash_vertex_array", "sizes_from_begins");
            Kokkos::parallel_for("sizes_from_begins", n_groups, KOKKOS_LAMBDA(const u32 g) {
                thm.group_n_vertices(g) = thm.group_begin(g + 1) - thm.group_begin(g);
            });
            // No fence needed for correctness unless host reads now
        }

        // Analyze bucket distribution
        /*
        {
            Kokkos::fence();
            auto host_group_sizes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                Kokkos::subview(thm.group_n_vertices, Kokkos::make_pair(0u, n_groups)));
            
            u32 min_size = UINT32_MAX, max_size = 0, total_vertices = 0;
            u32 large_buckets = 0, empty_buckets = 0;
            
            for (u32 i = 0; i < n_groups; ++i) {
                u32 size = host_group_sizes(i);
                min_size = std::min(min_size, size);
                max_size = std::max(max_size, size);
                total_vertices += size;
                if (size > 1024) large_buckets++;
                if (size == 0) empty_buckets++;
            }
            
            f64 avg_size = (f64)total_vertices / n_groups;
            
            std::cout << "Hash Bucket Analysis:" << std::endl;
            std::cout << "  Total groups: " << n_groups << std::endl;
            std::cout << "  Total vertices: " << total_vertices << std::endl;
            std::cout << "  Min bucket size: " << min_size << std::endl;
            std::cout << "  Max bucket size: " << max_size << std::endl;
            std::cout << "  Avg bucket size: " << avg_size << std::endl;
            std::cout << "  Large buckets (>1024): " << large_buckets << std::endl;
            std::cout << "  Empty buckets: " << empty_buckets << std::endl;
        }
        */
    }

    inline Mapping determine_mapping(const DeviceVertex &matching,
                                     const vertex_t n,
                                     KokkosMemoryStack &mem_stack,
                                     KokkosMemoryStack &small_mem_stack) {
        Mapping mapping;
        //
        {
            ScopedTimer _t("coarsening", "determine_mapping", "allocate_mapping");
            mapping = initialize_mapping(n, 0, mem_stack);
        }

        UnmanagedDeviceU32 needs_id;
        UnmanagedDeviceVertex assigned_id;
        //
        {
            ScopedTimer _t("coarsening", "determine_mapping", "allocate");
            auto *needs_id_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * n);
            auto *assigned_id_ptr = (vertex_t *) get_chunk(small_mem_stack, sizeof(vertex_t) * n);
            needs_id = UnmanagedDeviceU32(needs_id_ptr, n);
            assigned_id = UnmanagedDeviceVertex(assigned_id_ptr, n);
        }
        //
        {
            ScopedTimer _t("coarsening", "determine_mapping", "mark_needs_id");
            // Mark which vertices need IDs (one per unmatched vertex or per matched pair)
            Kokkos::parallel_for("mark_needs_id", n, KOKKOS_LAMBDA(const vertex_t u) {
                const vertex_t v = matching(u);
                // For a matched pair (u,v), only the smaller index assigns the ID.
                // For an unmatched vertex, v == u => it assigns an ID to itself.
                needs_id(u) = (v == u || u < v) ? 1u : 0u;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // Count how many IDs will be assigned => coarse_n
        vertex_t coarse_n = 0;
        //
        {
            ScopedTimer _t("coarsening", "determine_mapping", "count_ids");
            Kokkos::parallel_reduce("count_ids", n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                                        lsum += static_cast<vertex_t>(needs_id(u));
                                    },
                                    coarse_n
            );
            mapping.coarse_n = coarse_n;
        }
        //
        {
            ScopedTimer _t("coarsening", "determine_mapping", "assign_ids");
            // Exclusive scan to assign compact IDs [0, coarse_n)
            Kokkos::parallel_scan("assign_ids", n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
                                      if (needs_id(u)) {
                                          if (final) assigned_id(u) = update; // write exclusive scan result
                                          update += 1;
                                      }
                                  }
            );
            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("coarsening", "determine_mapping", "assign_old_to_new");
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
            KOKKOS_PROFILE_FENCE();
        }

        pop(small_mem_stack);
        pop(small_mem_stack);

        return mapping;
    }

    inline void heavy_edge_matching(TwoHopMatcher &thm,
                                    const Graph &g,
                                    UnmanagedDeviceVertex &matching,
                                    const Partition &partition) {
        for (u32 iteration = 0; iteration < thm.max_iterations_heavy; ++iteration) {
            vertex_t n_matched = 0;
            //
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "n_matched_v");
                n_matched = n_matched_v(matching, g.n);
            }

            if ((f64) n_matched >= thm.threshold * (f64) g.n) { return; }
            //
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "reset");
                Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                    thm.rating_vertex(u) = pack_f32_vertex(0.0f, u);
                });
                KOKKOS_PROFILE_FENCE();
            }
            //
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "pick_neighbor");

                using RP = Kokkos::RangePolicy<
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::Schedule<Kokkos::Static>,
                    Kokkos::IndexType<u32>
                >;
                RP pol(0, g.m);
                pol.set_chunk_size(1024);

                Kokkos::parallel_for("pick_neighbor", pol, KOKKOS_LAMBDA(const u32 i) {
                    vertex_t u = g.edges_u(i);
                    vertex_t v = g.edges_v(i);

                    // Early exit optimizations - check cheapest conditions first
                    if (matching(u) != u || matching(v) != v) { return; }
                    if (partition.map(u) != partition.map(v)) { return; }

                    // Cache weight lookups to reduce memory access
                    weight_t u_weight = g.weights(u);
                    weight_t v_weight = g.weights(v);
                    if (u_weight + v_weight > thm.lmax / 2) { return; }

                    weight_t w = g.edges_w(i);

                    // Optimized rating calculation - avoid expensive division when possible
                    f32 weight_product = (f32) (u_weight * v_weight);
                    f32 rating = (f32) (w * w) / weight_product;
                    rating += edge_noise(u, v);

                    Kokkos::atomic_max(&thm.rating_vertex(u), pack_f32_vertex(rating, v));
                });
                KOKKOS_PROFILE_FENCE();
            }
            //
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "apply_matching");
                Kokkos::parallel_for("apply_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                    if (matching(u) != u) { return; }

                    vertex_t v = unpack_vertex(thm.rating_vertex(u));
                    if (v >= g.n || v == u || matching(v) != v) { return; }

                    vertex_t pref_v = unpack_vertex(thm.rating_vertex(v));

                    if (pref_v == u && u < v) {
                        matching(u) = v;
                        matching(v) = u;
                    }
                });
                KOKKOS_PROFILE_FENCE();
            }
        }
    }

    inline void leaf_matching(TwoHopMatcher &thm,
                              const Graph &g,
                              UnmanagedDeviceVertex &matching,
                              const Partition &partition) {
        for (u32 iteration = 0; iteration < thm.max_iterations_leaf; ++iteration) {
            vertex_t n_matched = 0;
            //
            {
                ScopedTimer _t("coarsening", "thm_leaf_matching", "n_matched_v");
                n_matched = n_matched_v(matching, g.n);
            }

            if ((f64) n_matched >= thm.threshold * (f64) g.n) { return; }
            //
            {
                ScopedTimer _t("coarsening", "thm_leaf_matching", "pick_neighbor");
                Kokkos::parallel_for("pick_neighbor", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                    if (matching(u) != u || g.neighborhood(u + 1) - g.neighborhood(u) != 1) {
                        // already matched or not degree 1
                        thm.preferred_neighbor(u) = u;
                        return;
                    }

                    // Position of u in group
                    const u32 idx = thm.vertex_to_index(u);
                    const u32 gid = thm.index_to_group_id(idx);
                    const u32 b = thm.group_begin(gid);
                    const u32 e = thm.group_begin(gid + 1);

                    // Default: no match
                    vertex_t best = u;

                    // Pick direction based on parity (even → -1, odd → +1)
                    const bool is_even = ((idx - b) & 1u) == 0u;
                    const int dir = is_even ? -1 : +1;
                    const u32 ni = idx + dir;

                    if (ni >= b && ni < e) {
                        // neighbor exists
                        const vertex_t v = thm.hash_vertex_array(ni).v;
                        // Compact eligibility check
                        if (matching(v) == v && partition.map(u) == partition.map(v) && g.weights(u) + g.weights(v) <= thm.lmax / 2) {
                            best = v;
                        }
                    }

                    thm.preferred_neighbor(u) = best;
                });
                KOKKOS_PROFILE_FENCE();
            }
            //
            {
                ScopedTimer _t("coarsening", "thm_leaf_matching", "apply_matching");
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
                KOKKOS_PROFILE_FENCE();
            }
        }
    }

    inline void twin_matching(TwoHopMatcher &thm,
                              const Graph &g,
                              UnmanagedDeviceVertex &matching,
                              const Partition &partition) {
        for (u32 iteration = 0; iteration < thm.max_iterations_twins; ++iteration) {
            vertex_t n_matched = 0;
            //
            {
                ScopedTimer _t("coarsening", "thm_twin_matching", "n_matched_v");
                n_matched = n_matched_v(matching, g.n);
            }

            if ((f64) n_matched >= thm.threshold * (f64) g.n) { return; }
            //
            {
                ScopedTimer _t("coarsening", "thm_twin_matching", "pick_neighbor");

                Kokkos::parallel_for("pick_neighbor", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                    if (matching(u) != u) {
                        thm.preferred_neighbor(u) = u;
                        return;
                    }

                    // Position of u in group
                    const u32 idx = thm.vertex_to_index(u);
                    const u32 gid = thm.index_to_group_id(idx);
                    const u32 b = thm.group_begin(gid);
                    const u32 e = thm.group_begin(gid + 1);

                    // Default: no match
                    vertex_t best = u;

                    // Pick direction based on parity (even → -1, odd → +1)
                    const bool is_even = ((idx - b) & 1u) == 0u;
                    const int dir = is_even ? -1 : +1;
                    const u32 ni = idx + dir;

                    if (ni >= b && ni < e) {
                        // neighbor exists
                        const vertex_t v = thm.hash_vertex_array(ni).v;
                        // Compact eligibility check
                        if (matching(v) == v && partition.map(u) == partition.map(v) && g.weights(u) + g.weights(v) <= thm.lmax / 2) {
                            best = v;
                        }
                    }

                    thm.preferred_neighbor(u) = best;
                });

                KOKKOS_PROFILE_FENCE();
            }
            //
            {
                ScopedTimer _t("coarsening", "thm_twin_matching", "apply_matching");
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
                KOKKOS_PROFILE_FENCE();
            }
        }
    }

    inline void relative_matching(TwoHopMatcher &thm,
                                  const Graph &g,
                                  UnmanagedDeviceVertex &matching,
                                  const Partition &partition) {
        for (u32 iteration = 0; iteration < thm.max_iterations_relatives; ++iteration) {
            vertex_t n_matched = 0;
            //
            {
                ScopedTimer _t("coarsening", "thm_relative_matching", "n_matched_v");
                n_matched = n_matched_v(matching, g.n);
            }

            if ((f64) n_matched >= thm.threshold * (f64) g.n) { return; }
            //
            {
                ScopedTimer _t("coarsening", "thm_relative_matching", "reset");
                Kokkos::parallel_for("reset", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                    thm.preferred_neighbor(u) = u;
                });
                KOKKOS_PROFILE_FENCE();
            }
            //
            {
                ScopedTimer _t("coarsening", "thm_relative_matching", "pick_neighbor");
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
                            if (partition.map(u) != partition.map(v)) { continue ; }
                            if (g.weights(u) + g.weights(v) > thm.lmax / 2) { continue; } // resulting weight to large

                            f32 rating = (f32) ((mid_e_w + v_e_w) * (mid_e_w + v_e_w)) / (f32) (g.weights(u) * g.weights(v));
                            rating += edge_noise(u, v);

                            if (rating > best_rating || (rating == best_rating && v < best_v)) {
                                best_v = v;
                                best_rating = rating;
                            }
                        }
                    }

                    thm.preferred_neighbor(u) = best_v;
                });
                KOKKOS_PROFILE_FENCE();
            }
            //
            {
                ScopedTimer _t("coarsening", "thm_relative_matching", "apply_matching");
                Kokkos::parallel_for("apply_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                    vertex_t v = thm.preferred_neighbor(u);
                    vertex_t preferred_v = thm.preferred_neighbor(v);

                    if (matching(u) != u || matching(v) != v) { return; }

                    if (u == preferred_v && u < v) {
                        matching(u) = v;
                        matching(v) = u;
                    }
                });
                KOKKOS_PROFILE_FENCE();
            }
        }
    }

    inline Mapping two_hop_matcher_get_mapping(const Graph &g,
                                               const Partition &partition,
                                               const weight_t &lmax,
                                               KokkosMemoryStack &mem_stack,
                                               KokkosMemoryStack &small_mem_stack) {
        assert_is_empty(small_mem_stack);

        TwoHopMatcher thm = initialize_thm(g.n, g.m, lmax, small_mem_stack);

        UnmanagedDeviceVertex matching;
        //
        {
            ScopedTimer _t("coarsening", "TwoHopMatcher", "allocate_matching");
            auto *matching_ptr = (vertex_t *) get_chunk(small_mem_stack, sizeof(vertex_t) * g.n);
            matching = UnmanagedDeviceVertex(matching_ptr, g.n);
            Kokkos::parallel_for("set_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                matching(u) = u;
            });
            KOKKOS_PROFILE_FENCE();
        }

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

        pop(small_mem_stack); // pop the matching vec
        free_thm(thm, small_mem_stack);

        Mapping mapping = determine_mapping(matching, g.n, mem_stack, small_mem_stack);

        assert_is_empty(small_mem_stack);
        return mapping;
    }
}

#endif //GPU_HEIPA_TWO_HOP_MATCHING_H
