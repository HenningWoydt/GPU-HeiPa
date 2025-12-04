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
    struct TwoHopMatcher {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t lmax = 0;
        f64 threshold = 0.75;

        vertex_t n_matched = 0;
        UnmanagedDeviceF32 max_rating;
        UnmanagedDeviceVertex preferred_neighbor;
    };

    inline TwoHopMatcher initialize_thm(const vertex_t t_n,
                                        const vertex_t t_m,
                                        const weight_t t_lmax,
                                        KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "allocate");

        TwoHopMatcher thm;
        thm.n = t_n;
        thm.m = t_m;
        thm.lmax = t_lmax;

        thm.n_matched = 0;
        auto *preferred_neighbor_ptr = (vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * t_n);
        auto *max_rating_ptr = (f32 *) get_chunk_back(mem_stack, sizeof(f32) * t_n);
        thm.preferred_neighbor = UnmanagedDeviceVertex(preferred_neighbor_ptr, t_n);
        thm.max_rating = UnmanagedDeviceF32(max_rating_ptr, t_n);

        return thm;
    }

    inline void free_thm(TwoHopMatcher &thm,
                         KokkosMemoryStack &mem_stack) {
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    KOKKOS_INLINE_FUNCTION
    f32 edge_noise(vertex_t u, vertex_t v, u32 round) {
        // make (u,v) order-independent
        uint32_t a = (u < v) ? u : v;
        uint32_t b = (u < v) ? v : u;

        // combine the pair (boost::hash_combine style)
        uint32_t x = a;
        x ^= b + 0x9e3779b9u + (x << 6) + (x >> 2) + round;

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

    inline Mapping determine_mapping(const DeviceVertex &matching,
                                     const vertex_t n,
                                     KokkosMemoryStack &mem_stack) {
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
            needs_id = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * n), n);
            assigned_id = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n), n);
        }
        //
        {
            ScopedTimer _t("coarsening", "determine_mapping", "mark_and_count");
            Kokkos::parallel_reduce("mark_and_count", n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                                        vertex_t v = matching(u);
                                        u32 need = v <= u;
                                        needs_id(u) = need; // side-effect: mark array
                                        lsum += need;       // accumulate count
                                    },
                                    mapping.coarse_n
            );
            KOKKOS_PROFILE_FENCE();
        }
        // Exclusive scan to assign compact IDs [0, coarse_n)
        {
            ScopedTimer _t("coarsening", "determine_mapping", "assign_ids");
            Kokkos::parallel_scan("assign_ids", n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
                                      if (needs_id(u)) {
                                          if (final) assigned_id(u) = update; // write exclusive scan result
                                          update += 1;
                                      }
                                  }
            );
            KOKKOS_PROFILE_FENCE();
        }
        // Build the translation mapping: old vertex -> new vertex
        {
            ScopedTimer _t("coarsening", "determine_mapping", "assign_old_to_new");
            Kokkos::parallel_for("assign_old_to_new", n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t v = matching(u);
                mapping.mapping(u) = v <= u ? assigned_id(u) : assigned_id(v);
            });
            KOKKOS_PROFILE_FENCE();
        }

        Kokkos::parallel_for("assign_old_to_new", n, KOKKOS_LAMBDA(const vertex_t u) {
            MY_KOKKOS_ASSERT(mapping.mapping(u) < mapping.coarse_n);
        });

        pop_back(mem_stack);
        pop_back(mem_stack);

        return mapping;
    }


    inline void heavy_edge_matching(TwoHopMatcher &thm,
                                    const Graph &g,
                                    UnmanagedDeviceVertex &matching,
                                    const Partition &partition,
                                    KokkosMemoryStack &mem_stack) {
        vertex_t n_unmapped = thm.n - thm.n_matched;
        auto *unmapped_v_ptr = (vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n_unmapped);
        UnmanagedDeviceVertex unmapped_v = UnmanagedDeviceVertex(unmapped_v_ptr, n_unmapped);

        vertex_t next_n_unmapped = thm.n - thm.n_matched;
        auto *next_unmapped_v_ptr = (vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * next_n_unmapped);
        UnmanagedDeviceVertex next_unmapped_v = UnmanagedDeviceVertex(next_unmapped_v_ptr, next_n_unmapped);

        // init unmapped vertices
        {
            ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "init_unmapped");
            Kokkos::parallel_for("init_unmapped", n_unmapped, KOKKOS_LAMBDA(const vertex_t u) {
                unmapped_v(u) = u;
            });
            KOKKOS_PROFILE_FENCE();
        }

        const weight_t max_allowed = (weight_t) (6.0 * (f64) g.g_weight / (f64) thm.n);
        u32 round = 0;
        u32 made_pairs = 1;
        while (made_pairs > 0 && n_unmapped > 0) {
            made_pairs = 0;
            // each one picks a neighbor
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "pick_neighbor");

                Kokkos::parallel_for("pick_neighbor", n_unmapped, KOKKOS_LAMBDA(const u32 i) {
                    vertex_t u = unmapped_v(i);

                    // Cache weight lookups to reduce memory access
                    weight_t u_weight = g.weights(u);

                    if (u_weight >= max_allowed) {
                        thm.preferred_neighbor(u) = u;
                        return;
                    }

                    vertex_t best_v = u;
                    f32 best_rating = -max_sentinel<f32>();
                    for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
                        vertex_t v = g.edges_v(j);

                        if (matching(v) != v) { continue; }

                        weight_t w = g.edges_w(j);
                        weight_t v_weight = g.weights(v);

                        if (v_weight >= max_allowed) { continue; }

                        f32 weight_product = (f32) (u_weight * v_weight);
                        f32 rating = (f32) (w * w) / weight_product;
                        rating += edge_noise(u, v, round);

                        if (rating > best_rating) {
                            best_v = v;
                            best_rating = rating;
                        }
                    }

                    thm.preferred_neighbor(u) = best_v;
                });
                KOKKOS_PROFILE_FENCE();
            }
            // apply the matching
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "apply_matching");
                Kokkos::parallel_reduce("apply_matching", n_unmapped, KOKKOS_LAMBDA(const u32 i, u32 &local) {
                    vertex_t u = unmapped_v(i);

                    vertex_t v = thm.preferred_neighbor(u);
                    if (v == u) { return; }

                    vertex_t pref_v = thm.preferred_neighbor(v);
                    if (pref_v == u && u < v) {
                        matching(u) = v;
                        matching(v) = u;
                        local += 1; // count pairs
                    }
                }, made_pairs);

                thm.n_matched += 2 * made_pairs; // host scalar
                KOKKOS_PROFILE_FENCE();
            }
            // Build next_unmapped_v
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "update_unmapped");
                Kokkos::parallel_scan("update_unmapped", n_unmapped, KOKKOS_LAMBDA(const u32 i, vertex_t &offset, const bool final) {
                    vertex_t u = unmapped_v(i);
                    if (matching(u) == u) {
                        if (final) {
                            next_unmapped_v(offset) = u;
                        }
                        offset += 1;
                    }
                }, next_n_unmapped);
                KOKKOS_PROFILE_FENCE();
            }

            // Swap buffers for next round
            std::swap(unmapped_v, next_unmapped_v);
            std::swap(n_unmapped, next_n_unmapped);
            round += 1;
        }

        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline void leaf_matching(TwoHopMatcher &thm,
                              const Graph &g,
                              UnmanagedDeviceVertex &matching,
                              const Partition &partition,
                              KokkosMemoryStack &mem_stack) {
        vertex_t n_unmapped = thm.n - thm.n_matched;
        UnmanagedDeviceVertex unmapped_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n_unmapped), n_unmapped);

        vertex_t n_mappable = 0;
        // determine unmapped vertices
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching", "determine_unmapped");
            Kokkos::parallel_scan("fill_unmapped_deg1", g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &offset, const bool final) {
                bool is_unmatched_deg1 = (matching(u) == u) && (g.neighborhood(u + 1) - g.neighborhood(u) == 1);

                if (is_unmatched_deg1) {
                    if (final) { unmapped_v(offset) = u; }
                    offset += 1;
                }
            }, n_mappable);
            KOKKOS_PROFILE_FENCE();
        }

        if (n_mappable == 0) {
            pop_back(mem_stack); // unmapped_v
            return;
        }

        // large hash array, but no faulty collisions possible
        UnmanagedDeviceVertex hash;
        //
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching", "init_hash_array");

            hash = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
            Kokkos::deep_copy(hash, thm.n);
        }

        u32 made_pairs = 0;
        // pick partners for unmatched vertices
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching", "pick_neighbor");
            Kokkos::parallel_reduce("pick_neighbor", n_mappable, KOKKOS_LAMBDA(const u32 i, u32 &local) {
                vertex_t u = unmapped_v(i);
                vertex_t v = g.edges_v(g.neighborhood(u));

                while (true) {
                    vertex_t old_u = Kokkos::atomic_compare_exchange(&hash(v), thm.n, u);
                    if (old_u == thm.n) {
                        // we claimed the spot, no more to be done
                        break;
                    }

                    // slot not empty, but has old_u in it, try to claim old_u
                    vertex_t next_old_u = Kokkos::atomic_compare_exchange(&hash(v), old_u, thm.n);
                    if (next_old_u == old_u) {
                        // we claimed old_u, so now match them
                        matching(u) = old_u;
                        matching(old_u) = u;
                        local += 1; // count pairs
                        break;
                    }
                }
            }, made_pairs);

            thm.n_matched += 2 * made_pairs; // host scalar
            KOKKOS_PROFILE_FENCE();
        }

        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline void twin_matching(TwoHopMatcher &thm,
                              const Graph &g,
                              UnmanagedDeviceVertex &matching,
                              const Partition &partition,
                              KokkosMemoryStack &mem_stack) {
        vertex_t n_unmapped = thm.n - thm.n_matched;
        UnmanagedDeviceVertex unmapped_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n_unmapped), n_unmapped);

        vertex_t n_mappable = 0;
        // determine unmapped vertices
        {
            ScopedTimer _t("coarsening", "thm_twin_matching", "determine_unmapped");
            Kokkos::parallel_scan("fill_unmapped_deg1", g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &offset, const bool final) {
                const bool is_unmatched = (matching(u) == u);

                if (is_unmatched) {
                    if (final) { unmapped_v(offset) = u; }
                    offset += 1;
                }
            }, n_mappable);
            KOKKOS_PROFILE_FENCE();
        }

        if (n_mappable == 0) {
            pop_back(mem_stack); // unmapped_v
            return;
        }

        // large hash array
        UnmanagedDeviceVertex hash;
        //
        {
            ScopedTimer _t("coarsening", "thm_twin_matching", "init_hash_array");

            hash = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
            Kokkos::deep_copy(hash, thm.n);
        }

        u32 made_pairs = 0;
        // pick partners for unmatched vertices
        {
            ScopedTimer _t("coarsening", "thm_twin_matching", "pick_neighbor");
            Kokkos::parallel_reduce("pick_neighbor", n_mappable, KOKKOS_LAMBDA(const u32 i, u32 &local) {
                vertex_t u = unmapped_v(i);

                u64 h = 0;
                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                    u64 x = g.edges_v(j);
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    h += x * 0x9e3779b97f4a7c15ull + 1ull;
                }
                if (h == 0) h = 1;

                u32 key = (u32) h % thm.n;
                while (true) {
                    vertex_t old_u = Kokkos::atomic_compare_exchange(&hash(key), thm.n, u);
                    if (old_u == thm.n) {
                        // we claimed the spot, no more to be done
                        break;
                    }

                    // slot not empty, but has old_u in it, try to claim old_u
                    vertex_t next_old_u = Kokkos::atomic_compare_exchange(&hash(key), old_u, thm.n);
                    if (next_old_u == old_u) {
                        // we claimed old_u, so now match them
                        matching(u) = old_u;
                        matching(old_u) = u;
                        local += 1; // count pairs
                        break;
                    }
                }
            }, made_pairs);

            thm.n_matched += 2 * made_pairs; // host scalar
            KOKKOS_PROFILE_FENCE();
        }

        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline void relative_matching(TwoHopMatcher &thm,
                                  const Graph &g,
                                  UnmanagedDeviceVertex &matching,
                                  const Partition &partition) { {
            ScopedTimer _t("coarsening", "thm_relative_matching", "pick_neighbor");
            Kokkos::parallel_for("pick_neighbor", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (matching(u) != u) {
                    // ignore already matched
                    thm.preferred_neighbor(u) = u;
                    return;
                }

                vertex_t best_v = u;
                f32 best_rating = 0;

                for (u32 i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                    vertex_t mid_v = g.edges_v(i);
                    weight_t mid_e_w = g.edges_w(i);
                    vertex_t mid_deg = g.neighborhood(mid_v + 1) - g.neighborhood(mid_v);
                    if (mid_deg > 10) { continue; }             // avoid matchmaker with high degree
                    if (matching(mid_v) == mid_v) { continue; } // avoid unmatched matchmaker

                    for (u32 j = g.neighborhood(mid_v); j < g.neighborhood(mid_v + 1); ++j) {
                        vertex_t v = g.edges_v(j);
                        weight_t v_e_w = g.edges_w(j);

                        if (u == v) { continue; }
                        if (matching(v) != v) { continue; } // ignore already matched

                        f32 rating = (f32) ((mid_e_w + v_e_w) * (mid_e_w + v_e_w)) / (f32) (g.weights(u) * g.weights(v));
                        rating += edge_noise(u, v, 0);

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
        u32 made_pairs = 0;
        //
        {
            ScopedTimer _t("coarsening", "thm_relative_matching", "apply_matching");
            Kokkos::parallel_reduce("apply_matching", g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &local) {
                if (u == thm.preferred_neighbor(u)) { return; }

                vertex_t v = thm.preferred_neighbor(u);
                vertex_t preferred_v = thm.preferred_neighbor(v);

                if (matching(u) != u || matching(v) != v) { return; }

                if (u == preferred_v && u < v) {
                    matching(u) = v;
                    matching(v) = u;
                    local += 1;
                }
            }, made_pairs);

            thm.n_matched += 2 * made_pairs; // host scalar
            KOKKOS_PROFILE_FENCE();
        }
    }

    inline Mapping two_hop_matcher_get_mapping(const Graph &g,
                                               const Partition &partition,
                                               const weight_t &lmax,
                                               KokkosMemoryStack &mem_stack) {
        assert_back_is_empty(mem_stack);

        TwoHopMatcher thm = initialize_thm(g.n, g.m, lmax, mem_stack);

        UnmanagedDeviceVertex matching;
        //
        {
            ScopedTimer _t("coarsening", "TwoHopMatcher", "allocate_matching");
            auto *matching_ptr = (vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n);
            matching = UnmanagedDeviceVertex(matching_ptr, g.n);
            Kokkos::parallel_for("set_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                matching(u) = u;
            });
            KOKKOS_PROFILE_FENCE();
        }

        heavy_edge_matching(thm, g, matching, partition, mem_stack);

        if ((f64) thm.n_matched < thm.threshold * (f64) g.n) {
            leaf_matching(thm, g, matching, partition, mem_stack);
        }

        if ((f64) thm.n_matched < thm.threshold * (f64) g.n) {
            twin_matching(thm, g, matching, partition, mem_stack);
        }

        if ((f64) thm.n_matched < thm.threshold * (f64) g.n) {
            relative_matching(thm, g, matching, partition);
        }

        Mapping mapping = determine_mapping(matching, g.n, mem_stack);

        pop_back(mem_stack); // pop the matching vec
        free_thm(thm, mem_stack);

        assert_back_is_empty(mem_stack);
        return mapping;
    }
}

#endif //GPU_HEIPA_TWO_HOP_MATCHING_H
