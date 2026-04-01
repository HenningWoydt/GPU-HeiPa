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
        thm.preferred_neighbor = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * t_n), t_n);
        thm.max_rating = UnmanagedDeviceF32((f32 *) get_chunk_back(mem_stack, sizeof(f32) * t_n), t_n);

        return thm;
    }

    inline void free_thm(TwoHopMatcher &thm,
                         KokkosMemoryStack &mem_stack) {
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    KOKKOS_INLINE_FUNCTION
    f32 edge_rating(vertex_t u,
                    vertex_t v,
                    weight_t w,
                    weight_t u_weight,
                    weight_t v_weight,
                    u32 round) {
        // ---- base heavy-edge rating ----
        f32 base = (f32) (w * w) / (f32) (u_weight * v_weight);

        // ---- deterministic symmetric noise ----
        const u32 a = u < v ? u : v;
        const u32 b = u < v ? v : u;

        u32 x = a * 0x9e3779b1u;
        x ^= b * 0x85ebca77u;
        x ^= round * 0xc2b2ae3du;

        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;

        // ---- convert to small perturbation ----
        f32 noise = (f32) (x & 0x00ffffffu) * (1.0f / 16777216.0f);

        // scale noise relative to base (important!)
        return base * (1.0f + 1e-6f * noise);
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
                lsum += need; // accumulate count
            }, mapping.coarse_n);
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
            });
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

        pop_back(mem_stack);
        pop_back(mem_stack);

        return mapping;
    }

    KOKKOS_INLINE_FUNCTION
    u32 hash_vertex(vertex_t v, u32 seed) {
        u32 x = v + seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    // Jet-style neighbor selection: pick heaviest edge, break ties randomly (serial per-thread version)
    KOKKOS_INLINE_FUNCTION
    vertex_t pick_heaviest_neighbor(const Graph &g,
                                    const UnmanagedDeviceVertex &matching,
                                    vertex_t u,
                                    u32 rand_seed,
                                    bool filter_matched) {
        constexpr vertex_t SENTINEL = std::numeric_limits<vertex_t>::max();
        weight_t max_ewt = 0;
        u32 best_tiebreaker = 0;
        vertex_t best_v = SENTINEL;

        for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); ++j) {
            vertex_t v = g.edges_v(j);
            if (filter_matched && matching(v) != v) continue;

            weight_t w = g.edges_w(j);
            if (w > max_ewt) {
                max_ewt = w;
                best_v = v;
                best_tiebreaker = hash_vertex(v, rand_seed);
            } else if (w == max_ewt) {
                u32 tb = hash_vertex(v, rand_seed);
                if (tb >= best_tiebreaker) {
                    best_v = v;
                    best_tiebreaker = tb;
                }
            }
        }
        return best_v;
    }

    using TeamPolicy_t = Kokkos::TeamPolicy<DeviceExecutionSpace>;
    using TeamMember = TeamPolicy_t::member_type;

    // TeamPolicy pick_neighbor: each team handles one vertex, team threads scan adjacency in parallel
    template<bool filter_matched>
    inline void pick_neighbor_team(TwoHopMatcher &thm,
                                   const Graph &g,
                                   const UnmanagedDeviceVertex &matching,
                                   const UnmanagedDeviceVertex &perm,
                                   vertex_t perm_len,
                                   u32 rand_seed,
                                   bool use_perm) {
        constexpr vertex_t SENTINEL = std::numeric_limits<vertex_t>::max();
        using reducer_t = Kokkos::MaxFirstLoc<u64, u32, DeviceMemorySpace>;
        using val_t = typename reducer_t::value_type;

        Kokkos::parallel_for("pick_neighbor_team", TeamPolicy_t(perm_len, Kokkos::AUTO),
                             KOKKOS_LAMBDA(const TeamMember &team) {
                                 const u32 i = team.league_rank();
                                 const vertex_t u = use_perm ? perm(i) : (vertex_t) i;

                                 if (filter_matched && matching(u) != u) return;

                                 const u32 start = g.neighborhood(u);
                                 const u32 end = g.neighborhood(u + 1);

                                 val_t best{0, end};
                                 Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, start, end),
                                                         [=](const u32 j, val_t &local) {
                                                             vertex_t v = g.edges_v(j);
                                                             if (filter_matched && matching(v) != v) return;
                                                             weight_t w = g.edges_w(j);
                                                             u32 tb = hash_vertex(v, rand_seed);
                                                             u64 key = ((u64) (u32) w << 32) | (u64) tb;
                                                             if (key >= local.val) {
                                                                 local.val = key;
                                                                 local.loc = j;
                                                             }
                                                         }, reducer_t(best));

                                 Kokkos::single(Kokkos::PerTeam(team), [=]() {
                                     if (best.loc >= start && best.loc < end) {
                                         thm.preferred_neighbor(u) = g.edges_v(best.loc);
                                     } else {
                                         thm.preferred_neighbor(u) = SENTINEL;
                                     }
                                 });
                             });
    }

    inline void heavy_edge_matching(TwoHopMatcher &thm,
                                    const Graph &g,
                                    UnmanagedDeviceVertex &matching,
                                    const Partition &partition,
                                    KokkosMemoryStack &mem_stack) {
        constexpr vertex_t SENTINEL = std::numeric_limits<vertex_t>::max();
        constexpr vertex_t INACTIVE = SENTINEL - 1;

        UnmanagedDeviceVertex perm;
        UnmanagedDeviceVertex perm_scratch;
        vertex_t perm_length = thm.n;

        {
            ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "init_unmapped");

            perm = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
            perm_scratch = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);

            Kokkos::parallel_for("init_unmapped", thm.n, KOKKOS_LAMBDA(const vertex_t u) {
                perm(u) = u;
            });
            KOKKOS_PROFILE_FENCE();
        }

        // initial pick: heaviest neighbor (consider all neighbors, no filtering)
        {
            ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "first_pick_neighbor");

            if (g.uniform_edge_weights) {
                Kokkos::parallel_for("pick_neighbor_uniform", thm.n, KOKKOS_LAMBDA(const vertex_t u) {
                    u32 adj_size = g.neighborhood(u + 1) - g.neighborhood(u);
                    if (adj_size > 0) {
                        u32 offset = g.neighborhood(u) + (hash_vertex(u, 0x12345u) % adj_size);
                        thm.preferred_neighbor(u) = g.edges_v(offset);
                    } else {
                        thm.preferred_neighbor(u) = SENTINEL;
                    }
                });
            } else {
                pick_neighbor_team<false>(thm, g, matching, perm, thm.n, 0x12345u, false);
            }
            KOKKOS_PROFILE_FENCE();
        }

        u32 round = 0;
        while (perm_length > 0) {
            // commit matches: 4 sub-rounds with hash-based active/inactive
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "4_round_commit");

                for (u32 r = 0; r < 4; ++r) {
                    // phase 1: mark inactive (claimable)
                    Kokkos::parallel_for("mark_inactive", perm_length, KOKKOS_LAMBDA(const u32 i) {
                        vertex_t u = perm(i);
                        vertex_t v = thm.preferred_neighbor(u);
                        if (v == SENTINEL || matching(u) != u) return;

                        bool active = (r == 0) ? (u < v) : (hash_vertex(u, r) < hash_vertex(v, r));
                        if (!active) {
                            Kokkos::atomic_compare_exchange(&matching(u), u, INACTIVE);
                        }
                    });

                    // phase 2: active vertices claim their inactive target
                    Kokkos::parallel_for("claim_match", perm_length, KOKKOS_LAMBDA(const u32 i) {
                        vertex_t u = perm(i);
                        if (matching(u) != u) return;
                        vertex_t v = thm.preferred_neighbor(u);
                        if (v == SENTINEL) return;

                        if (Kokkos::atomic_compare_exchange(&matching(v), INACTIVE, u) == INACTIVE) {
                            matching(u) = v;
                        }
                    });

                    // phase 3: reset unclaimed inactive back to self
                    Kokkos::parallel_for("reset_inactive", perm_length, KOKKOS_LAMBDA(const u32 i) {
                        vertex_t u = perm(i);
                        Kokkos::atomic_compare_exchange(&matching(u), INACTIVE, u);
                    });
                }
            }

            // re-pick for vertices whose target got matched away
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "repick");

                u32 seed = round * 0x9e3779b1u + 0xdeadbeefu;

                if (g.uniform_edge_weights) {
                    // O(1) random pick for uniform weights — no adjacency scan needed
                    Kokkos::parallel_for("repick_uniform", perm_length, KOKKOS_LAMBDA(const u32 i) {
                        vertex_t u = perm(i);
                        if (matching(u) != u) return;
                        vertex_t old_hn = thm.preferred_neighbor(u);
                        if (old_hn != SENTINEL && matching(old_hn) == old_hn) return;

                        u32 adj_size = g.neighborhood(u + 1) - g.neighborhood(u);
                        vertex_t best_v = SENTINEL;
                        for (u32 t = 0; t < adj_size && best_v == SENTINEL; ++t) {
                            u32 offset = g.neighborhood(u) + ((hash_vertex(u, seed) + t) % adj_size);
                            vertex_t v = g.edges_v(offset);
                            if (matching(v) == v) best_v = v;
                        }
                        thm.preferred_neighbor(u) = best_v;
                    });
                } else {
                    vertex_t n_repick = 0;
                    Kokkos::parallel_scan("build_repick", perm_length, KOKKOS_LAMBDA(const u32 i, vertex_t &offset, const bool final) {
                        vertex_t u = perm(i);
                        if (matching(u) != u) return;
                        vertex_t old_hn = thm.preferred_neighbor(u);
                        if (old_hn != SENTINEL && matching(old_hn) == old_hn) return;
                        if (final) {
                            perm_scratch(offset) = u;
                        }
                        offset += 1;
                    }, n_repick);

                    if (n_repick > 0) {
                        pick_neighbor_team<true>(thm, g, matching, perm_scratch, n_repick, seed, true);
                    }
                }
            }

            // compact
            {
                ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "compact");

                Kokkos::parallel_scan("compact_perm", perm_length, KOKKOS_LAMBDA(const u32 i, vertex_t &offset, const bool final) {
                    vertex_t u = perm(i);
                    if (matching(u) == u && thm.preferred_neighbor(u) != SENTINEL) {
                        if (final) {
                            perm_scratch(offset) = u;
                        }
                        offset += 1;
                    }
                }, perm_length);
                std::swap(perm, perm_scratch);
            }

            round += 1;
        }

        // count total matched vertices
        vertex_t n_matched = 0;
        //
        {
            ScopedTimer _t("coarsening", "thm_heavy_edge_matching", "count_matched");

            Kokkos::parallel_reduce("count_matched", thm.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &local) {
                if (matching(u) != u) local += 1;
            }, n_matched);
            thm.n_matched = n_matched;
        }

        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline void leaf_matching(TwoHopMatcher &thm,
                              const Graph &g,
                              UnmanagedDeviceVertex &matching,
                              const Partition & /*partition*/,
                              KokkosMemoryStack &mem_stack) {
        const vertex_t n_unmapped_upper = thm.n - thm.n_matched;

        // unmatched degree-1 vertices
        UnmanagedDeviceVertex unmapped_v((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n_unmapped_upper), n_unmapped_upper);

        vertex_t n_mappable = 0;

        //
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching_bucketed", "determine_unmapped");

            Kokkos::parallel_scan("fill_unmapped_deg1", g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &offset, const bool final) {
                const bool is_unmatched_deg1 = (matching(u) == u) && (g.neighborhood(u + 1) - g.neighborhood(u) == 1);

                if (is_unmatched_deg1) {
                    if (final) {
                        unmapped_v(offset) = u;
                    }
                    offset += 1;
                }
            }, n_mappable);
            KOKKOS_PROFILE_FENCE();
        }

        if (n_mappable == 0) {
            pop_back(mem_stack); // unmapped_v
            return;
        }

        UnmanagedDeviceVertex centers((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n_mappable), n_mappable);
        UnmanagedDeviceVertex counts((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
        UnmanagedDeviceVertex offsets((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * (thm.n + 1)), thm.n + 1);
        UnmanagedDeviceVertex cursors((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
        UnmanagedDeviceVertex bucketed_leaves((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n_mappable), n_mappable);
        UnmanagedDeviceVertex centers_by_bucket((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * n_mappable), n_mappable);

        {
            ScopedTimer _t("coarsening", "thm_leaf_matching_bucketed", "reset");

            Kokkos::deep_copy(counts, vertex_t(0));
            Kokkos::deep_copy(cursors, vertex_t(0));
        }
        //
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching_bucketed", "count_per_center");

            Kokkos::parallel_for(
                "count_per_center",
                n_mappable,
                KOKKOS_LAMBDA(const vertex_t i) {
                    const vertex_t u = unmapped_v(i);
                    const vertex_t v = g.edges_v(g.neighborhood(u)); // unique neighbor for deg-1 vertex
                    centers(i) = v;
                    Kokkos::atomic_inc(&counts(v));
                }
            );
            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching_bucketed", "exclusive_scan_offsets");

            Kokkos::parallel_scan(
                "exclusive_scan_offsets",
                thm.n + 1,
                KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
                    const vertex_t val = (i < thm.n ? counts(i) : 0);
                    if (final) {
                        offsets(i) = update;
                    }
                    update += val;
                }
            );
            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching_bucketed", "bucket_leaves");

            Kokkos::parallel_for(
                "bucket_leaves",
                n_mappable,
                KOKKOS_LAMBDA(const vertex_t i) {
                    const vertex_t u = unmapped_v(i);
                    const vertex_t v = centers(i);

                    const vertex_t pos = Kokkos::atomic_fetch_add(&cursors(v), vertex_t(1));
                    const vertex_t dst = offsets(v) + pos;

                    bucketed_leaves(dst) = u;
                    centers_by_bucket(dst) = v;
                }
            );
            KOKKOS_PROFILE_FENCE();
        }
        //
        {
            ScopedTimer _t("coarsening", "thm_leaf_matching_bucketed", "pair_within_buckets");

            vertex_t made_pairs = 0;

            Kokkos::parallel_reduce("pair_within_buckets", n_mappable, KOKKOS_LAMBDA(const vertex_t i, vertex_t &local_pairs) {
                const vertex_t v = centers_by_bucket(i);
                const vertex_t local_idx = i - offsets(v);

                if ((local_idx & 1) != 0) return; // only even positions start a pair
                if (i + 1 >= n_mappable) return; // safety at end of array
                if (centers_by_bucket(i + 1) != v) return; // do not cross bucket boundary

                const vertex_t u0 = bucketed_leaves(i);
                const vertex_t u1 = bucketed_leaves(i + 1);

                matching(u0) = u1;
                matching(u1) = u0;
                local_pairs += 1;
            }, made_pairs);

            thm.n_matched += 2 * made_pairs;
            KOKKOS_PROFILE_FENCE();
        }

        pop_back(mem_stack); // centers_by_bucket
        pop_back(mem_stack); // bucketed_leaves
        pop_back(mem_stack); // cursors
        pop_back(mem_stack); // offsets
        pop_back(mem_stack); // counts
        pop_back(mem_stack); // centers
        pop_back(mem_stack); // unmapped_v
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

        // pick partners for unmatched vertices
        {
            ScopedTimer _t("coarsening", "thm_twin_matching", "pick_neighbor");

            u32 made_pairs = 0;
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
                                  const Partition &partition) {
        u32 made_pairs = 1;
        u32 round = 0;
        while (made_pairs > 0) {
            //
            {
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
                        if (mid_deg > 50) { continue; } // avoid matchmaker with high degree
                        // if (matching(mid_v) == mid_v) { continue; } // avoid unmatched matchmaker

                        for (u32 j = g.neighborhood(mid_v); j < g.neighborhood(mid_v + 1); ++j) {
                            vertex_t v = g.edges_v(j);
                            weight_t v_e_w = g.edges_w(j);

                            if (u == v) { continue; }
                            if (matching(v) != v) { continue; } // ignore already matched

                            f32 rating = edge_rating(u, v, mid_e_w + v_e_w, g.weights(u), g.weights(v), round);
                            // f32 rating = (f32) ((mid_e_w + v_e_w) * (mid_e_w + v_e_w)) / (f32) (g.weights(u) * g.weights(v));
                            // rating += edge_noise(u, v, 0);

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
            made_pairs = 0;
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

            round += 1;
        }
    }

    inline Mapping two_hop_matcher_get_mapping(const Graph &g,
                                               const Partition &partition,
                                               const weight_t &lmax,
                                               KokkosMemoryStack &mem_stack) {
        TwoHopMatcher thm = initialize_thm(g.n, g.m, lmax, mem_stack);

        UnmanagedDeviceVertex matching;
        //
        {
            ScopedTimer _t("coarsening", "TwoHopMatcher", "allocate_matching");

            matching = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
            Kokkos::parallel_for("set_matching", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                matching(u) = u;
            });
            KOKKOS_PROFILE_FENCE();
        }

        heavy_edge_matching(thm, g, matching, partition, mem_stack);

        // std::cout << "Heavy-Edge: " << thm.n_matched << " " << std::fixed << thm.threshold * (f64) g.n << " " << (thm.n_matched) / ((f64) g.n) << std::endl;

        if ((f64) thm.n_matched < thm.threshold * (f64) g.n) {
            leaf_matching(thm, g, matching, partition, mem_stack);
        }

        // std::cout << "Leaf     : " << thm.n_matched << " " << std::fixed << thm.threshold * (f64) g.n << " " << (thm.n_matched) / ((f64) g.n) << std::endl;

        if ((f64) thm.n_matched < thm.threshold * (f64) g.n) {
            twin_matching(thm, g, matching, partition, mem_stack);
        }

        // std::cout << "Twin     : " << thm.n_matched << " " << std::fixed << thm.threshold * (f64) g.n << " " << (thm.n_matched) / ((f64) g.n) << std::endl;

        if ((f64) thm.n_matched < thm.threshold * (f64) g.n) {
            relative_matching(thm, g, matching, partition);
        }

        // std::cout << "Rela     : " << thm.n_matched << " " << std::fixed << thm.threshold * (f64) g.n << " " << (thm.n_matched) / ((f64) g.n) << std::endl;

        Mapping mapping = determine_mapping(matching, g.n, mem_stack);

        pop_back(mem_stack); // pop the matching vec
        free_thm(thm, mem_stack);

        return mapping;
    }
}

#endif //GPU_HEIPA_TWO_HOP_MATCHING_H
