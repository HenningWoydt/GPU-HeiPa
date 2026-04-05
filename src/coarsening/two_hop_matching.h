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

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_Functional.hpp>

#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    using hasher_t = Kokkos::pod_hash<vertex_t>;
    using argmax_reducer_t = Kokkos::MaxFirstLoc<u32, u32, DeviceMemorySpace>;
    using argmax_t = typename argmax_reducer_t::value_type;

    using TeamPolicy_t = Kokkos::TeamPolicy<DeviceExecutionSpace>;
    using TeamMember = TeamPolicy_t::member_type;

    constexpr vertex_t SENTINEL = std::numeric_limits<vertex_t>::max();

    struct TwoHopMatcher {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        UnmanagedDeviceVertex vcmap;
        UnmanagedDeviceVertex hn;
        UnmanagedDeviceVertex vertex_list;
        UnmanagedDeviceVertex vertex_list_temp;
    };

    inline TwoHopMatcher initialize_two_hop_matcher(const vertex_t t_n,
                                                    const vertex_t t_m,
                                                    const partition_t t_k,
                                                    const weight_t t_lmax,
                                                    KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("coarsening", "TwoHopMatcher", "allocate");

        TwoHopMatcher thm;

        thm.n = t_n;
        thm.m = t_m;
        thm.k = t_k;
        thm.lmax = t_lmax;

        thm.vcmap = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
        thm.hn = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
        thm.vertex_list = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);
        thm.vertex_list_temp = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * thm.n), thm.n);

        return thm;
    }

    inline void free_TwoHopMatcher(const TwoHopMatcher &thm,
                                   KokkosMemoryStack &mem_stack) {
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    KOKKOS_INLINE_FUNCTION
    u32 xorshiftHash(u32 key) {
        u32 x = key;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    template<bool is_initial, bool is_uniform>
    inline void pick_neighbor_flat(const Graph &g,
                                   const UnmanagedDeviceVertex &vcmap,
                                   const UnmanagedDeviceVertex &hn,
                                   u32 seed,
                                   const UnmanagedDeviceVertex &vperm,
                                   vertex_t perm_length,
                                   DeviceExecutionSpace &exec_space) {
        Kokkos::parallel_for("pick_flat", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, perm_length), KOKKOS_LAMBDA(const vertex_t i) {
            vertex_t u = perm_length == g.n ? i : vperm(i);
            if (!is_initial && (vcmap(u) != SENTINEL || hn(u) == SENTINEL || vcmap(hn(u)) == SENTINEL)) return;

            vertex_t h = SENTINEL;
            u32 r = xorshiftHash(u ^ seed);

            weight_t max_ewt = 0;
            u32 tiebreaker = 0;

            for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                vertex_t v = g.edges_v(j);
                if (is_initial || vcmap(v) == SENTINEL) {
                    if constexpr (!is_uniform) {
                        if (max_ewt < g.edges_w(j)) {
                            max_ewt = g.edges_w(j);
                            h = v;
                            tiebreaker = xorshiftHash(v + r);
                            continue;
                        }
                        if (max_ewt != g.edges_w(j)) continue;
                    }
                    u32 tb = xorshiftHash(v + r);
                    if (tb >= tiebreaker) {
                        h = v;
                        tiebreaker = tb;
                    }
                }
            }
            hn(u) = h;
        });
    }

    template<bool is_initial, bool is_uniform>
    inline void pick_neighbor_team(const Graph &g,
                                   const UnmanagedDeviceVertex &vcmap,
                                   const UnmanagedDeviceVertex &hn,
                                   u32 seed,
                                   const UnmanagedDeviceVertex &vertex_list,
                                   vertex_t n_vertices,
                                   DeviceExecutionSpace &exec_space) {
        Kokkos::parallel_for("pick_team", TeamPolicy_t(exec_space, n_vertices, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &thread) {
            const vertex_t i = thread.league_rank();
            vertex_t u = n_vertices == g.n ? i : vertex_list(i);
            if (!is_initial && (vcmap(u) != SENTINEL || hn(u) == SENTINEL || vcmap(hn(u)) == SENTINEL)) return;

            u32 start = g.neighborhood(u);
            u32 end = g.neighborhood(u + 1);
            u32 r = xorshiftHash(u ^ seed);

            weight_t max_ewt = 0;
            if constexpr (!is_uniform) {
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const u32 j, weight_t &update) {
                    if (!is_initial && vcmap(g.edges_v(j)) != SENTINEL) return;
                    if (g.edges_w(j) > update) update = g.edges_w(j);
                }, Kokkos::Max<weight_t, DeviceMemorySpace>(max_ewt));
            }
            thread.team_barrier();

            argmax_t argmax{0, end};
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const u32 j, argmax_t &local) {
                if (!is_initial && vcmap(g.edges_v(j)) != SENTINEL) return;
                if constexpr (!is_uniform) {
                    if (g.edges_w(j) != max_ewt) return;
                }
                u32 v = g.edges_v(j);
                u32 tb = xorshiftHash(v + r);
                if (tb >= local.val) {
                    local.val = tb;
                    local.loc = j;
                }
            }, argmax_reducer_t(argmax));
            thread.team_barrier();

            Kokkos::single(Kokkos::PerTeam(thread), [=]() {
                hn(u) = (argmax.loc >= start && argmax.loc < end) ? g.edges_v(argmax.loc) : SENTINEL;
            });
        });
    }

    template<bool is_initial, bool is_uniform>
    inline void pick_neighbor(const Graph &g,
                              const UnmanagedDeviceVertex &vcmap,
                              const UnmanagedDeviceVertex &hn,
                              u32 seed,
                              const UnmanagedDeviceVertex &vperm,
                              vertex_t perm_length,
                              DeviceExecutionSpace &exec_space) {
        if (g.m / g.n > 32) {
            pick_neighbor_team<is_initial, is_uniform>(g, vcmap, hn, seed, vperm, perm_length, exec_space);
        } else {
            pick_neighbor_flat<is_initial, is_uniform>(g, vcmap, hn, seed, vperm, perm_length, exec_space);
        }
    }

    template<typename hash_t>
    inline void matchHash(const UnmanagedDeviceVertex &unmappedVtx,
                          const Kokkos::View<hash_t *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > &hashes,
                          const hash_t nullkey,
                          UnmanagedDeviceVertex &vcmap,
                          vertex_t mappable,
                          KokkosMemoryStack &mem_stack,
                          DeviceExecutionSpace &exec_space) {
        Kokkos::View<hash_t *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > htable((hash_t *) get_chunk_back(mem_stack, sizeof(hash_t) * mappable), mappable);
        UnmanagedDeviceVertex twins((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * mappable), mappable);

        Kokkos::deep_copy(exec_space, htable, nullkey);
        Kokkos::deep_copy(exec_space, twins, SENTINEL);

        Kokkos::parallel_for("match_by_hash", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, mappable), KOKKOS_LAMBDA(const vertex_t x) {
            vertex_t i = unmappedVtx(x);
            hash_t h = hashes(x);
            vertex_t key = static_cast<vertex_t>(h % mappable);
            bool found = false;
            while (!found) {
                if (htable(key) == nullkey) {
                    Kokkos::atomic_compare_exchange(&htable(key), nullkey, h);
                }
                if (htable(key) == h) {
                    found = true;
                } else {
                    ++key;
                    if (key >= mappable) key -= mappable;
                }
            }
            found = false;
            while (!found) {
                vertex_t twin = twins(key);
                if (twin == SENTINEL) {
                    if (Kokkos::atomic_compare_exchange(&twins(key), twin, i) == twin) found = true;
                } else {
                    if (Kokkos::atomic_compare_exchange(&twins(key), twin, SENTINEL) == twin) {
                        vertex_t cv = twin < i ? twin : i;
                        vcmap(twin) = cv;
                        vcmap(i) = cv;
                        found = true;
                    }
                }
            }
        });

        pop_back(mem_stack); // twins
        pop_back(mem_stack); // htable
    }

    inline vertex_t count_unmapped(const Graph &g,
                                   const TwoHopMatcher &thm,
                                   DeviceExecutionSpace &exec_space) {
        vertex_t unmapped = 0;
        {
            ScopedTimer _t("coarsening", "coarsen_match", "count_unmapped");

            Kokkos::parallel_reduce("count_unmapped", g.n, KOKKOS_LAMBDA(vertex_t i, vertex_t &update) {
                if (thm.vcmap(i) == SENTINEL) update++;
            }, unmapped);

            KOKKOS_PROFILE_FENCE(exec_space);
        }
        return unmapped;
    }

    template<bool uniform_v_weights, bool uniform_e_weights>
    inline void heavy_edge_matching(const Graph &g,
                                    TwoHopMatcher &thm,
                                    u32 seed,
                                    DeviceExecutionSpace &exec_space) {
        if (uniform_e_weights) {
            ScopedTimer _t("coarsening", "coarsen_match", "initial_pick_uniform");

            Kokkos::parallel_for("initial_pick_uniform", g.n, KOKKOS_LAMBDA(vertex_t i) {
                u32 adj_size = g.neighborhood(i + 1) - g.neighborhood(i);
                if (adj_size == 0) return;
                u32 offset = g.neighborhood(i) + (xorshiftHash(i ^ seed) % adj_size);
                thm.hn(i) = g.edges_v(offset);
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        } else {
            if (g.m / g.n > 32) {
                ScopedTimer _t("coarsening", "coarsen_match", "initial_pick_team");

                pick_neighbor_team<true, uniform_e_weights>(g, thm.vcmap, thm.hn, seed, thm.vertex_list, g.n, exec_space);

                KOKKOS_PROFILE_FENCE(exec_space);
            } else {
                ScopedTimer _t("coarsening", "coarsen_match", "initial_pick_flat");

                pick_neighbor_flat<true, uniform_e_weights>(g, thm.vcmap, thm.hn, seed, thm.vertex_list, g.n, exec_space);

                KOKKOS_PROFILE_FENCE(exec_space);
            }
        }

        // ---- Main matching loop ----
        vertex_t perm_length = g.n;
        u32 round = 0;
        while (perm_length > 0) {
            u32 round_seed = seed ^ (round * 0x9e3779b1u);
            // 4 sub-rounds of commit
            {
                ScopedTimer _t("coarsening", "coarsen_match", "commit");

                for (u32 r = 0; r < 4; r++) {
                    Kokkos::parallel_for("commit_p1", perm_length, KOKKOS_LAMBDA(vertex_t i) {
                        vertex_t u = perm_length == g.n ? i : thm.vertex_list(i);
                        vertex_t v = thm.hn(u);
                        if (v == SENTINEL || thm.vcmap(u) != SENTINEL) return;
                        hasher_t hash;
                        bool condition = (r > 0) ? (hash(u + r) < hash(v + r)) : (u < v);
                        if (!condition) thm.vcmap(u) = SENTINEL - 1;
                    });
                    Kokkos::parallel_for("commit_p2", perm_length, KOKKOS_LAMBDA(vertex_t i) {
                        vertex_t u = perm_length == g.n ? i : thm.vertex_list(i);
                        vertex_t v = thm.hn(u);
                        if (v == SENTINEL || thm.vcmap(u) != SENTINEL) return;
                        vertex_t cv = u < v ? u : v;
                        if (Kokkos::atomic_compare_exchange(&thm.vcmap(v), SENTINEL - 1, cv) == SENTINEL - 1) {
                            thm.vcmap(u) = cv;
                        }
                    });
                    Kokkos::parallel_for("commit_p3", perm_length, KOKKOS_LAMBDA(vertex_t i) {
                        vertex_t u = perm_length == g.n ? i : thm.vertex_list(i);
                        if (thm.vcmap(u) == SENTINEL - 1) thm.vcmap(u) = SENTINEL;
                    });
                }

                KOKKOS_PROFILE_FENCE(exec_space);
            }

            // Re-pick for unmatched
            {
                if (uniform_e_weights) {
                    if (g.m / g.n > 32) {
                        ScopedTimer _t("coarsening", "coarsen_match", "repick_uniform_team");

                        pick_neighbor_team<false, uniform_e_weights>(g, thm.vcmap, thm.hn, round_seed, thm.vertex_list, perm_length, exec_space);

                        KOKKOS_PROFILE_FENCE(exec_space);
                    } else {
                        ScopedTimer _t("coarsening", "coarsen_match", "repick_uniform_flat");

                        pick_neighbor_flat<false, uniform_e_weights>(g, thm.vcmap, thm.hn, round_seed, thm.vertex_list, perm_length, exec_space);

                        KOKKOS_PROFILE_FENCE(exec_space);
                    }
                } else {
                    if (g.m / g.n > 32) {
                        ScopedTimer _t("coarsening", "coarsen_match", "repick_team");

                        pick_neighbor_team<false, uniform_e_weights>(g, thm.vcmap, thm.hn, round_seed, thm.vertex_list, perm_length, exec_space);

                        KOKKOS_PROFILE_FENCE(exec_space);
                    } else {
                        ScopedTimer _t("coarsening", "coarsen_match", "repick_flat");

                        pick_neighbor_flat<false, uniform_e_weights>(g, thm.vcmap, thm.hn, round_seed, thm.vertex_list, perm_length, exec_space);

                        KOKKOS_PROFILE_FENCE(exec_space);
                    }
                }
            }

            // Compact
            {
                ScopedTimer _t("coarsening", "coarsen_match", "compact");

                if (perm_length != g.n) {
                    Kokkos::parallel_for("copy_perm", perm_length, KOKKOS_LAMBDA(vertex_t i) {
                        thm.vertex_list_temp(i) = thm.vertex_list(i);
                    });
                }
                Kokkos::parallel_scan("compact", perm_length, KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
                    vertex_t u = perm_length == g.n ? i : thm.vertex_list_temp(i);
                    if (thm.vcmap(u) == SENTINEL && thm.hn(u) != SENTINEL) {
                        if (final) thm.vertex_list(update) = u;
                        update++;
                    }
                }, perm_length);

                KOKKOS_PROFILE_FENCE(exec_space);
            }
            round++;
        }
    }

    inline void leaf_matching(const Graph &g,
                              TwoHopMatcher &thm,
                              vertex_t unmapped,
                              KokkosMemoryStack &mem_stack,
                              DeviceExecutionSpace &exec_space) {
        ScopedTimer _t("coarsening", "coarsen_match", "leaf");

        UnmanagedDeviceVertex unmappedVtx((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * unmapped), unmapped);
        UnmanagedDeviceVertex hashes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * unmapped), unmapped);

        vertex_t mappable = 0;
        Kokkos::parallel_scan("scan_leaf", g.n, KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
            if (thm.vcmap(i) == SENTINEL && g.neighborhood(i + 1) - g.neighborhood(i) == 1) {
                if (final) {
                    const vertex_t pos = update;
                    unmappedVtx(pos) = i;
                    hashes(pos) = g.edges_v(g.neighborhood(i));
                }
                update++;
            }
        }, mappable);

        if (mappable > 0) {
            matchHash<vertex_t>(unmappedVtx, hashes, SENTINEL, thm.vcmap, mappable, mem_stack, exec_space);
        }

        pop_back(mem_stack); // hashes
        pop_back(mem_stack); // unmappedVtx

        KOKKOS_PROFILE_FENCE(exec_space);
    }

    inline void twin_matching(const Graph &g,
                              TwoHopMatcher &thm,
                              vertex_t unmapped,
                              KokkosMemoryStack &mem_stack,
                              DeviceExecutionSpace &exec_space) {
        UnmanagedDeviceVertex unmappedVtx((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * unmapped), unmapped);
        UnmanagedDeviceU64 hashes((u64 *) get_chunk_back(mem_stack, sizeof(u64) * unmapped), unmapped);

        u64 table_size = g.n;
        UnmanagedDeviceU64 htable((u64 *) get_chunk_back(mem_stack, sizeof(u64) * table_size), table_size);
        UnmanagedDeviceVertex twins((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * table_size), table_size);

        {
            ScopedTimer _t("coarsening", "twin_matching", "unmapped");

            Kokkos::parallel_scan("scan_twin", g.n, KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
                if (thm.vcmap(i) == SENTINEL) {
                    if (final) unmappedVtx(update) = i;
                    update++;
                }
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }
        //
        {
            ScopedTimer _t("coarsening", "twin_matching", "hash");

            hasher_t hasher;
            Kokkos::parallel_for("twin_digests", TeamPolicy_t(unmapped, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &thread) {
                vertex_t u = unmappedVtx(thread.league_rank());
                u64 hash = 0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 j, u64 &thread_sum) {
                    u64 x = g.edges_v(j);
                    u64 y = hasher(static_cast<vertex_t>(x));
                    y = y * y + y;
                    thread_sum += y;
                }, hash);
                Kokkos::single(Kokkos::PerTeam(thread), [=]() {
                    hashes(thread.league_rank()) = hash;
                });
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }
        //
        {
            ScopedTimer _t("coarsening", "twin_matching", "reset_htable");

            Kokkos::deep_copy(htable, 0);
            Kokkos::deep_copy(twins, SENTINEL);

            KOKKOS_PROFILE_FENCE(exec_space);
        }
        //
        {
            ScopedTimer _t("coarsening", "twin_matching", "match");

            Kokkos::parallel_for("match_by_hash", unmapped, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = unmappedVtx(i);
                u64 h = hashes(i);
                vertex_t key = (vertex_t) (h % table_size);
                bool found = false;
                while (!found) {
                    if (htable(key) == 0) {
                        Kokkos::atomic_compare_exchange(&htable(key), 0, h);
                    }
                    if (htable(key) == h) {
                        found = true;
                    } else {
                        ++key;
                        if (key >= table_size) key -= table_size;
                    }
                }
                found = false;
                while (!found) {
                    vertex_t twin = twins(key);
                    if (twin == SENTINEL) {
                        if (Kokkos::atomic_compare_exchange(&twins(key), twin, u) == twin) found = true;
                    } else {
                        if (Kokkos::atomic_compare_exchange(&twins(key), twin, SENTINEL) == twin) {
                            vertex_t cv = twin < u ? twin : u;
                            thm.vcmap(twin) = cv;
                            thm.vcmap(u) = cv;
                            found = true;
                        }
                    }
                }
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        pop_back(mem_stack); // hashes
        pop_back(mem_stack); // unmappedVtx
        pop_back(mem_stack); // twins
        pop_back(mem_stack); // htable
    }

    template<bool uniform_e_weights>
    inline void relative_matching(const Graph &g,
                                  TwoHopMatcher &thm,
                                  vertex_t unmapped,
                                  KokkosMemoryStack &mem_stack,
                                  DeviceExecutionSpace &exec_space) {
        ScopedTimer _t("coarsening", "coarsen_match", "relative");

        UnmanagedDeviceVertex unmappedVtx((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * unmapped), unmapped);
        UnmanagedDeviceVertex hashes((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * unmapped), unmapped);

        vertex_t mappable = 0;
        Kokkos::parallel_scan("scan_rel", g.n, KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
            if (thm.vcmap(i) == SENTINEL) {
                if (final) unmappedVtx(update) = i;
                update++;
            }
        }, mappable);

        Kokkos::parallel_for("rel_digests", mappable, KOKKOS_LAMBDA(vertex_t i) {
            vertex_t u = unmappedVtx(i);
            vertex_t h = SENTINEL;
            weight_t max_wgt = 0;
            vertex_t min_deg = SENTINEL;
            for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                vertex_t v = g.edges_v(j);
                vertex_t vdeg = g.neighborhood(v + 1) - g.neighborhood(v);
                if (min_deg > vdeg) {
                    min_deg = vdeg;
                    max_wgt = uniform_e_weights ? 1 : g.edges_w(j);
                    h = v;
                } else if (min_deg == vdeg && max_wgt < (uniform_e_weights ? 1 : g.edges_w(j))) {
                    h = v;
                    max_wgt = uniform_e_weights ? 1 : g.edges_w(j);
                }
            }
            hashes(i) = h;
        });

        matchHash<vertex_t>(unmappedVtx, hashes, SENTINEL, thm.vcmap, mappable, mem_stack, exec_space);

        pop_back(mem_stack); // hashes
        pop_back(mem_stack); // unmappedVtx

        KOKKOS_PROFILE_FENCE(exec_space);
    }

    template<bool uniform_v_weights, bool uniform_e_weights>
    inline Mapping two_hop_matcher_get_mapping(const Graph &g,
                                               const Partition &partition,
                                               const weight_t &lmax,
                                               KokkosMemoryStack &mem_stack,
                                               DeviceExecutionSpace &exec_space) {
        TwoHopMatcher thm = initialize_two_hop_matcher(g.n, g.m, partition.k, lmax, mem_stack);

        {
            ScopedTimer _t("coarsening", "coarsen_match", "reset");

            Kokkos::deep_copy(thm.vcmap, SENTINEL);
            Kokkos::deep_copy(thm.hn, SENTINEL);

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        heavy_edge_matching<uniform_v_weights, uniform_e_weights>(g, thm, 12345u, exec_space);

        vertex_t unmapped = count_unmapped(g, thm, exec_space);
        if ((f64) unmapped / (f64) g.n > 0.25) {
            leaf_matching(g, thm, unmapped, mem_stack, exec_space);

            unmapped = count_unmapped(g, thm, exec_space);
        }

        // Twin matches
        if ((f64) unmapped / (f64) g.n > 0.25) {
            twin_matching(g, thm, unmapped, mem_stack, exec_space);

            unmapped = count_unmapped(g, thm, exec_space);
        }

        // Relative matches
        if ((f64) unmapped / (f64) g.n > 0.25) {
            relative_matching<uniform_e_weights>(g, thm, unmapped, mem_stack, exec_space);
        }
        //
        Mapping mapping;
        //
        {
            ScopedTimer _t("coarsening", "coarsen_match", "build_mapping");

            Kokkos::parallel_for("singletons", g.n, KOKKOS_LAMBDA(vertex_t i) {
                if (thm.vcmap(i) == SENTINEL) thm.vcmap(i) = i;
            });

            vertex_t nc = 0;
            Kokkos::parallel_scan("set_coarse_ids", g.n, KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
                if (thm.vcmap(i) == i) {
                    if (final) thm.vcmap(i) = update;
                    update++;
                } else if (final) {
                    thm.vcmap(i) += g.n;
                }
            }, nc);

            Kokkos::parallel_for("prop_coarse_ids", g.n, KOKKOS_LAMBDA(const vertex_t i) {
                if (thm.vcmap(i) >= g.n) thm.vcmap(i) = thm.vcmap(thm.vcmap(i) - g.n);
            });

            // ---- Build Mapping ----
            mapping = initialize_mapping(g.n, nc, mem_stack);
            Kokkos::parallel_for("copy_mapping", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                mapping.mapping(u) = thm.vcmap(u);
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        {
            ScopedTimer _t("coarsening", "coarsen_match", "free");

            free_TwoHopMatcher(thm, mem_stack);
        }

        return mapping;
    }
} // namespace GPU_HeiPa

#endif //GPU_HEIPA_TWO_HOP_MATCHING_H
