#ifndef GPU_HEIPA_HEAVY_EDGE_MATCHING_H
#define GPU_HEIPA_HEAVY_EDGE_MATCHING_H

#include <Kokkos_Core.hpp>
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "two_hop_matching.h"
#include "../datastructures/small_graph.h"

namespace GPU_HeiPa {
    template<bool uniform_v_weights, bool uniform_e_weights, EdgeRatingFunction rating_function = EdgeRatingFunction::EXPANSIONSTAR>
    inline Mapping heavy_edge_matching_small_get_mapping(const SmallGraph &g,
                                                         const Partition &partition,
                                                         const weight_t lmax,
                                                         KokkosMemoryStack &mem_stack,
                                                         DeviceExecutionSpace &exec_space) {
        // --- Hyperparameters ---
        constexpr u32 SEED = 0;
        constexpr u32 MAX_MATCHING_ROUNDS = 3;
        constexpr u32 COMMIT_SUB_ROUNDS = 3;
        // -----------------------

        HEIPA_PROFILE_SCOPE("initial_partitioning", "coarsen_match_small", "heavy_edge_matching_small");

        Mapping mapping = initialize_mapping(g.n, 0, mem_stack);
        TwoHopMatcher thm = initialize_two_hop_matcher(g.n, g.m, partition.k, lmax, mem_stack);
        UnmanagedDeviceU32 d_nc = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * 1), 1);

        auto thm_vcmap = thm.vcmap;
        auto thm_hn = thm.hn;
        auto g_n = g.n;
        auto g_weights = g.weights;
        auto g_edge_begin = g.edge_begin;
        auto g_edge_end = g.edge_end;
        auto g_edges_v = g.edges_v;
        auto g_edges_w = g.edges_w;
        auto mapping_partners = mapping.partners;
        auto mapping_mapping = mapping.mapping;

        Kokkos::parallel_for("hem_small_fused_mega", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
            // Phase 0: Initialize arrays
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                thm_vcmap(u) = SENTINEL;
                thm_hn(u) = SENTINEL;
            });
            team.team_barrier();

            // Phase 1: Initial pick
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                (void) g_weights;
                (void) g_edges_w;

                vertex_t h = SENTINEL;
                f32 max_ewt = -1.0f;
                u32 r = xorshiftHash(u ^ SEED);
                u32 tiebreaker = 0;

                weight_t u_w = uniform_v_weights ? 1 : g_weights(u);

                u32 start = g_edge_begin(u);
                u32 limit = g_edge_end(u);
                for (u32 j = start; j < limit; j++) {
                    vertex_t v = g_edges_v(j);

                    weight_t v_w = uniform_v_weights ? 1 : g_weights(v);
                    // if (u_w + v_w > lmax) continue;

                    if constexpr (!(uniform_v_weights && uniform_e_weights)) {
                        weight_t ew = uniform_e_weights ? 1 : g_edges_w(j);
                        f32 edge_rating;
                        if constexpr (rating_function == EdgeRatingFunction::WEIGHT) {
                            edge_rating = (f32) ew;
                        } else if constexpr (rating_function == EdgeRatingFunction::EXPANSION) {
                            edge_rating = (f32) ew / (f32) (u_w + v_w);
                        } else if constexpr (rating_function == EdgeRatingFunction::EXPANSIONSTAR) {
                            edge_rating = (f32) ew / (f32) (u_w * v_w);
                        } else if constexpr (rating_function == EdgeRatingFunction::EXPANSIONSTARSTAR) {
                            edge_rating = (f32) (ew * ew) / (f32) (u_w * v_w);
                        }

                        if (edge_rating < max_ewt) continue;
                        if (edge_rating > max_ewt) {
                            max_ewt = edge_rating;
                            h = v;
                            tiebreaker = xorshiftHash(v + r);
                            continue;
                        }
                    }

                    u32 tb = xorshiftHash(v + r);
                    if (tb >= tiebreaker) {
                        h = v;
                        tiebreaker = tb;
                    }
                }
                thm_hn(u) = h;
            });
            team.team_barrier();

            // Phase 2: Main matching loop - fixed iterations
            for (u32 round = 0; round < MAX_MATCHING_ROUNDS; ++round) {
                u32 round_seed = SEED ^ (round * 0x9e3779b1u);

                // Phase 2a: Commit sub-rounds
                for (u32 r = 0; r < COMMIT_SUB_ROUNDS; r++) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                        vertex_t v = thm_hn(u);
                        if (v == SENTINEL || thm_vcmap(u) != SENTINEL) return;

                        u32 h_u = xorshiftHash(u + r);
                        u32 h_v = xorshiftHash(v + r);
                        bool condition = (r > 0) ? (h_u < h_v) : (u < v);
                        if (!condition) thm_vcmap(u) = SENTINEL - 1;
                    });
                    team.team_barrier();

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                        vertex_t v = thm_hn(u);
                        if (v == SENTINEL || thm_vcmap(u) != SENTINEL) return;

                        vertex_t cv = u < v ? u : v;
                        if (Kokkos::atomic_compare_exchange(&thm_vcmap(v), SENTINEL - 1, cv) == SENTINEL - 1) {
                            thm_vcmap(u) = cv;
                        }
                    });
                    team.team_barrier();

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                        if (thm_vcmap(u) == SENTINEL - 1) thm_vcmap(u) = SENTINEL;
                    });
                    team.team_barrier();
                }

                // Phase 2b: Repick for unmatched
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t u) {
                    (void) g_weights;
                    (void) g_edges_w;

                    if (thm_vcmap(u) != SENTINEL) return;
                    if (thm_hn(u) == SENTINEL) return;
                    if (thm_vcmap(thm_hn(u)) == SENTINEL) return;

                    vertex_t h = SENTINEL;
                    f32 max_ewt = -1.0f;
                    u32 r = xorshiftHash(u ^ round_seed);
                    u32 tiebreaker = 0;

                    weight_t u_w = uniform_v_weights ? 1 : g_weights(u);

                    u32 start = g_edge_begin(u);
                    u32 limit = g_edge_end(u);
                    for (u32 j = start; j < limit; j++) {
                        vertex_t v = g_edges_v(j);
                        if (thm_vcmap(v) != SENTINEL) continue;

                        weight_t v_w = uniform_v_weights ? 1 : g_weights(v);
                        // if (u_w + v_w > lmax) continue;

                        if constexpr (!(uniform_v_weights && uniform_e_weights)) {
                            weight_t ew = uniform_e_weights ? 1 : g_edges_w(j);
                            f32 edge_rating;
                            if constexpr (rating_function == EdgeRatingFunction::WEIGHT) {
                                edge_rating = (f32) ew;
                            } else if constexpr (rating_function == EdgeRatingFunction::EXPANSION) {
                                edge_rating = (f32) ew / (f32) (u_w + v_w);
                            } else if constexpr (rating_function == EdgeRatingFunction::EXPANSIONSTAR) {
                                edge_rating = (f32) ew / (f32) (u_w * v_w);
                            } else if constexpr (rating_function == EdgeRatingFunction::EXPANSIONSTARSTAR) {
                                edge_rating = (f32) (ew * ew) / (f32) (u_w * v_w);
                            }

                            if (edge_rating < max_ewt) continue;
                            if (edge_rating > max_ewt) {
                                max_ewt = edge_rating;
                                h = v;
                                tiebreaker = xorshiftHash(v + r);
                                continue;
                            }
                        }

                        u32 tb = xorshiftHash(v + r);
                        if (tb >= tiebreaker) {
                            h = v;
                            tiebreaker = tb;
                        }
                    }
                    thm_hn(u) = h;
                });
                team.team_barrier();
            }

            // Phase 3: Singletons and partners
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t i) {
                if (thm_vcmap(i) == SENTINEL) thm_vcmap(i) = i;
                mapping_partners(i) = thm_vcmap(i);
            });
            team.team_barrier();

            // Phase 4: Set coarse ids (block scan)
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t i, vertex_t &update, const bool final) {
                if (thm_vcmap(i) == i) {
                    if (final) thm_vcmap(i) = update;
                    update++;
                } else if (final) {
                    thm_vcmap(i) += g_n;
                }
                if (final && i == g_n - 1) {
                    d_nc(0) = update;
                }
            });
            team.team_barrier();

            // Phase 5: Propagate coarse ids & copy to mapping array
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g_n), [&](const vertex_t i) {
                if (thm_vcmap(i) >= g_n) thm_vcmap(i) = thm_vcmap(thm_vcmap(i) - g_n);
                mapping_mapping(i) = thm_vcmap(i);
            });
        });

        u32 nc;
        Kokkos::deep_copy(exec_space, nc, Kokkos::subview(d_nc, 0));
        mapping.coarse_n = nc;

        pop_back(mem_stack); // d_nc
        free_TwoHopMatcher(thm, mem_stack);

        KOKKOS_PROFILE_FENCE(exec_space);

        return mapping;
    }

    inline Mapping dispatch_heavy_edge_matching_small_get_mapping(const SmallGraph &g,
                                                                  const Partition &partition,
                                                                  const weight_t lmax,
                                                                  KokkosMemoryStack &mem_stack,
                                                                  DeviceExecutionSpace &exec_space) {
        bool uvw = g.uniform_vertex_weights;
        bool uew = g.uniform_edge_weights;
        if (uvw && uew) {
            return heavy_edge_matching_small_get_mapping<true, true>(g, partition, lmax, mem_stack, exec_space);
        } else if (uvw) {
            return heavy_edge_matching_small_get_mapping<true, false>(g, partition, lmax, mem_stack, exec_space);
        } else if (uew) {
            return heavy_edge_matching_small_get_mapping<false, true>(g, partition, lmax, mem_stack, exec_space);
        } else {
            return heavy_edge_matching_small_get_mapping<false, false>(g, partition, lmax, mem_stack, exec_space);
        }
    }
}

#endif
