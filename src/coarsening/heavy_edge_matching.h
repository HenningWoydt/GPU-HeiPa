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

    template<bool uniform_v_weights, bool uniform_e_weights>
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

        Kokkos::parallel_for("hem_small_fused_mega", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
            // 0. Initialize arrays
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                thm.vcmap(u) = SENTINEL;
                thm.hn(u) = SENTINEL;
            });
            team.team_barrier();

            // 1. Initial pick
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                vertex_t h = SENTINEL;
                weight_t max_ewt = 0;
                u32 r = xorshiftHash(u ^ SEED);
                u32 tiebreaker = 0;

                weight_t wu = 1;
                if constexpr (!uniform_v_weights) wu = g.weights(u);

                u32 start = g.edge_begin(u);
                u32 limit = g.edge_end(u);
                for (u32 j = start; j < limit; j++) {
                    vertex_t v = g.edges_v(j);
                    
                    weight_t wv = 1;
                    if constexpr (!uniform_v_weights) wv = g.weights(v);
                    // if (wu + wv > lmax) continue;

                    if constexpr (!uniform_e_weights) {
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
                thm.hn(u) = h;
            });
            team.team_barrier();

            // Main matching loop - fixed iterations
            for (u32 round = 0; round < MAX_MATCHING_ROUNDS; ++round) {
                u32 round_seed = SEED ^ (round * 0x9e3779b1u);

                // 4 sub-rounds of commit
                for (u32 r = 0; r < COMMIT_SUB_ROUNDS; r++) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                        vertex_t v = thm.hn(u);
                        if (v == SENTINEL || thm.vcmap(u) != SENTINEL) return;
                        u32 h_u = xorshiftHash(u + r);
                        u32 h_v = xorshiftHash(v + r);
                        bool condition = (r > 0) ? (h_u < h_v) : (u < v);
                        if (!condition) thm.vcmap(u) = SENTINEL - 1;
                    });
                    team.team_barrier();

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                        vertex_t v = thm.hn(u);
                        if (v == SENTINEL || thm.vcmap(u) != SENTINEL) return;
                        vertex_t cv = u < v ? u : v;
                        if (Kokkos::atomic_compare_exchange(&thm.vcmap(v), SENTINEL - 1, cv) == SENTINEL - 1) {
                            thm.vcmap(u) = cv;
                        }
                    });
                    team.team_barrier();

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                        if (thm.vcmap(u) == SENTINEL - 1) thm.vcmap(u) = SENTINEL;
                    });
                    team.team_barrier();
                }

                // Repick for unmatched
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                    if (thm.vcmap(u) != SENTINEL || thm.hn(u) == SENTINEL || thm.vcmap(thm.hn(u)) == SENTINEL) return;

                    vertex_t h = SENTINEL;
                    weight_t max_ewt = 0;
                    u32 r = xorshiftHash(u ^ round_seed);
                    u32 tiebreaker = 0;

                    weight_t wu = 1;
                    if constexpr (!uniform_v_weights) wu = g.weights(u);

                    u32 start = g.edge_begin(u);
                    u32 limit = g.edge_end(u);
                    for (u32 j = start; j < limit; j++) {
                        vertex_t v = g.edges_v(j);
                        if (thm.vcmap(v) == SENTINEL) {
                            weight_t wv = 1;
                            if constexpr (!uniform_v_weights) wv = g.weights(v);
                            // if (wu + wv > lmax) continue;

                            if constexpr (!uniform_e_weights) {
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
                    thm.hn(u) = h;
                });
                team.team_barrier();
            }

            // 1. Singletons and partners
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t i) {
                if (thm.vcmap(i) == SENTINEL) thm.vcmap(i) = i;
                mapping.partners(i) = thm.vcmap(i);
            });
            team.team_barrier();

            // 2. Set coarse ids (block scan)
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t i, vertex_t &update, const bool final) {
                if (thm.vcmap(i) == i) {
                    if (final) thm.vcmap(i) = update;
                    update++;
                } else if (final) {
                    thm.vcmap(i) += g.n;
                }
                if (final && i == g.n - 1) {
                    d_nc(0) = update;
                }
            });
            team.team_barrier();

            // 3. Propagate coarse ids & copy to mapping array
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t i) {
                if (thm.vcmap(i) >= g.n) thm.vcmap(i) = thm.vcmap((vertex_t)thm.vcmap(i) - g.n);
                mapping.mapping(i) = thm.vcmap(i);
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
}

#endif
