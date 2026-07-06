#ifndef GPU_HEIPA_HEAVY_EDGE_MATCHING_H
#define GPU_HEIPA_HEAVY_EDGE_MATCHING_H

#include <Kokkos_Core.hpp>
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "two_hop_matching.h"

namespace GPU_HeiPa {

    template<bool uniform_e_weights>
    inline void heavy_edge_matching_small(const Graph &g,
                                          TwoHopMatcher &thm,
                                          u32 seed,
                                          DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("coarsening", "coarsen_match", "heavy_edge_matching_small");

        Kokkos::parallel_for("hem_small", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
            
            // 1. Initial pick
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                vertex_t h = SENTINEL;
                weight_t max_ewt = 0;
                u32 r = xorshiftHash(u ^ seed);
                u32 tiebreaker = 0;

                for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                    vertex_t v = g.edges_v(j);
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
                // Don't initialize vcmap here if it is already initialized before, 
                // but usually vcmap is initialized to SENTINEL elsewhere.
            });
            team.team_barrier();

            // Main matching loop - fixed 10 iterations max for small graphs
            for(u32 round = 0; round < 10; ++round) {
                u32 round_seed = seed ^ (round * 0x9e3779b1u);
                
                // 4 sub-rounds of commit
                for (u32 r = 0; r < 4; r++) {
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

                    for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                        vertex_t v = g.edges_v(j);
                        if (thm.vcmap(v) == SENTINEL) {
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
        });
    }

    template<bool uniform_v_weights, bool uniform_e_weights>
    inline Mapping two_hop_matcher_get_mapping_small(const Graph &g,
                                                     const Partition &partition,
                                                     const weight_t &lmax,
                                                     KokkosMemoryStack &mem_stack,
                                                     DeviceExecutionSpace &exec_space) {
        TwoHopMatcher thm = initialize_two_hop_matcher(g.n, g.m, partition.k, lmax, mem_stack);

        {
            HEIPA_PROFILE_SCOPE("coarsening", "coarsen_match_small", "reset");
            Kokkos::deep_copy(exec_space, thm.vcmap, SENTINEL);
            Kokkos::deep_copy(exec_space, thm.hn, SENTINEL);
        }

        heavy_edge_matching_small<uniform_e_weights>(g, thm, 12345u, exec_space);

        Mapping mapping;
        {
            HEIPA_PROFILE_SCOPE("coarsening", "coarsen_match_small", "build_mapping_fused");
            
            // For small graphs, we can fuse singletons, set_coarse_ids, prop_coarse_ids, and mapping copy
            // Since set_coarse_ids needs a prefix sum, we can do it in a single TeamPolicy kernel
            UnmanagedDeviceU32 d_nc = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * 1), 1);
            
            Kokkos::parallel_for("build_mapping_fused", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
                // 1. Singletons
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t i) {
                    if (thm.vcmap(i) == SENTINEL) thm.vcmap(i) = i;
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
                        if (thm.vcmap(i) < g.n) d_nc(0) = update;
                        else d_nc(0) = update; // update has the total count of coarse vertices
                    }
                });
                team.team_barrier();

                // 3. Propagate coarse ids
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t i) {
                    if (thm.vcmap(i) >= g.n) thm.vcmap(i) = thm.vcmap(thm.vcmap(i) - g.n);
                });
            });

            u32 nc;
            Kokkos::deep_copy(exec_space, nc, Kokkos::subview(d_nc, 0));
            
            mapping = initialize_mapping(g.n, nc, mem_stack);
            
            Kokkos::parallel_for("copy_mapping_small", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
                mapping.mapping(u) = thm.vcmap(u);
            });
            
            pop_back(mem_stack); // d_nc
        }

        {
            HEIPA_PROFILE_SCOPE("coarsening", "coarsen_match_small", "free");
            free_TwoHopMatcher(thm, mem_stack);
        }

        return mapping;
    }

}

#endif

