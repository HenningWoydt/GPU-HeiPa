#ifndef GPU_HEIPA_DUMB_MATCHING_H
#define GPU_HEIPA_DUMB_MATCHING_H

#include <Kokkos_Core.hpp>
#include "../definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "two_hop_matching.h"

namespace GPU_HeiPa {

    template<bool uniform_e_weights, bool constrain_partition = false>
    inline Mapping dumb_matcher_get_mapping(const Graph &g,
                                            const Partition &partition,
                                            const weight_t &lmax,
                                            KokkosMemoryStack &mem_stack,
                                            DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("coarsening", "dumb_match", "total");

        UnmanagedDeviceVertex vcmap((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);

        {
            HEIPA_PROFILE_SCOPE("coarsening", "dumb_match", "reset");
            Kokkos::deep_copy(exec_space, vcmap, SENTINEL);
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        // 3 iterations of 1-hop + 2-hop matching
        for (u32 iter = 0; iter < 2; ++iter) {
            u32 seed = 12345u ^ (iter * 0x9e3779b1u);
            
            // 1-hop matching
            {
                HEIPA_PROFILE_SCOPE("coarsening", "dumb_match", "1-hop");
                Kokkos::parallel_for("dumb_1hop", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(vertex_t u) {
                    if (vcmap(u) != SENTINEL) return;
                    
                    vertex_t best_v = SENTINEL;
                    weight_t max_wt = 0;
                    u32 tiebreaker = 0;
                    u32 r = xorshiftHash(u ^ seed);

                    for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                        vertex_t v = g.edges_v(j);
                        if (vcmap(v) == SENTINEL) {
                            if (constrain_partition) {
                                if (partition.map(u) != partition.map(v)) continue;
                            }
                            weight_t wt = uniform_e_weights ? 1 : g.edges_w(j);
                            if (wt > max_wt) {
                                max_wt = wt;
                                best_v = v;
                                tiebreaker = xorshiftHash(v + r);
                            } else if (wt == max_wt) {
                                u32 tb = xorshiftHash(v + r);
                                if (tb > tiebreaker) {
                                    best_v = v;
                                    tiebreaker = tb;
                                }
                            }
                        }
                    }
                    
                    if (best_v != SENTINEL) {
                        vertex_t min_v = (u < best_v) ? u : best_v;
                        vertex_t max_v = (u > best_v) ? u : best_v;
                        
                        if (Kokkos::atomic_compare_exchange(&vcmap(max_v), SENTINEL, min_v) == SENTINEL) {
                            if (Kokkos::atomic_compare_exchange(&vcmap(min_v), SENTINEL, min_v) != SENTINEL) {
                                Kokkos::atomic_exchange(&vcmap(max_v), SENTINEL); // rollback
                            }
                        }
                    }
                });
                KOKKOS_PROFILE_FENCE(exec_space);
            }

            // 2-hop matching
            {
                HEIPA_PROFILE_SCOPE("coarsening", "dumb_match", "2-hop");
                Kokkos::parallel_for("dumb_2hop", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(vertex_t u) {
                    if (vcmap(u) != SENTINEL) return;
                    
                    vertex_t best_v = SENTINEL;
                    for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                        vertex_t v = g.edges_v(j);
                        for (u32 k = g.neighborhood(v); k < g.neighborhood(v + 1); k++) {
                            vertex_t w = g.edges_v(k);
                            if (w != u && vcmap(w) == SENTINEL) {
                                if (constrain_partition) {
                                    if (partition.map(u) != partition.map(w)) continue;
                                }
                                best_v = w;
                                break;
                            }
                        }
                        if (best_v != SENTINEL) break;
                    }

                    if (best_v != SENTINEL) {
                        vertex_t min_v = (u < best_v) ? u : best_v;
                        vertex_t max_v = (u > best_v) ? u : best_v;
                        
                        if (Kokkos::atomic_compare_exchange(&vcmap(max_v), SENTINEL, min_v) == SENTINEL) {
                            if (Kokkos::atomic_compare_exchange(&vcmap(min_v), SENTINEL, min_v) != SENTINEL) {
                                Kokkos::atomic_exchange(&vcmap(max_v), SENTINEL); // rollback
                            }
                        }
                    }
                });
                KOKKOS_PROFILE_FENCE(exec_space);
            }
        }

        Mapping mapping;
        {
            HEIPA_PROFILE_SCOPE("coarsening", "dumb_match", "build_mapping");

            Kokkos::parallel_for("singletons", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(vertex_t i) {
                if (vcmap(i) == SENTINEL) vcmap(i) = i;
            });

            vertex_t nc = 0;
            Kokkos::parallel_scan("set_coarse_ids", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t i, vertex_t &update, const bool final) {
                if (vcmap(i) == i) {
                    if (final) vcmap(i) = update;
                    update++;
                } else if (final) {
                    vcmap(i) += g.n;
                }
            }, nc);

            Kokkos::parallel_for("prop_coarse_ids", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t i) {
                if (vcmap(i) >= g.n) vcmap(i) = vcmap(vcmap(i) - g.n);
            });

            mapping = initialize_mapping(g.n, nc, mem_stack);
            Kokkos::parallel_for("copy_mapping", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
                mapping.mapping(u) = vcmap(u);
            });

            KOKKOS_PROFILE_FENCE(exec_space);
        }

        pop_back(mem_stack); // vcmap
        return mapping;
    }
}
#endif
