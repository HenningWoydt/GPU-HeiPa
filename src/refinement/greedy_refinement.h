#ifndef GPU_HEIPA_GREEDY_REFINEMENT_H
#define GPU_HEIPA_GREEDY_REFINEMENT_H

#include <Kokkos_Core.hpp>
#include "../definitions.h"
#include "../utility/profiler.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {

    struct Move {
        weight_t gain;
        vertex_t u;
        partition_t to;
    };

    struct MoveReducer {
        using reducer = MoveReducer;
        using value_type = Move;
        using result_view_type = Kokkos::View<value_type, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

        KOKKOS_INLINE_FUNCTION
        void join(value_type &dst, const value_type &src) const {
            if (src.gain > dst.gain) {
                dst = src;
            } else if (src.gain == dst.gain && src.u < dst.u) {
                dst = src;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void init(value_type &val) const {
            val.gain = -1000000000;
            val.u = SENTINEL;
            val.to = 0;
        }

        value_type *value;
        KOKKOS_INLINE_FUNCTION MoveReducer(value_type &val) : value(&val) {}
        KOKKOS_INLINE_FUNCTION MoveReducer(result_view_type view) : value(view.data()) {}
        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }
        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }
        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    struct MovePair {
        weight_t gain;
        vertex_t u;
        vertex_t v;
        partition_t to;
    };

    struct MovePairReducer {
        using reducer = MovePairReducer;
        using value_type = MovePair;
        using result_view_type = Kokkos::View<value_type, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

        KOKKOS_INLINE_FUNCTION
        void join(value_type &dst, const value_type &src) const {
            if (src.gain > dst.gain) {
                dst = src;
            } else if (src.gain == dst.gain && src.u < dst.u) {
                dst = src;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void init(value_type &val) const {
            val.gain = -1000000000;
            val.u = SENTINEL;
            val.v = SENTINEL;
            val.to = 0;
        }

        value_type *value;
        KOKKOS_INLINE_FUNCTION MovePairReducer(value_type &val) : value(&val) {}
        KOKKOS_INLINE_FUNCTION MovePairReducer(result_view_type view) : value(view.data()) {}
        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }
        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }
        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    // Helper to find perfect matching partner in round-robin tournament
    KOKKOS_INLINE_FUNCTION
    partition_t get_partner(partition_t p, u32 r, partition_t k) {
        partition_t n = (k % 2 == 0) ? k : k + 1;
        if (n <= 1) return (partition_t)SENTINEL; 
        
        r = r % (n - 1);
        
        if (p == n - 1) return (partition_t)r;
        if (p == r) return (partition_t)(n - 1);
        
        partition_t q = (2 * r + (n - 1) - p) % (n - 1);
        return q;
    }

    template<bool uniform_vw, bool uniform_ew>
    inline void greedy_kway_refinement_small(const Graph& g, 
                                             UnmanagedDevicePartition& p_map, 
                                             UnmanagedDeviceWeight& bweights, 
                                             partition_t k, 
                                             weight_t lmax, 
                                             u32 num_iterations,
                                             DeviceExecutionSpace& exec_space) {
        HEIPA_PROFILE_SCOPE("refinement", "greedy_refinement", "small");

        num_iterations = 5; // Max full passes

        Kokkos::parallel_for("greedy_kway_refinement_single", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type& team) {
            partition_t num_rounds = (k % 2 == 0) ? k - 1 : k;
            
            // Phase 1: Single vertex moves (Round Robin)
            for (u32 iter = 0; iter < num_iterations; ++iter) {
                u32 total_active_moves = 0;
                
                for (u32 sub_iter = 0; sub_iter < num_rounds; ++sub_iter) {
                    uint64_t best_moves[256];
                    for (int i = 0; i < k && i < 256; i++) {
                        best_moves[i] = 0;
                    }
                    
                    uint32_t active_moves = 0;

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, g.n), [&](const vertex_t u) {
                        partition_t from = p_map(u);
                        if (from >= 256) return;
                        
                        partition_t to = get_partner(from, sub_iter, k);
                        if (to >= 256 || to >= k) return; // e.g., dummy block for odd K
                        
                        weight_t vw = uniform_vw ? 1 : g.weights(u);
                        if (bweights(to) + vw > lmax) return; 
                        
                        weight_t internal_deg = 0;
                        weight_t external_deg = 0;

                        for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                            partition_t neighbor_p = p_map(g.edges_v(j));
                            weight_t ew = uniform_ew ? 1 : g.edges_w(j);
                            if (neighbor_p == from) internal_deg += ew;
                            else if (neighbor_p == to) external_deg += ew;
                        }
                        
                        int32_t gain = external_deg - internal_deg;
                        if (gain > 0) {
                            uint64_t packed = ((uint64_t)gain << 32) | (uint32_t)u;
                            Kokkos::atomic_max(&best_moves[from], packed);
                        }
                    });

                    team.team_barrier();

                    Kokkos::single(Kokkos::PerTeam(team), [&](uint32_t& l_active) {
                        for (partition_t p = 0; p < k && p < 256; p++) {
                            partition_t partner = get_partner(p, sub_iter, k);
                            
                            // Only process each pair once to avoid double counting and race conditions
                            if (partner < 256 && partner < k && p < partner) {
                                uint64_t m1 = best_moves[p];
                                uint64_t m2 = best_moves[partner];
                                
                                int32_t g1 = (int32_t)(m1 >> 32);
                                uint32_t u1 = (uint32_t)(m1 & 0xFFFFFFFF);
                                
                                int32_t g2 = (int32_t)(m2 >> 32);
                                uint32_t u2 = (uint32_t)(m2 & 0xFFFFFFFF);
                                
                                // Apply the best move between the pair
                                if (g1 > 0 || g2 > 0) {
                                    l_active++;
                                    if (g1 >= g2) { // apply m1 (p -> partner)
                                        weight_t vw = uniform_vw ? 1 : g.weights(u1);
                                        p_map(u1) = partner;
                                        Kokkos::atomic_sub(&bweights(p), vw);
                                        Kokkos::atomic_add(&bweights(partner), vw);
                                    } else { // apply m2 (partner -> p)
                                        weight_t vw = uniform_vw ? 1 : g.weights(u2);
                                        p_map(u2) = p;
                                        Kokkos::atomic_sub(&bweights(partner), vw);
                                        Kokkos::atomic_add(&bweights(p), vw);
                                    }
                                }
                            }
                        }
                    }, active_moves);
                    
                    team.team_barrier();
                    
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        total_active_moves += active_moves;
                    });
                    
                    team.team_barrier();
                } // End of full round-robin tournament
                
                // If an entire tournament over all block pairs yielded zero moves, we've hit a strict local minimum
                if (total_active_moves == 0) break;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);
        return;

        HEIPA_PROFILE_SCOPE("refinement", "greedy_refinement", "double");
        Kokkos::parallel_for("greedy_kway_refinement_pair", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type& team) {
            // Phase 2: Pair vertex moves
            for (u32 iter = 0; iter < num_iterations; ++iter) {
                MovePair best_move;
                best_move.gain = -1000000000;
                best_move.u = SENTINEL;
                best_move.v = SENTINEL;
                best_move.to = 0;

                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, g.m), [&](const vertex_t e, MovePair& local_best) {
                    vertex_t u = g.edges_u(e);
                    vertex_t v = g.edges_v(e);
                    if (u >= v) return; // Process undirected edges once
                    
                    partition_t pu = p_map(u);
                    partition_t pv = p_map(v);
                    
                    if (pu >= 256 || pv >= 256) return; // Only support up to 256 blocks for small graphs
                    
                    weight_t wu = uniform_vw ? 1 : g.weights(u);
                    weight_t wv = uniform_vw ? 1 : g.weights(v);
                    weight_t edge_weight = uniform_ew ? 1 : g.edges_w(e);
                    
                    weight_t conn_u[256];
                    weight_t conn_v[256];
                    for(int i = 0; i < k && i < 256; i++) {
                        conn_u[i] = 0;
                        conn_v[i] = 0;
                    }

                    for (u32 j = g.neighborhood(u); j < g.neighborhood(u + 1); j++) {
                        partition_t to = p_map(g.edges_v(j));
                        if (to < 256) conn_u[to] += uniform_ew ? 1 : g.edges_w(j);
                    }
                    
                    for (u32 j = g.neighborhood(v); j < g.neighborhood(v + 1); j++) {
                        partition_t to = p_map(g.edges_v(j));
                        if (to < 256) conn_v[to] += uniform_ew ? 1 : g.edges_w(j);
                    }
                    
                    weight_t u_internal_deg = conn_u[pu];
                    weight_t v_internal_deg = conn_v[pv];
                    
                    weight_t max_gain = 0;
                    partition_t best_to = 0;

                    for (partition_t to = 0; to < k && to < 256; ++to) {
                        if (to == pu || to == pv) continue;
                        
                        if (bweights(to) + wu + wv <= lmax) {
                            weight_t gu = conn_u[to] - u_internal_deg;
                            weight_t gv = conn_v[to] - v_internal_deg;
                            
                            weight_t pair_gain = gu + gv + edge_weight * (2 - (pu != pv));
                            if (pair_gain > max_gain) {
                                max_gain = pair_gain;
                                best_to = to;
                            }
                        }
                    }
                    
                    if (max_gain > 0 && max_gain > local_best.gain) {
                        local_best.gain = max_gain;
                        local_best.u = u;
                        local_best.v = v;
                        local_best.to = best_to;
                    }
                }, MovePairReducer(best_move));

                team.team_barrier();

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    if (best_move.gain > 0 && best_move.u != SENTINEL) {
                        partition_t to = best_move.to;
                        partition_t from_u = p_map(best_move.u);
                        partition_t from_v = p_map(best_move.v);
                        
                        weight_t wu = uniform_vw ? 1 : g.weights(best_move.u);
                        weight_t wv = uniform_vw ? 1 : g.weights(best_move.v);
                        
                        p_map(best_move.u) = to;
                        Kokkos::atomic_sub(&bweights(from_u), wu);
                        Kokkos::atomic_add(&bweights(to), wu);
                        
                        p_map(best_move.v) = to;
                        Kokkos::atomic_sub(&bweights(from_v), wv);
                        Kokkos::atomic_add(&bweights(to), wv);
                    }
                });
                
                team.team_barrier();
                
                // Early exit if no valid moves were found
                if (best_move.gain <= 0) break;
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);
    }

}

#endif
