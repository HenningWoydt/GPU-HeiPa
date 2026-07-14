#ifndef GPU_HEIPA_SMALL_GRAPH_H
#define GPU_HEIPA_SMALL_GRAPH_H

#include <Kokkos_Core.hpp>
#include "../definitions.h"
#include "kokkos_memory_stack.h"
#include "graph.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {

    struct SmallGraph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;
        bool uniform_edge_weights = false;
        bool uniform_vertex_weights = false;

        u64 n_pops = 6;

        UnmanagedDeviceWeight weights;
        UnmanagedDeviceU32 edge_begin;
        UnmanagedDeviceU32 edge_end;
        UnmanagedDeviceVertex edges_u;
        UnmanagedDeviceVertex edges_v;
        UnmanagedDeviceWeight edges_w;
    };

    inline void free_graph(SmallGraph &sg, KokkosMemoryStack &mem_stack) {
        for (u32 i = 0; i < sg.n_pops; ++i) {
            pop_front(mem_stack);
        }
        sg.n_pops = 0;
    }

    inline SmallGraph from_Graph_to_SmallGraph(const Graph &g, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "from_Graph_to_SmallGraph");
        SmallGraph sg;
        sg.n = g.n;
        sg.m = g.m;
        sg.g_weight = g.g_weight;
        sg.uniform_edge_weights = g.uniform_edge_weights;
        sg.uniform_vertex_weights = g.uniform_vertex_weights;

        sg.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * sg.n), sg.n);
        sg.edge_begin = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * sg.n), sg.n);
        sg.edge_end = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * sg.n), sg.n);
        sg.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * sg.m), sg.m);
        sg.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * sg.m), sg.m);
        sg.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * sg.m), sg.m);

        Kokkos::deep_copy(exec_space, sg.weights, g.weights);
        Kokkos::deep_copy(exec_space, sg.edges_u, g.edges_u);
        Kokkos::deep_copy(exec_space, sg.edges_v, g.edges_v);
        Kokkos::deep_copy(exec_space, sg.edges_w, g.edges_w);

        auto g_neigh = g.neighborhood;
        auto sg_begin = sg.edge_begin;
        auto sg_end = sg.edge_end;
        Kokkos::parallel_for("convert_to_small", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sg.n), KOKKOS_LAMBDA(const vertex_t u) {
            sg_begin(u) = g_neigh(u);
            sg_end(u) = g_neigh(u + 1);
        });

        KOKKOS_PROFILE_FENCE(exec_space);
        return sg;
    }

    inline Graph from_SmallGraph_to_Graph(const SmallGraph &sg, KokkosMemoryStack &mem_stack, DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "gpu_rb_partition", "from_SmallGraph_to_Graph");

        Graph g;
        g.n = sg.n;
        g.m = sg.m;
        g.g_weight = sg.g_weight;
        g.uniform_edge_weights = sg.uniform_edge_weights;
        g.uniform_vertex_weights = sg.uniform_vertex_weights;
        g.n_pops = 5;

        g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.n), g.n);
        g.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
        g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.m), g.m);
        g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * g.m), g.m);

        Kokkos::deep_copy(exec_space, g.weights, sg.weights);
        Kokkos::deep_copy(exec_space, g.edges_u, sg.edges_u);
        Kokkos::deep_copy(exec_space, g.edges_v, sg.edges_v);
        Kokkos::deep_copy(exec_space, g.edges_w, sg.edges_w);

        auto g_neigh = g.neighborhood;
        auto sg_begin = sg.edge_begin;
        auto sg_end = sg.edge_end;
        Kokkos::parallel_for("convert_to_graph", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, sg.n), KOKKOS_LAMBDA(const vertex_t u) {
            g_neigh(u) = sg_begin(u);
            if (u == sg.n - 1) {
                g_neigh(u + 1) = sg_end(u);
            }
        });

        KOKKOS_PROFILE_FENCE(exec_space);
        return g;
    }

    template<bool uniform_vw, bool uniform_ew, bool sort_by_degree = false>
    inline SmallGraph from_Graph_Mapping_small(const SmallGraph &old_g,
                                               const Mapping &mapping,
                                               KokkosMemoryStack &mem_stack,
                                               DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small_SG", "from_Graph_Mapping_small_SG");

        SmallGraph coarse_g;
        coarse_g.n = mapping.coarse_n;
        coarse_g.g_weight = old_g.g_weight;
        coarse_g.uniform_edge_weights = false;
        coarse_g.uniform_vertex_weights = false;

        coarse_g.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.n), coarse_g.n);
        coarse_g.edge_begin = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * coarse_g.n), coarse_g.n);
        coarse_g.edge_end = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * coarse_g.n), coarse_g.n);

        UnmanagedDeviceU32 max_degrees((u32 *) get_chunk_back(mem_stack, sizeof(u32) * coarse_g.n), coarse_g.n);
        Kokkos::deep_copy(exec_space, max_degrees, 0);
        Kokkos::deep_copy(exec_space, coarse_g.weights, 0);
        KOKKOS_PROFILE_FENCE(exec_space);

        // 1. Calculate max degrees, coarse weights, and reduce max chunk size simultaneously
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small_SG", "calc_max_deg");
        u32 chunk_size = 0;
        Kokkos::parallel_for("calc_max_deg", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.n), KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t new_u = mapping.mapping(u);
            u32 deg = old_g.edge_end(u) - old_g.edge_begin(u);
            weight_t w = uniform_vw ? 1 : old_g.weights(u);

            Kokkos::atomic_add(&max_degrees(new_u), deg);
            Kokkos::atomic_add(&coarse_g.weights(new_u), w);
        });

        Kokkos::parallel_reduce("find_max_chunk", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n), KOKKOS_LAMBDA(const vertex_t u, u32 &local_max) {
            if (max_degrees(u) > local_max) local_max = max_degrees(u);
        }, Kokkos::Max<u32>(chunk_size));
        KOKKOS_PROFILE_FENCE(exec_space);

        // 3. Allocate edges using uniform chunk sizes
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small_SG", "allocate_edges");
        u32 max_m = coarse_g.n * chunk_size;
        coarse_g.m = max_m;
        coarse_g.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
        coarse_g.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * coarse_g.m), coarse_g.m);
        coarse_g.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * coarse_g.m), coarse_g.m);
        
        Kokkos::parallel_for("init_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.m), KOKKOS_LAMBDA(const u32 i) {
            coarse_g.edges_v(i) = SENTINEL;
            coarse_g.edges_w(i) = 0;
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // 4. Lock-free linear probing insertion (Edge-parallel)
        HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small_SG", "insert_edges");
        Kokkos::parallel_for("insert_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.edges_v.extent(0)), KOKKOS_LAMBDA(const u32 i) {
            vertex_t old_v = old_g.edges_v(i);
            if (old_v == SENTINEL) return; // skip sentinels in old_g

            vertex_t old_u = old_g.edges_u(i);
            vertex_t new_u = mapping.mapping(old_u);
            vertex_t new_v = mapping.mapping(old_v);
            if (new_u == new_v) return; // internal edge

            weight_t w = uniform_ew ? 1 : old_g.edges_w(i);
            u32 begin = new_u * chunk_size;
            u32 limit = begin + chunk_size;

            for (u32 idx = begin; idx < limit; ++idx) {
                vertex_t existing = coarse_g.edges_v(idx);
                if (existing == new_v) {
                    Kokkos::atomic_add(&coarse_g.edges_w(idx), w);
                    break;
                }
                if (existing == SENTINEL) {
                    vertex_t old_val = Kokkos::atomic_compare_exchange(&coarse_g.edges_v(idx), SENTINEL, new_v);
                    if (old_val == SENTINEL || old_val == new_v) {
                        Kokkos::atomic_add(&coarse_g.edges_w(idx), w);
                        coarse_g.edges_u(idx) = new_u;
                        break;
                    }
                }
            }
        });
        KOKKOS_PROFILE_FENCE(exec_space);

        // 5. Finalize edge_begin and edge_end (or fuse it into sorting!)
        if constexpr (!sort_by_degree) {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small_SG", "finalize");
            Kokkos::parallel_for("finalize_bounds", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, coarse_g.n), KOKKOS_LAMBDA(const vertex_t new_u) {
                u32 start = new_u * chunk_size;
                u32 limit = start + chunk_size;
                coarse_g.edge_begin(new_u) = start;
                u32 end = start;
                for (; end < limit; ++end) {
                    if (coarse_g.edges_v(end) == SENTINEL) break;
                }
                coarse_g.edge_end(new_u) = end;
            });
            KOKKOS_PROFILE_FENCE(exec_space);
        } else {
            HEIPA_PROFILE_SCOPE("initial_partitioning", "from_Graph_Mapping_small_SG", "sort_by_degree_csr");
            UnmanagedDeviceU32 bin_counts((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (coarse_g.n + 1)), coarse_g.n + 1);
            Kokkos::deep_copy(exec_space, bin_counts, 0);

            UnmanagedDeviceVertex sorted_old_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * coarse_g.n), coarse_g.n);
            UnmanagedDeviceVertex d_perm((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * coarse_g.n), coarse_g.n);
            UnmanagedDeviceWeight temp_weights((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * coarse_g.n), coarse_g.n);
            UnmanagedDeviceU32 temp_begin((u32 *) get_chunk_back(mem_stack, sizeof(u32) * coarse_g.n), coarse_g.n);
            UnmanagedDeviceU32 temp_end((u32 *) get_chunk_back(mem_stack, sizeof(u32) * coarse_g.n), coarse_g.n);

            auto c_weights = coarse_g.weights;
            auto c_begin = coarse_g.edge_begin;
            auto c_end = coarse_g.edge_end;
            auto c_edges_v = coarse_g.edges_v;

            typedef Kokkos::TeamPolicy<DeviceExecutionSpace> TeamPolicy;
            typedef TeamPolicy::member_type TeamMember;

            Kokkos::parallel_for("sort_by_degree_csr_fused", TeamPolicy(exec_space, 1, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &team) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, coarse_g.n), [&](const vertex_t u) {
                    // Fuse finalize with count_bins
                    u32 start = u * chunk_size;
                    u32 limit = start + chunk_size;
                    c_begin(u) = start;
                    u32 end = start;
                    for (; end < limit; ++end) {
                        if (c_edges_v(end) == SENTINEL) break;
                    }
                    c_end(u) = end;

                    Kokkos::atomic_add(&bin_counts(end - start), 1);
                });
                team.team_barrier();

                u32 total = 0;
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, coarse_g.n + 1), [&](const u32 i, u32 &running, const bool final) {
                    u32 val = bin_counts(i);
                    if (final) bin_counts(i) = running;
                    running += val;
                }, total);
                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, coarse_g.n), [&](const vertex_t u) {
                    u32 write_idx = Kokkos::atomic_fetch_add(&bin_counts(c_end(u) - c_begin(u)), 1);
                    sorted_old_ids(write_idx) = u;
                });
                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, coarse_g.n), [&](const vertex_t new_u) {
                    vertex_t old_u = sorted_old_ids(new_u);
                    d_perm(old_u) = new_u;
                    temp_begin(new_u) = c_begin(old_u);
                    temp_end(new_u) = c_end(old_u);
                    temp_weights(new_u) = c_weights(old_u);
                });
                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, coarse_g.n), [&](const vertex_t new_u) {
                    vertex_t old_u = sorted_old_ids(new_u);
                    for (u32 idx = c_begin(old_u); idx < c_end(old_u); ++idx) {
                        coarse_g.edges_u(idx) = new_u;
                        coarse_g.edges_v(idx) = d_perm(coarse_g.edges_v(idx));
                    }
                });
            });

            auto map = mapping.mapping;
            Kokkos::parallel_for("update_mapping", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, old_g.n), KOKKOS_LAMBDA(const vertex_t u) {
                map(u) = d_perm(map(u));
            });

            Kokkos::deep_copy(exec_space, coarse_g.weights, temp_weights);
            Kokkos::deep_copy(exec_space, coarse_g.edge_begin, temp_begin);
            Kokkos::deep_copy(exec_space, coarse_g.edge_end, temp_end);

            pop_back(mem_stack); // temp_end
            pop_back(mem_stack); // temp_begin
            pop_back(mem_stack); // temp_weights
            pop_back(mem_stack); // d_perm
            pop_back(mem_stack); // sorted_old_ids
            pop_back(mem_stack); // bin_counts
            KOKKOS_PROFILE_FENCE(exec_space);
        }

        pop_back(mem_stack); // max_degrees
        
        coarse_g.n_pops = 6;

        KOKKOS_PROFILE_FENCE(exec_space);
        return coarse_g;
    }

    template<bool sort_by_degree = false>
    inline SmallGraph dispatch_from_Graph_Mapping_small(const SmallGraph &old_g,
                                                        const Mapping &mapping,
                                                        KokkosMemoryStack &mem_stack,
                                                        DeviceExecutionSpace &exec_space) {
        bool uvw = old_g.uniform_vertex_weights;
        bool uew = old_g.uniform_edge_weights;
        if (uvw && uew) {
            return from_Graph_Mapping_small<true, true, sort_by_degree>(old_g, mapping, mem_stack, exec_space);
        } else if (uvw) {
            return from_Graph_Mapping_small<true, false, sort_by_degree>(old_g, mapping, mem_stack, exec_space);
        } else if (uew) {
            return from_Graph_Mapping_small<false, true, sort_by_degree>(old_g, mapping, mem_stack, exec_space);
        } else {
            return from_Graph_Mapping_small<false, false, sort_by_degree>(old_g, mapping, mem_stack, exec_space);
        }
    }

} // namespace GPU_HeiPa

#endif
