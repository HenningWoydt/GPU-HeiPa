#ifndef GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H
#define GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../datastructures/graph.h"
#include "../datastructures/partition.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "../coarsening/two_hop_matching.h"
#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/custom_reductions.h"

namespace GPU_HeiPa {
    inline void count_block_vertices(const Partition &partition,
                                     const Graph &g,
                                     UnmanagedDeviceVertex &d_counts,
                                     DeviceExecutionSpace &exec_space) {
        Kokkos::deep_copy(exec_space, d_counts, 0);
        auto part_map = partition.map;
        Kokkos::parallel_for("count_block_vertices", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = part_map(u);
            Kokkos::atomic_add(&d_counts(b), 1);
        });
    }

    struct BestBisectConfig {
        u64 penalty = 0xFFFFFFFFFFFFFFFFULL;
        weight_t cut = 0x7FFFFFFF;
        u64 config = 0;

        KOKKOS_INLINE_FUNCTION BestBisectConfig() = default;
    };

    struct BestBisectReducer {
        using reducer = BestBisectReducer;
        using value_type = BestBisectConfig;
        using result_view_type = Kokkos::View<value_type, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        KOKKOS_INLINE_FUNCTION void join(value_type &dst, const value_type &src) const {
            if (src.penalty < dst.penalty) {
                dst = src;
            } else if (src.penalty == dst.penalty) {
                if (src.cut < dst.cut) {
                    dst = src;
                }
            }
        }

        KOKKOS_INLINE_FUNCTION void init(value_type &dst) const {
            dst.penalty = 0xFFFFFFFFFFFFFFFFULL;
            dst.cut = 0x7FFFFFFF;
            dst.config = 0;
        }

        value_type *value;

        KOKKOS_INLINE_FUNCTION BestBisectReducer(value_type &val) : value(&val) {}

        KOKKOS_INLINE_FUNCTION BestBisectReducer(result_view_type view) : value(view.data()) {}

        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }

        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }

        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    template<bool uvw, bool uew, int CHUNK = 16>
    inline void brute_force_bisect_async(
        const Graph &g,
        weight_t lmax_left,
        weight_t lmax_right,
        UnmanagedDevicePartition &partition_map,
        Kokkos::View<BestBisectConfig, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged>> result_view,
        DeviceExecutionSpace &exec_space
    ) {
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
                partition_map(0) = 0;
            });
            return;
        }

        const vertex_t gn = g.n;
        const vertex_t last = gn - 1;
        const u64 num_configs = 1ULL << last;
        const u64 num_chunks = (num_configs + CHUNK - 1) / CHUNK;

        Kokkos::parallel_reduce("brute_force_bisect_reduction", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_chunks), KOKKOS_LAMBDA(const u64 chunk_id, BestBisectConfig &local_best) {
            const u64 begin = chunk_id * CHUNK;
            const u64 end = begin + CHUNK < num_configs ? begin + CHUNK : num_configs;

            u64 gray = begin ^ (begin >> 1);
            weight_t wr = 0;
            for (vertex_t u = 0; u < last; ++u) {
                if ((gray >> u) & 1ULL) {
                    wr += uvw ? 1 : g.weights(u);
                }
            }

            weight_t cut = 0;
            for (u32 e = 0; e < g.m; ++e) {
                const vertex_t u = g.edges_u(e);
                const vertex_t v = g.edges_v(e);
                if (u < v) {
                    const u64 pu = (gray >> u) & 1ULL;
                    const u64 pv = (gray >> v) & 1ULL;
                    if (pu != pv) {
                        cut += uew ? 1 : g.edges_w(e);
                    }
                }
            }

            auto evaluate_current = [&](const u64 config, const weight_t wr_cur, const weight_t cut_cur, BestBisectConfig &best_cur) {
                const weight_t wl = g.g_weight - wr_cur;
                const u64 p_l = wl > lmax_left ? (u64) (wl - lmax_left) : 0;
                const u64 p_r = wr_cur > lmax_right ? (u64) (wr_cur - lmax_right) : 0;
                const u64 penalty = p_l * p_l + p_r * p_r;

                if (penalty < best_cur.penalty || (penalty == best_cur.penalty && cut_cur < best_cur.cut)) {
                    best_cur.penalty = penalty;
                    best_cur.cut = cut_cur;
                    best_cur.config = config;
                }
            };

            evaluate_current(gray, wr, cut, local_best);

            for (u64 i = begin + 1; i < end; ++i) {
                const u64 next_gray = i ^ (i >> 1);
                const u64 diff = gray ^ next_gray;

                #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                const vertex_t flip_u = (vertex_t) __ffsll(diff) - 1;
                #else
                const vertex_t flip_u = (vertex_t) __builtin_ctzll(diff);
                #endif

                const u64 old_part_u = (gray >> flip_u) & 1ULL;
                const u64 new_part_u = old_part_u ^ 1ULL;
                const weight_t wu = uvw ? 1 : g.weights(flip_u);

                if (new_part_u) wr += wu; else wr -= wu;

                for (u32 e = g.neighborhood(flip_u); e < g.neighborhood(flip_u + 1); ++e) {
                    const vertex_t v = g.edges_v(e);
                    const u64 part_v = (gray >> v) & 1ULL;
                    const bool was_cut = old_part_u != part_v;
                    const bool now_cut = new_part_u != part_v;
                    const weight_t ew = uew ? 1 : g.edges_w(e);
                    if (was_cut && !now_cut) cut -= ew; else if (!was_cut && now_cut) cut += ew;
                }
                gray = next_gray;
                evaluate_current(gray, wr, cut, local_best);
            }
        }, BestBisectReducer(result_view));

        Kokkos::parallel_for("apply_best_config", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            partition_map(u) = (partition_t) ((result_view().config >> u) & 1ULL);
        });
    }

    inline void predict_block_distribution(const Mapping &mapping,
                                           const UnmanagedDevicePartition &partition_map,
                                           UnmanagedDeviceVertex &d_counts,
                                           DeviceExecutionSpace &exec_space) {
        Kokkos::deep_copy(exec_space, d_counts, 0);
        Kokkos::parallel_for("predict_block_distribution", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, mapping.old_n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t coarse_b = partition_map(mapping.mapping(u));
            Kokkos::atomic_add(&d_counts(coarse_b), 1);
        });
    }

    struct BatchedSubgraphs {
        u32 S = 0;

        UnmanagedDevicePartition split_blocks;
        UnmanagedDevicePartition block_to_split;

        UnmanagedDeviceVertex sub_n;
        UnmanagedDeviceU32 sub_m;
        UnmanagedDeviceWeight sub_weight;

        UnmanagedDeviceVertex sub_vertex_offsets;
        UnmanagedDeviceU32 sub_edge_offsets;
        UnmanagedDeviceVertex sub_neigh_offsets;

        UnmanagedDeviceVertex sub_write_pos;

        UnmanagedDeviceVertex sub_to_old;
        UnmanagedDeviceVertex old_to_sub;
        UnmanagedDevicePartition sub_vertex_to_split;

        UnmanagedDeviceU32 degrees;
        UnmanagedDeviceU32 neighborhood;

        UnmanagedDeviceVertex edges_u;
        UnmanagedDeviceVertex edges_v;
        UnmanagedDeviceWeight edges_w;
        UnmanagedDeviceWeight weights;

        vertex_t total_sub_n = 0;
        u32 total_sub_m = 0;
        vertex_t total_neigh_n = 0;

        std::vector<vertex_t> h_sub_n;
        std::vector<u32> h_sub_m;
        std::vector<weight_t> h_sub_weight;
        std::vector<vertex_t> h_sub_vertex_offsets;
        std::vector<u32> h_sub_edge_offsets;
        std::vector<vertex_t> h_sub_neigh_offsets;
    };

    inline void free_batched_subgraphs(BatchedSubgraphs &batch, bool uniform_vw, bool uniform_ew, KokkosMemoryStack &mem_stack) {
        if (batch.S == 0) return;
        if (!uniform_vw) pop_front(mem_stack);
        if (!uniform_ew) pop_front(mem_stack);
        pop_front(mem_stack); // edges_v
        pop_front(mem_stack); // edges_u
        pop_front(mem_stack); // neighborhood
        pop_front(mem_stack); // degrees
        pop_front(mem_stack); // sub_vertex_to_split
        pop_front(mem_stack); // old_to_sub
        pop_front(mem_stack); // sub_to_old
        pop_front(mem_stack); // sub_write_pos
        pop_front(mem_stack); // sub_neigh_offsets
        pop_front(mem_stack); // sub_edge_offsets
        pop_front(mem_stack); // sub_vertex_offsets
        pop_front(mem_stack); // sub_weight
        pop_front(mem_stack); // sub_m
        pop_front(mem_stack); // sub_n
        pop_front(mem_stack); // block_to_split
        pop_front(mem_stack); // split_blocks
    }

    inline void extract_block_subgraphs_batched(const Graph &g,
                                                const UnmanagedDevicePartition &partition_map,
                                                partition_t k,
                                                const std::vector<partition_t> &h_blocks_to_split,
                                                BatchedSubgraphs &batch,
                                                KokkosMemoryStack &mem_stack,
                                                DeviceExecutionSpace &exec_space) {
        const u32 S = (u32) h_blocks_to_split.size();
        batch.S = S;

        if (S == 0) {
            batch.total_sub_n = 0;
            batch.total_sub_m = 0;
            batch.total_neigh_n = 0;
            return;
        }

        constexpr partition_t INVALID_SPLIT = std::numeric_limits<partition_t>::max();

        batch.split_blocks = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * S), S);
        batch.block_to_split = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * k), k);
        batch.sub_n = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);
        batch.sub_m = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * S), S);
        batch.sub_weight = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * S), S);
        batch.sub_vertex_offsets = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);
        batch.sub_edge_offsets = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * S), S);
        batch.sub_neigh_offsets = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);
        batch.sub_write_pos = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);

        HostPartition h_split_blocks(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_split_blocks"), S);
        for (u32 s = 0; s < S; ++s) {
            h_split_blocks(s) = h_blocks_to_split[s];
        }
        Kokkos::deep_copy(exec_space, batch.split_blocks, h_split_blocks);

        Kokkos::deep_copy(exec_space, batch.block_to_split, INVALID_SPLIT);
        Kokkos::deep_copy(exec_space, batch.sub_n, 0);
        Kokkos::deep_copy(exec_space, batch.sub_m, 0);
        Kokkos::deep_copy(exec_space, batch.sub_weight, 0);

        auto d_split_blocks = batch.split_blocks;
        auto d_block_to_split = batch.block_to_split;

        Kokkos::parallel_for("batch_build_block_to_split", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, S), KOKKOS_LAMBDA(const u32 s) {
            d_block_to_split(d_split_blocks(s)) = s;
        });

        auto part_map = partition_map;
        auto sub_n = batch.sub_n;
        auto sub_m = batch.sub_m;
        auto sub_weight = batch.sub_weight;

        auto neigh = g.neighborhood;
        auto edges_v = g.edges_v;
        auto weights = g.weights;

        const bool uniform_vw = g.uniform_vertex_weights;

        Kokkos::parallel_for("batch_count_subgraphs", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = part_map(u);
            partition_t s = d_block_to_split(b);

            if (s != INVALID_SPLIT) {
                Kokkos::atomic_add(&sub_n(s), (vertex_t) 1);
                weight_t wu = uniform_vw ? 1 : weights(u);
                Kokkos::atomic_add(&sub_weight(s), wu);

                u32 local_m = 0;
                for (u32 e = neigh(u); e < neigh(u + 1); ++e) {
                    vertex_t v = edges_v(e);
                    if (part_map(v) == b) {
                        local_m++;
                    }
                }
                Kokkos::atomic_add(&sub_m(s), local_m);
            }
        });

        exec_space.fence();

        HostVertex h_sub_n_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_sub_n"), S);
        HostU32 h_sub_m_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_sub_m"), S);
        HostWeight h_sub_weight_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_sub_weight"), S);

        Kokkos::deep_copy(exec_space, h_sub_n_view, batch.sub_n);
        Kokkos::deep_copy(exec_space, h_sub_m_view, batch.sub_m);
        Kokkos::deep_copy(exec_space, h_sub_weight_view, batch.sub_weight);
        exec_space.fence();

        HostVertex h_sub_vertex_offsets_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_sub_vertex_offsets"), S);
        HostU32 h_sub_edge_offsets_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_sub_edge_offsets"), S);
        HostVertex h_sub_neigh_offsets_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_sub_neigh_offsets"), S);

        vertex_t total_sub_n = 0;
        u32 total_sub_m = 0;
        vertex_t total_neigh_n = 0;

        batch.h_sub_n.resize(S);
        batch.h_sub_m.resize(S);
        batch.h_sub_weight.resize(S);
        batch.h_sub_vertex_offsets.resize(S);
        batch.h_sub_edge_offsets.resize(S);
        batch.h_sub_neigh_offsets.resize(S);

        for (u32 s = 0; s < S; ++s) {
            h_sub_vertex_offsets_view(s) = total_sub_n;
            h_sub_edge_offsets_view(s) = total_sub_m;
            h_sub_neigh_offsets_view(s) = total_neigh_n;

            batch.h_sub_n[s] = h_sub_n_view(s);
            batch.h_sub_m[s] = h_sub_m_view(s);
            batch.h_sub_weight[s] = h_sub_weight_view(s);
            batch.h_sub_vertex_offsets[s] = total_sub_n;
            batch.h_sub_edge_offsets[s] = total_sub_m;
            batch.h_sub_neigh_offsets[s] = total_neigh_n;

            total_sub_n += h_sub_n_view(s);
            total_sub_m += h_sub_m_view(s);
            total_neigh_n += h_sub_n_view(s) + 1;
        }

        batch.total_sub_n = total_sub_n;
        batch.total_sub_m = total_sub_m;
        batch.total_neigh_n = total_neigh_n;

        Kokkos::deep_copy(exec_space, batch.sub_vertex_offsets, h_sub_vertex_offsets_view);
        Kokkos::deep_copy(exec_space, batch.sub_edge_offsets, h_sub_edge_offsets_view);
        Kokkos::deep_copy(exec_space, batch.sub_neigh_offsets, h_sub_neigh_offsets_view);

        batch.sub_to_old = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * total_sub_n), total_sub_n);
        batch.old_to_sub = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * g.n), g.n);
        batch.sub_vertex_to_split = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * total_sub_n), total_sub_n);
        batch.degrees = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * total_sub_n), total_sub_n);
        batch.neighborhood = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * total_neigh_n), total_neigh_n);
        batch.edges_u = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * total_sub_m), total_sub_m);
        batch.edges_v = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * total_sub_m), total_sub_m);

        if (!g.uniform_edge_weights) {
            batch.edges_w = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * total_sub_m), total_sub_m);
        }
        if (!g.uniform_vertex_weights) {
            batch.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * total_sub_n), total_sub_n);
        }

        Kokkos::deep_copy(exec_space, batch.sub_write_pos, 0);

        auto sub_vertex_offsets = batch.sub_vertex_offsets;
        auto sub_write_pos = batch.sub_write_pos;
        auto sub_to_old = batch.sub_to_old;
        auto old_to_sub = batch.old_to_sub;
        auto sub_vertex_to_split = batch.sub_vertex_to_split;
        auto batched_weights = batch.weights;

        Kokkos::parallel_for("batch_fill_vertices", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = part_map(u);
            partition_t s = d_block_to_split(b);

            if (s != INVALID_SPLIT) {
                vertex_t local_u = Kokkos::atomic_fetch_add(&sub_write_pos(s), (vertex_t) 1);
                vertex_t global_sub_u = sub_vertex_offsets(s) + local_u;

                sub_to_old(global_sub_u) = u;
                old_to_sub(u) = global_sub_u;
                sub_vertex_to_split(global_sub_u) = s;

                if (!uniform_vw) {
                    batched_weights(global_sub_u) = weights(u);
                }
            }
        });

        auto degrees = batch.degrees;

        Kokkos::parallel_for("batch_count_vertex_degrees", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, total_sub_n), KOKKOS_LAMBDA(const vertex_t global_sub_u) {
            vertex_t old_u = sub_to_old(global_sub_u);
            partition_t old_block = part_map(old_u);

            u32 deg = 0;
            for (u32 e = neigh(old_u); e < neigh(old_u + 1); ++e) {
                vertex_t old_v = edges_v(e);
                if (part_map(old_v) == old_block) {
                    deg++;
                }
            }
            degrees(global_sub_u) = deg;
        });

        exec_space.fence();

        HostU32 h_degrees(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_degrees"), total_sub_n);
        HostU32 h_neighborhood(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_batch_neighborhood"), total_neigh_n);

        Kokkos::deep_copy(exec_space, h_degrees, batch.degrees);
        exec_space.fence();

        for (u32 s = 0; s < S; ++s) {
            vertex_t voff = batch.h_sub_vertex_offsets[s];
            vertex_t noff = batch.h_sub_neigh_offsets[s];
            vertex_t ns = batch.h_sub_n[s];

            u32 prefix = 0;
            h_neighborhood(noff) = 0;

            for (vertex_t local_u = 0; local_u < ns; ++local_u) {
                prefix += h_degrees(voff + local_u);
                h_neighborhood(noff + local_u + 1) = prefix;
            }
        }

        Kokkos::deep_copy(exec_space, batch.neighborhood, h_neighborhood);

        auto sub_neigh_offsets = batch.sub_neigh_offsets;
        auto sub_edge_offsets = batch.sub_edge_offsets;
        auto neighborhood = batch.neighborhood;
        auto batched_edges_u = batch.edges_u;
        auto batched_edges_v = batch.edges_v;
        auto batched_edges_w = batch.edges_w;
        auto g_edges_w = g.edges_w;

        const bool uniform_ew = g.uniform_edge_weights;

        Kokkos::parallel_for("batch_fill_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, total_sub_n), KOKKOS_LAMBDA(const vertex_t global_sub_u) {
            partition_t s = sub_vertex_to_split(global_sub_u);

            vertex_t old_u = sub_to_old(global_sub_u);
            partition_t old_block = part_map(old_u);

            vertex_t voff = sub_vertex_offsets(s);
            vertex_t noff = sub_neigh_offsets(s);
            u32 eoff = sub_edge_offsets(s);

            vertex_t local_u = global_sub_u - voff;

            u32 local_pos = neighborhood(noff + local_u);
            u32 global_pos = eoff + local_pos;

            for (u32 e = neigh(old_u); e < neigh(old_u + 1); ++e) {
                vertex_t old_v = edges_v(e);

                if (part_map(old_v) == old_block) {
                    vertex_t global_sub_v = old_to_sub(old_v);
                    vertex_t local_v = global_sub_v - voff;

                    batched_edges_u(global_pos) = local_u;
                    batched_edges_v(global_pos) = local_v;

                    if (!uniform_ew) {
                        batched_edges_w(global_pos) = g_edges_w(e);
                    }
                    global_pos++;
                }
            }
        });

        exec_space.fence();
    }

    inline Graph make_batched_subgraph_view(const BatchedSubgraphs &batch,
                                            u32 s,
                                            bool uniform_vertex_weights,
                                            bool uniform_edge_weights) {
        Graph sub_g;

        sub_g.n = batch.h_sub_n[s];
        sub_g.m = batch.h_sub_m[s];
        sub_g.g_weight = batch.h_sub_weight[s];

        sub_g.uniform_vertex_weights = uniform_vertex_weights;
        sub_g.uniform_edge_weights = uniform_edge_weights;

        vertex_t noff = batch.h_sub_neigh_offsets[s];
        vertex_t voff = batch.h_sub_vertex_offsets[s];
        u32 eoff = batch.h_sub_edge_offsets[s];

        sub_g.neighborhood = Kokkos::subview(batch.neighborhood, std::make_pair(noff, noff + sub_g.n + 1));
        sub_g.edges_u = Kokkos::subview(batch.edges_u, std::make_pair(eoff, eoff + sub_g.m));
        sub_g.edges_v = Kokkos::subview(batch.edges_v, std::make_pair(eoff, eoff + sub_g.m));

        if (!uniform_edge_weights) {
            sub_g.edges_w = Kokkos::subview(batch.edges_w, std::make_pair(eoff, eoff + sub_g.m));
        }
        if (!uniform_vertex_weights) {
            sub_g.weights = Kokkos::subview(batch.weights, std::make_pair(voff, voff + sub_g.n));
        }

        return sub_g;
    }

    inline void update_partition_from_batched_subparts(const BatchedSubgraphs &batch,
                                                       const UnmanagedDevicePartition &sub_part,
                                                       const UnmanagedDevicePartition &split_rids,
                                                       UnmanagedDevicePartition &partition_map,
                                                       DeviceExecutionSpace &exec_space) {
        auto sub_to_old = batch.sub_to_old;
        auto sub_vertex_to_split = batch.sub_vertex_to_split;

        Kokkos::parallel_for("batch_update_partition_from_subparts", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, batch.total_sub_n), KOKKOS_LAMBDA(const vertex_t global_sub_u) {
            if (sub_part(global_sub_u) == 1) {
                vertex_t old_u = sub_to_old(global_sub_u);
                partition_t s = sub_vertex_to_split(global_sub_u);
                partition_t rid = split_rids(s);

                partition_map(old_u) = rid;
            }
        });
    }

    inline void gpu_progressive_partition(Graph &g,
                                          const std::vector<partition_t> &hierarchy,
                                          partition_t k,
                                          f64 imbalance,
                                          u64 seed,
                                          u32 threshold,
                                          Partition &partition,
                                          KokkosMemoryStack &mem_stack,
                                          DeviceExecutionSpace &exec_space) {
        // --- 1. COARSENING ---
        std::vector<Graph> graphs = {g};
        std::vector<Mapping> mappings;

        weight_t lmax_global = (weight_t) std::ceil((1.0 + imbalance) * (f64) g.g_weight / (f64) k);

        while (graphs.back().n > threshold) {
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "coarsening");
                mappings.push_back(two_hop_matcher_get_mapping<false, false>(graphs.back(), partition, lmax_global, mem_stack, exec_space));
            }
            //
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "graph_contraction");
                graphs.push_back(from_Graph_Mapping<false, false>(graphs.back(), mappings.back(), mem_stack, exec_space));
            }
            //
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "partition_contraction");
                contract(partition, mappings.back(), exec_space);
            }

            std::cout << "  [Progressive Coarsening] Level " << graphs.size() - 1 << ": n=" << graphs.back().n << std::endl;
        }

        std::cout << "  [Progressive Bisection] Starting loop at level " << graphs.size() - 1 << " (n=" << graphs.back().n << ")" << std::endl;

        UnmanagedDeviceVertex vertex_count = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);
        HostVertex h_counts(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_counts"), k);
        HostWeight h_weights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_weights"), k);

        // --- Hierarchy and Metadata Setup ---
        u32 num_levels = (u32) hierarchy.size();
        std::vector<partition_t> strides(num_levels);
        strides[0] = 1;
        for (u32 i = 1; i < num_levels; ++i) strides[i] = strides[i - 1] * hierarchy[i - 1];

        struct BlockMeta {
            u32 lvl = 0;
            partition_t fact = 0;
            vertex_t n = 0;
            weight_t weight = 0;
        };
        std::vector<BlockMeta> b_meta(k);
        b_meta[0].lvl = num_levels - 1;
        b_meta[0].fact = hierarchy[num_levels - 1];
        b_meta[0].n = graphs.back().n;
        b_meta[0].weight = graphs.back().g_weight;

        auto normalize_block_meta = [&](partition_t b) {
            while (b_meta[b].fact <= 1 && b_meta[b].lvl > 0) {
                b_meta[b].lvl--;
                b_meta[b].fact = hierarchy[b_meta[b].lvl];
            }
        };

        f64 avg_core_weight = (f64) g.g_weight / (f64) k;
        int cl = (int) graphs.size() - 1;

        // --- 2. PROGRESSIVE BISECTION / UNCOARSENING ---
        Kokkos::deep_copy(exec_space, partition.map, 0);

        {
            ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "initial_count");

            recalculate_weights<false>(partition, graphs.back(), exec_space);
            count_block_vertices(partition, graphs.back(), vertex_count, exec_space);
            Kokkos::deep_copy(exec_space, h_counts, vertex_count);
            Kokkos::deep_copy(exec_space, h_weights, partition.bweights);
            exec_space.fence();
        }

        for (partition_t b = 0; b < k; ++b) {
            b_meta[b].n = h_counts(b);
            b_meta[b].weight = h_weights(b);
        }

        while (true) {
            Graph &curr_g = graphs[cl];

            if (!mappings.empty()) {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "predict_block_distribution");

                predict_block_distribution(mappings.back(), partition.map, vertex_count, exec_space);
            } else {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "count_block_vertices");

                count_block_vertices(partition, curr_g, vertex_count, exec_space);
            }

            Kokkos::deep_copy(exec_space, h_counts, vertex_count);
            exec_space.fence();

            bool split_occurred;
            do {
                split_occurred = false;

                std::vector<partition_t> blocks_to_split;

                for (partition_t b = 0; b < k; ++b) {
                    if (h_counts(b) == 0) continue;

                    if (b_meta[b].fact > 1) {
                        if (h_counts(b) > threshold || cl == 0) {
                            blocks_to_split.push_back(b);
                        }
                    }
                }

                std::cout << "need to split " << blocks_to_split.size() << " blocks" << std::endl;

                if (!blocks_to_split.empty()) {
                    split_occurred = true;

                    BatchedSubgraphs batch;
                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "extract_block_subgraphs_batched");
                        extract_block_subgraphs_batched(curr_g, partition.map, k, blocks_to_split, batch, mem_stack, exec_space);
                        KOKKOS_PROFILE_FENCE(exec_space);
                    }

                    UnmanagedDevicePartition sub_part_batch = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * batch.total_sub_n), batch.total_sub_n);
                    UnmanagedDevicePartition split_rids = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * batch.S), batch.S);
                    
                    using result_view_t = Kokkos::View<BestBisectConfig*, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                    result_view_t results_batch((BestBisectConfig*)get_chunk_front(mem_stack, sizeof(BestBisectConfig) * batch.S), batch.S);
                    
                    HostPartition h_split_rids(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_split_rids"), batch.S);

                    u32 n_instances = std::min(batch.S, 16u);
                    std::vector<DeviceExecutionSpace> instances = Kokkos::Experimental::partition_space(exec_space, std::vector<int>(n_instances, 1));

                    for (u32 s = 0; s < batch.S; ++s) {
                        partition_t b = blocks_to_split[s];
                        u32 lvl = b_meta[b].lvl;
                        partition_t f = b_meta[b].fact;

                        partition_t rp = 1 << (u32) std::log2(f - 1);
                        partition_t lp = f - rp;
                        partition_t stride = strides[lvl];
                        partition_t rid = b + lp * stride;

                        h_split_rids(s) = rid;

                        weight_t lmax_l = (weight_t) std::ceil((1.0 + imbalance) * avg_core_weight * (lp * stride));
                        weight_t lmax_r = (weight_t) std::ceil((1.0 + imbalance) * avg_core_weight * (rp * stride));

                        std::cout << "      [Lvl " << lvl << "][Block " << b << "]" << " n=" << batch.h_sub_n[s] << " f=" << f << " stride=" << stride << " lp=" << lp << " rp=" << rp << " rid=" << rid << " -> Splitting" << std::endl;

                        Graph sub_g = make_batched_subgraph_view(batch, s, curr_g.uniform_vertex_weights, curr_g.uniform_edge_weights);
                        UnmanagedDevicePartition sub_part = Kokkos::subview(sub_part_batch, std::make_pair(batch.h_sub_vertex_offsets[s], batch.h_sub_vertex_offsets[s] + batch.h_sub_n[s]));
                        auto result_s = Kokkos::subview(results_batch, s);

                        {
                            ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "brute_force_bisect");
                            if (sub_g.uniform_vertex_weights && sub_g.uniform_edge_weights) {
                                brute_force_bisect_async<true, true, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            } else if (!sub_g.uniform_vertex_weights && sub_g.uniform_edge_weights) {
                                brute_force_bisect_async<false, true, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            } else if (sub_g.uniform_vertex_weights && !sub_g.uniform_edge_weights) {
                                brute_force_bisect_async<true, false, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            } else {
                                brute_force_bisect_async<false, false, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            }
                        }

                        b_meta[b].fact = lp;
                        b_meta[rid].lvl = lvl;
                        b_meta[rid].fact = rp;
                        normalize_block_meta(b);
                        normalize_block_meta(rid);
                    }

                    for (auto &st : instances) st.fence();

                    Kokkos::deep_copy(exec_space, split_rids, h_split_rids);

                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "update_partition");
                        update_partition_from_batched_subparts(batch, sub_part_batch, split_rids, partition.map, exec_space);
                        KOKKOS_PROFILE_FENCE(exec_space);
                    }

                    pop_front(mem_stack); // results_batch
                    pop_front(mem_stack); // split_rids
                    pop_front(mem_stack); // sub_part_batch
                    free_batched_subgraphs(batch, curr_g.uniform_vertex_weights, curr_g.uniform_edge_weights, mem_stack);

                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "recalc_stats");

                        recalculate_weights<false>(partition, curr_g, exec_space);
                        count_block_vertices(partition, curr_g, vertex_count, exec_space);

                        Kokkos::deep_copy(exec_space, h_counts, vertex_count);
                        Kokkos::deep_copy(exec_space, h_weights, partition.bweights);
                        KOKKOS_PROFILE_FENCE(exec_space);
                        exec_space.fence();
                    }

                    for (partition_t b = 0; b < k; ++b) {
                        b_meta[b].n = h_counts(b);
                        b_meta[b].weight = h_weights(b);
                    }
                }
            } while (split_occurred);

            if (mappings.empty()) {
                break;
            }

            cl--;

            std::cout << "  [Progressive Bisection] uncontract " << cl << " (n=" << graphs[cl].n << ")" << std::endl;

            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "uncontract");

                uncontract(partition, mappings.back(), exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);
            }
            //
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "recalc_stats");

                recalculate_weights<false>(partition, graphs[cl], exec_space);
                count_block_vertices(partition, graphs[cl], vertex_count, exec_space);

                Kokkos::deep_copy(exec_space, h_counts, vertex_count);
                Kokkos::deep_copy(exec_space, h_weights, partition.bweights);
                KOKKOS_PROFILE_FENCE(exec_space);
                exec_space.fence();
            }

            for (partition_t b = 0; b < k; ++b) {
                b_meta[b].n = h_counts(b);
                b_meta[b].weight = h_weights(b);
            }

            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();

            exec_space.fence();
        }

        // Deallocate buffers
        pop_front(mem_stack); // vertex_count
    }
}

#endif //GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H
