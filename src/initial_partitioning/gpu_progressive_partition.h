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
#include "../coarsening/independent_edge_set.h"
#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/custom_reductions.h"

namespace GPU_HeiPa {
    struct Totals {
        vertex_t v = 0;
        u32 e = 0;
        vertex_t n = 0;

        KOKKOS_INLINE_FUNCTION void operator+=(const Totals &rhs) {
            v += rhs.v;
            e += rhs.e;
            n += rhs.n;
        }
    };

    struct OffsetTriple {
        vertex_t v = 0;
        u32 e = 0;
        vertex_t n = 0;

        KOKKOS_INLINE_FUNCTION void operator+=(const OffsetTriple &rhs) {
            v += rhs.v;
            e += rhs.e;
            n += rhs.n;
        }
    };

    struct BestBisectConfig {
        u64 penalty = 0xFFFFFFFFFFFFFFFFULL;
        weight_t cut = 0x7FFFFFFF;
        u64 config = 0;

        KOKKOS_INLINE_FUNCTION BestBisectConfig() = default;
    };

    struct BestBisectReducer {
        using reducer = BestBisectReducer;
        using value_type = BestBisectConfig;
        using result_view_type = Kokkos::View<value_type, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

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

        KOKKOS_INLINE_FUNCTION BestBisectReducer(value_type &val) : value(&val) {
        }

        KOKKOS_INLINE_FUNCTION BestBisectReducer(result_view_type view) : value(view.data()) {
        }

        KOKKOS_INLINE_FUNCTION value_type &reference() const { return *value; }

        KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(value); }

        KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
    };

    template<bool uvw, bool uew, int CHUNK>
    inline void brute_force_bisect_async(const Graph &g,
                                         weight_t lmax_left,
                                         weight_t lmax_right,
                                         UnmanagedDevicePartition &partition_map,
                                         Kokkos::View<BestBisectConfig, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> > result_view,
                                         DeviceExecutionSpace &exec_space) {
        if (g.n == 0) return;
        if (g.n == 1) {
            Kokkos::parallel_for("bisect1", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
                partition_map(0) = 0;
            });
            return;
        }

        const vertex_t gn = g.n;
        const u32 gm = g.m;
        const vertex_t last = gn - 1;
        const u64 num_configs = 1ULL << last;

        const int team_size = 256;
        const u64 configs_per_team = (u64) team_size * CHUNK;
        const u32 num_teams = (u32) ((num_configs + configs_per_team - 1) / configs_per_team);

        size_t shmem_size = (gn + 1) * sizeof(u32); // neighborhood
        shmem_size += gm * sizeof(vertex_t);       // edges_u
        shmem_size += gm * sizeof(vertex_t);       // edges_v
        if (!uvw) shmem_size += gn * sizeof(weight_t); // weights
        if (!uew) shmem_size += gm * sizeof(weight_t); // edges_w

        auto policy = Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, num_teams, team_size)
                .set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_reduce("brute_force_bisect_reduction", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team, BestBisectConfig &team_best) {
            typedef Kokkos::View<u32 *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchU32;
            typedef Kokkos::View<vertex_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchVertex;
            typedef Kokkos::View<weight_t *, DeviceExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchWeight;

            ScratchU32 s_neigh(team.team_scratch(0), gn + 1);
            ScratchVertex s_edges_u(team.team_scratch(0), gm);
            ScratchVertex s_edges_v(team.team_scratch(0), gm);
            ScratchWeight s_weights;
            if (!uvw) s_weights = ScratchWeight(team.team_scratch(0), gn);
            ScratchWeight s_edges_w;
            if (!uew) s_edges_w = ScratchWeight(team.team_scratch(0), gm);

            // Load graph data into shared memory
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn + 1), [&](const u32 i) {
                s_neigh(i) = g.neighborhood(i);
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gm), [&](const u32 i) {
                s_edges_u(i) = g.edges_u(i);
                s_edges_v(i) = g.edges_v(i);
            });
            if (!uvw) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gn), [&](const vertex_t i) {
                    s_weights(i) = g.weights(i);
                });
            }
            if (!uew) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, gm), [&](const u32 i) {
                    s_edges_w(i) = g.edges_w(i);
                });
            }
            team.team_barrier();

            BestBisectConfig best_in_team;
            BestBisectReducer reducer(best_in_team);
            reducer.init(best_in_team);

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, team_size), [&](const int tid, BestBisectConfig &local_best) {
                const u64 chunk_id = (u64) team.league_rank() * team_size + tid;
                const u64 begin = chunk_id * CHUNK;
                if (begin >= num_configs) return;
                const u64 end = begin + CHUNK < num_configs ? begin + CHUNK : num_configs;

                u64 gray = begin ^ (begin >> 1);
                weight_t wr = 0;
                for (vertex_t u = 0; u < last; ++u) {
                    if ((gray >> u) & 1ULL) {
                        wr += uvw ? 1 : s_weights(u);
                    }
                }

                weight_t cut = 0;
                for (u32 e = 0; e < gm; ++e) {
                    const vertex_t u = s_edges_u(e);
                    const vertex_t v = s_edges_v(e);
                    if (u < v) {
                        const u64 pu = (gray >> u) & 1ULL;
                        const u64 pv = (gray >> v) & 1ULL;
                        if (pu != pv) {
                            cut += uew ? 1 : s_edges_w(e);
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

                for (u64 i = begin + 1; i < end; i++) {
                    const u64 next_gray = i ^ (i >> 1);
                    const u64 diff = gray ^ next_gray;

                    #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                    const vertex_t flip_u = (vertex_t) __ffsll(diff) - 1;
                    #else
                    const vertex_t flip_u = (vertex_t) __builtin_ctzll(diff);
                    #endif

                    const u64 old_part_u = (gray >> flip_u) & 1ULL;
                    const u64 new_part_u = old_part_u ^ 1ULL;
                    const weight_t wu = uvw ? 1 : s_weights(flip_u);

                    if (new_part_u) wr += wu;
                    else wr -= wu;

                    for (u32 e = s_neigh(flip_u); e < s_neigh(flip_u + 1); ++e) {
                        const vertex_t v = s_edges_v(e);
                        const u64 part_v = (gray >> v) & 1ULL;
                        const bool was_cut = old_part_u != part_v;
                        const bool now_cut = new_part_u != part_v;
                        const weight_t ew = uew ? 1 : s_edges_w(e);
                        if (was_cut && !now_cut) cut -= ew;
                        else if (!was_cut && now_cut) cut += ew;
                    }
                    gray = next_gray;
                    evaluate_current(gray, wr, cut, local_best);
                }
            }, reducer);

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                BestBisectReducer(team_best).join(team_best, best_in_team);
            });
        }, BestBisectReducer(result_view));

        Kokkos::parallel_for("apply_best_config", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            partition_map(u) = (partition_t) ((result_view().config >> u) & 1ULL);
        });
    }

    template<bool uvw>
    inline void recalculate_weights_and_counts(Partition &partition,
                                               const Graph &g,
                                               UnmanagedDeviceVertex &d_counts,
                                               DeviceExecutionSpace &exec_space) {
        Kokkos::deep_copy(exec_space, partition.bweights, 0);
        Kokkos::deep_copy(exec_space, d_counts, 0);
        auto part_map = partition.map;
        auto bweights = partition.bweights;
        auto weights = g.weights;
        Kokkos::parallel_for("recalculate_weights_and_counts", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = part_map(u);
            Kokkos::atomic_add(&d_counts(b), 1);
            Kokkos::atomic_add(&bweights(b), uvw ? 1 : weights(u));
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
        std::vector<partition_t> h_rid;
        std::vector<u32> h_lmax_l;
        std::vector<u32> h_lmax_r;

        UnmanagedDeviceU32 metadata;
        u32 rid_offset = 0;
        u32 totals_offset = 0;
    };

    inline void free_batched_subgraphs(BatchedSubgraphs &batch, bool uniform_vw, bool uniform_ew, KokkosMemoryStack &mem_stack) {
        if (batch.metadata.extent(0) == 0) return;
        if (batch.total_sub_n > 0) {
            if (!uniform_vw) pop_front(mem_stack);
            if (!uniform_ew) pop_front(mem_stack);
            pop_front(mem_stack); // edges_v
            pop_front(mem_stack); // edges_u
            pop_front(mem_stack); // neighborhood
            pop_front(mem_stack); // degrees
            pop_front(mem_stack); // sub_vertex_to_split
            pop_front(mem_stack); // old_to_sub
            pop_front(mem_stack); // sub_to_old
        }
        pop_front(mem_stack); // sub_write_pos
        pop_front(mem_stack); // sub_neigh_offsets
        pop_front(mem_stack); // sub_edge_offsets
        pop_front(mem_stack); // sub_vertex_offsets
        pop_front(mem_stack); // sub_weight
        pop_front(mem_stack); // sub_m
        pop_front(mem_stack); // sub_n
        pop_front(mem_stack); // block_to_split
        pop_front(mem_stack); // split_blocks
        pop_front(mem_stack); // metadata
    }

    inline u32 extract_block_subgraphs_batched(const Graph &g,
                                               const UnmanagedDevicePartition &partition_map,
                                               const partition_t k,
                                               const UnmanagedDeviceU32 &combined_stats,
                                               const DeviceU32 &d_block_fact,
                                               const DeviceU32 &d_block_lvl,
                                               const DeviceU32 &d_strides,
                                               const DeviceU32 &d_hierarchy,
                                               const u32 threshold,
                                               const u32 cl,
                                               BatchedSubgraphs &batch,
                                               const f64 imbalance,
                                               const f64 avg_core_weight,
                                               KokkosMemoryStack &mem_stack,
                                               DeviceExecutionSpace &exec_space) {
        constexpr partition_t INVALID_SPLIT = std::numeric_limits<partition_t>::max();

        // 1. Find split candidates and assign them to split indices 0..S-1
        batch.split_blocks = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * k), k);
        batch.block_to_split = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * k), k);
        Kokkos::deep_copy(exec_space, batch.block_to_split, INVALID_SPLIT);

        // Layout: [n(k), m(k), weight(k), v_off(k), e_off(k), n_off(k), rid(k), lmax_l(k), lmax_r(k), totals(3), S(1)]
        batch.metadata = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (k * 9 + 4)), k * 9 + 4);
        batch.rid_offset = 6 * k;
        const u32 lmax_l_off = 7 * k;
        const u32 lmax_r_off = 8 * k;
        batch.totals_offset = 9 * k;
        const u32 s_off = 9 * k + 3;
        auto d_metadata = batch.metadata;
        auto d_split_blocks = batch.split_blocks;
        auto d_block_to_split = batch.block_to_split;

        Kokkos::parallel_scan("find_split_candidates", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const u32 b, u32 &update, const bool final) {
            const u32 n = combined_stats(b);
            const u32 fact = d_block_fact(b);
            bool splittable = (n > 0) && (fact > 1) && ((n > threshold) || (cl == 0));
            if (splittable) {
                if (final) {
                    d_split_blocks(update) = b;
                    d_block_to_split(b) = update;
                }
                update++;
            }
            if (final && b == k - 1) d_metadata(s_off) = update;
        });

        batch.sub_n = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);
        batch.sub_m = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * k), k);
        batch.sub_weight = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * k), k);
        batch.sub_vertex_offsets = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);
        batch.sub_edge_offsets = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * k), k);
        batch.sub_neigh_offsets = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);
        batch.sub_write_pos = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);

        Kokkos::deep_copy(exec_space, batch.sub_n, 0);
        Kokkos::deep_copy(exec_space, batch.sub_m, 0);
        Kokkos::deep_copy(exec_space, batch.sub_weight, 0);

        auto sub_n = batch.sub_n;
        auto sub_m = batch.sub_m;
        auto sub_weight = batch.sub_weight;

        auto neigh = g.neighborhood;
        auto edges_v = g.edges_v;
        auto weights = g.weights;
        const bool uniform_vw = g.uniform_vertex_weights;

        // 2. Count vertices, internal edges, and weights for each split candidate
        Kokkos::parallel_for("batch_count_subgraphs", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, g.n), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = partition_map(u);
            partition_t s = d_block_to_split(b);

            if (s != INVALID_SPLIT) {
                Kokkos::atomic_add(&sub_n(s), (vertex_t) 1);
                weight_t wu = uniform_vw ? 1 : weights(u);
                Kokkos::atomic_add(&sub_weight(s), wu);

                u32 local_m = 0;
                for (u32 e = neigh(u); e < neigh(u + 1); ++e) {
                    vertex_t v = edges_v(e);
                    if (partition_map(v) == b) {
                        local_m++;
                    }
                }
                Kokkos::atomic_add(&sub_m(s), local_m);
            }
        });

        // 3. Compute offsets and further metadata
        Totals totals;
        Kokkos::parallel_scan("batch_compute_metadata", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const u32 s, Totals &update, const bool final) {
            const u32 S = d_metadata(s_off);
            if (s < S) {
                const vertex_t ns = sub_n(s);
                const u32 ms = sub_m(s);
                const weight_t ws = sub_weight(s);

                if (final) {
                    d_metadata(s) = ns;
                    d_metadata(k + s) = ms;
                    d_metadata(2 * k + s) = static_cast<u32>(ws);
                    d_metadata(3 * k + s) = update.v;
                    d_metadata(4 * k + s) = update.e;
                    d_metadata(5 * k + s) = update.n;

                    partition_t b = d_split_blocks(s);
                    u32 lvl = d_block_lvl(b);
                    partition_t f = d_block_fact(b);
                    partition_t rp = 1 << (u32) floor(log2(static_cast<float>(f - 1)));
                    partition_t lp = f - rp;
                    partition_t stride = d_strides(lvl);

                    d_metadata(6 * k + s) = b + lp * stride;
                    d_metadata(lmax_l_off + s) = static_cast<u32>(ceil((1.0 + imbalance) * avg_core_weight * (lp * stride)));
                    d_metadata(lmax_r_off + s) = static_cast<u32>(ceil((1.0 + imbalance) * avg_core_weight * (rp * stride)));
                }
                update.v += ns;
                update.e += ms;
                update.n += ns + 1;
            }
            if (final && s == k - 1) {
                d_metadata(batch.totals_offset) = update.v;
                d_metadata(batch.totals_offset + 1) = update.e;
                d_metadata(batch.totals_offset + 2) = update.n;
            }
        }, totals);

        // 4. Download metadata to host (this is the ONLY sync point in this function)
        HostU32 h_full_metadata(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_full_metadata"), batch.metadata.extent(0));
        Kokkos::deep_copy(exec_space, h_full_metadata, batch.metadata);
        exec_space.fence();

        batch.S = h_full_metadata(s_off);
        const u32 S = batch.S;
        if (S == 0) return 0;

        batch.total_sub_n = h_full_metadata(batch.totals_offset);
        batch.total_sub_m = h_full_metadata(batch.totals_offset + 1);
        batch.total_neigh_n = h_full_metadata(batch.totals_offset + 2);

        // Store host vectors
        batch.h_sub_n.assign(h_full_metadata.data(), h_full_metadata.data() + S);
        batch.h_sub_m.assign(h_full_metadata.data() + k, h_full_metadata.data() + k + S);
        batch.h_sub_weight.resize(S);
        for (u32 s = 0; s < S; ++s) batch.h_sub_weight[s] = static_cast<weight_t>(h_full_metadata(2 * k + s));
        batch.h_sub_vertex_offsets.assign(h_full_metadata.data() + 3 * k, h_full_metadata.data() + 3 * k + S);
        batch.h_sub_edge_offsets.assign(h_full_metadata.data() + 4 * k, h_full_metadata.data() + 4 * k + S);
        batch.h_sub_neigh_offsets.assign(h_full_metadata.data() + 5 * k, h_full_metadata.data() + 5 * k + S);
        batch.h_rid.assign(h_full_metadata.data() + 6 * k, h_full_metadata.data() + 6 * k + S);
        batch.h_lmax_l.assign(h_full_metadata.data() + lmax_l_off, h_full_metadata.data() + lmax_l_off + S);
        batch.h_lmax_r.assign(h_full_metadata.data() + lmax_r_off, h_full_metadata.data() + lmax_r_off + S);

        // We still need the structural offsets on the device for kernels
        Kokkos::parallel_for("fill_device_offsets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, S), KOKKOS_LAMBDA(const u32 s) {
            batch.sub_vertex_offsets(s) = d_metadata(3 * k + s);
            batch.sub_edge_offsets(s) = d_metadata(4 * k + s);
            batch.sub_neigh_offsets(s) = d_metadata(5 * k + s);
        });

        const vertex_t total_sub_n = batch.total_sub_n;
        const u32 total_sub_m = batch.total_sub_m;
        const vertex_t total_neigh_n = batch.total_neigh_n;

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
            partition_t b = partition_map(u);
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
            partition_t old_block = partition_map(old_u);

            u32 deg = 0;
            for (u32 e = neigh(old_u); e < neigh(old_u + 1); ++e) {
                vertex_t old_v = edges_v(e);
                if (partition_map(old_v) == old_block) {
                    deg++;
                }
            }
            degrees(global_sub_u) = deg;
        });

        auto neighborhood = batch.neighborhood;
        auto sub_neigh_offsets = batch.sub_neigh_offsets;
        auto d_sub_n = batch.sub_n;

        // Construct neighborhood on device
        Kokkos::parallel_for("batch_build_neighborhood", Kokkos::TeamPolicy<DeviceExecutionSpace>(exec_space, S, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DeviceExecutionSpace>::member_type &team) {
            const u32 s = team.league_rank();
            const vertex_t voff = sub_vertex_offsets(s);
            const vertex_t noff = sub_neigh_offsets(s);
            const vertex_t ns = d_sub_n(s);

            // Set the first entry to 0
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                neighborhood(noff) = 0;
            });

            // Exclusive prefix sum over degrees for this subgraph
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, ns), [=](const vertex_t i, u32 &update, const bool final) {
                const u32 d = degrees(voff + i);
                if (final) {
                    neighborhood(noff + i + 1) = update + d;
                }
                update += d;
            });
        });

        auto sub_edge_offsets = batch.sub_edge_offsets;
        auto batched_edges_u = batch.edges_u;
        auto batched_edges_v = batch.edges_v;
        auto batched_edges_w = batch.edges_w;
        auto g_edges_w = g.edges_w;

        const bool uniform_ew = g.uniform_edge_weights;

        Kokkos::parallel_for("batch_fill_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, total_sub_n), KOKKOS_LAMBDA(const vertex_t global_sub_u) {
            partition_t s = sub_vertex_to_split(global_sub_u);

            vertex_t old_u = sub_to_old(global_sub_u);
            partition_t old_block = partition_map(old_u);

            vertex_t voff = sub_vertex_offsets(s);
            vertex_t noff = sub_neigh_offsets(s);
            u32 eoff = sub_edge_offsets(s);

            vertex_t local_u = global_sub_u - voff;

            u32 local_pos = neighborhood(noff + local_u);
            u32 global_pos = eoff + local_pos;

            for (u32 e = neigh(old_u); e < neigh(old_u + 1); ++e) {
                vertex_t old_v = edges_v(e);

                if (partition_map(old_v) == old_block) {
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
        return S;
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
                                                       const UnmanagedDeviceU32 &d_metadata,
                                                       const partition_t k,
                                                       UnmanagedDevicePartition &partition_map,
                                                       DeviceExecutionSpace &exec_space) {
        auto sub_to_old = batch.sub_to_old;
        auto sub_vertex_to_split = batch.sub_vertex_to_split;

        Kokkos::parallel_for("batch_update_partition_from_subparts", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, batch.total_sub_n), KOKKOS_LAMBDA(const vertex_t global_sub_u) {
            if (sub_part(global_sub_u) == 1) {
                vertex_t old_u = sub_to_old(global_sub_u);
                partition_t s = sub_vertex_to_split(global_sub_u);
                partition_t rid = d_metadata(6 * k + s);

                partition_map(old_u) = rid;
            }
        });
    }

    inline void update_hierarchy(const u32 S,
                                 const UnmanagedDevicePartition &d_split_blocks,
                                 const UnmanagedDeviceU32 &d_metadata,
                                 const partition_t k,
                                 DeviceU32 &d_block_lvl,
                                 DeviceU32 &d_block_fact,
                                 const DeviceU32 &d_hierarchy,
                                 DeviceExecutionSpace &exec_space) {
        Kokkos::parallel_for("update_hierarchy", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, S), KOKKOS_LAMBDA(const u32 s) {
            partition_t b = d_split_blocks(s);
            partition_t rid = d_metadata(6 * k + s);

            u32 lvl = d_block_lvl(b);
            partition_t f = d_block_fact(b);
            partition_t rp = 1 << (u32) floor(log2(static_cast<float>(f - 1)));
            partition_t lp = f - rp;

            // Update parent and child metadata
            d_block_fact(b) = lp;
            d_block_lvl(rid) = lvl;
            d_block_fact(rid) = rp;

            // Normalize b
            while (d_block_fact(b) <= 1 && d_block_lvl(b) > 0) {
                d_block_lvl(b)--;
                d_block_fact(b) = d_hierarchy(d_block_lvl(b));
            }
            // Normalize rid
            while (d_block_fact(rid) <= 1 && d_block_lvl(rid) > 0) {
                d_block_lvl(rid)--;
                d_block_fact(rid) = d_hierarchy(d_block_lvl(rid));
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
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "graph_contraction");
                graphs.push_back(from_Graph_Mapping<false, false>(graphs.back(), mappings.back(), mem_stack, exec_space));
            }
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "partition_contraction");
                contract(partition, mappings.back(), exec_space);
            }
        }

        UnmanagedDeviceVertex vertex_count = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);
        UnmanagedDeviceU32 combined_stats = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * 2 * k), 2 * k);

        // --- Hierarchy and Metadata Setup ---
        u32 num_levels = (u32) hierarchy.size();
        std::vector<partition_t> strides(num_levels);
        strides[0] = 1;
        for (u32 i = 1; i < num_levels; ++i) strides[i] = strides[i - 1] * hierarchy[i - 1];

        // Move strides and hierarchy to device
        DeviceU32 d_strides("d_strides", num_levels);
        Kokkos::deep_copy(exec_space, d_strides, Kokkos::View<partition_t *, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(strides.data(), num_levels));

        DeviceU32 d_hierarchy("d_hierarchy", num_levels);
        Kokkos::deep_copy(exec_space, d_hierarchy, Kokkos::View<const partition_t *, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(hierarchy.data(), num_levels));

        // Persistent Hierarchy Metadata on Device
        DeviceU32 d_block_lvl("d_block_lvl", k);
        DeviceU32 d_block_fact("d_block_fact", k);

        // Initialize block 0
        Kokkos::parallel_for("init_hierarchy", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const u32 b) {
            d_block_lvl(b) = 0;
            d_block_fact(b) = 0;
            if (b == 0) {
                d_block_lvl(0) = num_levels - 1;
                d_block_fact(0) = d_hierarchy(num_levels - 1);
            }
        });

        f64 avg_core_weight = (f64) g.g_weight / (f64) k;
        int cl = (int) graphs.size() - 1;

        {
            ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "initial_count");

            if (graphs.back().uniform_vertex_weights) recalculate_weights_and_counts<true>(partition, graphs.back(), vertex_count, exec_space);
            else recalculate_weights_and_counts<false>(partition, graphs.back(), vertex_count, exec_space);

            Kokkos::parallel_for("combine_stats", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
                combined_stats(b) = vertex_count(b);
                combined_stats(k + b) = partition.bweights(b);
            });
            exec_space.fence();
        }

        while (true) {
            Graph &curr_g = graphs[cl];

            assert_state_after_partition(curr_g, partition, k, exec_space);
            assert_hierarchy(k, d_block_lvl, d_block_fact, d_hierarchy, exec_space);

            if (!mappings.empty()) {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "predict_block_distribution");
                predict_block_distribution(mappings.back(), partition.map, vertex_count, exec_space);
            } else {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "count_block_vertices");
                if (curr_g.uniform_vertex_weights) recalculate_weights_and_counts<true>(partition, curr_g, vertex_count, exec_space);
                else recalculate_weights_and_counts<false>(partition, curr_g, vertex_count, exec_space);
            }

            Kokkos::parallel_for("combine_stats", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
                combined_stats(b) = vertex_count(b);
                combined_stats(k + b) = partition.bweights(b);
            });

            bool split_occurred;
            do {
                split_occurred = false;

                BatchedSubgraphs batch;
                const u32 S = extract_block_subgraphs_batched(curr_g, partition.map, k, combined_stats, d_block_fact, d_block_lvl, d_strides, d_hierarchy, threshold, (u32)cl, batch, imbalance, avg_core_weight, mem_stack, exec_space);

                if (S > 0) {
                    split_occurred = true;

                    const vertex_t alloc_n = curr_g.n;
                    UnmanagedDevicePartition sub_part_batch = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * alloc_n), alloc_n);

                    using result_view_t = Kokkos::View<BestBisectConfig *, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
                    result_view_t results_batch((BestBisectConfig *) get_chunk_front(mem_stack, sizeof(BestBisectConfig) * S), S);

                    u32 n_instances = std::min(S, 16u);
                    std::vector<DeviceExecutionSpace> instances = Kokkos::Experimental::partition_space(exec_space, std::vector<int>(n_instances, 1));

                    for (u32 s = 0; s < S; ++s) {
                        weight_t lmax_l = static_cast<weight_t>(batch.h_lmax_l[s]);
                        weight_t lmax_r = static_cast<weight_t>(batch.h_lmax_r[s]);
                        vertex_t voff = batch.h_sub_vertex_offsets[s];

                        Graph sub_g = make_batched_subgraph_view(batch, s, curr_g.uniform_vertex_weights, curr_g.uniform_edge_weights);
                        UnmanagedDevicePartition sub_part = Kokkos::subview(sub_part_batch, std::make_pair(voff, voff + sub_g.n));

                        auto result_s = Kokkos::subview(results_batch, s);

                        {
                            ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "brute_force_bisect");
                            if (sub_g.uniform_vertex_weights && sub_g.uniform_edge_weights) {
                                brute_force_bisect_async<true, true, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            } else if (sub_g.uniform_vertex_weights && !sub_g.uniform_edge_weights) {
                                brute_force_bisect_async<true, false, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            } else if (!sub_g.uniform_vertex_weights && sub_g.uniform_edge_weights) {
                                brute_force_bisect_async<false, true, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            } else {
                                brute_force_bisect_async<false, false, 64>(sub_g, lmax_l, lmax_r, sub_part, result_s, instances[s % n_instances]);
                            }
                        }
                    }

                    for (auto &st: instances) st.fence();

                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "update_partition");
                        update_partition_from_batched_subparts(batch, sub_part_batch, batch.metadata, k, partition.map, exec_space);
                        update_hierarchy(S, batch.split_blocks, batch.metadata, k, d_block_lvl, d_block_fact, d_hierarchy, exec_space);
                        KOKKOS_PROFILE_FENCE(exec_space);
                    }

                    pop_front(mem_stack); // results_batch
                    pop_front(mem_stack); // sub_part_batch
                    free_batched_subgraphs(batch, curr_g.uniform_vertex_weights, curr_g.uniform_edge_weights, mem_stack);

                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "recalc_stats");
                        if (curr_g.uniform_vertex_weights) recalculate_weights_and_counts<true>(partition, curr_g, vertex_count, exec_space);
                        else recalculate_weights_and_counts<false>(partition, curr_g, vertex_count, exec_space);
                        Kokkos::parallel_for("combine_stats", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
                            combined_stats(b) = vertex_count(b);
                            combined_stats(k + b) = partition.bweights(b);
                        });
                        KOKKOS_PROFILE_FENCE(exec_space);
                    }
                } else {
                    free_batched_subgraphs(batch, curr_g.uniform_vertex_weights, curr_g.uniform_edge_weights, mem_stack);
                }
            } while (split_occurred);

            assert_state_after_partition(curr_g, partition, k, exec_space);
            assert_hierarchy(k, d_block_lvl, d_block_fact, d_hierarchy, exec_space);

            if (mappings.empty()) break;
            cl--;
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "uncontract");
                uncontract(partition, mappings.back(), exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);
            }
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "recalc_stats");
                if (graphs[cl].uniform_vertex_weights) recalculate_weights_and_counts<true>(partition, graphs[cl], vertex_count, exec_space);
                else recalculate_weights_and_counts<false>(partition, graphs[cl], vertex_count, exec_space);
                Kokkos::parallel_for("combine_stats", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
                    combined_stats(b) = vertex_count(b);
                    combined_stats(k + b) = partition.bweights(b);
                });
                KOKKOS_PROFILE_FENCE(exec_space);
            }
            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();
            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();
            exec_space.fence();
        }
        pop_front(mem_stack); // combined_stats
        pop_front(mem_stack); // vertex_count
    }
}
#endif //GPU_HEIPA_GPU_PROGRESSIVE_PARTITION_H
