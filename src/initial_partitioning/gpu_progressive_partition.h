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

    /**
     * Managed container for a batch of identical-size graphs.
     */
    struct GraphBatch {
        Kokkos::View<uint8_t*, DeviceMemorySpace> storage;
        Kokkos::View<Graph*, DeviceMemorySpace> graphs;
        size_t stride_bytes = 0; // Size of one graph in the chunk
    };

    /**
     * Allocates space for k graphs as one big chunk based on a reference graph.
     */
    inline GraphBatch allocate_graph_batch(const Graph& ref_g,
                                           const size_t k,
                                           DeviceExecutionSpace& exec_space) {
        // 1. Calculate space required for a single graph instance (CSR data)
        size_t stride = 0;
        stride += round_up_64(sizeof(u32) * (ref_g.n + 1));      // neighborhood
        stride += round_up_64(sizeof(vertex_t) * ref_g.m);       // edges_u
        stride += round_up_64(sizeof(vertex_t) * ref_g.m);       // edges_v

        if (!ref_g.uniform_edge_weights) {
            stride += round_up_64(sizeof(weight_t) * ref_g.m);   // edges_w
        }
        if (!ref_g.uniform_vertex_weights) {
            stride += round_up_64(sizeof(weight_t) * ref_g.n);   // weights
        }

        // 2. Perform the single large device allocation
        GraphBatch batch;
        batch.stride_bytes = stride;
        batch.storage = Kokkos::View<uint8_t*, DeviceMemorySpace>("batch_storage", stride * k);
        batch.graphs = Kokkos::View<Graph*, DeviceMemorySpace>("batch_graphs", k);

        // 3. Initialize Graph objects on Host with correct device pointers
        auto h_graphs = Kokkos::create_mirror_view(batch.graphs);
        uint8_t* base_ptr = batch.storage.data();

        for (size_t i = 0; i < k; ++i) {
            // Set metadata based on reference
            h_graphs(i).n = ref_g.n;
            h_graphs(i).m = ref_g.m;
            h_graphs(i).g_weight = ref_g.g_weight;
            h_graphs(i).uniform_vertex_weights = ref_g.uniform_vertex_weights;
            h_graphs(i).uniform_edge_weights = ref_g.uniform_edge_weights;

            // Offset to the i-th graph's memory region
            uint8_t* graph_mem = base_ptr + (i * stride);
            size_t local_cursor = 0;

            h_graphs(i).neighborhood = UnmanagedDeviceU32((u32*)(graph_mem + local_cursor), ref_g.n + 1);
            local_cursor += round_up_64(sizeof(u32) * (ref_g.n + 1));

            h_graphs(i).edges_u = UnmanagedDeviceVertex((vertex_t*)(graph_mem + local_cursor), ref_g.m);
            local_cursor += round_up_64(sizeof(vertex_t) * ref_g.m);

            h_graphs(i).edges_v = UnmanagedDeviceVertex((vertex_t*)(graph_mem + local_cursor), ref_g.m);
            local_cursor += round_up_64(sizeof(vertex_t) * ref_g.m);

            if (!ref_g.uniform_edge_weights) {
                h_graphs(i).edges_w = UnmanagedDeviceWeight((weight_t*)(graph_mem + local_cursor), ref_g.m);
                local_cursor += round_up_64(sizeof(weight_t) * ref_g.m);
            }
            if (!ref_g.uniform_vertex_weights) {
                h_graphs(i).weights = UnmanagedDeviceWeight((weight_t*)(graph_mem + local_cursor), ref_g.n);
            }
        }

        // 4. Transfer the Graph objects (containing the pointers) to the device
        Kokkos::deep_copy(exec_space, batch.graphs, h_graphs);

        return batch;
    }

    /**
     * Extracts all k subgraphs from a graph based on a partition into a GraphBatch.
     * Assumes all k subgraphs have the same dimensions as the Graphs in the batch.
     */
    inline void extract_all_subgraphs_batched(const Graph& g,
                                              const Partition& partition,
                                              GraphBatch& batch,
                                              KokkosMemoryStack& mem_stack,
                                              DeviceExecutionSpace& exec_space) {
        const partition_t k = partition.k;
        const vertex_t gn = g.n;
        const bool uniform_vw = g.uniform_vertex_weights;
        const bool uniform_ew = g.uniform_edge_weights;

        // 1. Temporary mapping from global vertex to local vertex index in its subgraph
        UnmanagedDeviceVertex global_to_local = UnmanagedDeviceVertex((vertex_t*)get_chunk_front(mem_stack, sizeof(vertex_t) * gn), gn);
        UnmanagedDeviceVertex local_counters = UnmanagedDeviceVertex((vertex_t*)get_chunk_front(mem_stack, sizeof(vertex_t) * k), k);
        UnmanagedDeviceU32 edge_counters = UnmanagedDeviceU32((u32*)get_chunk_front(mem_stack, sizeof(u32) * k), k);

        Kokkos::deep_copy(exec_space, local_counters, 0);
        Kokkos::deep_copy(exec_space, edge_counters, 0);

        auto p_map = partition.map;
        auto graphs = batch.graphs;

        // 2. Assign local IDs to vertices and copy vertex weights
        Kokkos::parallel_for("fill_sub_vertices", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = p_map(u);
            vertex_t local_u = Kokkos::atomic_fetch_add(&local_counters(b), 1);
            global_to_local(u) = local_u;

            if (!uniform_vw) {
                graphs(b).weights(local_u) = g.weights(u);
            }
        });

        // 3. Count edges and fill sub-neighborhoods (prefix sum style)
        // Note: For identical-size graphs in a batch, sub_g.neighborhood(local_u)
        // will be filled as we iterate.
        auto g_neigh = g.neighborhood;
        auto g_edges_v = g.edges_v;
        auto g_edges_w = g.edges_w;

        Kokkos::parallel_for("fill_sub_edges", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, gn), KOKKOS_LAMBDA(const vertex_t u) {
            partition_t b = p_map(u);
            vertex_t local_u = global_to_local(u);
            Graph& sub_g = graphs(b);

            u32 start_e = g_neigh(u);
            u32 end_e = g_neigh(u + 1);

            // We need to count how many edges stay within the same block
            u32 local_edge_count = 0;
            for (u32 e = start_e; e < end_e; ++e) {
                vertex_t v = g_edges_v(e);
                if (p_map(v) == b) {
                    local_edge_count++;
                }
            }

            // This is a simplified fill for identical-size batches.
            // In a real scenario, sub_g.n and sub_g.m might differ per block.
            // If the batch assumes MAXIMUM sizes, we must be careful with neighborhood offsets.
            // Here we assume the caller ensures sub_g has enough capacity.

            u32 sub_e_pos = Kokkos::atomic_fetch_add(&edge_counters(b), local_edge_count);

            // Fill neighborhood: this is tricky without a prefix sum per subgraph.
            // For now, we store the local count in the neighborhood array to scan it later.
            sub_g.neighborhood(local_u + 1) = local_edge_count;

            u32 current_e_idx = sub_e_pos;
            for (u32 e = start_e; e < end_e; ++e) {
                vertex_t v = g_edges_v(e);
                if (p_map(v) == b) {
                    vertex_t local_v = global_to_local(v);
                    sub_g.edges_u(current_e_idx) = local_u;
                    sub_g.edges_v(current_e_idx) = local_v;
                    if (!uniform_ew) {
                        sub_g.edges_w(current_e_idx) = g_edges_w(e);
                    }
                    current_e_idx++;
                }
            }
        });

        // 4. Finalize neighborhoods for each subgraph
        Kokkos::parallel_for("finalize_sub_neighborhoods", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
            Graph& sub_g = graphs(b);
            sub_g.neighborhood(0) = 0;
            u32 sum = 0;
            for (vertex_t i = 0; i < sub_g.n; ++i) {
                u32 deg = sub_g.neighborhood(i + 1);
                sub_g.neighborhood(i + 1) = sum + deg;
                sum += deg;
            }
        });

        pop_front(mem_stack); // edge_counters
        pop_front(mem_stack); // local_counters
        pop_front(mem_stack); // global_to_local
    }

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
                const weight_t wu = uvw ? 1 : g.weights(flip_u);

                if (new_part_u) wr += wu;
                else wr -= wu;

                for (u32 e = g.neighborhood(flip_u); e < g.neighborhood(flip_u + 1); ++e) {
                    const vertex_t v = g.edges_v(e);
                    const u64 part_v = (gray >> v) & 1ULL;
                    const bool was_cut = old_part_u != part_v;
                    const bool now_cut = new_part_u != part_v;
                    const weight_t ew = uew ? 1 : g.edges_w(e);
                    if (was_cut && !now_cut) cut -= ew;
                    else if (!was_cut && now_cut) cut += ew;
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

        UnmanagedDeviceU32 metadata;
        u32 rid_offset = 0;
        u32 totals_offset = 0;
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
        pop_front(mem_stack); // metadata
    }

    inline void extract_block_subgraphs_batched(const Graph &g,
                                                const UnmanagedDevicePartition &partition_map,
                                                partition_t k,
                                                const UnmanagedDevicePartition &d_blocks_to_split,
                                                const u32 S,
                                                BatchedSubgraphs &batch,
                                                const DeviceU32 &d_strides,
                                                const DeviceU32 &d_block_lvl,
                                                const DeviceU32 &d_block_fact,
                                                const f64 imbalance,
                                                const f64 avg_core_weight,
                                                KokkosMemoryStack &mem_stack,
                                                DeviceExecutionSpace &exec_space) {
        batch.S = S;

        if (S == 0) {
            batch.total_sub_n = 0;
            batch.total_sub_m = 0;
            batch.total_neigh_n = 0;
            return;
        }

        constexpr partition_t INVALID_SPLIT = std::numeric_limits<partition_t>::max();

        // Layout: [n(S), m(S), weight(S), v_off(S), e_off(S), n_off(S), rid(S), lmax_l(S), lmax_r(S), totals(3)]
        batch.metadata = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * (S * 9 + 3)), S * 9 + 3);
        batch.rid_offset = 6 * S;
        const u32 lmax_l_off = 7 * S;
        const u32 lmax_r_off = 8 * S;
        batch.totals_offset = 9 * S;
        auto d_metadata = batch.metadata;

        batch.split_blocks = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * S), S);
        batch.block_to_split = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * k), k);

        batch.sub_n = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);
        batch.sub_m = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * S), S);
        batch.sub_weight = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * S), S);
        batch.sub_vertex_offsets = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);
        batch.sub_edge_offsets = UnmanagedDeviceU32((u32 *) get_chunk_front(mem_stack, sizeof(u32) * S), S);
        batch.sub_neigh_offsets = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);
        batch.sub_write_pos = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * S), S);

        Kokkos::deep_copy(exec_space, batch.split_blocks, Kokkos::subview(d_blocks_to_split, std::make_pair(0u, S)));

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

        Totals totals;

        Kokkos::parallel_scan("batch_compute_metadata", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, S), KOKKOS_LAMBDA(const u32 s, Totals &update, const bool final) {
            const vertex_t ns = sub_n(s);
            const u32 ms = sub_m(s);
            const weight_t ws = sub_weight(s);

            if (final) {
                d_metadata(s) = ns;
                d_metadata(S + s) = ms;
                d_metadata(2 * S + s) = static_cast<u32>(ws);
                d_metadata(3 * S + s) = update.v;
                d_metadata(4 * S + s) = update.e;
                d_metadata(5 * S + s) = update.n;

                // RID and lmax calculation on device
                partition_t b = d_split_blocks(s);
                u32 lvl = d_block_lvl(b);
                partition_t f = d_block_fact(b);
                partition_t rp = 1 << (u32) floor(log2(static_cast<float>(f - 1)));
                partition_t lp = f - rp;
                partition_t stride = d_strides(lvl);

                d_metadata(6 * S + s) = b + lp * stride;
                d_metadata(lmax_l_off + s) = static_cast<u32>(ceil((1.0 + imbalance) * avg_core_weight * (lp * stride)));
                d_metadata(lmax_r_off + s) = static_cast<u32>(ceil((1.0 + imbalance) * avg_core_weight * (rp * stride)));
            }
            update.v += ns;
            update.e += ms;
            update.n += ns + 1;
        }, totals);

        Kokkos::parallel_for("fill_totals", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
            d_metadata(batch.totals_offset) = totals.v;
            d_metadata(batch.totals_offset + 1) = totals.e;
            d_metadata(batch.totals_offset + 2) = totals.n;
        });

        // Download all metadata to the host in a single contiguous transfer
        HostU32 h_full_metadata(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_full_metadata"), batch.metadata.extent(0));
        Kokkos::deep_copy(exec_space, h_full_metadata, batch.metadata);
        exec_space.fence();

        batch.total_sub_n = h_full_metadata(batch.totals_offset);
        batch.total_sub_m = h_full_metadata(batch.totals_offset + 1);
        batch.total_neigh_n = h_full_metadata(batch.totals_offset + 2);

        // Store local host vectors for loop management
        batch.h_sub_n.assign(h_full_metadata.data(), h_full_metadata.data() + S);
        batch.h_sub_m.assign(h_full_metadata.data() + S, h_full_metadata.data() + 2 * S);
        batch.h_sub_weight.resize(S);
        for (u32 s = 0; s < S; ++s) batch.h_sub_weight[s] = static_cast<weight_t>(h_full_metadata(2 * S + s));
        batch.h_sub_vertex_offsets.assign(h_full_metadata.data() + 3 * S, h_full_metadata.data() + 4 * S);
        batch.h_sub_edge_offsets.assign(h_full_metadata.data() + 4 * S, h_full_metadata.data() + 5 * S);
        batch.h_sub_neigh_offsets.assign(h_full_metadata.data() + 5 * S, h_full_metadata.data() + 6 * S);

        // Metadata on host for balance limits
        std::vector<u32> h_lmax_l(h_full_metadata.data() + lmax_l_off, h_full_metadata.data() + lmax_l_off + S);
        std::vector<u32> h_lmax_r(h_full_metadata.data() + lmax_r_off, h_full_metadata.data() + lmax_r_off + S);

        // We still need the structural offsets on the device for kernels
        Kokkos::parallel_for("fill_device_offsets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, S), KOKKOS_LAMBDA(const u32 s) {
            batch.sub_vertex_offsets(s) = d_metadata(3 * S + s);
            batch.sub_edge_offsets(s) = d_metadata(4 * S + s);
            batch.sub_neigh_offsets(s) = d_metadata(5 * S + s);
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
                                                       const UnmanagedDeviceU32 &d_metadata,
                                                       UnmanagedDevicePartition &partition_map,
                                                       DeviceExecutionSpace &exec_space) {
        auto sub_to_old = batch.sub_to_old;
        auto sub_vertex_to_split = batch.sub_vertex_to_split;
        const u32 S = batch.S;

        Kokkos::parallel_for("batch_update_partition_from_subparts", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, batch.total_sub_n), KOKKOS_LAMBDA(const vertex_t global_sub_u) {
            if (sub_part(global_sub_u) == 1) {
                vertex_t old_u = sub_to_old(global_sub_u);
                partition_t s = sub_vertex_to_split(global_sub_u);
                partition_t rid = d_metadata(6 * S + s);

                partition_map(old_u) = rid;
            }
        });
    }

    inline void update_hierarchy(const u32 S,
                                 const UnmanagedDevicePartition &d_split_blocks,
                                 const UnmanagedDeviceU32 &d_metadata,
                                 DeviceU32 &d_block_lvl,
                                 DeviceU32 &d_block_fact,
                                 const DeviceU32 &d_hierarchy,
                                 DeviceExecutionSpace &exec_space) {
        Kokkos::parallel_for("update_hierarchy", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, S), KOKKOS_LAMBDA(const u32 s) {
            partition_t b = d_split_blocks(s);
            partition_t rid = d_metadata(6 * S + s);

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

        GraphBatch graph_batch = allocate_graph_batch(g, k, exec_space);

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

            recalculate_weights<false>(partition, graphs.back(), exec_space);
            count_block_vertices(partition, graphs.back(), vertex_count, exec_space);

            Kokkos::parallel_for("combine_stats", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
                combined_stats(b) = vertex_count(b);
                combined_stats(k + b) = partition.bweights(b);
            });
            exec_space.fence();
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

            Kokkos::parallel_for("combine_stats", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
                combined_stats(b) = vertex_count(b);
                combined_stats(k + b) = partition.bweights(b);
            });

            bool split_occurred;
            do {
                split_occurred = false;

                UnmanagedDevicePartition d_blocks_to_split((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * k), k);
                HostScalarPinnedU32 h_S("h_S");

                Kokkos::parallel_scan("find_split_candidates", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const u32 b, u32 &update, const bool final) {
                    const u32 n = combined_stats(b);
                    const u32 fact = d_block_fact(b);
                    bool splittable = (n > 0) && (fact > 1) && ((n > threshold) || (cl == 0));
                    if (splittable) {
                        if (final) d_blocks_to_split(update) = b;
                        update++;
                    }
                }, h_S);
                exec_space.fence();

                const u32 S = h_S();

                if (S > 0) {
                    split_occurred = true;

                    extract_all_subgraphs_batched(g, partition, graph_batch, mem_stack, exec_space);

                    // --- UPPER BOUND ALLOCATION ---
                    // Allocate based on current graph level sizes to avoid waiting for exact counts
                    const vertex_t alloc_n = curr_g.n;

                    BatchedSubgraphs batch;
                    extract_block_subgraphs_batched(curr_g, partition.map, k, d_blocks_to_split, S, batch, d_strides, d_block_lvl, d_block_fact, imbalance, avg_core_weight, mem_stack, exec_space);

                    UnmanagedDevicePartition sub_part_batch = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * alloc_n), alloc_n);

                    using result_view_t = Kokkos::View<BestBisectConfig *, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
                    result_view_t results_batch((BestBisectConfig *) get_chunk_front(mem_stack, sizeof(BestBisectConfig) * S), S);

                    u32 n_instances = std::min(S, 16u);
                    std::vector<DeviceExecutionSpace> instances = Kokkos::Experimental::partition_space(exec_space, std::vector<int>(n_instances, 1));

                    // Use downloaded metadata for host-side loop control (memory subviewing and workload)
                    HostU32 h_full_metadata(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_full_metadata"), batch.metadata.extent(0));
                    Kokkos::deep_copy(exec_space, h_full_metadata, batch.metadata);
                    exec_space.fence();

                    for (u32 s = 0; s < S; ++s) {
                        weight_t lmax_l = static_cast<weight_t>(h_full_metadata(7 * S + s));
                        weight_t lmax_r = static_cast<weight_t>(h_full_metadata(8 * S + s));
                        vertex_t voff = h_full_metadata(3 * S + s);

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
                        update_partition_from_batched_subparts(batch, sub_part_batch, batch.metadata, partition.map, exec_space);
                        update_hierarchy(S, batch.split_blocks, batch.metadata, d_block_lvl, d_block_fact, d_hierarchy, exec_space);
                        KOKKOS_PROFILE_FENCE(exec_space);
                    }

                    pop_front(mem_stack); // results_batch
                    pop_front(mem_stack); // sub_part_batch
                    free_batched_subgraphs(batch, curr_g.uniform_vertex_weights, curr_g.uniform_edge_weights, mem_stack);

                    {
                        ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "recalc_stats");
                        recalculate_weights<false>(partition, curr_g, exec_space);
                        count_block_vertices(partition, curr_g, vertex_count, exec_space);
                        Kokkos::parallel_for("combine_stats", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k), KOKKOS_LAMBDA(const partition_t b) {
                            combined_stats(b) = vertex_count(b);
                            combined_stats(k + b) = partition.bweights(b);
                        });
                        KOKKOS_PROFILE_FENCE(exec_space);
                    }
                }
                pop_front(mem_stack); // d_blocks_to_split
            } while (split_occurred);

            if (mappings.empty()) break;
            cl--;
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "uncontract");
                uncontract(partition, mappings.back(), exec_space);
                KOKKOS_PROFILE_FENCE(exec_space);
            }
            {
                ScopedTimer _t("initial_partitioning", "gpu_progressive_partition", "recalc_stats");
                recalculate_weights<false>(partition, graphs[cl], exec_space);
                count_block_vertices(partition, graphs[cl], vertex_count, exec_space);
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

