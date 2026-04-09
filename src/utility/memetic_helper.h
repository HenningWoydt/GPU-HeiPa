#ifndef GPU_HEIPA_HELPERS_H
#define GPU_HEIPA_HELPERS_H

#include "definitions.h"
#include "../refinement/block_conn.h"
#include "../datastructures/partition.h"
#include "../utility/hungarian_algorithm.h"

namespace GPU_HeiPa {
    struct KeyTuple {
        u32 key_count;
        u64 key;
    };

    enum class PopulationManagement {
        steadystate,
        shrinking // #partitions == (level + 1)
    };


    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- different ways to distribute leftover vertices: -------------------------
    //! -------------------------------------------------------------------------------------------------




    inline void assign_leftovers_favorUnderloadedBlocks(
        const Graph &graph,
        Partition &child,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        DeviceExecutionSpace &exec_space
    ) {
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);
        UnmanagedDeviceF64 distribution = UnmanagedDeviceF64((f64 *) get_chunk_back(mem_stack, sizeof(f64) * k), k);


        Kokkos::parallel_scan(
            "create distribution",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k),
            KOKKOS_LAMBDA(partition_t id, f64 &update, bool final) {
                weight_t sum = 0;
                weight_t inverse_weight = (lmax - child.bweights(id));

                for (partition_t i = 0; i < k; ++i) {
                    sum += (lmax - child.bweights(i));
                }

                update += static_cast<double>(inverse_weight) / static_cast<double>(sum);
                if (final) {
                    distribution(id) = update;
                }
            }
        );


        // assign remaining vertices
        Kokkos::parallel_for(
            "assign leftovers",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    f64 rand = gen.drand(0.0, 1.0);
                    random_pool.free_state(gen);


                    for (u32 i = 0; i < k; ++i) {
                        if ((rand < distribution(i)) || (i == (k - 1))) {
                            child.map(u) = i;
                            Kokkos::atomic_fetch_add(&child.bweights(i), graph.weights(u));
                            break;
                        }
                    }
                }
            }
        );

        pop_back(mem_stack); //rm distribution
    }

    inline void assign_leftovers_gain_and_weight(
        const Graph &graph,
        Partition &child,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        f64 alpha,
        DeviceExecutionSpace &exec_space
    ) {
        // this determines how much underloaded blocks are weighted
        // alpha = 0 -> only gain, big alpha -> only underloaded blocks

        Kokkos::Random_XorShift64_Pool<> random_pool(12345);


        BlockConn bc;
        if (graph.uniform_edge_weights) {
            bc = init_BlockConn<true>(graph, child, mem_stack, exec_space);
        } else {
            bc = init_BlockConn<false>(graph, child, mem_stack, exec_space);
        }

        //! determine max gain
        DeviceScalarWeight max_gain = DeviceScalarWeight("highest gain value");;
        Kokkos::parallel_reduce(
            "determine max gain",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, bc.size),
            KOKKOS_LAMBDA(u32 index, weight_t &update) {
                weight_t val = bc.weights(index);
                if ((val > update) && (bc.ids(index) != 5 * k)) {
                    update = val;
                }
            }, Kokkos::Max(max_gain)
        );

        Kokkos::parallel_for(
            "distribute leftovers",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                // calculate gain
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    partition_t best_id = static_cast<partition_t>(gen.urand64() % static_cast<u64>(k));
                    random_pool.free_state(gen);

                    f64 best_score = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        if (id == 5 * k)
                            continue;

                        weight_t gain = bc.weights(i);

                        //! i think this is actually quite smart, because in the case that
                        //! child.bweights(id) > lmax, then the right part will get negative
                        //! -> i.e. overweight blocks are penalized
                        f64 my_score = gain + (alpha * max_gain() * (static_cast<double>(lmax - child.bweights(id)) / static_cast<double>(lmax)));

                        bool valid = (id != NULL_PART) & (id != HASH_RECLAIM); // single mask

                        // Update best if it's a candidate and better
                        bool better = valid & (my_score > best_score);
                        best_score = better ? my_score : best_score;
                        best_id = better ? id : best_id;
                    }

                    child.map(u) = best_id;
                    Kokkos::atomic_fetch_add(&child.bweights(best_id), graph.weights(u));
                }
            }
        );

        free_BlockConn(bc, mem_stack);

        return;
    }




    template<bool uniform_vw>
    inline void assign_leftovers_fullyRandom(
        const Graph &graph,
        Partition &child,
        partition_t k,
        DeviceExecutionSpace &exec_space
    ) {

        Kokkos::Random_XorShift64_Pool<> random_pool(12345);

        // assign remaining vertices
        Kokkos::parallel_for(
            "assign leftovers",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    partition_t id = static_cast<partition_t>(gen.urand64() % static_cast<u64>(k));
                    random_pool.free_state(gen);

                    child.map(u) = id;
                    Kokkos::atomic_add(&child.bweights(id), uniform_vw ? 1 : graph.weights(u));
                    
                }
            }
        );
    }




    inline void assign_leftovers_gain(
        const Graph &graph,
        Partition &child,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        DeviceExecutionSpace &exec_space
    ) {
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);


        BlockConn bc;
        if (graph.uniform_edge_weights) {
            bc = init_BlockConn<true>(graph, child, mem_stack, exec_space);
        } else {
            bc = init_BlockConn<false>(graph, child, mem_stack, exec_space);
        }


        Kokkos::parallel_for(
            "distribute leftovers",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                // calculate gain
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    partition_t best_id = static_cast<partition_t>(gen.urand64() % static_cast<u64>(k));
                    random_pool.free_state(gen);

                    weight_t best_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        if (id == 5 * k)
                            continue;

                        weight_t w = bc.weights(i);

                        bool valid = (id != NULL_PART) & (id != HASH_RECLAIM); // single mask

                        // Update best if it's a candidate and better
                        bool better = valid & (w > best_conn);
                        best_conn = better ? w : best_conn;
                        best_id = better ? id : best_id;
                    }

                    child.map(u) = best_id;
                    Kokkos::atomic_fetch_add(&child.bweights(best_id), graph.weights(u));
                }
            }
        );

        free_BlockConn(bc, mem_stack);

        return;
    }


    //! ----------------------------------------------------------------

    inline u32 next_power_of_two(
        u32 k
    ) {
        if (k <= 1) return 1;

        k--;
        k |= k >> 1;
        k |= k >> 2;
        k |= k >> 4;
        k |= k >> 8;
        k |= k >> 16;
        return k + 1;
    }

    // basically returns log2(k)
    // and k should always be power of 2
    u32 bits_needed(u32 k) {
        u32 b = 0;
        k--;
        while (k > 0) {
            k >>= 1;
            b++;
        }
        return b;
    }

    KOKKOS_FUNCTION u64 determine_key(
        vertex_t u,
        const Kokkos::View<int *> &parent_ids,
        const Kokkos::View<Partition *> &population,
        u32 num_bits
    ) {
        u64 key = 0;
        for (size_t i = 0; i < parent_ids.size(); ++i) {
            u64 val = static_cast<u64>(population[parent_ids[i]].map(u));
            key |= (val & 0xFF) << (num_bits * i); //! do i even need this &0xFF ?
        }
        return key;
    }


    inline u32 max_matching(
        const UnmanagedDeviceU32 &sim_matrix,
        const u32 k,
        DeviceExecutionSpace &exec_space
    ) {
        if (k == 0) return 0;

        // Copy matrix from GPU to CPU
        std::vector<u32> matrix_host(k * k);
        auto sim_matrix_host = Kokkos::create_mirror_view(sim_matrix);
        Kokkos::deep_copy(exec_space, sim_matrix_host, sim_matrix);

        for (u32 i = 0; i < k * k; ++i) {
            matrix_host[i] = sim_matrix_host(i);
        }

        // Apply Hungarian algorithm for maximum weight matching
        u32 result = HungarianAlgorithm::solve(matrix_host.data(), k);

        return result;
    }


    inline u32 determine_distance(
        const Graph &graph,
        const Partition &A,
        const Partition &B,
        const partition_t k,
        KokkosMemoryStack &mem_stack,
        DeviceExecutionSpace &exec_space
    ) {
        u32 distance;

        // build matrix
        auto sim_matrix = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * k * k), k * k);

        Kokkos::parallel_for(
            "init sim_matrix",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k * k),
            KOKKOS_LAMBDA(u32 index) {
                sim_matrix(index) = 0;
            }
        );
        exec_space.fence();

        Kokkos::parallel_for(
            "fill matrix",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                u32 row = A.map(u);
                u32 col = B.map(u);

                Kokkos::atomic_fetch_add(&sim_matrix(row * k + col), 1);
            }
        );
        exec_space.fence();

        // get maximum matching on matrix
        u32 similarity = max_matching(sim_matrix, k, exec_space);
        distance = graph.n - similarity;

        pop_back(mem_stack); //rm sim_matrix

        return distance;
    }
}

#endif
