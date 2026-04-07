#ifndef GPU_HEIPA_MEMETIC_REFINEMENT_SHRINKING_H
#define GPU_HEIPA_MEMETIC_REFINEMENT_SHRINKING_H

#include <Kokkos_Core.hpp>
#include <bitset>
#include <numeric>
#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>
#include <Kokkos_Random.hpp>

#include "../utility/definitions.h"
#include "../utility/memetic_helper.h"
#include "../utility/hungarian_algorithm.h"
#include "../datastructures/partition.h"
#include "block_conn.h"

#include "omp.h"

namespace GPU_HeiPa {
    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- selection: --------------------------------------------------------------
    //! -------------------------------------------------------------------------------------------------


    inline int tournament_selection(
        const std::vector<weight_t> &fitness_values,
        const u32 tournament_size,
        const size_t parents_curr
    ) {
        // get num_parents random numbers between [0, num_individuals)
        size_t num_individuals = parents_curr;
        std::vector<size_t> indices;
        std::unordered_set<size_t> unique_indices;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, num_individuals - 1);

        while (unique_indices.size() < tournament_size) {
            size_t idx = dis(gen);
            if (unique_indices.insert(idx).second) {
                indices.push_back(idx);
            }
        }


        size_t best_idx = indices[0];
        weight_t best_fitness = fitness_values[best_idx];
        for (size_t i = 1; i < indices.size(); ++i) {
            if (fitness_values[indices[i]] < best_fitness) {
                best_fitness = fitness_values[indices[i]];
                best_idx = indices[i];
            }
        }
        return static_cast<int>(best_idx);
    }


    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- different ways to distribute leftover vertices: -------------------------
    //! -------------------------------------------------------------------------------------------------











    //! -------------------------------------------------------------------------------------------------
    //! --------------------------- crossover: ----------------------------------------------------------
    //! -------------------------------------------------------------------------------------------------


    inline void backbone_based_crossover(
        Partition &child,
        const Graph &graph,
        const std::vector<int> &parent_ids,
        const std::vector<Partition> &population,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        const std::string &leftover_strategy,
        f64 alpha,
        partition_t extent,
        DeviceExecutionSpace &exec_space

    ) {
        //setup: get the vectors onto the GPU
        auto parent_ids_device = Kokkos::View<int *, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
            (int *) get_chunk_back(mem_stack, sizeof(int) * parent_ids.size()),
            parent_ids.size()
        );
        auto population_device = Kokkos::View<Partition *, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
            (Partition *) get_chunk_back(mem_stack, sizeof(Partition) * population.size()),
            population.size()
        );

        Kokkos::deep_copy(exec_space, parent_ids_device, Kokkos::View<const int *>(parent_ids.data(), parent_ids.size()));
        Kokkos::deep_copy(exec_space, population_device, Kokkos::View<const Partition *>(population.data(), population.size()));

        exec_space.fence();

        partition_t k_prime = next_power_of_two(k);
        u32 num_bits = bits_needed(k_prime);

        u64 num_buckets = static_cast<u64>(pow(k_prime, parent_ids.size()));

        auto buckets = Kokkos::View<KeyTuple *, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
            (KeyTuple *) get_chunk_back(mem_stack, sizeof(KeyTuple) * num_buckets), num_buckets
        );


        Kokkos::parallel_for(
            "init buckets",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_buckets),
            KOKKOS_LAMBDA(u64 index) {
                buckets(index).key_count = 0;
                buckets(index).key = index;
            }
        );

        Kokkos::parallel_for(
            "fill buckets",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits);
                Kokkos::atomic_fetch_add(&buckets(key).key_count, 1);
            }
        );

        // sort descending based on key_count
        // after sorting, the k most frequent keys will be at the top
        // and you can query them via .key
        Kokkos::sort(exec_space, buckets, KOKKOS_LAMBDA(const KeyTuple &a, const KeyTuple &b) {
            return a.key_count > b.key_count;
        });


        partition_t local_extent = std::min(extent, k);
        if (local_extent < 1) {
            local_extent = 1;
        }

        // assign vertices of the backbone to the offspring
        Kokkos::parallel_for(
            "create new offspring",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                partition_t id;
                bool in_backbone = false;
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits);

                for (partition_t j = 0; (j <= local_extent) && (!in_backbone); ++j) {
                    for (partition_t i = 0; i < k; ++i) {
                        if (key == buckets(i + (j * k)).key) {
                            if (j == 0)
                                id = i;
                            else
                                id = k - i - 1; //! "reverse assignment" from full buckets to underloaded partitions

                            Kokkos::atomic_fetch_add(&child.bweights(id), graph.weights(u));
                            in_backbone = true;
                            break;
                        }
                    }
                }


                if (!in_backbone) {
                    id = 5 * k; //! mark as not assigned 
                }

                child.map(u) = id;
            }
        );


        if (leftover_strategy == "random") {
            assign_leftovers_fullyRandom(graph, child, k, exec_space);
        } else if (leftover_strategy == "balanced") {
            assign_leftovers_favorUnderloadedBlocks(graph, child, k, lmax, mem_stack, exec_space);
        } else if (leftover_strategy == "gain") {
            assign_leftovers_gain(graph, child, k, lmax, mem_stack, exec_space);
        } else {
            assign_leftovers_gain_and_weight(graph, child, k, lmax, mem_stack, alpha, exec_space);
        }

        pop_back(mem_stack); //rm buckets
        pop_back(mem_stack); //rm population
        pop_back(mem_stack); //rm parent_ids
    }


    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- distance computation stuff: ---------------------------------------------
    //! -------------------------------------------------------------------------------------------------


    inline u32 determine_min_distance_offspring(
        const Graph &graph,
        const std::vector<Partition> &population,
        size_t parents_curr,
        const Partition &offspring,
        partition_t k,
        std::vector<KokkosMemoryStack> &mem_stacks,
        std::vector<DeviceExecutionSpace> &exec_spaces,
        size_t num_cpu_threads
    ) {
        // size_t pop_size = population.size();
        u32 min_distance = std::numeric_limits<u32>::max();


        #pragma omp parallel for reduction(min:min_distance) num_threads(static_cast<int>(num_cpu_threads))
        for (size_t i = 0; i < parents_curr; ++i) {
            size_t tid = static_cast<size_t>(omp_get_thread_num());
            u32 distance = determine_distance(
                graph,
                population[i],
                offspring,
                k,
                mem_stacks[tid],
                exec_spaces[tid]
            );

            min_distance = std::min(min_distance, distance);
        }


        return min_distance;
    }

    inline void determine_min_distances_population(
        const Graph &graph,
        const std::vector<Partition> &population,
        size_t parents_curr, // amount of non-offspring individuals in the population
        std::vector<u32> &min_distances,
        partition_t k,
        std::vector<KokkosMemoryStack> &mem_stacks,
        std::vector<DeviceExecutionSpace> &exec_spaces,
        size_t num_cpu_threads

    ) {
        //size_t pop_size = population.size();

        std::vector<u32> all_distances(parents_curr * parents_curr, std::numeric_limits<u32>::max());

        //! this can be trivially parallelized via
        //! #pragma omp parallel collapse
        #pragma omp parallel for collapse(2) num_threads(static_cast<int>(num_cpu_threads))
        for (size_t i = 0; i < parents_curr; ++i) {
            for (size_t j = i + 1; j < parents_curr; ++j) {
                size_t tid = static_cast<size_t>(omp_get_thread_num());
                u32 dis = determine_distance(
                    graph,
                    population[i],
                    population[j],
                    k,
                    mem_stacks[tid],
                    exec_spaces[tid]
                );

                all_distances[i * parents_curr + j] = dis;
                all_distances[j * parents_curr + i] = dis;
            }
        }

        for (u32 i = 0; i < parents_curr; ++i) {
            u32 min_val = std::numeric_limits<u32>::max();
            for (u32 j = 0; j < parents_curr; ++j) {
                min_val = std::min(min_val, all_distances[i * parents_curr + j]);
            }
            min_distances[i] = min_val;
        }

        return;
    }

    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- SAMPLED distance computation (faster alternative) -----------------------
    //! -------------------------------------------------------------------------------------------------

    inline u32 determine_min_distance_offspring_sampled(
        const Graph &graph,
        const std::vector<Partition> &population,
        const Partition &offspring,
        size_t parents_curr,
        partition_t k,
        std::vector<KokkosMemoryStack> &mem_stacks,
        std::vector<DeviceExecutionSpace> &exec_spaces,
        size_t num_cpu_threads,
        size_t sample_size
    ) {
        size_t pop_size = parents_curr;
        u32 min_distance = std::numeric_limits<u32>::max();

        // Create indices for sampling
        std::vector<size_t> candidate_indices;
        for (size_t i = 0; i < pop_size; ++i) {
            candidate_indices.push_back(i);
        }

        // Shuffle and take first sample_size indices
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(candidate_indices.begin(), candidate_indices.end(), g);

        const size_t num_to_check = std::min(sample_size, candidate_indices.size());

        // Evaluate offspring against sampled candidates
        #pragma omp parallel for reduction(min:min_distance) num_threads(static_cast<int>(num_cpu_threads))
        for (size_t s = 0; s < num_to_check; ++s) {
            size_t individual = candidate_indices[s];
            size_t tid = static_cast<size_t>(omp_get_thread_num());
            u32 distance = determine_distance(
                graph,
                population[individual],
                offspring,
                k,
                mem_stacks[tid],
                exec_spaces[tid]
            );

            min_distance = std::min(min_distance, distance);
        }

        return min_distance;
    }

    inline void determine_min_distances_population_sampled(
        const Graph &graph,
        const std::vector<Partition> &population,
        size_t parents_curr,
        std::vector<u32> &min_distances,
        partition_t k,
        std::vector<KokkosMemoryStack> &mem_stacks,
        std::vector<DeviceExecutionSpace> &exec_spaces,
        size_t num_cpu_threads,
        size_t sample_size
    ) {
        size_t pop_size = parents_curr;

        // For each individual, compute distance to a sampled subset of other individuals
        #pragma omp parallel for num_threads(static_cast<int>(num_cpu_threads))
        for (size_t i = 0; i < pop_size; ++i) {
            // Build candidate set: all individuals except i
            std::vector<size_t> candidates;
            for (size_t j = 0; j < pop_size; ++j) {
                if (i != j)
                    candidates.push_back(j);
            }

            // Shuffle and sample
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(candidates.begin(), candidates.end(), g);
            size_t num_to_check = std::min(sample_size, candidates.size());

            // Find minimum distance to sampled candidates
            u32 min_val = std::numeric_limits<u32>::max();
            size_t tid = static_cast<size_t>(omp_get_thread_num());

            for (size_t s = 0; s < num_to_check; ++s) {
                size_t j = candidates[s];
                u32 dis = determine_distance(
                    graph,
                    population[i],
                    population[j],
                    k,
                    mem_stacks[tid],
                    exec_spaces[tid]
                );

                min_val = std::min(min_val, dis);
            }

            min_distances[i] = min_val;
        }
    }
}

#endif
