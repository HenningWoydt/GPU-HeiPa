/*******************************************************************************
 * MIT License
 *
 * This file is part of GPU-HeiPa.
 *
 * Copyright (C) 2025 Henning Woydt <henning.woydt@informatik.uni-heidelberg.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef GPU_HEIPA_DISTANCES_SHRINKING_H
#define GPU_HEIPA_DISTANCES_SHRINKING_H

#include <Kokkos_Core.hpp>
#include <bitset>
#include <numeric>
#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>
#include <Kokkos_Random.hpp>

#include "../definitions.h"
#include "../utility/hungarian_algorithm.h"
#include "../datastructures/partition.h"
#include "block_conn.h"

#include "../utility/memetic_helper.h"
#include "omp.h"


namespace GPU_HeiPa {


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


    inline void determine_min_distances_population(
        const Graph &graph,
        const std::vector<Partition> &population,
        size_t parents_curr, // amount of non-offspring individuals in the population
        std::vector<u32> &min_distances,
        partition_t k,
        std::vector<KokkosMemoryStack> &mem_stacks,
        std::vector<DeviceExecutionSpace> &exec_spaces,
        size_t num_cpu_threads,
        std::vector<bool> &active_b

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
            if( !active_b[i] ) {continue;}

            u32 min_val = std::numeric_limits<u32>::max();
            for (u32 j = 0; j < parents_curr; ++j) {
                
                if( all_distances[i * parents_curr + j] == 0) {
                    active_b[j] = false; // deactivate duplicate
                    continue;
                }

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