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
#include "../utility/hungarian_algorithm.h"
#include "../datastructures/partition.h"
#include "block_conn.h"

#include "omp.h"

namespace GPU_HeiPa {



    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- utility / helpers: ------------------------------------------------------
    //! -------------------------------------------------------------------------------------------------


    struct KeyTuple {
        u32 key_count;
        u64 key;
    };
    
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
        const Kokkos::View<int*> &parent_ids,
        const Kokkos::View<Partition*> &population,
        u32 num_bits
    ) {

        u64 key = 0;
        for (size_t i = 0; i < parent_ids.size(); ++i) {
            u64 val = static_cast<u64>(population[parent_ids[i]].map(u));
            key |= (val & 0xFF) << (num_bits * i); //! do i even need this &0xFF ?
        }
        return key;

    }


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


    inline void assign_leftovers_fullyRandom(
            const Graph &graph,
            Partition &child,
            partition_t k,
            Kokkos::Cuda &exec_space
    ){
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);
        
        // assign remaining vertices
        Kokkos::parallel_for(
            "assign leftovers", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                
                if(child.map(u) == 5*k) {
                    
                    auto gen = random_pool.get_state();
                    partition_t id = static_cast<partition_t>(gen.urand(0, k));
                    random_pool.free_state(gen);
                    
                    child.map(u) = id;
                    Kokkos::atomic_fetch_add( &child.bweights(id), graph.weights(u) );
                    
                }
            }
        );
    
    }


    inline void assign_leftovers_gain(
            const Graph &graph,
            Partition &child,
            partition_t k,
            weight_t lmax,
            KokkosMemoryStack  &mem_stack,
            Kokkos::Cuda &exec_space
    ) {
        
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);
        
        
        BlockConn bc = init_BlockConn(graph, child, mem_stack, exec_space);


        Kokkos::parallel_for(
            "distribute leftovers", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {

                // calculate gain
                if( child.map(u) == 5*k) {
                    
                    auto gen = random_pool.get_state();
                    partition_t best_id = static_cast<partition_t>(gen.urand(0, k));
                    random_pool.free_state(gen);
                    
                    weight_t best_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        if (id == 5*k)
                            continue;
                        
                        weight_t w = bc.weights(i);

                        bool valid = (id != NULL_PART) & (id != HASH_RECLAIM); // single mask
                        
                        // Update best if it's a candidate and better
                        bool better = valid & (w > best_conn);
                        best_conn = better ? w : best_conn;
                        best_id = better ? id : best_id;
                    }

                    child.map(u) = best_id;
                    Kokkos::atomic_fetch_add( &child.bweights(best_id), graph.weights(u) );
                           
                }

            }
        );

        free_BlockConn(bc, mem_stack);

        return;

    }

    inline void assign_leftovers_gain_and_weight(
            const Graph &graph,
            Partition &child,
            partition_t k,
            weight_t lmax,
            KokkosMemoryStack  &mem_stack,
            f64 alpha,
            Kokkos::Cuda &exec_space
    ) {
        // this determines how much underloaded blocks are weighted
        // alpha = 0 -> only gain, big alpha -> only underloaded blocks
        
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);
        
        
        BlockConn bc = init_BlockConn(graph, child, mem_stack, exec_space);

        //! determine max gain
        DeviceScalarWeight max_gain = DeviceScalarWeight("highest gain value"); ;
        Kokkos::parallel_reduce(
            "determine max gain", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, bc.size),
            KOKKOS_LAMBDA(u32 index, weight_t &update) {
                weight_t val = bc.weights(index);
                if ( (val > update) && (bc.ids(index) != 5*k) ) {
                    update = val;
                }
            }, Kokkos::Max(max_gain)
        );

        Kokkos::parallel_for(
            "distribute leftovers", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {

                // calculate gain
                if( child.map(u) == 5*k) {
                    
                    auto gen = random_pool.get_state();
                    partition_t best_id = static_cast<partition_t>(gen.urand(0, k));
                    random_pool.free_state(gen);
                    
                    f64 best_score = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        if (id == 5*k)
                            continue;
                        
                        weight_t gain = bc.weights(i);

                        //! i think this is actually quite smart, because in the case that
                        //! child.bweights(id) > lmax, then the right part will get negative
                        //! -> i.e. overweight blocks are penalized
                        f64 my_score = gain + (alpha * max_gain() * ( static_cast<double>(lmax - child.bweights(id)) / static_cast<double>(lmax) ));

                        bool valid = (id != NULL_PART) & (id != HASH_RECLAIM); // single mask
                        
                        // Update best if it's a candidate and better
                        bool better = valid & (my_score > best_score);
                        best_score = better ? my_score : best_score;
                        best_id = better ? id : best_id;
                    }

                    child.map(u) = best_id;
                    Kokkos::atomic_fetch_add( &child.bweights(best_id), graph.weights(u) );
                           
                }

            }
        );

        free_BlockConn(bc, mem_stack);

        return;

    }

    inline void assign_leftovers_favorUnderloadedBlocks(
            const Graph &graph,
            Partition &child,
            partition_t k,
            weight_t lmax,
            KokkosMemoryStack  &mem_stack,
            Kokkos::Cuda &exec_space
    ){

        Kokkos::Random_XorShift64_Pool<> random_pool(12345);
        UnmanagedDeviceF64 distribution = UnmanagedDeviceF64((f64 *) get_chunk_back(mem_stack, sizeof(f64) * k), k);


        Kokkos::parallel_scan(
            "create distribution", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, k),
            KOKKOS_LAMBDA(partition_t id, f64 &update, bool final) {

                weight_t sum = 0;
                weight_t inverse_weight = ( lmax - child.bweights(id) );

                for(partition_t i = 0; i < k; ++i) {
                    sum += ( lmax - child.bweights(i) );
                }

                update += static_cast<double>(inverse_weight) / static_cast<double>(sum); 
                if(final) {
                    distribution(id) = update;
                }
                
            }
        );


        // assign remaining vertices
        Kokkos::parallel_for(
            "assign leftovers", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                
                if(child.map(u) == 5*k) {
                    
                    auto gen = random_pool.get_state();
                    f64 rand = gen.drand(0.0, 1.0);
                    random_pool.free_state(gen);
                    

                    for(u32 i = 0; i < k; ++i) {
                        if( ( rand < distribution(i) ) || ( i == (k-1) ) ) {
                            child.map(u) = i;
                            Kokkos::atomic_fetch_add( &child.bweights(i), graph.weights(u) );
                            break;
                        }
                    }

                }
            }
        );

        pop_back(mem_stack); //rm distribution

    }



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
        Kokkos::Cuda &exec_space
    
    ) {

        //setup: get the vectors onto the GPU
        auto parent_ids_device = Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            (int *) get_chunk_back(mem_stack, sizeof(int) * parent_ids.size()), 
            parent_ids.size()
        );
        auto population_device = Kokkos::View<Partition*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            (Partition *) get_chunk_back(mem_stack, sizeof(Partition) * population.size()), 
            population.size()
        );

        Kokkos::deep_copy(exec_space ,parent_ids_device, Kokkos::View<const int*>(parent_ids.data(), parent_ids.size()));
        Kokkos::deep_copy(exec_space ,population_device, Kokkos::View<const Partition*>(population.data(), population.size()));

        exec_space.fence();

        partition_t k_prime = next_power_of_two(k);
        u32 num_bits = bits_needed(k_prime);

        u64 num_buckets = static_cast<u64>(pow(k_prime , parent_ids.size())); 

        auto buckets = Kokkos::View<KeyTuple *, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            (KeyTuple *) get_chunk_back(mem_stack, sizeof(KeyTuple) * num_buckets), num_buckets 
        );

        
        Kokkos::parallel_for(
            "init buckets", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, num_buckets),
            KOKKOS_LAMBDA(u64 index) {
                buckets(index).key_count = 0;
                buckets(index).key = index;
            }
        );

        Kokkos::parallel_for(
            "fill buckets", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits); 
                Kokkos::atomic_fetch_add(&buckets( key ).key_count, 1);
            }
        );

        // sort descending based on key_count
        // after sorting, the k most frequent keys will be at the top
        // and you can query them via .key
        Kokkos::sort(exec_space, buckets, KOKKOS_LAMBDA(const KeyTuple& a, const KeyTuple& b) {
            return a.key_count > b.key_count;
        });


        partition_t local_extent = std::min(extent, k);
        if (local_extent < 1) {
            local_extent = 1;
        }

        // assign vertices of the backbone to the offspring
        Kokkos::parallel_for(
            "create new offspring", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                
                partition_t id;
                bool in_backbone = false;
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits); 
                
                for(partition_t j = 0; (j <= local_extent) && (!in_backbone) ; ++j) {
                    for(partition_t i = 0; i < k; ++i) {
                        if( key == buckets(i + (j*k ) ).key ){

                            if( j == 0)
                                id = i;
                            else
                                id = k - i - 1; //! "reverse assignment" from full buckets to underloaded partitions
                            
                            Kokkos::atomic_fetch_add( &child.bweights(id), graph.weights(u) );
                            in_backbone = true;
                            break;
                        }
                    }
                    
                }


                if( !in_backbone ){
                    id = 5*k ; //! mark as not assigned 
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

    inline u32 max_matching(
        const UnmanagedDeviceU32 &sim_matrix,
        const u32 k,
        Kokkos::Cuda &exec_space
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
        Kokkos::Cuda &exec_space
    ) {
        u32 distance;

        // build matrix
        auto sim_matrix = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * k * k), k * k);

        Kokkos::parallel_for(
            "init sim_matrix", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, k*k),
            KOKKOS_LAMBDA(u32 index) {
                sim_matrix(index) = 0;
            }
        );
        exec_space.fence();

        Kokkos::parallel_for(
            "fill matrix", 
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                u32 row = A.map(u);
                u32 col = B.map(u);

                Kokkos::atomic_fetch_add(&sim_matrix( row * k + col ), 1);
                
            }
        );
        exec_space.fence();

        // get maximum matching on matrix
        u32 similarity = max_matching(sim_matrix, k, exec_space);
        distance = graph.n - similarity;

        pop_back(mem_stack); //rm sim_matrix

        return distance;
    }


    inline u32 determine_min_distance_offspring(
        const Graph &graph,
        const std::vector<Partition> &population,
        size_t parents_curr,
        const Partition &offspring,
        partition_t k,
        std::vector<KokkosMemoryStack> &mem_stacks,
        std::vector<Kokkos::Cuda> &exec_spaces,
        size_t num_cpu_threads
    ) {

        // size_t pop_size = population.size();
        u32 min_distance = std::numeric_limits<u32>::max();
        

        #pragma omp parallel for reduction(min:min_distance) num_threads(static_cast<int>(num_cpu_threads))
        for(size_t i = 0; i < parents_curr; ++i) {
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
        std::vector<Kokkos::Cuda> &exec_spaces,
        size_t num_cpu_threads

    ) {

        //size_t pop_size = population.size();

        std::vector<u32> all_distances( parents_curr * parents_curr, std::numeric_limits<u32>::max());

        //! this can be trivially parallelized via
        //! #pragma omp parallel collapse
        #pragma omp parallel for collapse(2) num_threads(static_cast<int>(num_cpu_threads))
        for(size_t i = 0; i < parents_curr; ++i) {
            for(size_t j = i + 1; j < parents_curr; ++j) {
            
                size_t tid = static_cast<size_t>(  omp_get_thread_num() );
                u32 dis = determine_distance(
                        graph,
                        population[ i ],
                        population[ j ],
                        k,
                        mem_stacks[tid],
                        exec_spaces[tid]
                );

                all_distances[ i * parents_curr + j ] = dis;
                all_distances[ j * parents_curr + i ] = dis;
                
            }
        }

        for(u32 i = 0; i < parents_curr; ++i) {
            u32 min_val = std::numeric_limits<u32>::max();
            for(u32 j = 0; j < parents_curr; ++j) {
                min_val = std::min(min_val, all_distances[i * parents_curr + j]);
            }
            min_distances[ i ] = min_val;
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
        std::vector<Kokkos::Cuda> &exec_spaces,
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
        std::vector<Kokkos::Cuda> &exec_spaces,
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
                
                if(i != j)
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