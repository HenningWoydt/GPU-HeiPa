#ifndef GPU_HEIPA_MEMETIC_REFINEMENT_H
#define GPU_HEIPA_MEMETIC_REFINEMENT_H

#include <Kokkos_Core.hpp>
#include <bitset>
#include <numeric>
#include <unordered_set>
#include <random>
#include <vector>

#include "../utility/definitions.h"
#include "../datastructures/partition.h"


namespace GPU_HeiPa {


struct KeyTuple {
    u32 key_count;
    u64 key;
};

    inline int tournament_selection(
        const std::vector<weight_t> &fitness_values,
        u32 tournament_size
    ) {
        
        // get num_parents random numbers between [0, num_individuals)
        size_t num_individuals = fitness_values.size();
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


    inline Partition backbone_based_crossover(
        const Graph &graph,
        const std::vector<int> &parent_ids,       
        const std::vector<Partition> &population, 
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack 
    
    ) {

        Partition child;
        child = initialize_partition( graph.n , k, lmax, mem_stack);

        //setup: get the vectors onto the GPU
        auto parent_ids_device = Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            (int *) get_chunk_back(mem_stack, sizeof(int) * parent_ids.size()), 
            parent_ids.size()
        );
        auto population_device = Kokkos::View<Partition*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            (Partition *) get_chunk_back(mem_stack, sizeof(Partition) * population.size()), 
            population.size()
        );

        Kokkos::deep_copy(parent_ids_device, Kokkos::View<const int*>(parent_ids.data(), parent_ids.size()));
        Kokkos::deep_copy(population_device, Kokkos::View<const Partition*>(population.data(), population.size()));

        partition_t k_prime = next_power_of_two(k);
        u32 num_bits = bits_needed(k_prime);

        u64 num_buckets = static_cast<u64>(pow(k_prime , parent_ids.size())); 

        auto buckets = Kokkos::View<KeyTuple *, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            (KeyTuple *) get_chunk_back(mem_stack, sizeof(KeyTuple) * num_buckets), num_buckets 
        );
      
        
        Kokkos::parallel_for(
            "init buckets", num_buckets,
            KOKKOS_LAMBDA(u64 index) {
                buckets(index).key_count = 0;
                buckets(index).key = index;
            }
        );

        Kokkos::parallel_for(
            "fill buckets", graph.n,
            KOKKOS_LAMBDA(vertex_t u) {
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits); 
                Kokkos::atomic_fetch_add(&buckets( key ).key_count, 1);
            }
        );

        // sort descending based on key_count
        // after sorting, the k most frequent keys will be at the top
        // and you can query them via .key
        Kokkos::sort(buckets, KOKKOS_LAMBDA(const KeyTuple& a, const KeyTuple& b) {
            return a.key_count > b.key_count;
        });



        Kokkos::parallel_for(
            "create new offspring", graph.n,
            KOKKOS_LAMBDA(vertex_t u) {
                partition_t id;
                bool in_backbone = false;
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits); 
                
                for(partition_t i = 0; i < k; ++i) {
                    if( key == buckets(i).key ){
                        id = i;
                        in_backbone = true;
                        break;
                    }
                }

                if( !in_backbone ){
                    //! this kind of works quite well for some reason lol
                    id = (u % k); //TODO: determine random id between 0 and k-1
                }

                child.map(u) = id;
                Kokkos::atomic_fetch_add( &child.bweights(id), graph.weights(u) );
            }
        );

        
        pop_back(mem_stack); //rm buckets
        pop_back(mem_stack); //rm population
        pop_back(mem_stack); //rm parent_ids


        return child;

    }



}

#endif