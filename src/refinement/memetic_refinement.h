#ifndef GPU_HEIPA_MEMETIC_REFINEMENT_H
#define GPU_HEIPA_MEMETIC_REFINEMENT_H

#include <Kokkos_Core.hpp>
#include <bitset>
#include <numeric>

#include "../utility/definitions.h"
#include "../datastructures/partition.h"


namespace GPU_HeiPa {


struct KeyTuple {
    u32 key_count;
    u64 key;
};

    inline int tournament_selection() {

        // get num_parents random numbers between [0, num_individuals)

        // find smallest index for curr_edge_cut amongst these numbers

        // return this index

        
        return 0;
    }

    
    
    KOKKOS_FUNCTION u64 determine_key(
        vertex_t u,
        const Kokkos::View<int*> &parent_ids,
        const Kokkos::View<Partition*> &population
    ) {

        u64 key = 0;
        for (size_t i = 0; i < parent_ids.size(); ++i) {
            u64 val = static_cast<u64>(population[parent_ids[i]].map(u));
            key |= (val & 0xFF) << (4 * i);
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



        Partition child;
        child = initialize_partition( graph.n , k, lmax, mem_stack);

        u64 num_buckets = pow(k, parent_ids.size()); //! if k is not a power of 2, get next biggest power of two!

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
                u64 key = determine_key(u, parent_ids_device, population_device); 
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
                u64 key = determine_key(u, parent_ids_device, population_device); 
                
                for(size_t i = 0; i < k; ++i) {
                    if( key == buckets(i).key ){
                        id = i;
                        in_backbone = true;
                        break;
                    }
                }

                if( !in_backbone )
                    id = (u % k); //TODO: determine random id between 0 and k-1

                child.map(u) = id;
            }
        );

        
        pop_back(mem_stack); //rm buckets
        pop_back(mem_stack); //rm population
        pop_back(mem_stack); //rm parent_ids


        return child;

    }



}

#endif