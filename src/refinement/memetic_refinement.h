#ifndef GPU_HEIPA_MEMETIC_REFINEMENT_H
#define GPU_HEIPA_MEMETIC_REFINEMENT_H

#include <Kokkos_Core.hpp>
#include <bitset>

#include "../utility/definitions.h"
#include "../datastructures/partition.h"


namespace GPU_HeiPa {

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

        UnmanagedDeviceU64 keys = UnmanagedDeviceU64((u64 *) get_chunk_back(mem_stack, sizeof(u64) * graph.n), graph.n);
        Kokkos::parallel_for(
            "fill keys array", graph.n,
            KOKKOS_LAMBDA(vertex_t u) {
                keys(u) = determine_key(u, parent_ids_device, population_device); 
            }
        );

        /*
        */

        // reduce by key
        u32 num_buckets = static_cast<u32>(pow(16, parent_ids.size())); //TODO: no static assignment of 256
        UnmanagedDeviceU32 buckets = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * num_buckets), num_buckets);
        
        Kokkos::parallel_for(
            "count keys", graph.n,
            KOKKOS_LAMBDA(vertex_t u) {
                Kokkos::atomic_fetch_add(&buckets(keys(u)), 1 );
            }
        );

        // Copy buckets from device to host
        std::vector<u32> host_buckets(num_buckets);
        Kokkos::deep_copy(Kokkos::View<u32*, Kokkos::HostSpace>(host_buckets.data(), num_buckets), buckets);

        std::vector<std::pair<u32, u32>> bucket_counts(num_buckets);
        for (u32 i = 0; i < num_buckets; ++i) {
            bucket_counts[i] = {host_buckets[i], i};
        }

        std::partial_sort(
            bucket_counts.begin(),
            bucket_counts.begin() + k,
            bucket_counts.end(),
            [](const std::pair<u32, u32> &a, const std::pair<u32, u32> &b) {
                return a.first > b.first;
            }
        );
        std::vector<u32> top_k_indices(k);
        for (u32 i = 0; i < k; ++i) {
            top_k_indices[i] = bucket_counts[i].second;
        }
        
        // Print host_buckets
        std::cout << "host_buckets: ";
        for (u32 i = 0; i < num_buckets; ++i) {
            std::cout << host_buckets[i] << " ";
        }
        std::cout << std::endl;

        // Print top_k_indices
        std::cout << "top_k_indices: ";
        for (u32 i = 0; i < k; ++i) {
            std::cout << top_k_indices[i] << " ";
            std::cout << "with value: " << host_buckets[top_k_indices[i]] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "top_k_indices (binary): ";
        for (u32 i = 0; i < k; ++i) {
            std::bitset<32> binary(top_k_indices[i]);
            std::cout << binary.to_string().substr(32 - 4 * parent_ids.size()) << std::endl;
        }
        std::cout << std::endl;

        pop_back(mem_stack);

        
        pop_back(mem_stack); //rm keys
        pop_back(mem_stack); //rm population
        pop_back(mem_stack); //rm parent_ids

        //! stub for now:
        /*
        
        partition_t k_capture = k;
        Kokkos::parallel_for("create offspring", graph.n,
            KOKKOS_LAMBDA( u32 u ) {
                child.map(u) = u % k_capture;
                Kokkos::atomic_fetch_add(&child.bweights(u % k_capture), graph.weights(u) );
            }
        );
        Kokkos::fence();

        */
        return child;

    }



}

#endif