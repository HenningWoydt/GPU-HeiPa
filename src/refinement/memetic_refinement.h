#ifndef GPU_HEIPA_MEMETIC_REFINEMENT_H
#define GPU_HEIPA_MEMETIC_REFINEMENT_H

#include <Kokkos_Core.hpp>

#include "../utility/definitions.h"
#include "../datastructures/partition.h"


namespace GPU_HeiPa {

    inline int tournament_selection() {

        // get num_parents random numbers between [0, num_individuals)

        // find smallest index for curr_edge_cut amongst these numbers

        // return this index

        
        return 0;
    }

    inline Partition backbone_based_crossover(
        Graph &graph,
        std::vector<int> parent_ids, 
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack 
    
    ) {

        Partition child;
        //! child will have the size of the current graph
        child = initialize_partition( graph.n , k, lmax, mem_stack);
        // need "partitions" array
        // perform crossover on the parents:
        // get deviceview for all vertices:
        
        // UnmanagedDeviceScalarU64 keys = UnmanagedDeviceU64((u64 *) get_chunk_back(mem_stack, sizeof(u64) * lp.n), lp.n);
        // get key for each vertex
        // reduce by key
        //! stub for now:
        partition_t k_capture = k;
        Kokkos::parallel_for("create offspring", graph.n,
            KOKKOS_LAMBDA( u32 u ) {
                child.map(u) = u % k_capture;
                Kokkos::atomic_fetch_add(&child.bweights(u % k_capture), graph.weights(u) );
            }
        );
        Kokkos::fence();
        return child;

    }



}

#endif