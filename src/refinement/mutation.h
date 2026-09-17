#ifndef GPU_HEIPA_MUTATION_H
#define GPU_HEIPA_MUTATION_H

#include <Kokkos_Core.hpp>

#include "../utility/definitions.h"
#include "../datastructures/partition.h"
#include "../datastructures/solver.h"



namespace GPU_HeiPa {

    
    void mutate_individual(
        Partition &individual,
        Graph &graph,
        partition_t t_k,
        f64 imbalance,
        u64 seed,
        bool t_use_ultra,
        KokkosMemoryStack &mem_stack,
        DeviceExecutionSpace &exec_space
    ){

        Solver(
            graph,
            t_k,
            imbalance,
            seed,
            t_use_ultra,
            individual,
            mem_stack,
            exec_space
        );

        return;
    }


}

#endif