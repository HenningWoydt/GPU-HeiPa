#ifndef GPU_HEIPA_HELPERS_H
#define GPU_HEIPA_HELPERS_H

#include "definitions.h"
#include "../refinement/block_conn.h"
#include "../datastructures/partition.h"
#include "../utility/hungarian_algorithm.h"

namespace GPU_HeiPa {


    enum class PopulationManagement {
        steadystate,
        shrinking // #partitions == (level + 1)
    };



    //! ----------------------------------------------------------------

    


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
