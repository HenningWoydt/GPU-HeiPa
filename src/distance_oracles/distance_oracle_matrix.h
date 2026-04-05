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

#ifndef GPU_HEIPA_DISTANCE_ORACLE_MATRIX_H
#define GPU_HEIPA_DISTANCE_ORACLE_MATRIX_H

#include "../datastructures/kokkos_memory_stack.h"
#include "../utility/definitions.h"
#include "../utility/util.h"

namespace GPU_HeiPa {
    struct DistanceOracleMatrix {
        partition_t k = 0;
        partition_t l = 0;
        UnmanagedDeviceWeight w_mtx;
    };

    inline DistanceOracleMatrix initialize_distance_oracle_matrix(partition_t k,
                                                                  std::vector<partition_t> &hierarchy,
                                                                  std::vector<weight_t> &distance,
                                                                  KokkosMemoryStack &mem_stack,
                                                                  DeviceExecutionSpace &exec_space) {
        DistanceOracleMatrix d_oracle;

        d_oracle.k = k;
        d_oracle.w_mtx = UnmanagedDeviceWeight((weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * k * k), k * k);

        const size_t l = hierarchy.size();
        UnmanagedDevicePartition dev_hierarchy = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * l), l);
        UnmanagedDeviceWeight dev_distance = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * l), l);

        // Create host mirrors around original std:: vectors (no allocation)
        auto host_hierarchy = Kokkos::View<partition_t *, Kokkos::HostSpace>(hierarchy.data(), l);
        auto host_distance = Kokkos::View<weight_t *, Kokkos::HostSpace>(distance.data(), l);

        // Deep copy to device
        Kokkos::deep_copy(exec_space, dev_hierarchy, host_hierarchy);
        Kokkos::deep_copy(exec_space, dev_distance, host_distance);

        // Fill w_mtx and h_mtx in parallel
        Kokkos::parallel_for("build_distance_oracle", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k * k), KOKKOS_LAMBDA(const u32 idx) {
            partition_t i = idx / k;
            partition_t j = idx % k;
            if (i == j) {
                d_oracle.w_mtx(idx) = 0;
                return;
            }

            partition_t level = 0;
            partition_t group_size = 1;
            for (; level < l; ++level) {
                group_size *= dev_hierarchy(level);
                if ((i / group_size) == j / group_size) {
                    break;
                }
            }

            d_oracle.w_mtx(idx) = dev_distance(level);
        });
        exec_space.fence();


        pop_back(mem_stack);
        pop_back(mem_stack);

        return d_oracle;
    }

    inline void free_distance_oracle_matrix(DistanceOracleMatrix &d_oracle, KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack);
    }

    KOKKOS_INLINE_FUNCTION
    weight_t get(const DistanceOracleMatrix &d_oracle, const partition_t u_id, const partition_t v_id) {
        return d_oracle.w_mtx(u_id * d_oracle.k + v_id);
    }
}

#endif //GPU_HEIPA_DISTANCE_ORACLE_MATRIX_H
