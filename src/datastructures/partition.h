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

#ifndef GPU_HEIPA_PARTITION_H
#define GPU_HEIPA_PARTITION_H

#include "mapping.h"
#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    struct Partition {
        vertex_t n = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        DevicePartition map;
        DeviceWeight bweights;
    };

    inline Partition initial_partition(const vertex_t t_n,
                                       const partition_t t_k,
                                       const weight_t t_lmax) {
        ScopedTimer t{"io", "initial_partition", "initialize"};
        Partition partition;

        partition.n = t_n;
        partition.k = t_k;
        partition.lmax = t_lmax;

        partition.map = DevicePartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "partition"), t_n);
        partition.bweights = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b_weights"), t_k);

        return partition;
    }

    inline void contract(Partition &partition,
                         const Mapping &mapping) {
        ScopedTimer t{"coarsening", "partition", "contract"};
        // reset activity
        DevicePartition temp_device_partition = DevicePartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "partition"), partition.n);

        Kokkos::parallel_for("initialize", mapping.old_n, KOKKOS_LAMBDA(const vertex_t u) {
            // TODO: multiple threads may write the same value, is this bad?
            vertex_t u_new = mapping.mapping(u);
            temp_device_partition(u_new) = mapping.mapping(u);
        });
        Kokkos::fence();

        std::swap(partition.map, temp_device_partition);
    }
}

#endif //GPU_HEIPA_PARTITION_H
