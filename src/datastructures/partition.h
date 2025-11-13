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
#include "graph.h"
#include "../utility/definitions.h"
#include "../utility/util.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    struct Partition {
        vertex_t n = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        UnmanagedDevicePartition map;
        UnmanagedDevicePartition temp_map;
        UnmanagedDeviceWeight bweights;
    };

    inline Partition initialize_partition(const vertex_t t_n,
                                          const partition_t t_k,
                                          const weight_t t_lmax,
                                          KokkosMemoryStack &mem_stack) {
        Partition partition;

        partition.n = t_n;
        partition.k = t_k;
        partition.lmax = t_lmax;

        auto *map_ptr = (partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * t_n);
        auto *temp_map_ptr = (partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * t_n);
        auto *bweights_ptr = (weight_t *) get_chunk_front(mem_stack, sizeof(weight_t) * t_k);
        partition.map = UnmanagedDevicePartition(map_ptr, t_n);
        partition.temp_map = UnmanagedDevicePartition(temp_map_ptr, t_n);
        partition.bweights = UnmanagedDeviceWeight(bweights_ptr, t_k);

        Kokkos::deep_copy(partition.map, 0);
        Kokkos::deep_copy(partition.bweights, 0);

        return partition;
    }

    inline void free_partition(const Partition &partition,
                               KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack);
        pop_front(mem_stack);
        pop_front(mem_stack);
    }

    inline void contract(Partition &partition,
                         const Mapping &mapping) {
        ScopedTimer t{"contraction", "partition", "contract"};
        Kokkos::parallel_for("initialize", mapping.old_n, KOKKOS_LAMBDA(const vertex_t u) {
            // TODO: multiple threads may write the same value, is this bad?
            vertex_t u_new = mapping.mapping(u);
            partition.temp_map(u_new) = partition.map(u);
        });

        std::swap(partition.map, partition.temp_map);

        KOKKOS_PROFILE_FENCE();
    }

    inline void uncontract(Partition &partition,
                           const Mapping &mapping) {
        ScopedTimer _t("uncontraction", "partition", "uncontract");

        // reset activity
        Kokkos::parallel_for("initialize", mapping.old_n, KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t new_v = mapping.mapping(u);
            partition.temp_map(u) = partition.map(new_v);
        });
        std::swap(partition.map, partition.temp_map);

        KOKKOS_PROFILE_FENCE();
    }

    inline void recalculate_weights(Partition &partition,
                                    const Graph &g) {
        ScopedTimer _t("initial_partitioning", "Partition", "recalculate_weights");

        // reset weights
        Kokkos::deep_copy(partition.bweights, 0);

        // set weights
        Kokkos::parallel_for("set_block_weights", g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = partition.map(u);
            Kokkos::atomic_add(&partition.bweights(u_id), g.weights(u));
        });
        KOKKOS_PROFILE_FENCE();
    }

    inline weight_t max_weight(const Partition &partition) {
        weight_t max_val = 0;

        Kokkos::parallel_reduce("compute_max_weight", partition.k, KOKKOS_LAMBDA(const partition_t i, weight_t &local_max) {
                                    if (partition.bweights(i) > local_max) {
                                        local_max = partition.bweights(i);
                                    }
                                },
                                Kokkos::Max<weight_t>(max_val)
        );

        return max_val;
    }

    inline partition_t n_empty_blocks(const Partition &partition) {
        partition_t s = 0;

        Kokkos::parallel_reduce("compute_max_weight", partition.k, KOKKOS_LAMBDA(const partition_t i, partition_t &local_s) {
                                    if (partition.bweights(i) == 0) {
                                        local_s += 1;
                                    }
                                },
                                Kokkos::Sum<partition_t>(s)
        );

        return s;
    }

    inline partition_t n_oload_blocks(const Partition &partition) {
        partition_t s = 0;

        Kokkos::parallel_reduce("compute_max_weight", partition.k, KOKKOS_LAMBDA(const partition_t i, partition_t &local_s) {
                                    if (partition.bweights(i) > partition.lmax) {
                                        local_s += 1;
                                    }
                                },
                                Kokkos::Sum<partition_t>(s)
        );

        return s;
    }

    inline weight_t sum_oload_weight(const Partition &partition) {
        weight_t s = 0;

        Kokkos::parallel_reduce("compute_max_weight", partition.k, KOKKOS_LAMBDA(const partition_t i, weight_t &local_s) {
                                    if (partition.bweights(i) > partition.lmax) {
                                        local_s += partition.bweights(i) - partition.lmax;
                                    }
                                },
                                Kokkos::Sum<weight_t>(s)
        );

        return s;
    }

    inline void copy_into(Partition &dst, const Partition &src, u32 n) {
        dst.n = src.n;
        dst.k = src.k;
        dst.lmax = src.lmax;

        auto rN = std::make_pair<size_t, size_t>(0, n);
        Kokkos::deep_copy(Kokkos::subview(dst.map, rN), Kokkos::subview(src.map, rN));
        Kokkos::deep_copy(dst.bweights, src.bweights);
    }

    struct PartitionHost {
        vertex_t n = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        HostPartition map;
        HostWeight bweights;
    };

    inline PartitionHost to_host_partition(const Partition &partition) {
        PartitionHost host_partition;

        host_partition.n = partition.n;
        host_partition.k = partition.k;
        host_partition.lmax = partition.lmax;

        host_partition.map = HostPartition("partition", partition.n);
        host_partition.bweights = HostWeight("b_weights", partition.k);
        Kokkos::deep_copy(host_partition.map, partition.map);
        Kokkos::deep_copy(host_partition.bweights, partition.bweights);
        Kokkos::fence();

        return host_partition;
    }
}

#endif //GPU_HEIPA_PARTITION_H
