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

#ifndef GPU_HEIPA_METIS_WRAPPER_H
#define GPU_HEIPA_METIS_WRAPPER_H

#include <metis.h>
#include <vector>
#include "../definitions.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    inline void metis_partition_host(HostGraph &g,
                                     int k,
                                     f64 imbalance,
                                     u64 seed,
                                     HostPartition &host_part) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "metis_partition", "total");
        idx_t nvtxs = static_cast<idx_t>(g.n);
        idx_t ncon = 1; 
        idx_t nparts = static_cast<idx_t>(k);
        
        static_assert(sizeof(idx_t) == sizeof(int32_t) || sizeof(idx_t) == sizeof(int64_t), "Unsupported METIS idx_t size");

        std::vector<idx_t> metis_xadj(g.n + 1);
        std::vector<idx_t> metis_adjncy(g.m);
        std::vector<idx_t> metis_vwgt;
        std::vector<idx_t> metis_adjwgt;
        std::vector<idx_t> metis_part(g.n);

        HEIPA_PROFILE_SCOPE("misc", "metis_partition", "type_conversion");
        for (vertex_t i = 0; i <= g.n; ++i) metis_xadj[i] = static_cast<idx_t>(g.neighborhood(i));
        for (vertex_t i = 0; i < g.m; ++i) metis_adjncy[i] = static_cast<idx_t>(g.edges_v(i));

        if (!g.uniform_vertex_weights) {
            metis_vwgt.resize(g.n);
            for (vertex_t i = 0; i < g.n; ++i) metis_vwgt[i] = static_cast<idx_t>(g.weights(i));
        }

        if (!g.uniform_edge_weights) {
            metis_adjwgt.resize(g.m);
            for (vertex_t i = 0; i < g.m; ++i) metis_adjwgt[i] = static_cast<idx_t>(g.edges_w(i));
        }

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_SEED] = static_cast<idx_t>(seed);
        options[METIS_OPTION_UFACTOR] = static_cast<idx_t>(imbalance * 1000.0); 

        idx_t objval;
        HEIPA_PROFILE_SCOPE("misc", "metis_partition", "metis_core_call");
        int result = METIS_PartGraphKway(&nvtxs,
                                         &ncon,
                                         metis_xadj.data(),
                                         metis_adjncy.data(),
                                         g.uniform_vertex_weights ? nullptr : metis_vwgt.data(),
                                         nullptr, 
                                         g.uniform_edge_weights ? nullptr : metis_adjwgt.data(),
                                         &nparts,
                                         nullptr, 
                                         nullptr, 
                                         options,
                                         &objval,
                                         metis_part.data());

        if (result != METIS_OK) {
            printf("METIS_PartGraphKway failed with error code %d\n", result);
        }

        for (vertex_t i = 0; i < g.n; ++i) host_part(i) = static_cast<partition_t>(metis_part[i]);
    }

    inline void metis_partition(Graph &g,
                                int k,
                                f64 imbalance,
                                u64 seed,
                                Partition &partition,
                                DeviceExecutionSpace &exec_space) {
        HEIPA_PROFILE_SCOPE("initial_partitioning", "metis_partition", "total");

        // METIS works on host, so we need to copy the graph to host
        idx_t nvtxs = static_cast<idx_t>(g.n);
        idx_t ncon = 1; // Number of balancing constraints
        idx_t nparts = static_cast<idx_t>(k);

        // METIS expectations:
        // xadj: CSR row offsets
        // adjncy: CSR column indices
        // vwgt: vertex weights (NULL if uniform)
        // adjwgt: edge weights (NULL if uniform)
        // part: resulting partition

        // 1. Prepare host buffers
        HostU32 host_xadj("host_xadj", g.n + 1);
        HostVertex host_adjncy("host_adjncy", g.m);
        HostWeight host_vwgt;
        HostWeight host_adjwgt;
        HostPartition host_part("host_part", g.n);

        // 2. Deep copy from device to host
        HEIPA_PROFILE_SCOPE("up/download", "metis_partition", "copy_to_host");
        Kokkos::deep_copy(exec_space, host_xadj, g.neighborhood);
        Kokkos::deep_copy(exec_space, host_adjncy, g.edges_v);

        if (!g.uniform_vertex_weights) {
            host_vwgt = HostWeight("host_vwgt", g.n);
            Kokkos::deep_copy(exec_space, host_vwgt, g.weights);
        }

        if (!g.uniform_edge_weights) {
            host_adjwgt = HostWeight("host_adjwgt", g.m);
            Kokkos::deep_copy(exec_space, host_adjwgt, g.edges_w);
        }
        exec_space.fence();

        // 3. Convert types to METIS idx_t if necessary
        // In many systems idx_t is int32_t or int64_t.
        // Our vertex_t/weight_t might differ, so we use intermediate vectors for safety if needed,
        // but since we want efficiency, we check if we can cast directly.

        static_assert(sizeof(idx_t) == sizeof(int32_t) || sizeof(idx_t) == sizeof(int64_t), "Unsupported METIS idx_t size");

        std::vector<idx_t> metis_xadj(g.n + 1);
        std::vector<idx_t> metis_adjncy(g.m);
        std::vector<idx_t> metis_vwgt;
        std::vector<idx_t> metis_adjwgt;
        std::vector<idx_t> metis_part(g.n);

        HEIPA_PROFILE_SCOPE("misc", "metis_partition", "type_conversion");
        for (vertex_t i = 0; i <= g.n; ++i) metis_xadj[i] = static_cast<idx_t>(host_xadj(i));
        for (vertex_t i = 0; i < g.m; ++i) metis_adjncy[i] = static_cast<idx_t>(host_adjncy(i));

        if (!g.uniform_vertex_weights) {
            metis_vwgt.resize(g.n);
            for (vertex_t i = 0; i < g.n; ++i) metis_vwgt[i] = static_cast<idx_t>(host_vwgt(i));
        }

        if (!g.uniform_edge_weights) {
            metis_adjwgt.resize(g.m);
            for (vertex_t i = 0; i < g.m; ++i) metis_adjwgt[i] = static_cast<idx_t>(host_adjwgt(i));
        }

        // 4. METIS options
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_SEED] = static_cast<idx_t>(seed);
        options[METIS_OPTION_UFACTOR] = static_cast<idx_t>(imbalance * 1000.0); // METIS ufactor is imbalance * 1000

        idx_t objval;

        // 5. Call METIS
        HEIPA_PROFILE_SCOPE("misc", "metis_partition", "metis_core_call");
        int result = METIS_PartGraphKway(&nvtxs,
                                         &ncon,
                                         metis_xadj.data(),
                                         metis_adjncy.data(),
                                         g.uniform_vertex_weights ? nullptr : metis_vwgt.data(),
                                         nullptr, // vsize: NULL
                                         g.uniform_edge_weights ? nullptr : metis_adjwgt.data(),
                                         &nparts,
                                         nullptr, // tpwgts: NULL for equal weights
                                         nullptr, // ubvec: NULL
                                         options,
                                         &objval,
                                         metis_part.data());

        if (result != METIS_OK) {
            // Handle error if necessary
            printf("METIS_PartGraphKway failed with error code %d\n", result);
        }

        // 6. Copy back to Partition object
        HEIPA_PROFILE_SCOPE("up/download", "metis_partition", "copy_back");
        for (vertex_t i = 0; i < g.n; ++i) host_part(i) = static_cast<partition_t>(metis_part[i]);
        auto device_subview = Kokkos::subview(partition.map, std::make_pair((size_t) 0, (size_t) g.n));
        Kokkos::deep_copy(exec_space, device_subview, host_part);
        exec_space.fence();

        // 7. Recalculate partition weights on device
        if (g.uniform_vertex_weights) {
            recalculate_weights<true>(partition, g, exec_space);
        } else {
            recalculate_weights<false>(partition, g, exec_space);
        }
    }
}

#endif //GPU_HEIPA_METIS_WRAPPER_H
