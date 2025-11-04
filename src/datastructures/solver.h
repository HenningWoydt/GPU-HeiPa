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

#ifndef GPU_HEIPA_SOLVER_H
#define GPU_HEIPA_SOLVER_H

#include <vector>

#include "graph.h"
#include "host_graph.h"
#include "kokkos_memory_stack.h"
#include "mapping.h"
#include "partition.h"
#include "../coarsening/two_hop_matching.h"
#include "../refinement/jet_label_propagation.h"
#include "../refinement/jet_label_propagation_list.h"
// #include "../initial_partitioning/kaffpa_initial_partitioning.h"
#include "../initial_partitioning/metis_initial_partitioning.h"
#include "../utility/definitions.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/edge_cut.h"

namespace GPU_HeiPa {
    class Solver {
    public:
        Configuration config;
        std::chrono::time_point<std::chrono::system_clock> sp;

        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        HostGraph host_g;
        KokkosMemoryStack mem_stack;
        KokkosMemoryStack small_mem_stack;

        std::vector<Graph> graphs;
        std::vector<Mapping> mappings;

        Partition partition;

        weight_t initial_edge_cut = 0;
        weight_t initial_max_block_weight = 0;

        f64 io_ms = 0.0;
        f64 coarsening_ms = 0.0;
        f64 contraction_ms = 0.0;
        f64 initial_partitioning_ms = 0.0;
        f64 uncontraction_ms = 0.0;
        f64 refinement_ms = 0.0;

        explicit Solver(Configuration t_config) : config(std::move(t_config)) {
            sp = get_time_point();
        }

        std::vector<partition_t> solve() {
            internal_solve();

            ScopedTimer _t_write("io", "Solver", "write_partition");
            HostPartition host_partition = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), graphs.back().n);
            Kokkos::deep_copy(host_partition, partition.map);

            write_partition(host_partition, graphs.back().n, config.mapping_out);
            _t_write.stop();

            auto ep = get_time_point();
            f64 duration = get_seconds(sp, ep);

            std::cout << "Total time        : " << duration << std::endl;
            std::cout << "#Nodes            : " << graphs.back().n << std::endl;
            std::cout << "#Edges            : " << graphs.back().m << std::endl;
            std::cout << "k                 : " << config.k << std::endl;
            std::cout << "Lmax              : " << lmax << std::endl;
            std::cout << "Init. edge-cut    : " << initial_edge_cut << std::endl;
            std::cout << "Init. max block w : " << initial_max_block_weight << std::endl;
            std::cout << "Final edge-cut    : " << edge_cut(graphs.back(), partition) << std::endl;
            std::cout << "Final max block w : " << max_weight(partition) << std::endl;

            size_t n_empty_partitions = 0;
            size_t n_overloaded_partitions = 0;
            weight_t sum_too_much = 0;
            PartitionHost partition_host = to_host_partition(partition);
            for (partition_t id = 0; id < config.k; ++id) {
                n_empty_partitions += partition_host.bweights(id) == 0;
                n_overloaded_partitions += partition_host.bweights(id) > lmax;
                sum_too_much += std::max((weight_t) 0, partition_host.bweights(id) - lmax);
            }
            std::cout << "#empty partitions : " << n_empty_partitions << std::endl;
            std::cout << "#oload partitions : " << n_overloaded_partitions << std::endl;
            std::cout << "Sum oload weights : " << sum_too_much << std::endl;
            std::cout << "IO            : " << io_ms << std::endl;
            std::cout << "Coarsening    : " << coarsening_ms << std::endl;
            std::cout << "Contraction   : " << contraction_ms << std::endl;
            std::cout << "Init. Part.   : " << initial_partitioning_ms << std::endl;
            std::cout << "Uncontraction : " << uncontraction_ms << std::endl;
            std::cout << "Refinement    : " << refinement_ms << std::endl;

            free_partition(partition, mem_stack);

            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            assert_is_empty(mem_stack);
            destroy(mem_stack);
            assert_is_empty(small_mem_stack);
            destroy(small_mem_stack);

            return {};
        }

    private:
        void internal_solve() {
            initialize();

            const partition_t c = 8;

            u32 level = 0;
            while (graphs.back().n > c * k) {
                coarsening();
                contraction();

                level += 1;
            }

            u32 max_level = level - 1;
            initial_partitioning();

            while (!mappings.empty()) {
                level -= 1;

                uncontraction();
                refinement(max_level, level);
            }
        }

        void initialize() {
            auto p = get_time_point();
            host_g = from_file(config.graph_in);
            // Main stack: Graph + coarsening overhead
            mem_stack = initialize_kokkos_memory_stack(
                20 * host_g.n * sizeof(vertex_t) + // 20% buffer for vertices
                10 * host_g.m * sizeof(vertex_t),  // Graph + coarsening overhead
                "Stack"
            );

            // Small stack: Partitioning + temporary arrays
            small_mem_stack = initialize_kokkos_memory_stack(
                20 * host_g.n * sizeof(vertex_t) + // 33% buffer for partitioning
                2 * host_g.m * sizeof(vertex_t),
                "SmallStack"
            );

            n = host_g.n;
            m = host_g.m;
            k = config.k;
            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));

            graphs.emplace_back(from_HostGraph(host_g, mem_stack));

            partition = initialize_partition(n, k, lmax, mem_stack);

            io_ms += get_milli_seconds(p, get_time_point());

            assert_state_pre_partition(graphs.back());
        }

        void coarsening() {
            auto p = get_time_point();

            mappings.emplace_back(two_hop_matcher_get_mapping(graphs.back(), partition, lmax, mem_stack, small_mem_stack));

            Kokkos::fence();
            coarsening_ms += get_milli_seconds(p, get_time_point());

            assert_state_pre_partition(graphs.back());
        }

        void contraction() {
            auto p = get_time_point();

            graphs.emplace_back(from_Graph_Mapping(graphs.back(), mappings.back(), mem_stack, small_mem_stack));
            contract(partition, mappings.back());

            Kokkos::fence();
            contraction_ms += get_milli_seconds(p, get_time_point());

            assert_state_pre_partition(graphs.back());
        }

        void initial_partitioning() {
            auto p = get_time_point();

            // Use METIS for initial partitioning
            metis_initial_partition(graphs.back(), (int) k, config.imbalance, config.seed, partition);
            
            recalculate_weights(partition, graphs.back());

            initial_edge_cut = edge_cut(graphs.back(), partition);
            initial_max_block_weight = max_weight(partition);

            Kokkos::fence();
            initial_partitioning_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(graphs.back(), partition, config.k);
        }

        void refinement(u32 max_level, u32 level) {
            auto p = get_time_point();

            weight_t temp_lmax = (weight_t) std::ceil((1.0 + config.imbalance + (config.imbalance * ((f64) level / (f64) max_level))) * ((f64) host_g.g_weight / (f64) config.k));

            // refine(graphs.back(), partition, k, temp_lmax, max_level, level);
            // refine_list(graphs.back(), partition, k, temp_lmax, max_level, level);

            Kokkos::fence();
            refinement_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(graphs.back(), partition, config.k);
        }

        void uncontraction() {
            auto p = get_time_point();

            uncontract(partition, mappings.back());

            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();

            Kokkos::fence();
            uncontraction_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(graphs.back(), partition, config.k);
        }
    };
}

#endif //GPU_HEIPA_SOLVER_H
