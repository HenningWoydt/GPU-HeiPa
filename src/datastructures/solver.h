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
#include "mapping.h"
#include "partition.h"
#include "../coarsening/two_hop_matching.h"
#include "../refinement/jet_label_propagation.h"
#include "../initial_partitioning/kaffpa_initial_partitioning.h"
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

        std::vector<Graph> graphs;
        std::vector<Mapping> mappings;

        Partition partition;

        weight_t initial_edge_cut = 0;
        weight_t initial_max_block_weight = 0;

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

            return {};
        }

    private:
        void internal_solve() {
            initialize();

            const partition_t c = 8;

            u32 level = 0;
            while (graphs.back().n > c * k) {
                matching();

                // TODO: this should not be necessary
                if ((f64) mappings.back().coarse_n > TwoHopMatcher().threshold * (f64) mappings.back().old_n) {
                    mappings.pop_back();
                    break;
                }

                coarsening();

                // std::cout << "level " << level << " " << graphs.back().n << " " << graphs.back().m << std::endl;

                level += 1;
            }

            u32 max_level = level;
            initial_partitioning();

            while (!mappings.empty()) {
                level -= 1;

                // std::cout << "level " << level << " " << graphs.back().n << " " << graphs.back().m << " " << max_weight(partition) << " " << lmax << std::endl;

                uncoarsening();
                refinement(max_level, level);
            }
        }

        void initialize() {
            host_g = from_file(config.graph_in);

            n = host_g.n;
            m = host_g.m;
            k = config.k;
            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));

            graphs.emplace_back(from_HostGraph(host_g));

            partition = initialize_partition(n, k, lmax);
            Kokkos::deep_copy(partition.map, 0);
            Kokkos::deep_copy(partition.bweights, 0);
            Kokkos::fence();

            assert_state_pre_partition(graphs.back());
        }

        void matching() {
            TwoHopMatcher thm = initialize_thm(graphs.back().n, graphs.back().m, lmax);
            mappings.emplace_back(two_hop_matcher_get_mapping(thm, graphs.back(), partition));

            assert_state_pre_partition(graphs.back());
        }

        void coarsening() {
            graphs.emplace_back(from_Graph_Mapping(graphs.back(), mappings.back()));
            contract(partition, mappings.back());

            assert_state_pre_partition(graphs.back());
        }

        void initial_partitioning() {
            kaffpa_initial_partition(graphs.back(), (int) k, config.imbalance, (u32) config.seed, partition);
            recalculate_weights(partition, graphs.back());

            initial_edge_cut = edge_cut(graphs.back(), partition);
            initial_max_block_weight = max_weight(partition);

            assert_state_after_partition(graphs.back(), partition, config.k);
        }

        void refinement(u32 max_level, u32 level) {
            refine(graphs.back(), partition, k, lmax, max_level, level);
            assert_state_after_partition(graphs.back(), partition, config.k);
        }

        void uncoarsening() {
            uncontract(partition, mappings.back());

            graphs.pop_back();
            mappings.pop_back();

            assert_state_after_partition(graphs.back(), partition, config.k);
        }
    };
}

#endif //GPU_HEIPA_SOLVER_H
