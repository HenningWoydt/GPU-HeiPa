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
#include "../initial_partitioning/kaffpa_initial_partitioning.h"
#include "../utility/definitions.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"

namespace GPU_HeiPa {
    class Solver {
    public:
        Configuration config;
        weight_t lmax = 0;

        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;

        HostGraph host_g;

        std::vector<Graph> graphs;
        std::vector<Mapping> mappings;

        Partition partition;

        explicit Solver(Configuration t_config) : config(std::move(t_config)) {
        }

        std::vector<partition_t> solve() {
            internal_solve();

            ScopedTimer _t_write("io", "Solver", "write_partition");
            HostPartition host_partition = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), graphs.back().n);
            Kokkos::deep_copy(host_partition, partition.map);

            write_partition(host_partition, graphs.back().n, config.mapping_out);
            _t_write.stop();

            std::string config_JSON = config.to_JSON();
            std::string profile_JSON = Profiler::instance().to_JSON();

            // Combine manually into a single JSON string
            std::string combined_JSON = "{\n";
            combined_JSON += "  \"config\": " + config_JSON + ",\n";
            combined_JSON += "  \"profile\": " + profile_JSON + "\n";
            combined_JSON += "}";

            std::cout << combined_JSON << std::endl;

            return {};
        }

    private:
        void internal_solve() {
            initialize();

            const partition_t c = 8;

            u32 level = 0;
            while (graphs.back().n > c * k) {
                matching();

                if (mappings.back().old_n == mappings.back().coarse_n) {
                    mappings.pop_back();
                    break;
                }

                coarsening();

                std::cout << "level " << level << " " << graphs.back().n << " " << graphs.back().m << std::endl;

                level += 1;
            }

            initial_partitioning();

            while (!mappings.empty()) {
                level -= 1;

                std::cout << "level " << level << " " << graphs.back().n << " " << graphs.back().m << " " << max_weight(partition) << " " << lmax << std::endl;

                uncoarsening();
                refinement();
            }
        }

        void initialize() {
            host_g = from_file(config.graph_in);

            n = host_g.n;
            m = host_g.m;
            k = config.k;
            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));

            graphs.emplace_back(from_HostGraph(host_g));

            partition = initial_partition(n, k, lmax);
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
            kaffpa_initial_partition(graphs.back(), (int) k, config.imbalance, (int) config.seed, partition);
            recalculate_weights(partition, graphs.back());

            assert_state_after_partition(graphs.back(), partition, config.k);
        }

        void refinement() {
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
