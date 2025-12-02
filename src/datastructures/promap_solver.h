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

#ifndef GPU_HEIPA_PROMAP_SOLVER_H
#define GPU_HEIPA_PROMAP_SOLVER_H

#include <vector>

#include "graph.h"
#include "host_graph.h"
#include "kokkos_memory_stack.h"
#include "mapping.h"
#include "partition.h"
#include "../coarsening/two_hop_matching.h"
#include "../distance_oracles/distance_oracle_binary.h"
#include "../distance_oracles/distance_oracle_matrix.h"
#include "../refinement/promap_jet_label_propagation.h"
#include "../initial_partitioning/kaffpa_initial_partitioning.h"
#include "../initial_partitioning/metis_initial_partitioning.h"
#include "../utility/definitions.h"
#include "../utility/promap_configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../distance_oracles/distance_oracle_helpers.h"
#include "../utility/comm_cost.h"

namespace GPU_HeiPa {
    template<typename d_oracle_t>
    class ProMapSolver {
    public:
        ProMapConfiguration config;
        std::chrono::time_point<std::chrono::system_clock> sp;

        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        std::vector<partition_t> hierarchy;
        std::vector<weight_t> distances;
        weight_t lmax = 0;

        HostGraph host_g;
        KokkosMemoryStack mem_stack;

        std::vector<Graph> graphs;
        std::vector<Mapping> mappings;

        d_oracle_t d_oracle;
        Partition partition;

        weight_t curr_comm_cost = 0;
        weight_t initial_comm_cost = 0;
        weight_t initial_max_block_weight = 0;

        f64 io_ms = 0.0;
        f64 coarsening_ms = 0.0;
        f64 contraction_ms = 0.0;
        f64 initial_partitioning_ms = 0.0;
        f64 uncontraction_ms = 0.0;
        f64 refinement_ms = 0.0;

        struct level_info {
            u32 level;
            vertex_t n;
            vertex_t m;

            weight_t comm_cost;
            weight_t max_b_weight;
            f64 imb;
            partition_t empty_partitions;
            partition_t oload_partitions;
            weight_t sum_oload_weights;

            f64 t_coarsening;
            f64 t_contraction;
            f64 t_uncontraction;
            f64 t_refinement;
        };

        std::vector<level_info> level_infos;

        inline void print_level_row(const level_info &L) {
            std::cout
                    << std::setw(3) << L.level << " | "
                    << std::setw(8) << L.n << " | "
                    << std::setw(11) << L.m << " | "
                    << std::setw(8) << L.comm_cost << " | "
                    << std::setw(7) << L.max_b_weight << " | "
                    << std::setw(8) << L.imb << " | "
                    << std::setw(6) << (u32) L.empty_partitions << " | "
                    << std::setw(6) << (u32) L.oload_partitions << " | "
                    << std::setw(8) << L.sum_oload_weights << " | "
                    << std::setw(10) << L.t_coarsening << " | "
                    << std::setw(10) << L.t_contraction << " | "
                    << std::setw(10) << L.t_uncontraction << " | "
                    << std::setw(10) << L.t_refinement
                    << "\n";
        }

        inline void print_all_levels(const std::vector<level_info> &infos) {
            std::cout
                    << std::setw(3) << "Lvl" << " | "
                    << std::setw(8) << "n" << " | "
                    << std::setw(11) << "m" << " | "
                    << std::setw(8) << "comm cost" << " | "
                    << std::setw(7) << "maxW" << " | "
                    << std::setw(8) << "imb" << " | "
                    << std::setw(6) << "empty" << " | "
                    << std::setw(6) << "oload" << " | "
                    << std::setw(8) << "w_oload" << " | "
                    << std::setw(10) << "t_c" << " | "
                    << std::setw(10) << "t_con" << " | "
                    << std::setw(10) << "t_unc" << " | "
                    << std::setw(10) << "t_ref"
                    << "\n";

            std::cout << std::string(100, '-') << "\n";
            for (const auto &L: infos) {
                print_level_row(L);
            }
        }

        explicit ProMapSolver(ProMapConfiguration t_config) : config(std::move(t_config)) {
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
            std::cout << "Init. comm_cost   : " << initial_comm_cost << std::endl;
            std::cout << "Init. max block w : " << initial_max_block_weight << std::endl;
            std::cout << "Final comm-cost   : " << comm_cost(graphs.back(), partition, d_oracle) << std::endl;
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

            print_all_levels(level_infos);

            free_distance_oracle<d_oracle_t>(d_oracle, mem_stack);
            free_partition(partition, mem_stack);

            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            assert_is_empty(mem_stack);
            destroy(mem_stack);

            return {};
        }

    private:
        void internal_solve() {
            initialize();

            const partition_t c = 4;
            const partition_t max_n = c * k;

            u32 level = 0;
            while (graphs.back().n > max_n) {
                level_infos.emplace_back();
                level_infos[level].level = level;
                level_infos[level].n = graphs.back().n;
                level_infos[level].m = graphs.back().m;

                coarsening(level);
                contraction(level);

                level += 1;
            }

            level_infos.emplace_back();
            level_infos[level].level = level;
            level_infos[level].n = graphs.back().n;
            level_infos[level].m = graphs.back().m;

            initial_partitioning();

            level_infos[level].max_b_weight = max_weight(partition);
            level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
            level_infos[level].comm_cost = comm_cost(graphs.back(), partition, d_oracle);
            level_infos[level].empty_partitions = n_empty_blocks(partition);
            level_infos[level].oload_partitions = n_oload_blocks(partition);
            level_infos[level].sum_oload_weights = sum_oload_weight(partition);

            while (!mappings.empty()) {
                level -= 1;

                uncontraction(level);
                refinement(level);

                level_infos[level].max_b_weight = max_weight(partition);
                level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
                level_infos[level].comm_cost = comm_cost(graphs.back(), partition, d_oracle);
                level_infos[level].empty_partitions = n_empty_blocks(partition);
                level_infos[level].oload_partitions = n_oload_blocks(partition);
                level_infos[level].sum_oload_weights = sum_oload_weight(partition);
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

            n = host_g.n;
            m = host_g.m;
            k = config.k;
            hierarchy = config.hierarchy;
            distances = config.distance;
            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));

            graphs.emplace_back(from_HostGraph(host_g, mem_stack));

            // initialize distance oracle
            {
                ScopedTimer t{"io", "distance_oracle", "initialize"};
                d_oracle = initialize_distance_oracle<d_oracle_t>(k, hierarchy, distances, mem_stack);
            }

            // initialize partition
            {
                ScopedTimer t{"io", "partition", "initialize"};
                partition = initialize_partition(n, k, lmax, mem_stack);
            }

            io_ms += get_milli_seconds(p, get_time_point());

            assert_state_pre_partition(graphs.back());
        }

        void coarsening(u32 level) {
            auto p = get_time_point();

            mappings.emplace_back(two_hop_matcher_get_mapping(graphs.back(), partition, lmax, mem_stack));

            Kokkos::fence();
            coarsening_ms += get_milli_seconds(p, get_time_point());
            level_infos[level].t_coarsening = get_milli_seconds(p, get_time_point());

            assert_state_pre_partition(graphs.back());
        }

        void contraction(u32 level) {
            auto p = get_time_point();

            graphs.emplace_back(from_Graph_Mapping(graphs.back(), mappings.back(), mem_stack));
            contract(partition, mappings.back());

            Kokkos::fence();
            contraction_ms += get_milli_seconds(p, get_time_point());
            level_infos[level].t_contraction = get_milli_seconds(p, get_time_point());

            assert_state_pre_partition(graphs.back());
        }

        void initial_partitioning() {
            auto p = get_time_point();

            // Use METIS for initial partitioning
            metis_initial_partition(graphs.back(), (int) k, config.imbalance, config.seed, partition);

            recalculate_weights(partition, graphs.back());

            initial_comm_cost = comm_cost(graphs.back(), partition, d_oracle);
            curr_comm_cost = initial_comm_cost;
            initial_max_block_weight = max_weight(partition);

            Kokkos::fence();
            initial_partitioning_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(graphs.back(), partition, config.k);
        }

        void refinement(u32 level) {
            auto p = get_time_point();

            curr_comm_cost = promap_refine(graphs.back(), partition, d_oracle, k, lmax, level, curr_comm_cost, mem_stack);

            Kokkos::fence();
            refinement_ms += get_milli_seconds(p, get_time_point());
            level_infos[level].t_refinement = get_milli_seconds(p, get_time_point());

            assert_state_after_partition(graphs.back(), partition, config.k);
        }

        void uncontraction(u32 level) {
            auto p = get_time_point();

            uncontract(partition, mappings.back());

            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();

            Kokkos::fence();
            uncontraction_ms += get_milli_seconds(p, get_time_point());
            level_infos[level].t_uncontraction = get_milli_seconds(p, get_time_point());

            assert_state_after_partition(graphs.back(), partition, config.k);
        }
    };
}

#endif //GPU_HEIPA_PROMAP_SOLVER_H
