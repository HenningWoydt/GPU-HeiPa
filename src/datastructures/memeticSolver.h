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

#ifndef GPU_HEIPA_SOLVER_MEMETIC_H
#define GPU_HEIPA_SOLVER_MEMETIC_H

#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_StaticCrsGraph.hpp>



#include "graph.h"
#include "host_graph.h"
#include "kokkos_memory_stack.h"
#include "mapping.h"
#include "partition.h"
#include "../coarsening/two_hop_matching.h"
#include "../refinement/jet_label_propagation.h"
#include "../initial_partitioning/metis_partitioning.h"
#include "../utility/definitions.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/edge_cut.h"

#include <cuda_runtime.h>
#include "omp.h"

namespace GPU_HeiPa {
    class memeticSolver {
    public:
        Configuration config;

        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;
        bool use_ultra = true;

        std::vector<Graph> graphs;
        std::vector<Mapping> mappings;

        size_t num_cpu_threads = 4;
        size_t num_individuals = 4;
        

        std::vector<Partition> partitions = std::vector<Partition>(num_individuals);
        
        std::vector<cudaStream_t> cuda_streams = std::vector<cudaStream_t>(num_cpu_threads);
        std::vector<Kokkos::Cuda> exec_spaces; 


        std::vector<weight_t> curr_edge_cut = std::vector<weight_t>(num_individuals, 0);
        std::vector<weight_t> curr_max_block_weight = std::vector<weight_t>(num_individuals, 0);
        std::vector<weight_t> initial_edge_cut = std::vector<weight_t>(num_individuals, 0);
        std::vector<weight_t> initial_max_block_weight = std::vector<weight_t>(num_individuals, 0);

        f64 down_up_load_ms = 0.0;
        f64 misc_ms = 0.0;
        f64 coarsening_ms = 0.0;
        f64 contraction_ms = 0.0;
        f64 initial_partitioning_ms = 0.0;
        f64 uncontraction_ms = 0.0;
        f64 refinement_ms = 0.0;

        struct level_info {
            u32 level;
            vertex_t n;
            vertex_t m;

            weight_t edge_cut;
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

#if ENABLE_PROFILER
        std::vector<level_info> level_infos;
#endif

        inline void print_level_row(const level_info &L) {
            std::cout
                    << std::setw(3) << L.level << " | "
                    << std::setw(8) << L.n << " | "
                    << std::setw(11) << L.m << " | "
                    << std::setw(8) << L.edge_cut << " | "
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
                    << std::setw(8) << "cut" << " | "
                    << std::setw(7) << "maxW" << " | "
                    << std::setw(8) << "imb" << " | "
                    << std::setw(6) << "empty" << " | "
                    << std::setw(6) << "oload" << " | "
                    << std::setw(8) << "w_oload" << " | "
                    << std::setw(10) << "t_coars" << " | "
                    << std::setw(10) << "t_con" << " | "
                    << std::setw(10) << "t_unc" << " | "
                    << std::setw(10) << "t_ref"
                    << "\n";

            std::cout << std::string(100, '-') << "\n";
            for (const auto &L: infos) {
                print_level_row(L);
            }
        }

        explicit memeticSolver(Configuration t_config) : config(std::move(t_config)) {
        }

        //! do this later
        /*
        
        memeticSolver(Graph &dev_g,
               partition_t t_k,
               f64 imbalance,
               u64 seed,
               bool t_use_ultra,
               UnmanagedDevicePartition &dev_partition,
               KokkosMemoryStack &dev_mem_stack) {
            // Main stack: Graph + coarsening overhead
            ScopedTimer t_init{"hm", "solver", "initialize"};
            n = dev_g.n;
            m = dev_g.m;
            k = t_k;
            lmax = (weight_t) std::ceil((1.0 + imbalance) * ((f64) dev_g.g_weight / (f64) k));

            config.imbalance = imbalance;
            config.k = k;
            config.seed = seed;
            config.verbose_level = 0; // or whatever you want
            use_ultra = t_use_ultra;

            graphs.emplace_back(dev_g);

            partition = initialize_partition(n, k, lmax, dev_mem_stack);

            assert_state_pre_partition(graphs.back());

            t_init.stop();

            const partition_t c = 8;
            const partition_t max_n = c * k;

            u32 level = 0;
            while (graphs.back().n > max_n) {
#if ENABLE_PROFILER
                ScopedTimer t_profiler{"hm", "solver", "profiling"};
                level_infos.emplace_back();
                level_infos[level].level = level;
                level_infos[level].n = graphs.back().n;
                level_infos[level].m = graphs.back().m;
                t_profiler.stop();
#endif
                coarsening(level, dev_mem_stack);
                contraction(level, dev_mem_stack);

                level += 1;
            }

#if ENABLE_PROFILER
            ScopedTimer t_profiler{"hm", "solver", "profiling"};
            level_infos.emplace_back();
            level_infos[level].level = level;
            level_infos[level].n = graphs.back().n;
            level_infos[level].m = graphs.back().m;
            t_profiler.stop();
#endif

            initial_partitioning(dev_mem_stack);

#if ENABLE_PROFILER
            ScopedTimer t_profiler2{"hm", "solver", "profiling"};
            level_infos[level].max_b_weight = max_weight(partition);
            level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) dev_g.g_weight / (f64) config.k);
            level_infos[level].edge_cut = edge_cut(graphs.back(), partition);
            level_infos[level].empty_partitions = n_empty_blocks(partition);
            level_infos[level].oload_partitions = n_oload_blocks(partition);
            level_infos[level].sum_oload_weights = sum_oload_weight(partition);
            t_profiler2.stop();
#endif

            while (!mappings.empty()) {
                level -= 1;

                uncontraction(level, dev_mem_stack);
                refinement(level, dev_mem_stack);

#if ENABLE_PROFILER
                ScopedTimer t_profiler3{"hm", "solver", "profiling"};
                level_infos[level].max_b_weight = max_weight(partition);
                level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) dev_g.g_weight / (f64) config.k);
                level_infos[level].edge_cut = edge_cut(graphs.back(), partition);
                level_infos[level].empty_partitions = n_empty_blocks(partition);
                level_infos[level].oload_partitions = n_oload_blocks(partition);
                level_infos[level].sum_oload_weights = sum_oload_weight(partition);
                t_profiler3.stop();
#endif
            }

            ScopedTimer t{"hm", "solver", "copy_res"};
            Kokkos::deep_copy(dev_partition, partition.map);
            Kokkos::fence();

            free_partition(partition, dev_mem_stack);
        }
        */

        HostPartition solve(HostGraph &host_g) {
            auto sp = get_time_point();

            //! need a set of memstacks (for each thread)
            std::vector<KokkosMemoryStack> mem_stacks(num_cpu_threads);

            for (size_t i = 0; i < num_cpu_threads; ++i) {
                mem_stacks[i] = initialize_kokkos_memory_stack(
                    30 * (size_t) host_g.n * sizeof(vertex_t) +
                    10 * (size_t) host_g.m * sizeof(vertex_t),
                    "Stack_" + std::to_string(i)
                );
            }
            
            internal_solve(host_g, mem_stacks);

            auto p = get_time_point();
            
            //TODO: pick best partition out of all of them
            size_t min_id = 0;
            weight_t min_edgecut = curr_edge_cut[0];
            for(size_t i = 0; i < num_individuals; ++i) {
                
                std::cout << "fitness individual " << i << " = " << curr_edge_cut[i] << std::endl ;
                if ( curr_edge_cut[i] < min_edgecut) {
                    min_id = i;
                    min_edgecut = curr_edge_cut[i];
                }
            }
            
            HostPartition host_partition;
            //
            {
                ScopedTimer _t("up/download", "Solver", "download_partition");
                host_partition = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), graphs.back().n);
                Kokkos::deep_copy(host_partition, partitions[min_id].map);
            }
            f64 down_ms = get_milli_seconds(p, get_time_point());
            down_up_load_ms += down_ms;

            // calc stats
            size_t n_empty_partitions = 0;
            size_t n_overloaded_partitions = 0;
            weight_t sum_too_much = 0;
            if (config.verbose_level >= 2) {
                ScopedTimer _t("misc", "Solver", "calc_stats");
                PartitionHost partition_host = to_host_partition(partitions[min_id]);
                for (partition_t id = 0; id < config.k; ++id) {
                    n_empty_partitions += partition_host.bweights(id) == 0;
                    n_overloaded_partitions += partition_host.bweights(id) > lmax;
                    sum_too_much += std::max((weight_t) 0, partition_host.bweights(id) - lmax);
                }
            }

            // free all memory
            {
                ScopedTimer _t("misc", "Solver", "free_memory");

                for (size_t i = num_individuals; i-- > 0; ) {
                    free_partition(partitions[i], mem_stacks[0]);
                }

                free_graph(graphs.back(), mem_stacks[0]);
                graphs.pop_back();

                for(size_t i= 0; i < num_cpu_threads ; ++i ) {
   
                    assert_is_empty(mem_stacks[i]);
                    destroy(mem_stacks[i]);

                }

                free_streams();
            }

            misc_ms += get_milli_seconds(p, get_time_point());
            misc_ms -= down_ms;

            auto ep = get_time_point();
            f64 duration = get_milli_seconds(sp, ep);

            if (config.verbose_level >= 1) {
                std::cout << "------- Info -------" << std::endl;
                std::cout << "Total solve time  : " << duration << std::endl;
                std::cout << "#Vertices         : " << n << std::endl;
                std::cout << "#Edges            : " << m << std::endl;
                std::cout << "k                 : " << k << std::endl;
                std::cout << "imbalance         : " << config.imbalance << std::endl;
                std::cout << "Lmax              : " << lmax << std::endl;
            }
            if (config.verbose_level >= 2) {
                std::cout << "------- Stat -------" << std::endl;
                std::cout << "Init. edge-cut    : " << initial_edge_cut[min_id] << std::endl;
                std::cout << "Init. max block w : " << initial_max_block_weight[min_id] << std::endl;
                std::cout << "Final edge-cut    : " << curr_edge_cut[min_id] << std::endl;
                std::cout << "Final max block w : " << curr_max_block_weight[min_id] << std::endl;

                std::cout << "#empty partitions : " << n_empty_partitions << std::endl;
                std::cout << "#oload partitions : " << n_overloaded_partitions << std::endl;
                std::cout << "Sum oload weights : " << sum_too_much << std::endl;
            }
            if (config.verbose_level >= 1) {
                std::cout << "------- Time -------" << std::endl;
                std::cout << "Coarsening        : " << coarsening_ms << std::endl;
                std::cout << "Contraction       : " << contraction_ms << std::endl;
                std::cout << "Init. Part.       : " << initial_partitioning_ms << std::endl;
                std::cout << "Uncontraction     : " << uncontraction_ms << std::endl;
                std::cout << "Refinement        : " << refinement_ms << std::endl;
                std::cout << "Down/Upload       : " << down_up_load_ms << std::endl;
                std::cout << "Misc              : " << misc_ms << std::endl;
                std::cout << "ALL               : " << coarsening_ms + contraction_ms + initial_partitioning_ms + uncontraction_ms + refinement_ms + down_up_load_ms + misc_ms << std::endl;
            }
            if (config.verbose_level >= 2) {
#if ENABLE_PROFILER
                print_all_levels(level_infos);
#endif
            }

            return host_partition;
        }

    private:
        void internal_solve(HostGraph &host_g, std::vector<KokkosMemoryStack> &mem_stacks) {
            init_streams();
            initialize(host_g, mem_stacks[0]);


            const partition_t c = 8;
            const partition_t max_n = c * k;

            u32 level = 0;
            while (graphs.back().n > max_n) {
#if ENABLE_PROFILER
                level_infos.emplace_back();
                level_infos[level].level = level;
                level_infos[level].n = graphs.back().n;
                level_infos[level].m = graphs.back().m;
#endif

                coarsening(level, mem_stacks[0]);
                contraction(level, mem_stacks[0]);

                level += 1;
            }

#if ENABLE_PROFILER
            level_infos.emplace_back();
            level_infos[level].level = level;
            level_infos[level].n = graphs.back().n;
            level_infos[level].m = graphs.back().m;
#endif
            #pragma omp parallel for num_threads(num_cpu_threads)
            for(size_t i = 0; i < num_individuals ; ++i) {
                initial_partitioning(mem_stacks[0], i);
            }
            


#if ENABLE_PROFILER
            // level_infos[level].max_b_weight = max_weight(partitions[0]); //! übergangslösung
            // level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
            // level_infos[level].edge_cut = edge_cut(graphs.back(), partition);
            // level_infos[level].empty_partitions = n_empty_blocks(partition);
            // level_infos[level].oload_partitions = n_oload_blocks(partition);
            // level_infos[level].sum_oload_weights = sum_oload_weight(partition);
#endif

            while (!mappings.empty()) {
                level -= 1;

                //! do for all individuals in parallel
                #pragma omp parallel for num_threads(num_cpu_threads)
                for(size_t i = 0; i < num_individuals ; ++i) {
                    int tid = omp_get_thread_num();
                    //std::cout << "hi from  thread " << tid << std::endl;
                    uncontraction(level, i, tid); 
                }

                free_after_uncontraction(mem_stacks[0]); //! only one thread

                //! all threads: use tid to pick mem_stack
                //! this causes problems!
                //! maybe using different execution spaces will fix my issues...
                //! -> actually helps... wtf
                #pragma omp parallel for num_threads(num_cpu_threads)
                for(size_t i = 0; i < num_individuals ; ++i) {
                    int tid =  omp_get_thread_num();

    //                refinement(level, mem_stacks[i], i, i); //! mem_stacks[i] should be tid not i
                      refinement(level, mem_stacks[tid], i, tid); //! mem_stacks[i] should be tid not i
                }

#if ENABLE_PROFILER
               //TODO: make it work for partition vector
               // level_infos[level].max_b_weight = max_weight(partition);
               // level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
               // level_infos[level].edge_cut = edge_cut(graphs.back(), partition);
               // level_infos[level].empty_partitions = n_empty_blocks(partition);
               // level_infos[level].oload_partitions = n_oload_blocks(partition);
               // level_infos[level].sum_oload_weights = sum_oload_weight(partition);
#endif
            }
        }


        void init_streams() {
            
            for (size_t i = 0; i < num_cpu_threads; ++i) {
                cudaStreamCreateWithFlags(&cuda_streams[i], cudaStreamNonBlocking);
            }

            for (size_t i = 0; i < num_cpu_threads; ++i) {
                exec_spaces.emplace_back(Kokkos::Cuda(cuda_streams[i]));
            }

        }

        void free_streams() {
        
            for (size_t i = 0; i < num_cpu_threads; ++i) {
                cudaStreamDestroy(cuda_streams[i]);
            }
 
        }

        void initialize(HostGraph &host_g, KokkosMemoryStack &mem_stack) {
            auto p = get_time_point();

            n = host_g.n;
            m = host_g.m;
            k = config.k;
            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));
            use_ultra = config.config == "ultra";

            f64 up_ms = 0;
            
            graphs.emplace_back(from_HostGraph(host_g, mem_stack, up_ms));

    
            for(size_t i = 0; i < num_individuals ; ++i) {
                partitions[i] = initialize_partition(n, k, lmax, mem_stack);
            }


            misc_ms += get_milli_seconds(p, get_time_point());
            misc_ms -= up_ms;
            down_up_load_ms += up_ms;

            assert_state_pre_partition(graphs.back());
        }

        void coarsening(u32 level, KokkosMemoryStack &mem_stack) {
            auto p = get_time_point();

            //! only need one of these right?
            //! shouldnt matter which partition i pass, since this function doesnt use it anyway
            mappings.emplace_back(two_hop_matcher_get_mapping(graphs.back(), partitions[0], lmax, mem_stack));

            Kokkos::fence();
            coarsening_ms += get_milli_seconds(p, get_time_point());

#if ENABLE_PROFILER
            level_infos[level].t_coarsening = get_milli_seconds(p, get_time_point());
#endif

            assert_state_pre_partition(graphs.back());
        }

        void contraction(u32 level, KokkosMemoryStack &mem_stack) {
            auto p = get_time_point();

            graphs.emplace_back(from_Graph_Mapping(graphs.back(), mappings.back(), mem_stack));
            
            for(size_t i = 0; i < num_individuals ; ++i) {
                contract(partitions[i], mappings.back());
            }

            Kokkos::fence();
            contraction_ms += get_milli_seconds(p, get_time_point());

#if ENABLE_PROFILER
            level_infos[level].t_contraction = get_milli_seconds(p, get_time_point());
#endif

            assert_state_pre_partition(graphs.back());
        }

        //! this doesnt even use mem_stack...
        void initial_partitioning(KokkosMemoryStack &mem_stack, size_t individual_id) {
            auto p = get_time_point();

            metis_partition(graphs.back(), (int) k, config.imbalance, config.seed + individual_id, partitions[ individual_id ], METIS_RECURSIVE);

            recalculate_weights(partitions[individual_id], graphs.back());

            ScopedTimer _t("initial_partitioning", "Partition", "first_stats");

            //TODO: partition-local variables
            initial_edge_cut[individual_id] = edge_cut(graphs.back(), partitions[individual_id]); //! only reads graph and partition
            curr_edge_cut[individual_id] = initial_edge_cut[individual_id]; //! threadlocal variable
            initial_max_block_weight[individual_id] = max_weight(partitions[individual_id]); //! only reads graph and partition
            curr_max_block_weight[individual_id] = initial_max_block_weight[individual_id]; //! threadlocal variable

            Kokkos::fence(); //! threadlocal fence
            initial_partitioning_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(graphs.back(), partitions[individual_id], config.k);
        }

        void refinement(u32 level, KokkosMemoryStack &mem_stack, size_t individual_id, size_t tid) {
            auto p = get_time_point();

            //! each thread needs their own values for this call, only graphs is shared (right?)
            auto pair = jet_refine(graphs.back(), partitions[ individual_id ], k, lmax, use_ultra, level, curr_edge_cut[individual_id], curr_max_block_weight[individual_id], mem_stack, exec_spaces[tid]);
            //! set threadlocal variables
            curr_edge_cut[individual_id] = pair.first;
            curr_max_block_weight[individual_id] = pair.second;

            //! fence for correct execution space
            //Kokkos::fence();
            exec_spaces[tid].fence();

            //! timing vectors too?...
            refinement_ms += get_milli_seconds(p, get_time_point());

            //! threadlocal variables
            ASSERT(curr_edge_cut[individual_id] == edge_cut(graphs.back(), partitions[individual_id]));
            ASSERT(curr_max_block_weight[individual_id] == max_weight(partitions[individual_id]));

#if ENABLE_PROFILER
            level_infos[level].t_refinement = get_milli_seconds(p, get_time_point());
#endif

            assert_state_after_partition(graphs.back(), partitions[individual_id], config.k);
        }



        void uncontraction(u32 level, size_t individual_id, int tid) {
            auto p = get_time_point();

            uncontract(partitions[individual_id], mappings.back(), exec_spaces[tid]);

            //TODO: partition local value
            uncontraction_ms += get_milli_seconds(p, get_time_point());

#if ENABLE_PROFILER
            level_infos[level].t_uncontraction = get_milli_seconds(p, get_time_point());
#endif

            assert_state_after_partition(graphs.back(), partitions[individual_id], config.k);
        }

        void free_after_uncontraction(KokkosMemoryStack &mem_stack) {
           
            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();

            Kokkos::fence();
        }



    };
}

#endif //GPU_HEIPA_SOLVER_H
