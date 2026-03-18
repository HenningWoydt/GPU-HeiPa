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
#include "../refinement/memetic_refinement.h"
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
        u32 num_crossovers = 1;
        u32 num_parents = 2;
        u32 tournament_size = 2;

        size_t orga_stack = num_cpu_threads;
        size_t partition_stack = num_cpu_threads + 1;
        

        std::vector<Partition> partitions = std::vector<Partition>(num_individuals);
        
        std::vector<cudaStream_t> cuda_streams = std::vector<cudaStream_t>(num_cpu_threads);
        std::vector<Kokkos::Cuda> exec_spaces; 


        std::vector<weight_t> curr_edge_cut = std::vector<weight_t>(num_individuals, 0);
        std::vector<weight_t> curr_max_block_weight = std::vector<weight_t>(num_individuals, 0);
        std::vector<weight_t> initial_edge_cut = std::vector<weight_t>(num_individuals, 0);
        std::vector<weight_t> initial_max_block_weight = std::vector<weight_t>(num_individuals, 0);

        std::vector<weight_t> edge_cut_offspring = std::vector<weight_t>(num_crossovers, 0);
        std::vector<weight_t> max_block_weight_offspring = std::vector<weight_t>(num_crossovers, 0);

        // distances:
        // min_distances[ i ] := the smallest distance of individual i to another individual
        std::vector<u32> min_distances = std::vector<u32>(num_individuals, 0);

        f64 down_up_load_ms = 0.0;
        f64 misc_ms = 0.0;
        f64 coarsening_ms = 0.0; // basst
        f64 contraction_ms = 0.0; // basst
        f64 initial_partitioning_ms = 0.0;
        f64 uncontraction_ms = 0.0;
        f64 refinement_ms = 0.0;
        f64 memetic_ms = 0.0 ;

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
            f64 t_memetic;
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
                    << std::setw(10) << L.t_refinement << " | "
                    << std::setw(10) << L.t_memetic
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
                    << std::setw(10) << "t_ref" << " | "
                    << std::setw(10) << "t_meme"
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

            std::vector<KokkosMemoryStack> mem_stacks(num_cpu_threads + 2 );

            //! maybe adjust sizes here
            for (size_t i = 0; i < num_cpu_threads; ++i) {
                mem_stacks[i] = initialize_kokkos_memory_stack(
                    10 * (size_t) host_g.n * sizeof(vertex_t) +
                    10 *(size_t) host_g.m * sizeof(vertex_t),
                    "Stack_" + std::to_string(i)
                );
            }

            //! use this for the coarsening and mapping and stuff
            mem_stacks[ orga_stack ] = initialize_kokkos_memory_stack(
                    30 * (size_t) host_g.n * sizeof(vertex_t) +
                    10 *(size_t) host_g.m * sizeof(vertex_t),
                    "Stack for mappings and graphs"
                );
            
            //! use this to hold all partitions!
            mem_stacks[ partition_stack ] = initialize_kokkos_memory_stack(
                    ( num_individuals + 5 ) * (size_t) host_g.n * sizeof(vertex_t) +
                    10 *(size_t) host_g.m * sizeof(vertex_t),
                    "Stack for partitions" 
                );

            
            internal_solve(host_g, mem_stacks);

            auto p = get_time_point();
            
            // pick best partition out of all of them
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
                    free_partition(partitions[i], mem_stacks[ partition_stack ]);
                }

                free_graph(graphs.back(), mem_stacks[ orga_stack ]);
                graphs.pop_back();

                for(size_t i= 0; i < num_cpu_threads + 2 ; ++i ) {
   
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
                std::cout << "Memetic stuff     : " << memetic_ms << std::endl;  
                std::cout << "Down/Upload       : " << down_up_load_ms << std::endl;
                std::cout << "Misc              : " << misc_ms << std::endl;
                std::cout << "ALL               : " << coarsening_ms + contraction_ms + initial_partitioning_ms + uncontraction_ms + refinement_ms + memetic_ms + down_up_load_ms + misc_ms << std::endl;
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
            initialize(host_g, mem_stacks);


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

                coarsening(level, mem_stacks[ orga_stack ]);
                contraction(level, mem_stacks[ orga_stack ]);

                level += 1;
            }

#if ENABLE_PROFILER
            level_infos.emplace_back();
            level_infos[level].level = level;
            level_infos[level].n = graphs.back().n;
            level_infos[level].m = graphs.back().m;
#endif
        // extra scope for value p -> will appear multiple times
        { 
            auto p = get_time_point();

            #pragma omp parallel for num_threads(num_cpu_threads)
            for(size_t i = 0; i < num_individuals ; ++i) {
                size_t tid = static_cast<size_t>(  omp_get_thread_num() );
                initial_partitioning( i, tid );
                
            }
            ScopedTimer _t("initial_partitioning", "Partition", "first_stats");

            initial_partitioning_ms += get_milli_seconds(p, get_time_point());
        }


#if ENABLE_PROFILER
            //! übergangslösung
            level_infos[level].max_b_weight = max_weight(partitions[0]); 
            level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
            level_infos[level].edge_cut = edge_cut(graphs.back(), partitions[0]);
            level_infos[level].empty_partitions = n_empty_blocks(partitions[0]);
            level_infos[level].oload_partitions = n_oload_blocks(partitions[0]);
            level_infos[level].sum_oload_weights = sum_oload_weight(partitions[0]);
#endif

            while (!mappings.empty()) {
                level -= 1;

                {
                    // scope for uncontraction
                    auto p = get_time_point();
                    
                    
                    #pragma omp parallel for num_threads(num_cpu_threads)
                    for(size_t i = 0; i < num_individuals ; ++i) {
                        size_t tid = static_cast<size_t>(  omp_get_thread_num() );
                        uncontraction( level, i, tid ); 
                    }
                    
                    free_after_uncontraction(mem_stacks[ orga_stack ]); 
                    
                    
                    uncontraction_ms += get_milli_seconds(p, get_time_point());

#if ENABLE_PROFILER
                    level_infos[level].t_uncontraction = get_milli_seconds(p, get_time_point());
#endif

                }
            
                {
                    //scope for refinement
                    auto p = get_time_point();

                    #pragma omp parallel for num_threads(num_cpu_threads)
                    for(size_t i = 0; i < num_individuals ; ++i) {
                        size_t tid = static_cast<size_t>(  omp_get_thread_num() );
                        refinement(level, mem_stacks[tid], i, tid ); 
                    }

                    refinement_ms += get_milli_seconds(p, get_time_point());
#if ENABLE_PROFILER
                    level_infos[level].t_refinement = get_milli_seconds(p, get_time_point());
#endif
                }

                {
                   // std::cout << "Edge cuts of all individuals in level " << level  << ": ";
                   // for (size_t i = 0; i < curr_edge_cut.size(); ++i) {
                   //     std::cout << curr_edge_cut[i];
                   //     if (i + 1 < curr_edge_cut.size()) std::cout << ", ";
                   // }
                   // std::cout << std::endl;
                }

                {
                    auto p = get_time_point();
                    
                    memetic_refinement(level, mem_stacks[ partition_stack ]);
                    
                    memetic_ms += get_milli_seconds(p, get_time_point());

#if ENABLE_PROFILER
                    level_infos[level].t_memetic = get_milli_seconds(p, get_time_point());
#endif
                }



#if ENABLE_PROFILER
            //! übergangslösung
            level_infos[level].max_b_weight = max_weight(partitions[0]); 
            level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
            level_infos[level].edge_cut = edge_cut(graphs.back(), partitions[0]);
            level_infos[level].empty_partitions = n_empty_blocks(partitions[0]);
            level_infos[level].oload_partitions = n_oload_blocks(partitions[0]);
            level_infos[level].sum_oload_weights = sum_oload_weight(partitions[0]);
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

        void initialize(HostGraph &host_g, std::vector<KokkosMemoryStack> &mem_stacks) {
            auto p = get_time_point();

            n = host_g.n;
            m = host_g.m;
            k = config.k;
            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));
            use_ultra = config.config == "ultra";

            f64 up_ms = 0;
            
            graphs.emplace_back(from_HostGraph(host_g, mem_stacks[ orga_stack ], up_ms));

    
            for(size_t i = 0; i < num_individuals ; ++i) {
                partitions[i] = initialize_partition(n, k, lmax, mem_stacks[ partition_stack ]);
            }


            misc_ms += get_milli_seconds(p, get_time_point());
            misc_ms -= up_ms;
            down_up_load_ms += up_ms;

            assert_state_pre_partition(graphs.back());
        }

        void coarsening(u32 level, KokkosMemoryStack &mem_stack) {
            auto p = get_time_point();


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


        void initial_partitioning(size_t individual_id, size_t tid) {
            

            //! better way to pick the seed? more randomness?
            metis_partition(graphs.back(), (int) k, config.imbalance, config.seed + individual_id, partitions[ individual_id ], METIS_RECURSIVE);

            recalculate_weights(partitions[individual_id], graphs.back());
            
            
            initial_edge_cut[individual_id] = edge_cut(graphs.back(), partitions[individual_id], exec_spaces[tid]); 
            curr_edge_cut[individual_id] = initial_edge_cut[individual_id]; 

            initial_max_block_weight[individual_id] = max_weight(partitions[individual_id], exec_spaces[tid] ); 
            curr_max_block_weight[individual_id] = initial_max_block_weight[individual_id]; 

            exec_spaces[tid].fence();

            assert_state_after_partition(graphs.back(), partitions[individual_id], config.k);
        }

        void refinement(u32 level, KokkosMemoryStack &mem_stack, size_t individual_id, size_t tid) {
            
            //! each thread needs their own values for this call, only graphs is shared (right?)
            auto pair = jet_refine(graphs.back(), partitions[ individual_id ], k, lmax, use_ultra, level, curr_edge_cut[individual_id], curr_max_block_weight[individual_id], mem_stack, exec_spaces[tid]);
            
            curr_edge_cut[individual_id] = pair.first;
            curr_max_block_weight[individual_id] = pair.second;

            exec_spaces[tid].fence();

            ASSERT(curr_edge_cut[individual_id] == edge_cut(graphs.back(), partitions[individual_id]));
            ASSERT(curr_max_block_weight[individual_id] == max_weight(partitions[individual_id]));

            assert_state_after_partition(graphs.back(), partitions[individual_id], config.k);
        }



        void uncontraction(u32 level, size_t individual_id, size_t tid) {

            uncontract(partitions[individual_id], mappings.back(), exec_spaces[tid]);

            assert_state_after_partition(graphs.back(), partitions[individual_id], config.k);
        }

        void free_after_uncontraction(KokkosMemoryStack &mem_stack) {
           
            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();

            Kokkos::fence();
        }



        void memetic_refinement(u32 level, KokkosMemoryStack &mem_stack) {


            for(u32 i = 0; i < num_crossovers; ++i) {

                std::vector<int> parent_ids;
                {
                    ScopedTimer _t("memetic", "memetic_refinement", "parent selection");

                    
                    for(u32 j= 0; j < num_parents; ++j) {
                        parent_ids.push_back( tournament_selection( curr_edge_cut , tournament_size) );
                    }
                    
                    // Ensure parent_ids[0] != parent_ids[1]
                    if (parent_ids.size() == 2 && parent_ids[0] == parent_ids[1]) {
                        int new_parent = parent_ids[0];
                        while (new_parent == parent_ids[0]) {
                            new_parent = static_cast<int>(rand() % num_individuals);
                        }
                        parent_ids[1] = new_parent;
                    }
                }

                Partition offspring;
                {

                    ScopedTimer _t("memetic", "memetic_refinement", "create offspring");   
                    offspring = backbone_based_crossover( graphs.back(), parent_ids, partitions, k, lmax, mem_stack );
                    
                }
               // auto bweights_host = Kokkos::create_mirror_view(offspring.bweights);
               // Kokkos::deep_copy(bweights_host, offspring.bweights);
//
               // std::cout << "Block weights of offspring " << i << ": ";
               // for (partition_t j = 0; j < k; ++j) {
               //     std::cout << bweights_host(j);
               //     if (j + 1 < k) std::cout << ", ";
               // }
               // std::cout << std::endl;

                edge_cut_offspring[i] =  edge_cut(graphs.back(), offspring);
                
                // std::cout << "Edge cut of offspring " << i << ": " << edge_cut_offspring[i] << std::endl;
                
                max_block_weight_offspring[i] = max_weight(offspring);

                {
                    ScopedTimer _t("memetic", "memetic_refinement", "refine individual");  
                    auto pair = jet_refine( graphs.back(), offspring, k, lmax, use_ultra, level, 
                    edge_cut_offspring[i], max_block_weight_offspring[i], mem_stack, exec_spaces[0] );
                    
                    edge_cut_offspring[i] = pair.first;
                    max_block_weight_offspring[i] = pair.second;
                }

                //std::cout << "Edge cut of offspring after refinement " << i << ": " << edge_cut_offspring[i] << std::endl;


                assert_state_after_partition(graphs.back(), offspring, config.k);
                {
                    ScopedTimer _t("memetic", "memetic_refinement", "update population");   
                    UpdatePopulation(offspring, mem_stack, i);
                }

            }

            return;
        }

        

        //TODO: add distance stuff
        void UpdatePopulation(
            Partition &offspring, 
            KokkosMemoryStack &mem_stack,
            u32 offspring_id
        
        ) {

            determine_min_distances_population(graphs.back(), partitions, min_distances, k, mem_stack);
            u32 min_distance_population = *std::min_element(min_distances.begin(), min_distances.end());

            u32 offspring_distance = determine_min_distance_offspring(graphs.back(), partitions, offspring, k, mem_stack);

            std::cout << "Min distances: ";
            for (size_t i = 0; i < min_distances.size(); ++i) {
                std::cout << min_distances[i];
                if (i + 1 < min_distances.size()) std::cout << ", ";
            }
            std::cout << std::endl;

            std::cout << "min distance: " << min_distance_population << std::endl;
            std::cout << "Offspring distance: " << offspring_distance << std::endl;
            

            size_t worst_id = 0;
            weight_t worst_edgecut = curr_edge_cut[0];
            for (size_t i = 1; i < curr_edge_cut.size(); ++i) {
                if (curr_edge_cut[i] > worst_edgecut) {
                    worst_edgecut = curr_edge_cut[i];
                    worst_id = i;
                }
            }


            // for now: replace worst individual
            if( edge_cut_offspring[offspring_id] < worst_edgecut || ( offspring_distance > min_distance_population )) {
                
                auto rN = std::make_pair<size_t, size_t>(0, graphs.back().n);
                deep_copy(Kokkos::subview(partitions[ worst_id ].map, rN), Kokkos::subview(offspring.map, rN));
                deep_copy(partitions[ worst_id ].bweights, offspring.bweights);
            
                curr_edge_cut[ worst_id  ] = edge_cut_offspring[offspring_id];
                curr_max_block_weight[ worst_id ] = max_block_weight_offspring[offspring_id];

            }
                        
            free_partition(offspring, mem_stack);

            
            return;
        }


    };
}

#endif //GPU_HEIPA_SOLVER_H
