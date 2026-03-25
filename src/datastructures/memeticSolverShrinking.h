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

#ifndef GPU_HEIPA_SOLVER_MEMETIC_SHRINKING_H
#define GPU_HEIPA_SOLVER_MEMETIC_SHRINKING_H

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
#include "../refinement/memetic_refinementShrinking.h"
#include "../initial_partitioning/metis_partitioning.h"
#include "../utility/definitions.h"
#include "../utility/memetic_configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/edge_cut.h"

#include <cuda_runtime.h>
#include "omp.h"

namespace GPU_HeiPa {

    enum class PopulationManagement {
        steadystate,
        shrinking // #partitions == (level + 1)
        };

    class memeticSolverShrinking {
    public:
        MemeticConfiguration config;

        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;
        bool use_ultra = true;

        //! only for experiments
        weight_t average_weight = 0;

        std::vector<Graph> graphs;
        std::vector<Mapping> mappings;

        // configurable via MemeticConfiguration
        size_t num_cpu_threads = 4;
        size_t num_individuals = 20;
        
        u32 num_crossovers = 1;
        u32 num_parents = 2;
        u32 tournament_size = 2;

        PopulationManagement pop_management = PopulationManagement::shrinking;

        size_t orga_stack = num_cpu_threads;
        size_t partition_stack_a = num_cpu_threads + 1;
        size_t partition_stack_b = num_cpu_threads + 2;
        
        Partition dummy;

        std::vector<std::vector<Partition>> solutions = std::vector<std::vector<Partition>>();
        std::vector<bool> active_b = std::vector<bool>();
        std::vector<size_t> stack_ids = std::vector<size_t>(2);
        size_t parents_curr = num_individuals;
        size_t count_active;


        std::vector<size_t> active = std::vector<size_t>(num_individuals);
        
        
        std::vector<cudaStream_t> cuda_streams = std::vector<cudaStream_t>(num_cpu_threads);
        std::vector<Kokkos::Cuda> exec_spaces; 


        std::vector<weight_t> curr_edge_cut = std::vector<weight_t>(num_individuals + num_crossovers, 0);
        std::vector<weight_t> curr_max_block_weight = std::vector<weight_t>(num_individuals + num_crossovers, 0);
        std::vector<weight_t> initial_edge_cut = std::vector<weight_t>(num_individuals + num_crossovers, 0);
        std::vector<weight_t> initial_max_block_weight = std::vector<weight_t>(num_individuals + num_crossovers, 0);

        std::vector<u32> min_distances = std::vector<u32>(num_individuals + num_crossovers, 0);

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

        explicit memeticSolverShrinking(MemeticConfiguration t_config) : config(std::move(t_config)) {
            num_cpu_threads = config.num_cpu_threads;
            num_individuals = config.num_individuals;
            num_crossovers = config.num_crossovers;
            num_parents = config.num_parents;
            tournament_size = config.tournament_size;

            if (config.population_management == "steadystate") {
                pop_management = PopulationManagement::steadystate;
            } else {
                pop_management = PopulationManagement::shrinking;
            }

            orga_stack = num_cpu_threads;
            partition_stack_a = num_cpu_threads + 1;
            partition_stack_b = num_cpu_threads + 2;

            stack_ids[0] = partition_stack_a;
            stack_ids[1] = partition_stack_b;


            cuda_streams = std::vector<cudaStream_t>(num_cpu_threads);
            curr_edge_cut = std::vector<weight_t>(num_individuals + num_crossovers, 0);
            curr_max_block_weight = std::vector<weight_t>(num_individuals + num_crossovers, 0);
            initial_edge_cut = std::vector<weight_t>(num_individuals + num_crossovers, 0);
            initial_max_block_weight = std::vector<weight_t>(num_individuals + num_crossovers, 0);
            min_distances = std::vector<u32>(num_individuals + num_crossovers, 0);


            active = std::vector<size_t>(num_individuals);
            for(size_t i = 0; i < num_individuals; ++i)
                active.at(i) = i;
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

            std::vector<KokkosMemoryStack> mem_stacks(num_cpu_threads + 3 );

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
            mem_stacks[ partition_stack_a ] = initialize_kokkos_memory_stack(
                    ( 5 ) * (size_t) host_g.n * sizeof(vertex_t) +
                    10 *(size_t) host_g.m * sizeof(vertex_t),
                    "Stack for partitions" 
                );

            mem_stacks[ partition_stack_b ] = initialize_kokkos_memory_stack(
                    ( 5 ) * (size_t) host_g.n * sizeof(vertex_t) +
                    10 *(size_t) host_g.m * sizeof(vertex_t),
                    "Stack for partitions" 
                );
            
            
            internal_solve(host_g, mem_stacks);

            auto p = get_time_point();
            
            // pick best partition out of all of them
            size_t min_id;
            weight_t min_edgecut;
            if(pop_management == PopulationManagement::steadystate) {

                min_id = 0;
                min_edgecut = curr_edge_cut[0];
                for(size_t i = 0; i < num_individuals; ++i) {
                    
                    std::cout << "fitness individual " << i << " = " << curr_edge_cut[i] << std::endl ;
                    if ( curr_edge_cut[i] < min_edgecut) {
                        min_id = i;
                        min_edgecut = curr_edge_cut[i];
                    }
                }

            }else if(pop_management == PopulationManagement::shrinking) {

                min_id = active[0];
                min_edgecut = curr_edge_cut[min_id];
                std::cout << "fitness active individual " << min_id << " = " << min_edgecut << std::endl ;

                for(size_t i = 1; i < active.size(); ++i) {
                    std::cout << "fitness active individual " << active[i] << " = " << curr_edge_cut[active[i]] << std::endl ;
                    if ( curr_edge_cut[ active[i] ] < min_edgecut) {
                        min_id = active[i];
                        min_edgecut = curr_edge_cut[ active[i] ];
                    }
                }
            }
            
            HostPartition host_partition;
            //
            {
                ScopedTimer _t("up/download", "Solver", "download_partition");
                host_partition = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), graphs.back().n);
                Kokkos::deep_copy(host_partition, solutions[0][min_id].map);
            }
            f64 down_ms = get_milli_seconds(p, get_time_point());
            down_up_load_ms += down_ms;

            // calc stats
            size_t n_empty_partitions = 0;
            size_t n_overloaded_partitions = 0;
            weight_t sum_too_much = 0;
            if (config.verbose_level >= 2) {
                ScopedTimer _t("misc", "Solver", "calc_stats");
                PartitionHost partition_host = to_host_partition(solutions[0][min_id]);
                for (partition_t id = 0; id < config.k; ++id) {
                    n_empty_partitions += partition_host.bweights(id) == 0;
                    n_overloaded_partitions += partition_host.bweights(id) > lmax;
                    sum_too_much += std::max((weight_t) 0, partition_host.bweights(id) - lmax);
                }
            }

            // free all memory
            {
                ScopedTimer _t("misc", "Solver", "free_memory");

                for (size_t i = count_active; i-- > 0; ) {
                    free_partition(solutions[0][i], mem_stacks[ stack_ids[0] ]);
                }

                free_partition(dummy, mem_stacks[ orga_stack ]);
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
                std::cout << "distance mode     : " << config.distance << std::endl;
                std::cout << "population mgmt   : " << config.population_management << std::endl;
                std::cout << "leftover strategy : " << config.leftover_strategy << std::endl;
                std::cout << "alpha             : " << config.alpha << std::endl;
                std::cout << "extent            : " << config.extent << std::endl;
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


/*

    inline void uncontract(Partition &partition,
                           const Mapping &mapping,
                           Kokkos::Cuda &exec_space
                        ) {
        ScopedTimer _t("uncontraction", "partition", "uncontract");

        // reset activity
        Kokkos::parallel_for("initialize",   
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, mapping.old_n), 
            KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t new_v = mapping.mapping(u);
            partition.temp_map(u) = partition.map(new_v);
        });
        std::swap(partition.map, partition.temp_map);

        KOKKOS_PROFILE_FENCE();
    }
*/



        void carryover_solution(
            Partition &partition_old,
            Partition &partition_new,
            int tid
        ) {

            Kokkos::parallel_for("carryover",
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[tid], 0, mappings.back().old_n ),
                KOKKOS_LAMBDA(const vertex_t u) {
                    vertex_t new_v = mappings.back().mapping(u);
                    partition_new.map(u) = partition_old.map(new_v);
                }
            );

            exec_spaces[tid].fence();

        };



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



            for(size_t i = 0; i < num_individuals; ++i) {
                solutions[level % 2].push_back( initialize_partition( graphs.back().n , k, lmax, mem_stacks[ stack_ids[level % 2]]) );
                active_b.push_back(true);
            }


            #pragma omp parallel for num_threads(num_cpu_threads)
            for(size_t i = 0; i < num_individuals ; ++i) {
                size_t tid = static_cast<size_t>(  omp_get_thread_num() );
                initial_partitioning( i, tid, level );
                
            }
            ScopedTimer _t("initial_partitioning", "Partition", "first_stats");

            initial_partitioning_ms += get_milli_seconds(p, get_time_point());
        }


#if ENABLE_PROFILER
            //! übergangslösung
            level_infos[level].max_b_weight = max_weight(solutions[level % 2][0]); 
            level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
            level_infos[level].edge_cut = edge_cut(graphs.back(), solutions[level % 2][0]);
            level_infos[level].empty_partitions = n_empty_blocks(solutions[level % 2][0]);
            level_infos[level].oload_partitions = n_oload_blocks(solutions[level % 2][0]);
            level_infos[level].sum_oload_weights = sum_oload_weight(solutions[level % 2][0]);
#endif

            while (!mappings.empty()) {
                level -= 1;
                

                {
                    // scope for uncontraction
                    auto p = get_time_point();

                    //TODO: 1. count how many new partitiosn there will be ( #active flags)
                    //TODO: 2. create new partitions vector (pushback[] = init_partition)
                    //TODO: 3. "copy and uncoarsen": use the uncoarsen method to fill new partitions
                    //TODO:    with values, based on old partitions
                    //TODO: 4. deallocate / destroy old partitions

                    count_active = 0;
                    std::vector<size_t> active_indices;
                    for(size_t i = 0; i < active_b.size() ; ++i) {
                        if(active_b[i]) {
                            count_active++;
                            active_indices.push_back(i);
                        }
                    }
                    parents_curr = count_active;

                    for(size_t i = 0; i < count_active ; ++i) {
                        solutions[ level % 2].push_back( initialize_partition( mappings.back().old_n , k, lmax, mem_stacks[ stack_ids[level % 2]]) );
                    }

                    //TODO: I also need to change curr_edge_cut
                    //TODO: - i need to resize it to parents_curr + num_crossovers
                    //TODO: - i need to carry over the old values
                    //TODO: Best Idea is to make a temporary vector and swap the values afterwards
                    //TODO: other Idea: introduce two vectors and swap between them, just like solutions or the mem_stacks

                    #pragma omp parallel for num_threads(num_cpu_threads)
                    for(size_t i = 0; i < count_active ; ++i) {
                        size_t tid = static_cast<size_t>(  omp_get_thread_num() );
                        size_t old_id = active_indices[i];
                        carryover_solution( 
                          solutions[(level + 1) % 2][old_id],
                          solutions[level % 2][i],
                          tid  
                          );
                    }

                    //TODO: here is also the place to reset active_b:
                    //TODO: - resize to parents_curr
                    //TODO: - fill with all values = true 

                    //TODO: free the old partitions from the stack
                    //TODO: pop everything from the old solutions vector (level + 1) % 2

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
                    for(size_t i = 0; i < count_active; ++i) {
                        size_t id = active[i];
                        size_t tid = static_cast<size_t>(  omp_get_thread_num() );
                        refinement( level, mem_stacks[tid], id, tid);
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
                    if(active.size() >= tournament_size)
                        memetic_refinement(level, mem_stacks);
                    
                    memetic_ms += get_milli_seconds(p, get_time_point());

#if ENABLE_PROFILER
                    level_infos[level].t_memetic = get_milli_seconds(p, get_time_point());
#endif
                }



#if ENABLE_PROFILER
            //! übergangslösung
            level_infos[level].max_b_weight = max_weight(solutions[level % 2][0]); 
            level_infos[level].imb = (f64) level_infos[level].max_b_weight / ((f64) host_g.g_weight / (f64) config.k);
            level_infos[level].edge_cut = edge_cut(graphs.back(), solutions[level % 2][0]);
            level_infos[level].empty_partitions = n_empty_blocks(solutions[level % 2][0]);
            level_infos[level].oload_partitions = n_oload_blocks(solutions[level % 2][0]);
            level_infos[level].sum_oload_weights = sum_oload_weight(solutions[level % 2][0]);
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

            //! only for experiments
            average_weight = (weight_t) std::ceil((f64) host_g.g_weight / (f64) config.k) ;

            use_ultra = config.config == "ultra";

            f64 up_ms = 0;
            
            graphs.emplace_back(from_HostGraph(host_g, mem_stacks[ orga_stack ], up_ms));

            dummy = initialize_partition(n, k, lmax, mem_stacks[ orga_stack ]);
            


            misc_ms += get_milli_seconds(p, get_time_point());
            misc_ms -= up_ms;
            down_up_load_ms += up_ms;

            assert_state_pre_partition(graphs.back());
        }

        void coarsening(u32 level, KokkosMemoryStack &mem_stack) {
            auto p = get_time_point();


            //TODO: pass dummy mapping / remove entirely
            mappings.emplace_back(two_hop_matcher_get_mapping(graphs.back(), dummy, lmax, mem_stack));

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
            
            Kokkos::fence();
            contraction_ms += get_milli_seconds(p, get_time_point());

#if ENABLE_PROFILER
            level_infos[level].t_contraction = get_milli_seconds(p, get_time_point());
#endif

            assert_state_pre_partition(graphs.back());
        }


        void initial_partitioning(size_t individual_id, size_t tid, u32 level) {
            

            //! better way to pick the seed? more randomness?
            metis_partition(graphs.back(), (int) k, config.imbalance, config.seed + individual_id, solutions[level % 2][ individual_id ], METIS_RECURSIVE);

            recalculate_weights(solutions[level % 2][individual_id], graphs.back());
            
            
            initial_edge_cut[individual_id] = edge_cut(graphs.back(), solutions[level % 2][individual_id], exec_spaces[tid]); 
            curr_edge_cut[individual_id] = initial_edge_cut[individual_id]; 

            initial_max_block_weight[individual_id] = max_weight(solutions[level % 2][individual_id], exec_spaces[tid] ); 
            curr_max_block_weight[individual_id] = initial_max_block_weight[individual_id]; 

            exec_spaces[tid].fence();

            assert_state_after_partition(graphs.back(), solutions[level % 2][individual_id], config.k);
        }

        void refinement(u32 level, KokkosMemoryStack &mem_stack, size_t individual_id, size_t tid) {
            
            auto pair = jet_refine(graphs.back(), solutions[level % 2][ individual_id ], k, lmax, use_ultra, level, curr_edge_cut[individual_id], curr_max_block_weight[individual_id], mem_stack, exec_spaces[tid]);
            
            curr_edge_cut[individual_id] = pair.first;
            curr_max_block_weight[individual_id] = pair.second;

            exec_spaces[tid].fence();

            ASSERT(curr_edge_cut[individual_id] == edge_cut(graphs.back(), solutions[level % 2][individual_id]));
            ASSERT(curr_max_block_weight[individual_id] == max_weight(solutions[level % 2][individual_id]));

            assert_state_after_partition(graphs.back(), solutions[level % 2][individual_id], config.k);
        }

        void free_after_uncontraction(KokkosMemoryStack &mem_stack) {
           
            free_graph(graphs.back(), mem_stack);
            graphs.pop_back();

            free_mapping(mappings.back(), mem_stack);
            mappings.pop_back();

            Kokkos::fence();
        }

        f64 determine_goodness_score( size_t id ) {
            //TODO: reactivate
            // f64 BETA = 0.08 * graphs.back().n ;
            // return (curr_edge_cut[id] + BETA/min_distances[id]);
            return static_cast<f64>(curr_edge_cut[id]);
        }


        void memetic_refinement(u32 level, std::vector<KokkosMemoryStack> &mem_stacks) {

            for(u32 i = 0; i < num_crossovers; ++i) {
                solutions[level % 2].push_back( initialize_partition( mappings.back().old_n , k, lmax, mem_stacks[ stack_ids[level % 2]]) );
                active_b.push_back(true);
            }


            // TODO: #pragma omp parallel for num_threads(num_threads)
            for(u32 i = 0; i < num_crossovers; ++i) {

                std::vector<int> parent_ids;

                    //! curr edge cut is auch noch problematisch...
                    for(u32 j= 0; j < num_parents; ++j) {
                        parent_ids.push_back( tournament_selection( curr_edge_cut , tournament_size, parents_curr) );
                    }
                    
                    // Ensure parent_ids[0] != parent_ids[1]
                    //! des is auch noch weird
                    // if (parent_ids.size() == 2 && parent_ids[0] == parent_ids[1]) {
                    //     if (active.size() >= 2) {
                    //         size_t idx;
                    //         size_t new_parent = static_cast<size_t>(parent_ids[0]);
                    //         while (new_parent == static_cast<size_t>(parent_ids[0])) {
                    //             idx = static_cast<size_t>(rand()) % active.size();
                    //             new_parent = active.at(idx);
                    //         }
                    //         parent_ids[1] = static_cast<int>(new_parent);
                    //     }
                    // }
                

                
 
                backbone_based_crossover(
                        solutions[level % 2][ parents_curr + i ],        
                        graphs.back(),
                        parent_ids,
                        solutions[level % 2],
                        k,
                        lmax,
                        mem_stacks[ 0 /* tid */ ],
                        config.leftover_strategy,
                        config.alpha,
                        config.extent
                    );
                    
                


                curr_edge_cut[parents_curr + i] =  edge_cut(graphs.back(), solutions[level % 2][ parents_curr + i ]);
                
                curr_max_block_weight[parents_curr + i] = max_weight( solutions[level % 2][ parents_curr + i ] );


                auto pair = jet_refine( graphs.back(), solutions[level % 2][ parents_curr + i ], k, lmax, use_ultra, level, 
                curr_edge_cut[parents_curr + i], curr_max_block_weight[parents_curr + i], mem_stacks[ 0 /* tid */ ], exec_spaces[0 /* tid */ ] );
                curr_edge_cut[parents_curr + i] = pair.first;
                curr_max_block_weight[parents_curr + i] = pair.second;
                


                assert_state_after_partition(graphs.back(), solutions[level % 2][ parents_curr + i ], config.k);
                

            }

            {

                ScopedTimer _t("memetic", "memetic_refinement", "update population");   
                UpdatePopulation( mem_stacks );
                
                control_size(level);
            }
            
            return;
        }

        // deactivate some bad individuals
        void control_size(u32 level) {

            
            // fine tune
            size_t desired_count = level + 1;

            while(count_active > desired_count) {

                size_t worst_id;
                f64 worst_goodness_score = 0.0f;

                f64 curr_goodness_score;
                for (size_t i = 0; i < active_b.size(); ++i) {

                    if(!active_b[i])
                        continue;
    
                    curr_goodness_score = determine_goodness_score(i);
                    if( curr_goodness_score > worst_goodness_score) {
                        worst_goodness_score = curr_goodness_score;
                        worst_id = i;
                    }
                    
                    
                }

                count_active--;
                active_b[worst_id] = false;
                
            }

            return;
        }
        

        void UpdatePopulation(
            std::vector<KokkosMemoryStack> &mem_stacks
        
        ) {
            /*
            
            TODO: add later on when the rest already works
            TODO: -> focus on main idea first

            enum class DistanceMode {
                Exact,
                Sampled
            };

            const size_t SAMPLE_SIZE = 5;               
           
            DistanceMode distance_mode = DistanceMode::Exact;
            if (config.distance == "sampled") {
                distance_mode = DistanceMode::Sampled;
            }

            u32 min_distance_population = 0;
            u32 offspring_distance = 0;

            switch (distance_mode) {
                case DistanceMode::Sampled:
                    determine_min_distances_population_sampled(
                        graphs.back(),
                        partitions,
                        min_distances,
                        k,
                        mem_stacks,
                        exec_spaces,
                        num_cpu_threads,
                        SAMPLE_SIZE
                    );
                    break;

                case DistanceMode::Exact:
                default:
                    determine_min_distances_population(
                        graphs.back(),
                        partitions,
                        active,
                        min_distances,
                        k,
                        mem_stacks,
                        exec_spaces,
                        num_cpu_threads
                    );
                    break;
            }


             switch (distance_mode) {
                case DistanceMode::Sampled:
                    offspring_distance = determine_min_distance_offspring_sampled(
                        graphs.back(),
                        partitions,
                        offspring,
                        k,
                        mem_stacks,
                        exec_spaces,
                        num_cpu_threads,
                        SAMPLE_SIZE
                    );
                    break;

                case DistanceMode::Exact:
                default:
                    offspring_distance = determine_min_distance_offspring(
                        graphs.back(),
                        partitions,
                        active,
                        offspring,
                        k,
                        mem_stacks,
                        exec_spaces,
                        num_cpu_threads
                    );
                    break;
            }


            min_distance_population = std::numeric_limits<u32>::max();
            for (size_t id : active) {
                min_distance_population = std::min(min_distance_population, min_distances[id]);
            }

            */

            for(u32 offspring_id = 0; offspring_id < num_crossovers ; ++offspring_id) {
                
                weight_t best_edgecut = std::numeric_limits<u32>::max();
                
                size_t worst_id = 0;
                f64 worst_goodness_score = 0.0;
                
                
                f64 curr_goodness_score;
                for (size_t i = 0; i < parents_curr; ++i) {

                    if( !active_b[i])
                        continue;

                    if( curr_edge_cut[i] < best_edgecut)
                        best_edgecut = curr_edge_cut[i];
                    
                    curr_goodness_score = determine_goodness_score(i);
                    if( curr_goodness_score > worst_goodness_score) {
                        worst_goodness_score = curr_goodness_score;
                        worst_id = i;
                    }
                    
                }
                
                
                
                if( 
                    (curr_edge_cut[parents_curr + offspring_id] < best_edgecut)
                    // ||  ( offspring_distance > min_distance_population )
                ) {

                    active_b[worst_id] = false;
                    active_b[parents_curr + offspring_id] = true;
                }
            
            }


            
            return;
        }


    };
}

#endif //GPU_HEIPA_SOLVER_H
