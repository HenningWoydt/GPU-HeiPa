#ifndef GPU_HEIPA_SOLVER_REC_BISEC_GPU_H
#define GPU_HEIPA_SOLVER_REC_BISEC_GPU_H

#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_StaticCrsGraph.hpp>

#include "graph.h"
#include "host_graph.h"
#include "mapping.h"
#include "partition.h"
#include "../initial_partitioning/global_multisection.h"
#include "../utility/definitions.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "solver.h"



namespace GPU_HeiPa {

    struct Accumulators {
    u32 partial_0s = 0;
    u32 partial_1s = 0;
    
    KOKKOS_INLINE_FUNCTION
    void operator+=(const Accumulators& rhs) {
        partial_0s += rhs.partial_0s;
        partial_1s += rhs.partial_1s;
    }
    };

    class SolverRecursiveBisectionGood {


        public:
            Configuration config;
            weight_t global_g_w;
            HostPartition solution;

            explicit SolverRecursiveBisectionGood(Configuration t_config) : config(std::move(t_config)) {
                global_g_w = 0;
        }

            HostPartition solve(HostGraph &host_g) {

                if (!isPowerOfTwo(config.k)) {
                    throw std::invalid_argument("k must be a power of two");
                }

                if( config.k == 1) {
                    throw std::invalid_argument("k must be at least 2");    
                }
                
                internal_solve(host_g);

                return solution;
            }


        

        private:

            bool isPowerOfTwo(u32 n) {
                return (n > 0) && ((n & (n - 1)) == 0);
            }


            /*
             pos: Describes the partitions position in the recursin tree
             Each pos[i] describes if you went right (1) or left (0) on level i
             
             mappings: Holds the set of mappings to convert from the vertex IDs
             of a subgraph to the vertex IDs of the original graph (1 level above)
              
             
            */
            void internal_solve(HostGraph &host_g) {

                std::vector<int> pos = {};
                int level = (int)std::log2(config.k);
                std::vector<vertex_t> mapping(host_g.n); // mapping between subgraph and original vertices
                for (vertex_t u = 0; u < host_g.n; ++u) mapping[u] = u; // init with identity mapping

                solution = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "solution"), host_g.n);
    
                global_g_w = host_g.g_weight;

                recursive_bisection(host_g, level, pos, mapping);

                return;
            }


            void recursive_bisection(HostGraph &in_g, int level, std::vector<int> &pos, std::vector<u32> &mapping) {
                
                Configuration internal_config; 

                internal_config = config;
                internal_config.k = 2;
                internal_config.imbalance = determine_adaptive_imbalance(
                    config.imbalance,
                    global_g_w,
                    config.k,
                    in_g.g_weight,
                    pow(2, level),
                    level
                );
                internal_config.verbose_level = 0;

                HostPartition in_partition = Solver(internal_config).solve(in_g);

                if(level == 1) {
                    ScopedTimer _t("recursive_bisection", "recursive_bisection", "propagate_solution");
                    propagate_solution(in_partition, in_g, mapping, pos);
                    return;

                } else{
                    ScopedTimer _t("recursive_bisection", "recursive_bisection", "create_subgraph_like_Henning");
                    
                    HostGraph left_graph, right_graph;
                    std::vector<u32> left_new_to_old = {}; // the mappings between new and old vertex IDs
                    std::vector<u32> right_new_to_old = {}; 
                    create_subgraph_like_Henning(in_partition, in_g, left_graph, right_graph, left_new_to_old, right_new_to_old, mapping);

                    std::vector<int> pos_left_graph, pos_right_graph;
                    pos_left_graph = pos_right_graph = pos;  // copy wont hurt because this is super small (like 10 entries)
                    pos_left_graph.push_back(0);
                    pos_right_graph.push_back(1);

                    _t.stop();

                    recursive_bisection(left_graph, level-1, pos_left_graph, left_new_to_old); // go down to the next lower level
                    recursive_bisection(right_graph, level-1, pos_right_graph, right_new_to_old);
                    return;
                }
                
            }


            /**
             * This function takes a subgraph on the last level of the recursive bisection
             * and translates (propagates) the partition found on this graph, to a 
             * partition on the input graph
             */
            void propagate_solution(const HostPartition& local_partition, const HostGraph& local_graph,
                        const std::vector<u32>& mapping, const std::vector<int>& pos) {
                            
                            // build global partition id from pos bits
                            partition_t global_partition_id = 0;
                            for (size_t i = 0; i < pos.size(); ++i) {
                                //global_partition_id += pos[i] * pow(2, pos.size() - i);
                                global_partition_id += pos[i] * ( 1 << (pos.size() - i) );
                            }
                        
                            for (vertex_t u = 0; u < local_graph.n; ++u) {
                                vertex_t original_id = mapping[u];

                                partition_t full_id = global_partition_id + local_partition(u);
                            
                                solution(original_id) = full_id;
                            }
                            return;
            }


            /**
             * Returns (populate) two subgraphs:
             *  left graph : holds all nodes with partition ID 0
             *  right graph: holds all nodes with partition ID 1
             *
             *  
             * @param TODO
             * @return TODO
             */
            void create_subgraph(HostPartition input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph, HostGraph &right_graph,
                                 std::vector<u32> &left_new_to_old,
                                 std::vector<u32> &right_new_to_old
            ) {

                vertex_t num_vertices[2], num_edges[2];
                weight_t weights[2];
                
                num_vertices[0] = num_vertices[1] = num_edges[0] = num_edges[1] = 0;
                weights[0] = weights[1] = 0;

                HostU32 rename = HostU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_renaming"), input_graph.n  ) ;

                
                // Get the initial information to create the two subgraphs:
                partition_t partition;
                for(vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition = input_partition(u);

                    rename[u] = num_vertices[partition];
                    num_vertices[partition]++ ;
                    weights[partition] += input_graph.weights(u);
                    num_edges[partition] += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u); // fast upper bound
                }
                

                // init the two HostGraphs
                allocate_memory(left_graph, num_vertices[0], num_edges[0], weights[0]);
                allocate_memory(right_graph, num_vertices[1], num_edges[1], weights[1]);

                //TODO: parallelize

                //! This can be parallelized with a parallel_for (iterations are fully independent)
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            if(my_part == 0 ) {

                                left_graph.neighborhood( rename(u) )++;

                            }else{
                                
                                right_graph.neighborhood( rename(u) )++;

                            }

                        }


                    }
                }

                //! These can be parallelized with a parallel scan
                //! + You can combine these two into one loop with a minimum over the two n values
                //! + And then add a smaller loop for the remaining elements
                left_graph.neighborhood(left_graph.n) = 0;
                for(vertex_t u = 1; u < left_graph.n + 1; ++u)
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                
                right_graph.neighborhood(right_graph.n) = 0;
                for(vertex_t u = 1; u < right_graph.n + 1; ++u)
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);

                

                // work on the mapping of new vertex IDs to old vertex IDs :
                left_new_to_old.resize( num_vertices[0] );
                right_new_to_old.resize( num_vertices[1 ]);
                
                vertex_t idx_l, idx_r;
                //! This is again a parallel for (independent iterations)
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);

                    if( my_part == 0) {
                        left_graph.weights( rename(u) ) = input_graph.weights(u);

                        // update left new to old mapping
                        left_new_to_old.at( rename(u) ) = u ;
                    } else{
                        right_graph.weights( rename(u)) = input_graph.weights(u);

                        //update mapping
                        right_new_to_old.at( rename(u) ) = u;
                    }

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            if(my_part == 0 ) {
                                idx_l = --left_graph.neighborhood( rename(u) ) ;

                                left_graph.edges_v( idx_l ) = rename(v);
                                left_graph.edges_w( idx_l ) = input_graph.edges_w(i);

                            }else{
                                
                                idx_r = --right_graph.neighborhood( rename(u) ) ;

                                right_graph.edges_v( idx_r ) = rename(v);
                                right_graph.edges_w( idx_r ) = input_graph.edges_w(i);

                            }

                        }


                    }
                }


                return;
            }


            void create_subgraph_clean(HostPartition input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph, HostGraph &right_graph,
                                 std::vector<u32> &left_new_to_old,
                                 std::vector<u32> &right_new_to_old
            ) {

                HostGraph* subgraphs[2];
                subgraphs[0] = &left_graph;
                subgraphs[1] = &right_graph;

                vertex_t num_vertices[2], num_edges[2];
                weight_t weights[2];
                
                num_vertices[0] = num_vertices[1] = num_edges[0] = num_edges[1] = 0;
                weights[0] = weights[1] = 0;

                HostU32 rename = HostU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_renaming"), input_graph.n  ) ;

                
                // Get the initial information to create the two subgraphs:
                partition_t partition;
                for(vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition = input_partition(u);

                    rename[u] = num_vertices[partition];
                    num_vertices[partition]++ ;
                    weights[partition] += input_graph.weights(u);
                    num_edges[partition] += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u); // fast upper bound
                }
                

                // init the two HostGraphs
                allocate_memory(left_graph, num_vertices[0], num_edges[0], weights[0]);
                allocate_memory(right_graph, num_vertices[1], num_edges[1], weights[1]);

                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            subgraphs[my_part]->neighborhood( rename(u) )++;

                        }


                    }
                }


                left_graph.neighborhood(left_graph.n) = 0;
                right_graph.neighborhood(right_graph.n) = 0;

                vertex_t min_n = std::min(left_graph.n, right_graph.n);
                for(vertex_t u = 1; u < min_n + 1; ++u) {
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);
                }
                
                for(vertex_t u = min_n + 1; u < left_graph.n + 1; ++u)
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                
                for(vertex_t u = min_n + 1; u < right_graph.n + 1; ++u)
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);

                

                // work on the mapping of new vertex IDs to old vertex IDs :
                left_new_to_old.resize( num_vertices[0] );
                right_new_to_old.resize( num_vertices[1 ]);
                
                std::vector<u32>* mappings[2];
                mappings[0] = &left_new_to_old;
                mappings[1] = &right_new_to_old;

                
                vertex_t idx;
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);


                    subgraphs[my_part]->weights( rename(u) ) = input_graph.weights(u);
                    mappings[my_part]->at( rename(u) ) = u;

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            idx = --subgraphs[my_part]->neighborhood(rename(u));
                            subgraphs[my_part]->edges_v(idx) = rename(v);
                            subgraphs[my_part]->edges_w(idx) = input_graph.edges_w(i);

                        }

                    }
                }

                return;
            }


            void create_subgraph_like_Henning(HostPartition input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph, HostGraph &right_graph,
                                 std::vector<u32> &left_new_to_old,
                                 std::vector<u32> &right_new_to_old,
                                 std::vector<u32> &curr_mapping   // this is "parent_n_to_o" in Hennings method
            ) {

                HostGraph* subgraphs[2];
                subgraphs[0] = &left_graph;
                subgraphs[1] = &right_graph;

                vertex_t num_vertices[2], num_edges[2];
                weight_t weights[2];
                
                num_vertices[0] = num_vertices[1] = num_edges[0] = num_edges[1] = 0;
                weights[0] = weights[1] = 0;

                HostU32 rename = HostU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_renaming"), input_graph.n  ) ;

                
                // Get the initial information to create the two subgraphs:
                partition_t partition;
                for(vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition = input_partition(u);

                    rename[u] = num_vertices[partition];
                    num_vertices[partition]++ ;
                    weights[partition] += input_graph.weights(u);
                    num_edges[partition] += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u); // fast upper bound
                }
                

                // init the two HostGraphs
                allocate_memory(left_graph, num_vertices[0], num_edges[0], weights[0]);
                allocate_memory(right_graph, num_vertices[1], num_edges[1], weights[1]);

                // work on the mapping of new vertex IDs to old vertex IDs :
                left_new_to_old.resize( num_vertices[0] );
                right_new_to_old.resize( num_vertices[1 ]);
                
                std::vector<u32>* mappings[2];
                mappings[0] = &left_new_to_old;
                mappings[1] = &right_new_to_old;


                partition_t my_part;
                vertex_t idx;
                for(vertex_t u = 0; u < input_graph.n; ++u) {
                    my_part = input_partition(u);

                    subgraphs[my_part]->weights( rename(u) ) = input_graph.weights(u);

                    //! this is the crucial change: This maps the new mapping right back to the
                    //! input graph
                    mappings[my_part]->at( rename(u) ) = curr_mapping.at(u);

                    
                    subgraphs[my_part]->neighborhood( rename(u) + 1) = subgraphs[my_part]->neighborhood( rename(u) );

                    for(vertex_t i = input_graph.neighborhood(u) ; i < input_graph.neighborhood(u+1); ++i ) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            idx = subgraphs[my_part]->neighborhood(rename(u)+1);

                            subgraphs[my_part]->edges_v(idx) = rename(v);
                            subgraphs[my_part]->edges_w(idx) = input_graph.edges_w(i);
                            subgraphs[my_part]->neighborhood(rename(u)+1) += 1;
                        }
                    }

                }

                return;
            }

            void create_subgraph_GPU_wrapper(HostPartition &input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph, HostGraph &right_graph,
                                 std::vector<u32> &left_new_to_old,
                                 std::vector<u32> &right_new_to_old,
                                 std::vector<u32> &curr_mapping
            ) {
                // convert data from CPU to GPU
                
                create_subgraph_GPU_vertexParallel() ;

                // convert back

                return;
            }

            /*
            
            Its not quite clear if parallelizing over vertices or over the edges is better here.
            So i will try it out both versions and empirically find out which is better!
            
            First, for the sake of simplicity i will start with vertex parallelism, because then
            i can pretty much copy-paste most of the existing solutions
            -> replace #pragma omp parallel for with Kokkos::for and change locations of memory
            ! But first i need to fix the logic for the mapping from new vertex IDs to the original ones!
            */
            void create_subgraph_GPU_vertexParallel() {

                HostGraph* subgraphs[2]; //TODO: this should be a graph on the GPU -> change input 
                subgraphs[0] = &left_graph;
                subgraphs[1] = &right_graph;

                vertex_t num_vertices[2], num_edges[2]; //TODO: this should all be on the GPU
                weight_t weights[2]; //TODO: because i am using it in the following code
                
                //TODO: init on GPU
                num_vertices[0] = num_vertices[1] = num_edges[0] = num_edges[1] = 0;
                weights[0] = weights[1] = 0;

                //TODO: should be on GPU
                std::vector<u32> rename = std::vector<u32>(input_graph.n);
                

                //TODO: parallelize ? parallel scan over num_vertices[partition] ?
                /*partition_t partition;
                for(vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition = input_partition(u);

                    //? looks like a prefix sum wrt rename
                    //? but a typical reduction wrt num_vertices...
                    //? split this into two loops? reduction & scan ?
                    rename[u] = num_vertices[partition];
                    num_vertices[partition]++ ; 

                    //! these are definetly reductions, because ordering doesnt matter!
                    weights[partition] += input_graph.weights(u);
                    num_edges[partition] += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u); // fast upper bound
                }
                */

                Kokkos::parallel_scan("rename vertices", input_graph.n,
                KOKKOS_LAMBDA( u32 u, Accumulators &acc, bool final ) {
                    partition_t partition = in_partition(u);

                    if(partition == 0) {
                        if(final) rename(u) = acc.partial_0s;
                        acc.partial_0s += 1;
                    } else {
                        if(final) rename(u) = acc.partial_1s;
                        acc.partial_1s += 1;
                    }
                } //TODO: retrieve the final result of the accumulation
                );


                // init the two HostGraphs
                //TODO: do this on the GPU -> which function might be good for this?
                allocate_memory(left_graph, num_vertices[0], num_edges[0], weights[0]);
                allocate_memory(right_graph, num_vertices[1], num_edges[1], weights[1]);

                //TODO: kokkos parallel for
                //TODO: experiment: Is it better to parallelize over the vertices or edges? (no-atomics vs coalescing)
                #pragma omp parallel for
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            subgraphs[my_part]->neighborhood( rename.at(u) )++;

                        }


                    }
                }

                //? wie mache ich diese einzelnen calls, die nur ein thread machen soll ?
                left_graph.neighborhood(left_graph.n) = 0;
                right_graph.neighborhood(right_graph.n) = 0;

                vertex_t min_n = std::min(left_graph.n, right_graph.n);

                //TODO: parallel scan
                for(vertex_t u = 1; u < min_n + 1; ++u) {
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);
                }
                
                for(vertex_t u = min_n + 1; u < left_graph.n + 1; ++u)
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                
                for(vertex_t u = min_n + 1; u < right_graph.n + 1; ++u)
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);

                

                // work on the mapping of new vertex IDs to old vertex IDs :
                left_new_to_old.resize( num_vertices[0] );
                right_new_to_old.resize( num_vertices[1 ]);
                
                //TODO: i guess this should become a view?
                std::vector<u32>* mappings[2];
                mappings[0] = &left_new_to_old;
                mappings[1] = &right_new_to_old;


                //? This has to be a parallel for over the vertices?
                //? Loop over all edges?

                #pragma omp parallel for
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);


                    subgraphs[my_part]->weights( rename.at(u) ) = input_graph.weights(u);
                    mappings[my_part]->at( rename.at(u) ) = curr_mapping.at(u);

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            vertex_t idx = --subgraphs[my_part]->neighborhood(rename.at(u));
                            subgraphs[my_part]->edges_v(idx) = rename.at(v);
                            subgraphs[my_part]->edges_w(idx) = input_graph.edges_w(i);

                        }

                    }
                }

             



                return;
            }

            void create_subgraph_parallel(HostPartition &input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph, HostGraph &right_graph,
                                 std::vector<u32> &left_new_to_old,
                                 std::vector<u32> &right_new_to_old,
                                 std::vector<u32> &curr_mapping
            ) {

                HostGraph* subgraphs[2];
                subgraphs[0] = &left_graph;
                subgraphs[1] = &right_graph;

                vertex_t num_vertices[2], num_edges[2];
                weight_t weights[2];
                
                num_vertices[0] = num_vertices[1] = num_edges[0] = num_edges[1] = 0;
                weights[0] = weights[1] = 0;

                std::vector<u32> rename = std::vector<u32>(input_graph.n);
                
                // Get the initial information to create the two subgraphs:
                partition_t partition;
                for(vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition = input_partition(u);

                    rename[u] = num_vertices[partition];
                    num_vertices[partition]++ ;
                    weights[partition] += input_graph.weights(u);
                    num_edges[partition] += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u); // fast upper bound
                }
                

                // init the two HostGraphs
                allocate_memory(left_graph, num_vertices[0], num_edges[0], weights[0]);
                allocate_memory(right_graph, num_vertices[1], num_edges[1], weights[1]);

                #pragma omp parallel for
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            subgraphs[my_part]->neighborhood( rename.at(u) )++;

                        }


                    }
                }


                left_graph.neighborhood(left_graph.n) = 0;
                right_graph.neighborhood(right_graph.n) = 0;

                vertex_t min_n = std::min(left_graph.n, right_graph.n);
                for(vertex_t u = 1; u < min_n + 1; ++u) {
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);
                }
                
                for(vertex_t u = min_n + 1; u < left_graph.n + 1; ++u)
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                
                for(vertex_t u = min_n + 1; u < right_graph.n + 1; ++u)
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);

                

                // work on the mapping of new vertex IDs to old vertex IDs :
                left_new_to_old.resize( num_vertices[0] );
                right_new_to_old.resize( num_vertices[1 ]);
                
                std::vector<u32>* mappings[2];
                mappings[0] = &left_new_to_old;
                mappings[1] = &right_new_to_old;


                #pragma omp parallel for
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);

                    mappings[my_part]->at( rename.at(u) ) = curr_mapping.at(u);
                    
                    subgraphs[my_part]->weights( rename.at(u) ) = input_graph.weights(u);
                    

                    for(vertex_t i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                        vertex_t v = input_graph.edges_v(i);
                        partition_t neighbor_part = input_partition(v);

                        if( my_part == neighbor_part) {

                            vertex_t idx = --subgraphs[my_part]->neighborhood(rename.at(u));
                            subgraphs[my_part]->edges_v(idx) = rename.at(v);
                            subgraphs[my_part]->edges_w(idx) = input_graph.edges_w(i);

                        }

                    }
                }

                return;
            }
        
        
    };

}


#endif