#ifndef GPU_HEIPA_SOLVER_REC_BISEC_HOST_H
#define GPU_HEIPA_SOLVER_REC_BISEC_HOST_H

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

    

    class SolverRecursiveBisectionHost {


        public:
            Configuration config;
            weight_t global_g_w;
            HostPartition solution;

            explicit SolverRecursiveBisectionHost(Configuration t_config) : config(std::move(t_config)) {
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

                std::vector<u32> pos = {};
                int level = (int)std::log2(config.k);
                std::vector<vertex_t> mapping(host_g.n); // mapping between subgraph and original vertices
                for (vertex_t u = 0; u < host_g.n; ++u) mapping[u] = u; // init with identity mapping

                solution = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "solution"), host_g.n);
    
                global_g_w = host_g.g_weight;

                recursive_bisection(host_g, level, pos, mapping);

                return;
            }


            void recursive_bisection(HostGraph &in_g, int level, std::vector<u32> &pos, std::vector<u32> &mapping) {
                
                Configuration internal_config; 

                internal_config = config;
                internal_config.k = 2;
                f64 adapt_imbalance = determine_adaptive_imbalance(
                    config.imbalance,
                    global_g_w,
                    config.k,
                    in_g.g_weight,
                    ((1U) << level),
                    static_cast<u64>(level)
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
                    
                    std::vector<u32> pos_left_graph, pos_right_graph;
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
                        const std::vector<u32>& mapping, const std::vector<u32>& pos) {
                            
                            // build global partition id from pos bits
                            partition_t global_partition_id = 0;
                            for (size_t i = 0; i < pos.size(); ++i) {
                                global_partition_id += pos[i] * ( 1 << (pos.size() - i) );
                            }
                        
                            for (vertex_t u = 0; u < local_graph.n; ++u) {
                                //! map local u -> original id 
                                //! mapping from lowest level to original graph
                                vertex_t original_id = mapping[u];

                                partition_t full_id = global_partition_id + local_partition(u);
                            
                                solution(original_id) = full_id;
                            }
                            return;
            }


           


            void create_subgraphs(HostPartition input_partition, HostGraph &input_graph,
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


    };

}


#endif