#ifndef GPU_HEIPA_SOLVER_REC_BISEC_H
#define GPU_HEIPA_SOLVER_REC_BISEC_H

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
#include "solver.h"


namespace GPU_HeiPa {

    class SolverRecursiveBisection {


        public:
            Configuration config; 
            //? what does this hold?
            //? -> basically all the config info that you can supply to the algo upon calling the executable

            // constructor:
            explicit SolverRecursiveBisection(Configuration t_config) : config(std::move(t_config)) {
        }

            HostPartition solve(HostGraph &host_g) {
                
                //other stuff...
                
                internal_solve(host_g);
            
                //other stuff...

                HostPartition host_partition; 
                //? how does this look like? what do i actually need to return?
                //? -> is just a an array of u32 (but defined as a kokkos view)

                // retrieve the actual host partition from the solver

                return host_partition;
            }


        private:

            void internal_solve(HostGraph &host_g) {
                // do recursive bisection on the inputgraph

                //Step 1: Just partition once:
                
                Configuration conf1 = config;
                conf1.k = 2;
                conf1.verbose_level = 2;

                HostPartition first_partition = Solver(conf1).solve(host_g);
            
                HostGraph left_graph, right_graph;
                create_subgraph(first_partition, host_g, left_graph, right_graph);

                HostPartition left_par = Solver(conf1).solve(left_graph);
                HostPartition right_par = Solver(conf1).solve(right_graph);

            }


            /**
             * Returns two subgraphs:
             *  One for all nodes with partition ID 0
             *  One for all nodes with partition ID 1
             *
             *  First make a boring serial version, 
             *  later parallelize (openMP ? GPU ?)
             * 
             *  Important question: How do i convert between input and output graphs?
             *  I.e. how do i get the partition for the original graph from
             *  my smaller subgraphs?
             *  -> Save the mapping somewhere?
             * 
             * @param TODO
             * @return TODO
             */
            void create_subgraph(HostPartition input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph, HostGraph &right_graph
            ) {

                vertex_t num_vertices[2], num_edges[2];
                weight_t weights[2];
                num_vertices[0] = num_vertices[1] = num_edges[0] = num_edges[1] = 0;
                weights[0] = weights[1] = 0;

                //? Is this byte-allocation correct? Henning does the same in the host_graph.h ...
                HostU32 rename = HostU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_renaming"), input_graph.n * sizeof(u32) ) ;

                
                // Get the initial information to create the two subgraphs:
                partition_t partition;
                for(vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition = input_partition(u);

                    rename[u] = num_vertices[partition];
                    num_vertices[partition]++ ;
                    weights[partition] += input_graph.weights(u);
                    num_edges[partition] += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u); // fast upper bound
                }
                
                // Print initialization values for subgraphs
                std::cout << "Left subgraph:  vertices=" << num_vertices[0] 
                          << " edges=" << num_edges[0] << " weight=" << weights[0] << std::endl;
                std::cout << "Right subgraph: vertices=" << num_vertices[1] 
                          << " edges=" << num_edges[1] << " weight=" << weights[1] << std::endl;
                
                
                // init the two HostGraphs
                allocate_memory(left_graph, num_vertices[0], num_edges[0], weights[0]);
                allocate_memory(right_graph, num_vertices[1], num_edges[1], weights[1]);


                //TODO: Assign values to l_graph and r_graph (create the subgraphs)

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
                left_graph.neighborhood(left_graph.n) = 0;
                for(vertex_t u = 1; u < left_graph.n + 1; ++u)
                    left_graph.neighborhood(u) += left_graph.neighborhood(u-1);
                
                right_graph.neighborhood(right_graph.n) = 0;
                for(vertex_t u = 1; u < right_graph.n + 1; ++u)
                    right_graph.neighborhood(u) += right_graph.neighborhood(u-1);

                
                vertex_t idx_l, idx_r;
                //! This is again a parallel for (independent iterations)
                for( vertex_t u = 0; u < input_graph.n ; ++u) {
                    partition_t my_part = input_partition(u);

                    if( my_part == 0) {
                        left_graph.weights( rename(u) ) = input_graph.weights(u);
                    } else{
                        right_graph.weights( rename(u)) = input_graph.weights(u);
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
        
        
    };

}


#endif