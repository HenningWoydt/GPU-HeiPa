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

    struct WeightAccumulators {
    weight_t partial_0s = 0;
    weight_t partial_1s = 0;
    
    KOKKOS_INLINE_FUNCTION
    void operator+=(const WeightAccumulators& rhs) {
        partial_0s += rhs.partial_0s;
        partial_1s += rhs.partial_1s;
    }
    };

    // ...existing code...

inline void print_host_graph(const HostGraph& g, const std::string& name) {
    std::cout << "Graph: " << name << "\n";
    std::cout << "n = " << g.n << ", m = " << g.m << ", g_weight = " << g.g_weight << "\n";
    std::cout << "weights: ";
    for (vertex_t u = 0; u < std::min((vertex_t)30, g.n); ++u) std::cout << g.weights(u) << " ";
    std::cout << "\nneighborhood: ";
    for (vertex_t u = 0; u < std::min((vertex_t)30, (vertex_t)(g.n + 1)); ++u) std::cout << g.neighborhood(u) << " ";
    std::cout << "\nedges_v: ";
    for (vertex_t e = 0; e < std::min((vertex_t)30, g.m); ++e) std::cout << g.edges_v(e) << " ";
    std::cout << "\nedges_w: ";
    for (vertex_t e = 0; e < std::min((vertex_t)30, g.m); ++e) std::cout << g.edges_w(e) << " ";
    std::cout << "\n";
}

inline void print_host_vertex(const HostVertex& v, const std::string& name) {
    std::cout << "Mapping: " << name << "\n";
    for (vertex_t i = 0; i < std::min((vertex_t)30, (vertex_t)v.extent(0)); ++i) std::cout << v(i) << " ";
    std::cout << "\n";
}
// ...existing code...


    class SolverRecursiveBisectionGPU {


        public:
            Configuration config;
            weight_t global_g_w;
            HostPartition solution;

            explicit SolverRecursiveBisectionGPU(Configuration t_config) : config(std::move(t_config)) {
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
                HostVertex mapping("mapping", host_g.n); //TODO: this later be on the GPU 
                            
                for (vertex_t u = 0; u < host_g.n; ++u) {
                    mapping(u) = u;
                }
                solution = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "solution"), host_g.n);
    
                global_g_w = host_g.g_weight;

                KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(
                30 * (size_t) host_g.n * sizeof(vertex_t) + // 20% buffer for vertices
                10 * (size_t) host_g.m * sizeof(vertex_t), // Graph + coarsening overhead
                "Jacobs internal stack"
                );
                Kokkos::fence();


                recursive_bisection(host_g, level, pos, mapping, mem_stack);

                destroy(mem_stack);

                return;
            }


            void recursive_bisection(HostGraph &in_g, int level, std::vector<int> &pos, HostVertex &mapping, KokkosMemoryStack &mem_stack) {
                
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

                // HostPartition in_partition = Solver(internal_config).solve(in_g);
                // Kokkos::fence();

                HostPartition in_partition("lol", in_g.n) ;
                Kokkos::fence();
                for(int i = 0; i < in_g.n; ++i) {
                    in_partition(i) = i % 2;
                }

                if(level == 1) {
                    std::cout << "came until popagate step" << std::endl;

                    ScopedTimer _t("recursive_bisection", "recursive_bisection", "propagate_solution");
                    propagate_solution(in_partition, in_g, mapping, pos);
                    return;

                } else{
                    ScopedTimer _t("recursive_bisection", "recursive_bisection", "create_subgraph_like_Henning");
                    
                    HostGraph left_graph_host, right_graph_host;
                    HostVertex left_mapping_host ; // the mappings between new and old vertex IDs
                    HostVertex right_mapping_host ; 
                    create_subgraph_GPU_wrapper(
                        in_partition, in_g,
                        left_graph_host, right_graph_host,
                        left_mapping_host,
                        right_mapping_host,
                        mapping,
                        mem_stack
                    );
                    

                    //! print the graphs and mappings
                    // print_host_graph(left_graph_host, "left_graph_host");
                    // print_host_graph(right_graph_host, "right_graph_host");
                    // print_host_vertex(left_mapping_host, "left_new_to_old");
                    // print_host_vertex(right_mapping_host, "right_new_to_old");

                    std::vector<int> pos_left_graph, pos_right_graph;
                    pos_left_graph = pos_right_graph = pos;  // copy wont hurt because this is super small (like 10 entries)
                    pos_left_graph.push_back(0);
                    pos_right_graph.push_back(1);

                    _t.stop();

                    recursive_bisection(left_graph_host, level-1, pos_left_graph, left_mapping_host, mem_stack); // go down to the next lower level
                    recursive_bisection(right_graph_host, level-1, pos_right_graph, right_mapping_host, mem_stack);
                    return;
                }
                /**/
            }


            /**
             * This function takes a subgraph on the last level of the recursive bisection
             * and translates (propagates) the partition found on this graph, to a 
             * partition on the input graph
             * 
             * ! there seems to be a bug here, maybe because of mapping?
             */
            void propagate_solution(const HostPartition& local_partition, const HostGraph& local_graph,
                        const HostVertex& mapping, const std::vector<int>& pos) {
                            
                            // build global partition id from pos bits
                            partition_t global_partition_id = 0;
                            for (size_t i = 0; i < pos.size(); ++i) {
                                //global_partition_id += pos[i] * pow(2, pos.size() - i);
                                global_partition_id += pos[i] * ( 1 << (pos.size() - i) );
                            }
                        
                            int msize = mapping.extent(0);
                            std::cout << "mapping size: " << msize << std::endl;

                            for (vertex_t u = 0; u < local_graph.n; ++u) {
                                if(u >= msize)
                                    std::cout << "something wrong " <<std::endl;

                                std::cout << u <<std::endl;
                                vertex_t original_id = mapping(u);

                                partition_t full_id = global_partition_id + local_partition(u);
                            
                                solution(original_id) = full_id;
                            }
                            return;
            }


        
           
            void create_subgraph_GPU_wrapper(HostPartition &input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph_host, HostGraph &right_graph_host,
                                 HostVertex &left_mapping_host,
                                 HostVertex &right_mapping_host,
                                 HostVertex &curr_mapping,
                                 KokkosMemoryStack &mem_stack
            ) {
                // convert data from CPU to GPU
                
                UnmanagedDevicePartition in_partition_device = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * input_graph.n), input_graph.n);
                Kokkos::fence();

                Kokkos::deep_copy(in_partition_device, input_partition) ; //? legal conversion ?
                Kokkos::fence();

                f64 upload;
                Graph in_graph_device = from_HostGraph(input_graph, mem_stack, upload ) ;
                Kokkos::fence();

                DeviceVertex curr_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * input_graph.n ), input_graph.n ) ;
                Kokkos::fence();
                Kokkos::deep_copy(curr_mapping_device, curr_mapping);
                Kokkos::fence();
                

                create_subgraph_GPU_vertexParallel(
                    in_graph_device,
                    in_partition_device,
                    mem_stack,
                    curr_mapping_device,
                    left_mapping_host,
                    right_mapping_host,
                    left_graph_host,
                    right_graph_host
                ) ;

                // convert back

                pop_front(mem_stack);

                free_graph(in_graph_device, mem_stack);

                pop_front(mem_stack); //Remove the in_partition_device

                std::cout << "came until here 10" << std::endl;
                return;
            }



            /*
            
            Its not quite clear if parallelizing over vertices or over the edges is better here.
            So i will try it out both versions and empirically find out which is better!
            
            */
            void create_subgraph_GPU_vertexParallel(
                Graph &input_graph,
                UnmanagedDevicePartition &in_partition,
                KokkosMemoryStack &mem_stack,
                DeviceVertex &curr_mapping,
                HostVertex &left_mapping_host,
                HostVertex &right_mapping_host,
                HostGraph &left_graph_host,
                HostGraph &right_graph_host
            ) {

                UnmanagedDeviceVertex rename = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * input_graph.n), input_graph.n);
                
                Accumulators res_num_vertices;
                res_num_vertices.partial_0s = 0;
                res_num_vertices.partial_1s = 0;

                // init the rename view
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
                }, res_num_vertices
                );

                //TODO: Combine the loops into a single on
                WeightAccumulators res_weights;

                res_weights.partial_0s = 0;
                res_weights.partial_1s = 0;

                Kokkos::parallel_reduce("reduction over the weights", input_graph.n, 
                    KOKKOS_LAMBDA( u32 u, WeightAccumulators &acc) {
                        partition_t partition = in_partition(u);
                    
                        if(partition == 0) acc.partial_0s += input_graph.weights(u);
                        else acc.partial_1s += input_graph.weights(u);
                    
                }, res_weights);


                Accumulators res_neighbors;
                res_neighbors.partial_0s = 0;
                res_neighbors.partial_1s = 0;
        
                Kokkos::parallel_reduce("reduction over the neighbors", input_graph.n,
                    KOKKOS_LAMBDA( u32 u, Accumulators &acc) {
                        partition_t partition = in_partition(u);
                    
                        if(partition == 0) acc.partial_0s += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u);
                        else acc.partial_1s += input_graph.neighborhood( u+1 ) - input_graph.neighborhood(u);
                    }, res_neighbors);
                
                
                Kokkos::fence();
               

                Graph left_graph = make_graph( res_num_vertices.partial_0s , res_neighbors.partial_0s , res_weights.partial_0s, mem_stack );
                Graph right_graph = make_graph( res_num_vertices.partial_1s , res_neighbors.partial_1s , res_weights.partial_1s, mem_stack );
                Kokkos::fence();

                

                
                Kokkos::parallel_for("count active edges", input_graph.n,
                    KOKKOS_LAMBDA(vertex_t u) {
                        partition_t my_part = in_partition(u);

                        if(my_part == 0) {
                            left_graph.neighborhood( rename(u) ) = 0;
                        } else{
                            right_graph.neighborhood( rename(u) ) = 0;
                        }

                        for( u32 i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i ) {
                            vertex_t v = input_graph.edges_v(i);
                            partition_t neighbor_part = in_partition(v);

                            if( my_part == neighbor_part) {

                                if( my_part == 0) {
                                    left_graph.neighborhood( rename(u) ) += 1;
                                } else {
                                    right_graph.neighborhood( rename(u)) += 1;
                                }


                            }
                        }
                    }
                );


                Kokkos::parallel_for("InitNeighborhoodEnd", 1, KOKKOS_LAMBDA(const int) {
                    left_graph.neighborhood(left_graph.n) = 0;
                    right_graph.neighborhood(right_graph.n) = 0;
                });
                Kokkos::fence();

                
                Kokkos::parallel_scan("create neighborhood left graph", left_graph.n +1,
                    KOKKOS_LAMBDA( vertex_t u, vertex_t &temp, bool final ) {
                        u32 val = left_graph.neighborhood(u);
                        temp += val;
                        if(final) {
                            left_graph.neighborhood(u) = temp;
                        }
                        

                    }
                );
                

                Kokkos::parallel_scan("create neighborhood right graph", right_graph.n + 1, 
                    KOKKOS_LAMBDA (const vertex_t u, u32& temp, const bool final) {
                        u32 val = right_graph.neighborhood(u);
                        temp += val;
                        if (final) {
                            right_graph.neighborhood(u) = temp;
                        }
                        
                });
                Kokkos::fence();
                DeviceVertex left_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * left_graph.n ), left_graph.n ) ;
                DeviceVertex right_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * right_graph.n ), right_graph.n ) ;

                left_mapping_host = HostVertex("host mapping left subgraph", left_graph.n) ;
                right_mapping_host = HostVertex("host mapping right subgraph", right_graph.n) ;
                Kokkos::fence();

                Kokkos::parallel_for("write edges to subgraphs", input_graph.n,
                    KOKKOS_LAMBDA( vertex_t u ) 
                    {

                        partition_t my_part = in_partition(u);

                        if( my_part == 0) {
                            
                            left_graph.weights( rename(u) ) = input_graph.weights(u);
                            left_mapping_device( rename(u) ) = curr_mapping(u);

                        }else{
                            
                            right_graph.weights( rename(u) ) = input_graph.weights(u);
                            right_mapping_device( rename(u) ) = curr_mapping(u);

                        }
                        
                        
                        for( u32 i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                            vertex_t v = input_graph.edges_v(i);
                            partition_t neighbor_part = in_partition(v);

                            if(my_part == neighbor_part) {
                            
                                if(my_part == 0) {
                                    

                                    u32 idx = --left_graph.neighborhood(rename(u));
                                    left_graph.edges_v(idx) = rename(v);
                                    left_graph.edges_w(idx) = input_graph.edges_w(i);

                                }else{

                                    u32 idx = --right_graph.neighborhood(rename(u));
                                    right_graph.edges_v(idx) = rename(v);
                                    right_graph.edges_w(idx) = input_graph.edges_w(i);

                                }
                            
                                
                            }
                        }
                    }
                );
                Kokkos::fence();

                Kokkos::deep_copy(left_mapping_host, left_mapping_device);
                Kokkos::deep_copy(right_mapping_host, right_mapping_device);

                pop_front(mem_stack); // rm right_mapping_device
                pop_front(mem_stack); // rm left_mapping_device

                left_graph_host = to_host_graph(left_graph);
                right_graph_host = to_host_graph(right_graph);

                free_graph(left_graph, mem_stack);
                free_graph(right_graph, mem_stack);

                pop_front(mem_stack); //remove rename
                

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