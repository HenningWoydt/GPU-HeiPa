#ifndef GPU_HEIPA_SOLVER_REC_BISEC_GPU_GOOD_H
#define GPU_HEIPA_SOLVER_REC_BISEC_GPU_GOOD_H

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
#include "../utility/custom_reductions.h"

namespace GPU_HeiPa {

    class SolverRecursiveBisectionGPUGood {


        public:
            Configuration config;
            weight_t global_g_w;
            HostPartition solution;

            explicit SolverRecursiveBisectionGPUGood(Configuration t_config) : config(std::move(t_config)) {
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

                //TODO: configure this size
                KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(
                30 * 20 * (size_t) host_g.n * sizeof(vertex_t) + // 20% buffer for vertices
                10 * 20 * (size_t) host_g.m * sizeof(vertex_t), // Graph + coarsening overhead
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

                HostPartition in_partition = Solver(internal_config).solve(in_g);
                Kokkos::fence();


                if(level == 1) {
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
                           // std::cout << "mapping size: " << msize << std::endl;

                            for (vertex_t u = 0; u < local_graph.n; ++u) {
                                // if(u >= msize)
                                //     std::cout << "something wrong " <<std::endl;

                               // std::cout << u <<std::endl;
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
                

                pop_front(mem_stack);

                free_graph(in_graph_device, mem_stack);

                pop_front(mem_stack); //Remove the in_partition_device

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

                // step 1: count n,m,w for the subgraphs

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

                bigAccumulator edges_and_weights;

                Kokkos::parallel_reduce("get num edges and graph weight",
                    input_graph.n,
                    KOKKOS_LAMBDA( u32 u, bigAccumulator &acc) {

                        partition_t partition = in_partition(u);
                        vertex_t cnt = 0;

                        for(int i = input_graph.neighborhood(u); i < input_graph.neighborhood(u+1); ++i) {
                            const vertex_t v = input_graph.edges_v(i);
                            if(in_partition(v) == partition) ++cnt;
                        }                        


                        if(partition == 0) {
                            acc.num_edges_0s += cnt;
                            acc.weight_0s += input_graph.weights(u);
                        }else{
                            acc.num_edges_1s += cnt;
                            acc.weight_1s += input_graph.weights(u);
                        }

                    }, edges_and_weights
                );

                Kokkos::fence(); // wait for kernel completion

            
                // step 2: allocate and make the subgraphs

                Graph left_graph = make_graph( res_num_vertices.partial_0s , edges_and_weights.num_edges_0s , edges_and_weights.weight_0s  , mem_stack );
                Graph right_graph = make_graph( res_num_vertices.partial_1s , edges_and_weights.num_edges_1s , edges_and_weights.weight_1s, mem_stack );
                
                DeviceVertex left_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * left_graph.n ), left_graph.n ) ;
                DeviceVertex right_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * right_graph.n ), right_graph.n ) ;
                
                left_mapping_host = HostVertex("host mapping left subgraph", left_graph.n) ;
                right_mapping_host = HostVertex("host mapping right subgraph", right_graph.n) ;
                Kokkos::fence();
               
                // init neighborhood(0)
                Kokkos::parallel_for("InitNeighborhood0", 1, KOKKOS_LAMBDA(const int) { 
                        left_graph.neighborhood(0) = 0; 
                        right_graph.neighborhood(0) = 0; 
                    });   
                Kokkos::fence();

                //TODO: check if something is missing in between
                
                // step 3: fill edges + offsets

                Kokkos::parallel_scan("FillEdges", input_graph.n,
                    KOKKOS_LAMBDA(const vertex_t u, Accumulators &acc, const bool final) {

                        u32 cnt = 0;

                        if (in_partition(u) == 0) {
                            u32 start = acc.partial_0s;

                            for (u32 i = input_graph.neighborhood(u); i < input_graph.neighborhood(u + 1); ++i) {
                                const vertex_t v = input_graph.edges_v(i);
                                if (in_partition(v) == 0) {
                                    if (final) {
                                        const vertex_t sub_v = rename(v);
                                        left_graph.edges_v(start) = sub_v;
                                        left_graph.edges_w(start) = input_graph.edges_w(i);
                                    }
                                    ++start;
                                    ++cnt;
                                }
                            }
                        
                            if (final) {

                                left_graph.weights( rename(u) ) = input_graph.weights(u);
                                left_mapping_device( rename(u) ) = curr_mapping(u);                        

                                const vertex_t sub_u = rename(u);
                                left_graph.neighborhood(sub_u + 1) = acc.partial_0s + cnt;
                            }
                        
                            acc.partial_0s += cnt;
                        } else{
                            u32 start = acc.partial_1s;

                            for (u32 i = input_graph.neighborhood(u); i < input_graph.neighborhood(u + 1); ++i) {
                                const vertex_t v = input_graph.edges_v(i);
                                if (in_partition(v) == 1) {
                                    if (final) {
                                        const vertex_t sub_v = rename(v);
                                        right_graph.edges_v(start) = sub_v;
                                        right_graph.edges_w(start) = input_graph.edges_w(i);
                                    }
                                    ++start;
                                    ++cnt;
                                }
                            }
                        
                            if (final) {

                                right_graph.weights( rename(u) ) = input_graph.weights(u);
                                right_mapping_device( rename(u) ) = curr_mapping(u);                        

                                const vertex_t sub_u = rename(u);
                                right_graph.neighborhood(sub_u + 1) = acc.partial_1s + cnt;
                            }
                        
                            acc.partial_1s += cnt;
                        }

                    }
                );
                Kokkos::fence();

                Kokkos::deep_copy(left_mapping_host, left_mapping_device);
                Kokkos::deep_copy(right_mapping_host, right_mapping_device);

                left_graph_host = to_host_graph(left_graph);
                right_graph_host = to_host_graph(right_graph);

                Kokkos::fence();

                pop_front(mem_stack); // rm right_mapping_device
                pop_front(mem_stack); // rm left_mapping_device

                free_graph(left_graph, mem_stack);
                free_graph(right_graph, mem_stack);

                pop_front(mem_stack); //remove rename
                

                return;
            }

        
        
    };

}


#endif