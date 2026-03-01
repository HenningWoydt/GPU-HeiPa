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
                
                HostPartition solution = internal_solve(host_g);

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
            HostPartition internal_solve(HostGraph &host_g) {


                //TODO: configure this size
                KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(
                30 * 10 * (size_t) host_g.n * sizeof(vertex_t) + // 20% buffer for vertices
                10 * 10 * (size_t) host_g.m * sizeof(vertex_t), // Graph + coarsening overhead
                "Jacobs internal stack"
                );

                KokkosMemoryStack mem_stack_tmp = initialize_kokkos_memory_stack(
                30 * 2 * (size_t) host_g.n * sizeof(vertex_t) + // 20% buffer for vertices
                10 * 2 * (size_t) host_g.m * sizeof(vertex_t), // Graph + coarsening overhead
                "Stack for migrating my code"
                );
                Kokkos::fence();

                std::vector<int> pos = {};
                int level = (int)std::log2(config.k);

                UnmanagedDeviceVertex mapping = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack_tmp, sizeof(vertex_t) * host_g.n), host_g.n);
                
                Kokkos::parallel_for("fill mapping", host_g.n,
                    KOKKOS_LAMBDA(vertex_t u) {
                        mapping(u) = u;
                    }
                );
                
                global_g_w = host_g.g_weight;

                //TODO: need a gpu copy of this too
                HostPartition solution = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "solution"), host_g.n);
                UnmanagedDevicePartition solution_device = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * host_g.n), host_g.n);
                

                recursive_bisection(host_g, level, pos, mapping, mem_stack, mem_stack_tmp, solution_device);

                Kokkos::deep_copy(solution, solution_device);
                Kokkos::fence();

                pop_front(mem_stack_tmp); // rm mapping
                
                pop_front(mem_stack); // rm solution_device

                destroy(mem_stack);
                destroy(mem_stack_tmp);

                return solution;
            }


            void recursive_bisection(HostGraph &in_g, int level, std::vector<int> &pos, 
                UnmanagedDeviceVertex &mapping,
                KokkosMemoryStack &mem_stack,
                KokkosMemoryStack &mem_stack_tmp,
                UnmanagedDevicePartition &solution_d
            ) {
                
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
                    // HostVertex mapping_h("mh", mapping.extent(0));
                    // Kokkos::deep_copy(mapping_h, mapping);
                    // Kokkos::fence() ;

                    UnmanagedDevicePartition device_part = UnmanagedDevicePartition((partition_t *) get_chunk_front(mem_stack, sizeof(partition_t) * in_g.n), in_g.n);
                    
                    Kokkos::deep_copy(device_part, in_partition);
                    Kokkos::fence() ;

                    propagate_solution(device_part, in_g, mapping, pos, solution_d);
                    pop_front(mem_stack);
                    return;

                } else{
                    ScopedTimer _t("recursive_bisection", "recursive_bisection", "create_subgraph_like_Henning");
                    
                    HostGraph left_graph_host, right_graph_host;

                    
                    UnmanagedDeviceVertex left_mapping ; // the mappings between new and old vertex IDs
                    UnmanagedDeviceVertex right_mapping ; 
                    create_subgraph_GPU_wrapper(
                        in_partition, in_g,
                        left_graph_host, right_graph_host,
                        left_mapping,
                        right_mapping,
                        mapping,
                        mem_stack,
                        mem_stack_tmp
                    );
                    
                    std::vector<int> pos_left_graph, pos_right_graph;
                    pos_left_graph = pos_right_graph = pos;  
                    pos_left_graph.push_back(0);
                    pos_right_graph.push_back(1);

                    _t.stop();

                    recursive_bisection(left_graph_host, level-1, pos_left_graph, left_mapping, mem_stack, mem_stack_tmp, solution_d); // go down to the next lower level
                    pop_front(mem_stack_tmp);
                    
                    recursive_bisection(right_graph_host, level-1, pos_right_graph, right_mapping, mem_stack, mem_stack_tmp, solution_d);
                    pop_front(mem_stack_tmp);

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
            void propagate_solution(
                        //const HostPartition& local_partition,
                        const UnmanagedDevicePartition &local_partition,
                        const HostGraph& local_graph,
                        const UnmanagedDeviceVertex &mapping,
                        const std::vector<int>& pos,
                        UnmanagedDevicePartition &solution
                    ) {
                            
                            partition_t global_partition_id = 0;
                            for (size_t i = 0; i < pos.size(); ++i) {
                                //global_partition_id += pos[i] * pow(2, pos.size() - i);
                                global_partition_id += pos[i] * ( 1 << (pos.size() - i) );
                            }
                            
                            Kokkos::parallel_for("write final partition", local_graph.n,
                                KOKKOS_LAMBDA( vertex_t u) {
                                    vertex_t original_id = mapping(u);

                                    partition_t full_id = global_partition_id + local_partition(u); 
                            
                                    solution(original_id) = full_id; 
                                }
                            );

                            return;
            }


        
           
            void create_subgraph_GPU_wrapper(HostPartition &input_partition, HostGraph &input_graph,
                                 HostGraph &left_graph_host, HostGraph &right_graph_host,
                                 UnmanagedDeviceVertex &left_mapping,
                                 UnmanagedDeviceVertex &right_mapping,
                                 UnmanagedDeviceVertex &curr_mapping,
                                 KokkosMemoryStack &mem_stack,
                                 KokkosMemoryStack &mem_stack_tmp
            ) {
                // convert data from CPU to GPU
                
                UnmanagedDevicePartition in_partition_device = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * input_graph.n), input_graph.n);
                Kokkos::fence();

                Kokkos::deep_copy(in_partition_device, input_partition) ; //? legal conversion ?
                Kokkos::fence();

                f64 upload;
                Graph in_graph_device = from_HostGraph(input_graph, mem_stack, upload ) ;
                Kokkos::fence();


                // TODO: can remove this if mapping is already on the device
                // DeviceVertex curr_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * input_graph.n ), input_graph.n ) ;
                // Kokkos::fence();
                // Kokkos::deep_copy(curr_mapping_device, curr_mapping);
                // Kokkos::fence();
                
                
                
                create_subgraph_GPU_vertexParallel(
                    in_graph_device,
                    in_partition_device,
                    mem_stack,
                    mem_stack_tmp,
                    
                    curr_mapping,
                    left_mapping,
                    right_mapping,
                    
                    left_graph_host,
                    right_graph_host
                ) ;
                

                // pop_front(mem_stack);

                free_graph(in_graph_device, mem_stack);

                pop_back(mem_stack); //Remove the in_partition_device

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
                KokkosMemoryStack &mem_stack_tmp,
                UnmanagedDeviceVertex &curr_mapping, 
                UnmanagedDeviceVertex &left_mapping_device,
                UnmanagedDeviceVertex &right_mapping_device,
                HostGraph &left_graph_host,
                HostGraph &right_graph_host
            ) {

                UnmanagedDeviceVertex rename = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * input_graph.n), input_graph.n);

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
                
                right_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack_tmp, sizeof(vertex_t) * res_num_vertices.partial_1s ), res_num_vertices.partial_1s ) ;
                left_mapping_device = DeviceVertex( (vertex_t *) get_chunk_front(mem_stack_tmp, sizeof(vertex_t) * res_num_vertices.partial_0s ), res_num_vertices.partial_0s ) ;
                
                Graph right_graph = make_graph( res_num_vertices.partial_1s , edges_and_weights.num_edges_1s , edges_and_weights.weight_1s, mem_stack );
                Graph left_graph = make_graph( res_num_vertices.partial_0s , edges_and_weights.num_edges_0s , edges_and_weights.weight_0s  , mem_stack );
                
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

                //TODO: remove
                //Kokkos::deep_copy(left_mapping_host, left_mapping_device);
                //Kokkos::deep_copy(right_mapping_host, right_mapping_device);

                left_graph_host = to_host_graph(left_graph);
                right_graph_host = to_host_graph(right_graph);

                Kokkos::fence();

                //TODO: dont pop here!
                // pop_front(mem_stack); // rm right_mapping_device
                // pop_front(mem_stack); // rm left_mapping_device

                free_graph(left_graph, mem_stack);
                free_graph(right_graph, mem_stack);

                pop_back(mem_stack); //remove rename
                

                return;
            }

        
        
    };

}


#endif