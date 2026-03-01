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
#include "../utility/custom_reductions.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "solver.h"



namespace GPU_HeiPa {

   


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
                

/*
                build_subgraph_Henning_GPU(
                    in_graph_device,
                    in_partition_device,
                    curr_mapping,
                    //curr_mapping_device,
                    left_mapping_host,
                    right_mapping_host,
                    left_graph_host,
                    right_graph_host,
                    mem_stack
                ) ;*/
                // convert back

                pop_front(mem_stack);

                free_graph(in_graph_device, mem_stack);

                pop_front(mem_stack); //Remove the in_partition_device

                std::cout << "came until here 10" << std::endl;
                return;
            }




            /*
            Ich probier mal die methode von Henning irgendwie anzupassen, weil die funktioniert ja...
            Vielleicht finde ich dann meinen Fehler
            */
            inline void build_subgraph_Henning_GPU(Graph &device_g,
                                                      UnmanagedDevicePartition &tmp_part,
                                                      const HostVertex curr_mapping,

                                                      HostVertex &left_mapping_host,
                                                      HostVertex &right_mapping_host,
                                                      
                                                      HostGraph &left_graph_host,
                                                      HostGraph &right_graph_host,
                                                      
                                                      KokkosMemoryStack &mem_stack)
                 {
                
                //UnmanagedDeviceVertex rename = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * input_graph.n), input_graph.n);
                // (".", curr_mapping.extent(0));
                UnmanagedDeviceVertex n_to_o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * curr_mapping.extent(0)), curr_mapping.extent(0));
                Kokkos::deep_copy(n_to_o, curr_mapping);
                Kokkos::fence() ;


                


                const partition_t k = 2;
                                                
            
                // 3) Non-leaf: build each child 
                for (partition_t id = 0; id < k; ++id) {
                    // --- First pass: compute sub_n, sub_m, sub_weight for this id
                    vertex_t sub_n = 0;
                    vertex_t sub_m = 0;
                    weight_t sub_weight = 0;
                
                    Kokkos::parallel_reduce("SubN", device_g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                        if (tmp_part(u) == id) lsum += 1;
                    }, sub_n);
                
                    Kokkos::parallel_reduce("SubWeight", device_g.n, KOKKOS_LAMBDA(const vertex_t u, weight_t &lsum) {
                        if (tmp_part(u) == id) lsum += device_g.weights(u);
                    }, sub_weight);
                
                    //? I wonder if it makes a difference if you count the edges exact or make an approximation
                    Kokkos::parallel_reduce("SubM", device_g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &lsum) {
                        if (tmp_part(u) == id) {
                            vertex_t cnt = 0;
                            for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                                const vertex_t v = device_g.edges_v(i);
                                if (tmp_part(v) == id) ++cnt;
                            }
                            lsum += cnt;
                        }
                    }, sub_m);
                
                    Kokkos::fence();
                
                    // Empty block => skip
                    if (sub_n == 0) {
                        continue;
                    }
                
                    // --- Allocate child graph + mappings
                    Graph child_g = make_graph(sub_n, sub_m, sub_weight, mem_stack);
                    UnmanagedDeviceVertex child_n_to_o = UnmanagedDeviceVertex((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * sub_n), sub_n);
                    UnmanagedDeviceVertex child_o_to_n = UnmanagedDeviceVertex((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * solution.extent(0)), solution.extent(0));
                
                    // --- Fill translation tables and weights
                    Kokkos::parallel_scan("AssignLocalIndex", device_g.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &prefix, const bool final) {
                        if (tmp_part(u) == id) {
                            const vertex_t my_idx = prefix;
                            if (final) {
                                const vertex_t old_u = n_to_o(u);
                                child_o_to_n(old_u) = my_idx;
                                child_n_to_o(my_idx) = old_u;
                                child_g.weights(my_idx) = device_g.weights(u);
                            }
                            prefix += 1;
                        }
                    });
                    Kokkos::fence();
                
                    // init neighborhood(0)
                    Kokkos::parallel_for("InitNeighborhood0", 1, KOKKOS_LAMBDA(const int) { child_g.neighborhood(0) = 0; });
                    Kokkos::fence();
                
                    // --- Fill edges + neighborhood offsets
                    Kokkos::parallel_scan("FillEdges", device_g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &edge_prefix, const bool final) {
                        if (tmp_part(u) == id) {
                            u32 start = edge_prefix;
                            u32 cnt = 0;
                        
                            for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                                const vertex_t v = device_g.edges_v(i);
                                if (tmp_part(v) == id) {
                                    if (final) {
                                        const vertex_t sub_v = child_o_to_n(n_to_o(v));
                                        child_g.edges_v(start) = sub_v;
                                        child_g.edges_w(start) = device_g.edges_w(i);
                                    }
                                    ++start;
                                    ++cnt;
                                }
                            }
                        
                            if (final) {
                                const vertex_t sub_u = child_o_to_n(n_to_o(u));
                                child_g.neighborhood(sub_u + 1) = edge_prefix + cnt;
                            }
                        
                            edge_prefix += cnt;
                        }
                    });
                    Kokkos::fence();
                
                    // fill the u array
                    Kokkos::parallel_for("fill_edges_u", child_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                        u32 begin = child_g.neighborhood(u);
                        u32 end = child_g.neighborhood(u + 1);
                        for (u32 i = begin; i < end; ++i) {
                            child_g.edges_u(i) = u;
                        }
                    });
                    Kokkos::fence();
                
                    if( id == 0) {
                        left_graph_host = to_host_graph(child_g);
                        left_mapping_host = HostVertex("host mapping left subgraph", child_g.n) ;
                        Kokkos::fence();
                        Kokkos::deep_copy(left_mapping_host, child_n_to_o);
                    } else{
                        right_graph_host = to_host_graph(child_g);
                        right_mapping_host = HostVertex("host mapping right subgraph", child_g.n) ;
                        Kokkos::fence();
                        Kokkos::deep_copy(right_mapping_host, child_n_to_o);
                    }

                    // We no longer need child_o_to_n after edges built
                    pop_back(mem_stack);
                
                    // --- Free child allocations (reverse of allocations for this child)
                    pop_front(mem_stack); // child_n_to_o
                    free_graph(child_g, mem_stack); // whatever make_graph allocated
                }
            
                
                
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
                
                Kokkos::parallel_scan("count neighbors", input_graph.n, KOKKOS_LAMBDA(const vertex_t u, Accumulators &acc, const bool final) {
                       
                        u32 cnt = 0;
                        partition_t my_partition = in_partition(u);

                        for (u32 i = input_graph.neighborhood(u); i < input_graph.neighborhood(u + 1); ++i) {
                            const vertex_t v = input_graph.edges_v(i);
                            if (in_partition(v) == my_partition) {
                                ++cnt;
                            }
                        }

                        if (in_partition(u) == 0) {
                            acc.partial_0s += cnt;
                            if (final) {
                                const vertex_t sub_u = rename(u);
                                left_graph.neighborhood( sub_u ) = acc.partial_0s ;

                            }
                        }else{ 
                            acc.partial_1s += cnt;
                            if (final) {
                                const vertex_t sub_u = rename(u);
                                right_graph.neighborhood( sub_u ) = acc.partial_1s ;
                            }
                        }

                        
                });
                Kokkos::fence();
                Kokkos::parallel_for("InitNeighborhoodEnd", 1, KOKKOS_LAMBDA(const int) {
                    left_graph.neighborhood(left_graph.n) = left_graph.neighborhood(left_graph.n - 1);
                    right_graph.neighborhood(right_graph.n) = right_graph.neighborhood(right_graph.n - 1)  ;
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