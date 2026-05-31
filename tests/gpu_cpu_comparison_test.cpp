#include <gtest/gtest.h>
#include "initial_partitioning/gpu_bisection_partition.h"
#include "datastructures/host_graph.h"
#include <vector>
#include <algorithm>
#include <filesystem>
#include <map>

using namespace GPU_HeiPa;

// CPU version of subgraph extraction for comparison
void extract_all_subgraphs_cpu(const HostGraph &hg,
                              const HostPartition &h_map,
                              partition_t k,
                              std::vector<HostGraph> &subgraphs) {
    subgraphs.resize(k);
    std::vector<vertex_t> global_to_local(hg.n);
    std::vector<vertex_t> block_n(k, 0);

    // 1. Assign local IDs
    for (vertex_t u = 0; u < hg.n; ++u) {
        partition_t id = h_map(u);
        global_to_local[u] = block_n[id]++;
    }

    for (partition_t id = 0; id < k; ++id) {
        subgraphs[id].uniform_vertex_weights = hg.uniform_vertex_weights;
        subgraphs[id].uniform_edge_weights = hg.uniform_edge_weights;
        
        // First pass: count edges
        vertex_t m_count = 0;
        weight_t total_weight = 0;
        for (vertex_t u = 0; u < hg.n; ++u) {
            if (h_map(u) != id) continue;
            total_weight += hg.uniform_vertex_weights ? 1 : hg.weights(u);
            for (u32 e = hg.neighborhood(u); e < hg.neighborhood(u + 1); ++e) {
                if (h_map(hg.edges_v(e)) == id) {
                    m_count++;
                }
            }
        }

        allocate_memory(subgraphs[id], block_n[id], m_count, total_weight);

        // Second pass: fill CSR
        vertex_t current_local_u = 0;
        vertex_t current_m = 0;
        subgraphs[id].neighborhood(0) = 0;
        for (vertex_t u = 0; u < hg.n; ++u) {
            if (h_map(u) != id) continue;
            
            if (!hg.uniform_vertex_weights) {
                subgraphs[id].weights(current_local_u) = hg.weights(u);
            }

            for (u32 e = hg.neighborhood(u); e < hg.neighborhood(u + 1); ++e) {
                vertex_t v = hg.edges_v(e);
                if (h_map(v) == id) {
                    subgraphs[id].edges_v(current_m) = global_to_local[v];
                    if (!hg.uniform_edge_weights) {
                        subgraphs[id].edges_w(current_m) = hg.edges_w(e);
                    }
                    current_m++;
                }
            }
            subgraphs[id].neighborhood(current_local_u + 1) = current_m;
            current_local_u++;
        }
    }
}

class GpuCpuComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    std::string find_graph(const std::string& name) {
        std::vector<std::string> search_paths = {"data/" + name, "../data/" + name, "../../data/" + name};
        for (const auto& p : search_paths) {
            if (file_exists(p)) return p;
        }
        return "";
    }

    // Compare two subgraphs (one from CPU, one from GPU)
    // Since vertex ordering might differ, we use the global IDs to establish a mapping
    void compare_extracted_subgraphs(const HostGraph &cpu_g, 
                                    const HostPartition &h_map, // global partition map
                                    const std::vector<vertex_t> &cpu_local_to_global, // established by CPU extraction order
                                    const Graph &gpu_g, 
                                    const UnmanagedDeviceVertex &gpu_g_ids, // global IDs for GPU subgraph vertices
                                    DeviceExecutionSpace &exec_space) {
        
        ASSERT_EQ(cpu_g.n, gpu_g.n);
        ASSERT_EQ(cpu_g.m, gpu_g.m);
        EXPECT_EQ(cpu_g.g_weight, gpu_g.g_weight);

        // Map global ID -> local ID for GPU subgraph
        HostVertex h_gpu_g_ids("h_gpu_g_ids", gpu_g.n);
        Kokkos::deep_copy(exec_space, h_gpu_g_ids, gpu_g_ids);
        exec_space.fence();
        
        std::map<vertex_t, vertex_t> global_to_gpu_local;
        for (vertex_t i = 0; i < gpu_g.n; ++i) {
            global_to_gpu_local[h_gpu_g_ids(i)] = i;
        }

        // Map global ID -> local ID for CPU subgraph
        std::map<vertex_t, vertex_t> global_to_cpu_local;
        for (vertex_t i = 0; i < cpu_g.n; ++i) {
            global_to_cpu_local[cpu_local_to_global[i]] = i;
        }

        // Verify that they both contain the same global vertices
        ASSERT_EQ(global_to_gpu_local.size(), global_to_cpu_local.size());
        for (auto const& [gid, local_id] : global_to_cpu_local) {
            ASSERT_TRUE(global_to_gpu_local.count(gid)) << "Global vertex " << gid << " missing in GPU subgraph";
        }

        // Verify edges for each vertex
        HostU32 h_gpu_neighborhood("h_gpu_neighborhood", gpu_g.n + 1);
        Kokkos::deep_copy(exec_space, h_gpu_neighborhood, gpu_g.neighborhood);
        HostVertex h_gpu_edges_v("h_gpu_edges_v", gpu_g.m);
        Kokkos::deep_copy(exec_space, h_gpu_edges_v, gpu_g.edges_v);
        exec_space.fence();

        for (auto const& [gid, cpu_l] : global_to_cpu_local) {
            vertex_t gpu_l = global_to_gpu_local[gid];

            // Get CPU neighbors (as global IDs)
            std::vector<vertex_t> cpu_neighbors;
            for (u32 e = cpu_g.neighborhood(cpu_l); e < cpu_g.neighborhood(cpu_l + 1); ++e) {
                cpu_neighbors.push_back(cpu_local_to_global[cpu_g.edges_v(e)]);
            }
            std::sort(cpu_neighbors.begin(), cpu_neighbors.end());

            // Get GPU neighbors (as global IDs)
            std::vector<vertex_t> gpu_neighbors;
            for (u32 e = h_gpu_neighborhood(gpu_l); e < h_gpu_neighborhood(gpu_l + 1); ++e) {
                gpu_neighbors.push_back(h_gpu_g_ids(h_gpu_edges_v(e)));
            }
            std::sort(gpu_neighbors.begin(), gpu_neighbors.end());

            ASSERT_EQ(cpu_neighbors.size(), gpu_neighbors.size()) << "Degree mismatch for global vertex " << gid;
            for (size_t i = 0; i < cpu_neighbors.size(); ++i) {
                ASSERT_EQ(cpu_neighbors[i], gpu_neighbors[i]) << "Neighbor mismatch for global vertex " << gid;
            }
        }
    }

    void run_extraction_comparison(const std::string& graph_name, partition_t k) {
        std::string graph_path = find_graph(graph_name);
        if (graph_path.empty()) {
            GTEST_SKIP() << "Test graph " << graph_name << " not found.";
        }

        HostGraph hg = from_file(graph_path);
        
        // Create a simple partition (round robin)
        HostPartition h_map("h_map", hg.n);
        for (vertex_t i = 0; i < hg.n; ++i) {
            h_map(i) = i % k;
        }

        // 1. CPU Extraction
        std::vector<HostGraph> cpu_subgraphs;
        extract_all_subgraphs_cpu(hg, h_map, k, cpu_subgraphs);
        
        // We also need the local-to-global mapping for CPU subgraphs to compare correctly
        // extract_all_subgraphs_cpu assigns local IDs in global ID order per block.
        std::vector<std::vector<vertex_t>> cpu_local_to_global(k);
        for (vertex_t u = 0; u < hg.n; ++u) {
            cpu_local_to_global[h_map(u)].push_back(u);
        }

        // 2. GPU Extraction
        KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(512 * 1024 * 1024, "comparison_stack");
        DeviceExecutionSpace exec_space;
        f64 dummy_ms = 0;
        Graph g = from_HostGraph(hg, mem_stack, dummy_ms, exec_space);
        
        Partition p = initialize_partition(hg.n, k, (weight_t)hg.m, mem_stack, exec_space);
        Kokkos::deep_copy(exec_space, p.map, h_map);
        exec_space.fence();

        GraphBatch batch;
        init_GraphBatch(batch, g, k, mem_stack);
        
        DeviceU8 empty_mask;
        UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
        UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);

        extract_all_subgraphs(g, batch, p, empty_mask, local_ids, local_degree, exec_space);

        // 3. Compare CPU vs GPU result
        for (partition_t i = 0; i < k; ++i) {
            Graph gpu_sub_g = get_Graph(batch, i);
            UnmanagedDeviceVertex gpu_g_ids = get_global_ids(batch, i);
            compare_extracted_subgraphs(cpu_subgraphs[i], h_map, cpu_local_to_global[i], gpu_sub_g, gpu_g_ids, exec_space);
        }

        free_GraphBatch(batch, mem_stack);
        free_partition(p, mem_stack);
        free_graph(g, mem_stack);
        destroy(mem_stack);
    }
};

TEST_F(GpuCpuComparisonTest, ExtractAllSubgraphs_fe_ocean) {
    run_extraction_comparison("fe_ocean.graph", 4);
}

TEST_F(GpuCpuComparisonTest, ExtractAllSubgraphs_G2_circuit) {
    run_extraction_comparison("G2_circuit.mtx.graph", 8);
}

TEST_F(GpuCpuComparisonTest, ExtractAllSubgraphs_thermomech_TC) {
    run_extraction_comparison("thermomech_TC.mtx.graph", 16);
}

TEST_F(GpuCpuComparisonTest, RecalculateBlockWeightsComparison) {
    std::string graph_path = find_graph("fe_ocean.graph");
    if (graph_path.empty()) {
        GTEST_SKIP() << "Test graph fe_ocean.graph not found.";
    }

    HostGraph hg = from_file(graph_path);
    partition_t k = 8;
    HostPartition h_map("h_map", hg.n);
    for (vertex_t i = 0; i < hg.n; ++i) h_map(i) = i % k;

    // CPU Calculation
    std::vector<weight_t> cpu_bweights(k, 0);
    for (vertex_t i = 0; i < hg.n; ++i) {
        cpu_bweights[h_map(i)] += hg.uniform_vertex_weights ? 1 : hg.weights(i);
    }

    // GPU Calculation
    KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(256 * 1024 * 1024, "comparison_stack");
    DeviceExecutionSpace exec_space;
    f64 dummy_ms = 0;
    Graph g = from_HostGraph(hg, mem_stack, dummy_ms, exec_space);
    Partition p = initialize_partition(hg.n, k, (weight_t)hg.m, mem_stack, exec_space);
    Kokkos::deep_copy(exec_space, p.map, h_map);
    exec_space.fence();

    recalculate_block_weights(g, p.map, p.bweights, exec_space);

    HostWeight h_bweights("h_bweights", k);
    Kokkos::deep_copy(exec_space, h_bweights, p.bweights);
    exec_space.fence();

    for (partition_t i = 0; i < k; ++i) {
        EXPECT_EQ(cpu_bweights[i], h_bweights(i)) << "Weight mismatch for block " << i;
    }

    free_partition(p, mem_stack);
    free_graph(g, mem_stack);
    destroy(mem_stack);
}
