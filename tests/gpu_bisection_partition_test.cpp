#include <gtest/gtest.h>
#include "initial_partitioning/gpu_bisection_partition.h"

using namespace GPU_HeiPa;

class GPU_BisectionPartitionTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(GPU_BisectionPartitionTest, HierarchyManagerInit) {
    HierarchyManager manager;
    std::vector<partition_t> hierarchy = {2, 2, 2}; // 2*2*2 = 8
    partition_t k = 8;
    init_HierarchyManager(manager, hierarchy, 2 * k);

    EXPECT_EQ(manager.total_k, 8);
    EXPECT_EQ(manager.hierarchy.size(), 3);
    EXPECT_EQ(manager.unit_sizes[0], 1);
    EXPECT_EQ(manager.unit_sizes[1], 2);
    EXPECT_EQ(manager.unit_sizes[2], 4);
    
    // Initial state
    EXPECT_TRUE(manager.active[0]);
    EXPECT_EQ(manager.curr_level[0], 2);
    EXPECT_EQ(manager.curr_load[0], 2);
}

TEST_F(GPU_BisectionPartitionTest, HierarchyManagerSplit) {
    HierarchyManager manager;
    std::vector<partition_t> hierarchy = {2, 2, 2};
    partition_t k = 8;
    init_HierarchyManager(manager, hierarchy, 2 * k);

    partition_t l_k, r_k;
    split_into(manager, 0, l_k, r_k);
    EXPECT_EQ(l_k, 4);
    EXPECT_EQ(r_k, 4);

    split(manager, 0, l_k, r_k);
    EXPECT_TRUE(manager.active[0]);
    EXPECT_TRUE(manager.active[4]);
    EXPECT_EQ(manager.curr_level[0], 2);
    EXPECT_EQ(manager.curr_load[0], 1);
    EXPECT_EQ(manager.curr_level[4], 2);
    EXPECT_EQ(manager.curr_load[4], 1);
}

TEST_F(GPU_BisectionPartitionTest, HierarchyManagerDescend) {
    HierarchyManager manager;
    std::vector<partition_t> hierarchy = {2, 2, 2};
    partition_t k = 8;
    init_HierarchyManager(manager, hierarchy, 2 * k);

    // Initial load at level 2 is 2. Split it.
    partition_t l_k, r_k;
    split_into(manager, 0, l_k, r_k);
    split(manager, 0, l_k, r_k);

    // Now id 0 is at level 2 with load 1. Descend.
    EXPECT_TRUE(descend(manager, 0));
    EXPECT_EQ(manager.curr_level[0], 1);
    EXPECT_EQ(manager.curr_load[0], 2);

    EXPECT_TRUE(descend(manager, 0));
    EXPECT_EQ(manager.curr_level[0], 0);
    EXPECT_EQ(manager.curr_load[0], 2);

    EXPECT_FALSE(descend(manager, 0));
}

TEST_F(GPU_BisectionPartitionTest, GraphBatchTest) {
    KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(1024 * 1024, "test_stack");
    
    // Create a dummy graph
    Graph g;
    g.n = 10;
    g.m = 20;
    
    partition_t k = 2;
    GraphBatch batch;
    init_GraphBatch(batch, g, k, mem_stack);
    
    // Manually set actual dimensions (usually done by extract_all_subgraphs)
    batch.actual_n(0) = 10;
    batch.actual_m(0) = 20;

    EXPECT_EQ(batch.n, 10);
    EXPECT_EQ(batch.m, 20);
    EXPECT_EQ(batch.k, 2);
    
    Graph g0 = get_Graph(batch, 0);
    ASSERT_EQ(g0.n, 10);
    EXPECT_EQ(g0.neighborhood.extent(0), 11);
    EXPECT_EQ(g0.edges_v.extent(0), 20);
    
    free_GraphBatch(batch, mem_stack);
    destroy(mem_stack);
}

TEST_F(GPU_BisectionPartitionTest, RecalculateBlockWeights) {
    KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(1024 * 1024, "test_stack");
    DeviceExecutionSpace exec_space;

    // 1. Create a small graph: 4 vertices, 0 edges, weights {1, 2, 3, 4}
    HostGraph hg;
    hg.uniform_vertex_weights = false;
    hg.uniform_edge_weights = true;
    allocate_memory(hg, 4, 0, 10);
    hg.weights(0) = 1; hg.weights(1) = 2; hg.weights(2) = 3; hg.weights(3) = 4;
    hg.neighborhood(0) = 0; hg.neighborhood(1) = 0; hg.neighborhood(2) = 0; hg.neighborhood(3) = 0; hg.neighborhood(4) = 0;
    hg.g_weight = 10;
    
    f64 dummy_ms = 0;
    Graph g = from_HostGraph(hg, mem_stack, dummy_ms, exec_space);
    
    // 2. Create a partition: {0, 0, 1, 1}
    Partition p = initialize_partition(4, 2, 10, mem_stack, exec_space);
    HostPartition h_map("h_map", 4);
    h_map(0) = 0; h_map(1) = 0; h_map(2) = 1; h_map(3) = 1;
    Kokkos::deep_copy(exec_space, p.map, h_map);
    
    // 3. Recalculate
    recalculate_block_weights(g, p.map, p.bweights, exec_space);
    
    HostWeight h_bweights("h_bweights", 2);
    Kokkos::deep_copy(exec_space, h_bweights, p.bweights);
    exec_space.fence();
    
    EXPECT_EQ(static_cast<weight_t>(h_bweights(0)), 3); // 1 + 2
    EXPECT_EQ(static_cast<weight_t>(h_bweights(1)), 7); // 3 + 4
    
    free_partition(p, mem_stack);
    free_graph(g, mem_stack);
    destroy(mem_stack);
}

TEST_F(GPU_BisectionPartitionTest, ExtractSubgraph) {
    KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(2 * 1024 * 1024, "test_stack");
    DeviceExecutionSpace exec_space;

    // 1. Create a graph: 4 vertices, 4 edges (square: 0-1, 1-2, 2-3, 3-0)
    HostGraph hg;
    hg.uniform_vertex_weights = true;
    hg.uniform_edge_weights = true;
    allocate_memory(hg, 4, 4, 4);
    hg.neighborhood(0) = 0; hg.neighborhood(1) = 1; hg.neighborhood(2) = 2; hg.neighborhood(3) = 3; hg.neighborhood(4) = 4;
    hg.edges_v(0) = 1; hg.edges_v(1) = 2; hg.edges_v(2) = 3; hg.edges_v(3) = 0;
    hg.g_weight = 4;
    
    f64 dummy_ms = 0;
    Graph g = from_HostGraph(hg, mem_stack, dummy_ms, exec_space);
    
    // 2. Partition: {0, 0, 1, 1}
    Partition p = initialize_partition(4, 2, 4, mem_stack, exec_space);
    HostPartition h_map("h_map", 4);
    h_map(0) = 0; h_map(1) = 0; h_map(2) = 1; h_map(3) = 1;
    Kokkos::deep_copy(exec_space, p.map, h_map);
    exec_space.fence();
    
    // 3. Extract subgraph for block 0 (vertices 0 and 1)
    Graph sub_g = make_graph(2, 2, 2, true, true, mem_stack); // Upper bound for edges
    ASSERT_NE(sub_g.neighborhood.data(), nullptr);
    UnmanagedDeviceVertex global_ids((vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * 2), 2);
    
    extract_subgraph(g, sub_g, global_ids, 0, p, mem_stack, exec_space);
    
    EXPECT_EQ(sub_g.n, 2);
    EXPECT_EQ(sub_g.m, 1);
    
    HostU32 h_neighborhood("h_neighborhood", 3);
    Kokkos::deep_copy(exec_space, h_neighborhood, sub_g.neighborhood);
    exec_space.fence();
    EXPECT_EQ(h_neighborhood(0), 0);
    EXPECT_EQ(h_neighborhood(1), 1);
    EXPECT_EQ(h_neighborhood(2), 1);
    
    pop_front(mem_stack); // global_ids
    free_graph(sub_g, mem_stack);
    free_partition(p, mem_stack);
    free_graph(g, mem_stack);
    destroy(mem_stack);
}

TEST_F(GPU_BisectionPartitionTest, ExtractAllSubgraphs) {
    KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(4 * 1024 * 1024, "test_stack");
    DeviceExecutionSpace exec_space;

    // 1. Create a graph: 6 vertices, 10 edges
    // Edges: (0,1), (1,0), (1,2), (2,1), (2,3), (3,2), (3,4), (4,3), (4,5), (5,4)
    HostGraph hg;
    hg.uniform_vertex_weights = true;
    hg.uniform_edge_weights = true;
    allocate_memory(hg, 6, 10, 10);
    hg.neighborhood(0)=0; hg.neighborhood(1)=1; hg.neighborhood(2)=3; hg.neighborhood(3)=5; hg.neighborhood(4)=7; hg.neighborhood(5)=9; hg.neighborhood(6)=10;
    hg.edges_v(0)=1;
    hg.edges_v(1)=0; hg.edges_v(2)=2;
    hg.edges_v(3)=1; hg.edges_v(4)=3;
    hg.edges_v(5)=2; hg.edges_v(6)=4;
    hg.edges_v(7)=3; hg.edges_v(8)=5;
    hg.edges_v(9)=4;
    hg.m = 10;
    hg.g_weight = 6;
    
    f64 dummy_ms = 0;
    Graph g = from_HostGraph(hg, mem_stack, dummy_ms, exec_space);
    
    // 2. Partition: {0, 0, 1, 1, 2, 2}
    Partition p = initialize_partition(6, 3, 10, mem_stack, exec_space);
    HostPartition h_map("h_map", 6);
    h_map(0)=0; h_map(1)=0; h_map(2)=1; h_map(3)=1; h_map(4)=2; h_map(5)=2;
    Kokkos::deep_copy(exec_space, p.map, h_map);
    exec_space.fence();

    // 3. Extract all
    GraphBatch batch;
    init_GraphBatch(batch, g, 3, mem_stack);
    
    DeviceU8 empty_mask;
    UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
    UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);

    extract_all_subgraphs(g, batch, p, empty_mask, local_ids, local_degree, exec_space);
    
    // 4. Verify Block 0
    Graph g0 = get_Graph(batch, 0);
    EXPECT_EQ(g0.n, 2);
    EXPECT_EQ(g0.m, 2); // 0-1 and 1-0
    EXPECT_EQ(g0.g_weight, 2);
    
    // 5. Verify Block 1
    Graph g1 = get_Graph(batch, 1);
    EXPECT_EQ(g1.n, 2);
    EXPECT_EQ(g1.m, 2); // 2-3 and 3-2
    EXPECT_EQ(g1.g_weight, 2);
    
    // 6. Verify Block 2
    Graph g2 = get_Graph(batch, 2);
    EXPECT_EQ(g2.n, 2);
    EXPECT_EQ(g2.m, 2); // 4-5 and 5-4
    EXPECT_EQ(g2.g_weight, 2);
    
    pop_back(mem_stack); // local_degree
    pop_back(mem_stack); // local_ids
    free_GraphBatch(batch, mem_stack);
    free_partition(p, mem_stack);
    free_graph(g, mem_stack);
    destroy(mem_stack);
}

TEST_F(GPU_BisectionPartitionTest, ExtractMaskedSubgraphs) {
    KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(4 * 1024 * 1024, "test_stack");
    DeviceExecutionSpace exec_space;

    // 1. Create a graph: 6 vertices, 10 edges (same as ExtractAllSubgraphs)
    HostGraph hg;
    hg.uniform_vertex_weights = true;
    hg.uniform_edge_weights = true;
    allocate_memory(hg, 6, 10, 10);
    hg.neighborhood(0)=0; hg.neighborhood(1)=1; hg.neighborhood(2)=3; hg.neighborhood(3)=5; hg.neighborhood(4)=7; hg.neighborhood(5)=9; hg.neighborhood(6)=10;
    hg.edges_v(0)=1;
    hg.edges_v(1)=0; hg.edges_v(2)=2;
    hg.edges_v(3)=1; hg.edges_v(4)=3;
    hg.edges_v(5)=2; hg.edges_v(6)=4;
    hg.edges_v(7)=3; hg.edges_v(8)=5;
    hg.edges_v(9)=4;
    hg.m = 10; hg.g_weight = 6;
    f64 dummy_ms = 0;
    Graph g = from_HostGraph(hg, mem_stack, dummy_ms, exec_space);
    
    // 2. Partition: {0, 0, 1, 1, 2, 2}
    Partition p = initialize_partition(6, 3, 10, mem_stack, exec_space);
    HostPartition h_map("h_map", 6);
    h_map(0)=0; h_map(1)=0; h_map(2)=1; h_map(3)=1; h_map(4)=2; h_map(5)=2;
    Kokkos::deep_copy(exec_space, p.map, h_map);
    
    // 3. Extract with mask
    GraphBatch batch;
    init_GraphBatch(batch, g, 3, mem_stack);

    DeviceU8 active_mask("active_mask", batch.max_blocks);
    HostU8 h_active_mask("h_active_mask", batch.max_blocks);
    h_active_mask(0) = 1; h_active_mask(1) = 0; h_active_mask(2) = 1;
    Kokkos::deep_copy(exec_space, active_mask, h_active_mask);
    exec_space.fence();
    
    UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
    UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);

    extract_all_subgraphs(g, batch, p, active_mask, local_ids, local_degree, exec_space);
    
    // 5. Verify Results
    // Block 0: Should be extracted
    Graph g0 = get_Graph(batch, 0);
    EXPECT_EQ(g0.n, 2);
    EXPECT_EQ(g0.m, 2);
    
    // Block 1: Should be SKIPPED (n=0, m=0)
    Graph g1 = get_Graph(batch, 1);
    EXPECT_EQ(g1.n, 0);
    EXPECT_EQ(g1.m, 0);
    
    // Block 2: Should be extracted
    Graph g2 = get_Graph(batch, 2);
    EXPECT_EQ(g2.n, 2);
    EXPECT_EQ(g2.m, 2);
    
    pop_back(mem_stack); // local_degree
    pop_back(mem_stack); // local_ids
    free_GraphBatch(batch, mem_stack);
    free_partition(p, mem_stack);
    free_graph(g, mem_stack);
    destroy(mem_stack);
}

TEST_F(GPU_BisectionPartitionTest, BatchedBisectCorrectness) {
    KokkosMemoryStack mem_stack = initialize_kokkos_memory_stack(8 * 1024 * 1024, "test_stack");
    DeviceExecutionSpace exec_space;

    // 1. Create a graph: 12 vertices, 3 blocks of 4 vertices each
    HostGraph hg;
    hg.uniform_vertex_weights = true;
    hg.uniform_edge_weights = true;
    allocate_memory(hg, 12, 12, 12);
    hg.neighborhood(0)=0; hg.neighborhood(1)=1; hg.neighborhood(2)=2; hg.neighborhood(3)=3;
    hg.neighborhood(4)=4; hg.neighborhood(5)=5; hg.neighborhood(6)=6; hg.neighborhood(7)=7;
    hg.neighborhood(8)=8; hg.neighborhood(9)=9; hg.neighborhood(10)=10; hg.neighborhood(11)=11;
    hg.neighborhood(12)=12;
    for (int i = 0; i < 12; ++i) hg.edges_v(i) = (i % 4 == 3) ? i - 3 : i + 1; // 0->1, 1->2, 2->3, 3->0 etc.
    hg.m = 12; hg.g_weight = 12;
    f64 dummy_ms = 0;
    Graph g = from_HostGraph(hg, mem_stack, dummy_ms, exec_space);

    Partition p = initialize_partition(12, 3, 12, mem_stack, exec_space);
    HostPartition h_map("h_map", 12);
    for (int i = 0; i < 12; ++i) h_map(i) = i / 4;
    Kokkos::deep_copy(exec_space, p.map, h_map);

    GraphBatch batch;
    init_GraphBatch(batch, g, 3, mem_stack);
    DeviceU8 empty_mask;
    UnmanagedDeviceVertex local_ids((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);
    UnmanagedDeviceVertex local_degree((vertex_t *) get_chunk_back(mem_stack, sizeof(vertex_t) * g.n), g.n);

    extract_all_subgraphs(g, batch, p, empty_mask, local_ids, local_degree, exec_space);

    // 2. Bisect using Loop
    std::vector<HostPartition> loop_results(3);
    for (partition_t i = 0; i < 3; ++i) {
        Graph sub_g = get_Graph(batch, i);
        UnmanagedDevicePartition sub_part_map = get_partition(batch, i);
        bisect(sub_g, 2, 2, sub_part_map, exec_space);
        loop_results[i] = HostPartition("h_sub", sub_g.n);
        Kokkos::deep_copy(exec_space, loop_results[i], sub_part_map);
    }
    exec_space.fence();

    // 3. Bisect using Batch (Re-extract to reset partition buffers)
    extract_all_subgraphs(g, batch, p, empty_mask, local_ids, local_degree, exec_space);
    
    DeviceU8 active_mask("active_mask", batch.max_blocks);
    HostU8 h_active_mask("h_active_mask", batch.max_blocks);
    h_active_mask(0) = 1; h_active_mask(1) = 1; h_active_mask(2) = 1;
    Kokkos::deep_copy(exec_space, active_mask, h_active_mask);
    
    DeviceWeight lmax_l("lmax_l", batch.max_blocks);
    DeviceWeight lmax_r("lmax_r", batch.max_blocks);
    HostWeight h_lmax("h_lmax", batch.max_blocks);
    for (int i=0; i<3; ++i) h_lmax(i) = 2;
    Kokkos::deep_copy(exec_space, lmax_l, h_lmax);
    Kokkos::deep_copy(exec_space, lmax_r, h_lmax);

    batched_bisect(batch, active_mask, lmax_l, lmax_r, exec_space);

    // 4. Compare
    for (partition_t i = 0; i < 3; ++i) {
        Graph sub_g = get_Graph(batch, i);
        UnmanagedDevicePartition sub_part = get_partition(batch, i);
        HostPartition h_sub_part("h_sub_part", sub_g.n);
        Kokkos::deep_copy(exec_space, h_sub_part, sub_part);
        exec_space.fence();

        for (vertex_t j = 0; j < sub_g.n; ++j) {
            EXPECT_EQ(h_sub_part(j), loop_results[i](j)) << "Mismatch in block " << (int)i << " at vertex " << j;
        }
    }

    pop_back(mem_stack); // local_degree
    pop_back(mem_stack); // local_ids
    free_GraphBatch(batch, mem_stack);
    free_partition(p, mem_stack);
    free_graph(g, mem_stack);
    destroy(mem_stack);
}
