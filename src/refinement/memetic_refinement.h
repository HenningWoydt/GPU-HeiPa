#ifndef GPU_HEIPA_MEMETIC_REFINEMENT_H
#define GPU_HEIPA_MEMETIC_REFINEMENT_H

#include <Kokkos_Core.hpp>
#include <bitset>
#include <numeric>
#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <Kokkos_Random.hpp>

#include "../utility/definitions.h"
#include "../utility/memetic_helper.h"
#include "../utility/hungarian_algorithm.h"
#include "../datastructures/partition.h"
#include "block_conn.h"

#include "omp.h"

namespace GPU_HeiPa {




    struct KeyTuple {
        u32 key_count;
        u64 key;
    };


    inline u32 next_power_of_two(
        u32 k
    ) {
        if (k <= 1) return 1;

        k--;
        k |= k >> 1;
        k |= k >> 2;
        k |= k >> 4;
        k |= k >> 8;
        k |= k >> 16;
        return k + 1;
    }

    // basically returns log2(k)
    // and k should always be power of 2
    u32 bits_needed(u32 k) {
        u32 b = 0;
        k--;
        while (k > 0) {
            k >>= 1;
            b++;
        }
        return b;
    }

    KOKKOS_FUNCTION u64 determine_key(
        vertex_t u,
        const Kokkos::View<int *> &parent_ids,
        const Kokkos::View<Partition *> &population,
        u32 num_bits
    ) {
        u64 key = 0;
        for (size_t i = 0; i < parent_ids.size(); ++i) {
            u64 val = static_cast<u64>(population[parent_ids[i]].map(u));
            key |= (val & 0xFF) << (num_bits * i); //! do i even need this &0xFF ?
        }
        return key;
    }


    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- selection: --------------------------------------------------------------
    //! -------------------------------------------------------------------------------------------------


    inline int tournament_selection(
        const std::vector<weight_t> &fitness_values,
        const u32 tournament_size,
        const std::vector<size_t> &active
    ) {
        // get num_parents random numbers between [0, num_individuals)
        size_t num_individuals = active.size();
        std::vector<size_t> indices;
        std::unordered_set<size_t> unique_indices;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, num_individuals - 1);

        while (unique_indices.size() < tournament_size) {
            size_t idx = dis(gen);
            size_t parent_id = active[idx];
            if (unique_indices.insert(parent_id).second) {
                indices.push_back(parent_id);
            }
        }


        size_t best_idx = indices[0];
        weight_t best_fitness = fitness_values[best_idx];
        for (size_t i = 1; i < indices.size(); ++i) {
            if (fitness_values[indices[i]] < best_fitness) {
                best_fitness = fitness_values[indices[i]];
                best_idx = indices[i];
            }
        }
        return static_cast<int>(best_idx);
    }


    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- selection (shrinking solver): -------------------------------------------
    //! -------------------------------------------------------------------------------------------------


    inline int tournament_selection(
        const std::vector<weight_t> &fitness_values,
        const u32 tournament_size,
        const size_t parents_curr
    ) {
        // get num_parents random numbers between [0, num_individuals)
        size_t num_individuals = parents_curr;
        std::vector<size_t> indices;
        std::unordered_set<size_t> unique_indices;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, num_individuals - 1);

        while (unique_indices.size() < tournament_size) {
            size_t idx = dis(gen);
            if (unique_indices.insert(idx).second) {
                indices.push_back(idx);
            }
        }


        size_t best_idx = indices[0];
        weight_t best_fitness = fitness_values[best_idx];
        for (size_t i = 1; i < indices.size(); ++i) {
            if (fitness_values[indices[i]] < best_fitness) {
                best_fitness = fitness_values[indices[i]];
                best_idx = indices[i];
            }
        }
        return static_cast<int>(best_idx);
    }



    //! -------------------------------------------------------------------------------------------------
    //! ----------------------- different ways to distribute leftover vertices: -------------------------
    //! -------------------------------------------------------------------------------------------------



    template<bool uniform_vw>
    inline void assign_leftovers_favorUnderloadedBlocks(
        const Graph &graph,
        Partition &child,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        DeviceExecutionSpace &exec_space
    ) {
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);
        UnmanagedDeviceF64 distribution = UnmanagedDeviceF64((f64 *) get_chunk_back(mem_stack, sizeof(f64) * k), k);


        Kokkos::parallel_scan(
            "create distribution",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, k),
            KOKKOS_LAMBDA(partition_t id, f64 &update, bool final) {
                weight_t sum = 0;
                weight_t inverse_weight = (lmax - child.bweights(id));

                for (partition_t i = 0; i < k; ++i) {
                    sum += (lmax - child.bweights(i));
                }

                update += static_cast<double>(inverse_weight) / static_cast<double>(sum);
                if (final) {
                    distribution(id) = update;
                }
            }
        );


        // assign remaining vertices
        Kokkos::parallel_for(
            "assign leftovers",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    f64 rand = gen.drand(0.0, 1.0);
                    random_pool.free_state(gen);


                    for (u32 i = 0; i < k; ++i) {
                        if ((rand < distribution(i)) || (i == (k - 1))) {
                            child.map(u) = i;
                            Kokkos::atomic_add(&child.bweights(i), uniform_vw ? 1 : graph.weights(u));
                            break;
                        }
                    }
                }
            }
        );

        pop_back(mem_stack); //rm distribution
    }

    template<bool uniform_vw>
    inline void assign_leftovers_gain_and_weight(
        const Graph &graph,
        Partition &child,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        f64 alpha,
        DeviceExecutionSpace &exec_space
    ) {
        // this determines how much underloaded blocks are weighted
        // alpha = 0 -> only gain, big alpha -> only underloaded blocks

        Kokkos::Random_XorShift64_Pool<> random_pool(12345);


        BlockConn bc;
        if (graph.uniform_edge_weights) {
            bc = init_BlockConn<true>(graph, child, mem_stack, exec_space);
        } else {
            bc = init_BlockConn<false>(graph, child, mem_stack, exec_space);
        }

        //! determine max gain
        DeviceScalarWeight max_gain = DeviceScalarWeight("highest gain value");;
        Kokkos::parallel_reduce(
            "determine max gain",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, bc.size),
            KOKKOS_LAMBDA(u32 index, weight_t &update) {
                weight_t val = bc.weights(index);
                if ((val > update) && (bc.ids(index) != 5 * k)) {
                    update = val;
                }
            }, Kokkos::Max(max_gain)
        );

        Kokkos::parallel_for(
            "distribute leftovers",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                // calculate gain
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    partition_t best_id = static_cast<partition_t>(gen.urand64() % static_cast<u64>(k));
                    random_pool.free_state(gen);

                    f64 best_score = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        if (id == 5 * k)
                            continue;

                        weight_t gain = bc.weights(i);

                        //! i think this is actually quite smart, because in the case that
                        //! child.bweights(id) > lmax, then the right part will get negative
                        //! -> i.e. overweight blocks are penalized
                        f64 my_score = gain + (alpha * max_gain() * (static_cast<double>(lmax - child.bweights(id)) / static_cast<double>(lmax)));

                        bool valid = (id != NULL_PART) & (id != HASH_RECLAIM); // single mask

                        // Update best if it's a candidate and better
                        bool better = valid & (my_score > best_score);
                        best_score = better ? my_score : best_score;
                        best_id = better ? id : best_id;
                    }

                    child.map(u) = best_id;
                    Kokkos::atomic_add(&child.bweights(best_id), uniform_vw ? 1 : graph.weights(u));
                }
            }
        );

        free_BlockConn(bc, mem_stack);

        return;
    }




    template<bool uniform_vw>
    inline void assign_leftovers_fullyRandom(
        const Graph &graph,
        Partition &child,
        partition_t k,
        DeviceExecutionSpace &exec_space
    ) {

        Kokkos::Random_XorShift64_Pool<> random_pool(12345);

        // assign remaining vertices
        Kokkos::parallel_for(
            "assign leftovers",
            Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    partition_t id = static_cast<partition_t>(gen.urand64() % static_cast<u64>(k));
                    random_pool.free_state(gen);

                    child.map(u) = id;
                    Kokkos::atomic_add(&child.bweights(id), uniform_vw ? 1 : graph.weights(u));
                    
                }
            }
        );
    }



    template<bool uniform_vw>
    inline void assign_leftovers_gain(
        const Graph &graph,
        Partition &child,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        DeviceExecutionSpace &exec_space
    ) {
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);


        BlockConn bc;
        if (graph.uniform_edge_weights) {
            bc = init_BlockConn<true>(graph, child, mem_stack, exec_space);
        } else {
            bc = init_BlockConn<false>(graph, child, mem_stack, exec_space);
        }


        Kokkos::parallel_for(
            "distribute leftovers",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                // calculate gain
                if (child.map(u) == 5 * k) {
                    auto gen = random_pool.get_state();
                    partition_t best_id = static_cast<partition_t>(gen.urand64() % static_cast<u64>(k));
                    random_pool.free_state(gen);

                    weight_t best_conn = 0;

                    u32 r_beg = bc.row(u);
                    u32 r_len = bc.sizes(u);
                    u32 r_end = r_beg + r_len;

                    for (u32 i = r_beg; i < r_end; ++i) {
                        partition_t id = bc.ids(i);
                        if (id == 5 * k)
                            continue;

                        weight_t w = bc.weights(i);

                        bool valid = (id != NULL_PART) & (id != HASH_RECLAIM); // single mask

                        // Update best if it's a candidate and better
                        bool better = valid & (w > best_conn);
                        best_conn = better ? w : best_conn;
                        best_id = better ? id : best_id;
                    }

                    child.map(u) = best_id;
                    Kokkos::atomic_add(&child.bweights(best_id), uniform_vw ? 1 : graph.weights(u));
                }
            }
        );

        free_BlockConn(bc, mem_stack);

        return;
    }



    //! -------------------------------------------------------------------------------------------------
    //! --------------------------- crossover: ----------------------------------------------------------
    //! -------------------------------------------------------------------------------------------------

    template<bool uniform_vw>
    inline void backbone_based_crossover_paper_cpu(
        Partition &child,
        const Graph &graph,
        const std::vector<int> &parent_ids,
        const std::vector<Partition> &population,
        partition_t k,
        DeviceExecutionSpace &exec_space,
        u64 seed = 0
    ) {
        const size_t p = parent_ids.size();
        if (p == 0 || k == 0 || graph.n == 0) {
            return;
        }

        const partition_t REMOVED = NO_BLOCK_ID;
        const weight_t total_weight = uniform_vw ? static_cast<weight_t>(graph.n) : graph.g_weight;
        const weight_t w_opt = static_cast<weight_t>(std::ceil(static_cast<f64>(total_weight) / static_cast<f64>(k)));
        const u32 max_picks_per_parent = static_cast<u32>((k + static_cast<partition_t>(p) - 1) / static_cast<partition_t>(p));

        std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
        std::uniform_real_distribution<f64> uni01(0.0, 1.0);

        std::vector<HostPartition> parent_maps;
        parent_maps.reserve(p);
        for (size_t pi = 0; pi < p; ++pi) {
            HostPartition hp("paper_bbc_parent_map", graph.n);
            Kokkos::deep_copy(exec_space, hp, population[parent_ids[pi]].map);
            parent_maps.push_back(hp);
        }

        HostWeight host_vertex_weights;
        if (!uniform_vw) {
            host_vertex_weights = HostWeight("paper_bbc_vertex_weights", graph.n);
            Kokkos::deep_copy(exec_space, host_vertex_weights, graph.weights);
        }
        exec_space.fence();

        auto vw = [&](vertex_t v) -> weight_t {
            return uniform_vw ? 1 : host_vertex_weights(v);
        };

        std::vector<std::vector<partition_t> > labels(p, std::vector<partition_t>(graph.n, REMOVED));
        std::vector<std::vector<weight_t> > subset_weights(p, std::vector<weight_t>(k, 0));
        std::vector<u32> picked_count(p, 0);

        for (size_t pi = 0; pi < p; ++pi) {
            for (vertex_t v = 0; v < graph.n; ++v) {
                partition_t b = parent_maps[pi](v);
                labels[pi][v] = b;
                if (b < k) {
                    subset_weights[pi][b] += vw(v);
                }
            }
        }

        std::vector<partition_t> child_map(graph.n, REMOVED);
        std::vector<weight_t> child_bweights(k, 0);

        //! ---------- setup done ------------

        for (partition_t mu = 0; mu < k; ++mu) {
            // ! ----------- start step 1 ------------
            weight_t best_w = std::numeric_limits<weight_t>::lowest();
            size_t best_parent = 0;
            partition_t best_subset = 0;
            bool found = false;

            // understandable have a good day
            for (size_t pi = 0; pi < p; ++pi) {
                if (picked_count[pi] >= max_picks_per_parent) {
                    continue;
                }
                for (partition_t b = 0; b < k; ++b) {
                    if (subset_weights[pi][b] > best_w) {
                        best_w = subset_weights[pi][b]; //! richtig initialisiert?
                        best_parent = pi;
                        best_subset = b;
                        found = true;
                    }
                }
            }

            if (!found || best_w <= 0) {
                break;
            }

            std::vector<vertex_t> selected_vertices; //! this is S_i,j from the paper
            selected_vertices.reserve(graph.n);
            for (vertex_t v = 0; v < graph.n; ++v) {
                if (labels[best_parent][v] == best_subset) {
                    selected_vertices.push_back(v);
                }
            }

            // ! ----------- finished step 1 ------------

            std::vector<partition_t> best_match_subset(p, REMOVED);
            for (size_t pt = 0; pt < p; ++pt) {
                if (pt == best_parent) {
                    continue;
                }

                // count where the vertices lie in the other parent
                // basically create intersection-sizes for all subsets of pt with the selected subset S_i,j
                std::vector<u32> overlap(k, 0);
                for (vertex_t v: selected_vertices) {
                    partition_t b = labels[pt][v];
                    if (b < k) {
                        overlap[b] += 1; //? should this be 1 or the vertex-weight?
                    }
                }

                // determine maximum
                u32 best_overlap = 0;
                partition_t best_b = 0;
                for (partition_t b = 0; b < k; ++b) {
                    if (overlap[b] > best_overlap) {
                        best_overlap = overlap[b];
                        best_b = b;
                    }
                }
                // save the best intersecting subset
                best_match_subset[pt] = best_b;
            }

            std::vector<u8> in_intersection(graph.n, 0);
            std::vector<vertex_t> s_mu;
            s_mu.reserve(selected_vertices.size());

            for (vertex_t v: selected_vertices) {
                bool inside = true;
                for (size_t pt = 0; pt < p; ++pt) {
                    if (pt == best_parent) {
                        continue;
                    }
                    if (labels[pt][v] != best_match_subset[pt]) {
                        inside = false;
                        break;
                    }
                }

                if (inside) {
                    in_intersection[v] = 1;
                    s_mu.push_back(v);
                }
            }

            //! ------------ finished step 2 important stuff -----------------

            for (vertex_t v: selected_vertices) {
                if (in_intersection[v]) {
                    continue;
                }

                u32 occur = 0;
                for (size_t pt = 0; pt < p; ++pt) {
                    if (pt == best_parent) {
                        continue;
                    }
                    if (labels[pt][v] == best_match_subset[pt]) {
                        occur += 1;
                    }
                }

                const f64 acceptance = (p > 1) ? (static_cast<f64>(occur) / static_cast<f64>(p - 1)) : 1.0;
                if (acceptance >= uni01(rng)) {
                    s_mu.push_back(v);
                }
            }
            //! ------------ finished step 2 random assignment -----------------

            for (vertex_t v: s_mu) {
                child_map[v] = mu;
                child_bweights[mu] += vw(v);
            }

            picked_count[best_parent] += 1;

            for (vertex_t v: s_mu) {
                for (size_t pi = 0; pi < p; ++pi) {
                    partition_t old_b = labels[pi][v];
                    if (old_b < k) {
                        subset_weights[pi][old_b] -= vw(v);
                        labels[pi][v] = REMOVED;
                    }
                }
            }
        }

        for (vertex_t v = 0; v < graph.n; ++v) {
            if (child_map[v] != REMOVED) {
                continue;
            }

            const weight_t w = vw(v);
            std::vector<partition_t> feasible;
            feasible.reserve(k);
            for (partition_t b = 0; b < k; ++b) {
                if (child_bweights[b] + w <= w_opt) {
                    feasible.push_back(b);
                }
            }

            partition_t target = 0;
            if (!feasible.empty()) {
                std::uniform_int_distribution<size_t> dis(0, feasible.size() - 1);
                target = feasible[dis(rng)];
            } else {
                for (partition_t b = 1; b < k; ++b) {
                    if (child_bweights[b] < child_bweights[target]) {
                        target = b;
                    }
                }
            }

            child_map[v] = target;
            child_bweights[target] += w;
        }

        HostPartition host_child_map("paper_bbc_child_map", graph.n);
        HostWeight host_child_bweights("paper_bbc_child_bweights", k);

        for (vertex_t v = 0; v < graph.n; ++v) {
            host_child_map(v) = child_map[v];
        }
        for (partition_t b = 0; b < k; ++b) {
            host_child_bweights(b) = child_bweights[b];
        }

        Kokkos::deep_copy(exec_space, child.map, host_child_map);
        Kokkos::deep_copy(exec_space, child.bweights, host_child_bweights);
        exec_space.fence();
    }


    template<bool uniform_vw>
    inline void backbone_based_crossover(
        Partition &child,
        const Graph &graph,
        const std::vector<int> &parent_ids,
        const std::vector<Partition> &population,
        partition_t k,
        weight_t lmax,
        KokkosMemoryStack &mem_stack,
        const std::string &leftover_strategy,
        f64 alpha,
        partition_t extent,
        DeviceExecutionSpace &exec_space
    ) {
        
        //setup: get the vectors onto the GPU
        auto parent_ids_device = Kokkos::View<int *, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
            (int *) get_chunk_back(mem_stack, sizeof(int) * parent_ids.size()),
            parent_ids.size()
        );
        auto population_device = Kokkos::View<Partition *, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
            (Partition *) get_chunk_back(mem_stack, sizeof(Partition) * population.size()),
            population.size()
        );

        Kokkos::deep_copy(exec_space, parent_ids_device, Kokkos::View<const int *>(parent_ids.data(), parent_ids.size()));
        Kokkos::deep_copy(exec_space, population_device, Kokkos::View<const Partition *>(population.data(), population.size()));
        

        partition_t k_prime = next_power_of_two(k);
        u32 num_bits = bits_needed(k_prime);

        u64 num_buckets = static_cast<u64>(pow(k_prime, parent_ids.size()));

        auto buckets = Kokkos::View<KeyTuple *, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
            (KeyTuple *) get_chunk_back(mem_stack, sizeof(KeyTuple) * num_buckets), num_buckets
        );
        

        Kokkos::parallel_for(
            "init buckets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, num_buckets),
            KOKKOS_LAMBDA(u64 index) {
                buckets(index).key_count = 0;
                buckets(index).key = index;
            }
        );

        Kokkos::parallel_for(
            "fill buckets", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits);
                Kokkos::atomic_fetch_add(&buckets(key).key_count, 1);
            }
        );
        
        // sort descending based on key_count
        // after sorting, the k most frequent keys will be at the top
        // and you can query them via .key
        Kokkos::sort(exec_space, buckets, KOKKOS_LAMBDA(const KeyTuple &a, const KeyTuple &b) {
            return a.key_count > b.key_count;
        });



        
        partition_t local_extent = std::min(extent, k);
        if (local_extent < 1) {
            local_extent = 1;
        }

        // assign vertices of the backbone to the offspring
        Kokkos::parallel_for(
            "create new offspring", Kokkos::RangePolicy<DeviceExecutionSpace>(exec_space, 0, graph.n),
            KOKKOS_LAMBDA(vertex_t u) {
                partition_t id;
                bool in_backbone = false;
                u64 key = determine_key(u, parent_ids_device, population_device, num_bits);
                for (partition_t j = 0; (j < local_extent) && (!in_backbone); ++j) {
                    for (partition_t i = 0; i < k; ++i) {
                        if (key == buckets(i + (j * k)).key) {
                            if (j == 0)
                                id = i;
                            else
                                id = k - i - 1; //! "reverse assignment" from full buckets to underloaded partitions
                         
                            Kokkos::atomic_add(&child.bweights(id), uniform_vw ? 1 : graph.weights(u));
                            in_backbone = true;
                            break;
                        }
                    }
                }
                if (!in_backbone) {
                    id = 5 * k; //! mark as not assigned 
                }

                child.map(u) = id;
        }
        );
        

        
    
        if (leftover_strategy == "random") {
            if(uniform_vw) {

                assign_leftovers_fullyRandom<true>(graph, child, k, exec_space);
            }else{
                assign_leftovers_fullyRandom<false>(graph, child, k, exec_space);
            }

        } else if (leftover_strategy == "balanced") {

            if(uniform_vw) {

                assign_leftovers_favorUnderloadedBlocks<true>(graph, child, k, lmax, mem_stack, exec_space);
            
            }else{
                assign_leftovers_favorUnderloadedBlocks<false>(graph, child, k, lmax, mem_stack, exec_space);
            }

        } else if (leftover_strategy == "gain") {

            if(uniform_vw) {

                assign_leftovers_gain<true>(graph, child, k, lmax, mem_stack, exec_space);
            
            }else{
                assign_leftovers_gain<false>(graph, child, k, lmax, mem_stack, exec_space);
            }

        } else {

            if(uniform_vw) {

                assign_leftovers_gain_and_weight<true>(graph, child, k, lmax, mem_stack, alpha, exec_space);
            
            }else{
                assign_leftovers_gain_and_weight<false>(graph, child, k, lmax, mem_stack, alpha, exec_space);
            }

        }
            
        
        pop_back(mem_stack); //rm buckets
        pop_back(mem_stack); //rm population
        pop_back(mem_stack); //rm parent_ids
        

    }

}

#endif
