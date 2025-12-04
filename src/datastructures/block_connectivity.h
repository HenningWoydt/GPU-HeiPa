/*******************************************************************************
 * MIT License
 *
 * This file is part of GPU-HeiPa.
 *
 * Copyright (C) 2025 Henning Woydt <henning.woydt@informatik.uni-heidelberg.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef GPU_HEIPA_BLOCK_CONNECTIVITY_H
#define GPU_HEIPA_BLOCK_CONNECTIVITY_H

#include <unordered_set>

#include "../utility/definitions.h"
#include "../utility/kokkos_util.h"
#include "../datastructures/graph.h"
#include "../datastructures/partition.h"

namespace GPU_HeiPa {
    struct BlockConnectivity {
        vertex_t n = 0;
        u32 size = 0;

        UnmanagedDeviceU32 row;
        UnmanagedDeviceU32 sizes;
        UnmanagedDevicePartition ids;
        UnmanagedDeviceWeight weights;

        UnmanagedDeviceU32 lock;
        UnmanagedDevicePartition dest_cache;
    };

    inline void free_bc(BlockConnectivity &bc,
                        KokkosMemoryStack &mem_stack) {
        ScopedTimer _t("refinement", "BlockConnectivity", "free");

        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
        pop_back(mem_stack);
    }

    inline BlockConnectivity from_scratch(const Graph &g,
                                          const Partition &partition,
                                          KokkosMemoryStack &mem_stack) {
        BlockConnectivity bc;
        // allocate rows
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "allocate_rows");

            bc.row = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * (g.n + 1)), g.n + 1);
            bc.sizes = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * g.n), g.n);
            bc.n = g.n;
        }
        // set rows
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "set_rows");

            Kokkos::parallel_scan("set_rows", g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (i == 0) {
                    // first slot is 0
                    if (final) bc.row(0) = 0;
                    return;
                }

                const vertex_t u = i - 1;
                const u32 len = g.neighborhood(u + 1) - g.neighborhood(u);
                const u32 c = len < partition.k ? len : partition.k;

                // write inclusive row[i] = running + c
                if (final) {
                    bc.row(i) = running + c;
                    bc.sizes(u) = 0; // c;
                }

                running += c;
            });
            Kokkos::deep_copy(bc.size, Kokkos::subview(bc.row, g.n));
            Kokkos::fence();
        }
        // allocate rest
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "allocate");

            bc.ids = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * bc.size), bc.size);
            bc.weights = UnmanagedDeviceWeight((weight_t *) get_chunk_back(mem_stack, sizeof(weight_t) * bc.size), bc.size);
            Kokkos::deep_copy(bc.ids, NULL_PART);
            Kokkos::deep_copy(bc.weights, 0);

            bc.lock = UnmanagedDeviceU32((u32 *) get_chunk_back(mem_stack, sizeof(u32) * bc.n), bc.n);
            bc.dest_cache = UnmanagedDevicePartition((partition_t *) get_chunk_back(mem_stack, sizeof(partition_t) * bc.n), bc.n);
            Kokkos::deep_copy(bc.lock, 0);
            Kokkos::deep_copy(bc.dest_cache, NULL_PART);
            KOKKOS_PROFILE_FENCE();
        }
        // first fill of the structure
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "fill");

            Kokkos::parallel_for("fill", g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = g.edges_u(i);
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                u32 r_beg = bc.row(u);
                u32 r_end = bc.row(u + 1);
                u32 r_len = r_end - r_beg;

                if (r_len == 0) { return; }

                partition_t v_id = partition.map(v);

                for (u32 j = r_beg; j < r_end; j++) {
                    partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), NULL_PART, v_id);
                    if (val == NULL_PART) {
                        Kokkos::atomic_add(&bc.weights(j), w);
                        Kokkos::atomic_inc(&bc.sizes(u));
                        break;
                    }
                    if (val == v_id) {
                        Kokkos::atomic_add(&bc.weights(j), w);
                        break;
                    }
                }
            });
            KOKKOS_PROFILE_FENCE();
        }

        return bc;
    }

    inline void move_bc(BlockConnectivity &bc,
                        const Graph &g,
                        const Partition &partition,
                        const UnmanagedDevicePartition &id,
                        const UnmanagedDeviceVertex &moves,
                        const u32 n_moves) {
        // remove_weight
        {
            ScopedTimer _t("refinement", "BlockConnectivity_move", "remove_weight");

            using TeamPolicy = Kokkos::TeamPolicy<DeviceExecutionSpace, Kokkos::IndexType<u32>>;
            Kokkos::parallel_for("remove_weight", TeamPolicy(n_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamPolicy::member_type &t) {
                vertex_t u = moves(t.league_rank());
                partition_t old_u_id = partition.map(u);
                partition_t new_u_id = id(u);

                if (old_u_id == new_u_id) { return; }

                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 k) {
                    vertex_t v = g.edges_v(k);
                    weight_t w = g.edges_w(k);

                    // search in v's neighborhood for u_id
                    u32 r_beg = bc.row(v);
                    u32 r_len = bc.sizes(v);
                    u32 r_end = r_beg + r_len;

                    bc.dest_cache(v) = NULL_PART;

                    if (r_len == 0) { return; }

                    u32 j = r_beg + hash32(old_u_id) % r_len;
                    for (u32 t = 0; t < r_len; t++) {
                        if (j == r_end) { j = r_beg; }
                        if (bc.ids(j) == old_u_id) {
                            weight_t old_w = Kokkos::atomic_fetch_sub(&bc.weights(j), w);
                            if (old_w == w) {
                                bc.ids(j) = NULL_PART;
                                break;
                            }
                        }
                        j += 1;
                    }
                });
            });
            KOKKOS_PROFILE_FENCE();
        }
        // add the connections
        {
            ScopedTimer _t("refinement", "BlockConnectivity_move", "add_conn");

            using TeamPolicy = Kokkos::TeamPolicy<DeviceExecutionSpace, Kokkos::IndexType<u32>>;
            Kokkos::parallel_for("add_conn", TeamPolicy(n_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamPolicy::member_type &t) {
                vertex_t u = moves(t.league_rank());
                partition_t old_u_id = partition.map(u);
                partition_t new_u_id = id(u);

                if (old_u_id == new_u_id) { return; }

                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.neighborhood(u), g.neighborhood(u + 1)), [=](const u32 k) {
                    vertex_t v = g.edges_v(k);
                    weight_t w = g.edges_w(k);

                    // search in v's neighborhood for new_u_id
                    u32 r_beg = bc.row(v);
                    u32 r_len = bc.sizes(v);
                    u32 r_end = r_beg + r_len;

                    if (r_len == 0) { return; }

                    // first pass check if new_u_id exists anywhere
                    bool exists = false;
                    u32 j = r_beg + hash32(new_u_id) % r_len;
                    u32 empty_j = j;
                    for (u32 t = 0; t < r_len; t++) {
                        if (j == r_end) { j = r_beg; }
                        if (bc.ids(j) == NULL_PART) { empty_j = j; }
                        if (bc.ids(j) == new_u_id) {
                            // found the spot, add the weight
                            Kokkos::atomic_add(&bc.weights(j), w);
                            exists = true;
                            break;
                        }
                        j += 1;
                    }

                    if (exists) { return;; }

                    // new_u_id does not exist, now search for an empty spot, start at last seen empty spot and hope
                    j = empty_j;
                    for (u32 t = 0; t < r_len; t++) {
                        if (j == r_end) { j = r_beg; }
                        partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), NULL_PART, new_u_id);
                        if (val == NULL_PART || val == new_u_id) {
                            // found empty spot or good spot, add the weight
                            Kokkos::atomic_add(&bc.weights(j), w);
                            exists = true;
                            break;
                        }
                        j += 1;
                    }

                    if (exists) { return; }

                    // new_u_id does not exist and no empty spot -> grow the size
                    u32 r_reserve_beg = r_beg + r_len;
                    u32 r_reserve_end = bc.row(v + 1);

                    j = r_reserve_beg;
                    while (j < r_reserve_end) {
                        partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), NULL_PART, new_u_id);
                        if (val == NULL_PART) {
                            // we claimed this spot
                            Kokkos::atomic_add(&bc.weights(j), w);
                            Kokkos::atomic_inc(&bc.sizes(v));
                            break;
                        }
                        if (val == new_u_id) {
                            // also cool
                            Kokkos::atomic_add(&bc.weights(j), w);
                            break;
                        }
                        j += 1;
                    }
                });
            });
            KOKKOS_PROFILE_FENCE();
        }
    }

    struct HostBlockConnectivity {
        Kokkos::View<u32 *, Kokkos::HostSpace> row;
        Kokkos::View<u32 *, Kokkos::HostSpace> sizes;
        Kokkos::View<partition_t *, Kokkos::HostSpace> ids;
        Kokkos::View<weight_t *, Kokkos::HostSpace> weights;

        vertex_t n = 0;
        u32 size = 0;
    };

    inline HostBlockConnectivity download_block_connectivity(const BlockConnectivity &bc) {
        HostBlockConnectivity h_bc;

        // Copy metadata
        h_bc.n = bc.n;
        h_bc.size = bc.size;

        // Allocate host mirrors
        h_bc.row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.row);
        h_bc.sizes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.sizes);
        h_bc.ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.ids);
        h_bc.weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.weights);

        return h_bc;
    }

    inline void assert_bc(const BlockConnectivity &bc,
                          const Graph &g,
                          const Partition &partition,
                          partition_t k) {
        // Host mirrors
        HostBlockConnectivity h_bc = download_block_connectivity(bc);
        HostGraph h_g = to_host_graph(g);
        auto h_part = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), partition.map);

        const vertex_t n = h_g.n;

        // --- Global shape checks ---
        ASSERT(h_bc.n == n);
        ASSERT(h_bc.row.extent(0) == static_cast<size_t>(n + 1));
        ASSERT(h_bc.sizes.extent(0) == static_cast<size_t>(n));
        ASSERT(h_bc.ids.extent(0) == static_cast<size_t>(h_bc.size));
        ASSERT(h_bc.weights.extent(0) == static_cast<size_t>(h_bc.size));
        ASSERT(h_bc.row(0) == 0);
        for (vertex_t u = 0; u < n; ++u) {
            ASSERT(h_bc.row(u) <= h_bc.row(u + 1));
        }
        ASSERT(h_bc.row(n) == h_bc.size);

        // --- Per-vertex checks ---
        for (vertex_t u = 0; u < n; ++u) {
            const u32 r0 = h_bc.row(u);
            const u32 r1 = h_bc.row(u + 1);
            const u32 cap = r1 - r0;       // physical capacity
            const u32 len = h_bc.sizes(u); // logical length

            ASSERT(r1 >= r0 && r1 <= h_bc.size);
            ASSERT(len <= cap); // logical size cannot exceed capacity

            const u32 deg_u = static_cast<u32>(h_g.neighborhood(u + 1) - h_g.neighborhood(u));
            const u32 max_cap = std::min<u32>(deg_u, static_cast<u32>(k));
            // capacity is bounded by min(deg, k)
            ASSERT(cap <= max_cap);

            // Build partition->weight map from u's neighborhood
            std::unordered_map<partition_t, weight_t> neigh_pw;
            neigh_pw.reserve(std::min<u32>(deg_u, 32u));

            for (vertex_t ei = h_g.neighborhood(u); ei < h_g.neighborhood(u + 1); ++ei) {
                const vertex_t v = h_g.edges_v(ei);
                const partition_t pid = h_part(v);
                ASSERT(pid < k);
                neigh_pw[pid] += h_g.edges_w(ei);
            }

            // Row-level invariants
            std::unordered_set<partition_t> seen_ids;
            seen_ids.reserve(len);

            weight_t row_sum = 0;

            // 1) Logical region [r0, r0 + len)
            const u32 rend = r0 + len;
            ASSERT(rend <= r1);

            for (u32 i = r0; i < rend; ++i) {
                const partition_t id = h_bc.ids(i);
                const weight_t w = h_bc.weights(i);

                ASSERT(w >= static_cast<weight_t>(0));

                if (id == NULL_PART) {
                    // We allow empty slots in the logical region,
                    // though in practice you pack them densely.
                    ASSERT(w == static_cast<weight_t>(0));
                    continue;
                }

                // Valid partition id
                ASSERT(id < k);

                // Unique per row
                const bool inserted = seen_ids.insert(id).second;
                ASSERT(inserted);

                // Must correspond to real neighborhood mass
                auto it = neigh_pw.find(id);
                ASSERT(it != neigh_pw.end());
                ASSERT(it->second == w);

                row_sum += w;
            }

            // 2) Overflow region [r0 + len, r1) must be completely empty
            for (u32 i = rend; i < r1; ++i) {
                const partition_t id = h_bc.ids(i);
                const weight_t w = h_bc.weights(i);

                ASSERT(id == NULL_PART);
                ASSERT(w == static_cast<weight_t>(0));
            }

            // Sanity: row_sum doesn't exceed sum of all incident weights
            {
                weight_t neigh_sum = 0;
                for (const auto &kv: neigh_pw) neigh_sum += kv.second;
                ASSERT(row_sum <= neigh_sum);
            }

            // Optional: if you ever want to enforce that BC stores all nonzero
            // partition sums for this vertex, you can uncomment this block.
            //
            // if (seen_ids.size() == neigh_pw.size()) {
            //     for (auto &kv : neigh_pw) {
            //         const partition_t id = kv.first;
            //         if (kv.second == static_cast<weight_t>(0)) continue;
            //         ASSERT(seen_ids.count(id) == 1);
            //     }
            // }
        }
    }


    inline void analyze_block_connectivity(const BlockConnectivity &bc,
                                           const Graph &g,
                                           const Partition &partition,
                                           partition_t k) {
        using std::cout;
        using std::endl;

        const vertex_t n = bc.n;

        // --- Mirror device data to host ---
        auto h_row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.row);
        auto h_sizes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.sizes);
        auto h_ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.ids);
        auto h_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.weights);
        auto h_lock = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.lock);
        auto h_cache = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.dest_cache);

        // --- Basic shape ---
        u64 total_slots_phys = bc.size; // physical slots: sum of capacities
        u64 total_slots_logic = 0;      // sum of logical sizes
        u64 nonempty_slots_log = 0;     // ids != NULL_PART within [row, row+size)
        u64 nonempty_slots_ovf = 0;     // ids != NULL_PART beyond logical size, < capacity
        u64 used_slots = 0;             // ids in [0,k) within logical region

        // Per-row stats
        u64 total_row_cap = 0;  // sum capacities
        u64 total_row_len = 0;  // sum logical sizes
        u64 total_row_used = 0; // sum used slots (logical region only)

        u64 rows_with_any = 0;
        u64 rows_empty = 0;
        u64 rows_full_logic = 0; // logical load factor == 1
        u64 rows_high_load = 0;  // logical load factor > 0.75

        u32 max_row_cap = 0;
        vertex_t max_row_cap_v = 0;

        u32 max_row_len = 0;
        vertex_t max_row_len_v = 0;

        u32 max_row_used = 0;
        vertex_t max_row_used_v = 0;

        double max_row_load = 0.0;
        vertex_t max_row_load_v = 0;

        // histogram of load factors (logical) 0–1 in 10 bins
        static constexpr int NUM_BINS = 10;
        u64 load_hist[NUM_BINS] = {0};

        // Cache + lock stats
        u64 cache_null = 0;
        u64 cache_no_move = 0;
        u64 cache_valid = 0;
        u64 cache_oob = 0; // invalid part ids
        u64 lock_set = 0;

        // Graph stats
        auto h_part = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), partition.map);
        auto h_g = to_host_graph(g);

        u64 deg_sum = 0;
        u64 deg_nonzero = 0;
        u32 max_deg = 0;
        vertex_t max_deg_v = 0;

        for (vertex_t u = 0; u < n; ++u) {
            const u32 r0 = h_row(u);
            const u32 r1 = h_row(u + 1);
            const u32 cap = r1 - r0;    // physical capacity for row u
            const u32 len = h_sizes(u); // logical hash-table size
            const u32 rend = r0 + len;

            total_row_cap += cap;
            total_row_len += len;

            if (cap > max_row_cap) {
                max_row_cap = cap;
                max_row_cap_v = u;
            }
            if (len > max_row_len) {
                max_row_len = len;
                max_row_len_v = u;
            }

            // Graph degree
            const u32 deg_u = static_cast<u32>(h_g.neighborhood(u + 1) - h_g.neighborhood(u));
            deg_sum += deg_u;
            if (deg_u > 0) deg_nonzero++;
            if (deg_u > max_deg) {
                max_deg = deg_u;
                max_deg_v = u;
            }

            if (len == 0) {
                rows_empty++;
                continue;
            }

            u32 row_used = 0;
            u32 row_nonempty = 0;
            u32 row_nonempty_ovf = 0;

            // Logical region [r0, r0+len)
            for (u32 i = r0; i < rend; ++i) {
                partition_t id = h_ids(i);
                weight_t w = h_weights(i);

                if (id != NULL_PART) {
                    row_nonempty++;
                    row_used++;

                    if (id < k) {
                        used_slots++;
                    }

                    (void) w;
                }
            }

            // Overflow region [r0+len, r0+cap)
            for (u32 i = rend; i < r1; ++i) {
                partition_t id = h_ids(i);
                if (id != NULL_PART) {
                    row_nonempty_ovf++;
                }
            }

            if (row_used > 0) {
                rows_with_any++;
            }

            total_row_used += row_used;
            nonempty_slots_log += row_nonempty;
            nonempty_slots_ovf += row_nonempty_ovf;

            // logical load factor = used / len
            double load = len > 0 ? static_cast<double>(row_used) / static_cast<double>(len) : 0.0;

            if (row_used == len) {
                rows_full_logic++;
            }
            if (row_used > max_row_used) {
                max_row_used = row_used;
                max_row_used_v = u;
            }
            if (load > 0.75) {
                rows_high_load++;
            }
            if (load > max_row_load) {
                max_row_load = load;
                max_row_load_v = u;
            }

            // histogram bin for logical load
            int bin = 0;
            if (len > 0) {
                bin = static_cast<int>(load * NUM_BINS);
                if (bin >= NUM_BINS) bin = NUM_BINS - 1;
            }
            load_hist[bin]++;
        }

        // Analyze cache + lock
        for (vertex_t u = 0; u < n; ++u) {
            partition_t c = h_cache(u);
            if (h_lock(u) != 0) {
                lock_set++;
            }

            if (c == NULL_PART) {
                cache_null++;
            } else if (c == NO_MOVE) {
                cache_no_move++;
            } else if (c < k) {
                cache_valid++;
            } else {
                cache_oob++;
            }
        }

        // Averages
        double avg_row_cap = n > 0 ? static_cast<double>(total_row_cap) / static_cast<double>(n) : 0.0;
        double avg_row_len = n > 0 ? static_cast<double>(total_row_len) / static_cast<double>(n) : 0.0;
        double avg_row_used = n > 0 ? static_cast<double>(total_row_used) / static_cast<double>(n) : 0.0;

        double avg_load_factor = total_row_len > 0
                                     ? static_cast<double>(total_row_used) / static_cast<double>(total_row_len)
                                     : 0.0;

        double global_occupancy_logic = total_slots_logic > 0
                                            ? static_cast<double>(nonempty_slots_log) / static_cast<double>(total_slots_logic)
                                            : 0.0;

        double global_occupancy_phys = total_slots_phys > 0
                                           ? static_cast<double>(nonempty_slots_log + nonempty_slots_ovf) /
                                             static_cast<double>(total_slots_phys)
                                           : 0.0;

        double avg_deg = n > 0 ? static_cast<double>(deg_sum) / static_cast<double>(n) : 0.0;
        double avg_deg_nonzero = deg_nonzero > 0
                                     ? static_cast<double>(deg_sum) / static_cast<double>(deg_nonzero)
                                     : 0.0;

        // total_slots_logic = sum of sizes(u)
        total_slots_logic = total_row_len;

        // Print report
        cout << "===== BlockConnectivity Analysis =====" << endl;
        cout << "Vertices (n)                    : " << n << endl;
        cout << "Total BC slots (physical)       : " << total_slots_phys << endl;
        cout << "Total BC slots (logical)        : " << total_slots_logic << endl;
        cout << "Rows total capacity (sum cap)   : " << total_row_cap << endl;
        cout << "Rows total logical len (sum len): " << total_row_len << endl;
        cout << "Non-empty slots (logical region): " << nonempty_slots_log << endl;
        cout << "Non-empty slots (overflow)      : " << nonempty_slots_ovf << endl;
        cout << "Slots with valid part id        : " << used_slots << endl;
        cout << "Global occupancy (logical)      : " << global_occupancy_logic << endl;
        cout << "Global occupancy (physical)     : " << global_occupancy_phys << endl;

        cout << endl;
        cout << "Row capacity / logical size stats:" << endl;
        cout << "  avg row capacity             : " << avg_row_cap << endl;
        cout << "  max row capacity             : " << max_row_cap
                << " (vertex " << max_row_cap_v << ")" << endl;
        cout << "  avg logical row_len          : " << avg_row_len << endl;
        cout << "  max logical row_len          : " << max_row_len
                << " (vertex " << max_row_len_v << ")" << endl;
        cout << "  rows with any entries        : " << rows_with_any << endl;
        cout << "  rows empty (len == 0)        : " << rows_empty << endl;

        cout << endl;
        cout << "Per-row used-slot stats (logical region):" << endl;
        cout << "  avg used per row             : " << avg_row_used << endl;
        cout << "  avg load factor (used/len)   : " << avg_load_factor << endl;
        cout << "  rows full (load=1.0)         : " << rows_full_logic << endl;
        cout << "  rows load > 0.75             : " << rows_high_load << endl;
        cout << "  max row_used                 : " << max_row_used
                << " (vertex " << max_row_used_v << ")" << endl;
        cout << "  max load factor              : " << max_row_load
                << " (vertex " << max_row_load_v << ")" << endl;

        cout << endl;
        cout << "Load factor histogram (per row, logical region):" << endl;
        for (int b = 0; b < NUM_BINS; ++b) {
            double lo = static_cast<double>(b) / static_cast<double>(NUM_BINS);
            double hi = static_cast<double>(b + 1) / static_cast<double>(NUM_BINS);
            cout << "  [" << lo << ", " << hi << "): " << load_hist[b] << " rows" << endl;
        }

        cout << endl;
        cout << "Graph degree stats (for comparison):" << endl;
        cout << "  avg degree (all vertices)    : " << avg_deg << endl;
        cout << "  avg degree (deg>0)           : " << avg_deg_nonzero << endl;
        cout << "  max degree                   : " << max_deg
                << " (vertex " << max_deg_v << ")" << endl;

        cout << endl;
        cout << "Cache / lock stats:" << endl;
        cout << "  dest_cache == NULL_PART      : " << cache_null << endl;
        cout << "  dest_cache == NO_MOVE        : " << cache_no_move << endl;
        cout << "  dest_cache valid [0,k)       : " << cache_valid << endl;
        cout << "  dest_cache invalid / oob     : " << cache_oob << endl;
        cout << "  lock != 0                    : " << lock_set << endl;

        cout << "===== End BlockConnectivity Analysis =====" << endl;
    }
}

#endif //GPU_HEIPA_BLOCK_CONNECTIVITY_H
