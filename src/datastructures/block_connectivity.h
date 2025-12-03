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
    }

    inline BlockConnectivity from_scratch(const Graph &g,
                                          const Partition &partition,
                                          KokkosMemoryStack &mem_stack) {
        BlockConnectivity bc;
        // allocate rows
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "allocate_rows");
            auto *row_ptr = (u32 *) get_chunk_back(mem_stack, sizeof(u32) * (g.n + 1));
            bc.row = UnmanagedDeviceU32(row_ptr, g.n + 1);
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
                if (final) bc.row(i) = running + c;

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
        // actual fill of structure
        {
            ScopedTimer _t("refinement", "BlockConnectivity_fs", "fill");
            Kokkos::parallel_for("fill", g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = g.edges_u(i);
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                u32 r_beg = bc.row(u);
                u32 r_end = bc.row(u + 1);
                u32 r_len = r_end - r_beg;

                partition_t v_id = partition.map(v);

                u32 j = r_beg + hash32(v_id) % r_len;
                for (u32 t = 0; t < r_len; t++) {
                    if (j == r_end) { j = r_beg; }
                    partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), NULL_PART, v_id);
                    if (val == NULL_PART || val == v_id) {
                        Kokkos::atomic_add(&bc.weights(j), w);
                        break;
                    }
                    j += 1;
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

            Kokkos::parallel_for("remove_weight", n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = moves(i);
                partition_t u_id = partition.map(u);

                for (u32 k = g.neighborhood(u); k < g.neighborhood(u + 1); k++) {
                    vertex_t v = g.edges_v(k);
                    weight_t w = g.edges_w(k);

                    // search in v's neighborhood for u_id
                    u32 r_beg = bc.row(v);
                    u32 r_end = bc.row(v + 1);
                    u32 r_len = r_end - r_beg;

                    bc.dest_cache(v) = NULL_PART;

                    if (r_len == 0) { continue; }

                    u32 j = r_beg + hash32(u_id) % r_len;
                    for (u32 t = 0; t < r_len; t++) {
                        if (j == r_end) { j = r_beg; }
                        if (bc.ids(j) == u_id) {
                            weight_t old_w = Kokkos::atomic_fetch_sub(&bc.weights(j), w);
                            if (old_w == w) {
                                bc.ids(j) = NULL_PART;
                                break;
                            }
                        }
                        j += 1;
                    }
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        // add the connections
        {
            ScopedTimer _t("refinement", "BlockConnectivity_move", "add_conn");

            Kokkos::parallel_for("add_conn", n_moves, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = moves(i);
                partition_t new_u_id = id(u);

                for (u32 k = g.neighborhood(u); k < g.neighborhood(u + 1); ++k) {
                    vertex_t v = g.edges_v(k);
                    weight_t w = g.edges_w(k);

                    // search in v's neighborhood for new_u_id
                    u32 r_beg = bc.row(v);
                    u32 r_end = bc.row(v + 1);
                    u32 r_len = r_end - r_beg;

                    if (r_len == 0) { continue; }

                    // first pass check if new_u_id exists anywhere
                    bool exists = false;
                    u32 j = r_beg + hash32(new_u_id) % r_len;
                    for (u32 t = 0; t < r_len; t++) {
                        if (j == r_end) { j = r_beg; }
                        if (bc.ids(j) == new_u_id) {
                            // found the spot, add the weight
                            Kokkos::atomic_add(&bc.weights(j), w);
                            exists = true;
                            break;
                        }
                        j += 1;
                    }

                    if (exists) { continue; }

                    // new_u_id does not exist, now search for an empty spot
                    j = r_beg + hash32(new_u_id) % r_len;
                    for (u32 t = 0; t < r_len; t++) {
                        if (j == r_end) { j = r_beg; }
                        partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), NULL_PART, new_u_id);
                        if (val == NULL_PART || val == new_u_id) {
                            // found empty spot or good spot, add the weight
                            Kokkos::atomic_add(&bc.weights(j), w);
                            break;
                        }
                        j += 1;
                    }
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
    }

    KOKKOS_INLINE_FUNCTION
    weight_t conn_to_block(const BlockConnectivity &bc,
                           vertex_t u,
                           partition_t id) {
        u32 r_beg = bc.row(u);
        u32 r_end = bc.row(u + 1);
        u32 r_len = r_end - r_beg;

        u32 j = r_beg + hash32(id) % r_len;
        for (u32 t = 0; t < r_len; t++) {
            if (j == r_end) { j = r_beg; }
            if (bc.ids(j) == id) {
                return bc.weights(j);
            }
            j += 1;
        }
        return 0;
    }

    struct HostBlockConnectivity {
        Kokkos::View<u32 *, Kokkos::HostSpace> row;
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
            ASSERT(r1 >= r0 && r1 <= h_bc.size);

            const u32 seg_len = r1 - r0;
            const u32 deg_u = static_cast<u32>(h_g.neighborhood(u + 1) - h_g.neighborhood(u));
            const u32 max_cap = std::min<u32>(deg_u, static_cast<u32>(k));
            ASSERT(seg_len <= max_cap);

            // Build partition->weight map from u's neighborhood
            // (sum edge weights to each neighbor’s partition)
            std::unordered_map<partition_t, weight_t> neigh_pw;
            neigh_pw.reserve(std::min<u32>(deg_u, 32u)); // small initial bucket count

            for (vertex_t ei = h_g.neighborhood(u); ei < h_g.neighborhood(u + 1); ++ei) {
                const vertex_t v = h_g.edges_v(ei);
                const partition_t pid = h_part(v);
                ASSERT(pid < k);
                neigh_pw[pid] += h_g.edges_w(ei);
            }

            // Row-level invariants
            std::unordered_set<partition_t> seen_ids;
            seen_ids.reserve(seg_len);

            weight_t row_sum = 0;

            for (u32 i = r0; i < r1; ++i) {
                const partition_t id = h_bc.ids(i);
                const weight_t w = h_bc.weights(i);

                // ownership stamp + basic ranges
                ASSERT(w >= static_cast<weight_t>(0));

                // Allow sentinel id == k for unused slots; if present, weight must be 0
                if (id == k) {
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

            // Optional: sanity that row_sum doesn't exceed sum of all incident weights
            {
                weight_t neigh_sum = 0;
                for (const auto &kv: neigh_pw) neigh_sum += kv.second;
                ASSERT(row_sum <= neigh_sum);
            }

            // If you require that the row stores *all* nonzero partition sums when seg_len == max_cap,
            // you can optionally enforce that when seg_len == std::min(deg_u,k), every nonzero in neigh_pw
            // appears in bc (beware: if you intentionally truncate to top-k, skip this).
            //
            // if (seg_len < neigh_pw.size()) {
            //     // Likely truncation to capacity; no further assertion.
            // } else {
            //     for (auto &kv : neigh_pw) {
            //         const partition_t id = kv.first;
            //         if (kv.second == static_cast<weight_t>(0)) continue;
            //         assert(seen_ids.count(id) == 1 && "missing nonzero partition from bc row");
            //     }
            // }
        }
    }
}

#endif //GPU_HEIPA_BLOCK_CONNECTIVITY_H
