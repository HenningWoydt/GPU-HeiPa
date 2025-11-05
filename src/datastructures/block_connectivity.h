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
        UnmanagedDeviceU32 row;
        UnmanagedDeviceVertex us;
        UnmanagedDevicePartition ids;
        UnmanagedDeviceWeight weights;
        vertex_t n = 0;
        u32 size = 0;
    };

    inline void free_bc(BlockConnectivity &bc,
                        KokkosMemoryStack &small_mem_stack) {
        ScopedTimer _t("refine", "BlockConnectivity", "free");

        bc.n = 0;
        bc.size = 0;
        bc.row = UnmanagedDeviceU32(nullptr, 0);
        bc.us = UnmanagedDeviceVertex(nullptr, 0);
        bc.ids = UnmanagedDevicePartition(nullptr, 0);
        bc.weights = UnmanagedDeviceWeight(nullptr, 0);

        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
        pop(small_mem_stack);
    }

    inline BlockConnectivity from_scratch(const Graph &g,
                                          const Partition &partition,
                                          KokkosMemoryStack &small_mem_stack) {
        BlockConnectivity bc;
        // allocate rows
        {
            ScopedTimer _t("refine", "BlockConnectivity_fs", "allocate_rows");
            auto *row_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * (g.n + 1));
            bc.row = UnmanagedDeviceU32(row_ptr, g.n + 1);
            bc.n = g.n;
        }
        // set rows
        {
            ScopedTimer _t("refine", "BlockConnectivity_fs", "set_rows");
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
            ScopedTimer _t("refine", "BlockConnectivity_fs", "allocate");
            auto *us_ptr = (vertex_t *) get_chunk(small_mem_stack, sizeof(vertex_t) * bc.size);
            auto *ids_ptr = (partition_t *) get_chunk(small_mem_stack, sizeof(partition_t) * bc.size);
            auto *weights_ptr = (weight_t *) get_chunk(small_mem_stack, sizeof(weight_t) * bc.size);
            bc.us = UnmanagedDeviceVertex(us_ptr, bc.size);
            bc.ids = UnmanagedDevicePartition(ids_ptr, bc.size);
            bc.weights = UnmanagedDeviceWeight(weights_ptr, bc.size);
            Kokkos::deep_copy(bc.ids, partition.k);
            Kokkos::deep_copy(bc.weights, 0);
            KOKKOS_PROFILE_FENCE();
        }
        // set u values
        {
            ScopedTimer _t("refine", "BlockConnectivity_fs", "fill_u");
            Kokkos::parallel_for("fill_u", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                for (size_t j = bc.row(u); j < bc.row(u + 1); j++) {
                    bc.us(j) = u;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        // actual fill of structure
        {
            ScopedTimer _t("refine", "BlockConnectivity_fs", "fill");
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
                    partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), partition.k, v_id);
                    if (val == partition.k || val == v_id) {
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

    inline BlockConnectivity rebuild(BlockConnectivity &old_bc,
                                     UnmanagedDeviceU32 &needs_more,
                                     const Graph &g,
                                     const Partition &partition,
                                     const UnmanagedDeviceU32 &to_move,
                                     const UnmanagedDeviceU64 &weight_id,
                                     KokkosMemoryStack &mem_stack,
                                     KokkosMemoryStack &small_mem_stack) {
        // we first build it on the mem stack and afterwards copy it over
        BlockConnectivity bc;
        // allocate rows
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "allocate_rows");
            auto *row_ptr = (u32 *) get_chunk(mem_stack, sizeof(u32) * (g.n + 1));
            bc.row = UnmanagedDeviceU32(row_ptr, g.n + 1);
            bc.n = g.n;
        }
        // set rows
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "set_rows");
            Kokkos::parallel_scan("set_rows", g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
                if (i == 0) {
                    if (final) bc.row(0) = 0;
                    return;
                }

                const vertex_t u = i - 1;

                const u32 deg = g.neighborhood(u + 1) - g.neighborhood(u);
                const u32 old_c = old_bc.row(u + 1) - old_bc.row(u);
                const u32 add = needs_more(u) == 0 ? 0u : (needs_more(u) + 2u); // extra slots
                const u32 max_c = (deg < partition.k) ? deg : partition.k;

                const u32 target = old_c + add;
                const u32 c = (target < max_c) ? target : max_c;

                if (final) bc.row(i) = running + c;
                running += c;
            });
            KOKKOS_PROFILE_FENCE();
            Kokkos::deep_copy(bc.size, Kokkos::subview(bc.row, g.n));
            Kokkos::fence();
        }
        // allocate rest
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "allocate");
            auto *us_ptr = (vertex_t *) get_chunk(mem_stack, sizeof(vertex_t) * bc.size);
            auto *ids_ptr = (partition_t *) get_chunk(mem_stack, sizeof(partition_t) * bc.size);
            auto *weights_ptr = (weight_t *) get_chunk(mem_stack, sizeof(weight_t) * bc.size);
            bc.us = UnmanagedDeviceVertex(us_ptr, bc.size);
            bc.ids = UnmanagedDevicePartition(ids_ptr, bc.size);
            bc.weights = UnmanagedDeviceWeight(weights_ptr, bc.size);
            Kokkos::deep_copy(bc.ids, partition.k);
            Kokkos::deep_copy(bc.weights, 0);
            KOKKOS_PROFILE_FENCE();
        }
        // set u values
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "fill_u");
            Kokkos::parallel_for("fill_u", g.n, KOKKOS_LAMBDA(const vertex_t u) {
                for (size_t j = bc.row(u); j < bc.row(u + 1); j++) {
                    bc.us(j) = u;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        // copy old values
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "copy_old_values");
            Kokkos::parallel_for("copy_old_values", old_bc.size, KOKKOS_LAMBDA(const u32 old_j) {
                vertex_t u = old_bc.us(old_j);
                if (needs_more(u) >= 1) { return; }

                u32 old_r_beg = old_bc.row(u);
                u32 new_r_beg = bc.row(u);
                u32 new_j = old_j - old_r_beg;

                bc.ids(new_r_beg + new_j) = old_bc.ids(old_j);
                bc.weights(new_r_beg + new_j) = old_bc.weights(old_j);
            });
            KOKKOS_PROFILE_FENCE();
        }
        // insert new values
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "insert_new_values");
            Kokkos::parallel_for("insert_new_values", g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = g.edges_u(i);
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);

                if (needs_more(u) == 0) { return; }

                partition_t v_id = partition.map(v);
                if (to_move(v) == 1) { v_id = unpack_partition(weight_id(v)); }

                u32 r_beg = bc.row(u);
                u32 r_end = bc.row(u + 1);
                u32 r_len = r_end - r_beg;

                u32 j = r_beg + hash32(v_id) % r_len;
                for (u32 t = 0; t < r_len; t++) {
                    if (j == r_end) { j = r_beg; }
                    partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), partition.k, v_id);
                    if (val == partition.k || val == v_id) {
                        Kokkos::atomic_add(&bc.weights(j), w);
                        break;
                    }
                    j += 1;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        // release old block connectivity
        {
            free_bc(old_bc, small_mem_stack);
        }
        BlockConnectivity true_bc;
        // allocate new on small mem stack
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "allocate_new_copy");

            auto *row_ptr = (u32 *) get_chunk(small_mem_stack, sizeof(u32) * (g.n + 1));
            auto *us_ptr = (vertex_t *) get_chunk(small_mem_stack, sizeof(vertex_t) * bc.size);
            auto *ids_ptr = (partition_t *) get_chunk(small_mem_stack, sizeof(partition_t) * bc.size);
            auto *weights_ptr = (weight_t *) get_chunk(small_mem_stack, sizeof(weight_t) * bc.size);
            true_bc.row = UnmanagedDeviceU32(row_ptr, g.n + 1);
            true_bc.us = UnmanagedDeviceVertex(us_ptr, bc.size);
            true_bc.ids = UnmanagedDevicePartition(ids_ptr, bc.size);
            true_bc.weights = UnmanagedDeviceWeight(weights_ptr, bc.size);
            true_bc.size = bc.size;
        }
        // copy to small mem stack
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "copy");
            Kokkos::deep_copy(true_bc.row, bc.row);
            Kokkos::deep_copy(true_bc.us, bc.us);
            Kokkos::deep_copy(true_bc.ids, bc.ids);
            Kokkos::deep_copy(true_bc.weights, bc.weights);
        }
        // free on mem stack
        {
            ScopedTimer _t("refine", "BlockConnectivity_re", "free_copy");
            pop(mem_stack);
            pop(mem_stack);
            pop(mem_stack);
            pop(mem_stack);
        }

        return true_bc;
    }

    inline void move(BlockConnectivity &bc,
                     const Graph &g,
                     const Partition &partition,
                     const UnmanagedDeviceU32 &to_move,
                     const UnmanagedDeviceU64 &weight_id,
                     KokkosMemoryStack &mem_stack,
                     KokkosMemoryStack &small_mem_stack) {
        // remove_weight
        {
            ScopedTimer _t("refine", "BlockConnectivity_move", "remove_weight");
            Kokkos::parallel_for("remove_weight", g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = g.edges_u(i);
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);
                if (to_move(u) == 0) { return; }

                partition_t u_id = partition.map(u);

                // search in v's neighborhood for u_id
                u32 r_beg = bc.row(v);
                u32 r_end = bc.row(v + 1);
                u32 r_len = r_end - r_beg;

                if (r_len == 0) { return; }

                u32 j = r_beg + hash32(u_id) % r_len;
                for (u32 t = 0; t < r_len; t++) {
                    if (j == r_end) { j = r_beg; }
                    if (bc.ids(j) == u_id) {
                        weight_t old_w = Kokkos::atomic_fetch_sub(&bc.weights(j), w);
                        if (old_w == w) {
                            bc.ids(j) = partition.k;
                            return;
                        }
                    }
                    j += 1;
                }
            });
            KOKKOS_PROFILE_FENCE();
        }
        UnmanagedDeviceU32 needs_more;
        // temp array to count
        {
            ScopedTimer _t("refine", "BlockConnectivity_move", "allocate_temp");
            auto *needs_more_ptr = (u32 *) get_chunk(mem_stack, sizeof(u32) * g.n); // on mem stack
            needs_more = UnmanagedDeviceU32(needs_more_ptr, g.n);
            Kokkos::deep_copy(needs_more, 0);
            KOKKOS_PROFILE_FENCE();
        }
        // add the connections
        {
            ScopedTimer _t("refine", "BlockConnectivity_move", "add_conn");
            Kokkos::parallel_for("add_conn", g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = g.edges_u(i);
                vertex_t v = g.edges_v(i);
                weight_t w = g.edges_w(i);
                if (to_move(u) == 0) { return; }

                partition_t new_u_id = unpack_partition(weight_id(u));

                // search in v's neighborhood for new_u_id
                u32 r_beg = bc.row(v);
                u32 r_end = bc.row(v + 1);
                u32 r_len = r_end - r_beg;

                if (r_len == 0) { return; }

                // first pass check if new_u_id exists anywhere
                u32 j = r_beg + hash32(new_u_id) % r_len;
                for (u32 t = 0; t < r_len; t++) {
                    if (j == r_end) { j = r_beg; }
                    if (bc.ids(j) == new_u_id) {
                        // found the spot, add the weight
                        Kokkos::atomic_add(&bc.weights(j), w);
                        return;
                    }
                    j += 1;
                }

                // new_u_id does not exist, now search for an empty spot
                j = r_beg + hash32(new_u_id) % r_len;
                for (u32 t = 0; t < r_len; t++) {
                    if (j == r_end) { j = r_beg; }
                    partition_t val = Kokkos::atomic_compare_exchange(&bc.ids(j), partition.k, new_u_id);
                    if (val == partition.k || val == new_u_id) {
                        // found empty spot or good spot, add the weight
                        Kokkos::atomic_add(&bc.weights(j), w);
                        return;
                    }
                    j += 1;
                }

                // could not insert in v, it needs more space
                Kokkos::atomic_inc(&needs_more(v));
            });
            KOKKOS_PROFILE_FENCE();
        }
        u32 sum = 0;
        // count how much more space is needed
        {
            ScopedTimer _t("refine", "BlockConnectivity_move", "sum");
            Kokkos::parallel_reduce("add_conn", g.n, KOKKOS_LAMBDA(const u32 i, u32 &update) {
                                        update += needs_more(i); // accumulate per-thread
                                    },
                                    sum);
            KOKKOS_PROFILE_FENCE();
        }

        if (sum > 0) {
            // not enough space, we need to rebuild
            bc = rebuild(bc, needs_more, g, partition, to_move, weight_id, mem_stack, small_mem_stack);
        }

        pop(mem_stack); // the needs_more array
    }

    struct HostBlockConnectivity {
        Kokkos::View<u32 *, Kokkos::HostSpace> row;
        Kokkos::View<vertex_t *, Kokkos::HostSpace> us;
        Kokkos::View<partition_t *, Kokkos::HostSpace> ids;
        Kokkos::View<weight_t *, Kokkos::HostSpace> weights;

        vertex_t n = 0;
        u32 size = 0;
        bool is_free = false;
    };

    inline HostBlockConnectivity download_block_connectivity(const BlockConnectivity &bc) {
        HostBlockConnectivity h_bc;

        // Copy metadata
        h_bc.n = bc.n;
        h_bc.size = bc.size;

        // Allocate host mirrors
        h_bc.row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.row);
        h_bc.us = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.us);
        h_bc.ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.ids);
        h_bc.weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc.weights);

        return h_bc;
    }

    /*
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
        assert(h_bc.n == n);
        assert(h_bc.row.extent(0) == static_cast<size_t>(n + 1));
        assert(h_bc.us.extent(0) == static_cast<size_t>(h_bc.size));
        assert(h_bc.ids.extent(0) == static_cast<size_t>(h_bc.size));
        assert(h_bc.weights.extent(0) == static_cast<size_t>(h_bc.size));
        assert(h_bc.row(0) == 0);
        for (vertex_t u = 0; u < n; ++u) {
            assert(h_bc.row(u) <= h_bc.row(u + 1));
        }
        assert(h_bc.row(n) == h_bc.size);

        // --- Per-vertex checks ---
        for (vertex_t u = 0; u < n; ++u) {
            const u32 r0 = h_bc.row(u);
            const u32 r1 = h_bc.row(u + 1);
            assert(r1 >= r0 && r1 <= h_bc.size);

            const u32 seg_len = r1 - r0;
            const u32 deg_u = static_cast<u32>(h_g.neighborhood(u + 1) - h_g.neighborhood(u));
            const u32 max_cap = std::min<u32>(deg_u, static_cast<u32>(k));
            assert(seg_len <= max_cap);

            // Build partition->weight map from u's neighborhood
            // (sum edge weights to each neighbor’s partition)
            std::unordered_map<partition_t, weight_t> neigh_pw;
            neigh_pw.reserve(std::min<u32>(deg_u, 32u)); // small initial bucket count

            for (vertex_t ei = h_g.neighborhood(u); ei < h_g.neighborhood(u + 1); ++ei) {
                const vertex_t v = h_g.edges_v(ei);
                const partition_t pid = h_part(v);
                assert(pid >= 0 && pid < k);
                neigh_pw[pid] += h_g.edges_w(ei);
            }

            // Row-level invariants
            std::unordered_set<partition_t> seen_ids;
            seen_ids.reserve(seg_len);

            weight_t row_sum = 0;

            for (u32 i = r0; i < r1; ++i) {
                const vertex_t uu = h_bc.us(i);
                const partition_t id = h_bc.ids(i);
                const weight_t w = h_bc.weights(i);

                // ownership stamp + basic ranges
                assert(uu == u);
                assert(w >= static_cast<weight_t>(0));

                // Allow sentinel id == k for unused slots; if present, weight must be 0
                if (id == k) {
                    assert(w == static_cast<weight_t>(0));
                    continue;
                }

                // Valid partition id
                assert(id >= 0 && id < k);

                // Unique per row
                const bool inserted = seen_ids.insert(id).second;
                assert(inserted && "duplicate partition id in row");

                // Must correspond to real neighborhood mass
                auto it = neigh_pw.find(id);
                assert(it != neigh_pw.end() && "bc contains id with no incident edges");
                assert(it->second == w && "bc weight != sum of incident edge weights for that partition");

                row_sum += w;
            }

            // Optional: sanity that row_sum doesn't exceed sum of all incident weights
            {
                weight_t neigh_sum = 0;
                for (const auto &kv: neigh_pw) neigh_sum += kv.second;
                assert(row_sum <= neigh_sum);
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
    */
}

#endif //GPU_HEIPA_BLOCK_CONNECTIVITY_H
