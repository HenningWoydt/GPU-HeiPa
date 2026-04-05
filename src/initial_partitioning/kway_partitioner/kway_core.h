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

#ifndef GPU_HEIPA_KWAY_CORE_H
#define GPU_HEIPA_KWAY_CORE_H

#include <cmath>
#include <numeric>

#include "kway_utility.h"
#include "kway_prio_queue.h"
#include "kway_indexed_prio_queue.h"
#include "kway_boundary.h"
#include "kway_graph.h"
#include "kway_block_conn.h"

#include "../../utility/definitions.h"
#include "../../datastructures/graph.h"


namespace GPU_HeiPa::ModifiedMetis {
    inline double ms_match = 0;
    inline double ms_coarse = 0;
    inline double ms_part = 0;
    inline double ms_refine = 0;

    inline void UpdateAdjacentVertexInfoAndBND(int v, int v_id, int u_id, int new_id, int e_weight, BlockConn &kway_mem, Boundary &bnd, int bndtype) {
        // Update global ID/ED and boundary
        if (v_id == u_id) {
            kway_mem.add_ext_w(v, e_weight);
            kway_mem.add_int_w(v, -e_weight);

            if (bndtype == BNDTYPE_REFINE) {
                if (kway_mem.get_ext_w(v) - kway_mem.get_int_w(v) >= 0 && !bnd.is_boundary(v)) { bnd.add(v); }
            } else {
                if (kway_mem.get_ext_w(v) > 0 && !bnd.is_boundary(v)) { bnd.add(v); }
            }
        } else if (v_id == new_id) {
            kway_mem.add_int_w(v, e_weight);
            kway_mem.add_ext_w(v, -e_weight);

            if (bndtype == BNDTYPE_REFINE) {
                if (kway_mem.get_ext_w(v) - kway_mem.get_int_w(v) < 0 && bnd.is_boundary(v)) { bnd.remove(v); }
            } else {
                if (kway_mem.get_ext_w(v) <= 0 && bnd.is_boundary(v)) { bnd.remove(v); }
            }
        }

        // Update edge weights using helper functions
        if (v_id != u_id) {
            kway_mem.update(v, u_id, -e_weight);
        }
        if (v_id != new_id) {
            kway_mem.update(v, new_id, e_weight);
        }
    }

    inline void UpdateMovedVertexInfoAndBND(int u, int u_id, int new_id, BlockConn &kway_mem, Boundary &bnd, std::vector<int> &partition, int bndtype) {
        partition[u] = new_id;

        int edto_to = kway_mem.get(u, new_id);
        kway_mem.add_ext_w(u, kway_mem.get_int_w(u) - edto_to);

        int tmp = kway_mem.get_int_w(u);
        kway_mem.set_int_w(u, edto_to);

        // Update: remove edge to 'to', add edge to 'from'
        kway_mem.update(u, new_id, -edto_to);
        kway_mem.update(u, u_id, tmp);

        // Update the boundary information. Both deletion and addition is
        // allowed as this routine can be used for moving arbitrary nodes.
        if (bndtype == BNDTYPE_REFINE) {
            if (bnd.is_boundary(u) && kway_mem.get_ext_w(u) - kway_mem.get_int_w(u) < 0) { bnd.remove(u); }
            if (!bnd.is_boundary(u) && kway_mem.get_ext_w(u) - kway_mem.get_int_w(u) >= 0) { bnd.add(u); }
        } else {
            if (bnd.is_boundary(u) && kway_mem.get_ext_w(u) <= 0) { bnd.remove(u); }
            if (!bnd.is_boundary(u) && kway_mem.get_ext_w(u) > 0) { bnd.add(u); }
        }
    }

    inline void Project2WayPartition(const Graph &g, BisectInfo &info, std::vector<int> &mapping, Boundary &bnd, const Boundary &c_bnd, std::vector<int> &partition, const std::vector<int> &c_partition) {
        for (int i = 0; i < g.n; ++i) {
            int j = mapping[i];
            partition[i] = c_partition[j];
            mapping[i] = c_bnd.idx[j];
        }

        for (int i = 0; i < g.n; ++i) {
            int tid = 0;
            int ted = 0;

            if (mapping[i] == -1) {
                for (int j = g.rows[i]; j < g.rows[i + 1]; ++j)
                    tid += g.edges_w[j];
            } else {
                int me = partition[i];
                for (int j = g.rows[i]; j < g.rows[i + 1]; ++j) {
                    if (me == partition[g.edges_v[j]])
                        tid += g.edges_w[j];
                    else
                        ted += g.edges_w[j];
                }
            }

            info.int_w[i] = tid;
            info.ext_w[i] = ted;

            if (ted > 0 || g.rows[i] == g.rows[i + 1]) { bnd.add(i); }
        }
    }

    inline void balance_bisection_boundary(Graph &g, BisectInfo &info, Boundary &bnd, std::vector<int> &p_weights, const std::vector<float> &ntpwgts, std::vector<int> &partition) {
        std::vector<int> moved(g.n, -1);
        std::vector<int> perm(g.n);

        int tpwgts[2];
        tpwgts[0] = (int) g.g_weight * ntpwgts[0];
        tpwgts[1] = g.g_weight - tpwgts[0];
        int mindiff = std::abs(tpwgts[0] - p_weights[0]);
        int from = p_weights[0] < tpwgts[0] ? 1 : 0;
        int to = (from + 1) % 2;

        PriorityQueue<int> queue(g.n);

        // permute
        for (int i = 0; i < bnd.curr_n; ++i) { perm[i] = i; }
        permute(bnd.curr_n, perm, bnd.curr_n / 5);

        for (int i = 0; i < bnd.curr_n; ++i) {
            const int v = bnd.val[perm[i]];
            if (partition[v] == from && g.v_weights[v] <= mindiff) {
                queue.insert(v, info.ext_w[v] - info.int_w[v]);
            }
        }

        int mincut = g.mincut;

        for (int nswaps = 0; nswaps < g.n; ++nswaps) {
            if (queue.empty()) { break; }
            const int higain = queue.top_pop();

            if (p_weights[to] + g.v_weights[higain] > tpwgts[to]) break;

            mincut -= (info.ext_w[higain] - info.int_w[higain]);

            p_weights[to] += g.v_weights[higain];
            p_weights[from] -= g.v_weights[higain];

            partition[higain] = to;
            moved[higain] = nswaps;

            std::swap(info.int_w[higain], info.ext_w[higain]);

            // If it is no longer a boundary vertex, remove it from boundary structures
            if (info.ext_w[higain] == 0 && g.rows[higain] < g.rows[higain + 1]) {
                bnd.remove(higain);
            }

            // Update neighbors
            for (int j = g.rows[higain]; j < g.rows[higain + 1]; ++j) {
                const int k = g.edges_v[j];
                const int kwgt = to == partition[k] ? g.edges_w[j] : -g.edges_w[j];

                info.int_w[k] += kwgt;
                info.ext_w[k] -= kwgt;

                if (bnd.is_boundary(k)) {
                    // k is currently on boundary
                    if (info.ext_w[k] == 0) {
                        // remove from boundary
                        bnd.remove(k);

                        if (moved[k] == -1 && partition[k] == from && g.v_weights[k] <= mindiff) {
                            queue.erase(k);
                        }
                    } else {
                        // still boundary, maybe update key
                        if (moved[k] == -1 && partition[k] == from && g.v_weights[k] <= mindiff) {
                            queue.update(k, info.ext_w[k] - info.int_w[k]);
                        }
                    }
                } else {
                    // k is not currently on boundary
                    if (info.ext_w[k] > 0) {
                        bnd.add(k);
                        if (moved[k] == -1 && partition[k] == from && g.v_weights[k] <= mindiff) {
                            queue.insert(k, info.ext_w[k] - info.int_w[k]);
                        }
                    }
                }
            }
        }

        g.mincut = mincut;
    }

    inline void balance_bisection_general(Graph &g, BisectInfo &info, Boundary &bnd, std::vector<int> &p_weights, const std::vector<float> &ntpwgts, std::vector<int> &partition) {
        std::vector<int> moved(g.n, -1);
        std::vector<int> perm(g.n);

        int tpwgts[2];
        tpwgts[0] = (int) g.g_weight * ntpwgts[0];
        tpwgts[1] = g.g_weight - tpwgts[0];
        int mindiff = abs(tpwgts[0] - p_weights[0]);
        int from = p_weights[0] < tpwgts[0] ? 1 : 0;
        int to = (from + 1) % 2;

        PriorityQueue<int> queue(g.n);

        for (int i = 0; i < g.n; ++i) { perm[i] = i; }
        permute(g.n, perm, g.n / 5);

        for (int ii = 0; ii < g.n; ii++) {
            int i = perm[ii];
            if (partition[i] == from && g.v_weights[i] <= mindiff) { queue.insert(i, info.ext_w[i] - info.int_w[i]); }
        }

        int mincut = g.mincut;
        for (int nswaps = 0; nswaps < g.n; nswaps++) {
            if (queue.empty()) { break; }
            int higain = queue.top_pop();

            if (p_weights[to] + g.v_weights[higain] > tpwgts[to])
                break;

            mincut -= info.ext_w[higain] - info.int_w[higain];
            p_weights[to] += g.v_weights[higain];
            p_weights[from] -= g.v_weights[higain];

            partition[higain] = to;
            moved[higain] = nswaps;


            std::swap(info.int_w[higain], info.ext_w[higain]);
            if (info.ext_w[higain] == 0 && bnd.is_boundary(higain) && g.rows[higain] < g.rows[higain + 1]) {
                bnd.remove(higain);
            }
            if (info.ext_w[higain] > 0 && !bnd.is_boundary(higain)) {
                bnd.add(higain);
            }

            for (int j = g.rows[higain]; j < g.rows[higain + 1]; j++) {
                int k = g.edges_v[j];

                int kwgt = to == partition[k] ? g.edges_w[j] : -g.edges_w[j];
                info.int_w[k] += kwgt;
                info.ext_w[k] -= kwgt;

                if (moved[k] == -1 && partition[k] == from && g.v_weights[k] <= mindiff) { queue.update(k, info.ext_w[k] - info.int_w[k]); }

                bool is_bnd = bnd.is_boundary(k);
                if (info.ext_w[k] == 0 && is_bnd) { bnd.remove(k); }
                if (info.ext_w[k] > 0 && !is_bnd) { bnd.add(k); }
            }
        }


        g.mincut = mincut;
    }

    inline int Match_2HopAny(const Graph &g, std::vector<int> &matching, std::vector<int> &mapping, int coarse_n, int &n_unmatched, int max_deg, const std::vector<int> &perm) {
        std::vector<int> colptr(g.n + 1, 0);
        for (int v = 0; v < g.n; v++) {
            if (matching[v] == UNMATCHED && g.rows[v + 1] - g.rows[v] < max_deg) {
                for (int j = g.rows[v]; j < g.rows[v + 1]; j++) { colptr[g.edges_v[j]]++; }
            }
        }
        for (int i = 1; i < g.n; i++) colptr[i] += colptr[i - 1];
        for (int i = g.n; i > 0; i--) colptr[i] = colptr[i - 1];
        colptr[0] = 0;

        std::vector<int> rowind(colptr[g.n]);
        for (int pi = 0; pi < g.n; pi++) {
            int i = perm[pi];
            if (matching[i] == UNMATCHED && g.rows[i + 1] - g.rows[i] < max_deg) {
                for (int j = g.rows[i]; j < g.rows[i + 1]; j++) { rowind[colptr[g.edges_v[j]]++] = i; }
            }
        }
        for (int i = g.n; i > 0; i--) colptr[i] = colptr[i - 1];
        colptr[0] = 0;


        for (int pi = 0; pi < g.n; pi++) {
            int i = perm[pi];
            if (colptr[i + 1] - colptr[i] < 2) { continue; }

            for (int jj = colptr[i + 1], j = colptr[i]; j < jj; j++) {
                if (matching[rowind[j]] == UNMATCHED) {
                    for (jj--; jj > j; jj--) {
                        if (matching[rowind[jj]] == UNMATCHED) {
                            mapping[rowind[j]] = mapping[rowind[jj]] = coarse_n++;
                            matching[rowind[j]] = rowind[jj];
                            matching[rowind[jj]] = rowind[j];
                            n_unmatched -= 2;
                            break;
                        }
                    }
                }
            }
        }

        return coarse_n;
    }

    inline int Match_2HopAll(const Graph &g, std::vector<int> &matching, std::vector<int> &mapping, int coarse_n, int &n_unmatched, int max_deg, const std::vector<int> &perm) {
        struct VertexKey {
            int key;
            int vertex;
        };

        int mask = INT32_MAX / max_deg;

        int ncand = 0;
        std::vector<VertexKey> keys(n_unmatched);
        for (int pi = 0; pi < g.n; pi++) {
            int i = perm[pi];
            int idegree = g.rows[i + 1] - g.rows[i];
            if (matching[i] == UNMATCHED && idegree > 1 && idegree < max_deg) {
                int k = 0;
                for (int j = g.rows[i]; j < g.rows[i + 1]; j++) { k += g.edges_v[j] % mask; }
                keys[ncand].vertex = i;
                keys[ncand].key = (k % mask) * max_deg + idegree;
                ncand++;
            }
        }
        qsort(keys.data(), ncand, sizeof(VertexKey), [](const void *a, const void *b) {
            const VertexKey *ia = (const VertexKey *) a;
            const VertexKey *ib = (const VertexKey *) b;
            return (ia->key > ib->key) - (ia->key < ib->key);
        });

        std::vector<int> mark(g.n, 0);


        for (int pi = 0; pi < ncand; pi++) {
            int i = keys[pi].vertex;
            if (matching[i] != UNMATCHED) { continue; }

            for (int j = g.rows[i]; j < g.rows[i + 1]; j++) { mark[g.edges_v[j]] = i; }

            for (int pk = pi + 1; pk < ncand; pk++) {
                int k = keys[pk].vertex;
                if (matching[k] != UNMATCHED) { continue; }

                if (keys[pi].key != keys[pk].key) { break; }
                if (g.rows[i + 1] - g.rows[i] != g.rows[k + 1] - g.rows[k]) { break; }

                int jj = g.rows[k];
                for (; jj < g.rows[k + 1]; jj++) {
                    if (mark[g.edges_v[jj]] != i) { break; }
                }
                if (jj == g.rows[k + 1]) {
                    mapping[i] = mapping[k] = coarse_n++;
                    matching[i] = k;
                    matching[k] = i;
                    n_unmatched -= 2;
                    break;
                }
            }
        }

        return coarse_n;
    }

    inline int Match_2Hop(const Graph &g, std::vector<int> &matching, std::vector<int> &mapping, int coarse_n, int n_unmatched, const std::vector<int> &perm) {
        coarse_n = Match_2HopAny(g, matching, mapping, coarse_n, n_unmatched, 2, perm);
        coarse_n = Match_2HopAll(g, matching, mapping, coarse_n, n_unmatched, 64, perm);

        if (n_unmatched > 1.5 * 0.10 * g.n) { coarse_n = Match_2HopAny(g, matching, mapping, coarse_n, n_unmatched, 3, perm); }
        if (n_unmatched > 2.0 * 0.10 * g.n) { coarse_n = Match_2HopAny(g, matching, mapping, coarse_n, n_unmatched, g.n, perm); }

        return coarse_n;
    }

    inline int heavy_edge_matching(const Graph &g, std::vector<int> &matching, std::vector<int> &mapping, int max_v_weight) {
        std::vector<int> perm(g.n);
        std::vector<int> tperm(g.n);
        std::vector<int> degrees(g.n);

        std::fill_n(matching.data(), g.n, UNMATCHED);

        for (int i = 0; i < g.n; ++i) { tperm[i] = i; }
        permute(g.n, tperm, g.n / 8);

        int avgdegree = 4.0 * (g.rows[g.n] / g.n);
        for (int i = 0; i < g.n; i++) {
            int bnum = std::sqrt(1 + g.rows[i + 1] - g.rows[i]);
            degrees[i] = (bnum > avgdegree ? avgdegree : bnum);
        }
        counting_sort(g.n, avgdegree, degrees, tperm, perm);

        int nunmatched = 0;
        int cnvtxs = 0, last_unmatched = 0;
        for (int pi = 0; pi < g.n; pi++) {
            int i = perm[pi];

            if (matching[i] == UNMATCHED) {
                int maxidx = i;

                if (g.v_weights[i] < max_v_weight) {
                    if (g.rows[i] == g.rows[i + 1]) {
                        last_unmatched = (pi > last_unmatched ? pi : last_unmatched) + 1;
                        for (; last_unmatched < g.n; last_unmatched++) {
                            int j = perm[last_unmatched];
                            if (matching[j] == UNMATCHED) {
                                maxidx = j;
                                break;
                            }
                        }
                    } else {
                        int maxwgt = -1;
                        for (int j = g.rows[i]; j < g.rows[i + 1]; j++) {
                            int k = g.edges_v[j];
                            if (matching[k] == UNMATCHED && maxwgt < g.edges_w[j] && g.v_weights[i] + g.v_weights[k] <= max_v_weight) {
                                maxidx = k;
                                maxwgt = g.edges_w[j];
                            }
                        }

                        if (maxidx == i && 2 * g.v_weights[i] < max_v_weight) {
                            nunmatched++;
                            maxidx = UNMATCHED;
                        }
                    }
                }

                if (maxidx != UNMATCHED) {
                    mapping[i] = mapping[maxidx] = cnvtxs++;
                    matching[i] = maxidx;
                    matching[maxidx] = i;
                }
            }
        }

        if (nunmatched > 0.10 * g.n) {
            cnvtxs = Match_2Hop(g, matching, mapping, cnvtxs, nunmatched, perm);
        }

        cnvtxs = 0;
        for (int i = 0; i < g.n; i++) {
            if (matching[i] == UNMATCHED) {
                matching[i] = i;
                mapping[i] = cnvtxs++;
            } else {
                if (i <= matching[i]) {
                    mapping[i] = mapping[matching[i]] = cnvtxs++;
                }
            }
        }

        return cnvtxs;
    }

    inline Graph coarse_graph(const Graph &g, const std::vector<int> &matching, const std::vector<int> &mapping, int coarse_n) {
        constexpr int mask = (1 << 13) - 1;

        Graph cgraph(coarse_n, g.m + 1);
        cgraph.mincut = -1;

        std::vector<int> htable(mask + 1, -1);
        std::vector<int> dtable(coarse_n, -1);

        int cvertex = 0;
        int cnedges = 0;
        cgraph.rows[0] = 0;

        for (int v = 0; v < g.n; ++v) {
            int u = matching[v];

            // Only process each matched pair once.
            if (u < v) {
                continue;
            }

            cgraph.v_weights[cvertex] = g.v_weights[v];
            if (v != u) {
                cgraph.v_weights[cvertex] += g.v_weights[u];
            }

            int degree_v = g.rows[v + 1] - g.rows[v];
            int degree_u = g.rows[u + 1] - g.rows[u];
            bool use_hash_table = (degree_v + degree_u) < (mask >> 2);

            int nedges = 0;
            int base = cnedges;

            if (use_hash_table) {
                htable[cvertex & mask] = 0;
                cgraph.edges_v[base + 0] = cvertex;
                nedges = 1;

                auto add_neighbors_hash = [&](int vertex) {
                    for (int j = g.rows[vertex]; j < g.rows[vertex + 1]; ++j) {
                        int coarse_neighbor = mapping[g.edges_v[j]];

                        int kk = coarse_neighbor & mask;
                        while (htable[kk] != -1 && cgraph.edges_v[base + htable[kk]] != coarse_neighbor) { kk = (kk + 1) & mask; }

                        int pos = htable[kk];
                        if (pos == -1) {
                            cgraph.edges_v[base + nedges] = coarse_neighbor;
                            cgraph.edges_w[base + nedges] = g.edges_w[j];
                            htable[kk] = nedges;
                            ++nedges;
                        } else {
                            cgraph.edges_w[base + pos] += g.edges_w[j];
                        }
                    }
                };

                add_neighbors_hash(v);
                if (v != u) {
                    add_neighbors_hash(u);
                }

                for (int j = nedges - 1; j >= 0; --j) {
                    int coarse_neighbor = cgraph.edges_v[base + j];
                    int kk = coarse_neighbor & mask;
                    while (cgraph.edges_v[base + htable[kk]] != coarse_neighbor) { kk = (kk + 1) & mask; }
                    htable[kk] = -1;
                }

                --nedges;
                cgraph.edges_v[base + 0] = cgraph.edges_v[base + nedges];
                cgraph.edges_w[base + 0] = cgraph.edges_w[base + nedges];
            } else {
                auto add_neighbors_direct = [&](int vertex) {
                    for (int j = g.rows[vertex]; j < g.rows[vertex + 1]; ++j) {
                        int coarse_neighbor = mapping[g.edges_v[j]];
                        int pos = dtable[coarse_neighbor];

                        if (pos == -1) {
                            cgraph.edges_v[base + nedges] = coarse_neighbor;
                            cgraph.edges_w[base + nedges] = g.edges_w[j];
                            dtable[coarse_neighbor] = nedges;
                            ++nedges;
                        } else {
                            cgraph.edges_w[base + pos] += g.edges_w[j];
                        }
                    }
                };

                add_neighbors_direct(v);

                if (v != u) {
                    add_neighbors_direct(u);

                    int self_pos = dtable[cvertex];
                    if (self_pos != -1) {
                        --nedges;
                        cgraph.edges_v[base + self_pos] = cgraph.edges_v[base + nedges];
                        cgraph.edges_w[base + self_pos] = cgraph.edges_w[base + nedges];
                        dtable[cvertex] = -1;
                    }
                }

                for (int j = 0; j < nedges; ++j) {
                    dtable[cgraph.edges_v[base + j]] = -1;
                }
            }

            cnedges += nedges;
            cgraph.rows[++cvertex] = cnedges;
        }

        cgraph.n = cvertex;
        cgraph.m = cnedges;
        cgraph.g_weight = g.g_weight;

        return cgraph;
    }

    inline void FM_2WayCutRefine(Graph &g, Boundary &bnd, BisectInfo &info, std::vector<int> &p_weights, const std::vector<float> &ntpwgts, std::vector<int> &partition, int max_iter) {
        std::vector<int> moved(g.n, -1);
        std::vector<int> swaps(g.n);
        std::vector<int> perm(g.n);

        int tpwgts[2];
        tpwgts[0] = g.g_weight * ntpwgts[0];
        tpwgts[1] = g.g_weight - tpwgts[0];

        int limit = (((0.01 * g.n) > (15) ? (0.01 * g.n) : (15)) < (100) ? ((0.01 * g.n) > (15) ? (0.01 * g.n) : (15)) : (100));
        int avgvwgt = (((p_weights[0] + p_weights[1]) / 20) < (2 * (p_weights[0] + p_weights[1]) / g.n) ? ((p_weights[0] + p_weights[1]) / 20) : (2 * (p_weights[0] + p_weights[1]) / g.n));

        PriorityQueue<int> queue0(g.n);
        PriorityQueue<int> queue1(g.n);
        PriorityQueue<int> *queues[2];
        queues[0] = &queue0;
        queues[1] = &queue1;

        int origdiff = abs(tpwgts[0] - p_weights[0]);

        for (int pass = 0; pass < max_iter; pass++) {
            queues[0]->reset();
            queues[1]->reset();

            int mincutorder = -1;
            int newcut = g.mincut;
            int mincut = g.mincut;
            int initcut = g.mincut;
            int mindiff = abs(tpwgts[0] - p_weights[0]);

            for (int i = 0; i < bnd.curr_n; ++i) { perm[i] = i; }
            permute(bnd.curr_n, perm, bnd.curr_n);

            for (int ii = 0; ii < bnd.curr_n; ii++) {
                int i = perm[ii];
                int v = bnd.val[i];
                queues[partition[v]]->insert(v, info.ext_w[v] - info.int_w[v]);
            }

            int nswaps = 0;
            for (; nswaps < g.n; nswaps++) {
                int from = (tpwgts[0] - p_weights[0] < tpwgts[1] - p_weights[1] ? 0 : 1);
                int to = (from + 1) % 2;

                if (queues[from]->empty()) { break; }
                int higain = queues[from]->top_pop();

                newcut -= (info.ext_w[higain] - info.int_w[higain]);
                p_weights[to] += g.v_weights[higain];
                p_weights[from] -= g.v_weights[higain];

                if ((newcut < mincut && abs(tpwgts[0] - p_weights[0]) <= origdiff + avgvwgt) ||
                    (newcut == mincut && abs(tpwgts[0] - p_weights[0]) < mindiff)) {
                    mincut = newcut;
                    mindiff = abs(tpwgts[0] - p_weights[0]);
                    mincutorder = nswaps;
                } else if (nswaps - mincutorder > limit) {
                    newcut += (info.ext_w[higain] - info.int_w[higain]);
                    p_weights[from] += g.v_weights[higain];
                    p_weights[to] -= g.v_weights[higain];
                    break;
                }

                partition[higain] = to;
                moved[higain] = nswaps;
                swaps[nswaps] = higain;

                std::swap(info.int_w[higain], info.ext_w[higain]);
                if (info.ext_w[higain] == 0 && g.rows[higain] < g.rows[higain + 1]) {
                    bnd.remove(higain);
                }

                for (int j = g.rows[higain]; j < g.rows[higain + 1]; j++) {
                    int k = g.edges_v[j];
                    int kwgt = (to == partition[k] ? g.edges_w[j] : -g.edges_w[j]);

                    info.int_w[k] += kwgt;
                    info.ext_w[k] -= kwgt;

                    if (bnd.is_boundary(k)) {
                        if (info.ext_w[k] == 0) {
                            bnd.remove(k);
                            if (moved[k] == -1) {
                                queues[partition[k]]->erase(k);
                            }
                        } else {
                            if (moved[k] == -1) {
                                queues[partition[k]]->update(k, info.ext_w[k] - info.int_w[k]);
                            }
                        }
                    } else {
                        if (info.ext_w[k] > 0) {
                            bnd.add(k);
                            if (moved[k] == -1) {
                                queues[partition[k]]->insert(k, info.ext_w[k] - info.int_w[k]);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < nswaps; i++) {
                moved[swaps[i]] = -1;
            }

            for (nswaps--; nswaps > mincutorder; nswaps--) {
                int higain = swaps[nswaps];

                int to = (partition[higain] + 1) % 2;
                partition[higain] = to;

                std::swap(info.int_w[higain], info.ext_w[higain]);

                if (info.ext_w[higain] == 0 && bnd.is_boundary(higain) && g.rows[higain] < g.rows[higain + 1]) {
                    bnd.remove(higain);
                } else if (info.ext_w[higain] > 0 && !bnd.is_boundary(higain)) {
                    bnd.add(higain);
                }

                p_weights[to] += g.v_weights[higain];
                p_weights[(to + 1) % 2] -= g.v_weights[higain];

                for (int j = g.rows[higain]; j < g.rows[higain + 1]; j++) {
                    int k = g.edges_v[j];
                    int kwgt = (to == partition[k] ? g.edges_w[j] : -g.edges_w[j]);

                    info.int_w[k] += kwgt;
                    info.ext_w[k] -= kwgt;

                    if (bnd.is_boundary(k) && info.ext_w[k] == 0) {
                        bnd.remove(k);
                    }
                    if (!bnd.is_boundary(k) && info.ext_w[k] > 0) {
                        bnd.add(k);
                    }
                }
            }

            g.mincut = mincut;

            if (mincutorder <= 0 || mincut == initcut) {
                break;
            }
        }
    }

    inline void ComputeKWayPartitionParams(Graph &g, Boundary &bnd, int k, std::vector<int> &p_weights, const std::vector<int> &partition, BlockConn &kway_mem) {
        std::fill_n(p_weights.data(), k, 0);
        bnd.reset();
        kway_mem.reset();

        int mincut = 0;

        for (int u = 0; u < g.n; ++u) {
            p_weights[partition[u]] += g.v_weights[u];
        }

        for (int u = 0; u < g.n; ++u) {
            const int me = partition[u];

            for (int j = g.rows[u]; j < g.rows[u + 1]; ++j) {
                if (me == partition[g.edges_v[j]])
                    kway_mem.add_int_w(u, g.edges_w[j]);
                else
                    kway_mem.add_ext_w(u, g.edges_w[j]);
            }

            if (kway_mem.get_ext_w(u) == 0) {
                continue;
            }

            mincut += kway_mem.get_ext_w(u);

            for (int j = g.rows[u]; j < g.rows[u + 1]; ++j) {
                int other = partition[g.edges_v[j]];
                if (me == other)
                    continue;

                kway_mem.update(u, other, g.edges_w[j]);
            }

            if (kway_mem.get_ext_w(u) >= kway_mem.get_int_w(u)) {
                bnd.add(u);
            }
        }

        g.mincut = mincut / 2;
    }

    inline void ProjectKWayPartition(Graph &g, Boundary &bnd, int k, std::vector<int> &mapping, std::vector<int> &partition, Graph &coarse_g, const std::vector<int> &c_partition, BlockConn &kway_mem) {
        std::vector<int> htable(k, -1);

        for (int u = 0; u < g.n; u++) {
            int v = mapping[u];
            partition[u] = c_partition[v];
            mapping[u] = kway_mem.get_temp_ext_w(v);
        }

        kway_mem.reset();

        coarse_g.resize(0, 0);
        bnd.reset();

        for (int i = 0; i < g.n; i++) {
            int istart = g.rows[i];
            int iend = g.rows[i + 1];

            if (mapping[i] == 0) {
                int tid = 0;
                for (int j = istart; j < iend; j++)
                    tid += g.edges_w[j];

                kway_mem.set_int_w(i, tid);
            } else {
                int me = partition[i];
                int tid = 0;
                int ted = 0;
                for (int j = istart; j < iend; j++) {
                    int other = partition[g.edges_v[j]];
                    if (me == other) {
                        tid += g.edges_w[j];
                    } else {
                        ted += g.edges_w[j];
                        kway_mem.update(i, other, g.edges_w[j]);
                    }
                }
                kway_mem.set_int_w(i, tid);
                kway_mem.set_ext_w(i, ted);


                if (ted > 0) {
                    if (ted - tid >= 0) {
                        bnd.add(i);
                    }

                    // Clear htable
                    for (int j = 0; j < kway_mem.get_n_conns(i); j++)
                        htable[kway_mem.get_conn_id(i, j)] = -1;
                }
            }
        }

        g.mincut = coarse_g.mincut;
    }

    inline void compute_boundary(const Graph &g, Boundary &bnd, int bndtype, const BlockConn &kway_mem) {
        bnd.reset();

        for (int u = 0; u < g.n; ++u) {
            if (kway_mem.get_ext_w(u) > 0 && (bndtype != BNDTYPE_REFINE || kway_mem.get_ext_w(u) >= kway_mem.get_int_w(u))) {
                bnd.add(u);
            }
        }
    }

    inline float ComputeLoadImbalance(const Graph &g, int k, const std::vector<int> &p_weights) {
        float max = 1.0;
        for (int j = 0; j < k; j++) {
            float cur = p_weights[j] * ((float) k / g.g_weight);
            if (cur > max)
                max = cur;
        }

        return max;
    }

    inline float ComputeLoadImbalanceDiff(const Graph &g, int k, const std::vector<int> &p_weights, float ubfactor) {
        float max = -1.0f;

        for (int j = 0; j < k; j++) {
            float cur = p_weights[j] * ((float) k / g.g_weight) - ubfactor;
            if (cur > max)
                max = cur;
        }

        return max;
    }

    inline void balance_bisection(Graph &g, Boundary &bnd, BisectInfo &info, const std::vector<float> &ntpwgts, std::vector<int> &p_weights, std::vector<int> &partition, float ubfactor) {
        if (ComputeLoadImbalanceDiff(g, 2, p_weights, ubfactor) <= 0) { return; }
        if (fabsf(ntpwgts[0] * g.g_weight - p_weights[0]) < 3 * g.g_weight / g.n) { return; }

        if (bnd.curr_n > 0) {
            balance_bisection_boundary(g, info, bnd, p_weights, ntpwgts, partition);
        } else {
            balance_bisection_general(g, info, bnd, p_weights, ntpwgts, partition);
        }
    }

    inline int is_balanced(const Graph &g, int k, const std::vector<int> &p_weights, float ubfactor, float ffactor) {
        return (ComputeLoadImbalanceDiff(g, k, p_weights, ubfactor) <= ffactor);
    }

    inline float ComputeLoadImbalanceDiffWeighted(const Graph &g, const std::vector<int> &p_weights, const std::vector<float> &tpwgts, float ubfactor) {
        float max = -1.0f;

        for (int i = 0; i < 2; i++) {
            float cur = p_weights[i] * (1.0f / (g.g_weight * tpwgts[i])) - ubfactor;
            if (cur > max)
                max = cur;
        }

        return max;
    }

    inline void Greedy_KWayCutOptimize(Graph &g, Boundary &bnd, int k, float imb, std::vector<int> &p_weights, std::vector<int> &partition, int max_iter, int mode, BlockConn &kway_mem) {
        int bndtype = (mode == OMODE_REFINE ? BNDTYPE_REFINE : BNDTYPE_BALANCE);

        if (mode != OMODE_BALANCE) {
            float curr_imb = ComputeLoadImbalance(g, k, p_weights);
            imb = (imb > curr_imb) ? imb : curr_imb;
        }

        int max_p_weight = (1.0f / k) * g.g_weight * imb;
        int min_p_weight = (1.0f / k) * g.g_weight * (1.0f / imb);

        std::vector<int> perm(g.n);

        std::vector<int> moved(g.n, -1);

        IndexedPriorityQueue<float> queue(g.n);

        for (int pass = 0; pass < max_iter; ++pass) {
            if (mode == OMODE_BALANCE) {
                bool stop = true;
                for (int i = 0; i < k; ++i) {
                    if (p_weights[i] > max_p_weight || p_weights[i] < min_p_weight) {
                        stop = false;
                        break;
                    }
                }
                if (stop) {
                    break;
                }
            }

            int oldcut = g.mincut;

            for (int i = 0; i < bnd.curr_n; ++i) {
                perm[i] = i;
            }
            permute(bnd.curr_n, perm, bnd.curr_n / 4);

            for (int i = 0; i < bnd.curr_n; ++i) {
                int u = bnd.val[perm[i]];
                float rgain = (kway_mem.get_n_conns(u) > 0 ? 1.0f * kway_mem.get_ext_w(u) / std::sqrt(static_cast<float>(kway_mem.get_n_conns(u))) : 0.0f) - kway_mem.get_int_w(u);

                queue.insert(u, rgain);
            }

            int nmoved = 0;

            while (!queue.empty()) {
                int u = queue.top_pop();
                moved[u] = pass;

                int u_id = partition[u];
                int new_id;
                int u_weight = g.v_weights[u];

                int jj;
                if (mode == OMODE_REFINE) {
                    for (jj = kway_mem.get_n_conns(u) - 1; jj >= 0; --jj) {
                        new_id = kway_mem.get_conn_id(u, jj);
                        if ((kway_mem.get_conn_w(u, jj) > kway_mem.get_int_w(u) && (p_weights[u_id] - u_weight >= min_p_weight ||
                                                                                    p_weights[new_id] < p_weights[u_id] - u_weight) &&
                             (p_weights[new_id] + u_weight <= max_p_weight ||
                              p_weights[new_id] < p_weights[u_id] - u_weight)) ||
                            (kway_mem.get_conn_w(u, jj) == kway_mem.get_int_w(u) &&
                             p_weights[new_id] < p_weights[u_id] - u_weight)) {
                            break;
                        }
                    }

                    if (jj < 0) {
                        continue;
                    }

                    for (int j = jj - 1; j >= 0; --j) {
                        new_id = kway_mem.get_conn_id(u, j);
                        if ((kway_mem.get_conn_w(u, j) > kway_mem.get_conn_w(u, jj) &&
                             (p_weights[u_id] - u_weight >= min_p_weight ||
                              p_weights[new_id] < p_weights[u_id] - u_weight) &&
                             (p_weights[new_id] + u_weight <= max_p_weight ||
                              p_weights[new_id] < p_weights[u_id] - u_weight)) ||
                            (kway_mem.get_conn_w(u, j) == kway_mem.get_conn_w(u, jj) &&
                             p_weights[new_id] < p_weights[kway_mem.get_conn_id(u, jj)])) {
                            jj = j;
                        }
                    }

                    new_id = kway_mem.get_conn_id(u, jj);
                } else {
                    for (jj = kway_mem.get_n_conns(u) - 1; jj >= 0; --jj) {
                        new_id = kway_mem.get_conn_id(u, jj);
                        if (u_id >= k || p_weights[new_id] < p_weights[u_id] - u_weight) {
                            break;
                        }
                    }

                    if (jj < 0) {
                        continue;
                    }

                    for (int j = jj - 1; j >= 0; --j) {
                        new_id = kway_mem.get_conn_id(u, j);
                        if (p_weights[new_id] < p_weights[kway_mem.get_conn_id(u, jj)]) {
                            jj = j;
                        }
                    }

                    new_id = kway_mem.get_conn_id(u, jj);
                }

                g.mincut -= kway_mem.get_conn_w(u, jj) - kway_mem.get_int_w(u);
                ++nmoved;

                p_weights[new_id] += u_weight;
                p_weights[u_id] -= u_weight;

                UpdateMovedVertexInfoAndBND(u, u_id, new_id, kway_mem, bnd, partition, bndtype);

                for (int j = g.rows[u]; j < g.rows[u + 1]; ++j) {
                    int v = g.edges_v[j];
                    int e_weight = g.edges_w[j];
                    int v_id = partition[v];

                    int oldnnbrs = kway_mem.get_n_conns(v);

                    UpdateAdjacentVertexInfoAndBND(v, v_id, u_id, new_id, e_weight, kway_mem, bnd, bndtype);

                    if (!(v_id == new_id || v_id == u_id || oldnnbrs != kway_mem.get_n_conns(v))) { continue; }

                    if (moved[v] == pass) { continue; }

                    float rgain = (kway_mem.get_n_conns(v) > 0 ? 1.0f * kway_mem.get_ext_w(v) / std::sqrt(static_cast<float>(kway_mem.get_n_conns(v))) : 0.0f) - kway_mem.get_int_w(v);

                    if (bndtype == BNDTYPE_REFINE) {
                        if (queue.contains(v)) {
                            if (kway_mem.get_ext_w(v) - kway_mem.get_int_w(v) >= 0) {
                                queue.update(v, rgain);
                            } else {
                                queue.erase(v);
                            }
                        } else if (kway_mem.get_ext_w(v) - kway_mem.get_int_w(v) >= 0) {
                            queue.insert(v, rgain);
                        }
                    } else {
                        if (queue.contains(v)) {
                            if (kway_mem.get_ext_w(v) > 0) {
                                queue.update(v, rgain);
                            } else {
                                queue.erase(v);
                            }
                        } else if (kway_mem.get_ext_w(v) > 0) {
                            queue.insert(v, rgain);
                        }
                    }
                }
            }

            if (nmoved == 0 || (mode == OMODE_REFINE && g.mincut == oldcut)) {
                break;
            }
        }
    }

    inline void Compute2WayPartitionParams(Graph &g, Boundary &bnd, std::vector<int> &p_weights, const std::vector<int> &partition, BisectInfo &info) {
        std::fill_n(p_weights.data(), 2, 0);
        bnd.reset();
        int mincut = 0;

        for (int i = 0; i < g.n; ++i)
            p_weights[partition[i]] += g.v_weights[i];

        for (int i = 0; i < g.n; ++i) {
            int istart = g.rows[i];
            int iend = g.rows[i + 1];
            int me = partition[i];
            int tid = 0, ted = 0;

            for (int j = istart; j < iend; ++j) {
                if (me == partition[g.edges_v[j]])
                    tid += g.edges_w[j];
                else
                    ted += g.edges_w[j];
            }

            info.int_w[i] = tid;
            info.ext_w[i] = ted;

            if (ted > 0 || istart == iend) {
                bnd.add(i);
                mincut += ted;
            }
        }

        g.mincut = mincut / 2;
    }

    inline void bisection(Graph &g, Boundary &bnd, float ubfactor, const std::vector<float> &ntpwgts, std::vector<int> &p_weights, std::vector<int> &partition, BisectInfo &info, int max_tries, int max_iter) {
        std::vector<int> temp_partition(g.n);
        std::vector<int> queue(g.n);
        std::vector<int> touched(g.n);

        int max_p_weight = ubfactor * g.g_weight * ntpwgts[1];
        int min_p_weight = (1.0 / ubfactor) * g.g_weight * ntpwgts[1];

        int bestcut = 0;
        for (int try_i = 0; try_i < max_tries; try_i++) {
            std::fill_n(temp_partition.data(), g.n, 1);

            std::fill(touched.begin(), touched.end(), 0);

            int pwgts[2];
            pwgts[1] = g.g_weight;
            pwgts[0] = 0;

            queue[0] = rand() % g.n;
            touched[queue[0]] = 1;

            int first = 0;
            int last = 1;
            int nleft = g.n - 1;
            int drain = 0;

            while (true) {
                int v = 0;
                if (first == last) {
                    if (nleft == 0 || drain) { break; }

                    int k = rand() % nleft;
                    for (v = 0; v < g.n; v++) {
                        if (touched[v] == 0) {
                            if (k == 0) { break; }
                            k--;
                        }
                    }

                    queue[0] = v;
                    touched[v] = 1;
                    first = 0;
                    last = 1;
                    nleft--;
                }

                v = queue[first++];
                if (pwgts[0] > 0 && pwgts[1] - g.v_weights[v] < min_p_weight) {
                    drain = 1;
                    continue;
                }

                temp_partition[v] = 0;

                pwgts[0] += g.v_weights[v];
                pwgts[1] -= g.v_weights[v];

                if (pwgts[1] <= max_p_weight) { break; }

                drain = 0;
                for (int j = g.rows[v]; j < g.rows[v + 1]; j++) {
                    int u = g.edges_v[j];
                    if (touched[u] == 0) {
                        queue[last++] = u;
                        touched[u] = 1;
                        nleft--;
                    }
                }
            }

            if (pwgts[1] == 0) { temp_partition[(rand() % g.n)] = 1; }
            if (pwgts[0] == 0) { temp_partition[(rand() % g.n)] = 0; }

            Compute2WayPartitionParams(g, bnd, p_weights, temp_partition, info);

            balance_bisection(g, bnd, info, ntpwgts, p_weights, temp_partition, ubfactor);

            FM_2WayCutRefine(g, bnd, info, p_weights, ntpwgts, temp_partition, max_iter);

            if (try_i == 0 || bestcut > g.mincut) {
                bestcut = g.mincut;
                std::copy_n(temp_partition.data(), g.n, partition.data());

                if (bestcut == 0) { break; }
            }
        }

        g.mincut = bestcut;
    }

    inline void split_graph(const Graph &g, const Boundary &bnd, const std::vector<int> &label, const std::vector<int> &partition, Graph &l_graph, std::vector<int> &l_label, Graph &r_graph, std::vector<int> &r_label) {
        int ns[2] = {0, 0};
        int ms[2] = {0, 0};
        int ws[2] = {0, 0};
        int *sxadj[2];
        int *svwgt[2];
        int *sadjncy[2];
        int *sadjwgt[2];
        std::vector<int> rename(g.n);

        for (int i = 0; i < g.n; ++i) {
            int part = partition[i];
            rename[i] = ns[part]++;
            ws[part] += g.v_weights[i];
            ms[part] += g.rows[i + 1] - g.rows[i]; // this overestimates
        }

        l_graph.resize(ns[0], ms[0], ws[0]);
        r_graph.resize(ns[1], ms[1], ws[1]);

        l_label.resize(ns[0]);
        r_label.resize(ns[1]);

        sxadj[0] = l_graph.rows.data();
        sxadj[1] = r_graph.rows.data();
        svwgt[0] = l_graph.v_weights.data();
        svwgt[1] = r_graph.v_weights.data();
        sadjncy[0] = l_graph.edges_v.data();
        sadjncy[1] = r_graph.edges_v.data();
        sadjwgt[0] = l_graph.edges_w.data();
        sadjwgt[1] = r_graph.edges_w.data();

        ns[0] = ns[1] = ms[0] = ms[1] = 0;
        sxadj[0][0] = sxadj[1][0] = 0;

        for (int i = 0; i < g.n; ++i) {
            int part = partition[i];

            if (!bnd.is_boundary(i)) {
                int *auxadjncy = sadjncy[part] + ms[part] - g.rows[i];
                int *auxadjwgt = sadjwgt[part] + ms[part] - g.rows[i];

                for (int j = g.rows[i]; j < g.rows[i + 1]; ++j) {
                    auxadjncy[j] = g.edges_v[j];
                    auxadjwgt[j] = g.edges_w[j];
                }
                ms[part] += g.rows[i + 1] - g.rows[i];
            } else {
                int *auxadjncy = sadjncy[part];
                int *auxadjwgt = sadjwgt[part];
                int l = ms[part];

                for (int j = g.rows[i]; j < g.rows[i + 1]; ++j) {
                    int k = g.edges_v[j];
                    if (partition[k] == part) {
                        auxadjncy[l] = k;
                        auxadjwgt[l++] = g.edges_w[j];
                    }
                }
                ms[part] = l;
            }

            svwgt[part][ns[part]] = g.v_weights[i];
            if (part == 0)
                l_label[ns[part]] = label[i];
            else
                r_label[ns[part]] = label[i];
            sxadj[part][++ns[part]] = ms[part];
        }

        for (int part = 0; part < 2; ++part) {
            int iend = sxadj[part][ns[part]];
            for (int i = 0; i < iend; ++i)
                sadjncy[part][i] = rename[sadjncy[part][i]];
        }

        l_graph.m = ms[0];
        r_graph.m = ms[1];
    }

    inline void multilevel_bisection(Graph &g, Boundary &bnd, std::vector<int> &partition, const std::vector<float> &tpwgts, int max_tries, int threshold, float ubfactor, int max_iter, Boundary &temp_bnd, std::vector<int> &temp_partition) {
        int bestobj = 0, curobj = 0;
        float bestbal = 0.0, curbal = 0.0;

        if (g.n == 0) { return; }

        std::vector<int> pwgts(2);
        BisectInfo info(g.n);
        std::vector<int> match(g.n);

        int *const where1_orig = partition.data();

        std::vector<int> bestwhere;
        if (max_tries > 1) {
            bestwhere.resize(g.n);
        }

        for (int cut = 0; cut < max_tries; cut++) {
            std::vector<Graph> graph_stack;
            std::vector<std::vector<int> > cmap_stack;

            graph_stack.push_back(g);

            int maxvwgt = 1.5 * graph_stack.back().g_weight / threshold;

            do {
                Graph &fine_g = graph_stack.back();

                cmap_stack.emplace_back(fine_g.n);
                std::vector<int> &cmap = cmap_stack.back();

                int cnvtxs = heavy_edge_matching(fine_g, match, cmap, maxvwgt);
                Graph coarse_g = coarse_graph(fine_g, match, cmap, cnvtxs);

                graph_stack.push_back(std::move(coarse_g));
            } while (
                graph_stack.back().n > threshold &&
                graph_stack.back().n < 0.85 * graph_stack[graph_stack.size() - 2].n &&
                graph_stack.back().m > graph_stack.back().n / 2
            );

            const int nlevels = (int) graph_stack.size() - 1;

            int niparts = 2; // graph_stack.back().n <= threshold ? 5 : 7;
            bnd.curr_n = 0;

            bisection(graph_stack.back(), bnd, ubfactor, tpwgts, pwgts, partition, info, niparts, max_iter);
            Compute2WayPartitionParams(graph_stack.back(), bnd, pwgts, partition, info);

            for (int level = nlevels; level > 0; --level) {
                Graph &coarse_g = graph_stack[level];
                std::vector<int> &cmap = cmap_stack[level - 1];

                balance_bisection(coarse_g, bnd, info, tpwgts, pwgts, partition, ubfactor);
                FM_2WayCutRefine(coarse_g, bnd, info, pwgts, tpwgts, partition, max_iter);

                Graph &fine_g = graph_stack[level - 1];
                temp_bnd.reset();
                Project2WayPartition(fine_g, info, cmap, temp_bnd, bnd, temp_partition, partition);
                fine_g.mincut = coarse_g.mincut;

                std::swap(bnd, temp_bnd);
                std::swap(partition, temp_partition);
            }

            balance_bisection(graph_stack[0], bnd, info, tpwgts, pwgts, partition, ubfactor);
            FM_2WayCutRefine(graph_stack[0], bnd, info, pwgts, tpwgts, partition, max_iter);

            g.mincut = graph_stack[0].mincut;

            curobj = g.mincut;
            curbal = ComputeLoadImbalanceDiffWeighted(g, pwgts, tpwgts, ubfactor);

            if (cut == 0 || (curbal <= 0.0005 && bestobj > curobj) || (bestbal > 0.0005 && curbal < bestbal)) {
                bestobj = curobj;
                bestbal = curbal;
                if (cut < max_tries - 1) {
                    std::copy_n(partition.data(), g.n, bestwhere.data());
                }
            }

            if (bestobj == 0) {
                break;
            }
        }

        if (bestobj != curobj) {
            std::copy_n(bestwhere.data(), g.n, partition.data());
            Compute2WayPartitionParams(g, bnd, pwgts, partition, info);
        }

        if (partition.data() != where1_orig) {
            std::copy_n(partition.data(), g.n, where1_orig);
        }
    }

    inline void recursive_bisection(Graph &g, Boundary &bnd, int k, const std::vector<int> &label, std::vector<int> &partition, float *tpwgts, int fpart, int max_tries, int threshold, float imb, int max_iter, Boundary &temp_bnd, std::vector<int> &temp_partition_1, std::vector<int> &temp_partition_2) {
        std::vector<float> tpwgts2(2);
        tpwgts2[0] = std::accumulate(tpwgts, tpwgts + (k >> 1), 0.0f);
        tpwgts2[1] = 1.0 - tpwgts2[0];

        multilevel_bisection(g, bnd, temp_partition_1, tpwgts2, max_tries, threshold, imb, max_iter, temp_bnd, temp_partition_2);

        for (int i = 0; i < g.n; i++) { partition[label[i]] = temp_partition_1[i] + fpart; }

        Graph lgraph;
        Graph rgraph;

        std::vector<int> l_label;
        std::vector<int> r_label;

        if (k > 2) { split_graph(g, bnd, label, temp_partition_1, lgraph, l_label, rgraph, r_label); }

        float wsum = std::accumulate(tpwgts, tpwgts + (k >> 1), 0.0f);
        std::transform(tpwgts, tpwgts + (k >> 1), tpwgts, [wsum](float val) { return val / wsum; });
        std::transform(tpwgts + (k >> 1), tpwgts + k, tpwgts + (k >> 1), [wsum](float val) { return val / (1.0f - wsum); });

        if (k > 3) {
            recursive_bisection(lgraph, bnd, (k >> 1), l_label, partition, tpwgts, fpart, max_tries, threshold, imb, max_iter, temp_bnd, temp_partition_1, temp_partition_2);
            recursive_bisection(rgraph, bnd, k - (k >> 1), r_label, partition, tpwgts + (k >> 1) * 1, fpart + (k >> 1), max_tries, threshold, imb, max_iter, temp_bnd, temp_partition_1, temp_partition_2);
        } else if (k == 3) {
            recursive_bisection(rgraph, bnd, k - (k >> 1), r_label, partition, tpwgts + (k >> 1) * 1, fpart + (k >> 1), max_tries, threshold, imb, max_iter, temp_bnd, temp_partition_1, temp_partition_2);
        }
    }

    inline int recursive_bisection_head(Graph &g, int k, float imb, int max_tries, std::vector<int> &partition) {
        std::fill_n(partition.data(), g.n, 0);

        if (k == 1) { return 0; }

        srand(4321);

        imb += 0.0000499f;

        std::vector<float> tpwgts(k, 1.0f / k);

        std::vector<int> label(g.n);
        for (int i = 0; i < g.n; i++) { label[i] = i; }

        Boundary bnd(g.n);
        Boundary temp_bnd(g.n);
        std::vector<int> temp_partition_1(g.n);
        std::vector<int> temp_partition_2(g.n);

        recursive_bisection(g, bnd, k, label, partition, tpwgts.data(), 0, max_tries, 50, imb, 3, temp_bnd, temp_partition_1, temp_partition_2);

        return 1;
    }

    inline int multilevel_partition(Graph &g, int k, float imb, std::vector<int> &partition, int threshold, int n_init_part, int max_tries, int max_iter) {
        int curobj = 0, bestobj = 0;
        float curbal = 0.0f, bestbal = 0.0f;

        Boundary bnd(g.n);

        std::vector<int> pwgts(k);
        std::vector<int> where1(g.n);
        std::vector<int> where2(g.n);
        std::vector<int> match(g.n);

        std::vector<Graph> graphs;
        std::vector<std::vector<int> > mappings;

        BlockConn kway_mem(g.n, k);

        for (int cut = 0; cut < max_tries; cut++) {
            graphs.clear();
            mappings.clear();

            graphs.push_back(g);

            int maxvwgt = 1.5 * graphs.back().g_weight / threshold;

            do {
                Graph &fine_g = graphs.back();

                mappings.emplace_back(fine_g.n);
                std::vector<int> &mapping = mappings.back();

                auto sp_match = get_time_point();
                int cnvtxs = heavy_edge_matching(fine_g, match, mapping, maxvwgt);
                auto ep_match = get_time_point();

                auto sp_coarse = get_time_point();
                Graph coarse_g = coarse_graph(fine_g, match, mapping, cnvtxs);
                auto ep_coarse = get_time_point();

                ms_match += get_milli_seconds(sp_match, ep_match);
                ms_coarse += get_milli_seconds(sp_coarse, ep_coarse);

                graphs.push_back(std::move(coarse_g));
            } while (graphs.back().n > threshold && graphs.back().n < 0.85 * graphs[graphs.size() - 2].n && graphs.back().m > graphs.back().n / 2);

            float temp_ubfactor = (float) pow(imb, 1.0 / log(k));
            auto sp_part = get_time_point();
            recursive_bisection_head(graphs.back(), k, temp_ubfactor, n_init_part, where1);
            auto ep_part = get_time_point();
            ms_part += get_milli_seconds(sp_part, ep_part);

            const int nlevels = (int) mappings.size();

            bnd.curr_n = 0;
            ComputeKWayPartitionParams(graphs.back(), bnd, k, pwgts, where1, kway_mem);

            int coarse_level = nlevels;
            for (int pass = 0; coarse_level >= 0; pass++) {
                Graph &cgraph = graphs[coarse_level];

                auto sp_refine = get_time_point();
                if (2 * pass >= nlevels && !is_balanced(cgraph, k, pwgts, imb, .02f)) {
                    compute_boundary(cgraph, bnd, BNDTYPE_BALANCE, kway_mem);
                    Greedy_KWayCutOptimize(cgraph, bnd, k, imb, pwgts, where1, 1, OMODE_BALANCE, kway_mem);
                    compute_boundary(cgraph, bnd, BNDTYPE_REFINE, kway_mem);
                }

                Greedy_KWayCutOptimize(cgraph, bnd, k, imb, pwgts, where1, max_iter, OMODE_REFINE, kway_mem);
                auto ep_refine = get_time_point();
                ms_refine += get_milli_seconds(sp_refine, ep_refine);

                if (coarse_level > 0) {
                    Graph &fine_g = graphs[coarse_level - 1];
                    std::vector<int> &mapping = mappings[coarse_level - 1];

                    kway_mem.swap_infos();

                    ProjectKWayPartition(fine_g, bnd, k, mapping, where2, cgraph, where1, kway_mem);

                    std::swap(where1, where2);
                }

                --coarse_level;
            }

            g.mincut = graphs[0].mincut;

            if (!is_balanced(graphs[0], k, pwgts, imb, 0.0f)) {
                compute_boundary(graphs[0], bnd, BNDTYPE_BALANCE, kway_mem);
                Greedy_KWayCutOptimize(graphs[0], bnd, k, imb, pwgts, where1, 3, OMODE_BALANCE, kway_mem);
                compute_boundary(graphs[0], bnd, BNDTYPE_REFINE, kway_mem);
                Greedy_KWayCutOptimize(graphs[0], bnd, k, imb, pwgts, where1, max_iter, OMODE_REFINE, kway_mem);
            }

            g.mincut = graphs[0].mincut;

            curobj = g.mincut;
            curbal = ComputeLoadImbalanceDiff(g, k, pwgts, imb);

            if (cut == 0 || (curbal <= 0.0005f && bestobj > curobj) || (bestbal > 0.0005f && curbal < bestbal)) {
                std::copy_n(where1.data(), g.n, partition.data());
                bestobj = curobj;
                bestbal = curbal;
            }

            if (bestobj == 0)
                break;
        }

        return bestobj;
    }

    inline void partition(Graph &g, int k, float imb, int seed, std::vector<int> &partition) {
        std::fill_n(partition.data(), g.n, 0);
        if (k == 1) { return; }

        int ufactor = static_cast<int>(imb * 1000.0f);
        float ubfactor = 1.0f + 0.001f * ufactor;
        ubfactor += 0.0000499f;

        srand((unsigned int) seed);

        int coarsenCandidate = static_cast<int>((double) g.n / (40.0 * log2(k)));
        int c_threshold = (coarsenCandidate > 30 * k) ? coarsenCandidate : 30 * k;

        int n_init_part = 1; // c_threshold == 30 * k ? 4 : 5;

        multilevel_partition(g, k, ubfactor, partition, c_threshold, n_init_part, 1, 3);
    }
}


namespace GPU_HeiPa {
    inline void kway_partition(HostGraph &g,
                               int k,
                               f64 imbalance,
                               u64 seed,
                               HostPartition &partition) {
        ModifiedMetis::Graph temp_g(g.n, g.m, g.g_weight);
        for (vertex_t u = 0; u < g.n; ++u) {
            temp_g.rows[u] = (int) g.neighborhood(u);
            temp_g.v_weights[u] = (int) g.weights(u);
            for (size_t i = g.neighborhood(u); i < g.neighborhood(u + 1); ++i) {
                temp_g.edges_v[i] = (int) g.edges_v(i);
                temp_g.edges_w[i] = (int) g.edges_w(i);
            }
        }
        temp_g.rows[g.n] = (int) g.neighborhood(g.n);

        std::vector<int> temp(g.n);

        ModifiedMetis::partition(temp_g, k, (float) imbalance, seed, temp);

        for (vertex_t u = 0; u < g.n; u++) {
            partition(u) = temp[u];
        }
    }

    inline void kway_partition(Graph &g,
                               int k,
                               f64 imbalance,
                               u64 seed,
                               Partition &partition,
                               DeviceExecutionSpace &exec_space) {
        // Convert device graph to simple CSR arrays on host
        HostGraph host_g = to_host_graph(g, exec_space);
        HostPartition host_partition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), g.n);

        kway_partition(host_g, k, imbalance, seed, host_partition);

        auto device_subview = Kokkos::subview(partition.map, std::pair<size_t, size_t>(0, host_partition.extent(0)));
        Kokkos::deep_copy(device_subview, host_partition);
        Kokkos::fence();
    }
}


#endif
