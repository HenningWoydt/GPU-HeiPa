import sys
import math
import random
import time
import numpy as np


# --- MINIMAL CSR GRAPH ---
class CSRGraph:
    def __init__(self, n, vw, xadj, adjncy, adjwgt):
        self.n = n
        self.vw = vw.astype(np.int32)
        self.xadj = xadj.astype(np.int32)
        self.adjncy = adjncy.astype(np.int32)
        self.adjwgt = adjwgt.astype(np.int32)
        self.total_weight = np.sum(vw)


def read_graph_csr(filename):
    t0 = time.time()
    with open(filename, 'r') as f:
        lines = f.readlines()
    line_idx = 0
    while line_idx < len(lines) and (lines[line_idx].startswith('%') or not lines[line_idx].strip()):
        line_idx += 1
    if line_idx >= len(lines): return None
    header = lines[line_idx].split()
    line_idx += 1
    n = int(header[0])
    fmt = header[2] if len(header) > 2 else "000"
    has_vw, has_ew = (fmt[-2] if len(fmt) >= 2 else '0') == '1', (fmt[-1] if len(fmt) >= 1 else '0') == '1'
    vw, xadj = np.ones(n, dtype=np.int32), np.zeros(n + 1, dtype=np.int32)
    adjncy_list, adjwgt_list, curr_m = [], [], 0
    for i in range(n):
        while line_idx < len(lines) and (lines[line_idx].startswith('%') or not lines[line_idx].strip()): line_idx += 1
        if line_idx >= len(lines): break
        parts = [int(x) for x in lines[line_idx].split()]
        line_idx += 1
        if has_vw: vw[i] = parts.pop(0)
        xadj[i] = curr_m
        if has_ew:
            for j in range(0, len(parts), 2):
                adjncy_list.append(parts[j] - 1);
                adjwgt_list.append(parts[j + 1]);
                curr_m += 1
        else:
            for v in parts: adjncy_list.append(v - 1); adjwgt_list.append(1); curr_m += 1
    xadj[n] = curr_m
    g = CSRGraph(n, vw, xadj, np.array(adjncy_list, dtype=np.int32), np.array(adjwgt_list, dtype=np.int32))
    return g, time.time() - t0


# --- COARSENING (SIMPLIFIED) ---
def coarsen_simple(graph, max_vw):
    n, mapping, cn, order = graph.n, np.full(graph.n, -1, dtype=np.int32), 0, np.random.permutation(graph.n)
    for u in order:
        if mapping[u] != -1: continue
        best_v, max_w = -1, -1
        # Constraint: no coarse vertex should exceed max_vw (lmax)
        for i in range(graph.xadj[u], graph.xadj[u + 1]):
            v, w = graph.adjncy[i], graph.adjwgt[i]
            if mapping[v] == -1 and graph.vw[u] + graph.vw[v] <= max_vw:
                if w > max_w: max_w, best_v = w, v
        mapping[u] = cn
        if best_v != -1: mapping[best_v] = cn
        cn += 1
    cvw = np.bincount(mapping, weights=graph.vw, minlength=cn).astype(np.int32)
    edges = {}
    for u in range(n):
        uc = mapping[u]
        for i in range(graph.xadj[u], graph.xadj[u + 1]):
            vc = mapping[graph.adjncy[i]]
            if uc == vc: continue
            pair = (uc, vc) if uc < vc else (vc, uc)
            edges[pair] = edges.get(pair, 0) + graph.adjwgt[i]
    cxadj, cadjncy, cadjwgt = np.zeros(cn + 1, dtype=np.int32), [], []
    adj_lists = [[] for _ in range(cn)]
    for (u, v), w in edges.items():
        adj_lists[u].append((v, w));
        adj_lists[v].append((u, w))
    curr_m = 0
    for i in range(cn):
        cxadj[i] = curr_m
        for v, w in adj_lists[i]:
            cadjncy.append(v);
            cadjwgt.append(w);
            curr_m += 1
    cxadj[cn] = curr_m
    return CSRGraph(cn, cvw, cxadj, np.array(cadjncy, dtype=np.int32), np.array(cadjwgt, dtype=np.int32)), mapping


# --- BISECTION ---
P_CACHE = {}


def get_p_matrix(n):
    if n <= 1 or n > 20: return None
    if n not in P_CACHE:
        num_combos = 1 << (n - 1)
        P = np.zeros((num_combos, n), dtype=np.int8)
        P[:, 1:] = (np.arange(num_combos)[:, None] >> np.arange(n - 1)) & 1
        P_CACHE[n] = P
    return P_CACHE[n]


def brute_force_bisect(graph, lmax_l, lmax_r):
    n, vw = graph.n, graph.vw
    if n <= 1:
        side = 0 if lmax_l >= lmax_r else 1
        return np.array([side], dtype=np.int8), (vw[0] if side == 0 else 0, 0 if side == 0 else vw[0])
    if n > 20:  # Greedy fallback
        idx = np.argsort(vw)[::-1]
        part, wl = np.zeros(n, dtype=np.int8), 0
        for i in idx:
            if wl + vw[i] <= lmax_l:
                part[i], wl = 0, wl + vw[i]
            else:
                part[i] = 1
        return part, (int(wl), int(np.sum(vw) - wl))
    P = get_p_matrix(n)
    vw_f = vw.astype(np.float64)
    wr, wl = P @ vw_f, vw_f.sum() - (P @ vw_f)
    imb = np.maximum(0, wl - lmax_l) ** 2 + np.maximum(0, wr - lmax_r) ** 2
    imb[(np.sum(P, axis=1) == 0) | (np.sum(P, axis=1) == n)] = np.inf
    cuts = np.zeros(P.shape[0])
    for u in range(n):
        for i in range(graph.xadj[u], graph.xadj[u + 1]):
            v, w = graph.adjncy[i], graph.adjwgt[i]
            if u < v: cuts += (P[:, u] != P[:, v]) * w
    idx = np.lexsort((cuts, imb))[0]
    return P[idx], (int(wl[idx]), int(wr[idx]))


# --- PARALLELIZABLE MATCHING (PLACEHOLDER) ---
# ... already integrated in coarsen_simple logic

def main():
    if len(sys.argv) < 4:
        print("Usage: python progressive_partition.py <graph> <hierarchy> <distances> [imbalance] [threshold]")
        return
    filename, raw_h = sys.argv[1], [int(x) for x in sys.argv[2].split(":")]
    dists = [int(x) for x in sys.argv[3].split(":")]
    imb_v = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
    threshold = int(sys.argv[5]) if len(sys.argv) > 5 else 16

    # Timing buckets
    timers = {
        "io": 0.0,
        "coarsen": 0.0,
        "bisect_extract": 0.0,
        "bisect_core": 0.0,
        "uncoarsen": 0.0,
        "stats": 0.0
    }

    graph, timers["io"] = read_graph_csr(filename)
    if not graph: return
    tk, tw = math.prod(raw_h), graph.total_weight
    lmax_global = math.ceil((1 + imb_v) * tw / tk)

    # Hierarchy Setup
    num_lvls = len(raw_h)
    strides = [math.prod(raw_h[:i]) for i in range(num_lvls)]

    t_solve_start = time.time()

    # --- COARSENING ---
    graphs, mappings = [graph], []
    curr_g = graph
    # Constraint: No coarse vertex should exceed the allowed maximum block weight.
    while curr_g.n > threshold:
        t0 = time.time()
        next_g, cmap = coarsen_simple(curr_g, lmax_global)
        timers["coarsen"] += time.time() - t0
        if next_g.n >= curr_g.n * 0.95: break
        graphs.append(next_g);
        mappings.append(cmap);
        curr_g = next_g

    # --- PROGRESSIVE BISECTION / UNCOARSENING ---
    cl = len(graphs) - 1
    part = np.zeros(graphs[cl].n, dtype=np.int32)
    b_weights = np.zeros(tk, dtype=np.float64)
    b_counts = np.zeros(tk, dtype=np.int32)
    b_h_lvl = np.zeros(tk, dtype=np.int32)
    b_h_fact = np.zeros(tk, dtype=np.int32)
    b_weights[0], b_counts[0] = graphs[cl].total_weight, graphs[cl].n
    b_h_lvl[0], b_h_fact[0] = 0, raw_h[0]

    # Pre-calculate fine node counts for each coarse graph level
    fnc_list = [np.bincount(m, minlength=graphs[i + 1].n) for i, m in enumerate(mappings)]

    while True:
        curr_g = graphs[cl]
        split_occurred = False

        # Efficiently calculate lookahead counts for all blocks
        if cl > 0:
            lookahead_counts = np.bincount(part, weights=fnc_list[cl - 1], minlength=tk).astype(np.int32)
        else:
            lookahead_counts = None

        for b in range(tk):
            if b_counts[b] == 0: continue
            lvl = b_h_lvl[b]
            if lvl >= num_lvls: continue

            f = b_h_fact[b]
            if f > 1:
                # Postponement logic using tuneable threshold:
                can_partition = False
                if cl == 0:
                    can_partition = True
                else:
                    if lookahead_counts[b] > threshold or b_counts[b] > threshold:
                        can_partition = True

                # Minimum bisection size is threshold // 2
                if can_partition and b_counts[b] >= (threshold // 2):
                    t0 = time.time()
                    nodes = np.where(part == b)[0]
                    sub_n = len(nodes)
                    l2g, g2l = nodes, np.full(curr_g.n, -1, dtype=np.int32)
                    g2l[l2g] = np.arange(sub_n)
                    sub_vw, sub_xadj, sub_adjncy, sub_adjwgt, curr_m = curr_g.vw[nodes], np.zeros(sub_n + 1,
                                                                                                  dtype=np.int32), [], [], 0
                    for i, u in enumerate(nodes):
                        sub_xadj[i] = curr_m
                        for j in range(curr_g.xadj[u], curr_g.xadj[u + 1]):
                            v, w = curr_g.adjncy[j], curr_g.adjwgt[j]
                            if g2l[v] != -1: sub_adjncy.append(g2l[v]); sub_adjwgt.append(w); curr_m += 1
                    sub_xadj[sub_n] = curr_m
                    sub_g = CSRGraph(sub_n, sub_vw, sub_xadj, np.array(sub_adjncy, dtype=np.int32),
                                     np.array(sub_adjwgt, dtype=np.int32))
                    timers["bisect_extract"] += time.time() - t0

                    t0 = time.time()
                    rp = 1 << int(math.log2(f - 1))
                    lp = f - rp
                    unit_w = tw / tk
                    stride = math.prod(raw_h[lvl + 1:]) if lvl + 1 < num_lvls else 1
                    lmax_l = math.ceil((1 + imb_v) * unit_w * (lp * stride))
                    lmax_r = math.ceil((1 + imb_v) * unit_w * (rp * stride))
                    lpart, (wl, wr) = brute_force_bisect(sub_g, lmax_l, lmax_r)
                    timers["bisect_core"] += time.time() - t0

                    # Track Imbalance
                    imb_l = max(0, wl - lmax_l)
                    imb_r = max(0, wr - lmax_r)
                    if imb_l > 0 or imb_r > 0:
                        print(
                            f"[IMB] Split Block {b:3} (lvl {lvl}, cl {cl}): L={wl:6}/{lmax_l:6} (+{imb_l:5}), R={wr:6}/{lmax_r:6} (+{imb_r:5}) | SubN: {sub_n:3}")

                    rid = b + lp * stride
                    part[nodes] = np.where(lpart == 0, b, rid)
                    b_weights[b], b_weights[rid] = wl, wr
                    b_counts[rid] = np.sum(lpart)
                    b_counts[b] = sub_n - b_counts[rid]
                    b_h_lvl[rid] = lvl
                    b_h_fact[b], b_h_fact[rid] = lp, rp
                    split_occurred = True
            else:
                if lvl + 1 < num_lvls:
                    b_h_lvl[b] += 1
                    b_h_fact[b] = raw_h[lvl + 1]
                    split_occurred = True
                else:
                    b_h_lvl[b] = num_lvls

        if not split_occurred:
            if cl > 0:
                t0 = time.time()
                cl -= 1
                part = part[mappings[cl]]
                b_counts = np.bincount(part, minlength=tk).astype(np.int32)
                b_weights = np.bincount(part, weights=graphs[cl].vw, minlength=tk)
                timers["uncoarsen"] += time.time() - t0
            else:
                break

    t_solve_end = time.time()

    # --- FINAL STATS ---
    t0 = time.time()
    cut, comm_cost = 0, 0
    for u in range(graph.n):
        for i in range(graph.xadj[u], graph.xadj[u + 1]):
            v, w = graph.adjncy[i], graph.adjwgt[i]
            if u < v and part[u] != part[v]:
                cut += w
                b1, b2 = part[u], part[v]
                for l in range(num_lvls):
                    s = math.prod(raw_h[l + 1:]) if l + 1 < num_lvls else 1
                    if (b1 // (s * raw_h[l])) != (b2 // (s * raw_h[l])):
                        comm_cost += w * dists[l];
                        break
                else:
                    comm_cost += w * dists[-1]
    timers["stats"] = time.time() - t0

    print(f"\n{'=' * 30}")
    print(f"PROFILING REPORT")
    print(f"{'=' * 30}")
    print(f"Graph IO:         {timers['io']:.4f}s")
    print(f"Coarsening:       {timers['coarsen']:.4f}s")
    print(f"Sub-Extract:      {timers['bisect_extract']:.4f}s")
    print(f"Bisection Core:   {timers['bisect_core']:.4f}s")
    print(f"Uncoarsen/Proj:   {timers['uncoarsen']:.4f}s")
    print(f"Stats Calculation: {timers['stats']:.4f}s")
    print(f"{'-' * 30}")
    print(f"Total Solve Time: {t_solve_end - t_solve_start:.4f}s")
    print(f"{'=' * 30}")
    print(f"\nRESULT")
    print(f"Lmax (Target): {lmax_global}")
    print(f"Max Block W:   {int(b_weights.max())}")
    print(f"Edge Cut:      {cut}")
    print(f"Comm Cost:     {comm_cost}")


if __name__ == "__main__":
    main()
