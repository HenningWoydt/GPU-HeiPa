import sys
import math
import random
import time
import numpy as np

# Pre-allocate P matrices for bisection (up to n=20)
P_CACHE = {}
def get_p_matrix(n):
    if n <= 1 or n > 20: return None
    if n not in P_CACHE:
        num_combos = 1 << (n - 1)
        P = np.zeros((num_combos, n), dtype=np.int8)
        P[:, 1:] = (np.arange(num_combos)[:, None] >> np.arange(n - 1)) & 1
        P_CACHE[n] = P
    return P_CACHE[n]

def read_graph(filename):
    """Reads a METIS-style graph file efficiently."""
    with open(filename, 'r') as f:
        all_lines = f.readlines()
    
    line_idx = 0
    while line_idx < len(all_lines) and (all_lines[line_idx].startswith('%') or not all_lines[line_idx].strip()):
        line_idx += 1
    
    if line_idx >= len(all_lines): return 0, np.array([]), []
    
    header = all_lines[line_idx].split()
    line_idx += 1
    n = int(header[0])
    fmt = header[2] if len(header) > 2 else "000"
    has_vw, has_ew = (fmt[-2] if len(fmt) >= 2 else '0') == '1', (fmt[-1] if len(fmt) >= 1 else '0') == '1'
    
    vw = np.ones(n, dtype=np.int32)
    adj = []
    for i in range(n):
        while line_idx < len(all_lines) and (all_lines[line_idx].startswith('%') or not all_lines[line_idx].strip()):
            line_idx += 1
        parts = [int(x) for x in all_lines[line_idx].split()]
        line_idx += 1
        if has_vw: vw[i] = parts.pop(0)
        
        node_adj = []
        if has_ew:
            for j in range(0, len(parts), 2): node_adj.append((parts[j]-1, parts[j+1]))
        else:
            for v in parts: node_adj.append((v-1, 1))
        adj.append(node_adj)
    return n, vw, adj

def brute_force_bisect(n, vw, adj, lmax_l, lmax_r):
    if n <= 1:
        side = 0 if lmax_l >= lmax_r else 1
        return np.array([side], dtype=np.int8), (vw[0] if side == 0 else 0, 0 if side == 0 else vw[0])
    
    if n > 20:
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
    w_r, w_l = P @ vw_f, vw_f.sum() - (P @ vw_f)
    
    imb = np.maximum(0, w_l - lmax_l)**2 + np.maximum(0, w_r - lmax_r)**2
    imb[(np.sum(P, axis=1) == 0) | (np.sum(P, axis=1) == n)] = np.inf
    
    edges = [(u, v, w) for u in range(n) for v, w in adj[u] if u < v]
    cuts = (P[:, [e[0] for e in edges]] != P[:, [e[1] for e in edges]]) @ [e[2] for e in edges] if edges else np.zeros(P.shape[0])
    idx = np.lexsort((cuts, imb))[0]
    return P[idx], (int(w_l[idx]), int(w_r[idx]))

def main():
    if len(sys.argv) < 3:
        print("Usage: python progressive_partition.py <graph> <hierarchy> [distances] [imbalance]")
        return

    filename = sys.argv[1]
    raw_h = [int(x) for x in sys.argv[2].split(":")]
    imb_v = float(sys.argv[4]) if len(sys.argv) > 4 else (float(sys.argv[3]) if len(sys.argv) > 3 and ":" not in sys.argv[3] else 0.03)
    
    n, vw, adj = read_graph(filename)
    tk, tw = math.prod(raw_h), vw.sum()

    # Pre-allocate State Arrays
    num_lvls = len(raw_h)
    b_factors = np.zeros((num_lvls, tk), dtype=np.int32)
    b_strides = np.zeros((num_lvls, tk), dtype=np.int32)
    b_h_idx = np.full(tk, num_lvls - 1, dtype=np.int32)

    for l in range(num_lvls):
        b_factors[l, :] = raw_h[l]
        b_strides[l, :] = math.prod(raw_h[:l]) if l > 0 else 1

    t_start = time.time()
    graphs, mappings, coarse_node_counts = [(n, vw, adj)], [], []
    curr_n, curr_vw, curr_adj = n, vw, adj
    max_vw = math.ceil((1 + imb_v) * tw / tk) * 0.5
    
    while curr_n > 20:
        cn, cvw, cadj, cmap = coarsen(curr_n, curr_vw, curr_adj, max_vw)
        if cn >= curr_n * 0.95:
            cn, cvw, cadj, cmap = coarsen(curr_n, curr_vw, curr_adj, 0, force_coarsen=True)
            if cn >= curr_n: break
        
        # Keep track of how many fine nodes each coarse node contains
        coarse_node_counts.append(np.bincount(cmap, minlength=cn))
        mappings.append(cmap)
        graphs.append((cn, cvw, cadj))
        curr_n, curr_vw, curr_adj = cn, cvw, cadj
    
    t_coarse = time.time()

    cl = len(graphs) - 1
    gn, gvw, gadj = graphs[cl]
    part = np.zeros(gn, dtype=np.int32)
    
    # Initialize block weights and counts at the coarsest level
    b_weights = np.zeros(tk, dtype=np.float64)
    b_counts = np.zeros(tk, dtype=np.int32)
    b_weights[0] = gvw.sum()
    b_counts[0] = gn
    
    while True:
        split_any = False
        gn, gvw, gadj = graphs[cl]
        
        for b in range(tk):
            h = b_h_idx[b]
            if h < 0 or b_counts[b] == 0: continue
            
            f, s = b_factors[h, b], b_strides[h, b]
            if f > 1:
                bn = b_counts[b]
                # Split condition
                if (bn >= 10 and bn <= 20) or cl == 0 or bn > 20:
                    nodes = np.where(part == b)[0]
                    rp = 1 << int(math.log2(f - 1))
                    lp = f - rp
                    lid, rid = b, b + lp * s
                    
                    lmax_l = math.ceil((1 + imb_v) * tw * (lp * s) / tk)
                    lmax_r = math.ceil((1 + imb_v) * tw * (rp * s) / tk)
                    
                    l2g = nodes; g2l = {u: i for i, u in enumerate(l2g)}
                    lpart, (wl, wr) = brute_force_bisect(len(nodes), gvw[nodes], [[(g2l[v], w) for v, w in gadj[u] if v in g2l] for u in nodes], lmax_l, lmax_r)
                    part[nodes] = np.where(lpart == 0, lid, rid)
                    
                    # Update metadata incrementally
                    # Weights are already calculated by bisect
                    b_weights[lid], b_weights[rid] = wl, wr
                    # Counts must be calculated from lpart
                    cnt_r = np.sum(lpart)
                    cnt_l = len(lpart) - cnt_r
                    b_counts[lid], b_counts[rid] = cnt_l, cnt_r
                    
                    # Update hierarchy state
                    b_factors[h, lid], b_factors[h, rid], b_h_idx[rid] = lp, rp, h
                    split_any = True
            else:
                b_h_idx[b] -= 1
                split_any = True

        if not split_any:
            if cl > 0:
                # Uncoarsen: Projection
                cmap = mappings[cl-1]
                part = part[cmap]
                cl -= 1
                # b_weights is invariant during uncoarsening!
                # b_counts must be updated based on coarse_node_counts
                # We use bincount with weights to sum up the sizes of fine nodes per block
                node_sizes = coarse_node_counts[cl]
                # Note: 'part' is now the finer partition, but we need the coarse one for the bincount
                # Actually, we can just use the previous 'part' before projection.
                # Let's save it.
                pass 
                # Re-calculate b_counts at the beginning of the loop for safety if not incremental
                # But we can do it better:
                # b_counts = np.bincount(part_before_projection, weights=node_sizes, minlength=tk)
            else:
                if np.all(b_h_idx < 0): break
                active_mask = b_h_idx >= 0
                if not np.any(b_factors[b_h_idx[active_mask], np.where(active_mask)[0]] > 1): break

        # Refresh b_counts and b_weights after uncoarsening to ensure they are up to date
        if not split_any and cl >= 0:
             gn, gvw, gadj = graphs[cl]
             b_counts = np.bincount(part, minlength=tk).astype(np.int32)
             b_weights = np.bincount(part, weights=gvw, minlength=tk)

    t_end = time.time()
    cut = sum(w for u in range(n) for v, w in adj[u] if u < v and part[u] != part[v])
    print(f"\nRESULT")
    print(f"Max Block W: {int(b_weights.max())}")
    print(f"Edge Cut:    {cut}")
    print(f"Time:        {t_end - t_start:.3f}s")

def coarsen(n, vw, adj, max_vw, force_coarsen=False):
    mapping, cn, order = np.full(n, -1, dtype=np.int32), 0, np.random.permutation(n)
    for u in order:
        if mapping[u] != -1: continue
        bv, bw = -1, -1
        for v, w in adj[u]:
            if mapping[v] == -1 and (force_coarsen or vw[u] + vw[v] <= max_vw):
                if w > bw: bv, bw = v, w
        mapping[u] = cn
        if bv != -1: mapping[bv] = cn
        cn += 1
    cvw = np.bincount(mapping, weights=vw).astype(np.int32)
    cadj_m = [{} for _ in range(cn)]
    for u in range(n):
        for v, w in adj[u]:
            if mapping[u] != mapping[v]: cadj_m[mapping[u]][mapping[v]] = cadj_m[mapping[u]].get(mapping[v], 0) + w
    return cn, cvw, [[(v, w) for v, w in m.items()] for m in cadj_m], mapping

if __name__ == "__main__":
    main()
