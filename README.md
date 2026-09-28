<div align="center">

# **GPU-HeiPa**
### **GPU-Accelerated Graph Partitioning & Process Mapping Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v1.1.0-brightgreen.svg)](https://github.com/HenningWoydt/GPU-HeiPa/releases)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Kokkos](https://img.shields.io/badge/Kokkos-5.0.0-orange.svg)](https://github.com/kokkos/kokkos)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

</div>

## Overview

**GPU-HeiPa** is a high-performance graph partitioning and process mapping framework built for modern GPU architectures using **C++20** and **[Kokkos 5.0](https://github.com/kokkos/kokkos)**. It achieves fast execution and high solution quality through parallel multilevel coarsening, initial partitioning, and deep GPU refinement.

### Core Tools

| Executable | Purpose |
| :--- | :--- |
| `GPU-HeiPa` | Fast $k$-way graph partitioning with parallel multilevel refinement |
| `GPU-HeiProMap` | Hierarchical process mapping minimizing communication cost |
| `GPU-MemHeiPa` | Evolutionary memetic partitioning combining GPU operators with population search |

---

## Problem Formulations

<details>
<summary><b>1. k-way Graph Partitioning</b></summary>

Given an undirected graph $G = (V, E)$ with vertex weights $c(v)$ and edge weights $\omega(e)$, partition $V$ into $k$ disjoint blocks $V_1, \ldots, V_k$:
- **Balance constraint:**
  $$\forall i \in \{1, \ldots, k\}: \sum_{v \in V_i} c(v) \leq L_{\max} := (1 + \varepsilon) \cdot \left\lceil \frac{c(V)}{k} \right\rceil$$
- **Objective:** Minimize total edge cut:
  $$\text{cut}(V_1, \ldots, V_k) = \sum_{\{u,v\} \in E, \Pi(u) \neq \Pi(v)} \omega(u,v)$$
</details>

<details>
<summary><b>2. Hierarchical Process Mapping</b></summary>

Given a communication graph $G = (V, E)$ and a target hierarchy $H = a_1 : a_2 : \ldots : a_\ell$ with distance costs $D = d_1 : d_2 : \ldots : d_\ell$:
- **Balance constraint:**
  $$\forall i \leq k: \sum_{j: \Pi(j) = i} c(j) \leq (1 + \varepsilon) \frac{c(V)}{k}$$
- **Objective:** Minimize communication metric:
  $$J(C, D, \Pi) = \sum_{u, v \in V} C_{uv} \cdot D_{\Pi(u)\Pi(v)}$$
</details>

---

## Quick Start

### Prerequisites

- **CMake** $\ge$ 3.16
- **C++20 Compiler** (GCC 10+, Clang 11+)
- **CUDA Toolkit** $\ge$ 11.0 (Compute Capability $\ge$ 7.0)
- **OpenMP**

### Build

```bash
# Complete build (fetches & builds Kokkos 5.0 dependencies)
./build.sh

# Rebuild application only (fast)
./build.sh --download-kokkos=OFF
```

#### Custom Build Options
```bash
./build.sh --max-threads=16 --kokkos-arch=Kokkos_ARCH_AMPERE86
```

Output binaries in `build/`:
- `build/GPU-HeiPa`
- `build/GPU-HeiProMap`
- `build/GPU-MemHeiPa`

---

## Usage & CLI Guide

### 1. `GPU-HeiPa` (Graph Partitioning)

Partition a graph into $k$ balanced blocks:
```bash
./build/GPU-HeiPa -g data/graph.metis -k 64 -e 0.03 -c ultra -m output.part
```

<details>
<summary><b>View CLI Options Table</b></summary>

| Flag | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--graph` | `-g` | Path to METIS graph file | *required* |
| `--k` | `-k` | Number of blocks $k$ | *required* |
| `--config` | `-c` | Preset: `default`, `ultra` | *required* |
| `--imbalance` | `-e` | Allowed imbalance $\varepsilon$ (e.g. `0.03`) | `0.03` |
| `--mapping` | `-m` | Output partition file path | `GPU-HeiPa_par.txt` |
| `--coarsening` | | Coarsening: `two-hop`, `independent-edge-set` | `two-hop` |
| `--initial-partitioning` | | Initial partitioner: `kway`, `metis` | `kway` |
| `--c-limit` | | Contraction limit $c$ | `64` |
| `--seed` | | Random seed | `0` (random) |
| `--n-bytes-requested` | | Allocated GPU memory (bytes) | `8589934592` |
| `--verbose-level` | | Verbosity (0=quiet, 1=normal, 2=detailed) | `1` |
| `--help` | | Show help message | |

</details>

---

### 2. `GPU-HeiProMap` (Process Mapping)

Map processes to a multi-level hardware hierarchy:
```bash
./build/GPU-HeiProMap \
  -g data/comm_graph.metis \
  --hierarchy 2:4:8 \
  --distance 1:5:50 \
  --config HM-ultra \
  -m output.map
```

<details>
<summary><b>View CLI Options Table</b></summary>

| Flag | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--graph` | `-g` | Path to METIS graph file | *required* |
| `--hierarchy` | `-h` | Hierarchy levels `a1:a2:...:al` | *required* |
| `--distance` | `-d` | Distance costs `d1:d2:...:dl` | *required* |
| `--config` | `-c` | Preset: `IM`, `HM`, `HM-ultra` | *required* |
| `--imbalance` | `-e` | Allowed imbalance $\varepsilon$ | `0.03` |
| `--distance-oracle` | | Distance oracle: `matrix` | `matrix` |
| `--mapping` | `-m` | Output mapping file path | `GPU-HeiProMap_par.txt` |
| `--initial-partitioning` | | Initial partitioner: `global_multisection` | `global_multisection` |
| `--seq-partitioner` | | Partitioner for multisection: `kway`, `metis` | `kway` |
| `--c-limit` | | Contraction limit $c$ | `8` |
| `--seed` | | Random seed | `0` (random) |
| `--n-bytes-requested` | | Allocated GPU memory (bytes) | `8589934592` |
| `--verbose-level` | | Verbosity (0=quiet, 1=normal, 2=detailed) | `1` |
| `--help` | | Show help message | |

</details>

---

### 3. `GPU-MemHeiPa` (Memetic Evolutionary Algorithm)

Run population-based evolutionary graph partitioning:
```bash
./build/GPU-MemHeiPa \
  -g data/graph.metis \
  -k 16 \
  -e 0.03 \
  -c default \
  --num-individuals 10 \
  --num-cpu-threads 4 \
  --population-management shrinking \
  -m output.part
```

<details>
<summary><b>View CLI Options Table</b></summary>

| Flag | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--graph` | `-g` | Path to METIS graph file | *required* |
| `--k` | `-k` | Number of blocks $k$ | *required* |
| `--config` | `-c` | Configuration preset | *required* |
| `--imbalance` | `-e` | Allowed imbalance $\varepsilon$ | `0.03` |
| `--mapping` | `-m` | Output partition file path | `GPU-HeiPa_par.txt` |
| `--statistics` | | Output JSON stats file path | `GPU-HeiPa_stats.JSON` |
| `--seed` | `-s` | Random seed | `0` (random) |
| `--verbose-level` | | Verbosity (0=quiet, 1=normal, 2=detailed) | `2` |
| `--population-management` | | Strategy: `shrinking`, `steadystate` | `shrinking` |
| `--num-individuals` | | Population size | `20` |
| `--num-cpu-threads` | | CPU worker threads | `4` |
| `--reduction-factor` | | Population reduction factor | `1` |
| `--num-crossovers` | | Crossovers per generation | `1` |
| `--num-parents` | | Parents per crossover | `2` |
| `--tournament-size` | | Selection tournament size | `2` |
| `--crossover-mode` | | Crossover operator: `signature`, `paper` | `signature` |
| `--leftover-strategy` | | Leftover strategy: `random`, `balanced`, `gain`, `mixed` | `mixed` |
| `--alpha` | | Alpha parameter for mixed strategy | `100.0` |
| `--extent` | | Backbone crossover extent in $[1, k]$ | `1` |
| `--distance` | | Distance calculation: `exact`, `sampled` | `exact` |
| `--inactive-percentile` | | Crossover lower bound percentile in $[0, 1]$ | `0.1` |
| `--mutation-percentile` | | Mutation upper bound percentile in $[0, 1]$ | `0.1` |
| `--mutation-rate` | | Mutation rate in $[0, 1]$ | `0.5` |
| `--help` | | Show help message | |

</details>

---

## Dependencies

Dependencies managed automatically via CMake / `build.sh`:

- [**Kokkos 5.0.0**](https://github.com/kokkos/kokkos) — Core performance portability framework
- [**Kokkos-Kernels 5.0.0**](https://github.com/kokkos/kokkos-kernels) — High-performance linear algebra kernels

---

## Citation

If you use **GPU-HeiPa** in academic research, please cite:

```bibtex
@Misc{Samoldekin25,
    author        = {Petr Samoldekin and Christian Schulz and Henning Woydt},
    title         = {{GPU-Accelerated Algorithms for Process Mapping}},
    year          = {2025},
    archiveprefix = {arXiv},
    doi           = {10.48550/arxiv.2510.12196},
    eprint        = {2510.12196},
    eprinttype    = {arxiv},
}
```

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

**Copyright (c) 2026 Henning Woydt**

> *Note: Includes a modified version of METIS (Apache License 2.0, Copyright (c) Regents of the University of Minnesota).*
