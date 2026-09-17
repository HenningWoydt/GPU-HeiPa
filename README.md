<div align="center">

# **GPU**-Accelerated Heidelberg **Partitioning** and **Process Mapping**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Kokkos](https://img.shields.io/badge/Kokkos-5.0.0-orange.svg)](https://github.com/kokkos/kokkos)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## Overview

GPU-HeiPa is a high-performance graph partitioning and process mapping framework designed for modern GPU architectures. Built with C++20 and [Kokkos](https://github.com/kokkos/kokkos), it delivers portable performance across CUDA-enabled GPUs.

### Key Features

- **GPU-Accelerated**: Leverages CUDA through Kokkos for massive parallelism
- **Suite of Tools**: 
  - `GPU-HeiPa` - Fast k-way graph partitioning with multilevel refinement
  - `GPU-HeiProMap` - Hierarchical process mapping with communication cost optimization
  - `GPU-MemHeiPa` - Memetic evolutionary graph partitioning algorithm combining GPU acceleration with population management

---

### k-way Graph Partitioning

Given a graph $G = (V, E)$ with vertex weights and edge weights, partition $V$ into $k$ disjoint blocks $V_1, \ldots, V_k$ such that:
- **Balance**: Each block satisfies $|V_i| \leq L_{\max} = (1 + \varepsilon) \cdot \lceil |V|/k \rceil$ for imbalance $\varepsilon$
- **Objective**: Minimize edge cut $= |\{(u,v) \in E : u \in V_i, v \in V_j, i \neq j\}|$

### Hierarchical Process Mapping

Given a communication graph $G = (V, E)$ with edge weights $\omega : E \to \mathbb{R}_+$ and a hierarchical architecture:
- **Hierarchy**: $H = a_1 : a_2 : \ldots : a_\ell$ where $k = \prod_{i=1}^{\ell} a_i$ total cores
- **Distance**: $D = d_1 : d_2 : \ldots : d_\ell$ where $d_i$ is the communication cost at level $i$

Find mapping $\Pi : V \to [k]$ that:
- **Balance**: $\forall i \leq k : \sum_{j : \Pi(j) = i} c(j) \leq (1 + \varepsilon) \frac{c(V)}{k}$
- **Objective**: Minimize $J(C, D, \Pi) = \sum_{i, j \leq n} C_{ij} \cdot D_{\Pi(i)\Pi(j)}$

---

## Quick Start

### Prerequisites

- **CMake** 3.16+
- **C++20 compiler** (GCC 10+, Clang 11+)
- **CUDA Toolkit** 11.0+ with compute capability 7.0+
- **OpenMP**

### Build

```bash
# Full build (downloads and builds Kokkos dependencies)
./build.sh

# Fast rebuild (app only, after dependencies are built)
./build.sh --download-kokkos=OFF
```

The build script automatically:
- Detects your GPU architecture (ensure correct detection for best performance)
- Downloads and builds Kokkos 5.0.0 and Kokkos-Kernels 5.0.0
- Compiles the project executables

**Build outputs:**
- `build/GPU-HeiPa` - Graph partitioning tool
- `build/GPU-HeiProMap` - Process mapping tool
- `build/GPU-MemHeiPa` - Memetic graph partitioning tool

### Build Options

- `--download-kokkos=ON|OFF` - Download and build Kokkos dependencies (default: ON)
- `--max-threads=N` - Override parallel build jobs (default: nproc - 2)
- `--kokkos-arch=ARCH` - Manually specify GPU architecture (e.g., `Kokkos_ARCH_AMPERE86`)

```bash
./build.sh --max-threads=16 --kokkos-arch=Kokkos_ARCH_AMPERE86
```

---

## Usage

### GPU-HeiPa: Graph Partitioning

Partition a graph into k balanced blocks:

```bash
./build/GPU-HeiPa \
  --graph input.graph \
  --k 32 \
  --imbalance 0.03 \
  --config default
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--graph` | `-g` | Filepath to the graph (METIS format) | *required* |
| `--k` | `-k` | Number of blocks $k$ | *required* |
| `--config` | `-c` | Algorithm configuration preset: `default`, `ultra` | *required* |
| `--imbalance` | `-e` | Allowed imbalance $\varepsilon$ (e.g. `0.03`) | `0.03` |
| `--mapping` | `-m` | Output filepath to the generated partition mapping | `GPU-HeiPa_par.txt` |
| `--coarsening` | | Coarsening method: `two-hop`, `independent-edge-set` | `two-hop` |
| `--initial-partitioning` | | Initial partitioning algorithm: `kway`, `biml_bisection`, `recursive-bisection`, `metis` | `kway` |
| `--bisection-method` | | Bisection method: `brute-force`, `heuristic`, `brute-force-with-heuristic`, `grasp` | `grasp` |
| `--c-limit` | | Contraction limit parameter $c$ | `64` |
| `--seed` | | Random seed | `0` (random if omitted) |
| `--n-bytes-requested` | | Total memory in bytes requested from device | `8589934592` |
| `--verbose-level` | | Verbosity level (0=quiet, 1=normal, 2=detailed) | `1` |
| `--help` | | Produce help message | |

**Configuration Presets:**
- `default` - Fast multilevel partitioning with good cut quality
- `ultra` - Deep refinement with best cut quality

**Example:**
```bash
# Partition a graph into 64 blocks with 3% imbalance tolerance using ultra preset
./build/GPU-HeiPa -g data/graph.metis -k 64 -e 0.03 -c ultra -m output.part
```

---

### GPU-HeiProMap: Process Mapping

Map processes to a hierarchical architecture optimizing for communication costs:

```bash
./build/GPU-HeiProMap \
  --graph input.graph \
  --hierarchy 4:8:6 \
  --distance 1:10:100 \
  --imbalance 0.03 \
  --config HM-ultra
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--graph` | `-g` | Input graph file (METIS format) | *required* |
| `--hierarchy` | `-h` | Hierarchical core structure (levels separated by `:`) | *required* |
| `--distance` | `-d` | Communication distance costs between levels separated by `:` | *required* |
| `--config` | `-c` | Algorithm configuration: `IM`, `HM`, `HM-ultra` | *required* |
| `--imbalance` | `-e` | Allowed imbalance $\varepsilon$ | `0.03` |
| `--distance-oracle` | | Distance oracle type: `matrix` | `matrix` |
| `--mapping` | `-m` | Output filepath to the generated mapping | `GPU-HeiProMap_par.txt` |
| `--initial-partitioning` | | Initial partitioning algorithm: `global_multisection`, `biml_bisection` | `global_multisection` |
| `--bisection-method` | | Bisection method: `brute-force`, `heuristic`, `brute-force-with-heuristic` | `brute-force` |
| `--seq-partitioner` | | Sequential partitioner for `global_multisection`: `kway`, `metis` | `kway` |
| `--c-limit` | | Contraction limit parameter $c$ | `8` |
| `--seed` | | Random seed | `0` (random if omitted) |
| `--n-bytes-requested` | | Total memory in bytes requested from device | `8589934592` |
| `--verbose-level` | | Verbosity level (0=quiet, 1=normal, 2=detailed) | `1` |
| `--help` | | Produce help message | |

**Configuration Presets:**
- `IM` - Iterative mapping (fastest, lower quality)
- `HM` - Hierarchical multisection (balanced speed and quality)
- `HM-ultra` - Hierarchical multisection with deep refinement (best quality)

**Example:**
```bash
# Map to 2 nodes × 4 sockets × 8 cores with hierarchical distances
./build/GPU-HeiProMap \
  -g data/comm_graph.metis \
  --hierarchy 2:4:8 \
  --distance 1:5:50 \
  --config HM-ultra \
  -m output.map
```

---

### GPU-MemHeiPa: Memetic Graph Partitioning

Memetic evolutionary partitioning algorithm combining GPU multilevel operators with population-based search:

```bash
./build/GPU-MemHeiPa \
  --graph input.graph \
  --k 32 \
  --imbalance 0.03 \
  --config default
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--graph` | `-g` | Filepath to the graph (METIS format) | *required* |
| `--k` | `-k` | Number of blocks $k$ | *required* |
| `--config` | `-c` | Broad algorithm configuration preset | *required* |
| `--imbalance` | `-e` | Allowed imbalance $\varepsilon$ | `0.03` |
| `--mapping` | `-m` | Output filepath to the generated partition | `GPU-HeiPa_par.txt` |
| `--statistics` | | Output filepath to save execution statistics (JSON format) | `GPU-HeiPa_stats.JSON` |
| `--seed` | `-s` | Random seed | `0` (random if omitted) |
| `--verbose-level` | | Verbosity level (0=quiet, 1=normal, 2=detailed) | `2` |
| `--population-management` | | Population management mode: `shrinking`, `steadystate` | `shrinking` |
| `--num-individuals` | | Population size | `20` |
| `--num-cpu-threads` | | Number of CPU threads | `4` |
| `--reduction-factor` | | Population reduction factor for `shrinking` mode | `1` |
| `--num-crossovers` | | Number of crossovers per generation | `1` |
| `--num-parents` | | Number of parents for crossover | `2` |
| `--tournament-size` | | Tournament size for parent selection | `2` |
| `--crossover-mode` | | Crossover mode: `signature`, `paper` (BBC from paper) | `signature` |
| `--leftover-strategy` | | Leftover distribution strategy: `random`, `balanced`, `gain`, `mixed` | `mixed` |
| `--alpha` | | Alpha parameter for mixed leftover strategy | `100.0` |
| `--extent` | | Extent parameter for backbone crossover in range $[1, k]$ | `1` |
| `--distance` | | Distance computation mode: `exact`, `sampled` | `exact` |
| `--inactive-percentile` | | Disable crossover below normalized level percentile in $[0, 1]$ | `0.1` |
| `--mutation-percentile` | | Enable mutation below normalized level percentile in $[0, 1]$ | `0.1` |
| `--mutation-rate` | | Mutation probability threshold in $[0, 1]$ | `0.5` |
| `--help` | | Produce help message | |

**Example:**
```bash
# Run memetic partitioning with 10 individuals and 4 CPU threads
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

---

## Dependencies

Automatically downloaded and built by `build.sh`:

- [**Kokkos 5.0.0**](https://github.com/kokkos/kokkos) - Performance portability layer
- [**Kokkos-Kernels 5.0.0**](https://github.com/kokkos/kokkos-kernels) - Sparse linear algebra kernels

---

## Citation

If you use GPU-HeiPa in your research, please cite [10.48550/arxiv.2510.12196](https://arxiv.org/abs/2510.12196):

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

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2026 Henning Woydt**

This project includes a modified version of METIS, which is licensed under the Apache License 2.0.  
Copyright (c) Regents of the University of Minnesota.
