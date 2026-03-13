<div align="center">

# **GPU**-accelerated Heidelberg **Partitioning** and **Process Mapping**

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
- **Dual Tools**: 
  - `GPU-HeiPa` - Fast k-way graph partitioning with multilevel refinement
  - `GPU-HeiProMap` - Hierarchical process mapping with communication cost optimization

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
- Compiles both executables

**Build outputs:**
- `build/GPU-HeiPa` - Graph partitioning tool
- `build/GPU-HeiProMap` - Process mapping tool

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

**Common Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--graph` | `-g` | Input graph file (METIS format) | *required* |
| `--k` | `-k` | Number of partitions | *required* |
| `--imbalance` | `-e` | Maximum imbalance (0.0-1.0) | `0.03` |
| `--config` | `-c` | Configuration: `default`, `ultra` | *required* |
| `--mapping` | `-m` | Output partition file | `GPU-HeiPa_par.txt` |
| `--verbose-level` | | Verbosity (0=quiet, 1=normal) | `1` |

**Configuration Presets:**
- `default` - Faster, good quality
- `ultra` - Best quality, slower runtime

**Example:**
```bash
# Partition a graph into 64 blocks with 5% imbalance tolerance
./build/GPU-HeiPa -g data/graph.metis -k 64 -e 0.05 -c ultra -m output.part
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
| `--graph` | `-g` | Input graph (METIS format) | *required* |
| `--hierarchy` | `-h` | Hierarchical structure (levels separated by `:`) | *required* |
| `--distance` | `-d` | Distance costs between levels | *required* |
| `--imbalance` | `-e` | Maximum imbalance | `0.03` |
| `--config` | `-c` | Configuration: `IM`, `HM`, `HM-ultra` | *required* |
| `--mapping` | `-m` | Output mapping file | `GPU-HeiProMap_par.txt` |
| `--verbose-level` | | Verbosity (0=quiet, 1=normal) | `1` |

**Configuration Presets:**
- `IM` - Fastest, lowest quality
- `HM` - Balanced speed and quality
- `HM-ultra` - Best quality, slowest runtime

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

This project includes modified versions of METIS, which is licensed under the Apache License 2.0.  
Copyright (c) Regents of the University of Minnesota.
