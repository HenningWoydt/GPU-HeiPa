# GPU-HeiPa

**GPU-accelerated Heidelberg Partitioning and Process Mapping**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Kokkos](https://img.shields.io/badge/Kokkos-5.0.0-orange.svg)](https://github.com/kokkos/kokkos)

GPU-HeiPa is a high-performance graph partitioning and process mapping framework designed for modern GPU architectures. It provides two main tools for optimizing computational workloads on parallel systems.

## Features

- **GPU Acceleration**: Leverages [Kokkos](https://github.com/kokkos/kokkos) for portable performance across CUDA and OpenMP backends
- **Dual Tools**:
  - `GPU-HeiPa`: Fast k-way graph partitioning
  - `GPU-HeiProMap`: Hierarchical process mapping with communication cost optimization

## Building

### Prerequisites

- CMake 3.16+
- C++20 compatible compiler (GCC 10+, Clang 11+)
- CUDA Toolkit 11.0+ (for GPU backend)

### Quick Build

The project includes an automated build script that handles all dependencies:

```bash
# Build with CUDA backend
./build.sh
```

The script automatically:
- Detects GPU architecture (for CUDA builds)
- Builds Kokkos and Kokkos-Kernels
- Builds KaHIP, METIS, and GKlib
- Compiles both executables

### Manual Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Executables will be in `build/`:
- `GPU-HeiPa`
- `GPU-HeiProMap`

## Usage

### GPU-HeiPa: Graph Partitioning

Partition a graph into k balanced blocks:

```bash
./GPU-HeiPa \
  --graph input.graph \
  --k 32 \
  --imbalance 0.03 \
  --config default \
  --mapping output.txt \
  --seed 1
```

**Parameters:**
- `--graph, -g`: Input graph file (METIS format)
- `--k, -k`: Number of partitions
- `--imbalance, -e`: Maximum imbalance (default: 0.03)
- `--config, -c`: Configuration preset
- `--mapping, -m`: Output partition file (optional)
- `--seed, -s`: Random seed (default: 0)
- `--verbose-level`: Verbosity 0-2 (default: 2)

### GPU-HeiProMap: Process Mapping

Map processes to a hierarchical architecture with communication cost optimization:

```bash
./GPU-HeiProMap \
  --graph input.graph \
  --hierarchy 4:8:6 \
  --distance 1:10:100 \
  --imbalance 0.03 \
  --config HM-ultra \
  --distance-oracle matrix \
  --mapping output.txt
```

**Parameters:**
- `--hierarchy`: Hierarchical structure (e.g., `4:8:6` = 4 nodes × 8 sockets × 6 cores)
- `--distance`: Distance costs between hierarchy levels
- `--distance-oracle`: Oracle type (`matrix` or `binary`)
- `--config`: Configuration (`IM`, `HM`, `HM-ultra`)

## Dependencies

Automatically built by `build.sh`:
- [Kokkos 5.0.0](https://github.com/kokkos/kokkos) - Performance portability
- [Kokkos-Kernels 5.0.0](https://github.com/kokkos/kokkos-kernels) - Linear algebra kernels

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

It includes METIS, which is licensed under the Apache License 2.0
Copyright (c) Regents of the University of Minnesota.

