#!/usr/bin/env bash
set -euo pipefail

# Get the directory where the script is located
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

echo "==> Building GPU-HeiPa in silent mode with assertions..."
if ./build.sh --silent --download-kokkos=OFF --assert-enabled=ON; then
    echo "==> Build successful."
else
    echo "==> Build failed! See output above."
    exit 1
fi

echo "==> Running GPU-HeiProMap..."
if [ -f "${ROOT}/build/GPU-HeiProMap" ]; then
    cd "${ROOT}/build"
    ./GPU-HeiProMap
else
    echo "Error: GPU-HeiProMap executable not found in build directory."
    exit 1
fi
