#!/usr/bin/env bash
set -euo pipefail

BACKEND="Cuda"
BACKEND_LOWER="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"

case "$BACKEND_LOWER" in
  cuda|nvidia|gpu)
    USE_CUDA=ON
    ;;
  *)
    echo "Error: Invalid backend '$BACKEND'. Only 'Cuda' is supported."
    exit 1
    ;;
esac

echo "==> Building with backend: ${BACKEND}"

# ---- detect GPU arch and map to Kokkos flag ----
detect_kokkos_arch() {
  # Allow manual override (e.g. KOKKOS_ARCH=Kokkos_ARCH_AMPERE86)
  if [ -n "${KOKKOS_ARCH:-}" ]; then
    echo "${KOKKOS_ARCH}=ON"
    return 0
  fi

  # Try nvidia-smi compute capability
  if command -v nvidia-smi >/dev/null 2>&1; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | awk -F. '{printf "%d%d\n", $1, $2}' \
         | sort -nr | head -n1)
    case "$cc" in
      120) echo "Kokkos_ARCH_BLACKWELL120=ON" ;;
      90)  echo "Kokkos_ARCH_HOPPER90=ON" ;;
      89)  echo "Kokkos_ARCH_ADA89=ON" ;;
      86)  echo "Kokkos_ARCH_AMPERE86=ON" ;;
      80)  echo "Kokkos_ARCH_AMPERE80=ON" ;;
      75)  echo "Kokkos_ARCH_TURING75=ON" ;;
      70)  echo "Kokkos_ARCH_VOLTA70=ON" ;;
      *)
        echo ""
        ;;
    esac
    return 0
  fi

  # Last resort: let Kokkos autodetect
  echo ""
}

ARCH_FLAG="$(detect_kokkos_arch)"
echo "Auto-detected Kokkos arch flag: ${ARCH_FLAG:-<autodetect>}"

# ----- pick a reasonable parallelism (leave 2 cores free) -----
calc_jobs() {
  local cores
  cores=$( nproc 2>/dev/null \
        || getconf _NPROCESSORS_ONLN 2>/dev/null \
        || sysctl -n hw.ncpu 2>/dev/null \
        || echo 4 )
  local j=$(( cores - 2 ))
  if [ "$j" -lt 1 ]; then j=1; fi
  echo "$j"
}
JOBS="${MAX_THREADS:-$(calc_jobs)}"
echo "Building with $JOBS parallel jobs (override with MAX_THREADS)."

# clean previous externals
rm -rf extern/local
rm -rf extern/kokkos-5.0.0
rm -rf extern/kokkos-kernels-5.0.0

ROOT=${PWD}
GCC=$(which gcc || true)

echo "Root            : ${ROOT}"
echo "Using C compiler: ${GCC:-<system default>}"

# make local folder for all includes
mkdir -p extern
cd extern && rm -rf local && mkdir local && cd "${ROOT}"

# --- Download Kokkos-Kernels 5.0.0 ---
echo "Downloading Kokkos-Kernels 5.0.0..."
if (
  cd extern \
  && rm -f kokkos-kernels-5.0.0.tar.gz \
  && rm -rf kokkos-kernels-5.0.0 \
  && wget -q https://github.com/kokkos/kokkos-kernels/releases/download/5.0.0/kokkos-kernels-5.0.0.tar.gz \
  && tar -xzf kokkos-kernels-5.0.0.tar.gz \
  && rm -f kokkos-kernels-5.0.0.tar.gz
); then
  echo "Kokkos-Kernels 5.0.0 downloaded and extracted successfully."
else
  echo "Failed to download Kokkos-Kernels!" >&2
  exit 1
fi

# --- Download Kokkos 5.0.0 ---
echo "Downloading Kokkos 5.0.0..."
if (
  cd extern \
  && rm -f kokkos-5.0.0.tar.gz \
  && rm -rf kokkos-5.0.0 \
  && wget -q https://github.com/kokkos/kokkos/releases/download/5.0.0/kokkos-5.0.0.tar.gz \
  && tar -xzf kokkos-5.0.0.tar.gz \
  && rm -f kokkos-5.0.0.tar.gz
); then
  echo "Kokkos 5.0.0 downloaded and extracted successfully."
else
  echo "Failed to download Kokkos!" >&2
  exit 1
fi

# Compiler for CMake (C++): Kokkos nvcc_wrapper for CUDA
export CXX="${ROOT}/extern/kokkos-5.0.0/bin/nvcc_wrapper"
if [ ! -x "$CXX" ]; then
  echo "Error: nvcc_wrapper not found at $CXX"
  echo "Make sure kokkos source was extracted and CUDA toolkit is installed."
  exit 2
fi

# Disable CUDA lazy loading - force eager module loading
export CUDA_MODULE_LOADING=EAGER
echo "CUDA lazy loading disabled (CUDA_MODULE_LOADING=EAGER)"

echo "Using C++ compiler: ${CXX}"

# ---- backend-specific flags ----
KOKKOS_COMMON="-DCMAKE_INSTALL_PREFIX=${ROOT}/extern/local/kokkos \
               -DKokkos_ENABLE_SERIAL=ON \
               -DCMAKE_BUILD_TYPE=Release \
               -DKokkos_ENABLE_DEBUG=OFF \
               -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
               -DKokkos_ENABLE_TUNING=ON"

KOKKOS_BACKEND="-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA_LAMBDA=ON"

# Strong optimization defaults for Release
CXX_RELEASE_FLAGS="-O3 -DNDEBUG -march=native -mtune=native -fno-math-errno -fomit-frame-pointer"

# --- build kokkos ---
echo "Building Kokkos 5.0.0..."
if (
  cd "${ROOT}/extern/kokkos-5.0.0" \
  && mkdir -p build && cd build \
  && cmake .. \
    ${KOKKOS_COMMON} \
    ${KOKKOS_BACKEND} \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_CXX_FLAGS_RELEASE="${CXX_RELEASE_FLAGS}" \
    -DCMAKE_CXX_FLAGS="-w" \
    ${ARCH_FLAG:+-D${ARCH_FLAG}} \
  && make install -j "$JOBS"
); then
  echo "Kokkos 5.0.0 build completed successfully."
else
  echo "Kokkos 5.0.0 build failed!" >&2
  exit 1
fi

echo "Building Kokkos-Kernels 5.0.0..."
if (
  cd "${ROOT}/extern/kokkos-kernels-5.0.0" \
  && mkdir -p build && cd build \
  && cmake .. \
    -DCMAKE_INSTALL_PREFIX="${ROOT}/extern/local/kokkos-kernels" \
    -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DKokkosKernels_ENABLE_TESTS=OFF \
    -DKokkosKernels_ENABLE_EXAMPLES=OFF \
    -DKokkosKernels_ENABLE_PERFTESTS=OFF \
    -DCMAKE_CXX_FLAGS_RELEASE="${CXX_RELEASE_FLAGS}" \
    -DCMAKE_CXX_FLAGS="-w" \
    ${KOKKOS_BACKEND} \
  && make install -j "$JOBS"
); then
  echo "Kokkos-Kernels 5.0.0 build completed successfully."
else
  echo "Kokkos-Kernels 5.0.0 build failed!" >&2
  exit 1
fi
cd "${ROOT}"

# --- build GPU-HeiPa ---
echo "Building GPU-HeiPa..."
rm -rf build && mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos;${ROOT}/extern/local/kokkos-kernels" \
         -DCMAKE_CXX_STANDARD=20 \
         -DCMAKE_CXX_EXTENSIONS=OFF
cmake --build . --parallel "$JOBS" --target GPU-HeiPa
cmake --build . --parallel "$JOBS" --target GPU-HeiProMap