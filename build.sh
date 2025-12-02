#!/usr/bin/env bash
set -euo pipefail

# ---------------- arg parsing ----------------
if [ $# -ne 1 ]; then
  echo "Usage: $0 [OpenMP|Cuda]"
  exit 1
fi

BACKEND="$1"
BACKEND_LOWER="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"

case "$BACKEND_LOWER" in
  openmp|omp)
    USE_CUDA=OFF
    USE_OPENMP=ON
    ;;
  cuda|nvidia|gpu)
    USE_CUDA=ON
    USE_OPENMP=OFF
    ;;
  *)
    echo "Error: Invalid backend '$BACKEND'. Use 'OpenMP' or 'Cuda'."
    exit 1
    ;;
esac

echo "==> Building with backend: ${BACKEND}"

# ---- detect GPU arch and map to Kokkos flag (CUDA path) ----
detect_kokkos_arch() {
  # Allow manual override (e.g. KOKKOS_ARCH=Kokkos_ARCH_AMPERE86)
  if [ -n "${KOKKOS_ARCH:-}" ]; then
    echo "${KOKKOS_ARCH}=ON"
    return 0
  fi

  # Try nvidia-smi compute capability (needs driver utils; CUDA 11.6+ exposes compute_cap)
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Pick the highest CC among all GPUs (good default for multi-GPU hosts)
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | awk -F. '{printf "%d%d\n", $1, $2}' \
         | sort -nr | head -n1)
    case "$cc" in
      90)  echo "Kokkos_ARCH_HOPPER90=ON" ;;
      89)  echo "Kokkos_ARCH_ADA89=ON" ;;
      86)  echo "Kokkos_ARCH_AMPERE86=ON" ;;
      80)  echo "Kokkos_ARCH_AMPERE80=ON" ;;
      75)  echo "Kokkos_ARCH_TURING75=ON" ;;
      70)  echo "Kokkos_ARCH_VOLTA70=ON" ;;
      120) echo "Kokkos_ARCH_BLACKWELL120=ON" ;;  # for very new GPUs, if supported by your Kokkos/CUDA
      *)
        # Unknown/new CC: let Kokkos try its own autodetect
        echo ""
        ;;
    esac
    return 0
  fi

  # Last resort: no detection -> let Kokkos autodetect at configure-time
  echo ""
}

# In your CUDA branch, before configuring Kokkos:
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
rm -rf extern/kokkos-4.7.00
rm -rf extern/kokkos-kernels-4.7.00
rm -rf extern/KaHIP

ROOT=${PWD}
GCC=$(which gcc || true)

echo "Root          : ${ROOT}"
echo "Using C compiler: ${GCC:-<system default>}"


# make local folder for all includes
mkdir -p extern
cd extern && rm -rf local && mkdir local && cd "${ROOT}"

# --- Download GKlib (latest release) ---
echo "Downloading GKlib..."
if (
  cd extern \
  && rm -f gklib.tar.gz \
  && rm -rf GKlib \
  && wget -q https://github.com/KarypisLab/GKlib/archive/refs/heads/master.tar.gz -O gklib.tar.gz \
  && tar -xzf gklib.tar.gz \
  && mv GKlib-master GKlib \
  && rm -f gklib.tar.gz
); then
  echo "GKlib downloaded and extracted successfully."
else
  echo "Failed to download GKlib!" >&2
  exit 1
fi

# --- Download METIS 5.2.1 ---
echo "Downloading METIS 5.2.1..."
if (
  cd extern \
  && rm -f metis-5.2.1.tar.gz \
  && rm -rf METIS \
  && wget -q https://github.com/KarypisLab/METIS/archive/refs/tags/v5.2.1.tar.gz -O metis-5.2.1.tar.gz \
  && tar -xzf metis-5.2.1.tar.gz \
  && mv METIS-5.2.1 METIS \
  && rm -f metis-5.2.1.tar.gz
); then
  echo "METIS 5.2.1 downloaded and extracted successfully."
else
  echo "Failed to download METIS v5.2.1!" >&2
  exit 1
fi

# --- Download KaHIP 3.19 ---
echo "Downloading KaHIP 3.19..."
if (
  cd extern \
  && rm -f v3.19.tar.gz \
  && rm -rf KaHIP \
  && wget -q https://github.com/KaHIP/KaHIP/archive/refs/tags/v3.19.tar.gz \
  && tar -xzf v3.19.tar.gz \
  && mv KaHIP-3.19 KaHIP \
  && rm -f v3.19.tar.gz
); then
  echo "KaHIP 3.19 downloaded and extracted successfully."
else
  echo "Failed to download KaHIP!" >&2
  exit 1
fi

# --- Download Kokkos-Kernels 4.7.00 ---
echo "Downloading Kokkos-Kernels 4.7.00..."
if (
  cd extern \
  && rm -f kokkos-kernels-4.7.00.tar.gz \
  && rm -rf kokkos-kernels-4.7.00 \
  && wget -q https://github.com/kokkos/kokkos-kernels/releases/download/4.7.00/kokkos-kernels-4.7.00.tar.gz \
  && tar -xzf kokkos-kernels-4.7.00.tar.gz \
  && rm -f kokkos-kernels-4.7.00.tar.gz
); then
  echo "Kokkos-Kernels 4.7.00 downloaded and extracted successfully."
else
  echo "Failed to download Kokkos-Kernels!" >&2
  exit 1
fi

# --- Download Kokkos 4.7.00 ---
echo "Downloading Kokkos 4.7.00..."
if (
  cd extern \
  && rm -f kokkos-4.7.00.tar.gz \
  && rm -rf kokkos-4.7.00 \
  && wget -q https://github.com/kokkos/kokkos/releases/download/4.7.00/kokkos-4.7.00.tar.gz \
  && tar -xzf kokkos-4.7.00.tar.gz \
  && rm -f kokkos-4.7.00.tar.gz
); then
  echo "Kokkos 4.7.00 downloaded and extracted successfully."
else
  echo "Failed to download Kokkos!" >&2
  exit 1
fi

# Compiler for CMake (C++): GCC by default; Kokkos nvcc_wrapper when CUDA
if [ "$USE_CUDA" = "ON" ]; then
  export CXX="${ROOT}/extern/kokkos-4.7.00/bin/nvcc_wrapper"
  if [ ! -x "$CXX" ]; then
    echo "Error: nvcc_wrapper not found at $CXX"
    echo "Make sure kokkos source was extracted and CUDA toolkit is installed."
    exit 2
  fi
else
  export CXX="${CXX:-g++}"
fi
echo "Using C++ compiler: ${CXX}"

# ---- backend-specific flags ----
# Common
KOKKOS_COMMON="-DCMAKE_INSTALL_PREFIX=${ROOT}/extern/local/kokkos \
               -DKokkos_ENABLE_SERIAL=ON \
               -DKokkos_ENABLE_TESTS=OFF \
               -DCMAKE_BUILD_TYPE=Release \
               -DKokkos_ENABLE_DEBUG=OFF \
               -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
               -DKokkos_ENABLE_PROFILING=OFF \
               -DKokkos_ENABLE_TUNING=OFF"

if [ "$USE_CUDA" = "ON" ]; then
  # Kokkos w/ CUDA
  KOKKOS_BACKEND="-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA_LAMBDA=ON"
  KOKKOS_ARCH=""  # optional: set -DKokkos_ARCH_AMPERE80=ON, etc.
else
  # Kokkos w/ OpenMP
  KOKKOS_BACKEND="-DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_OPENMP=ON"
  KOKKOS_ARCH="-DKokkos_ARCH_NATIVE=ON"
fi

# Strong optimization defaults for Release (keep warnings mute flag if you want)
CXX_RELEASE_FLAGS="-O3 -DNDEBUG -march=native -mtune=native -fno-math-errno -fomit-frame-pointer"
# If you accept non-IEEE behavior, add (optional): -ffast-math

# --- build kokkos ---
echo "Building Kokkos 4.7.00..."
if (
  cd "${ROOT}/extern/kokkos-4.7.00" \
  && mkdir -p build && cd build \
  && cmake .. \
    ${KOKKOS_COMMON} \
    ${KOKKOS_BACKEND} \
    ${KOKKOS_ARCH} \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DKokkosKernels_ENABLE_TESTS=OFF \
    -DCMAKE_CXX_FLAGS_RELEASE="${CXX_RELEASE_FLAGS}" \
    -DCMAKE_CXX_FLAGS="-w" \
    ${ARCH_FLAG:+-D${ARCH_FLAG}} \
    > /dev/null 2>&1 \
  && make install -j "$JOBS" > /dev/null 2>&1
); then
  echo "Kokkos 4.7.00 build completed successfully."
else
  echo "Kokkos 4.7.00 build failed!" >&2
  exit 1
fi

echo "Building Kokkos-Kernels 4.7.00..."
if (
  cd "${ROOT}/extern/kokkos-kernels-4.7.00" \
  && mkdir -p build && cd build \
  && cmake .. \
    -DCMAKE_INSTALL_PREFIX="${ROOT}/extern/local/kokkos-kernels" \
    -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DKokkosKernels_ENABLE_TESTS=OFF \
    -DKokkosKernels_ENABLE_EXAMPLES=OFF \
    -DKokkosKernels_ENABLE_PERFTESTS=OFF \
    -DCMAKE_CXX_FLAGS_RELEASE="${CXX_RELEASE_FLAGS}" \
    -DCMAKE_CXX_FLAGS="-w" \
    ${KOKKOS_BACKEND} \
    ${KOKKOS_ARCH} \
    > /dev/null 2>&1 \
  && make install -j "$JOBS" > /dev/null 2>&1
); then
  echo "Kokkos-Kernels 4.7.00 build completed successfully."
else
  echo "Kokkos-Kernels 4.7.00 build failed!" >&2
  exit 1
fi
cd "${ROOT}"

# install GKLIB into a local folder
export CFLAGS="-Wall -Wno-error=pedantic -Wno-error -D_GNU_SOURCE -DHAVE_STRDUP=1"
export CPPFLAGS="-Wall -Wno-error=pedantic -Wno-error -D_GNU_SOURCE -DHAVE_STRDUP=1"

echo "Building GKlib..."
if cd "${ROOT}/extern/GKlib" && rm -rf build \
  && make config prefix="${ROOT}/extern/local/GKlib" cc="${GCC}" > /dev/null 2>&1 \
  && make install > /dev/null 2>&1; then
  echo "GKlib build completed successfully."
else
  echo "GKlib build failed!" >&2
  exit 1
fi
cd "${ROOT}"

echo "Building METIS..."
if cd "${ROOT}/extern/METIS" \
  && rm -rf build \
  && make config prefix="${ROOT}/extern/local/METIS" gklib_path="${ROOT}/extern/local/GKlib" cc="${GCC}" > /dev/null 2>&1 \
  && make install > /dev/null 2>&1; then
  echo "METIS build completed successfully."
else
  echo "METIS build failed!" >&2
  exit 1
fi
cd "${ROOT}"

# --- build KaHIP ---
echo "Building KaHIP 3.19..."
if (
  cd "${ROOT}/extern/KaHIP" \
  && mkdir -p build && cd build \
  && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ROOT}/extern/local/kahip" \
    -DNOMPI=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    > /dev/null 2>&1 \
  && make install -j "$JOBS" > /dev/null 2>&1
); then
  echo "KaHIP 3.19 build completed successfully."
else
  echo "KaHIP 3.19 build failed!" >&2
  exit 1
fi
cd "${ROOT}"

# --- build GPU-HeiPa ---
echo "Building GPU-HeiPa..."
rm -rf build && mkdir -p build
cd build
# Ensure our Kokkos/Kernels are found first
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos;${ROOT}/extern/local/kokkos-kernels" \
         -DCMAKE_CXX_STANDARD=17 \
         -DCMAKE_CXX_EXTENSIONS=OFF
cmake --build . --parallel "$JOBS" --target GPU-HeiPa
cmake --build . --parallel "$JOBS" --target GPU-HeiProMap
