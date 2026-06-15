#!/usr/bin/env bash
set -euo pipefail

# Parse arguments
DOWNLOAD_KOKKOS="OFF" # Default to ON, can be overridden by AUTO or OFF
DOWNLOAD_METIS="OFF"
ENABLE_PROFILER="OFF"
ASSERT_ENABLED="OFF"
BUILD_TESTING="OFF"
MAX_THREADS=""
KOKKOS_ARCH=""
SILENT="ON"

for arg in "$@"; do
  case "$arg" in
    --enable-profiler=*) 
      ENABLE_PROFILER="${arg#*=}" 
      ;; 
    --assert-enabled=*) 
      ASSERT_ENABLED="${arg#*=}" 
      ;;
    --download-kokkos=*)
      DOWNLOAD_KOKKOS="${arg#*=}"
      ;;
    --download-metis=*)
      DOWNLOAD_METIS="${arg#*=}"
      ;;
    --test)
      BUILD_TESTING="ON"
      ;;
    --max-threads=*)
      MAX_THREADS="${arg#*=}"
      ;;
    --kokkos-arch=*)
      KOKKOS_ARCH="${arg#*=}"
      ;;
    --silent)
      SILENT="ON"
      ;;
    --verbose)
      SILENT="OFF"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--download-kokkos=ON|OFF|AUTO] [--download-metis=ON|OFF|AUTO] [--max-threads=N] [--kokkos-arch=ARCH] [--test] [--verbose]"
      exit 1
      ;;
  esac
done

SUCCESS="false"
if [ "$SILENT" = "ON" ]; then
  LOG_FILE=$(mktemp)
  # Save original stderr to fd 4 to ensure we can print the log on error
  exec 4>&2
  exec > "$LOG_FILE" 2>&1
  trap 'if [ "$SUCCESS" = "false" ]; then cat "$LOG_FILE" >&4; fi; rm -f "$LOG_FILE"' EXIT
fi

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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCC=$(which gcc || true)

echo "Root            : ${ROOT}"
echo "Using C compiler: ${GCC:-<system default>}"

SHOULD_DOWNLOAD_KOKKOS="false"

if [ "$DOWNLOAD_KOKKOS" = "ON" ]; then
  SHOULD_DOWNLOAD_KOKKOS="true"
elif [ "$DOWNLOAD_KOKKOS" = "AUTO" ]; then
  KOKKOS_LOCAL_DIR="${ROOT}/extern/local/kokkos"
  KOKKOS_KERNELS_LOCAL_DIR="${ROOT}/extern/local/kokkos-kernels"

  # Debugging output for AUTO mode check
  echo "Checking for Kokkos installation at: ${KOKKOS_LOCAL_DIR}/lib/cmake/Kokkos"
  echo "Checking for KokkosKernels installation at: ${KOKKOS_KERNELS_LOCAL_DIR}/lib/cmake/KokkosKernels"
  
  # Check for existence of the Kokkos installation directory
  if [ -d "${KOKKOS_LOCAL_DIR}/lib/cmake/Kokkos" ]; then
    echo "Found Kokkos directory: ${KOKKOS_LOCAL_DIR}/lib/cmake/Kokkos"
    KOKKOS_FOUND="true"
  else
    echo "Kokkos directory NOT found at: ${KOKKOS_LOCAL_DIR}/lib/cmake/Kokkos"
    KOKKOS_FOUND="false"
  fi

  if [ -d "${KOKKOS_KERNELS_LOCAL_DIR}/lib/cmake/KokkosKernels" ]; then
    echo "Found KokkosKernels directory: ${KOKKOS_KERNELS_LOCAL_DIR}/lib/cmake/KokkosKernels"
    KOKKOS_KERNELS_FOUND="true"
  else
    echo "KokkosKernels directory NOT found at: ${KOKKOS_KERNELS_LOCAL_DIR}/lib/cmake/KokkosKernels"
    KOKKOS_KERNELS_FOUND="false"
  fi

  if [ "$KOKKOS_FOUND" = "true" ] && [ "$KOKKOS_KERNELS_FOUND" = "true" ]; then
    echo "Existing Kokkos installation detected (AUTO mode). Skipping download and build."
    SHOULD_DOWNLOAD_KOKKOS="false"
  else
    echo "Existing Kokkos installation not detected (AUTO mode). Proceeding with download and build."
    SHOULD_DOWNLOAD_KOKKOS="true"
  fi
elif [ "$DOWNLOAD_KOKKOS" = "OFF" ]; then
  SHOULD_DOWNLOAD_KOKKOS="false"
else
  echo "Error: Invalid value for --download-kokkos. Must be ON, OFF, or AUTO." >&2
  exit 1
fi

SHOULD_DOWNLOAD_METIS="false"
if [ "$DOWNLOAD_METIS" = "ON" ]; then
  SHOULD_DOWNLOAD_METIS="true"
elif [ "$DOWNLOAD_METIS" = "AUTO" ]; then
  METIS_LOCAL_DIR="${ROOT}/extern/local/METIS"
  if [ -d "${METIS_LOCAL_DIR}/lib" ] && [ -d "${METIS_LOCAL_DIR}/include" ]; then
    echo "Existing METIS installation detected (AUTO mode). Skipping download and build."
    SHOULD_DOWNLOAD_METIS="false"
  else
    echo "Existing METIS installation not detected (AUTO mode). Proceeding with download and build."
    SHOULD_DOWNLOAD_METIS="true"
  fi
elif [ "$DOWNLOAD_METIS" = "OFF" ]; then
  SHOULD_DOWNLOAD_METIS="false"
else
  echo "Error: Invalid value for --download-metis. Must be ON, OFF, or AUTO." >&2
  exit 1
fi

echo "==> Download Kokkos: ${DOWNLOAD_KOKKOS} (Effective: $SHOULD_DOWNLOAD_KOKKOS)"
echo "==> Download METIS: ${DOWNLOAD_METIS} (Effective: $SHOULD_DOWNLOAD_METIS)"

# ---- detect GPU arch and map to Kokkos flag ----
detect_kokkos_arch() {
  # Use command-line argument if provided
  if [ -n "${KOKKOS_ARCH}" ]; then
    if [[ "${KOKKOS_ARCH}" == Kokkos_ARCH_* ]]; then
      echo "${KOKKOS_ARCH}=ON"
    else
      echo "Kokkos_ARCH_${KOKKOS_ARCH}=ON"
    fi
    return 0
  fi

  # Try nvidia-smi compute capability
  if command -v nvidia-smi >/dev/null 2>&1; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -nr | head -n1)
    if [ -z "$cc" ]; then return 0; fi
    
    case "$cc" in
      120) echo "Kokkos_ARCH_BLACKWELL120=ON" ;;
      90)  echo "Kokkos_ARCH_HOPPER90=ON" ;;
      89)  echo "Kokkos_ARCH_ADA89=ON" ;;
      86)  echo "Kokkos_ARCH_AMPERE86=ON" ;;
      80)  echo "Kokkos_ARCH_AMPERE80=ON" ;;
      75)  echo "Kokkos_ARCH_TURING75=ON" ;;
      70)  echo "Kokkos_ARCH_VOLTA70=ON" ;;
      *)
        return 0
        ;;
    esac
    return 0
  fi

  # Last resort: let Kokkos autodetect
  echo ""
}

ARCH_FLAG="$(detect_kokkos_arch)"
if [ -n "${ARCH_FLAG}" ]; then
  echo "==> GPU architecture detected! Kokkos flag: ${ARCH_FLAG}"
else
  echo "==> GPU architecture NOT detected. Kokkos will attempt auto-detection during build."
fi

# ----- pick a reasonable parallelism (leave 2 cores free) -----
calc_jobs() {
  local cores
  cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  local j=$(( cores - 2 ))
  if [ "$j" -lt 1 ]; then j=1; fi
  echo "$j"
}

# Use command-line argument if provided, otherwise calculate
if [ -n "${MAX_THREADS}" ]; then
  JOBS="${MAX_THREADS}"
else
  JOBS="$(calc_jobs)"
fi
echo "Building with $JOBS parallel jobs."

# make local folder for all includes if it doesn't exist
mkdir -p extern/local

if [ "$SHOULD_DOWNLOAD_METIS" = "true" ]; then
  # ---- Download and build METIS dependencies ----
  rm -rf extern/METIS
  rm -rf extern/GKlib
  rm -rf extern/local/METIS
  rm -rf extern/local/GKlib

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

  # --- Build GKlib ---
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

  # --- Build METIS ---
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
fi


if [ "$SHOULD_DOWNLOAD_KOKKOS" = "true" ]; then
  # ---- Download and build Kokkos dependencies ----
  
  # clean previous externals
  rm -rf extern/local/kokkos
  rm -rf extern/local/kokkos-kernels
  rm -rf extern/kokkos-5.0.0
  rm -rf extern/kokkos-kernels-5.0.0

  # --- Download Kokkos-Kernels 5.0.0 ---
  echo "Downloading Kokkos-Kernels 5.0.0..."
  if (cd extern && rm -f kokkos-kernels-5.0.0.tar.gz && rm -rf kokkos-kernels-5.0.0 && wget -q https://github.com/kokkos/kokkos-kernels/releases/download/5.0.0/kokkos-kernels-5.0.0.tar.gz && tar -xzf kokkos-kernels-5.0.0.tar.gz && rm -f kokkos-kernels-5.0.0.tar.gz); then
    echo "Kokkos-Kernels 5.0.0 downloaded and extracted successfully."
  else
    echo "Failed to download Kokkos-Kernels!" >&2
    exit 1
  fi

  # --- Download Kokkos 5.0.0 ---
  echo "Downloading Kokkos 5.0.0..."
  if (cd extern && rm -f kokkos-5.0.0.tar.gz && rm -rf kokkos-5.0.0 && wget -q https://github.com/kokkos/kokkos/releases/download/5.0.0/kokkos-5.0.0.tar.gz && tar -xzf kokkos-5.0.0.tar.gz && rm -f kokkos-5.0.0.tar.gz); then
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
  KOKKOS_COMMON="-DCMAKE_INSTALL_PREFIX=${ROOT}/extern/local/kokkos 
                 -DKokkos_ENABLE_SERIAL=ON 
                 -DCMAKE_BUILD_TYPE=Release 
                 -DKokkos_ENABLE_DEBUG=OFF 
                 -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF 
                 -DKokkos_ENABLE_TUNING=ON"

  KOKKOS_BACKEND="-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA_LAMBDA=ON"

  # Strong optimization defaults for Release
  CXX_RELEASE_FLAGS="-O3 -DNDEBUG -march=native -mtune=native -fno-math-errno -fomit-frame-pointer"

  # --- build kokkos ---
  echo "Building Kokkos 5.0.0..."
  if (cd "${ROOT}/extern/kokkos-5.0.0" && mkdir -p build && cd build && cmake .. ${KOKKOS_COMMON} ${KOKKOS_BACKEND} -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_EXTENSIONS=OFF -DENABLE_PROFILER=${ENABLE_PROFILER} -DASSERT_ENABLED=${ASSERT_ENABLED} -DCMAKE_CXX_FLAGS_RELEASE="${CXX_RELEASE_FLAGS}" -DCMAKE_CXX_FLAGS="-w" ${ARCH_FLAG:+-D${ARCH_FLAG}} && make install -j "$JOBS"); then
    echo "Kokkos 5.0.0 build completed successfully."
  else
    echo "Kokkos 5.0.0 build failed!" >&2
    exit 1
  fi

  echo "Building Kokkos-Kernels 5.0.0..."
  if (cd "${ROOT}/extern/kokkos-kernels-5.0.0" && mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX="${ROOT}/extern/local/kokkos-kernels" -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_EXTENSIONS=OFF -DENABLE_PROFILER=${ENABLE_PROFILER} -DASSERT_ENABLED=${ASSERT_ENABLED} -DKokkosKernels_ENABLE_TESTS=OFF -DKokkosKernels_ENABLE_EXAMPLES=OFF -DKokkosKernels_ENABLE_PERFTESTS=OFF -DCMAKE_CXX_FLAGS_RELEASE="${CXX_RELEASE_FLAGS}" -DCMAKE_CXX_FLAGS="-w" ${KOKKOS_BACKEND} && make install -j "$JOBS"); then
    echo "Kokkos-Kernels 5.0.0 build completed successfully."
  else
    echo "Kokkos-Kernels 5.0.0 build failed!" >&2
    exit 1
  fi

  cd "${ROOT}"
else
  echo "Skipping Kokkos download and build (using existing installation)."
  # Still try to set CXX if it's there
  if [ -x "${ROOT}/extern/kokkos-5.0.0/bin/nvcc_wrapper" ]; then
    export CXX="${ROOT}/extern/kokkos-5.0.0/bin/nvcc_wrapper"
    echo "Using C++ compiler: ${CXX}"
  fi
fi

# --- build GPU-HeiPa ---
echo "Building GPU-HeiPa..."
if [ "$SHOULD_DOWNLOAD_KOKKOS" = "true" ]; then
  # Only clean build directory if we are doing a full rebuild of Kokkos.
  # Otherwise, we want to reuse the existing build directory which might link to an existing Kokkos installation.
  rm -rf "${ROOT}/build"
fi
mkdir -p "${ROOT}/build"
cd "${ROOT}/build"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos;${ROOT}/extern/local/kokkos-kernels" -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_EXTENSIONS=OFF -DENABLE_PROFILER=${ENABLE_PROFILER} -DASSERT_ENABLED=${ASSERT_ENABLED} -DBUILD_TESTING=${BUILD_TESTING}
cmake --build . --parallel "$JOBS" --target GPU-HeiPa
cmake --build . --parallel "$JOBS" --target GPU-HeiProMap

if [ "$BUILD_TESTING" = "ON" ]; then
  cmake --build . --parallel "$JOBS" --target unit_tests
  echo "Running tests..."
  ./tests/unit_tests
fi

SUCCESS="true"
