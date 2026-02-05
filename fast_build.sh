#!/usr/bin/env bash
 set -euo pipefail

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

 ROOT=${PWD}
 GCC=$(which gcc || true)

 echo "Root          : ${ROOT}"
 echo "Using C compiler: ${GCC:-<system default>}"


 # --- build GPU-HeiPa ---
 echo "Building GPU-HeiPa..."
 cd build
 # Ensure our Kokkos/Kernels are found first
 cmake .. -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos;${ROOT}/extern/local/kokkos-kernels" \
          -DCMAKE_CXX_STANDARD=20 \
          -DCMAKE_CXX_EXTENSIONS=OFF \
          -DENABLE_PROFILER=0
 cmake --build . --parallel "$JOBS" --target GPU-HeiPa
 # cmake --build . --parallel "$JOBS" --target GPU-HeiProMap
