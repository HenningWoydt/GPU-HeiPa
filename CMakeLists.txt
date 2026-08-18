cmake_minimum_required(VERSION 3.16)
project(GPU-HeiPa)

set(PROJECT_VENDOR "Henning Woydt")
set(PROJECT_CONTACT "henning.woydt@informatik.uni-heidelberg.de")
set(PROJECT_URL "https://github.com/HenningWoydt/GPU_HeiProMap")
set(PROJECT_DESCRIPTION "GPU-HeiPa: Heidelberger Partitioning on GPU")
set(PROJECT_VERSION "1.0.0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Global language/standard policy ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Kokkos Setup ---
set(KokkosKernels_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kokkos-kernels/lib/cmake/KokkosKernels")
find_package(KokkosKernels CONFIG REQUIRED)
find_package(Kokkos REQUIRED)

# --- GKlib library ---
set(GKLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/GKlib")
find_library(GKLIB_LIBRARY
        NAMES GKlib libGKlib gklib libgklib
        PATHS ${GKLIB_ROOT}/lib
        NO_DEFAULT_PATH
)
find_path(GKLIB_INCLUDE_DIR
        NAMES gk_types.h GKlib.h
        PATHS ${GKLIB_ROOT}/include
        NO_DEFAULT_PATH
)
message(STATUS "Found GKlib: ${GKLIB_LIBRARY}")
include_directories(${GKLIB_INCLUDE_DIR})

# --- METIS library ---
set(METIS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/METIS")
find_library(METIS_LIBRARY
        NAMES metis libmetis
        PATHS ${METIS_ROOT}/lib
        NO_DEFAULT_PATH
)
find_path(METIS_INCLUDE_DIR
        NAMES metis.h
        PATHS ${METIS_ROOT}/include
        NO_DEFAULT_PATH
)
message(STATUS "Found METIS: ${METIS_LIBRARY}")
include_directories(${METIS_INCLUDE_DIR})

# --- Compiler Flags ---
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -pipe -finline-functions -ftree-vectorize -funroll-loops -falign-loops -fno-rtti -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -march=native -DNDEBUG -pipe -fno-inline-functions -ftree-vectorize -funroll-loops -falign-loops -fno-rtti -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -pipe -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion -pedantic")

option(ENABLE_PROFILER "Enable profiler" OFF)
option(ASSERT_ENABLED "Enable assertions" OFF)

if(ENABLE_PROFILER)
    add_compile_definitions(ENABLE_PROFILER=1)
endif()

if(ASSERT_ENABLED)
    add_compile_definitions(ASSERT_ENABLED=1)
endif()


file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "src/*.h")

# --- Kway Partitioning (Warnings Suppressed) ---
file(GLOB_RECURSE KWAY_SOURCES CONFIGURE_DEPENDS
        "src/initial_partitioning/kway_partitioner/*.cpp"
)
file(GLOB_RECURSE KWAY_HEADERS CONFIGURE_DEPENDS
        "src/initial_partitioning/kway_partitioner/*.h"
)

# Remove kway sources from the general executable source list
list(REMOVE_ITEM SOURCES ${KWAY_SOURCES})
list(REMOVE_ITEM HEADERS ${KWAY_HEADERS})

add_library(kway_lib ${KWAY_SOURCES} ${KWAY_HEADERS})

target_compile_options(kway_lib PRIVATE
        -w
)

target_include_directories(kway_lib SYSTEM PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/initial_partitioning/kway_partitioner"
)

target_link_libraries(kway_lib PUBLIC
        Kokkos::kokkos
        Kokkos::kokkoskernels
)

add_executable(GPU-HeiPa app/GPU-HeiPa.cpp ${SOURCES} ${HEADERS})
add_executable(GPU-HeiProMap app/GPU-HeiProMap.cpp ${SOURCES} ${HEADERS})

target_link_libraries(GPU-HeiPa PRIVATE kway_lib ${METIS_LIBRARY} ${GKLIB_LIBRARY})
target_link_libraries(GPU-HeiProMap PRIVATE kway_lib ${METIS_LIBRARY} ${GKLIB_LIBRARY})


# Disable CUDA lazy loading via linker flags
if(Kokkos_ENABLE_CUDA)
    target_link_options(GPU-HeiPa PRIVATE "LINKER:--require-defined,cudaDeviceSynchronize")
    target_link_options(GPU-HeiProMap PRIVATE "LINKER:--require-defined,cudaDeviceSynchronize")
endif()


target_compile_options(GPU-HeiPa PRIVATE
        -Wno-sign-conversion
)

target_compile_options(GPU-HeiProMap PRIVATE
        -Wno-sign-conversion
)

