cmake_minimum_required(VERSION 3.16)
project(GPU-HeiPa)

set(PROJECT_VENDOR "Henning Woydt")
set(PROJECT_CONTACT "henning.woydt@informatik.uni-heidelberg.de")
set(PROJECT_URL "https://github.com/HenningWoydt/GPU_HeiProMap")
set(PROJECT_DESCRIPTION "GPU-HeiPa: Heidelberger Partitioning on GPU")
set(PROJECT_VERSION "1.0.0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Global language/standard policy ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- KaHIP library ---
set(KaHIP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kahip")
find_library(KAHIP_LIBRARY
        NAMES kahip libkahip
        PATHS ${KaHIP_ROOT}/lib
        NO_DEFAULT_PATH
)
find_path(KAHIP_INCLUDE_DIR
        NAMES kaHIP_interface.h
        PATHS ${KaHIP_ROOT}/include
        NO_DEFAULT_PATH
)

message(STATUS "Found KaHIP: ${KAHIP_LIBRARY}")
include_directories(${KAHIP_INCLUDE_DIR})

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


# --- Kokkos Setup ---
set(KokkosKernels_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kokkos-kernels/lib/cmake/KokkosKernels")
find_package(KokkosKernels CONFIG REQUIRED)
find_package(Kokkos REQUIRED)

# --- OpenMP Setup ---
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
endif()

# --- Compiler Flags ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -pipe -finline-functions -ftree-vectorize -funroll-loops -falign-loops -fno-rtti -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -DNDEBUG -pipe -fno-inline-functions -ftree-vectorize -funroll-loops -falign-loops -fno-rtti -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion")
set(CMAKE_CXX_FLAGS_DEBUG "-g -march=native -pipe -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion -pedantic")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "src/*.h")

add_executable(GPU-HeiPa main.cpp ${SOURCES} ${HEADERS})

target_link_libraries(GPU-HeiPa
        PRIVATE
        ${KAHIP_LIBRARY}
        ${METIS_LIBRARY}
        ${GKLIB_LIBRARY}
        Kokkos::kokkos
        OpenMP::OpenMP_CXX
)
