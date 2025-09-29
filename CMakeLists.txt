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

if(NOT KAHIP_LIBRARY OR NOT KAHIP_INCLUDE_DIR)
    message(FATAL_ERROR "KaHIP library not found. Please run ./build_kahip.sh first")
endif()

message(STATUS "Found KaHIP: ${KAHIP_LIBRARY}")
include_directories(${KAHIP_INCLUDE_DIR})

# --- Kokkos Setup ---
set(KokkosKernels_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kokkos-kernels/lib/cmake/KokkosKernels")
find_package(KokkosKernels CONFIG REQUIRED)
find_package(Kokkos REQUIRED)

# --- Compiler Flags ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pipe -finline-functions -ftree-vectorize -funroll-loops -falign-loops -fno-rtti -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pipe -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion -pedantic")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "src/*.h")

add_executable(GPU-HeiPa main.cpp ${SOURCES} ${HEADERS})

target_link_libraries(GPU-HeiPa
        PRIVATE
        ${KAHIP_LIBRARY}
        Kokkos::kokkos
)
