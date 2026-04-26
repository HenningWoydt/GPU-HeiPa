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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- openMP setup ---
find_package(OpenMP REQUIRED)


# --- Kokkos Setup ---
# Respect externally provided Kokkos paths (e.g., via CMAKE_PREFIX_PATH).
# Fall back to the local layout created by build.sh when available.
if(NOT Kokkos_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kokkos/lib/cmake/Kokkos")
    set(Kokkos_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kokkos/lib/cmake/Kokkos")
endif()

if(NOT KokkosKernels_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kokkos-kernels/lib/cmake/KokkosKernels")
    set(KokkosKernels_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/local/kokkos-kernels/lib/cmake/KokkosKernels")
endif()

find_package(KokkosKernels CONFIG REQUIRED)
find_package(Kokkos REQUIRED)

# --- Compiler Flags ---
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -pipe -finline-functions -ftree-vectorize -funroll-loops -falign-loops -fno-rtti -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -DNDEBUG -pipe -fno-inline-functions -ftree-vectorize -funroll-loops -falign-loops -fno-rtti -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion")
set(CMAKE_CXX_FLAGS_DEBUG "-g -march=native -pipe -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wconversion -Wsign-conversion -Wfloat-conversion -pedantic")

option(ENABLE_PROFILER "Enable profiler" OFF)
option(ASSERT_ENABLED "Enable assertions" OFF)
option(GPU_HEIPA_BUILD_APPS "Build GPU-HeiPa executables" ${PROJECT_IS_TOP_LEVEL})

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "src/*.h")

# Header-centric library target for embedding into parent projects.
add_library(gpu_heipa INTERFACE)
add_library(GPU_HeiPa::gpu_heipa ALIAS gpu_heipa)
target_include_directories(gpu_heipa INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(gpu_heipa INTERFACE
        Kokkos::kokkos
        Kokkos::kokkoskernels
        OpenMP::OpenMP_CXX
)

if(ASSERT_ENABLED)
    target_compile_definitions(gpu_heipa INTERFACE ASSERT_ENABLED=1)
endif()

if(ENABLE_PROFILER)
    target_compile_definitions(gpu_heipa INTERFACE ENABLE_PROFILER=1)
endif()

if(GPU_HEIPA_BUILD_APPS)
    add_executable(GPU-HeiPa app/GPU-HeiPa.cpp ${SOURCES} ${HEADERS})
    #add_executable(GPU-HeiProMap app/GPU-HeiProMap.cpp ${SOURCES} ${HEADERS})
    #add new executable for new solver (to execute both)
    #add_executable(GPU-HeiPa2k app/GPU-HeiPa-k2.cpp ${SOURCES} ${HEADERS})
    add_executable(GPU-HeiPa-memetic app/GPU-HeiPa-memetic.cpp ${SOURCES} ${HEADERS})
    #add_executable(shrinking app/GPU-HeiPa-memeticShrinking.cpp ${SOURCES} ${HEADERS})

    target_link_libraries(GPU-HeiPa PRIVATE gpu_heipa)

#target_link_libraries(GPU-HeiProMap
#        PRIVATE
#        Kokkos::kokkos
#        Kokkos::kokkoskernels
#)

    target_link_libraries(GPU-HeiPa-memetic PRIVATE gpu_heipa)

    # Apply ASSERT_ENABLED flag to all targets
    if(ASSERT_ENABLED)
        target_compile_definitions(GPU-HeiPa PRIVATE ASSERT_ENABLED=1)
        #target_compile_definitions(GPU-HeiProMap PRIVATE ASSERT_ENABLED=1)
        target_compile_definitions(GPU-HeiPa-memetic PRIVATE ASSERT_ENABLED=1)
    endif()

    if(ENABLE_PROFILER)
        target_compile_definitions(GPU-HeiPa PRIVATE ENABLE_PROFILER=1)
        #target_compile_definitions(GPU-HeiProMap PRIVATE ENABLE_PROFILER=1)
        target_compile_definitions(GPU-HeiPa-memetic PRIVATE ENABLE_PROFILER=1)
    endif()

#target_link_libraries(shrinking
#        PRIVATE
#        ${KAHIP_LIBRARY}
#        ${METIS_LIBRARY}
#        ${GKLIB_LIBRARY}
#        Kokkos::kokkos
#        Kokkos::kokkoskernels
#        OpenMP::OpenMP_CXX
#)

#target_link_libraries(GPU-HeiProMap
#        PRIVATE
#        ${KAHIP_LIBRARY}
#        ${METIS_LIBRARY}
#        ${GKLIB_LIBRARY}
#        Kokkos::kokkos
#        Kokkos::kokkoskernels
#        OpenMP::OpenMP_CXX
#)

    # Disable CUDA lazy loading via linker flags
    if(Kokkos_ENABLE_CUDA)
        target_link_options(GPU-HeiPa PRIVATE "LINKER:--require-defined,cudaDeviceSynchronize")
        target_link_options(GPU-HeiPa-memetic PRIVATE "LINKER:--require-defined,cudaDeviceSynchronize")
        #target_link_options(shrinking PRIVATE "LINKER:--require-defined,cudaDeviceSynchronize")
        #target_link_options(GPU-HeiProMap PRIVATE "LINKER:--require-defined,cudaDeviceSynchronize")
    endif()
endif()
