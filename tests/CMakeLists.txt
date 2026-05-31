add_executable(unit_tests 
    test_runner.cpp
    gpu_bisection_partition_test.cpp
    gpu_cpu_comparison_test.cpp
)

target_link_libraries(unit_tests PRIVATE 
    GTest::gtest 
    Kokkos::kokkos
    Kokkos::kokkoskernels
    kway_lib
)

target_include_directories(unit_tests PRIVATE 
    ${PROJECT_SOURCE_DIR}/src
)

add_test(NAME unit_tests COMMAND unit_tests)
