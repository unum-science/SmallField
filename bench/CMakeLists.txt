# Benchmark executables linking against the SmallField library + RDKit for cross-validation.

add_executable(bench_conformer_backends bench_conformer_backends.cpp)
set_source_files_properties(bench_conformer_backends.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(bench_conformer_backends PRIVATE
    smallfield_lib
    OpenMP::OpenMP_CXX
    $<$<BOOL:${JEMALLOC_LIB}>:${JEMALLOC_LIB}>)
# The repo root is on the path for `test/test_rdkit_reference_helpers.hpp`, which
# this benchmark shares with the test suite.
target_include_directories(bench_conformer_backends PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_conformer_backends PROPERTIES
    LINKER_LANGUAGE CUDA)

add_executable(bench_profile bench_profile.cpp)
set_source_files_properties(bench_profile.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(bench_profile PRIVATE smallfield_lib OpenMP::OpenMP_CXX $<$<BOOL:${JEMALLOC_LIB}>:${JEMALLOC_LIB}>)
target_include_directories(bench_profile PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_profile PROPERTIES LINKER_LANGUAGE CUDA)

# MMFF kernel scaling benchmark: measures GPU kernel trio isolation.
# Mixed .cpp/.cu build: .cpp drives RDKit + OpenMP CPU timing, .cu launches
# the kernels with cudaEvent timing.
add_executable(bench_mmff_kernels bench_mmff_kernels.cpp bench_mmff_kernels_device.cu)
set_source_files_properties(bench_mmff_kernels.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(bench_mmff_kernels PRIVATE smallfield_lib OpenMP::OpenMP_CXX $<$<BOOL:${JEMALLOC_LIB}>:${JEMALLOC_LIB}>)
target_include_directories(bench_mmff_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_mmff_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA)

