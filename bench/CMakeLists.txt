# Benchmark executables linking against the SmallField library + RDKit for cross-validation.

add_executable(bench_conformers bench_conformers.cpp)
set_source_files_properties(bench_conformers.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(bench_conformers PRIVATE
    smallfield_lib
    OpenMP::OpenMP_CXX
    $<$<BOOL:${JEMALLOC_LIB}>:${JEMALLOC_LIB}>)
target_include_directories(bench_conformers PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_conformers PROPERTIES
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

