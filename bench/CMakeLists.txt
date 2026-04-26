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

# MACE-OFF inference batched-throughput benchmark.  Same source compiles
# under both the fp32 reference build and the ThunderKittens (TK) build;
# the build flavour is selected at compile time by `SMALLFIELD_USE_THUNDERKITTENS`.
# The CMake target below tracks whatever flavour `smallfield_lib` was built
# with — the two-flavour comparison reported in `MACE_BENCH_REPORT.md` is
# produced by the standalone `nvcc` invocations documented in the bench
# file's header comment, not by this CMake target.
add_executable(bench_mace_inference bench_mace_inference.cu)
target_link_libraries(bench_mace_inference PRIVATE smallfield_lib)
target_include_directories(bench_mace_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_mace_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA)
