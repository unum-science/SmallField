# Benchmark executables linking against the SmallField library + RDKit for cross-validation.

add_executable(bench_pipeline_throughput bench_pipeline_throughput.cpp)
set_source_files_properties(bench_pipeline_throughput.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(bench_pipeline_throughput PRIVATE
    smallfield_lib
    OpenMP::OpenMP_CXX)
target_include_directories(bench_pipeline_throughput PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_pipeline_throughput PROPERTIES
    LINKER_LANGUAGE CUDA)

add_executable(profile_pipeline profile_pipeline.cpp)
set_source_files_properties(profile_pipeline.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(profile_pipeline PRIVATE smallfield_lib OpenMP::OpenMP_CXX)
target_include_directories(profile_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(profile_pipeline PROPERTIES LINKER_LANGUAGE CUDA)

# MMFF kernel scaling benchmark: measures GPU kernel trio isolation.
# Mixed .cpp/.cu build: .cpp drives RDKit + OpenMP CPU timing, .cu launches
# the kernels with cudaEvent timing.
add_executable(bench_mmff_kernels bench_mmff_kernels.cpp bench_mmff_kernels_device.cu)
set_source_files_properties(bench_mmff_kernels.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(bench_mmff_kernels PRIVATE smallfield_lib OpenMP::OpenMP_CXX)
target_include_directories(bench_mmff_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_mmff_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA)
