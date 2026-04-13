# Benchmark executable linking against the SmallField library + RDKit for cross-validation.

add_executable(bench_chembl bench_chembl.cpp)
set_source_files_properties(bench_chembl.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(bench_chembl PRIVATE
    smallfield_lib
    OpenMP::OpenMP_CXX)
target_include_directories(bench_chembl PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bench_chembl PROPERTIES
    LINKER_LANGUAGE CUDA)

add_executable(profile_driver profile_driver.cpp)
set_source_files_properties(profile_driver.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(profile_driver PRIVATE smallfield_lib OpenMP::OpenMP_CXX)
target_include_directories(profile_driver PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(profile_driver PROPERTIES LINKER_LANGUAGE CUDA)
