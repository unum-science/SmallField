# C++ / CUDA test executables linking against RDKit for cross-validation.
# Each test is a standalone executable with its own main().

# shared test utilities (header-only)
add_library(test_helpers INTERFACE)
target_include_directories(test_helpers INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR})

# Tests that use mixed .cpp/.cu compilation (g++ for RDKit headers, nvcc for CUDA kernels).
macro(add_mixed_test test_name)
    add_executable(${test_name} ${test_name}.cpp ${test_name}_kernels.cu)
    set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CXX)
    target_link_libraries(${test_name} PRIVATE
        test_helpers
        smallfield_lib
        OpenMP::OpenMP_CXX)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA)
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_mixed_test(test_triangle_smooth)
add_mixed_test(test_dg_energy)
add_mixed_test(test_chirality)
add_mixed_test(test_mmff_gradient)

# test_descriptor is a pure C++ unit test (no CUDA).
add_executable(test_descriptor test_descriptor.cpp)
target_link_libraries(test_descriptor PRIVATE test_helpers smallfield_lib)
target_include_directories(test_descriptor PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_descriptor PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME test_descriptor COMMAND test_descriptor)

# test_torsion_predicates cross-validates the hand-coded recursive-SMARTS predicates
# against RDKit's SubstructMatch on a large set of ChEMBL-derived molecules.
add_executable(test_torsion_predicates test_torsion_predicates.cpp)
target_link_libraries(test_torsion_predicates PRIVATE test_helpers smallfield_lib)
target_include_directories(test_torsion_predicates PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_torsion_predicates PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME test_torsion_predicates COMMAND test_torsion_predicates)

# test_pipeline uses the library API directly (no separate _kernels.cu)
add_executable(test_pipeline test_pipeline.cpp)
set_source_files_properties(test_pipeline.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(test_pipeline PRIVATE
    test_helpers
    smallfield_lib
    OpenMP::OpenMP_CXX)
target_include_directories(test_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_pipeline PROPERTIES
    LINKER_LANGUAGE CUDA)
add_test(NAME test_pipeline COMMAND test_pipeline)
