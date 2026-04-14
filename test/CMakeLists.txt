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

# test_mmff94_tables (MG-0 + MG-A gate): cross-validate our extracted
# MMFF94 parameter tables against RDKit's live operator() lookups on
# host, and against device-side lookups via CUDA kernels.  Must be
# bitwise-equal on both sides for thousands of random keys.
add_mixed_test(test_mmff94_tables)

# test_mmff_cpu_port (MG-1 gate): cross-validate our native MMFF CPU port
# against the existing RDKit-backed path on a drug-like probe set.
add_executable(test_mmff_cpu_port test_mmff_cpu_port.cpp)
target_link_libraries(test_mmff_cpu_port PRIVATE test_helpers smallfield_lib)
target_include_directories(test_mmff_cpu_port PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_mmff_cpu_port PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME test_mmff_cpu_port COMMAND test_mmff_cpu_port)

# test_mmff_typer (MG-1.5/1.6 gate): native atom typer vs RDKit.
add_executable(test_mmff_typer test_mmff_typer.cpp)
target_link_libraries(test_mmff_typer PRIVATE test_helpers smallfield_lib)
target_include_directories(test_mmff_typer PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_mmff_typer PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME test_mmff_typer COMMAND test_mmff_typer)

# test_native_murcko cross-validates compute_native_murcko against RDKit's
# MurckoDecompose on a 500-molecule ChEMBL set.  Gate: ≥ 99 % agreement.
add_executable(test_native_murcko test_native_murcko.cpp)
target_link_libraries(test_native_murcko PRIVATE test_helpers smallfield_lib)
target_include_directories(test_native_murcko PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_native_murcko PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME test_native_murcko COMMAND test_native_murcko)

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
