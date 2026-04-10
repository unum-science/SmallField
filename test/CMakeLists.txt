# C++ / CUDA test executables linking against RDKit for cross-validation.
# Each test is a standalone executable with its own main().

# shared test utilities (header-only)
add_library(test_helpers INTERFACE)
target_include_directories(test_helpers INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR})

# collect the RDKit targets we need
set(RDKIT_LIBS
    RDKit::SmilesParse
    RDKit::GraphMol
    RDKit::RDGeneral
    RDKit::ForceFieldHelpers
    RDKit::ForceField
    RDKit::DistGeomHelpers
    RDKit::DistGeometry
    RDKit::Depictor
    RDKit::SubstructMatch
    RDKit::Alignment
    RDKit::EigenSolvers
    RDKit::MolTransforms)

# RDKit headers use C++20 constexpr virtual, which nvcc cannot parse.
# Solution: compile .cpp files with the host compiler (g++) and .cu files
# with nvcc, then link them together.

macro(add_mixed_test test_name)
    add_executable(${test_name} ${test_name}.cpp ${test_name}_kernels.cu)
    set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CXX)
    target_link_libraries(${test_name} PRIVATE
        test_helpers
        ${RDKIT_LIBS}
        ${CURAND_LIBRARY}
        ${CUSOLVER_LIBRARY}
        cudart
        OpenMP::OpenMP_CXX)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src)
    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA)
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_mixed_test(test_triangle_smooth)
add_mixed_test(test_dg_energy)
add_mixed_test(test_chirality)
add_mixed_test(test_mmff_gradient)
add_mixed_test(test_pipeline)
