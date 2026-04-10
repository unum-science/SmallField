# C++ / CUDA benchmark executables linking against RDKit for comparison.

# reuse the same RDKit targets from the parent
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

macro(add_mixed_bench bench_name)
    add_executable(${bench_name} ${bench_name}.cpp ${bench_name}_kernels.cu)
    set_source_files_properties(${bench_name}.cpp PROPERTIES LANGUAGE CXX)
    target_link_libraries(${bench_name} PRIVATE
        ${RDKIT_LIBS}
        ${CURAND_LIBRARY}
        cudart
        OpenMP::OpenMP_CXX)
    target_include_directories(${bench_name} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${bench_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA)
endmacro()

add_mixed_bench(bench_chembl)
