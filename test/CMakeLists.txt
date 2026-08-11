# C++ / CUDA test executables linking against RDKit for cross-validation. Each
# test is a standalone executable with its own main().  Tests are organised into
# subfolders mirroring src/: api, cuda, parameters, preprocess, mmff94, postprocess.

# Shared test utilities (header-only).  Only two anchors are exposed: the repo
# root, so a cross-directory include reads "test/cuda/foo.hpp", and src/, so a
# library include reads "smallfield/...".  The test directory itself is
# deliberately absent — a bare "foo.hpp" must resolve as the includer's own
# sidecar and nothing else.
add_library(test_helpers INTERFACE)
target_include_directories(
  test_helpers INTERFACE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)

# Mixed .cpp/.cu test: given a base name like "triangle_smooth" and a subfolder
# like "cuda", builds an executable from test_<base>.cpp +
# test_<base>_kernels.cu.
macro(add_mixed_test subfolder base_name)
  set(test_name test_${base_name})
  add_executable(${test_name} ${subfolder}/${test_name}.cpp
    ${subfolder}/${test_name}_kernels.cu)
  set_source_files_properties(${subfolder}/${test_name}.cpp PROPERTIES LANGUAGE
    CXX)
  target_link_libraries(${test_name} PRIVATE test_helpers smallfield_lib
    OpenMP::OpenMP_CXX)
  target_include_directories(
    ${test_name} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA)
  add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

# Pure C++ test (no CUDA kernels).  Still linked with CUDA because
# smallfield_lib is CUDA.
macro(add_cpp_test subfolder base_name)
  set(test_name test_${base_name})
  add_executable(${test_name} ${subfolder}/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE test_helpers smallfield_lib
    OpenMP::OpenMP_CXX)
  target_include_directories(
    ${test_name} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CUDA)
  add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

# --- cuda/ ---
add_mixed_test(cuda triangle_smooth)
add_mixed_test(cuda dg_energy)
add_mixed_test(cuda mmff_gradient)
add_mixed_test(cuda bond_distance)
add_mixed_test(cuda csd_torsion_device)

# The reproducibility check compiles metric_embedding_prepass.cuh itself so it can drive the
# embed kernel over thousands of identical blocks. It therefore must not also link
# smallfield_lib, which carries its own device copy of that kernel: nvlink would see the symbol
# defined twice. RDKit comes in only for the test_rdkit_reference_helpers.hpp assertion macros.
add_executable(test_embedding_determinism cuda/test_embedding_determinism.cpp
  cuda/test_embedding_determinism_kernels.cu)
set_source_files_properties(cuda/test_embedding_determinism.cpp PROPERTIES
  LANGUAGE CXX)
target_link_libraries(test_embedding_determinism PRIVATE test_helpers
  ${RDKIT_LIBS} fmt::fmt)
target_include_directories(
  test_embedding_determinism PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_embedding_determinism PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA)
add_test(NAME test_embedding_determinism COMMAND test_embedding_determinism)


# --- preprocess/ ---
add_mixed_test(preprocess chirality)
add_cpp_test(api conformer_slot_layout)
add_cpp_test(preprocess descriptor)
add_cpp_test(preprocess csd_torsion_predicates)
add_cpp_test(preprocess element_gate)
add_cpp_test(preprocess chiral_parity)

# --- postprocess/ ---
add_cpp_test(postprocess shape_descriptors)

# --- parameters/ ---
add_mixed_test(parameters mmff94_tables)

# --- mmff94/ ---
add_mixed_test(mmff94 mmff_gpu_typer)
add_mixed_test(mmff94 mmff_gpu_extract)
add_cpp_test(mmff94 mmff_cpu_port)
add_cpp_test(mmff94 mmff_typer)

# Per-term parity reuses the single-molecule energy entry point that the gradient
# test's kernel wrapper already exposes.
add_executable(test_mmff_term_parity mmff94/test_mmff_term_parity.cpp
  cuda/test_mmff_gradient_kernels.cu)
set_source_files_properties(mmff94/test_mmff_term_parity.cpp PROPERTIES LANGUAGE
  CXX)
target_link_libraries(test_mmff_term_parity PRIVATE test_helpers smallfield_lib
  OpenMP::OpenMP_CXX)
target_include_directories(
  test_mmff_term_parity PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_mmff_term_parity PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA)
add_test(NAME test_mmff_term_parity COMMAND test_mmff_term_parity)

# --- api/ ---
add_cpp_test(api pipeline)
add_cpp_test(api seed_determinism)
add_cpp_test(api ez_stereo)
