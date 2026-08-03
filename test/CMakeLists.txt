# C++ / CUDA test executables linking against RDKit for cross-validation. Each
# test is a standalone executable with its own main().  Tests are organised into
# subfolders mirroring src/: api, parameters, preprocess, mmff94, kernels.

# shared test utilities (header-only)
add_library(test_helpers INTERFACE)
target_include_directories(
  test_helpers INTERFACE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR})

# Mixed .cpp/.cu test: given a base name like "triangle_smooth" and a subfolder
# like "kernels", builds an executable from test_<base>.cpp +
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

# --- kernels/ ---
add_mixed_test(kernels triangle_smooth)
add_mixed_test(kernels dg_energy)
add_mixed_test(kernels mmff_gradient)

# --- preprocess/ ---
add_mixed_test(preprocess chirality)
add_cpp_test(api conformer_slot_layout)
add_cpp_test(preprocess descriptor)
add_cpp_test(preprocess native_murcko)
add_cpp_test(preprocess csd_torsion_predicates)
add_cpp_test(preprocess element_gate)

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
  kernels/test_mmff_gradient_kernels.cu)
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
add_executable(test_pipeline api/test_pipeline.cpp)
set_source_files_properties(api/test_pipeline.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(test_pipeline PRIVATE test_helpers smallfield_lib
  OpenMP::OpenMP_CXX)
target_include_directories(
  test_pipeline PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_pipeline PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME test_pipeline COMMAND test_pipeline)

add_executable(test_ez_stereo api/test_ez_stereo.cpp)
set_source_files_properties(api/test_ez_stereo.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(test_ez_stereo PRIVATE test_helpers smallfield_lib
  OpenMP::OpenMP_CXX)
target_include_directories(
  test_ez_stereo PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_ez_stereo PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME test_ez_stereo COMMAND test_ez_stereo)
