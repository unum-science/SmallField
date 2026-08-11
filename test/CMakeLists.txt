# C/C++/CUDA test executables, each a standalone binary with its own main(), organised into subfolders mirroring src/:
# api, cuda, parameters, preprocess, mmff94, postprocess.
#
# A mixed test pairs `<subfolder>/test_<name>.cpp` with `<subfolder>/test_<name>_kernels.cu`.  The .cpp calls a plain
# host wrapper the .cu defines, which is ordinary host linkage -- no relocatable device code is involved, which is why
# `define_smallfield_target` leaves it off.
function (define_smallfield_test subfolder base_name)
    cmake_parse_arguments(ARG "MIXED" "" "LINK" ${ARGN})
    set(sources "${subfolder}/test_${base_name}.cpp")
    if (ARG_MIXED)
        list(APPEND sources "${subfolder}/test_${base_name}_kernels.cu")
    endif ()
    if (NOT ARG_LINK)
        set(ARG_LINK smallfield_lib)
    endif ()
    define_smallfield_target(
        test_${base_name}
        SOURCES ${sources}
        LINK ${ARG_LINK}
    )
    add_test(NAME test_${base_name} COMMAND test_${base_name})
endfunction ()

define_smallfield_test(cuda triangle_smooth MIXED)
define_smallfield_test(cuda dg_energy MIXED)
define_smallfield_test(cuda mmff_gradient MIXED)
define_smallfield_test(cuda bond_distance MIXED)
define_smallfield_test(cuda csd_torsion_device MIXED)

# The reproducibility check compiles `metric_embedding_prepass.cuh` itself so it can drive the embed kernel over
# thousands of identical blocks.  It therefore must not also link `smallfield_lib`, which carries its own copy of that
# kernel: the launch stub is a non-inline symbol defined in a header, so the linker would see it twice.  RDKit comes in
# only for the assertion macros.
define_smallfield_test(
    cuda embedding_determinism
    MIXED
    LINK ${SMALLFIELD_RDKIT_TARGETS} fmt::fmt
)

define_smallfield_test(preprocess chirality MIXED)
define_smallfield_test(preprocess descriptor)
define_smallfield_test(preprocess csd_torsion_predicates)
define_smallfield_test(preprocess element_gate)
define_smallfield_test(preprocess chiral_parity)

define_smallfield_test(postprocess shape_descriptors)

define_smallfield_test(parameters mmff94_tables MIXED)

define_smallfield_test(mmff94 mmff_gpu_typer MIXED)
define_smallfield_test(mmff94 mmff_gpu_extract MIXED)
define_smallfield_test(mmff94 mmff_cpu_port)
define_smallfield_test(mmff94 mmff_typer)

# Per-term parity reuses the single-molecule energy entry point that the gradient test's kernel wrapper already exposes,
# so it takes that sidecar rather than one of its own.
define_smallfield_target(
    test_mmff_term_parity
    SOURCES mmff94/test_mmff_term_parity.cpp cuda/test_mmff_gradient_kernels.cu
    LINK smallfield_lib
)
add_test(NAME test_mmff_term_parity COMMAND test_mmff_term_parity)

define_smallfield_test(api conformer_slot_layout)
define_smallfield_test(api pipeline)
define_smallfield_test(api seed_determinism)
define_smallfield_test(api ez_stereo)
