# C/C++/CUDA test executables, each a standalone binary with its own main(), organised into subfolders mirroring src/:
# api, cuda, parameters, preprocess, mmff94, postprocess.
#
# A mixed test pairs `<subfolder>/test_<name>.cpp` with `<subfolder>/test_<name>_kernels.cu`.  The .cpp calls a plain
# host wrapper the .cu defines, which is ordinary host linkage -- no relocatable device code is involved, which is why
# `define_smallfield_target` leaves it off.
#
# EXTRA_SOURCES carries a source a test owns but does not want in its own translation unit: a table of plain data
# belongs in a host-only .cpp for the same reason `csd_torsion_table.cpp` is one, and a sidecar another test already
# defines is named here rather than copied.
#
# GPU marks a test that drives the device, and sets the `RESOURCE_LOCK` CTest holds while it runs, so `ctest -j` never
# starts two of them at once.  Several of these size a device allocation against the whole card, so run together they
# fail on allocation rather than on anything they assert -- red for contention costs whoever reproduces it more than
# running them one at a time ever costs the suite.  A MIXED test compiles its own kernels to launch them, so it locks
# without saying so; GPU is for a test whose device work arrives through `smallfield_lib`.  Everything unmarked is
# host-only and still runs in parallel with the locked ones, which is what the flag buys over `ctest -j 1`.
#
# `RESOURCE_LOCK` rather than `RESOURCE_GROUPS`: the latter needs a resource spec file named on the command line, so a
# plain `ctest -j 8` would silently get no serialization at all.
function (define_smallfield_test subfolder base_name)
    cmake_parse_arguments(ARG "MIXED;GPU" "" "LINK;EXTRA_SOURCES" ${ARGN})
    set(sources "${subfolder}/test_${base_name}.cpp" ${ARG_EXTRA_SOURCES})
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
    # The lock and the label are one fact: a test that must not run beside another device test is exactly a test a
    # GPU-less runner cannot run at all.  Written together so `ctest --label-exclude gpu` and the serialization can
    # never disagree about which half a binary is in.
    if (ARG_MIXED OR ARG_GPU)
        set_tests_properties(test_${base_name} PROPERTIES RESOURCE_LOCK smallfield_gpu LABELS gpu)
    endif ()
endfunction ()

define_smallfield_test(cuda triangle_smooth MIXED)
define_smallfield_test(cuda dg_energy MIXED)
define_smallfield_test(cuda mmff_gradient MIXED)
define_smallfield_test(cuda bond_distance MIXED)
define_smallfield_test(cuda csd_torsion_device MIXED)

# The reproducibility check compiles `metric_embedding_prepass.cuh` itself so it can drive the embed kernel over
# thousands of identical blocks.  It therefore must not also link `smallfield_lib`, which carries its own copy of that
# kernel: the launch stub is a non-inline symbol defined in a header, so the linker would see it twice.  RDKit comes in
# only for the assertion macros, and `CUDA::cudart` only for its include directories, which carry the libcu++ headers
# behind `smallfield/types/views.hpp`.
define_smallfield_test(
    cuda embedding_determinism
    MIXED
    LINK smallfield_rdkit fmt::fmt CUDA::cudart
)

# The infeasibility check compiles the metric prepass and the pipeline kernel itself, so it is bound by the same rule as
# the reproducibility check above and must not also link `smallfield_lib`.  RDKit comes in for the headers the flat
# molecule descriptor is declared against, and `CUDA::curand` for the Philox header the pipeline kernel seeds its jitter
# from.
define_smallfield_test(
    cuda bounds_infeasible
    MIXED
    LINK smallfield_rdkit fmt::fmt CUDA::cudart CUDA::curand
)

define_smallfield_test(preprocess chirality MIXED)
define_smallfield_test(preprocess descriptor)
define_smallfield_test(preprocess csd_torsion_predicates)
define_smallfield_test(preprocess element_gate)
define_smallfield_test(preprocess chiral_parity)
define_smallfield_test(preprocess batch_layout)
define_smallfield_test(preprocess rdkit_divergences EXTRA_SOURCES preprocess/rdkit_divergence_table.cpp)

define_smallfield_test(postprocess shape_descriptors)

define_smallfield_test(parameters mmff94_tables MIXED)

define_smallfield_test(mmff94 mmff_gpu_typer MIXED)
define_smallfield_test(mmff94 mmff_gpu_extract MIXED)
define_smallfield_test(mmff94 mmff_cpu_port)
define_smallfield_test(mmff94 mmff_typer)

# Per-term parity reuses the single-molecule energy entry point that the gradient test's kernel wrapper already exposes,
# so it takes that sidecar rather than one of its own.  Registered through the function like everything else, or it is
# the one test that silently misses whatever the function grows next.
define_smallfield_test(mmff94 mmff_term_parity GPU EXTRA_SOURCES cuda/test_mmff_gradient_kernels.cu)

# The fallback policy is a `constexpr` switch and a span walk, so the test asserts it without ever reaching a device.
# `smallfield_lib` would only add the `rdkit_fallback` definition it never calls; `smallfield_rdkit` and `CUDA::cudart`
# are here for their include directories alone, carrying the RDKit headers `molecule_descriptor_t` is declared against
# and the libcu++ headers behind `smallfield/types/views.hpp`.
define_smallfield_test(cli fallback_policy LINK smallfield_rdkit fmt::fmt CUDA::cudart)

# The zero-row shard goes through the shipped writer rather than a copy of it, so it takes the object library both CLI
# binaries link -- which exists only when the CLI is built.  Arrow is found here as well as in `cli/` because
# `find_package` imports its targets into the directory that calls it, so a name arriving from another subdirectory
# would not resolve.  Host-only: `write_shard` reaches no device, and the coordinates it is handed are three halves
# written by hand.
#
# Sealing and parking are the two ways an input shard leaves the set a later run is offered, so they are pinned against a
# real temporary tree rather than a fake.  `cli/shard_inventory.hpp` carries both inline, so the test needs no source of
# its own beyond the CLI headers -- and it is registered here, beside the writer test, because the header ships behaviour
# only to the two binaries this option builds.
if (SMALLFIELD_BUILD_CLI)
    find_package(Arrow CONFIG REQUIRED)
    find_package(Parquet CONFIG REQUIRED)
    define_smallfield_test(
        cli empty_shard
        LINK smallfield_parquet_writer Arrow::arrow_shared Parquet::parquet_shared
    )
    define_smallfield_test(cli shard_sealing LINK smallfield_rdkit fmt::fmt CUDA::cudart)
endif ()

define_smallfield_test(api conformer_slot_layout)
define_smallfield_test(api pipeline GPU)
define_smallfield_test(api seed_determinism GPU)
define_smallfield_test(api ez_stereo GPU)
