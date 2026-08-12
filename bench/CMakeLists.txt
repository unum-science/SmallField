# Benchmark executables linking the SmallField library plus RDKit for cross-validation.

define_smallfield_target(
    bench_conformer_backends
    SOURCES bench_conformer_backends.cpp
    LINK smallfield_lib
)
define_smallfield_target(
    bench_profile
    SOURCES bench_profile.cpp
    LINK smallfield_lib
)

# MMFF kernel scaling benchmark, measuring the GPU kernel trio in isolation.  Mixed build: the .cpp drives RDKit and
# OpenMP CPU timing, the .cu launches the kernels with cudaEvent timing.
define_smallfield_target(
    bench_mmff_kernels
    SOURCES bench_mmff_kernels.cpp bench_mmff_kernels_device.cu
    LINK smallfield_lib
)

# Kernel replay of one captured atom-limited bin, and the one target here that does not go through
# `define_smallfield_target`.  Every NVIDIA profiler dies at the first CUDA call inside a binary carrying the
# conda-forge stack, so this one links neither `smallfield_lib` nor jemalloc -- linking the CUDA runtime alone is the
# whole reason it exists, and the helper links both unconditionally.  It compiles the pipeline kernels itself, which is
# also why it must not link the archive that already carries them.
#
# `smallfield_settings` stays, since it is where `--use_fast_math`, `-Xptxas=-O3` and the include roots come from: a
# replay compiled under different flags measures a kernel the pipeline never launches.  It is built rather than left
# out of tree because it launches both kernels by hand and reinterprets a captured pool through a dozen descriptor
# types, so a signature or a layout change lands here and a hand-run `nvcc` line is the only thing that would say so.
# The manifest keys it binds by name stay a run-time agreement with `src/smallfield/api/wave_capture.hpp`, which is
# what the missing-key refusal reports.
add_executable(bench_kernel_replay bench_kernel_replay.cu)
target_link_libraries(bench_kernel_replay PRIVATE smallfield_settings CUDA::cudart)
set_target_properties(bench_kernel_replay PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
