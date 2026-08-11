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
