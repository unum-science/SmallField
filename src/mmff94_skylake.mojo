# SmallField: AVX-512 Optimized MMFF94 Non-bonded Energy Kernels
#
# Target: All AVX-512 CPUs (Skylake-X, Cascade Lake, Ice Lake, Sapphire Rapids)
# SIMD Width: 16 (512-bit registers, f32)
# Storage: f32
# Compute: f32 with fast approximations (VRSQRT14PS, VRCP14PS)
#
# Optimizations:
# - 16-way SIMD parallelism (processes 16 pairs simultaneously)
# - Fast reciprocal sqrt (rsqrt14) for distance computation
# - Fast reciprocal (rcp14) for division operations
# - FMA chains for distance squared computation
# - Multi-accumulator pattern to break dependency chains
# - Arithmetic-based 1-4 scaling (no branches in hot loop)
#
# Performance Characteristics:
# - VRSQRT14PS: 4 cycle latency, 1/cycle throughput, ~14-bit precision
# - VRCP14PS: 4 cycle latency, 1/cycle throughput, ~14-bit precision
# - VFMADD: 4 cycle latency, 0.5/cycle throughput
# - VMULPS: 4 cycle latency, 0.5/cycle throughput
# - VADDPS: 4 cycle latency, 0.5/cycle throughput

from math import rsqrt
from memory import UnsafePointer
from sys.intrinsics import llvm_intrinsic

# MMFF94 constants
alias MMFF_VDW_DELTA: Float32 = 0.07
alias MMFF_VDW_GAMMA: Float32 = 0.12
alias MMFF_ELEC_CONSTANT: Float32 = 332.0716
alias MMFF_ELEC_BUFFER: Float32 = 0.05

# SIMD width for AVX-512
alias WIDTH: Int = 16


# =============================================================================
# AVX-512 Intrinsics (Inlined for WIDTH=16)
# =============================================================================

@always_inline
fn rsqrt14(a: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    """AVX-512 VRSQRT14PS - 14-bit precision reciprocal square root."""
    return llvm_intrinsic[
        "llvm.x86.avx512.rsqrt14.ps.512",
        SIMD[DType.float32, 16],
        SIMD[DType.float32, 16],  # src
        SIMD[DType.float32, 16],  # passthru
        Int16,                     # mask
        has_side_effect=False
    ](a, SIMD[DType.float32, 16](0), Int16(-1))


@always_inline
fn rcp14(a: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    """AVX-512 VRCP14PS - 14-bit precision reciprocal."""
    return llvm_intrinsic[
        "llvm.x86.avx512.rcp14.ps.512",
        SIMD[DType.float32, 16],
        SIMD[DType.float32, 16],  # src
        SIMD[DType.float32, 16],  # passthru
        Int16,                     # mask
        has_side_effect=False
    ](a, SIMD[DType.float32, 16](0), Int16(-1))


# =============================================================================
# Multi-Accumulator Pattern (Reduces Dependency Chains)
# =============================================================================

struct SIMDMultiAccumulator[width: Int, N: Int = 4](Copyable, Movable):
    """
    N independent SIMD accumulators for vectorized loops.

    Each accumulator is a SIMD vector of 'width' f32 elements.
    Final reduce() returns a scalar sum of all elements.
    """
    var a0: SIMD[DType.float32, width]
    var a1: SIMD[DType.float32, width]
    var a2: SIMD[DType.float32, width]
    var a3: SIMD[DType.float32, width]

    fn __init__(out self):
        self.a0 = SIMD[DType.float32, width](0)
        self.a1 = SIMD[DType.float32, width](0)
        self.a2 = SIMD[DType.float32, width](0)
        self.a3 = SIMD[DType.float32, width](0)

    fn __copyinit__(out self, other: Self):
        self.a0 = other.a0
        self.a1 = other.a1
        self.a2 = other.a2
        self.a3 = other.a3

    fn __moveinit__(out self, deinit other: Self):
        self.a0 = other.a0
        self.a1 = other.a1
        self.a2 = other.a2
        self.a3 = other.a3

    @always_inline
    fn add(mut self, idx: Int, value: SIMD[DType.float32, width]):
        """Add vector to accumulator idx % 4."""
        var i = idx & 3
        if i == 0:
            self.a0 += value
        elif i == 1:
            self.a1 += value
        elif i == 2:
            self.a2 += value
        else:
            self.a3 += value

    @always_inline
    fn reduce(self) -> Float32:
        """Reduce all SIMD accumulators to single scalar."""
        var sum_vec = (self.a0 + self.a1) + (self.a2 + self.a3)
        return sum_vec.reduce_add()


# =============================================================================
# Helper: x^7 Computation (for vdW potential)
# =============================================================================

@always_inline
fn pow7_f32(x: SIMD[DType.float32, WIDTH]) -> SIMD[DType.float32, WIDTH]:
    """Compute x^7 efficiently: x^7 = x * x^2 * x^4."""
    var x2 = x * x
    var x4 = x2 * x2
    return x * x2 * x4


# =============================================================================
# Van der Waals Energy (Buffered 14-7 Potential)
# =============================================================================

fn compute_vdw_energy_skylake(
    x: UnsafePointer[Scalar[DType.float32]],
    y: UnsafePointer[Scalar[DType.float32]],
    z: UnsafePointer[Scalar[DType.float32]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_epsilon: UnsafePointer[Scalar[DType.float32]],
    nb_r_star: UnsafePointer[Scalar[DType.float32]],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int
) -> Float32:
    """
    Compute van der Waals energy using AVX-512 SIMD with fast approximations.

    Energy formula (MMFF94 buffered 14-7):
        E = epsilon * (1.07*R*/(r+0.07*R*))^7 * (1.12*R*^7/(r^7+0.12*R*^7) - 2)

    Optimizations:
    - 16-way SIMD: Process 16 atom pairs in parallel
    - Fast distance: inv_r = rsqrt14(r_sq); r = r_sq * inv_r
    - FMA chains: r_sq = dx.fma(dx, dy.fma(dy, dz*dz))
    - Multi-accumulator: 4 independent SIMD accumulators

    Args:
        x, y, z: Atomic coordinates (f32 arrays)
        nb_i, nb_j: Nonbonded pair atom indices
        nb_epsilon: Well depth parameters
        nb_r_star: Equilibrium distance parameters
        atom_start: Starting atom index (unused, for API compatibility)
        nb_start: Starting pair index
        nb_end: Ending pair index (exclusive)

    Returns:
        Total van der Waals energy in kcal/mol

    Note:
        MMFF94 does NOT apply 1-4 scaling to vdW interactions.
        Only electrostatic interactions use 1-4 scaling.
    """
    # Multi-accumulator for independent dependency chains
    var accum = SIMDMultiAccumulator[WIDTH, 4]()

    # Constants as SIMD vectors
    var delta = SIMD[DType.float32, WIDTH](MMFF_VDW_DELTA)
    var gamma = SIMD[DType.float32, WIDTH](MMFF_VDW_GAMMA)
    var one = SIMD[DType.float32, WIDTH](1.0)
    var two = SIMD[DType.float32, WIDTH](2.0)
    var one_plus_delta = SIMD[DType.float32, WIDTH](1.0 + MMFF_VDW_DELTA)
    var one_plus_gamma = SIMD[DType.float32, WIDTH](1.0 + MMFF_VDW_GAMMA)
    var min_dist_sq = SIMD[DType.float32, WIDTH](0.01)  # 0.1^2 = 0.01

    # Process pairs in batches of WIDTH (16)
    var num_pairs = nb_end - nb_start
    var simd_pairs = (num_pairs // WIDTH) * WIDTH

    # Main SIMD loop: Process WIDTH pairs at a time
    for pair_base in range(nb_start, nb_start + simd_pairs, WIDTH):
        # Gather atom positions and parameters for this batch
        var x_i = SIMD[DType.float32, WIDTH]()
        var y_i = SIMD[DType.float32, WIDTH]()
        var z_i = SIMD[DType.float32, WIDTH]()
        var x_j = SIMD[DType.float32, WIDTH]()
        var y_j = SIMD[DType.float32, WIDTH]()
        var z_j = SIMD[DType.float32, WIDTH]()
        var eps = SIMD[DType.float32, WIDTH]()
        var rstar = SIMD[DType.float32, WIDTH]()

        @parameter
        for k in range(WIDTH):
            var idx = pair_base + k
            var ii = Int(nb_i[idx])
            var jj = Int(nb_j[idx])
            x_i[k] = x[ii]
            y_i[k] = y[ii]
            z_i[k] = z[ii]
            x_j[k] = x[jj]
            y_j[k] = y[jj]
            z_j[k] = z[jj]
            eps[k] = nb_epsilon[idx]
            rstar[k] = nb_r_star[idx]

        # Compute displacement vectors
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i

        # Distance squared using FMA chain: r² = dx² + dy² + dz²
        # Optimal order: dx.fma(dx, dy.fma(dy, dz*dz))
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))

        # Fast distance computation: inv_r = rsqrt(r²), r = r² * inv_r
        var inv_r = rsqrt14(r_sq)
        var r = r_sq * inv_r

        # Precompute r_star^7 and r^7 using efficient pow7
        var r_star_7 = pow7_f32(rstar)
        var r_7 = pow7_f32(r)

        # Repulsive term: (1.07*R*/(r+0.07*R*))^7
        var denom_rep = r + delta * rstar
        var ratio_rep = one_plus_delta * rstar * rcp14(denom_rep)
        var rep_term = pow7_f32(ratio_rep)

        # Attractive term: 1.12*R*^7/(r^7+0.12*R*^7) - 2
        var denom_att = r_7 + gamma * r_star_7
        var att_term = one_plus_gamma * r_star_7 * rcp14(denom_att) - two

        # Total energy: E = epsilon * rep_term * att_term
        var energy = eps * rep_term * att_term

        # Structures are padded to SIMD width - no masking needed

        # Accumulate to independent accumulator (breaks dependency chains)
        var accum_idx = (pair_base - nb_start) // WIDTH
        accum.add(accum_idx, energy)

    # Tail loop: Process remaining pairs (if num_pairs not divisible by WIDTH)
    var total_energy = accum.reduce()

    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        var dx = x[j] - x[i]
        var dy = y[j] - y[i]
        var dz = z[j] - z[i]
        var r_sq = dx * dx + dy * dy + dz * dz

        if r_sq < Float32(0.01):
            continue

        # Scalar fallback for tail
        var inv_r = rsqrt(r_sq)
        var r = r_sq * inv_r
        var epsilon = nb_epsilon[pair_idx]
        var r_star = nb_r_star[pair_idx]

        # Scalar pow7
        var r_star_2 = r_star * r_star
        var r_star_4 = r_star_2 * r_star_2
        var r_star_7 = r_star * r_star_2 * r_star_4

        var r_2 = r * r
        var r_4 = r_2 * r_2
        var r_7 = r * r_2 * r_4

        # Repulsive and attractive terms
        var denom_rep = r + MMFF_VDW_DELTA * r_star
        var ratio_rep = (Float32(1.0) + MMFF_VDW_DELTA) * r_star / denom_rep
        var ratio_rep_2 = ratio_rep * ratio_rep
        var ratio_rep_4 = ratio_rep_2 * ratio_rep_2
        var rep_term = ratio_rep * ratio_rep_2 * ratio_rep_4

        var denom_att = r_7 + MMFF_VDW_GAMMA * r_star_7
        var att_term = (Float32(1.0) + MMFF_VDW_GAMMA) * r_star_7 / denom_att - Float32(2.0)

        total_energy += epsilon * rep_term * att_term

    return total_energy


# =============================================================================
# Electrostatic Energy (Coulomb with Buffering)
# =============================================================================

fn compute_electrostatic_energy_skylake(
    x: UnsafePointer[Scalar[DType.float32]],
    y: UnsafePointer[Scalar[DType.float32]],
    z: UnsafePointer[Scalar[DType.float32]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_qi: UnsafePointer[Scalar[DType.float32]],
    nb_qj: UnsafePointer[Scalar[DType.float32]],
    nb_is_1_4: UnsafePointer[Int32],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int,
    scale_1_4: Float32 = 0.75
) -> Float32:
    """
    Compute electrostatic energy using AVX-512 SIMD with fast approximations.

    Energy formula (MMFF94 with buffering):
        E = 332.0716 * qi * qj / (r + 0.05)

    Optimizations:
    - 16-way SIMD: Process 16 atom pairs in parallel
    - Fast distance: inv_r = rsqrt14(r_sq); r = r_sq * inv_r
    - Fast reciprocal: inv_r_eff = rcp14(r + 0.05)
    - Arithmetic 1-4 scaling: scale = 1.0 + is_1_4 * (0.75 - 1.0)
    - Multi-accumulator: 4 independent SIMD accumulators

    Args:
        x, y, z: Atomic coordinates (f32 arrays)
        nb_i, nb_j: Nonbonded pair atom indices
        nb_qi, nb_qj: Partial charges (pre-fetched for pairs)
        nb_is_1_4: 1-4 interaction flags (0 or 1)
        atom_start: Starting atom index (unused, for API compatibility)
        nb_start: Starting pair index
        nb_end: Ending pair index (exclusive)
        scale_1_4: Scaling factor for 1-4 interactions (default 0.75)

    Returns:
        Total electrostatic energy in kcal/mol

    Note:
        This implementation uses pre-fetched charges (nb_qi, nb_qj) for
        better memory access patterns. The caller should prepare these
        arrays: nb_qi[k] = charge[nb_i[k]], nb_qj[k] = charge[nb_j[k]].
    """
    # Multi-accumulator for independent dependency chains
    var accum = SIMDMultiAccumulator[WIDTH, 4]()

    # Constants as SIMD vectors
    var elec_const = SIMD[DType.float32, WIDTH](MMFF_ELEC_CONSTANT)
    var elec_buffer = SIMD[DType.float32, WIDTH](MMFF_ELEC_BUFFER)
    var one = SIMD[DType.float32, WIDTH](1.0)
    var scale_1_4_vec = SIMD[DType.float32, WIDTH](scale_1_4)
    var scale_diff = scale_1_4_vec - one  # Precompute for arithmetic scaling
    var min_dist_sq = SIMD[DType.float32, WIDTH](0.01)  # 0.1^2 = 0.01
    var zero = SIMD[DType.float32, WIDTH](0.0)

    # Process pairs in batches of WIDTH (16)
    var num_pairs = nb_end - nb_start
    var simd_pairs = (num_pairs // WIDTH) * WIDTH

    # Main SIMD loop: Process WIDTH pairs at a time
    for pair_base in range(nb_start, nb_start + simd_pairs, WIDTH):
        # Gather atom positions, charges, and flags for this batch
        var x_i = SIMD[DType.float32, WIDTH]()
        var y_i = SIMD[DType.float32, WIDTH]()
        var z_i = SIMD[DType.float32, WIDTH]()
        var x_j = SIMD[DType.float32, WIDTH]()
        var y_j = SIMD[DType.float32, WIDTH]()
        var z_j = SIMD[DType.float32, WIDTH]()
        var qi = SIMD[DType.float32, WIDTH]()
        var qj = SIMD[DType.float32, WIDTH]()
        var is_1_4 = SIMD[DType.float32, WIDTH]()

        @parameter
        for k in range(WIDTH):
            var idx = pair_base + k
            var ii = Int(nb_i[idx])
            var jj = Int(nb_j[idx])
            x_i[k] = x[ii]
            y_i[k] = y[ii]
            z_i[k] = z[ii]
            x_j[k] = x[jj]
            y_j[k] = y[jj]
            z_j[k] = z[jj]
            qi[k] = nb_qi[idx]
            qj[k] = nb_qj[idx]
            # Convert int to float: 0 -> 0.0, non-zero -> 1.0
            is_1_4[k] = Float32(1.0) if nb_is_1_4[idx] != 0 else Float32(0.0)

        # Compute displacement vectors
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i

        # Distance squared using FMA chain
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))

        # Fast distance computation: inv_r = rsqrt(r²), r = r² * inv_r
        var inv_r = rsqrt14(r_sq)
        var r = r_sq * inv_r

        # Effective distance with buffer: r_eff = r + 0.05
        var r_eff = r + elec_buffer

        # Charge product
        var qq = qi * qj

        # Energy: E = 332.0716 * qi * qj / r_eff
        # Use fast reciprocal: E = 332.0716 * qq * rcp14(r_eff)
        var energy = elec_const * qq * rcp14(r_eff)

        # Arithmetic 1-4 scaling (no branches):
        # scale = 1.0 + is_1_4 * (scale_1_4 - 1.0)
        # If is_1_4 = 0: scale = 1.0 + 0 * (-0.25) = 1.0
        # If is_1_4 = 1: scale = 1.0 + 1 * (-0.25) = 0.75
        var scale = one + is_1_4 * scale_diff
        energy = energy * scale

        # Structures are padded to SIMD width - no masking needed

        # Accumulate to independent accumulator
        var accum_idx = (pair_base - nb_start) // WIDTH
        accum.add(accum_idx, energy)

    # Tail loop: Process remaining pairs
    var total_energy = accum.reduce()

    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        var qi_val = nb_qi[pair_idx]
        var qj_val = nb_qj[pair_idx]

        # Skip if either charge is zero
        if qi_val == Float32(0.0) or qj_val == Float32(0.0):
            continue

        var dx = x[j] - x[i]
        var dy = y[j] - y[i]
        var dz = z[j] - z[i]
        var r_sq = dx * dx + dy * dy + dz * dz

        if r_sq < Float32(0.01):
            continue

        # Scalar fallback for tail
        var inv_r = rsqrt(r_sq)
        var r = r_sq * inv_r
        var r_eff = r + MMFF_ELEC_BUFFER

        var qq = qi_val * qj_val
        var energy = MMFF_ELEC_CONSTANT * qq / r_eff

        # Apply 1-4 scaling
        if nb_is_1_4[pair_idx] != 0:
            energy *= scale_1_4

        total_energy += energy

    return total_energy


# =============================================================================
# Convenience Wrappers (for Systems Without Pre-fetched Charges)
# =============================================================================

fn compute_electrostatic_energy_skylake_simple(
    x: UnsafePointer[Scalar[DType.float32]],
    y: UnsafePointer[Scalar[DType.float32]],
    z: UnsafePointer[Scalar[DType.float32]],
    charge: UnsafePointer[Scalar[DType.float32]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_is_1_4: UnsafePointer[Int32],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int,
    scale_1_4: Float32 = 0.75
) -> Float32:
    """
    Simplified electrostatic energy that fetches charges on-the-fly.

    This is a convenience wrapper that doesn't require pre-fetched charge arrays.
    For optimal performance, use compute_electrostatic_energy_skylake() with
    pre-fetched nb_qi and nb_qj arrays.

    Args:
        x, y, z: Atomic coordinates (f32 arrays)
        charge: Partial charges for all atoms
        nb_i, nb_j: Nonbonded pair atom indices
        nb_is_1_4: 1-4 interaction flags
        atom_start: Starting atom index (unused)
        nb_start: Starting pair index
        nb_end: Ending pair index (exclusive)
        scale_1_4: Scaling factor for 1-4 interactions

    Returns:
        Total electrostatic energy in kcal/mol
    """
    # Multi-accumulator for independent dependency chains
    var accum = SIMDMultiAccumulator[WIDTH, 4]()

    # Constants as SIMD vectors
    var elec_const = SIMD[DType.float32, WIDTH](MMFF_ELEC_CONSTANT)
    var elec_buffer = SIMD[DType.float32, WIDTH](MMFF_ELEC_BUFFER)
    var one = SIMD[DType.float32, WIDTH](1.0)
    var scale_1_4_vec = SIMD[DType.float32, WIDTH](scale_1_4)
    var scale_diff = scale_1_4_vec - one
    var min_dist_sq = SIMD[DType.float32, WIDTH](0.01)
    var zero = SIMD[DType.float32, WIDTH](0.0)

    # Process pairs in batches of WIDTH (16)
    var num_pairs = nb_end - nb_start
    var simd_pairs = (num_pairs // WIDTH) * WIDTH

    # Main SIMD loop
    for pair_base in range(nb_start, nb_start + simd_pairs, WIDTH):
        # Gather data for this batch
        var x_i = SIMD[DType.float32, WIDTH]()
        var y_i = SIMD[DType.float32, WIDTH]()
        var z_i = SIMD[DType.float32, WIDTH]()
        var x_j = SIMD[DType.float32, WIDTH]()
        var y_j = SIMD[DType.float32, WIDTH]()
        var z_j = SIMD[DType.float32, WIDTH]()
        var qi = SIMD[DType.float32, WIDTH]()
        var qj = SIMD[DType.float32, WIDTH]()
        var is_1_4 = SIMD[DType.float32, WIDTH]()

        @parameter
        for k in range(WIDTH):
            var idx = pair_base + k
            var ii = Int(nb_i[idx])
            var jj = Int(nb_j[idx])
            x_i[k] = x[ii]
            y_i[k] = y[ii]
            z_i[k] = z[ii]
            x_j[k] = x[jj]
            y_j[k] = y[jj]
            z_j[k] = z[jj]
            qi[k] = charge[ii]  # Fetch from global charge array
            qj[k] = charge[jj]  # Fetch from global charge array
            is_1_4[k] = Float32(1.0) if nb_is_1_4[idx] != 0 else Float32(0.0)

        # Compute displacement and distance
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))
        var inv_r = rsqrt14(r_sq)
        var r = r_sq * inv_r
        var r_eff = r + elec_buffer

        # Compute energy
        var qq = qi * qj
        var energy = elec_const * qq * rcp14(r_eff)

        # Apply 1-4 scaling
        var scale = one + is_1_4 * scale_diff
        energy = energy * scale

        # Structures are padded to SIMD width - no masking needed

        # Accumulate
        var accum_idx = (pair_base - nb_start) // WIDTH
        accum.add(accum_idx, energy)

    # Tail loop
    var total_energy = accum.reduce()

    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        var qi_val = charge[i]
        var qj_val = charge[j]

        if qi_val == Float32(0.0) or qj_val == Float32(0.0):
            continue

        var dx = x[j] - x[i]
        var dy = y[j] - y[i]
        var dz = z[j] - z[i]
        var r_sq = dx * dx + dy * dy + dz * dz

        if r_sq < Float32(0.01):
            continue

        var inv_r = rsqrt(r_sq)
        var r = r_sq * inv_r
        var r_eff = r + MMFF_ELEC_BUFFER

        var qq = qi_val * qj_val
        var energy = MMFF_ELEC_CONSTANT * qq / r_eff

        if nb_is_1_4[pair_idx] != 0:
            energy *= scale_1_4

        total_energy += energy

    return total_energy
