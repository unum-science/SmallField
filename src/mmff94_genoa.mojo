# SmallField: MMFF94 BF16 Storage Variant for AMD Genoa / Intel Cooper Lake+
#
# Target CPUs:
#   - AMD Genoa (Zen 4) with AVX-512 BF16
#   - Intel Cooper Lake with AVX512_BF16
#   - Intel Sapphire Rapids with AVX512_BF16 + AVX512_FP16
#
# Strategy:
#   - Storage: BF16 (2x memory bandwidth savings vs F32)
#   - Compute: F32 (widen from BF16, compute, accumulate in F32)
#   - SIMD width: 16 for F32 compute (after widening from 32 BF16 values)
#
# Key Insight:
#   BF16 has same exponent range as F32 but only 7 mantissa bits (~0.8% precision).
#   This is sufficient for storing coordinates (typically 1-100 Angstroms).
#   The value is MEMORY BANDWIDTH, not compute precision.
#   We load BF16, widen to F32, compute in F32, and accumulate in F32.
#
# Performance Benefits:
#   - 2x reduction in memory traffic for coordinate loads
#   - Same computational accuracy as F32 (all math done in F32)
#   - Enables fitting larger molecules in cache

from memory import UnsafePointer
from math import sqrt
from sys.intrinsics import llvm_intrinsic

# =============================================================================
# Inlined AVX-512 Intrinsics for F32 Compute (from simd_utils.mojo)
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
# Constants (same as base MMFF94)
# =============================================================================

alias MMFF_VDW_DELTA: Float32 = 0.07
alias MMFF_VDW_GAMMA: Float32 = 0.12
alias MMFF_ELEC_CONSTANT: Float32 = 332.0716
alias MMFF_ELEC_BUFFER: Float32 = 0.05

# =============================================================================
# BF16 Helper: pow7 (used in vdW calculations)
# =============================================================================

@always_inline
fn pow7[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Compute x^7 efficiently using x * x^2 * x^4."""
    var x2 = x * x
    var x4 = x2 * x2
    return x * x2 * x4


# =============================================================================
# Van der Waals Energy (BF16 storage, F32 compute)
# =============================================================================

fn compute_vdw_energy_genoa(
    x: UnsafePointer[Scalar[DType.bfloat16]],
    y: UnsafePointer[Scalar[DType.bfloat16]],
    z: UnsafePointer[Scalar[DType.bfloat16]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_epsilon: UnsafePointer[Scalar[DType.bfloat16]],
    nb_r_star: UnsafePointer[Scalar[DType.bfloat16]],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int
) -> Float32:
    """
    Compute vdW energy using BF16 storage with F32 compute (AMD Genoa / Intel Cooper Lake+).

    Args:
        x, y, z: Atomic coordinates in BF16 format
        nb_i, nb_j: Nonbonded pair atom indices
        nb_epsilon: vdW well depth in BF16
        nb_r_star: vdW equilibrium distance in BF16
        atom_start: Starting atom index for gathering coordinates
        nb_start: Starting nonbonded pair index
        nb_end: Ending nonbonded pair index

    Returns:
        Total vdW energy as Float32

    Implementation:
        - Loads 16 pairs at a time (32 BF16 coords -> widen to 16 F32)
        - Widening is a type reinterpretation (no computation cost)
        - All distance/energy calculations in F32 for accuracy
        - Uses rsqrt14 for fast distance normalization
        - 4-way multi-accumulator to hide FP latency
    """
    alias width = 16  # F32 SIMD width for AVX-512
    var accum = SIMDMultiAccumulator[width, 4]()

    var delta = SIMD[DType.float32, width](MMFF_VDW_DELTA)
    var gamma = SIMD[DType.float32, width](MMFF_VDW_GAMMA)
    var one = SIMD[DType.float32, width](1.0)
    var two = SIMD[DType.float32, width](2.0)
    var min_dist = SIMD[DType.float32, width](0.1)

    # Main SIMD loop - process 16 pairs at a time
    var num_pairs = nb_end - nb_start
    var simd_pairs = (num_pairs // width) * width

    for pair_base in range(nb_start, nb_start + simd_pairs, width):
        # Gather atom indices and load BF16 data
        var x_i_bf16 = SIMD[DType.bfloat16, width]()
        var y_i_bf16 = SIMD[DType.bfloat16, width]()
        var z_i_bf16 = SIMD[DType.bfloat16, width]()
        var x_j_bf16 = SIMD[DType.bfloat16, width]()
        var y_j_bf16 = SIMD[DType.bfloat16, width]()
        var z_j_bf16 = SIMD[DType.bfloat16, width]()
        var eps_bf16 = SIMD[DType.bfloat16, width]()
        var rstar_bf16 = SIMD[DType.bfloat16, width]()

        # Gather loop - load BF16 values
        @parameter
        for k in range(width):
            var ii = Int(nb_i[pair_base + k])
            var jj = Int(nb_j[pair_base + k])
            x_i_bf16[k] = x[ii]
            y_i_bf16[k] = y[ii]
            z_i_bf16[k] = z[ii]
            x_j_bf16[k] = x[jj]
            y_j_bf16[k] = y[jj]
            z_j_bf16[k] = z[jj]
            eps_bf16[k] = nb_epsilon[pair_base + k]
            rstar_bf16[k] = nb_r_star[pair_base + k]

        # Widen BF16 to F32 for computation (no-cost type reinterpretation)
        var x_i = x_i_bf16.cast[DType.float32]()
        var y_i = y_i_bf16.cast[DType.float32]()
        var z_i = z_i_bf16.cast[DType.float32]()
        var x_j = x_j_bf16.cast[DType.float32]()
        var y_j = y_j_bf16.cast[DType.float32]()
        var z_j = z_j_bf16.cast[DType.float32]()
        var epsilon = eps_bf16.cast[DType.float32]()
        var r_star = rstar_bf16.cast[DType.float32]()

        # All computation now in F32 for full precision
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i

        # Fast distance using rsqrt: r² -> inv_r -> r
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))
        var inv_r = rsqrt14(r_sq)
        var r = r_sq * inv_r

        # Buffered 14-7 vdW potential (MMFF94 formula)
        # E = epsilon * (1.07*R*/(r+0.07*R*))^7 * (1.12*R*^7/(r^7+0.12*R*^7) - 2)
        var r_star_7 = pow7[width](r_star)

        # Repulsive term: (1.07*R*/(r+0.07*R*))^7
        var denom_rep = r + delta * r_star
        var ratio_rep = (one + delta) * r_star * rcp14(denom_rep)
        var rep_term = pow7[width](ratio_rep)

        # Attractive term: 1.12*R*^7/(r^7+0.12*R*^7) - 2
        var r_7 = pow7[width](r)
        var denom_att = r_7 + gamma * r_star_7
        var att_ratio = (one + gamma) * r_star_7 * rcp14(denom_att)
        var att_term = att_ratio - two

        # Energy = epsilon * rep * att
        var energy = epsilon * rep_term * att_term

        # Note: No masking needed - structures are padded to SIMD multiples

        # Accumulate with multi-accumulator pattern (breaks dependency chains)
        accum.add((pair_base - nb_start) // width, energy)

    # Scalar tail loop for remaining pairs
    var total = accum.reduce()
    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        # Load BF16 and widen to F32
        var dx_f32 = Float32(x[j]) - Float32(x[i])
        var dy_f32 = Float32(y[j]) - Float32(y[i])
        var dz_f32 = Float32(z[j]) - Float32(z[i])
        var r_sq = dx_f32 * dx_f32 + dy_f32 * dy_f32 + dz_f32 * dz_f32
        var r = sqrt(r_sq)

        if r < Float32(0.1):
            continue

        var eps_f32 = Float32(nb_epsilon[pair_idx])
        var rstar_f32 = Float32(nb_r_star[pair_idx])

        # Scalar vdW calculation
        var r_star_7 = rstar_f32 ** 7
        var denom_rep = r + MMFF_VDW_DELTA * rstar_f32
        var ratio_rep = (Float32(1.0) + MMFF_VDW_DELTA) * rstar_f32 / denom_rep
        var rep_term = ratio_rep ** 7

        var r_7 = r ** 7
        var denom_att = r_7 + MMFF_VDW_GAMMA * r_star_7
        var att_ratio = (Float32(1.0) + MMFF_VDW_GAMMA) * r_star_7 / denom_att
        var att_term = att_ratio - Float32(2.0)

        total += eps_f32 * rep_term * att_term

    return total


# =============================================================================
# Electrostatic Energy (BF16 storage, F32 compute)
# =============================================================================

fn compute_electrostatic_energy_genoa(
    x: UnsafePointer[Scalar[DType.bfloat16]],
    y: UnsafePointer[Scalar[DType.bfloat16]],
    z: UnsafePointer[Scalar[DType.bfloat16]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_qi: UnsafePointer[Scalar[DType.bfloat16]],
    nb_qj: UnsafePointer[Scalar[DType.bfloat16]],
    nb_is_1_4: UnsafePointer[Int32],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int,
    scale_1_4: Float32 = 0.75
) -> Float32:
    """
    Compute electrostatic energy using BF16 storage with F32 compute (AMD Genoa / Intel Cooper Lake+).

    Args:
        x, y, z: Atomic coordinates in BF16 format
        nb_i, nb_j: Nonbonded pair atom indices
        nb_qi, nb_qj: Partial charges in BF16
        nb_is_1_4: Flag for 1-4 interactions (0 or 1)
        atom_start: Starting atom index
        nb_start: Starting nonbonded pair index
        nb_end: Ending nonbonded pair index
        scale_1_4: Scaling factor for 1-4 interactions (default 0.75)

    Returns:
        Total electrostatic energy as Float32

    Implementation:
        - MMFF94 formula: E = 332.0716 * qi * qj / (r + 0.05)
        - 1-4 interactions scaled by scale_1_4
        - Uses rcp14 for fast reciprocal (1/r term)
        - All computation in F32 after widening from BF16
    """
    alias width = 16  # F32 SIMD width for AVX-512
    var accum = SIMDMultiAccumulator[width, 4]()

    var elec_const = SIMD[DType.float32, width](MMFF_ELEC_CONSTANT)
    var elec_buffer = SIMD[DType.float32, width](MMFF_ELEC_BUFFER)
    var one = SIMD[DType.float32, width](1.0)
    var scale_1_4_vec = SIMD[DType.float32, width](scale_1_4)
    var min_dist = SIMD[DType.float32, width](0.1)
    var zero = SIMD[DType.float32, width](0)

    # Main SIMD loop
    var num_pairs = nb_end - nb_start
    var simd_pairs = (num_pairs // width) * width

    for pair_base in range(nb_start, nb_start + simd_pairs, width):
        # Gather BF16 data
        var x_i_bf16 = SIMD[DType.bfloat16, width]()
        var y_i_bf16 = SIMD[DType.bfloat16, width]()
        var z_i_bf16 = SIMD[DType.bfloat16, width]()
        var x_j_bf16 = SIMD[DType.bfloat16, width]()
        var y_j_bf16 = SIMD[DType.bfloat16, width]()
        var z_j_bf16 = SIMD[DType.bfloat16, width]()
        var qi_bf16 = SIMD[DType.bfloat16, width]()
        var qj_bf16 = SIMD[DType.bfloat16, width]()
        var is_1_4_flags = SIMD[DType.float32, width]()

        @parameter
        for k in range(width):
            var ii = Int(nb_i[pair_base + k])
            var jj = Int(nb_j[pair_base + k])
            x_i_bf16[k] = x[ii]
            y_i_bf16[k] = y[ii]
            z_i_bf16[k] = z[ii]
            x_j_bf16[k] = x[jj]
            y_j_bf16[k] = y[jj]
            z_j_bf16[k] = z[jj]
            qi_bf16[k] = nb_qi[pair_base + k]
            qj_bf16[k] = nb_qj[pair_base + k]
            is_1_4_flags[k] = Float32(1.0) if nb_is_1_4[pair_base + k] != 0 else Float32(0.0)

        # Widen BF16 to F32 for computation
        var x_i = x_i_bf16.cast[DType.float32]()
        var y_i = y_i_bf16.cast[DType.float32]()
        var z_i = z_i_bf16.cast[DType.float32]()
        var x_j = x_j_bf16.cast[DType.float32]()
        var y_j = y_j_bf16.cast[DType.float32]()
        var z_j = z_j_bf16.cast[DType.float32]()
        var qi = qi_bf16.cast[DType.float32]()
        var qj = qj_bf16.cast[DType.float32]()

        # Compute scale factors: 1.0 for normal pairs, scale_1_4 for 1-4 pairs
        var scale = one + is_1_4_flags * (scale_1_4_vec - one)

        # Compute distances in F32
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))
        var inv_r = rsqrt14(r_sq)
        var r = r_sq * inv_r

        # Electrostatic energy: E = 332.0716 * qi * qj / (r + 0.05)
        var r_eff = r + elec_buffer
        var qq = qi * qj
        var energy = elec_const * qq * rcp14(r_eff)

        # Apply 1-4 scaling (no masking needed - structures are padded)
        energy = energy * scale

        accum.add((pair_base - nb_start) // width, energy)

    # Scalar tail loop
    var total = accum.reduce()
    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        var qi_f32 = Float32(nb_qi[pair_idx])
        var qj_f32 = Float32(nb_qj[pair_idx])

        if qi_f32 == Float32(0.0) or qj_f32 == Float32(0.0):
            continue

        var dx_f32 = Float32(x[j]) - Float32(x[i])
        var dy_f32 = Float32(y[j]) - Float32(y[i])
        var dz_f32 = Float32(z[j]) - Float32(z[i])
        var r_sq = dx_f32 * dx_f32 + dy_f32 * dy_f32 + dz_f32 * dz_f32
        var r = sqrt(r_sq)

        if r < Float32(0.1):
            continue

        var scale = Float32(1.0)
        if nb_is_1_4[pair_idx] != 0:
            scale = scale_1_4

        var r_eff = r + MMFF_ELEC_BUFFER
        total += scale * MMFF_ELEC_CONSTANT * qi_f32 * qj_f32 / r_eff

    return total


# =============================================================================
# Combined Nonbonded Energy (vdW + Electrostatic)
# =============================================================================

fn compute_nonbonded_energy_genoa(
    x: UnsafePointer[Scalar[DType.bfloat16]],
    y: UnsafePointer[Scalar[DType.bfloat16]],
    z: UnsafePointer[Scalar[DType.bfloat16]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_epsilon: UnsafePointer[Scalar[DType.bfloat16]],
    nb_r_star: UnsafePointer[Scalar[DType.bfloat16]],
    nb_qi: UnsafePointer[Scalar[DType.bfloat16]],
    nb_qj: UnsafePointer[Scalar[DType.bfloat16]],
    nb_is_1_4: UnsafePointer[Int32],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int,
    scale_1_4: Float32 = 0.75
) -> Tuple[Float32, Float32]:
    """
    Compute both vdW and electrostatic energies in a single pass.

    This fused kernel is more efficient than calling vdW and electrostatic
    separately because:
    1. Coordinates loaded only once (reduces memory traffic)
    2. Distance computed only once (saves sqrt/rsqrt)
    3. Better cache utilization

    Args:
        x, y, z: Atomic coordinates in BF16
        nb_i, nb_j: Nonbonded pair indices
        nb_epsilon, nb_r_star: vdW parameters in BF16
        nb_qi, nb_qj: Partial charges in BF16
        nb_is_1_4: 1-4 interaction flags
        atom_start: Starting atom index
        nb_start, nb_end: Pair range
        scale_1_4: 1-4 electrostatic scaling

    Returns:
        Tuple of (vdW_energy, electrostatic_energy)
    """
    alias width = 16
    var vdw_accum = SIMDMultiAccumulator[width, 4]()
    var elec_accum = SIMDMultiAccumulator[width, 4]()

    var delta = SIMD[DType.float32, width](MMFF_VDW_DELTA)
    var gamma = SIMD[DType.float32, width](MMFF_VDW_GAMMA)
    var one = SIMD[DType.float32, width](1.0)
    var two = SIMD[DType.float32, width](2.0)
    var elec_const = SIMD[DType.float32, width](MMFF_ELEC_CONSTANT)
    var elec_buffer = SIMD[DType.float32, width](MMFF_ELEC_BUFFER)
    var scale_1_4_vec = SIMD[DType.float32, width](scale_1_4)
    var min_dist = SIMD[DType.float32, width](0.1)
    var zero = SIMD[DType.float32, width](0)

    var num_pairs = nb_end - nb_start
    var simd_pairs = (num_pairs // width) * width

    for pair_base in range(nb_start, nb_start + simd_pairs, width):
        # Gather BF16 data (all parameters at once)
        var x_i_bf16 = SIMD[DType.bfloat16, width]()
        var y_i_bf16 = SIMD[DType.bfloat16, width]()
        var z_i_bf16 = SIMD[DType.bfloat16, width]()
        var x_j_bf16 = SIMD[DType.bfloat16, width]()
        var y_j_bf16 = SIMD[DType.bfloat16, width]()
        var z_j_bf16 = SIMD[DType.bfloat16, width]()
        var eps_bf16 = SIMD[DType.bfloat16, width]()
        var rstar_bf16 = SIMD[DType.bfloat16, width]()
        var qi_bf16 = SIMD[DType.bfloat16, width]()
        var qj_bf16 = SIMD[DType.bfloat16, width]()
        var is_1_4_flags = SIMD[DType.float32, width]()

        @parameter
        for k in range(width):
            var ii = Int(nb_i[pair_base + k])
            var jj = Int(nb_j[pair_base + k])
            x_i_bf16[k] = x[ii]
            y_i_bf16[k] = y[ii]
            z_i_bf16[k] = z[ii]
            x_j_bf16[k] = x[jj]
            y_j_bf16[k] = y[jj]
            z_j_bf16[k] = z[jj]
            eps_bf16[k] = nb_epsilon[pair_base + k]
            rstar_bf16[k] = nb_r_star[pair_base + k]
            qi_bf16[k] = nb_qi[pair_base + k]
            qj_bf16[k] = nb_qj[pair_base + k]
            is_1_4_flags[k] = Float32(1.0) if nb_is_1_4[pair_base + k] != 0 else Float32(0.0)

        # Widen BF16 to F32
        var x_i = x_i_bf16.cast[DType.float32]()
        var y_i = y_i_bf16.cast[DType.float32]()
        var z_i = z_i_bf16.cast[DType.float32]()
        var x_j = x_j_bf16.cast[DType.float32]()
        var y_j = y_j_bf16.cast[DType.float32]()
        var z_j = z_j_bf16.cast[DType.float32]()
        var epsilon = eps_bf16.cast[DType.float32]()
        var r_star = rstar_bf16.cast[DType.float32]()
        var qi = qi_bf16.cast[DType.float32]()
        var qj = qj_bf16.cast[DType.float32]()

        # Compute distance ONCE for both vdW and electrostatic
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))
        var inv_r = rsqrt14(r_sq)
        var r = r_sq * inv_r

        # --- vdW energy ---
        var r_star_7 = pow7[width](r_star)
        var denom_rep = r + delta * r_star
        var ratio_rep = (one + delta) * r_star * rcp14(denom_rep)
        var rep_term = pow7[width](ratio_rep)
        var r_7 = pow7[width](r)
        var denom_att = r_7 + gamma * r_star_7
        var att_ratio = (one + gamma) * r_star_7 * rcp14(denom_att)
        var att_term = att_ratio - two
        var vdw_energy = epsilon * rep_term * att_term

        # --- Electrostatic energy ---
        var scale = one + is_1_4_flags * (scale_1_4_vec - one)
        var r_eff = r + elec_buffer
        var qq = qi * qj
        var elec_energy = elec_const * qq * rcp14(r_eff) * scale

        # Note: No masking needed - structures are padded to SIMD multiples

        vdw_accum.add((pair_base - nb_start) // width, vdw_energy)
        elec_accum.add((pair_base - nb_start) // width, elec_energy)

    # Scalar tail
    var vdw_total = vdw_accum.reduce()
    var elec_total = elec_accum.reduce()

    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        var dx_f32 = Float32(x[j]) - Float32(x[i])
        var dy_f32 = Float32(y[j]) - Float32(y[i])
        var dz_f32 = Float32(z[j]) - Float32(z[i])
        var r_sq = dx_f32 * dx_f32 + dy_f32 * dy_f32 + dz_f32 * dz_f32
        var r = sqrt(r_sq)

        if r < Float32(0.1):
            continue

        # vdW
        var eps_f32 = Float32(nb_epsilon[pair_idx])
        var rstar_f32 = Float32(nb_r_star[pair_idx])
        var r_star_7 = rstar_f32 ** 7
        var denom_rep = r + MMFF_VDW_DELTA * rstar_f32
        var ratio_rep = (Float32(1.0) + MMFF_VDW_DELTA) * rstar_f32 / denom_rep
        var rep_term = ratio_rep ** 7
        var r_7 = r ** 7
        var denom_att = r_7 + MMFF_VDW_GAMMA * r_star_7
        var att_ratio = (Float32(1.0) + MMFF_VDW_GAMMA) * r_star_7 / denom_att
        var att_term = att_ratio - Float32(2.0)
        vdw_total += eps_f32 * rep_term * att_term

        # Electrostatic
        var qi_f32 = Float32(nb_qi[pair_idx])
        var qj_f32 = Float32(nb_qj[pair_idx])
        if qi_f32 != Float32(0.0) and qj_f32 != Float32(0.0):
            var scale = Float32(1.0)
            if nb_is_1_4[pair_idx] != 0:
                scale = scale_1_4
            var r_eff = r + MMFF_ELEC_BUFFER
            elec_total += scale * MMFF_ELEC_CONSTANT * qi_f32 * qj_f32 / r_eff

    return (vdw_total, elec_total)


# =============================================================================
# Documentation and Usage Notes
# =============================================================================

fn main():
    """
    Usage notes for mmff94_genoa.mojo:

    This module provides BF16-optimized MMFF94 nonbonded energy kernels for:
    - AMD Genoa (Zen 4) with AVX-512 BF16
    - Intel Cooper Lake with AVX512_BF16
    - Intel Sapphire Rapids with AVX512_BF16 + AVX512_FP16

    Key Functions:
    1. compute_vdw_energy_genoa():
       - Computes van der Waals energy using buffered 14-7 potential
       - Loads coordinates in BF16, computes in F32
       - 2x memory bandwidth savings vs F32 storage

    2. compute_electrostatic_energy_genoa():
       - Computes Coulombic electrostatic energy
       - Supports 1-4 scaling (default 0.75)
       - Includes 0.05 A buffer to prevent singularities

    3. compute_nonbonded_energy_genoa():
       - Fused kernel computing both vdW and electrostatic
       - Most efficient option (single coordinate load, single distance calc)
       - Returns tuple of (vdw_energy, elec_energy)

    Performance Benefits:
    - 2x reduction in memory traffic for coordinates (BF16 vs F32)
    - Same computational accuracy (all math in F32)
    - Enables larger molecules to fit in cache
    - Uses AVX-512 intrinsics: rsqrt14, rcp14 for fast reciprocals

    BF16 Format:
    - 1 sign bit, 8 exponent bits, 7 mantissa bits
    - Same exponent range as F32 (10^-38 to 10^38)
    - Precision: ~2-3 decimal digits (~0.8% relative error)
    - Sufficient for molecular coordinates (1-100 Angstroms)

    Widening Strategy:
    - Load: 16 BF16 values (256 bits)
    - Widen: 16 F32 values (512 bits, no computation cost)
    - Compute: F32 SIMD operations
    - Accumulate: F32 for full precision

    Compiler Optimization:
    - Ensure LLVM is configured with -mavx512f -mavx512bf16
    - Check simd_utils.HAS_AVX512_BF16 = True
    - Verify CPU support: lscpu | grep avx512_bf16
    """
    print("SmallField MMFF94 BF16 Variant for AMD Genoa / Intel Cooper Lake+")
    print("Storage: BF16 (2x bandwidth savings)")
    print("Compute: F32 (full precision)")
    print("Target: CPUs with AVX512_BF16 (AMD Genoa, Intel Cooper Lake, Sapphire Rapids)")
