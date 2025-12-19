# SmallField: Generic SIMD MMFF94 Implementation
# Target: ARM NEON (width=4) and AVX2 (width=8)
# Compute: Full precision (f32/f64 storage, no approximations)
# Purpose: Accuracy baseline and portable SIMD code

from math import sqrt
from memory import UnsafePointer

# MMFF94 Constants
alias MMFF_VDW_DELTA: Float32 = 0.07
alias MMFF_VDW_GAMMA: Float32 = 0.12
alias MMFF_ELEC_CONSTANT: Float32 = 332.0716
alias MMFF_ELEC_BUFFER: Float32 = 0.05


# =============================================================================
# Helper: x^7 computation for vdW potential
# =============================================================================

@always_inline
fn pow7[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Compute x^7 efficiently using x * x^2 * x^4."""
    var x2 = x * x
    var x4 = x2 * x2
    return x * x2 * x4


# =============================================================================
# Multi-Accumulator Pattern (4 independent accumulators)
# =============================================================================

struct MultiAccumulator4(Copyable, Movable):
    """
    4 independent scalar accumulators to break dependency chains.

    On modern CPUs with ~5 cycle FP add latency, using 4 accumulators
    allows the CPU to overlap operations and hide latency.
    """
    var a0: Float32
    var a1: Float32
    var a2: Float32
    var a3: Float32

    fn __init__(out self):
        self.a0 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0

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
    fn add(mut self, idx: Int, value: Float32):
        """Add to accumulator idx % 4 (fast modulo via bitwise AND)."""
        var i = idx & 3  # Equivalent to idx % 4
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
        """Reduce all accumulators to single value."""
        return (self.a0 + self.a1) + (self.a2 + self.a3)


struct SIMDMultiAccumulator4[dtype: DType, width: Int](Copyable, Movable):
    """
    4 independent SIMD accumulators for vectorized loops.

    Each accumulator is a SIMD vector of 'width' elements.
    Final reduce() returns a scalar sum of all elements.
    """
    var a0: SIMD[dtype, width]
    var a1: SIMD[dtype, width]
    var a2: SIMD[dtype, width]
    var a3: SIMD[dtype, width]

    fn __init__(out self):
        self.a0 = SIMD[dtype, width](0)
        self.a1 = SIMD[dtype, width](0)
        self.a2 = SIMD[dtype, width](0)
        self.a3 = SIMD[dtype, width](0)

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
    fn add(mut self, idx: Int, value: SIMD[dtype, width]):
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
        return Float32(sum_vec.reduce_add())


# =============================================================================
# Van der Waals Energy (Buffered 14-7 Potential)
# =============================================================================

fn compute_vdw_energy_simd[
    StorageType: DType,
    width: Int = 4
](
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_epsilon: UnsafePointer[Scalar[StorageType]],
    nb_r_star: UnsafePointer[Scalar[StorageType]],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int
) -> Float32:
    """
    Generic SIMD van der Waals energy computation.

    MMFF94 buffered 14-7 potential:
        E = epsilon * (1.07*R*/(r+0.07*R*))^7 * (1.12*R*^7/(r^7+0.12*R*^7) - 2)

    Implementation:
        - SIMD width: 4 (ARM NEON) or 8 (AVX2)
        - Storage: f32 or f64
        - Compute: Full precision (standard sqrt, no approximations)
        - Multi-accumulator pattern with 4 independent accumulators
        - Process 'width' pairs per iteration

    Args:
        x, y, z: Atom coordinates (storage type)
        nb_i, nb_j: Nonbonded pair atom indices
        nb_epsilon: vdW well depth parameters
        nb_r_star: vdW equilibrium distance parameters
        atom_start: Starting atom index (for per-atom loops)
        nb_start: Starting pair index
        nb_end: Ending pair index

    Returns:
        Total vdW energy for the given range (Float32)
    """
    alias ComputeType = DType.float32

    # Multi-accumulator for scalar computation
    var accum = SIMDMultiAccumulator4[ComputeType, width]()

    # Constants
    var delta = SIMD[ComputeType, width](MMFF_VDW_DELTA)
    var gamma = SIMD[ComputeType, width](MMFF_VDW_GAMMA)
    var one = SIMD[ComputeType, width](1.0)
    var two = SIMD[ComputeType, width](2.0)

    # Main SIMD loop - process 'width' pairs at a time
    var simd_pairs = ((nb_end - nb_start) // width) * width

    for pair_base in range(nb_start, nb_start + simd_pairs, width):
        # Gather positions and parameters for 'width' pairs
        var x_i = SIMD[ComputeType, width]()
        var y_i = SIMD[ComputeType, width]()
        var z_i = SIMD[ComputeType, width]()
        var x_j = SIMD[ComputeType, width]()
        var y_j = SIMD[ComputeType, width]()
        var z_j = SIMD[ComputeType, width]()
        var eps = SIMD[ComputeType, width]()
        var rstar = SIMD[ComputeType, width]()

        @parameter
        for k in range(width):
            var ii = Int(nb_i[pair_base + k])
            var jj = Int(nb_j[pair_base + k])
            x_i[k] = Scalar[ComputeType](x[ii])
            y_i[k] = Scalar[ComputeType](y[ii])
            z_i[k] = Scalar[ComputeType](z[ii])
            x_j[k] = Scalar[ComputeType](x[jj])
            y_j[k] = Scalar[ComputeType](y[jj])
            z_j[k] = Scalar[ComputeType](z[jj])
            eps[k] = Scalar[ComputeType](nb_epsilon[pair_base + k])
            rstar[k] = Scalar[ComputeType](nb_r_star[pair_base + k])

        # Compute displacements
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i

        # Compute distance squared using FMA: dx*dx + dy*dy + dz*dz
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))

        # Compute distance using standard sqrt (full precision)
        var r = sqrt(r_sq)

        # Compute vdW energy components
        var r_star_7 = pow7[ComputeType, width](rstar)

        # Repulsive term: (1.07*R*/(r+0.07*R*))^7
        var denom_rep = r + delta * rstar
        var ratio_rep = (one + delta) * rstar / denom_rep
        var rep_term = pow7[ComputeType, width](ratio_rep)

        # Attractive term: 1.12*R*^7/(r^7+0.12*R*^7) - 2
        var r_7 = pow7[ComputeType, width](r)
        var denom_att = r_7 + gamma * r_star_7
        var att_term = (one + gamma) * r_star_7 / denom_att - two

        # Energy: epsilon * rep_term * att_term
        var energy = eps * rep_term * att_term

        # Note: No masking needed - structures are padded to SIMD multiples

        # Accumulate with multi-accumulator pattern
        accum.add((pair_base - nb_start) // width, energy)

    # Tail loop for remaining pairs
    var scalar_energy: Float32 = 0.0
    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        var dx = Float32(x[j]) - Float32(x[i])
        var dy = Float32(y[j]) - Float32(y[i])
        var dz = Float32(z[j]) - Float32(z[i])
        var r = sqrt(dx * dx + dy * dy + dz * dz)

        if r < Float32(0.1):
            continue

        var epsilon_val = Float32(nb_epsilon[pair_idx])
        var r_star = Float32(nb_r_star[pair_idx])
        var r_star_7 = r_star ** 7

        # Repulsive term
        var denom_rep = r + MMFF_VDW_DELTA * r_star
        var ratio_rep = (Float32(1.0) + MMFF_VDW_DELTA) * r_star / denom_rep
        var rep_term = ratio_rep ** 7

        # Attractive term
        var r_7 = r ** 7
        var denom_att = r_7 + MMFF_VDW_GAMMA * r_star_7
        var att_term = (Float32(1.0) + MMFF_VDW_GAMMA) * r_star_7 / denom_att - Float32(2.0)

        scalar_energy += epsilon_val * rep_term * att_term

    return accum.reduce() + scalar_energy


# =============================================================================
# Electrostatic Energy (Coulomb with Buffer)
# =============================================================================

fn compute_electrostatic_energy_simd[
    StorageType: DType,
    width: Int = 4
](
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_qi: UnsafePointer[Scalar[StorageType]],
    nb_qj: UnsafePointer[Scalar[StorageType]],
    nb_is_1_4: UnsafePointer[Int32],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int,
    dielectric: Float32 = 1.0,
    scale_1_4: Float32 = 0.75
) -> Float32:
    """
    Generic SIMD electrostatic energy computation.

    MMFF94 electrostatic potential:
        E = 332.0716 * qi * qj / (D * (r + 0.05))

    where:
        - D is the dielectric constant (or r for distance-dependent)
        - 0.05 A buffer prevents singularities
        - 1-4 interactions are scaled by scale_1_4 (typically 0.75)

    Implementation:
        - SIMD width: 4 (ARM NEON) or 8 (AVX2)
        - Storage: f32 or f64
        - Compute: Full precision (standard sqrt and division)
        - Multi-accumulator pattern with 4 independent accumulators
        - Process 'width' pairs per iteration

    Args:
        x, y, z: Atom coordinates (storage type)
        nb_i, nb_j: Nonbonded pair atom indices
        nb_qi, nb_qj: Partial charges for each pair
        nb_is_1_4: 1-4 interaction flags
        atom_start: Starting atom index (for per-atom loops)
        nb_start: Starting pair index
        nb_end: Ending pair index
        dielectric: Dielectric constant (default 1.0)
        scale_1_4: 1-4 scaling factor (default 0.75)

    Returns:
        Total electrostatic energy for the given range (Float32)
    """
    alias ComputeType = DType.float32

    # Multi-accumulator for scalar computation
    var accum = SIMDMultiAccumulator4[ComputeType, width]()

    # Constants
    var elec_const = SIMD[ComputeType, width](MMFF_ELEC_CONSTANT)
    var elec_buffer = SIMD[ComputeType, width](MMFF_ELEC_BUFFER)
    var one = SIMD[ComputeType, width](1.0)
    var scale_1_4_vec = SIMD[ComputeType, width](scale_1_4)
    var dielectric_vec = SIMD[ComputeType, width](dielectric)

    # Main SIMD loop - process 'width' pairs at a time
    var simd_pairs = ((nb_end - nb_start) // width) * width

    for pair_base in range(nb_start, nb_start + simd_pairs, width):
        # Gather positions and charges for 'width' pairs
        var x_i = SIMD[ComputeType, width]()
        var y_i = SIMD[ComputeType, width]()
        var z_i = SIMD[ComputeType, width]()
        var x_j = SIMD[ComputeType, width]()
        var y_j = SIMD[ComputeType, width]()
        var z_j = SIMD[ComputeType, width]()
        var qi = SIMD[ComputeType, width]()
        var qj = SIMD[ComputeType, width]()
        var is_1_4 = SIMD[ComputeType, width]()

        @parameter
        for k in range(width):
            var ii = Int(nb_i[pair_base + k])
            var jj = Int(nb_j[pair_base + k])
            x_i[k] = Scalar[ComputeType](x[ii])
            y_i[k] = Scalar[ComputeType](y[ii])
            z_i[k] = Scalar[ComputeType](z[ii])
            x_j[k] = Scalar[ComputeType](x[jj])
            y_j[k] = Scalar[ComputeType](y[jj])
            z_j[k] = Scalar[ComputeType](z[jj])
            qi[k] = Scalar[ComputeType](nb_qi[pair_base + k])
            qj[k] = Scalar[ComputeType](nb_qj[pair_base + k])
            is_1_4[k] = Scalar[ComputeType](1.0) if nb_is_1_4[pair_base + k] != 0 else Scalar[ComputeType](0.0)

        # Compute scale factors: 1.0 or scale_1_4 based on is_1_4 flag
        var scale = one + is_1_4 * (scale_1_4_vec - one)

        # Compute displacements
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i

        # Compute distance squared using FMA
        var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))

        # Compute distance using standard sqrt (full precision)
        var r = sqrt(r_sq)
        var r_eff = r + elec_buffer

        # Compute energy: E = 332.0716 * qi * qj / (D * r_eff)
        var qq = qi * qj
        var denom = dielectric_vec * r_eff
        var energy = elec_const * qq / denom

        # Apply 1-4 scaling
        energy = energy * scale

        # Note: No masking needed - structures are padded to SIMD multiples

        # Accumulate with multi-accumulator pattern
        accum.add((pair_base - nb_start) // width, energy)

    # Tail loop for remaining pairs
    var scalar_energy: Float32 = 0.0
    for pair_idx in range(nb_start + simd_pairs, nb_end):
        var i = Int(nb_i[pair_idx])
        var j = Int(nb_j[pair_idx])

        var qi_val = Float32(nb_qi[pair_idx])
        var qj_val = Float32(nb_qj[pair_idx])

        # Skip if either charge is zero
        if qi_val == Float32(0.0) or qj_val == Float32(0.0):
            continue

        var dx = Float32(x[j]) - Float32(x[i])
        var dy = Float32(y[j]) - Float32(y[i])
        var dz = Float32(z[j]) - Float32(z[i])
        var r = sqrt(dx * dx + dy * dy + dz * dz)

        if r < Float32(0.1):
            continue

        # Apply 1-4 scaling
        var scale = Float32(1.0)
        if nb_is_1_4[pair_idx] != 0:
            scale = scale_1_4

        var r_eff = r + MMFF_ELEC_BUFFER
        var energy = MMFF_ELEC_CONSTANT * qi_val * qj_val / (dielectric * r_eff)
        scalar_energy += energy * scale

    return accum.reduce() + scalar_energy


# =============================================================================
# Convenience Wrappers for Common Use Cases
# =============================================================================

fn compute_vdw_energy_neon[StorageType: DType](
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_epsilon: UnsafePointer[Scalar[StorageType]],
    nb_r_star: UnsafePointer[Scalar[StorageType]],
    num_pairs: Int
) -> Float32:
    """
    ARM NEON optimized vdW energy (width=4).

    Args:
        x, y, z: Atom coordinates
        nb_i, nb_j: Nonbonded pair atom indices
        nb_epsilon: vdW well depth parameters
        nb_r_star: vdW equilibrium distance parameters
        num_pairs: Total number of pairs

    Returns:
        Total vdW energy
    """
    return compute_vdw_energy_simd[StorageType, 4](
        x, y, z, nb_i, nb_j, nb_epsilon, nb_r_star, 0, 0, num_pairs
    )


fn compute_vdw_energy_avx2[StorageType: DType](
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_epsilon: UnsafePointer[Scalar[StorageType]],
    nb_r_star: UnsafePointer[Scalar[StorageType]],
    num_pairs: Int
) -> Float32:
    """
    AVX2 optimized vdW energy (width=8).

    Args:
        x, y, z: Atom coordinates
        nb_i, nb_j: Nonbonded pair atom indices
        nb_epsilon: vdW well depth parameters
        nb_r_star: vdW equilibrium distance parameters
        num_pairs: Total number of pairs

    Returns:
        Total vdW energy
    """
    return compute_vdw_energy_simd[StorageType, 8](
        x, y, z, nb_i, nb_j, nb_epsilon, nb_r_star, 0, 0, num_pairs
    )


fn compute_electrostatic_energy_neon[StorageType: DType](
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_qi: UnsafePointer[Scalar[StorageType]],
    nb_qj: UnsafePointer[Scalar[StorageType]],
    nb_is_1_4: UnsafePointer[Int32],
    num_pairs: Int,
    dielectric: Float32 = 1.0,
    scale_1_4: Float32 = 0.75
) -> Float32:
    """
    ARM NEON optimized electrostatic energy (width=4).

    Args:
        x, y, z: Atom coordinates
        nb_i, nb_j: Nonbonded pair atom indices
        nb_qi, nb_qj: Partial charges for each pair
        nb_is_1_4: 1-4 interaction flags
        num_pairs: Total number of pairs
        dielectric: Dielectric constant (default 1.0)
        scale_1_4: 1-4 scaling factor (default 0.75)

    Returns:
        Total electrostatic energy
    """
    return compute_electrostatic_energy_simd[StorageType, 4](
        x, y, z, nb_i, nb_j, nb_qi, nb_qj, nb_is_1_4, 0, 0, num_pairs,
        dielectric, scale_1_4
    )


fn compute_electrostatic_energy_avx2[StorageType: DType](
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_qi: UnsafePointer[Scalar[StorageType]],
    nb_qj: UnsafePointer[Scalar[StorageType]],
    nb_is_1_4: UnsafePointer[Int32],
    num_pairs: Int,
    dielectric: Float32 = 1.0,
    scale_1_4: Float32 = 0.75
) -> Float32:
    """
    AVX2 optimized electrostatic energy (width=8).

    Args:
        x, y, z: Atom coordinates
        nb_i, nb_j: Nonbonded pair atom indices
        nb_qi, nb_qj: Partial charges for each pair
        nb_is_1_4: 1-4 interaction flags
        num_pairs: Total number of pairs
        dielectric: Dielectric constant (default 1.0)
        scale_1_4: 1-4 scaling factor (default 0.75)

    Returns:
        Total electrostatic energy
    """
    return compute_electrostatic_energy_simd[StorageType, 8](
        x, y, z, nb_i, nb_j, nb_qi, nb_qj, nb_is_1_4, 0, 0, num_pairs,
        dielectric, scale_1_4
    )


# =============================================================================
# Test/Build Entry Point
# =============================================================================

fn main():
    """Minimal main function for compilation testing."""
    print("mmff94_simd.mojo: Generic SIMD MMFF94 implementation compiled successfully")
    print("Target: ARM NEON (width=4) and AVX2 (width=8)")
    print("Compute: Full precision (standard sqrt, no approximations)")
