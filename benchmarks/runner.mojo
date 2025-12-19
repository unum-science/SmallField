# SmallField Unified Benchmark: Native Mojo Runner
#
# Reads pre-parameterized molecule data from binary file and computes
# MMFF94 energies using native f16, bf16, or f32 storage types.
#
# Usage: mojo run runner.mojo <input.bin> <storage_type>
#   storage_type: f16, bf16, or f32

from sys import argv, num_physical_cores
from time import perf_counter_ns
from math import sqrt, cos, acos, atan2, rsqrt
from memory import UnsafePointer
from algorithm import parallelize

# SIMD configuration (compile-time) - Enabled for Intel Xeon Platinum 8468 (Sapphire Rapids)
alias HAS_AVX512: Bool = True
alias HAS_AVX512_BF16: Bool = True   # Native VDPBF16PS instruction
alias HAS_AVX512_FP16: Bool = True   # Native FP16 instructions
alias SIMD_F32_WIDTH: Int = 16

# =============================================================================
# MMFF94 Constants (must match Python baseline exactly)
# =============================================================================

alias PI: Float32 = 3.14159265358979323846
alias DEG_TO_RAD: Float32 = 0.017453292519943295
alias RAD_TO_DEG: Float32 = 57.29577951308232

alias MMFF_BOND_PREFACTOR: Float32 = 143.9325
alias MMFF_CUBIC_STRETCH: Float32 = -2.0
alias MMFF_ANGLE_PREFACTOR: Float32 = 0.021922
alias MMFF_ANGLE_CUBIC: Float32 = -0.006981317
alias MMFF_TORSION_PREFACTOR: Float32 = 0.5
alias MMFF_VDW_DELTA: Float32 = 0.07
alias MMFF_VDW_GAMMA: Float32 = 0.12
alias MMFF_ELEC_CONSTANT: Float32 = 332.0716
alias MMFF_ELEC_BUFFER: Float32 = 0.05


# =============================================================================
# Energy Components
# =============================================================================

@fieldwise_init
struct EnergyComponents(Copyable, Movable):
    var bond: Float32
    var angle: Float32
    var torsion: Float32
    var vdw: Float32
    var elec: Float32

    fn __init__(out self):
        self.bond = 0.0
        self.angle = 0.0
        self.torsion = 0.0
        self.vdw = 0.0
        self.elec = 0.0

    fn __copyinit__(out self, other: Self):
        self.bond = other.bond
        self.angle = other.angle
        self.torsion = other.torsion
        self.vdw = other.vdw
        self.elec = other.elec

    fn __moveinit__(out self, deinit other: Self):
        self.bond = other.bond
        self.angle = other.angle
        self.torsion = other.torsion
        self.vdw = other.vdw
        self.elec = other.elec

    fn total(self) -> Float32:
        return self.bond + self.angle + self.torsion + self.vdw + self.elec


# =============================================================================
# Energy Functions
# =============================================================================

@always_inline
fn bond_energy_single(r: Float32, r0: Float32, kb: Float32) -> Float32:
    var dr = r - r0
    var dr2 = dr * dr
    var cs = MMFF_CUBIC_STRETCH
    var cs2 = cs * cs
    return Float32(0.5) * MMFF_BOND_PREFACTOR * kb * dr2 * (
        Float32(1.0) + cs * dr + Float32(7.0 / 12.0) * cs2 * dr2
    )


@always_inline
fn angle_energy_single(theta: Float32, theta0: Float32, ka: Float32) -> Float32:
    var dtheta_deg = (theta - theta0) * RAD_TO_DEG
    var dtheta_deg2 = dtheta_deg * dtheta_deg
    return MMFF_ANGLE_PREFACTOR * ka * dtheta_deg2 * (
        Float32(1.0) + MMFF_ANGLE_CUBIC * dtheta_deg
    )


@always_inline
fn torsion_energy_single(phi: Float32, v1: Float32, v2: Float32, v3: Float32) -> Float32:
    var cos_phi = cos(phi)
    var cos_2phi = cos(Float32(2.0) * phi)
    var cos_3phi = cos(Float32(3.0) * phi)
    return MMFF_TORSION_PREFACTOR * (
        v1 * (Float32(1.0) + cos_phi) +
        v2 * (Float32(1.0) - cos_2phi) +
        v3 * (Float32(1.0) + cos_3phi)
    )


@always_inline
fn pow7(x: Float32) -> Float32:
    var x2 = x * x
    var x4 = x2 * x2
    return x * x2 * x4


@always_inline
fn vdw_energy_single(r: Float32, epsilon: Float32, r_star: Float32) -> Float32:
    var r_star_7 = pow7(r_star)
    var denom_rep = r + MMFF_VDW_DELTA * r_star
    var ratio_rep = (Float32(1.0) + MMFF_VDW_DELTA) * r_star / denom_rep
    var rep_term = pow7(ratio_rep)

    var r_7 = pow7(r)
    var denom_att = r_7 + MMFF_VDW_GAMMA * r_star_7
    var att_term = (Float32(1.0) + MMFF_VDW_GAMMA) * r_star_7 / denom_att - Float32(2.0)

    return epsilon * rep_term * att_term


@always_inline
fn elec_energy_single(r: Float32, qi: Float32, qj: Float32, scale: Float32) -> Float32:
    var r_eff = r + MMFF_ELEC_BUFFER
    return scale * MMFF_ELEC_CONSTANT * qi * qj / r_eff


# =============================================================================
# SIMD Fast Math (rsqrt14, rcp14)
# =============================================================================

from sys.intrinsics import llvm_intrinsic


@always_inline
fn rsqrt14_16(a: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    """Approximate 1/sqrt(a) with ~14-bit precision.
    Uses LLVM's intrinsic which maps to VRSQRT14PS on AVX-512."""
    return rsqrt(a)


@always_inline
fn rcp14_16(a: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    """Approximate 1/a with fast reciprocal.
    Uses LLVM's division which can use VRCP14PS on AVX-512."""
    return SIMD[DType.float32, 16](1.0) / a


@always_inline
fn rsqrt14_32(a: SIMD[DType.float32, 32]) -> SIMD[DType.float32, 32]:
    """Approximate 1/sqrt(a) for 32-wide vectors (two AVX-512 ops)."""
    var lo = rsqrt14_16(SIMD[DType.float32, 16](
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]))
    var hi = rsqrt14_16(SIMD[DType.float32, 16](
        a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
        a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31]))
    return SIMD[DType.float32, 32](
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
        lo[8], lo[9], lo[10], lo[11], lo[12], lo[13], lo[14], lo[15],
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7],
        hi[8], hi[9], hi[10], hi[11], hi[12], hi[13], hi[14], hi[15])


@always_inline
fn rcp14_32(a: SIMD[DType.float32, 32]) -> SIMD[DType.float32, 32]:
    """Approximate 1/a for 32-wide vectors (two AVX-512 ops)."""
    var lo = rcp14_16(SIMD[DType.float32, 16](
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]))
    var hi = rcp14_16(SIMD[DType.float32, 16](
        a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
        a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31]))
    return SIMD[DType.float32, 32](
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
        lo[8], lo[9], lo[10], lo[11], lo[12], lo[13], lo[14], lo[15],
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7],
        hi[8], hi[9], hi[10], hi[11], hi[12], hi[13], hi[14], hi[15])


# =============================================================================
# SIMD Energy Functions
# =============================================================================

@always_inline
fn pow7_simd[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Compute x^7 = x * x^2 * x^4 for SIMD vectors."""
    var x2 = x * x
    var x4 = x2 * x2
    return x * x2 * x4


@always_inline
fn vdw_energy_simd[width: Int](
    r: SIMD[DType.float32, width],
    epsilon: SIMD[DType.float32, width],
    r_star: SIMD[DType.float32, width]
) -> SIMD[DType.float32, width]:
    """SIMD vdW energy for 'width' pairs simultaneously."""
    var delta = SIMD[DType.float32, width](MMFF_VDW_DELTA)
    var gamma = SIMD[DType.float32, width](MMFF_VDW_GAMMA)
    var one = SIMD[DType.float32, width](1.0)
    var two = SIMD[DType.float32, width](2.0)

    var r_star_7 = pow7_simd[width](r_star)
    var denom_rep = r + delta * r_star
    var ratio_rep = (one + delta) * r_star / denom_rep
    var rep_term = pow7_simd[width](ratio_rep)

    var r_7 = pow7_simd[width](r)
    var denom_att = r_7 + gamma * r_star_7
    var att_term = (one + gamma) * r_star_7 / denom_att - two

    return epsilon * rep_term * att_term


@always_inline
fn vdw_energy(r: Float32, epsilon: Float32, r_star: Float32) -> Float32:
    """Scalar vdW energy computation for tail processing."""
    var delta = Float32(MMFF_VDW_DELTA)
    var gamma = Float32(MMFF_VDW_GAMMA)

    var r_star_7 = r_star * r_star * r_star * r_star * r_star * r_star * r_star
    var denom_rep = r + delta * r_star
    var ratio_rep = (1.0 + delta) * r_star / denom_rep
    var rep_term = ratio_rep * ratio_rep * ratio_rep * ratio_rep * ratio_rep * ratio_rep * ratio_rep

    var r_7 = r * r * r * r * r * r * r
    var denom_att = r_7 + gamma * r_star_7
    var att_term = (1.0 + gamma) * r_star_7 / denom_att - 2.0

    return epsilon * rep_term * att_term


# =============================================================================
# Multi-Accumulator for Reduced Dependency Chains
# =============================================================================

struct MultiAccum4:
    """4 independent scalar accumulators to break dependency chains."""
    var a0: Float32
    var a1: Float32
    var a2: Float32
    var a3: Float32

    fn __init__(out self):
        self.a0 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0

    @always_inline
    fn add(mut self, idx: Int, value: Float32):
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
        return (self.a0 + self.a1) + (self.a2 + self.a3)


struct SIMDMultiAccum4[width: Int]:
    """4 independent SIMD accumulators."""
    var a0: SIMD[DType.float32, width]
    var a1: SIMD[DType.float32, width]
    var a2: SIMD[DType.float32, width]
    var a3: SIMD[DType.float32, width]

    fn __init__(out self):
        self.a0 = SIMD[DType.float32, width](0)
        self.a1 = SIMD[DType.float32, width](0)
        self.a2 = SIMD[DType.float32, width](0)
        self.a3 = SIMD[DType.float32, width](0)

    @always_inline
    fn add(mut self, idx: Int, value: SIMD[DType.float32, width]):
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
        var sum_vec = (self.a0 + self.a1) + (self.a2 + self.a3)
        return sum_vec.reduce_add()


@always_inline
fn compute_distance(x1: Float32, y1: Float32, z1: Float32,
                    x2: Float32, y2: Float32, z2: Float32) -> Float32:
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    return sqrt(dx * dx + dy * dy + dz * dz)


@always_inline
fn compute_angle(x_i: Float32, y_i: Float32, z_i: Float32,
                 x_j: Float32, y_j: Float32, z_j: Float32,
                 x_k: Float32, y_k: Float32, z_k: Float32) -> Float32:
    var v1x = x_i - x_j
    var v1y = y_i - y_j
    var v1z = z_i - z_j
    var v2x = x_k - x_j
    var v2y = y_k - y_j
    var v2z = z_k - z_j

    var r1 = sqrt(v1x*v1x + v1y*v1y + v1z*v1z)
    var r2 = sqrt(v2x*v2x + v2y*v2y + v2z*v2z)

    if r1 < Float32(1e-10) or r2 < Float32(1e-10):
        return Float32(0.0)

    var dot = v1x*v2x + v1y*v2y + v1z*v2z
    var cos_theta = dot / (r1 * r2)

    if cos_theta > Float32(1.0):
        cos_theta = Float32(1.0)
    elif cos_theta < Float32(-1.0):
        cos_theta = Float32(-1.0)

    return acos(cos_theta)


@always_inline
fn compute_torsion(x_i: Float32, y_i: Float32, z_i: Float32,
                   x_j: Float32, y_j: Float32, z_j: Float32,
                   x_k: Float32, y_k: Float32, z_k: Float32,
                   x_l: Float32, y_l: Float32, z_l: Float32) -> Float32:
    var b1x = x_j - x_i
    var b1y = y_j - y_i
    var b1z = z_j - z_i
    var b2x = x_k - x_j
    var b2y = y_k - y_j
    var b2z = z_k - z_j
    var b3x = x_l - x_k
    var b3y = y_l - y_k
    var b3z = z_l - z_k

    # Cross products
    var n1x = b1y * b2z - b1z * b2y
    var n1y = b1z * b2x - b1x * b2z
    var n1z = b1x * b2y - b1y * b2x
    var n2x = b2y * b3z - b2z * b3y
    var n2y = b2z * b3x - b2x * b3z
    var n2z = b2x * b3y - b2y * b3x

    var n1_norm = sqrt(n1x*n1x + n1y*n1y + n1z*n1z)
    var n2_norm = sqrt(n2x*n2x + n2y*n2y + n2z*n2z)

    if n1_norm < Float32(1e-10) or n2_norm < Float32(1e-10):
        return Float32(0.0)

    n1x /= n1_norm
    n1y /= n1_norm
    n1z /= n1_norm
    n2x /= n2_norm
    n2y /= n2_norm
    n2z /= n2_norm

    var cos_phi = n1x*n2x + n1y*n2y + n1z*n2z
    if cos_phi > Float32(1.0):
        cos_phi = Float32(1.0)
    elif cos_phi < Float32(-1.0):
        cos_phi = Float32(-1.0)

    var b2_norm = sqrt(b2x*b2x + b2y*b2y + b2z*b2z)
    if b2_norm > Float32(1e-10):
        var b2_nx = b2x / b2_norm
        var b2_ny = b2y / b2_norm
        var b2_nz = b2z / b2_norm
        var m1x = n1y * b2_nz - n1z * b2_ny
        var m1y = n1z * b2_nx - n1x * b2_nz
        var m1z = n1x * b2_ny - n1y * b2_nx
        var sin_phi = m1x*n2x + m1y*n2y + m1z*n2z
        return atan2(sin_phi, cos_phi)
    else:
        return acos(cos_phi)


# =============================================================================
# Molecule Batch Data Structure
# =============================================================================

struct MoleculeBatch[StorageType: DType](Movable):
    """Container for batch of molecules with specified storage precision."""

    var num_molecules: Int
    var num_atoms: Int
    var num_bonds: Int
    var num_angles: Int
    var num_torsions: Int
    var num_nb: Int

    # Offset arrays (CSR-style)
    var atom_starts: List[UInt32]
    var bond_starts: List[UInt32]
    var angle_starts: List[UInt32]
    var torsion_starts: List[UInt32]
    var nb_starts: List[UInt32]

    # Atom data (stored in StorageType, loaded as Float32)
    var x: List[Scalar[StorageType]]
    var y: List[Scalar[StorageType]]
    var z: List[Scalar[StorageType]]
    var charge: List[Scalar[StorageType]]

    # Bond data
    var bond_i: List[UInt32]
    var bond_j: List[UInt32]
    var bond_kb: List[Scalar[StorageType]]
    var bond_r0: List[Scalar[StorageType]]

    # Angle data
    var angle_i: List[UInt32]
    var angle_j: List[UInt32]
    var angle_k: List[UInt32]
    var angle_ka: List[Scalar[StorageType]]
    var angle_theta0: List[Scalar[StorageType]]

    # Torsion data
    var torsion_i: List[UInt32]
    var torsion_j: List[UInt32]
    var torsion_k: List[UInt32]
    var torsion_l: List[UInt32]
    var torsion_v1: List[Scalar[StorageType]]
    var torsion_v2: List[Scalar[StorageType]]
    var torsion_v3: List[Scalar[StorageType]]

    # Nonbonded data
    var nb_i: List[UInt32]
    var nb_j: List[UInt32]
    var nb_is_1_4: List[UInt8]
    var nb_epsilon: List[Scalar[StorageType]]
    var nb_r_star: List[Scalar[StorageType]]
    var nb_qi: List[Scalar[StorageType]]
    var nb_qj: List[Scalar[StorageType]]

    fn __init__(out self):
        self.num_molecules = 0
        self.num_atoms = 0
        self.num_bonds = 0
        self.num_angles = 0
        self.num_torsions = 0
        self.num_nb = 0

        self.atom_starts = List[UInt32]()
        self.bond_starts = List[UInt32]()
        self.angle_starts = List[UInt32]()
        self.torsion_starts = List[UInt32]()
        self.nb_starts = List[UInt32]()

        self.x = List[Scalar[StorageType]]()
        self.y = List[Scalar[StorageType]]()
        self.z = List[Scalar[StorageType]]()
        self.charge = List[Scalar[StorageType]]()

        self.bond_i = List[UInt32]()
        self.bond_j = List[UInt32]()
        self.bond_kb = List[Scalar[StorageType]]()
        self.bond_r0 = List[Scalar[StorageType]]()

        self.angle_i = List[UInt32]()
        self.angle_j = List[UInt32]()
        self.angle_k = List[UInt32]()
        self.angle_ka = List[Scalar[StorageType]]()
        self.angle_theta0 = List[Scalar[StorageType]]()

        self.torsion_i = List[UInt32]()
        self.torsion_j = List[UInt32]()
        self.torsion_k = List[UInt32]()
        self.torsion_l = List[UInt32]()
        self.torsion_v1 = List[Scalar[StorageType]]()
        self.torsion_v2 = List[Scalar[StorageType]]()
        self.torsion_v3 = List[Scalar[StorageType]]()

        self.nb_i = List[UInt32]()
        self.nb_j = List[UInt32]()
        self.nb_is_1_4 = List[UInt8]()
        self.nb_epsilon = List[Scalar[StorageType]]()
        self.nb_r_star = List[Scalar[StorageType]]()
        self.nb_qi = List[Scalar[StorageType]]()
        self.nb_qj = List[Scalar[StorageType]]()

    fn __moveinit__(out self, deinit other: Self):
        self.num_molecules = other.num_molecules
        self.num_atoms = other.num_atoms
        self.num_bonds = other.num_bonds
        self.num_angles = other.num_angles
        self.num_torsions = other.num_torsions
        self.num_nb = other.num_nb

        self.atom_starts = other.atom_starts^
        self.bond_starts = other.bond_starts^
        self.angle_starts = other.angle_starts^
        self.torsion_starts = other.torsion_starts^
        self.nb_starts = other.nb_starts^

        self.x = other.x^
        self.y = other.y^
        self.z = other.z^
        self.charge = other.charge^

        self.bond_i = other.bond_i^
        self.bond_j = other.bond_j^
        self.bond_kb = other.bond_kb^
        self.bond_r0 = other.bond_r0^

        self.angle_i = other.angle_i^
        self.angle_j = other.angle_j^
        self.angle_k = other.angle_k^
        self.angle_ka = other.angle_ka^
        self.angle_theta0 = other.angle_theta0^

        self.torsion_i = other.torsion_i^
        self.torsion_j = other.torsion_j^
        self.torsion_k = other.torsion_k^
        self.torsion_l = other.torsion_l^
        self.torsion_v1 = other.torsion_v1^
        self.torsion_v2 = other.torsion_v2^
        self.torsion_v3 = other.torsion_v3^

        self.nb_i = other.nb_i^
        self.nb_j = other.nb_j^
        self.nb_is_1_4 = other.nb_is_1_4^
        self.nb_epsilon = other.nb_epsilon^
        self.nb_r_star = other.nb_r_star^
        self.nb_qi = other.nb_qi^
        self.nb_qj = other.nb_qj^


fn compute_molecule_energy[StorageType: DType](
    batch: MoleculeBatch[StorageType],
    mol_idx: Int
) -> EnergyComponents:
    """Compute MMFF94 energy for a single molecule in the batch."""
    var result = EnergyComponents()

    var atom_start = Int(batch.atom_starts[mol_idx])
    var bond_start = Int(batch.bond_starts[mol_idx])
    var bond_end = Int(batch.bond_starts[mol_idx + 1])
    var angle_start = Int(batch.angle_starts[mol_idx])
    var angle_end = Int(batch.angle_starts[mol_idx + 1])
    var torsion_start = Int(batch.torsion_starts[mol_idx])
    var torsion_end = Int(batch.torsion_starts[mol_idx + 1])
    var nb_start = Int(batch.nb_starts[mol_idx])
    var nb_end = Int(batch.nb_starts[mol_idx + 1])

    # Bond stretching
    for b in range(bond_start, bond_end):
        var i = Int(batch.bond_i[b])
        var j = Int(batch.bond_j[b])
        var ai = atom_start + i
        var aj = atom_start + j

        var r = compute_distance(
            Float32(batch.x[ai]), Float32(batch.y[ai]), Float32(batch.z[ai]),
            Float32(batch.x[aj]), Float32(batch.y[aj]), Float32(batch.z[aj])
        )
        result.bond += bond_energy_single(r, Float32(batch.bond_r0[b]), Float32(batch.bond_kb[b]))

    # Angle bending
    for a in range(angle_start, angle_end):
        var i = Int(batch.angle_i[a])
        var j = Int(batch.angle_j[a])
        var k = Int(batch.angle_k[a])
        var ai = atom_start + i
        var aj = atom_start + j
        var ak = atom_start + k

        var theta = compute_angle(
            Float32(batch.x[ai]), Float32(batch.y[ai]), Float32(batch.z[ai]),
            Float32(batch.x[aj]), Float32(batch.y[aj]), Float32(batch.z[aj]),
            Float32(batch.x[ak]), Float32(batch.y[ak]), Float32(batch.z[ak])
        )
        result.angle += angle_energy_single(theta, Float32(batch.angle_theta0[a]), Float32(batch.angle_ka[a]))

    # Torsion
    for t in range(torsion_start, torsion_end):
        var i = Int(batch.torsion_i[t])
        var j = Int(batch.torsion_j[t])
        var k = Int(batch.torsion_k[t])
        var l = Int(batch.torsion_l[t])
        var ai = atom_start + i
        var aj = atom_start + j
        var ak = atom_start + k
        var al = atom_start + l

        var phi = compute_torsion(
            Float32(batch.x[ai]), Float32(batch.y[ai]), Float32(batch.z[ai]),
            Float32(batch.x[aj]), Float32(batch.y[aj]), Float32(batch.z[aj]),
            Float32(batch.x[ak]), Float32(batch.y[ak]), Float32(batch.z[ak]),
            Float32(batch.x[al]), Float32(batch.y[al]), Float32(batch.z[al])
        )
        result.torsion += torsion_energy_single(
            phi,
            Float32(batch.torsion_v1[t]),
            Float32(batch.torsion_v2[t]),
            Float32(batch.torsion_v3[t])
        )

    # Nonbonded (vdW + electrostatic) - SIMD batch processing
    # Structures are padded to SIMD width multiples, no tail code needed
    var num_pairs = nb_end - nb_start

    # Process only complete SIMD batches (16 pairs at a time)
    # Structures should be padded to SIMD multiples, but be safe
    var simd_pairs = (num_pairs // 16) * 16

    var vdw_accum = SIMDMultiAccum4[16]()
    var elec_accum = SIMDMultiAccum4[16]()

    for nb_base in range(0, simd_pairs, 16):
        var n = nb_start + nb_base

        # Gather atom indices and positions for 16 pairs
        var x_i = SIMD[DType.float32, 16]()
        var y_i = SIMD[DType.float32, 16]()
        var z_i = SIMD[DType.float32, 16]()
        var x_j = SIMD[DType.float32, 16]()
        var y_j = SIMD[DType.float32, 16]()
        var z_j = SIMD[DType.float32, 16]()
        var eps = SIMD[DType.float32, 16]()
        var rstar = SIMD[DType.float32, 16]()
        var qi = SIMD[DType.float32, 16]()
        var qj = SIMD[DType.float32, 16]()
        var is_1_4 = SIMD[DType.float32, 16]()

        @parameter
        for k in range(16):
            var pair_idx = n + k
            var i = Int(batch.nb_i[pair_idx])
            var j = Int(batch.nb_j[pair_idx])
            var ai = atom_start + i
            var aj = atom_start + j

            x_i[k] = Float32(batch.x[ai])
            y_i[k] = Float32(batch.y[ai])
            z_i[k] = Float32(batch.z[ai])
            x_j[k] = Float32(batch.x[aj])
            y_j[k] = Float32(batch.y[aj])
            z_j[k] = Float32(batch.z[aj])
            eps[k] = Float32(batch.nb_epsilon[pair_idx])
            rstar[k] = Float32(batch.nb_r_star[pair_idx])
            qi[k] = Float32(batch.nb_qi[pair_idx])
            qj[k] = Float32(batch.nb_qj[pair_idx])
            is_1_4[k] = Float32(1.0) if batch.nb_is_1_4[pair_idx] == 1 else Float32(0.0)

        # Compute distances using rsqrt14 (fast approximation)
        var dx = x_j - x_i
        var dy = y_j - y_i
        var dz = z_j - z_i
        var r_sq = dx * dx + dy * dy + dz * dz
        var inv_r = rsqrt14_16(r_sq)
        var r = r_sq * inv_r

        # vdW energy (no 1-4 scaling)
        # Note: Invalid distances (r < 0.1) will produce large positive vdW energies
        # but since all structures are padded, all pairs should be valid
        var vdw_e = vdw_energy_simd[16](r, eps, rstar)
        vdw_accum.add(nb_base // 16, vdw_e)

        # Electrostatic energy (0.75 scaling for 1-4 pairs)
        var qq = qi * qj
        var elec_const = SIMD[DType.float32, 16](MMFF_ELEC_CONSTANT)
        var elec_buffer = SIMD[DType.float32, 16](MMFF_ELEC_BUFFER)
        var r_eff = r + elec_buffer

        # Scale: 1.0 for normal pairs, 0.75 for 1-4 pairs
        var one = SIMD[DType.float32, 16](1.0)
        var scale_1_4 = SIMD[DType.float32, 16](0.75)
        var scale = one + is_1_4 * (scale_1_4 - one)

        var elec_e = scale * elec_const * qq * rcp14_16(r_eff)
        elec_accum.add(nb_base // 16, elec_e)

    result.vdw = vdw_accum.reduce()
    result.elec = elec_accum.reduce()

    # Scalar tail for remaining pairs (if num_pairs not divisible by 16)
    for nb_idx in range(simd_pairs, num_pairs):
        var pair_idx = nb_start + nb_idx
        var i = Int(batch.nb_i[pair_idx])
        var j = Int(batch.nb_j[pair_idx])
        var ai = atom_start + i
        var aj = atom_start + j

        var xi = Float32(batch.x[ai])
        var yi = Float32(batch.y[ai])
        var zi = Float32(batch.z[ai])
        var xj = Float32(batch.x[aj])
        var yj = Float32(batch.y[aj])
        var zj = Float32(batch.z[aj])

        var dx = xj - xi
        var dy = yj - yi
        var dz = zj - zi
        var r = sqrt(dx*dx + dy*dy + dz*dz)

        if r >= 0.1:
            var eps = Float32(batch.nb_epsilon[pair_idx])
            var rstar = Float32(batch.nb_r_star[pair_idx])
            result.vdw += vdw_energy(r, eps, rstar)

            var qi_val = Float32(batch.nb_qi[pair_idx])
            var qj_val = Float32(batch.nb_qj[pair_idx])
            var qi_abs = qi_val if qi_val > 0 else -qi_val
            var qj_abs = qj_val if qj_val > 0 else -qj_val
            if qi_abs > 1e-10 and qj_abs > 1e-10:
                var r_eff = r + MMFF_ELEC_BUFFER
                var scale: Float32 = Float32(0.75) if batch.nb_is_1_4[pair_idx] == 1 else Float32(1.0)
                result.elec += scale * MMFF_ELEC_CONSTANT * qi_val * qj_val / r_eff

    return result^


fn read_binary_data[StorageType: DType](file_path: String) raises -> MoleculeBatch[StorageType]:
    """Read molecule data from binary file and convert to StorageType."""
    var batch = MoleculeBatch[StorageType]()

    # Read entire file - keep data in scope
    var f = open(file_path, "r")
    var data = f.read_bytes()
    f.close()

    var ptr = data.unsafe_ptr()
    var offset = 0

    # Parse header (64 bytes)
    var magic = ptr.bitcast[UInt32]().load()
    offset += 4

    if magic != 0x4C4D4653:
        raise Error("Invalid magic number in binary file")

    _ = (ptr + 4).bitcast[UInt32]().load()  # version (unused)
    offset += 4

    batch.num_molecules = Int((ptr + 8).bitcast[UInt32]().load())
    batch.num_atoms = Int((ptr + 12).bitcast[UInt32]().load())
    batch.num_bonds = Int((ptr + 16).bitcast[UInt32]().load())
    batch.num_angles = Int((ptr + 20).bitcast[UInt32]().load())
    batch.num_torsions = Int((ptr + 24).bitcast[UInt32]().load())
    batch.num_nb = Int((ptr + 28).bitcast[UInt32]().load())

    offset = 64  # Skip to end of header

    var n_mol_plus_1 = batch.num_molecules + 1

    # Read offset arrays
    var offset_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(n_mol_plus_1):
        batch.atom_starts.append(offset_ptr[i])
    offset += n_mol_plus_1 * 4

    offset_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(n_mol_plus_1):
        batch.bond_starts.append(offset_ptr[i])
    offset += n_mol_plus_1 * 4

    offset_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(n_mol_plus_1):
        batch.angle_starts.append(offset_ptr[i])
    offset += n_mol_plus_1 * 4

    offset_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(n_mol_plus_1):
        batch.torsion_starts.append(offset_ptr[i])
    offset += n_mol_plus_1 * 4

    offset_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(n_mol_plus_1):
        batch.nb_starts.append(offset_ptr[i])
    offset += n_mol_plus_1 * 4

    # Read atom data (f64 in file, convert to StorageType)
    var atom_f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_atoms):
        batch.x.append(Scalar[StorageType](atom_f64_ptr[i]))
    offset += batch.num_atoms * 8

    atom_f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_atoms):
        batch.y.append(Scalar[StorageType](atom_f64_ptr[i]))
    offset += batch.num_atoms * 8

    atom_f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_atoms):
        batch.z.append(Scalar[StorageType](atom_f64_ptr[i]))
    offset += batch.num_atoms * 8

    atom_f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_atoms):
        batch.charge.append(Scalar[StorageType](atom_f64_ptr[i]))
    offset += batch.num_atoms * 8

    # Read bond data
    var u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_bonds):
        batch.bond_i.append(u32_ptr[i])
    offset += batch.num_bonds * 4

    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_bonds):
        batch.bond_j.append(u32_ptr[i])
    offset += batch.num_bonds * 4

    var f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_bonds):
        batch.bond_kb.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_bonds * 8

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_bonds):
        batch.bond_r0.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_bonds * 8

    # Read angle data
    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_angles):
        batch.angle_i.append(u32_ptr[i])
    offset += batch.num_angles * 4

    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_angles):
        batch.angle_j.append(u32_ptr[i])
    offset += batch.num_angles * 4

    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_angles):
        batch.angle_k.append(u32_ptr[i])
    offset += batch.num_angles * 4

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_angles):
        batch.angle_ka.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_angles * 8

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_angles):
        batch.angle_theta0.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_angles * 8

    # Read torsion data
    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_torsions):
        batch.torsion_i.append(u32_ptr[i])
    offset += batch.num_torsions * 4

    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_torsions):
        batch.torsion_j.append(u32_ptr[i])
    offset += batch.num_torsions * 4

    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_torsions):
        batch.torsion_k.append(u32_ptr[i])
    offset += batch.num_torsions * 4

    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_torsions):
        batch.torsion_l.append(u32_ptr[i])
    offset += batch.num_torsions * 4

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_torsions):
        batch.torsion_v1.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_torsions * 8

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_torsions):
        batch.torsion_v2.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_torsions * 8

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_torsions):
        batch.torsion_v3.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_torsions * 8

    # Read nonbonded data
    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_nb):
        batch.nb_i.append(u32_ptr[i])
    offset += batch.num_nb * 4

    u32_ptr = (ptr + offset).bitcast[UInt32]()
    for i in range(batch.num_nb):
        batch.nb_j.append(u32_ptr[i])
    offset += batch.num_nb * 4

    var u8_ptr = (ptr + offset).bitcast[UInt8]()
    for i in range(batch.num_nb):
        batch.nb_is_1_4.append(u8_ptr[i])
    offset += batch.num_nb

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_nb):
        batch.nb_epsilon.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_nb * 8

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_nb):
        batch.nb_r_star.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_nb * 8

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_nb):
        batch.nb_qi.append(Scalar[StorageType](f64_ptr[i]))
    offset += batch.num_nb * 8

    f64_ptr = (ptr + offset).bitcast[Float64]()
    for i in range(batch.num_nb):
        batch.nb_qj.append(Scalar[StorageType](f64_ptr[i]))

    return batch^


fn run_benchmark[StorageType: DType](input_path: String) raises:
    """Run benchmark for specified storage type."""

    # Log SIMD capabilities
    print("# SIMD Configuration")
    print("# AVX-512:", HAS_AVX512)
    print("# AVX-512 BF16 (VDPBF16PS):", HAS_AVX512_BF16)
    print("# AVX-512 FP16:", HAS_AVX512_FP16)
    print("# CPU cores:", num_physical_cores())
    print("#")

    # Read data
    var batch = read_binary_data[StorageType](input_path)

    print("# Loaded", batch.num_molecules, "molecules")
    print("# Atoms:", batch.num_atoms, "Bonds:", batch.num_bonds)
    print("# Angles:", batch.num_angles, "Torsions:", batch.num_torsions)
    print("# Nonbonded pairs:", batch.num_nb)
    print("#")

    # Warm-up
    for i in range(min(10, batch.num_molecules)):
        _ = compute_molecule_energy[StorageType](batch, i)

    # Timed benchmark
    var start = perf_counter_ns()

    for mol_idx in range(batch.num_molecules):
        var energy = compute_molecule_energy[StorageType](batch, mol_idx)
        print(mol_idx, energy.bond, energy.angle, energy.torsion, energy.vdw, energy.elec)

    var elapsed_ns = perf_counter_ns() - start

    # Summary line
    print("TIMING", elapsed_ns, batch.num_molecules)


fn main() raises:
    var args = argv()

    if len(args) < 3:
        print("Usage: runner.mojo <input.bin> <storage_type>")
        print("  storage_type: f16, bf16, or f32")
        return

    var input_path = String(args[1])
    var storage_type = String(args[2])

    if storage_type == "f16":
        run_benchmark[DType.float16](input_path)
    elif storage_type == "bf16":
        run_benchmark[DType.bfloat16](input_path)
    elif storage_type == "f32":
        run_benchmark[DType.float32](input_path)
    else:
        print("Unknown storage type:", storage_type)
        print("Use: f16, bf16, or f32")
