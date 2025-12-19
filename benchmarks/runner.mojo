# SmallField Unified Benchmark: Native Mojo Runner
#
# Reads pre-parameterized molecule data from binary file and computes
# MMFF94 energies using native f16, bf16, or f32 storage types.
#
# Usage: mojo run runner.mojo <input.bin> <storage_type>
#   storage_type: f16, bf16, or f32

from sys import argv, num_physical_cores
from time import perf_counter_ns
from math import sqrt, cos, acos, atan2
from memory import UnsafePointer
from algorithm import parallelize

# SIMD configuration (compile-time)
alias HAS_AVX512: Bool = True
alias HAS_AVX512_BF16: Bool = False
alias HAS_AVX512_FP16: Bool = False
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

    # Nonbonded (vdW + electrostatic)
    for n in range(nb_start, nb_end):
        var i = Int(batch.nb_i[n])
        var j = Int(batch.nb_j[n])
        var ai = atom_start + i
        var aj = atom_start + j

        var r = compute_distance(
            Float32(batch.x[ai]), Float32(batch.y[ai]), Float32(batch.z[ai]),
            Float32(batch.x[aj]), Float32(batch.y[aj]), Float32(batch.z[aj])
        )

        if r < Float32(0.1):
            continue

        # vdW (no 1-4 scaling)
        result.vdw += vdw_energy_single(r, Float32(batch.nb_epsilon[n]), Float32(batch.nb_r_star[n]))

        # Electrostatic (0.75 scaling for 1-4)
        var qi = Float32(batch.nb_qi[n])
        var qj = Float32(batch.nb_qj[n])
        if qi * qi > Float32(1e-20) and qj * qj > Float32(1e-20):
            var scale = Float32(0.75) if batch.nb_is_1_4[n] == 1 else Float32(1.0)
            result.elec += elec_energy_single(r, qi, qj, scale)

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
