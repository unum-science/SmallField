# SmallField: MMFF94 Force Field Implementation
# Merged from: bond_stretch.mojo, angle_bend.mojo, torsion.mojo, vdw.mojo, electrostatic.mojo, total.mojo
# Updated for Mojo 0.25.7

from math import sqrt, cos, sin, acos, atan2
from memory import UnsafePointer

fn abs_f32(x: Float32) -> Float32:
    """Absolute value for Float32."""
    if x < Float32(0.0):
        return -x
    return x

# Mathematical constants
alias PI: Float32 = 3.14159265358979323846
alias DEG_TO_RAD: Float32 = 0.017453292519943295
alias RAD_TO_DEG: Float32 = 57.29577951308232

# Bond stretching
alias MMFF_BOND_PREFACTOR: Float32 = 143.9325  # mdyn*A/rad^2 -> kcal/mol
alias MMFF_CUBIC_STRETCH: Float32 = -2.0       # A^-1

# Angle bending
# c2 = MDYNE_A_TO_KCAL_MOL * DEG2RAD^2 = 143.9325 * (π/180)^2 = 0.043844
# MMFF_ANGLE_PREFACTOR = 0.5 * c2 = 0.021922
alias MMFF_ANGLE_PREFACTOR: Float32 = 0.021922  # 0.5 * 0.043844 (mdyn*A/rad^2 -> kcal/mol)
alias MMFF_ANGLE_CUBIC: Float32 = -0.006981317  # Exact RDKit cb value (deg^-1)

# Electrostatics
alias MMFF_ELEC_CONSTANT: Float32 = 332.0716  # kcal*A/mol/e^2
alias MMFF_ELEC_BUFFER: Float32 = 0.05        # Angstroms - prevents singularity

# Default parameters
alias DEFAULT_DIELECTRIC: Float32 = 1.0
alias DEFAULT_1_4_VDW_SCALE: Float32 = 0.75
alias DEFAULT_1_4_ELEC_SCALE: Float32 = 0.75

# Utility functions
fn compute_distance(
    x1: Float32, y1: Float32, z1: Float32,
    x2: Float32, y2: Float32, z2: Float32
) -> Float32:
    """Compute Euclidean distance between two points."""
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    return sqrt(dx * dx + dy * dy + dz * dz)

# SIMD width constants
alias SIMD_WIDTH_SCALAR: Int = 1
alias SIMD_WIDTH_AVX512: Int = 16

# Bond stretching
# E_bond = 0.5 * 143.9325 * kb * dr^2 * (1 + cs*dr + 7/12*cs^2*dr^2)
fn bond_energy[width: Int = 1](
    r: SIMD[DType.float32, width],
    r0: SIMD[DType.float32, width],
    kb: SIMD[DType.float32, width]
) -> SIMD[DType.float32, width]:
    """
    MMFF94 bond stretching energy.

    E = 0.5 * 143.9325 * kb * dr^2 * (1 + cs*dr + 7/12*cs^2*dr^2)

    where:
        dr = r - r0
        cs = -2.0 (cubic stretch constant)

    When width=1, operates on scalars.
    When width=16, processes 16 bonds in parallel (AVX-512).
    """
    var dr = r - r0
    var dr2 = dr * dr
    var cs = SIMD[DType.float32, width](MMFF_CUBIC_STRETCH)
    var cs2 = cs * cs
    var half = SIMD[DType.float32, width](0.5)
    var one = SIMD[DType.float32, width](1.0)
    var prefactor = SIMD[DType.float32, width](MMFF_BOND_PREFACTOR)
    var cubic_factor = SIMD[DType.float32, width](7.0 / 12.0)

    return half * prefactor * kb * dr2 * (one + cs * dr + cubic_factor * cs2 * dr2)


# Scalar convenience wrapper for backward compatibility
fn bond_energy_single(r: Float32, r0: Float32, kb: Float32) -> Float32:
    """Scalar bond energy (calls SIMD version with width=1)."""
    return bond_energy[1](r, r0, kb)[0]

fn bond_gradient[width: Int = 1](
    r: SIMD[DType.float32, width],
    r0: SIMD[DType.float32, width],
    kb: SIMD[DType.float32, width]
) -> SIMD[DType.float32, width]:
    """
    Compute bond energy gradient dE/dr.

    dE/dr = 143.9325 * kb * dr * (1 + 1.5*cs*dr + 7/3*cs^2*dr^2)
    """
    var dr = r - r0
    var dr2 = dr * dr
    var cs = SIMD[DType.float32, width](MMFF_CUBIC_STRETCH)
    var cs2 = cs * cs
    var one = SIMD[DType.float32, width](1.0)
    var prefactor = SIMD[DType.float32, width](MMFF_BOND_PREFACTOR)

    return prefactor * kb * dr * (
        one + SIMD[DType.float32, width](1.5) * cs * dr +
        SIMD[DType.float32, width](7.0 / 3.0) * cs2 * dr2
    )


fn bond_energy_and_gradient(
    r: Float32,
    r0: Float32,
    kb: Float32
) -> Tuple[Float32, Float32]:
    """Scalar bond energy and gradient (backward compatibility)."""
    var energy = bond_energy[1](r, r0, kb)[0]
    var gradient = bond_gradient[1](r, r0, kb)[0]
    return (energy, gradient)

fn compute_bond_forces[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Gradients (output)
    mut grad_x: UnsafePointer[Scalar[StorageType]],
    mut grad_y: UnsafePointer[Scalar[StorageType]],
    mut grad_z: UnsafePointer[Scalar[StorageType]],
    # Bond topology
    bond_i: UnsafePointer[Int32],
    bond_j: UnsafePointer[Int32],
    # Bond parameters
    bond_kb: UnsafePointer[Scalar[StorageType]],
    bond_r0: UnsafePointer[Scalar[StorageType]],
    # Range
    bond_start: Int,
    bond_end: Int
) -> Float32:
    """
    Compute bond stretching energy and accumulate forces.

    Returns total bond energy for the given range.
    Accumulates gradients into grad_x, grad_y, grad_z.
    """
    var total_energy: Float32 = 0.0

    for b in range(bond_start, bond_end):
        var i = Int(bond_i[b])
        var j = Int(bond_j[b])

        # Get positions and convert to Float32
        var x1 = Float32(x[i])
        var y1 = Float32(y[i])
        var z1 = Float32(z[i])
        var x2 = Float32(x[j])
        var y2 = Float32(y[j])
        var z2 = Float32(z[j])

        # Compute distance
        var dx = x2 - x1
        var dy = y2 - y1
        var dz = z2 - z1
        var r = sqrt(dx * dx + dy * dy + dz * dz)

        if r < Float32(1e-8):
            continue

        # Compute energy and gradient
        var r0 = Float32(bond_r0[b])
        var kb = Float32(bond_kb[b])
        var result = bond_energy_and_gradient(r, r0, kb)
        var energy = result[0]
        var dE_dr = result[1]

        total_energy += energy

        # Convert dE/dr to dE/d(xyz) using chain rule
        var inv_r = Float32(1.0) / r
        var fx = dE_dr * dx * inv_r
        var fy = dE_dr * dy * inv_r
        var fz = dE_dr * dz * inv_r

        # Accumulate gradients using store()
        grad_x[i] = grad_x[i] - Scalar[StorageType](fx)
        grad_y[i] = grad_y[i] - Scalar[StorageType](fy)
        grad_z[i] = grad_z[i] - Scalar[StorageType](fz)
        grad_x[j] = grad_x[j] + Scalar[StorageType](fx)
        grad_y[j] = grad_y[j] + Scalar[StorageType](fy)
        grad_z[j] = grad_z[j] + Scalar[StorageType](fz)

    return total_energy

fn compute_total_bond_energy[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Bond topology
    bond_i: UnsafePointer[Int32],
    bond_j: UnsafePointer[Int32],
    # Bond parameters
    bond_kb: UnsafePointer[Scalar[StorageType]],
    bond_r0: UnsafePointer[Scalar[StorageType]],
    # Number of bonds
    num_bonds: Int
) -> Float32:
    """
    Compute total bond stretching energy (no gradients).

    Optimized for energy-only evaluation.
    """
    var total_energy: Float32 = 0.0

    for b in range(num_bonds):
        var i = Int(bond_i[b])
        var j = Int(bond_j[b])

        # Get positions and convert to Float32
        var x1 = Float32(x[i])
        var y1 = Float32(y[i])
        var z1 = Float32(z[i])
        var x2 = Float32(x[j])
        var y2 = Float32(y[j])
        var z2 = Float32(z[j])

        # Compute distance and energy
        var r = compute_distance(x1, y1, z1, x2, y2, z2)
        var energy = bond_energy_single(r, Float32(bond_r0[b]), Float32(bond_kb[b]))
        total_energy += energy

    return total_energy

fn compute_total_bond_energy_kahan[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Bond topology
    bond_i: UnsafePointer[Int32],
    bond_j: UnsafePointer[Int32],
    # Bond parameters
    bond_kb: UnsafePointer[Scalar[StorageType]],
    bond_r0: UnsafePointer[Scalar[StorageType]],
    # Number of bonds
    num_bonds: Int
) -> Float32:
    """
    Compute total bond stretching energy with Kahan summation.

    More accurate accumulation for many bonds to prevent FP32 drift.
    """
    var total_energy: Float32 = 0.0
    var compensation: Float32 = 0.0

    for b in range(num_bonds):
        var i = Int(bond_i[b])
        var j = Int(bond_j[b])

        # Get positions and convert to Float32
        var x1 = Float32(x[i])
        var y1 = Float32(y[i])
        var z1 = Float32(z[i])
        var x2 = Float32(x[j])
        var y2 = Float32(y[j])
        var z2 = Float32(z[j])

        # Compute distance and energy
        var r = compute_distance(x1, y1, z1, x2, y2, z2)
        var energy = bond_energy_single(r, Float32(bond_r0[b]), Float32(bond_kb[b]))

        # Kahan summation
        var y_val = energy - compensation
        var t = total_energy + y_val
        compensation = (t - total_energy) - y_val
        total_energy = t

    return total_energy

# Angle bending
# E_angle = 0.043844 * ka * dtheta^2 * (1 - 0.007*dtheta)
fn compute_angle(
    x1: Float32, y1: Float32, z1: Float32,  # Atom i
    x2: Float32, y2: Float32, z2: Float32,  # Atom j (central)
    x3: Float32, y3: Float32, z3: Float32   # Atom k
) -> Float32:
    """
    Compute angle i-j-k in radians.

    Returns the angle at central atom j.
    """
    # Vectors from central atom j to i and k
    var v1x = x1 - x2
    var v1y = y1 - y2
    var v1z = z1 - z2

    var v2x = x3 - x2
    var v2y = y3 - y2
    var v2z = z3 - z2

    # Magnitudes
    var r1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    var r2 = sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

    if r1 < Float32(1e-8) or r2 < Float32(1e-8):
        return Float32(0.0)

    # Dot product and cosine
    var dot = v1x * v2x + v1y * v2y + v1z * v2z
    var cos_theta = dot / (r1 * r2)

    # Clamp to [-1, 1] to avoid acos domain errors
    if cos_theta > Float32(1.0):
        cos_theta = Float32(1.0)
    elif cos_theta < Float32(-1.0):
        cos_theta = Float32(-1.0)

    return acos(cos_theta)

fn angle_energy[width: Int = 1](
    theta: SIMD[DType.float32, width],
    theta0: SIMD[DType.float32, width],
    ka: SIMD[DType.float32, width]
) -> SIMD[DType.float32, width]:
    """
    MMFF94 angle bending energy.

    E = 0.043844 * ka * dtheta^2 * (1 - 0.007*dtheta)

    where:
        dtheta = theta - theta0 (in degrees for the cubic term)

    When width=1, operates on scalars.
    When width=16, processes 16 angles in parallel (AVX-512).
    """
    var dtheta_rad = theta - theta0
    var rad_to_deg = SIMD[DType.float32, width](RAD_TO_DEG)
    var dtheta_deg = dtheta_rad * rad_to_deg
    var dtheta_deg2 = dtheta_deg * dtheta_deg

    var prefactor = SIMD[DType.float32, width](MMFF_ANGLE_PREFACTOR)
    var cubic = SIMD[DType.float32, width](MMFF_ANGLE_CUBIC)
    var one = SIMD[DType.float32, width](1.0)

    return prefactor * ka * dtheta_deg2 * (one + cubic * dtheta_deg)


fn angle_energy_single(theta: Float32, theta0: Float32, ka: Float32) -> Float32:
    """Scalar angle energy (calls SIMD version with width=1)."""
    return angle_energy[1](theta, theta0, ka)[0]

fn angle_energy_and_gradient(
    theta: Float32,
    theta0: Float32,
    ka: Float32
) -> Tuple[Float32, Float32]:
    """
    Compute angle energy and its gradient dE/dtheta.

    dE/dtheta = 0.043844 * ka * 2*dtheta * (1 - 0.007*dtheta) +
                0.043844 * ka * dtheta^2 * (-0.007)
              = 0.043844 * ka * dtheta * (2 - 0.021*dtheta)

    Returns (energy, dE/d_theta_deg) - gradient w.r.t degrees.
    """
    var dtheta_rad = theta - theta0
    var dtheta_deg = dtheta_rad * RAD_TO_DEG
    var dtheta_deg2 = dtheta_deg * dtheta_deg

    var energy = MMFF_ANGLE_PREFACTOR * ka * dtheta_deg2 * (
        Float32(1.0) + MMFF_ANGLE_CUBIC * dtheta_deg
    )

    # Gradient w.r.t. theta in degrees
    var gradient_deg = MMFF_ANGLE_PREFACTOR * ka * dtheta_deg * (
        Float32(2.0) + Float32(3.0) * MMFF_ANGLE_CUBIC * dtheta_deg
    )

    # Convert to gradient w.r.t. theta in radians
    var gradient_rad = gradient_deg * RAD_TO_DEG

    return (energy, gradient_rad)

fn compute_angle_forces[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Gradients (output)
    mut grad_x: UnsafePointer[Scalar[StorageType]],
    mut grad_y: UnsafePointer[Scalar[StorageType]],
    mut grad_z: UnsafePointer[Scalar[StorageType]],
    # Angle topology
    angle_i: UnsafePointer[Int32],
    angle_j: UnsafePointer[Int32],
    angle_k: UnsafePointer[Int32],
    # Angle parameters
    angle_ka: UnsafePointer[Scalar[StorageType]],
    angle_theta0: UnsafePointer[Scalar[StorageType]],
    # Range
    angle_start: Int,
    angle_end: Int
) -> Float32:
    """
    Compute angle bending energy and accumulate forces.

    Returns total angle energy for the given range.
    Accumulates gradients into grad_x, grad_y, grad_z.
    """
    var total_energy: Float32 = 0.0

    for a in range(angle_start, angle_end):
        var i = Int(angle_i[a])
        var j = Int(angle_j[a])  # Central atom
        var k = Int(angle_k[a])

        # Get positions and convert to Float32
        var x1 = Float32(x[i])
        var y1 = Float32(y[i])
        var z1 = Float32(z[i])
        var x2 = Float32(x[j])
        var y2 = Float32(y[j])
        var z2 = Float32(z[j])
        var x3 = Float32(x[k])
        var y3 = Float32(y[k])
        var z3 = Float32(z[k])

        # Vectors from central atom
        var v1x = x1 - x2
        var v1y = y1 - y2
        var v1z = z1 - z2
        var v2x = x3 - x2
        var v2y = y3 - y2
        var v2z = z3 - z2

        var r1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        var r2 = sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

        if r1 < Float32(1e-8) or r2 < Float32(1e-8):
            continue

        # Compute angle
        var dot = v1x * v2x + v1y * v2y + v1z * v2z
        var cos_theta = dot / (r1 * r2)
        if cos_theta > Float32(1.0):
            cos_theta = Float32(1.0)
        elif cos_theta < Float32(-1.0):
            cos_theta = Float32(-1.0)

        var theta = acos(cos_theta)
        var sin_theta = sin(theta)

        # Avoid singularity at linear angles
        if sin_theta < Float32(1e-8):
            sin_theta = Float32(1e-8)

        # Compute energy and gradient
        var result = angle_energy_and_gradient(theta, Float32(angle_theta0[a]), Float32(angle_ka[a]))
        var energy = result[0]
        var dE_dtheta = result[1]

        total_energy += energy

        # Compute dtheta/d(xyz) using chain rule
        var inv_r1 = Float32(1.0) / r1
        var inv_r2 = Float32(1.0) / r2
        var inv_sin = Float32(1.0) / sin_theta

        # Normalize vectors
        var u1x = v1x * inv_r1
        var u1y = v1y * inv_r1
        var u1z = v1z * inv_r1
        var u2x = v2x * inv_r2
        var u2y = v2y * inv_r2
        var u2z = v2z * inv_r2

        # dtheta/d(atom_i) = -1/(r1*sin_theta) * (u2 - cos_theta*u1)
        var factor_i = -dE_dtheta * inv_r1 * inv_sin
        var fix = factor_i * (u2x - cos_theta * u1x)
        var fiy = factor_i * (u2y - cos_theta * u1y)
        var fiz = factor_i * (u2z - cos_theta * u1z)

        # dtheta/d(atom_k) = -1/(r2*sin_theta) * (u1 - cos_theta*u2)
        var factor_k = -dE_dtheta * inv_r2 * inv_sin
        var fkx = factor_k * (u1x - cos_theta * u2x)
        var fky = factor_k * (u1y - cos_theta * u2y)
        var fkz = factor_k * (u1z - cos_theta * u2z)

        # Force on central atom j = -(force_i + force_k)
        var fjx = -(fix + fkx)
        var fjy = -(fiy + fky)
        var fjz = -(fiz + fkz)

        # Accumulate gradients using store()
        grad_x[i] = grad_x[i] + Scalar[StorageType](fix)
        grad_y[i] = grad_y[i] + Scalar[StorageType](fiy)
        grad_z[i] = grad_z[i] + Scalar[StorageType](fiz)
        grad_x[j] = grad_x[j] + Scalar[StorageType](fjx)
        grad_y[j] = grad_y[j] + Scalar[StorageType](fjy)
        grad_z[j] = grad_z[j] + Scalar[StorageType](fjz)
        grad_x[k] = grad_x[k] + Scalar[StorageType](fkx)
        grad_y[k] = grad_y[k] + Scalar[StorageType](fky)
        grad_z[k] = grad_z[k] + Scalar[StorageType](fkz)

    return total_energy

fn compute_total_angle_energy[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Angle topology
    angle_i: UnsafePointer[Int32],
    angle_j: UnsafePointer[Int32],
    angle_k: UnsafePointer[Int32],
    # Angle parameters
    angle_ka: UnsafePointer[Scalar[StorageType]],
    angle_theta0: UnsafePointer[Scalar[StorageType]],
    # Number of angles
    num_angles: Int
) -> Float32:
    """
    Compute total angle bending energy (no gradients).
    """
    var total_energy: Float32 = 0.0

    for a in range(num_angles):
        var i = Int(angle_i[a])
        var j = Int(angle_j[a])
        var k = Int(angle_k[a])

        # Compute angle
        var theta = compute_angle(
            Float32(x[i]), Float32(y[i]), Float32(z[i]),
            Float32(x[j]), Float32(y[j]), Float32(z[j]),
            Float32(x[k]), Float32(y[k]), Float32(z[k])
        )

        var energy = angle_energy_single(theta, Float32(angle_theta0[a]), Float32(angle_ka[a]))
        total_energy += energy

    return total_energy

# Stretch-bend coupling
# E_stbn = c5 * dtheta * (kba_ijk * dr1 + kba_kji * dr2)
# where c5 = MDYNE_A_TO_KCAL_MOL * DEG2RAD = 143.9325 * (π/180)
alias MMFF_STBN_PREFACTOR: Float32 = 2.51210  # 143.9325 * π / 180

fn stretch_bend_energy_single(
    dr1: Float32,       # Bond 1 deviation (r1 - r0_1)
    dr2: Float32,       # Bond 2 deviation (r2 - r0_2)
    dtheta_deg: Float32,  # Angle deviation in degrees
    kba_ijk: Float32,   # Stretch-bend force constant i-j
    kba_kji: Float32    # Stretch-bend force constant k-j
) -> Float32:
    """
    MMFF94 stretch-bend coupling energy.

    E = c5 * dtheta * (kba_ijk * dr1 + kba_kji * dr2)
    where c5 = MDYNE_A_TO_KCAL_MOL * DEG2RAD
    """
    return MMFF_STBN_PREFACTOR * dtheta_deg * (kba_ijk * dr1 + kba_kji * dr2)

fn compute_total_stretch_bend_energy[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Stretch-bend topology (same as angles)
    stbn_i: UnsafePointer[Int32],
    stbn_j: UnsafePointer[Int32],
    stbn_k: UnsafePointer[Int32],
    # Stretch-bend parameters
    stbn_kba_ijk: UnsafePointer[Scalar[StorageType]],
    stbn_kba_kji: UnsafePointer[Scalar[StorageType]],
    # Bond equilibrium lengths (for the two bonds in the angle)
    stbn_r0_ij: UnsafePointer[Scalar[StorageType]],
    stbn_r0_jk: UnsafePointer[Scalar[StorageType]],
    # Angle equilibrium value
    stbn_theta0: UnsafePointer[Scalar[StorageType]],
    # Number of stretch-bend terms
    num_stbn: Int
) -> Float32:
    """
    Compute total stretch-bend coupling energy.
    """
    var total_energy: Float32 = 0.0

    for s in range(num_stbn):
        var i = Int(stbn_i[s])
        var j = Int(stbn_j[s])  # Central atom
        var k = Int(stbn_k[s])

        # Compute current bond lengths
        var dx_ij = Float32(x[j]) - Float32(x[i])
        var dy_ij = Float32(y[j]) - Float32(y[i])
        var dz_ij = Float32(z[j]) - Float32(z[i])
        var r_ij = sqrt(dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij)

        var dx_jk = Float32(x[k]) - Float32(x[j])
        var dy_jk = Float32(y[k]) - Float32(y[j])
        var dz_jk = Float32(z[k]) - Float32(z[j])
        var r_jk = sqrt(dx_jk * dx_jk + dy_jk * dy_jk + dz_jk * dz_jk)

        # Compute current angle
        var theta = compute_angle(
            Float32(x[i]), Float32(y[i]), Float32(z[i]),
            Float32(x[j]), Float32(y[j]), Float32(z[j]),
            Float32(x[k]), Float32(y[k]), Float32(z[k])
        )
        var theta_deg = theta * RAD_TO_DEG

        # Compute deviations
        var dr1 = r_ij - Float32(stbn_r0_ij[s])
        var dr2 = r_jk - Float32(stbn_r0_jk[s])
        var dtheta_deg = theta_deg - Float32(stbn_theta0[s])

        # Compute stretch-bend energy
        var energy = stretch_bend_energy_single(
            dr1, dr2, dtheta_deg,
            Float32(stbn_kba_ijk[s]), Float32(stbn_kba_kji[s])
        )
        total_energy += energy

    return total_energy

# Out-of-plane bending (for sp2 centers)
# E_oop = 0.5 * c2 * koop * chi^2
# where c2 = MDYNE_A_TO_KCAL_MOL * DEG2RAD^2 = 0.043844
alias MMFF_OOP_PREFACTOR: Float32 = 0.021922  # 0.5 * 0.043844

fn oop_energy_single(
    chi_deg: Float32,  # Out-of-plane angle in degrees
    koop: Float32      # Force constant
) -> Float32:
    """
    MMFF94 out-of-plane bending energy.

    E = 0.5 * c2 * koop * chi^2
    where c2 = MDYNE_A_TO_KCAL_MOL * DEG2RAD^2
    """
    return MMFF_OOP_PREFACTOR * koop * chi_deg * chi_deg

fn compute_oop_chi(
    x_i: Float32, y_i: Float32, z_i: Float32,  # Atom i
    x_j: Float32, y_j: Float32, z_j: Float32,  # Atom j (central sp2)
    x_k: Float32, y_k: Float32, z_k: Float32,  # Atom k
    x_l: Float32, y_l: Float32, z_l: Float32   # Atom l (out-of-plane atom)
) -> Float32:
    """
    Compute out-of-plane angle chi in degrees.

    chi is the angle between the j-l vector and the plane defined by i-j-k.
    """
    # Vectors from central atom j
    var rJI_x = x_i - x_j
    var rJI_y = y_i - y_j
    var rJI_z = z_i - z_j

    var rJK_x = x_k - x_j
    var rJK_y = y_k - y_j
    var rJK_z = z_k - z_j

    var rJL_x = x_l - x_j
    var rJL_y = y_l - y_j
    var rJL_z = z_l - z_j

    # Normalize vectors
    var dJI = sqrt(rJI_x * rJI_x + rJI_y * rJI_y + rJI_z * rJI_z)
    var dJK = sqrt(rJK_x * rJK_x + rJK_y * rJK_y + rJK_z * rJK_z)
    var dJL = sqrt(rJL_x * rJL_x + rJL_y * rJL_y + rJL_z * rJL_z)

    if dJI < Float32(1e-8) or dJK < Float32(1e-8) or dJL < Float32(1e-8):
        return Float32(0.0)

    rJI_x /= dJI
    rJI_y /= dJI
    rJI_z /= dJI
    rJK_x /= dJK
    rJK_y /= dJK
    rJK_z /= dJK
    rJL_x /= dJL
    rJL_y /= dJL
    rJL_z /= dJL

    # Normal to the i-j-k plane (rJI x rJK)
    var n_x = rJI_y * rJK_z - rJI_z * rJK_y
    var n_y = rJI_z * rJK_x - rJI_x * rJK_z
    var n_z = rJI_x * rJK_y - rJI_y * rJK_x

    var n_len = sqrt(n_x * n_x + n_y * n_y + n_z * n_z)
    if n_len < Float32(1e-8):
        return Float32(0.0)

    n_x /= n_len
    n_y /= n_len
    n_z /= n_len

    # sin(chi) = rJL dot n
    var sin_chi = rJL_x * n_x + rJL_y * n_y + rJL_z * n_z

    # Clamp to [-1, 1]
    if sin_chi > Float32(1.0):
        sin_chi = Float32(1.0)
    elif sin_chi < Float32(-1.0):
        sin_chi = Float32(-1.0)

    # Convert to degrees using small angle approximation: asin(x) ≈ x for small x
    # For accuracy, we could use asin but that's in math module
    # chi_deg = asin(sin_chi) * RAD_TO_DEG
    # For small angles (typical in OOP), sin_chi ≈ chi in radians
    return sin_chi * RAD_TO_DEG  # Approximation valid for small angles

fn compute_total_oop_energy[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # OOP topology (i, j, k, l where j is central)
    oop_i: UnsafePointer[Int32],
    oop_j: UnsafePointer[Int32],
    oop_k: UnsafePointer[Int32],
    oop_l: UnsafePointer[Int32],
    # OOP parameters
    oop_koop: UnsafePointer[Scalar[StorageType]],
    # Number of OOP terms
    num_oop: Int
) -> Float32:
    """
    Compute total out-of-plane bending energy.
    """
    var total_energy: Float32 = 0.0

    for o in range(num_oop):
        var i = Int(oop_i[o])
        var j = Int(oop_j[o])  # Central atom
        var k = Int(oop_k[o])
        var l = Int(oop_l[o])  # Out-of-plane atom

        var chi_deg = compute_oop_chi(
            Float32(x[i]), Float32(y[i]), Float32(z[i]),
            Float32(x[j]), Float32(y[j]), Float32(z[j]),
            Float32(x[k]), Float32(y[k]), Float32(z[k]),
            Float32(x[l]), Float32(y[l]), Float32(z[l])
        )

        var energy = oop_energy_single(chi_deg, Float32(oop_koop[o]))
        total_energy += energy

    return total_energy

# Torsion
# E_torsion = 0.5 * (V1*(1+cos(phi)) + V2*(1-cos(2*phi)) + V3*(1+cos(3*phi)))
fn compute_torsion_angle(
    x1: Float32, y1: Float32, z1: Float32,  # Atom i
    x2: Float32, y2: Float32, z2: Float32,  # Atom j
    x3: Float32, y3: Float32, z3: Float32,  # Atom k
    x4: Float32, y4: Float32, z4: Float32   # Atom l
) -> Float32:
    """
    Compute torsion angle i-j-k-l in radians.

    Uses the standard IUPAC definition with atan2.
    """
    # Bond vectors
    var b1x = x2 - x1
    var b1y = y2 - y1
    var b1z = z2 - z1

    var b2x = x3 - x2
    var b2y = y3 - y2
    var b2z = z3 - z2

    var b3x = x4 - x3
    var b3y = y4 - y3
    var b3z = z4 - z3

    # Normal vectors to the planes
    # n1 = b1 x b2
    var n1x = b1y * b2z - b1z * b2y
    var n1y = b1z * b2x - b1x * b2z
    var n1z = b1x * b2y - b1y * b2x

    # n2 = b2 x b3
    var n2x = b2y * b3z - b2z * b3y
    var n2y = b2z * b3x - b2x * b3z
    var n2z = b2x * b3y - b2y * b3x

    # m1 = n1 x b2/|b2|
    var b2_len = sqrt(b2x * b2x + b2y * b2y + b2z * b2z)
    if b2_len < Float32(1e-8):
        return Float32(0.0)

    var b2_inv = Float32(1.0) / b2_len
    var m1x = n1y * b2z * b2_inv - n1z * b2y * b2_inv
    var m1y = n1z * b2x * b2_inv - n1x * b2z * b2_inv
    var m1z = n1x * b2y * b2_inv - n1y * b2x * b2_inv

    # x = n1 . n2
    var x_val = n1x * n2x + n1y * n2y + n1z * n2z
    # y = m1 . n2
    var y_val = m1x * n2x + m1y * n2y + m1z * n2z

    return atan2(y_val, x_val)

fn torsion_energy_single(
    phi: Float32,     # Current torsion angle in radians
    v1: Float32,      # Fourier coefficient for cos(phi)
    v2: Float32,      # Fourier coefficient for cos(2*phi)
    v3: Float32       # Fourier coefficient for cos(3*phi)
) -> Float32:
    """
    MMFF94 torsion energy for a single torsion.

    E = 0.5 * (V1*(1+cos(phi)) + V2*(1-cos(2*phi)) + V3*(1+cos(3*phi)))
    """
    var cos_phi = cos(phi)
    var cos_2phi = cos(Float32(2.0) * phi)
    var cos_3phi = cos(Float32(3.0) * phi)

    var energy = Float32(0.5) * (
        v1 * (Float32(1.0) + cos_phi) +
        v2 * (Float32(1.0) - cos_2phi) +
        v3 * (Float32(1.0) + cos_3phi)
    )
    return energy

fn torsion_energy_and_gradient(
    phi: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32
) -> Tuple[Float32, Float32]:
    """
    Compute torsion energy and its gradient dE/dphi.

    dE/dphi = 0.5 * (-V1*sin(phi) + 2*V2*sin(2*phi) - 3*V3*sin(3*phi))
    """
    var cos_phi = cos(phi)
    var cos_2phi = cos(Float32(2.0) * phi)
    var cos_3phi = cos(Float32(3.0) * phi)
    var sin_phi = sin(phi)
    var sin_2phi = sin(Float32(2.0) * phi)
    var sin_3phi = sin(Float32(3.0) * phi)

    var energy = Float32(0.5) * (
        v1 * (Float32(1.0) + cos_phi) +
        v2 * (Float32(1.0) - cos_2phi) +
        v3 * (Float32(1.0) + cos_3phi)
    )

    var gradient = Float32(0.5) * (
        -v1 * sin_phi +
        Float32(2.0) * v2 * sin_2phi -
        Float32(3.0) * v3 * sin_3phi
    )

    return (energy, gradient)

fn compute_torsion_forces[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Gradients (output)
    mut grad_x: UnsafePointer[Scalar[StorageType]],
    mut grad_y: UnsafePointer[Scalar[StorageType]],
    mut grad_z: UnsafePointer[Scalar[StorageType]],
    # Torsion topology
    torsion_i: UnsafePointer[Int32],
    torsion_j: UnsafePointer[Int32],
    torsion_k: UnsafePointer[Int32],
    torsion_l: UnsafePointer[Int32],
    # Torsion parameters
    torsion_v1: UnsafePointer[Scalar[StorageType]],
    torsion_v2: UnsafePointer[Scalar[StorageType]],
    torsion_v3: UnsafePointer[Scalar[StorageType]],
    # Range
    torsion_start: Int,
    torsion_end: Int
) -> Float32:
    """
    Compute torsion energy and accumulate forces.

    Uses the standard formulation for torsion derivatives.
    """
    var total_energy: Float32 = 0.0

    for t in range(torsion_start, torsion_end):
        var i = Int(torsion_i[t])
        var j = Int(torsion_j[t])
        var k = Int(torsion_k[t])
        var l = Int(torsion_l[t])

        # Get positions and convert to Float32
        var x1 = Float32(x[i])
        var y1 = Float32(y[i])
        var z1 = Float32(z[i])
        var x2 = Float32(x[j])
        var y2 = Float32(y[j])
        var z2 = Float32(z[j])
        var x3 = Float32(x[k])
        var y3 = Float32(y[k])
        var z3 = Float32(z[k])
        var x4 = Float32(x[l])
        var y4 = Float32(y[l])
        var z4 = Float32(z[l])

        # Bond vectors
        var b1x = x2 - x1
        var b1y = y2 - y1
        var b1z = z2 - z1

        var b2x = x3 - x2
        var b2y = y3 - y2
        var b2z = z3 - z2

        var b3x = x4 - x3
        var b3y = y4 - y3
        var b3z = z4 - z3

        # Cross products for plane normals
        var n1x = b1y * b2z - b1z * b2y
        var n1y = b1z * b2x - b1x * b2z
        var n1z = b1x * b2y - b1y * b2x

        var n2x = b2y * b3z - b2z * b3y
        var n2y = b2z * b3x - b2x * b3z
        var n2z = b2x * b3y - b2y * b3x

        # Magnitudes
        var n1_sq = n1x * n1x + n1y * n1y + n1z * n1z
        var n2_sq = n2x * n2x + n2y * n2y + n2z * n2z
        var b2_sq = b2x * b2x + b2y * b2y + b2z * b2z

        if n1_sq < Float32(1e-10) or n2_sq < Float32(1e-10) or b2_sq < Float32(1e-10):
            continue

        var b2_len = sqrt(b2_sq)

        # Compute torsion angle
        var m1x = n1y * b2z / b2_len - n1z * b2y / b2_len
        var m1y = n1z * b2x / b2_len - n1x * b2z / b2_len
        var m1z = n1x * b2y / b2_len - n1y * b2x / b2_len

        var x_val = n1x * n2x + n1y * n2y + n1z * n2z
        var y_val = m1x * n2x + m1y * n2y + m1z * n2z
        var phi = atan2(y_val, x_val)

        # Compute energy and gradient
        var result = torsion_energy_and_gradient(
            phi, Float32(torsion_v1[t]), Float32(torsion_v2[t]), Float32(torsion_v3[t])
        )
        var energy = result[0]
        var dE_dphi = result[1]

        total_energy += energy

        # Compute forces using standard torsion gradient formulas
        var factor_i = dE_dphi * b2_len / n1_sq
        var factor_l = -dE_dphi * b2_len / n2_sq

        # Force on atom i
        var fix = factor_i * n1x
        var fiy = factor_i * n1y
        var fiz = factor_i * n1z

        # Force on atom l
        var flx = factor_l * n2x
        var fly = factor_l * n2y
        var flz = factor_l * n2z

        # Projections for j and k
        var b1_dot_b2 = b1x * b2x + b1y * b2y + b1z * b2z
        var b3_dot_b2 = b3x * b2x + b3y * b2y + b3z * b2z

        var factor_j_from_i = -b1_dot_b2 / b2_sq - Float32(1.0)
        var factor_j_from_l = b3_dot_b2 / b2_sq

        var fjx = factor_j_from_i * fix + factor_j_from_l * flx
        var fjy = factor_j_from_i * fiy + factor_j_from_l * fly
        var fjz = factor_j_from_i * fiz + factor_j_from_l * flz

        var factor_k_from_i = b1_dot_b2 / b2_sq
        var factor_k_from_l = -b3_dot_b2 / b2_sq - Float32(1.0)

        var fkx = factor_k_from_i * fix + factor_k_from_l * flx
        var fky = factor_k_from_i * fiy + factor_k_from_l * fly
        var fkz = factor_k_from_i * fiz + factor_k_from_l * flz

        # Accumulate gradients using store()
        grad_x[i] = grad_x[i] + Scalar[StorageType](fix)
        grad_y[i] = grad_y[i] + Scalar[StorageType](fiy)
        grad_z[i] = grad_z[i] + Scalar[StorageType](fiz)
        grad_x[j] = grad_x[j] + Scalar[StorageType](fjx)
        grad_y[j] = grad_y[j] + Scalar[StorageType](fjy)
        grad_z[j] = grad_z[j] + Scalar[StorageType](fjz)
        grad_x[k] = grad_x[k] + Scalar[StorageType](fkx)
        grad_y[k] = grad_y[k] + Scalar[StorageType](fky)
        grad_z[k] = grad_z[k] + Scalar[StorageType](fkz)
        grad_x[l] = grad_x[l] + Scalar[StorageType](flx)
        grad_y[l] = grad_y[l] + Scalar[StorageType](fly)
        grad_z[l] = grad_z[l] + Scalar[StorageType](flz)

    return total_energy

fn compute_total_torsion_energy[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Torsion topology
    torsion_i: UnsafePointer[Int32],
    torsion_j: UnsafePointer[Int32],
    torsion_k: UnsafePointer[Int32],
    torsion_l: UnsafePointer[Int32],
    # Torsion parameters
    torsion_v1: UnsafePointer[Scalar[StorageType]],
    torsion_v2: UnsafePointer[Scalar[StorageType]],
    torsion_v3: UnsafePointer[Scalar[StorageType]],
    # Number of torsions
    num_torsions: Int
) -> Float32:
    """
    Compute total torsion energy (no gradients).
    """
    var total_energy: Float32 = 0.0

    for t in range(num_torsions):
        var i = Int(torsion_i[t])
        var j = Int(torsion_j[t])
        var k = Int(torsion_k[t])
        var l = Int(torsion_l[t])

        var phi = compute_torsion_angle(
            Float32(x[i]), Float32(y[i]), Float32(z[i]),
            Float32(x[j]), Float32(y[j]), Float32(z[j]),
            Float32(x[k]), Float32(y[k]), Float32(z[k]),
            Float32(x[l]), Float32(y[l]), Float32(z[l])
        )

        var energy = torsion_energy_single(
            phi, Float32(torsion_v1[t]), Float32(torsion_v2[t]), Float32(torsion_v3[t])
        )
        total_energy += energy

    return total_energy

# Van der Waals
# Buffered 14-7 potential: E_vdw = epsilon * (1.07*R*/(r+0.07*R*))^7 * (1.12*R*^7/(r^7+0.12*R*^7) - 2)
# Helper for x^7 = x * x^2 * x^4 (efficient for SIMD)
fn pow7[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Compute x^7 efficiently using x * x^2 * x^4."""
    var x2 = x * x
    var x4 = x2 * x2
    return x * x2 * x4


fn vdw_energy[width: Int = 1](
    r: SIMD[DType.float32, width],
    epsilon: SIMD[DType.float32, width],
    r_star: SIMD[DType.float32, width]
) -> SIMD[DType.float32, width]:
    """
    MMFF94 buffered 14-7 van der Waals energy.

    E = epsilon * (1.07*R*/(r+0.07*R*))^7 * (1.12*R*^7/(r^7+0.12*R*^7) - 2)

    When width=1, operates on scalars.
    When width=16, processes 16 pairs in parallel (AVX-512).
    """
    var delta = SIMD[DType.float32, width](0.07)
    var gamma = SIMD[DType.float32, width](0.12)
    var one = SIMD[DType.float32, width](1.0)
    var two = SIMD[DType.float32, width](2.0)

    var r_star_7 = pow7[width](r_star)

    # Repulsive term: (1.07*R*/(r+0.07*R*))^7
    var denom_rep = r + delta * r_star
    var ratio_rep = (one + delta) * r_star / denom_rep
    var rep_term = pow7[width](ratio_rep)

    # Attractive term: 1.12*R*^7/(r^7+0.12*R*^7) - 2
    var r_7 = pow7[width](r)
    var denom_att = r_7 + gamma * r_star_7
    var att_term = (one + gamma) * r_star_7 / denom_att - two

    return epsilon * rep_term * att_term


fn vdw_energy_single(r: Float32, epsilon: Float32, r_star: Float32) -> Float32:
    """Scalar vdW energy (calls SIMD version with width=1)."""
    return vdw_energy[1](r, epsilon, r_star)[0]

fn vdw_energy_and_gradient(
    r: Float32,
    epsilon: Float32,
    r_star: Float32
) -> Tuple[Float32, Float32]:
    """
    Compute vdW energy and its gradient dE/dr.
    """
    alias DELTA: Float32 = 0.07
    alias GAMMA: Float32 = 0.12

    var r_star_7 = r_star ** 7
    var r_6 = r ** 6
    var r_7 = r * r_6

    # Repulsive components
    var denom_rep = r + DELTA * r_star
    var ratio_rep = (Float32(1.0) + DELTA) * r_star / denom_rep
    var rep_term = ratio_rep ** 7

    # Attractive components
    var denom_att = r_7 + GAMMA * r_star_7
    var att_ratio = (Float32(1.0) + GAMMA) * r_star_7 / denom_att
    var att_term = att_ratio - Float32(2.0)

    var energy = epsilon * rep_term * att_term

    # Derivatives
    # d(rep_term)/dr = -7 * rep_term / denom_rep
    var d_rep = -Float32(7.0) * rep_term / denom_rep

    # d(att_ratio)/dr = -7 * att_ratio * r^6 / denom_att
    var d_att = -Float32(7.0) * att_ratio * r_6 / denom_att

    # Chain rule
    var gradient = epsilon * (d_rep * att_term + rep_term * d_att)

    return (energy, gradient)

fn compute_vdw_forces[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Gradients (output)
    mut grad_x: UnsafePointer[Scalar[StorageType]],
    mut grad_y: UnsafePointer[Scalar[StorageType]],
    mut grad_z: UnsafePointer[Scalar[StorageType]],
    # Nonbonded pair indices
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    # vdW parameters
    nb_epsilon: UnsafePointer[Scalar[StorageType]],
    nb_r_star: UnsafePointer[Scalar[StorageType]],
    # Range
    nb_start: Int,
    nb_end: Int
) -> Float32:
    """
    Compute vdW energy and accumulate forces.

    NOTE: MMFF94 does NOT apply 1-4 scaling to vdW energy.
    1-4 scaling (0.75) only applies to electrostatic energy.
    """
    var total_energy: Float32 = 0.0

    for p in range(nb_start, nb_end):
        var i = Int(nb_i[p])
        var j = Int(nb_j[p])

        # Get positions and convert to Float32
        var x1 = Float32(x[i])
        var y1 = Float32(y[i])
        var z1 = Float32(z[i])
        var x2 = Float32(x[j])
        var y2 = Float32(y[j])
        var z2 = Float32(z[j])

        # Compute distance
        var dx = x2 - x1
        var dy = y2 - y1
        var dz = z2 - z1
        var r = sqrt(dx * dx + dy * dy + dz * dz)

        if r < Float32(0.1):
            continue

        # Compute energy and gradient (no 1-4 scaling for vdW)
        var result = vdw_energy_and_gradient(r, Float32(nb_epsilon[p]), Float32(nb_r_star[p]))
        var energy = result[0]
        var dE_dr = result[1]

        total_energy += energy

        # Convert to Cartesian forces
        var inv_r = Float32(1.0) / r
        var fx = dE_dr * dx * inv_r
        var fy = dE_dr * dy * inv_r
        var fz = dE_dr * dz * inv_r

        # Accumulate gradients using store()
        grad_x[i] = grad_x[i] - Scalar[StorageType](fx)
        grad_y[i] = grad_y[i] - Scalar[StorageType](fy)
        grad_z[i] = grad_z[i] - Scalar[StorageType](fz)
        grad_x[j] = grad_x[j] + Scalar[StorageType](fx)
        grad_y[j] = grad_y[j] + Scalar[StorageType](fy)
        grad_z[j] = grad_z[j] + Scalar[StorageType](fz)

    return total_energy

fn compute_total_vdw_energy[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Nonbonded pair indices
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    # vdW parameters
    nb_epsilon: UnsafePointer[Scalar[StorageType]],
    nb_r_star: UnsafePointer[Scalar[StorageType]],
    # Number of pairs
    num_pairs: Int
) -> Float32:
    """
    Compute total vdW energy (no gradients).

    NOTE: MMFF94 does NOT apply 1-4 scaling to vdW energy.
    1-4 scaling (0.75) only applies to electrostatic energy.
    """
    var total_energy: Float32 = 0.0

    for p in range(num_pairs):
        var i = Int(nb_i[p])
        var j = Int(nb_j[p])

        var dx = Float32(x[j]) - Float32(x[i])
        var dy = Float32(y[j]) - Float32(y[i])
        var dz = Float32(z[j]) - Float32(z[i])
        var r = sqrt(dx * dx + dy * dy + dz * dz)

        if r < Float32(0.1):
            continue

        var energy = vdw_energy_single(r, Float32(nb_epsilon[p]), Float32(nb_r_star[p]))
        total_energy += energy

    return total_energy

# Electrostatics
# E_elec = 332.0716 * qi * qj / (D * (r + delta))
fn electrostatic_energy_single(
    r: Float32,       # Distance
    qi: Float32,      # Charge on atom i
    qj: Float32,      # Charge on atom j
    dielectric: Float32  # Dielectric constant (1.0 for vacuum, r for distance-dependent)
) -> Float32:
    """
    MMFF94 electrostatic energy.

    E = 332.0716 * qi * qj / (D * (r + 0.05))

    The buffer of 0.05 A prevents singularities at short distances.
    """
    var r_eff = r + MMFF_ELEC_BUFFER
    return MMFF_ELEC_CONSTANT * qi * qj / (dielectric * r_eff)

fn electrostatic_energy_and_gradient(
    r: Float32,
    qi: Float32,
    qj: Float32,
    dielectric: Float32,
    use_distance_dielectric: Bool
) -> Tuple[Float32, Float32]:
    """
    Compute electrostatic energy and its gradient dE/dr.

    If use_distance_dielectric=True, D=r and:
        E = 332.0716 * qi * qj / (r * (r + 0.05))
        dE/dr = -332.0716 * qi * qj * (2r + 0.05) / (r^2 * (r + 0.05)^2)

    If use_distance_dielectric=False, D=dielectric (constant) and:
        E = 332.0716 * qi * qj / (D * (r + 0.05))
        dE/dr = -332.0716 * qi * qj / (D * (r + 0.05)^2)
    """
    var r_eff = r + MMFF_ELEC_BUFFER
    var qq = qi * qj

    if use_distance_dielectric:
        # Distance-dependent dielectric: D = r
        var energy = MMFF_ELEC_CONSTANT * qq / (r * r_eff)

        # Gradient: d/dr [1/(r*(r+d))] = -(2r+d)/(r^2*(r+d)^2)
        var gradient = -MMFF_ELEC_CONSTANT * qq * (Float32(2.0) * r + MMFF_ELEC_BUFFER) / (
            r * r * r_eff * r_eff
        )
        return (energy, gradient)
    else:
        # Constant dielectric
        var energy = MMFF_ELEC_CONSTANT * qq / (dielectric * r_eff)

        # Gradient: d/dr [1/(r+d)] = -1/(r+d)^2
        var gradient = -MMFF_ELEC_CONSTANT * qq / (dielectric * r_eff * r_eff)
        return (energy, gradient)

fn compute_electrostatic_forces[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Gradients (output)
    mut grad_x: UnsafePointer[Scalar[StorageType]],
    mut grad_y: UnsafePointer[Scalar[StorageType]],
    mut grad_z: UnsafePointer[Scalar[StorageType]],
    # Charges
    charge: UnsafePointer[Scalar[StorageType]],
    # Nonbonded pair indices
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_is_1_4: UnsafePointer[Int32],
    # Range
    nb_start: Int,
    nb_end: Int,
    # Options
    dielectric: Float32,
    use_distance_dielectric: Bool,
    scale_1_4: Float32
) -> Float32:
    """
    Compute electrostatic energy and accumulate forces.

    1-4 interactions are scaled by scale_1_4 (typically 0.75).
    """
    var total_energy: Float32 = 0.0

    for p in range(nb_start, nb_end):
        var i = Int(nb_i[p])
        var j = Int(nb_j[p])

        var qi = Float32(charge[i])
        var qj = Float32(charge[j])

        # Skip if either charge is zero
        if qi == Float32(0.0) or qj == Float32(0.0):
            continue

        # Get positions and convert to Float32
        var x1 = Float32(x[i])
        var y1 = Float32(y[i])
        var z1 = Float32(z[i])
        var x2 = Float32(x[j])
        var y2 = Float32(y[j])
        var z2 = Float32(z[j])

        # Compute distance
        var dx = x2 - x1
        var dy = y2 - y1
        var dz = z2 - z1
        var r = sqrt(dx * dx + dy * dy + dz * dz)

        if r < Float32(0.1):
            continue

        # Apply 1-4 scaling
        var scale = Float32(1.0)
        if nb_is_1_4[p] != 0:
            scale = scale_1_4

        # Compute energy and gradient
        var result = electrostatic_energy_and_gradient(
            r, qi, qj, dielectric, use_distance_dielectric
        )
        var energy = result[0] * scale
        var dE_dr = result[1] * scale

        total_energy += energy

        # Convert to Cartesian forces
        var inv_r = Float32(1.0) / r
        var fx = dE_dr * dx * inv_r
        var fy = dE_dr * dy * inv_r
        var fz = dE_dr * dz * inv_r

        # Accumulate gradients using store()
        grad_x[i] = grad_x[i] - Scalar[StorageType](fx)
        grad_y[i] = grad_y[i] - Scalar[StorageType](fy)
        grad_z[i] = grad_z[i] - Scalar[StorageType](fz)
        grad_x[j] = grad_x[j] + Scalar[StorageType](fx)
        grad_y[j] = grad_y[j] + Scalar[StorageType](fy)
        grad_z[j] = grad_z[j] + Scalar[StorageType](fz)

    return total_energy

fn compute_total_electrostatic_energy[
    StorageType: DType
](
    # Positions
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    # Charges
    charge: UnsafePointer[Scalar[StorageType]],
    # Nonbonded pair indices
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_is_1_4: UnsafePointer[Int32],
    # Number of pairs
    num_pairs: Int,
    # Options
    dielectric: Float32,
    use_distance_dielectric: Bool,
    scale_1_4: Float32
) -> Float32:
    """
    Compute total electrostatic energy (no gradients).
    """
    var total_energy: Float32 = 0.0

    for p in range(num_pairs):
        var i = Int(nb_i[p])
        var j = Int(nb_j[p])

        var qi = Float32(charge[i])
        var qj = Float32(charge[j])

        if qi == Float32(0.0) or qj == Float32(0.0):
            continue

        var dx = Float32(x[j]) - Float32(x[i])
        var dy = Float32(y[j]) - Float32(y[i])
        var dz = Float32(z[j]) - Float32(z[i])
        var r = sqrt(dx * dx + dy * dy + dz * dz)

        if r < Float32(0.1):
            continue

        var scale = Float32(1.0)
        if nb_is_1_4[p] != 0:
            scale = scale_1_4

        var energy: Float32 = 0.0
        if use_distance_dielectric:
            energy = MMFF_ELEC_CONSTANT * qi * qj / (r * (r + MMFF_ELEC_BUFFER))
        else:
            energy = MMFF_ELEC_CONSTANT * qi * qj / (dielectric * (r + MMFF_ELEC_BUFFER))

        total_energy += energy * scale

    return total_energy

# Total energy and options
struct MMFF94Options(Copyable, Movable):
    """Configuration options for MMFF94 calculations."""
    var dielectric: Float32
    var use_distance_dielectric: Bool
    var vdw_1_4_scale: Float32
    var elec_1_4_scale: Float32
    var compute_bonds: Bool
    var compute_angles: Bool
    var compute_torsions: Bool
    var compute_vdw: Bool
    var compute_elec: Bool

    fn __init__(out self):
        """Default MMFF94 options."""
        self.dielectric = DEFAULT_DIELECTRIC
        self.use_distance_dielectric = True
        self.vdw_1_4_scale = DEFAULT_1_4_VDW_SCALE
        self.elec_1_4_scale = DEFAULT_1_4_ELEC_SCALE
        self.compute_bonds = True
        self.compute_angles = True
        self.compute_torsions = True
        self.compute_vdw = True
        self.compute_elec = True

    fn __copyinit__(out self, other: Self):
        self.dielectric = other.dielectric
        self.use_distance_dielectric = other.use_distance_dielectric
        self.vdw_1_4_scale = other.vdw_1_4_scale
        self.elec_1_4_scale = other.elec_1_4_scale
        self.compute_bonds = other.compute_bonds
        self.compute_angles = other.compute_angles
        self.compute_torsions = other.compute_torsions
        self.compute_vdw = other.compute_vdw
        self.compute_elec = other.compute_elec

    fn __moveinit__(out self, deinit other: Self):
        self.dielectric = other.dielectric
        self.use_distance_dielectric = other.use_distance_dielectric
        self.vdw_1_4_scale = other.vdw_1_4_scale
        self.elec_1_4_scale = other.elec_1_4_scale
        self.compute_bonds = other.compute_bonds
        self.compute_angles = other.compute_angles
        self.compute_torsions = other.compute_torsions
        self.compute_vdw = other.compute_vdw
        self.compute_elec = other.compute_elec

struct EnergyComponents(Copyable, Movable):
    """Breakdown of energy components."""
    var bond: Float32
    var angle: Float32
    var torsion: Float32
    var vdw: Float32
    var elec: Float32
    var total: Float32

    fn __init__(out self):
        self.bond = 0.0
        self.angle = 0.0
        self.torsion = 0.0
        self.vdw = 0.0
        self.elec = 0.0
        self.total = 0.0

    fn __copyinit__(out self, other: Self):
        self.bond = other.bond
        self.angle = other.angle
        self.torsion = other.torsion
        self.vdw = other.vdw
        self.elec = other.elec
        self.total = other.total

    fn __moveinit__(out self, deinit other: Self):
        self.bond = other.bond
        self.angle = other.angle
        self.torsion = other.torsion
        self.vdw = other.vdw
        self.elec = other.elec
        self.total = other.total

    fn sum(mut self):
        """Compute total from components."""
        self.total = self.bond + self.angle + self.torsion + self.vdw + self.elec

fn compute_mmff94_energy[
    StorageType: DType
](
    # Atom data
    x: UnsafePointer[Scalar[StorageType]],
    y: UnsafePointer[Scalar[StorageType]],
    z: UnsafePointer[Scalar[StorageType]],
    charge: UnsafePointer[Scalar[StorageType]],
    num_atoms: Int,
    # Bond data
    bond_i: UnsafePointer[Int32],
    bond_j: UnsafePointer[Int32],
    bond_kb: UnsafePointer[Scalar[StorageType]],
    bond_r0: UnsafePointer[Scalar[StorageType]],
    num_bonds: Int,
    # Angle data
    angle_i: UnsafePointer[Int32],
    angle_j: UnsafePointer[Int32],
    angle_k: UnsafePointer[Int32],
    angle_ka: UnsafePointer[Scalar[StorageType]],
    angle_theta0: UnsafePointer[Scalar[StorageType]],
    num_angles: Int,
    # Torsion data
    torsion_i: UnsafePointer[Int32],
    torsion_j: UnsafePointer[Int32],
    torsion_k: UnsafePointer[Int32],
    torsion_l: UnsafePointer[Int32],
    torsion_v1: UnsafePointer[Scalar[StorageType]],
    torsion_v2: UnsafePointer[Scalar[StorageType]],
    torsion_v3: UnsafePointer[Scalar[StorageType]],
    num_torsions: Int,
    # Nonbonded data
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    nb_is_1_4: UnsafePointer[Int32],
    nb_epsilon: UnsafePointer[Scalar[StorageType]],
    nb_r_star: UnsafePointer[Scalar[StorageType]],
    num_nb_pairs: Int,
    # Options
    opts: MMFF94Options
) -> EnergyComponents:
    """
    Compute total MMFF94 energy with component breakdown.
    """
    var result = EnergyComponents()

    if opts.compute_bonds:
        result.bond = compute_total_bond_energy(
            x, y, z, bond_i, bond_j, bond_kb, bond_r0, num_bonds
        )

    if opts.compute_angles:
        result.angle = compute_total_angle_energy(
            x, y, z, angle_i, angle_j, angle_k, angle_ka, angle_theta0, num_angles
        )

    if opts.compute_torsions:
        result.torsion = compute_total_torsion_energy(
            x, y, z, torsion_i, torsion_j, torsion_k, torsion_l,
            torsion_v1, torsion_v2, torsion_v3, num_torsions
        )

    if opts.compute_vdw:
        result.vdw = compute_total_vdw_energy(
            x, y, z, nb_i, nb_j, nb_epsilon, nb_r_star, num_nb_pairs
        )

    if opts.compute_elec:
        result.elec = compute_total_electrostatic_energy(
            x, y, z, charge, nb_i, nb_j, nb_is_1_4, num_nb_pairs,
            opts.dielectric, opts.use_distance_dielectric, opts.elec_1_4_scale
        )

    result.sum()
    return result^

fn gradient_rms[
    StorageType: DType
](
    mut grad_x: UnsafePointer[Scalar[StorageType]],
    mut grad_y: UnsafePointer[Scalar[StorageType]],
    mut grad_z: UnsafePointer[Scalar[StorageType]],
    num_atoms: Int
) -> Float32:
    """Compute RMS gradient for convergence checking."""
    var sum_sq: Float32 = 0.0
    for i in range(num_atoms):
        var gx = Float32(grad_x[i])
        var gy = Float32(grad_y[i])
        var gz = Float32(grad_z[i])
        sum_sq += gx * gx
        sum_sq += gy * gy
        sum_sq += gz * gz

    return sqrt(sum_sq / Float32(3 * num_atoms))

fn gradient_max[
    StorageType: DType
](
    mut grad_x: UnsafePointer[Scalar[StorageType]],
    mut grad_y: UnsafePointer[Scalar[StorageType]],
    mut grad_z: UnsafePointer[Scalar[StorageType]],
    num_atoms: Int
) -> Float32:
    """Find maximum gradient component magnitude."""
    var max_grad: Float32 = 0.0
    for i in range(num_atoms):
        var gx = abs_f32(Float32(grad_x[i]))
        var gy = abs_f32(Float32(grad_y[i]))
        var gz = abs_f32(Float32(grad_z[i]))
        if gx > max_grad:
            max_grad = gx
        if gy > max_grad:
            max_grad = gy
        if gz > max_grad:
            max_grad = gz

    return max_grad
