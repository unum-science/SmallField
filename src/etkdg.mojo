# SmallField: ETKDG Distance Geometry
# Merged module: bounds, triangle smoothing, MDS embedding, and coordinate refinement
# Updated for Mojo 0.25.7

from math import sqrt, cos
from memory import UnsafePointer

# Bond length estimates by bond order
alias BOND_LENGTH_SINGLE: Float32 = 1.5     # A (typical C-C)
alias BOND_LENGTH_DOUBLE: Float32 = 1.34    # A (typical C=C)
alias BOND_LENGTH_TRIPLE: Float32 = 1.20    # A (typical C#C)
alias BOND_LENGTH_AROMATIC: Float32 = 1.40  # A (typical aromatic)

# Van der Waals radii (Angstroms)
alias VDW_RADIUS_H: Float32 = 1.20
alias VDW_RADIUS_C: Float32 = 1.70
alias VDW_RADIUS_N: Float32 = 1.55
alias VDW_RADIUS_O: Float32 = 1.52
alias VDW_RADIUS_S: Float32 = 1.80
alias VDW_RADIUS_DEFAULT: Float32 = 1.70

# Random number generator constants
alias LCG_A: Int = 1664525
alias LCG_C: Int = 1013904223
alias LCG_M: Int = 2147483647


fn get_pair_index(i: Int, j: Int, n: Int) -> Int:
    """
    Get linear index for pair (i, j) in upper triangular storage.

    For symmetric matrix with n atoms, index into n*(n-1)/2 element array.
    Works for both i < j and i > j.
    """
    var a = i
    var b = j
    if a > b:
        a = j
        b = i
    return a * n - a * (a + 1) // 2 + b - a - 1


struct DistanceBounds(Copyable, Movable):
    """
    Symmetric matrix storing lower and upper distance bounds.

    Uses flat storage with row-major upper triangle for efficiency.
    Access pattern: bounds[i,j] where i < j.
    """
    var lower: UnsafePointer[Float32]
    var upper: UnsafePointer[Float32]
    var num_atoms: Int
    var matrix_size: Int  # n*(n-1)/2 elements

    fn __init__(out self, num_atoms: Int):
        """Allocate bounds matrices."""
        self.num_atoms = num_atoms
        self.matrix_size = num_atoms * (num_atoms - 1) // 2

        self.lower = UnsafePointer[Float32].alloc(self.matrix_size)
        self.upper = UnsafePointer[Float32].alloc(self.matrix_size)

        # Initialize to infinity bounds
        for i in range(self.matrix_size):
            self.lower[i] = Float32(0.0)
            self.upper[i] = Float32(1000.0)  # Large upper bound

    fn __copyinit__(out self, other: Self):
        self.num_atoms = other.num_atoms
        self.matrix_size = other.matrix_size

        self.lower = UnsafePointer[Float32].alloc(self.matrix_size)
        self.upper = UnsafePointer[Float32].alloc(self.matrix_size)

        for i in range(self.matrix_size):
            self.lower[i] = other.lower[i]
            self.upper[i] = other.upper[i]

    fn __moveinit__(out self, deinit other: Self):
        self.num_atoms = other.num_atoms
        self.matrix_size = other.matrix_size
        self.lower = other.lower
        self.upper = other.upper

    fn free(self):
        """Release allocated memory."""
        self.lower.free()
        self.upper.free()

    fn index(self, i: Int, j: Int) -> Int:
        """Get linear index for pair (i, j) where i < j."""
        return get_pair_index(i, j, self.num_atoms)

    fn get_lower(self, i: Int, j: Int) -> Float32:
        """Get lower bound for pair (i, j)."""
        if i == j:
            return Float32(0.0)
        return self.lower[self.index(i, j)]

    fn get_upper(self, i: Int, j: Int) -> Float32:
        """Get upper bound for pair (i, j)."""
        if i == j:
            return Float32(0.0)
        return self.upper[self.index(i, j)]

    fn set_lower(self, i: Int, j: Int, value: Float32):
        """Set lower bound for pair (i, j)."""
        if i != j:
            self.lower[self.index(i, j)] = value

    fn set_upper(self, i: Int, j: Int, value: Float32):
        """Set upper bound for pair (i, j)."""
        if i != j:
            self.upper[self.index(i, j)] = value

    fn set_bounds(self, i: Int, j: Int, lower_val: Float32, upper_val: Float32):
        """Set both bounds for pair (i, j)."""
        if i != j:
            var idx = self.index(i, j)
            self.lower[idx] = lower_val
            self.upper[idx] = upper_val


fn initialize_bounds_from_topology(
    mut bounds: DistanceBounds,
    # Bond topology
    bond_i: UnsafePointer[Int32],
    bond_j: UnsafePointer[Int32],
    bond_length: UnsafePointer[Float32],
    num_bonds: Int,
    # Angle topology
    angle_i: UnsafePointer[Int32],
    angle_j: UnsafePointer[Int32],
    angle_k: UnsafePointer[Int32],
    angle_theta0: UnsafePointer[Float32],
    num_angles: Int,
    # VdW radii for all atoms
    vdw_radii: UnsafePointer[Float32],
    num_atoms: Int
):
    """
    Initialize distance bounds from molecular topology.

    Sets:
    - 1,2 bounds (bonded): r0 +/- 0.1 A
    - 1,3 bounds (angles): from law of cosines
    - 1,4+ bounds (non-bonded): sum of vdW radii as lower bound
    """
    # Set 1,2 bounds from bond lengths
    for bond_idx in range(num_bonds):
        var i = Int(bond_i[bond_idx])
        var j = Int(bond_j[bond_idx])
        var r0 = bond_length[bond_idx]

        # Tight bounds around equilibrium
        bounds.set_bounds(i, j, r0 - Float32(0.1), r0 + Float32(0.1))

    # Set 1,3 bounds from angles using law of cosines
    for angle_idx in range(num_angles):
        var i = Int(angle_i[angle_idx])
        var j = Int(angle_j[angle_idx])  # Central atom
        var k = Int(angle_k[angle_idx])
        var theta0 = angle_theta0[angle_idx]

        # Need bond lengths i-j and j-k
        var r_ij: Float32 = BOND_LENGTH_SINGLE
        var r_jk: Float32 = BOND_LENGTH_SINGLE

        # Find actual bond lengths from bond list
        for bond_idx in range(num_bonds):
            var bi = Int(bond_i[bond_idx])
            var bj = Int(bond_j[bond_idx])
            if (bi == i and bj == j) or (bi == j and bj == i):
                r_ij = bond_length[bond_idx]
            if (bi == j and bj == k) or (bi == k and bj == j):
                r_jk = bond_length[bond_idx]

        # Law of cosines: d_ik^2 = r_ij^2 + r_jk^2 - 2*r_ij*r_jk*cos(theta)
        var cos_theta = cos(theta0)
        var d_ik_sq = r_ij * r_ij + r_jk * r_jk - Float32(2.0) * r_ij * r_jk * cos_theta
        var d_ik = sqrt(d_ik_sq) if d_ik_sq > Float32(0.0) else Float32(0.1)

        # Allow some flexibility
        bounds.set_bounds(i, k, d_ik - Float32(0.2), d_ik + Float32(0.2))

    # Set non-bonded bounds (vdW contact)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            var current_lower = bounds.get_lower(i, j)

            # If not already set by bonded terms, use vdW contact
            if current_lower < Float32(0.1):
                var vdw_sum = vdw_radii[i] + vdw_radii[j]
                # Lower bound: vdW contact with some tolerance
                bounds.set_lower(i, j, vdw_sum * Float32(0.8))


fn sample_distance_matrix(
    bounds: DistanceBounds,
    mut distances: UnsafePointer[Float32],
    seed: Int
):
    """
    Sample a distance matrix within the bounds.

    Uses simple uniform sampling between lower and upper bounds.
    """
    var state = seed

    for i in range(bounds.num_atoms):
        for j in range(i + 1, bounds.num_atoms):
            var idx = bounds.index(i, j)
            var lower = bounds.lower[idx]
            var upper = bounds.upper[idx]

            # Generate random number in [0, 1)
            state = (LCG_A * state + LCG_C) & LCG_M
            var rand = Float32(state) / Float32(LCG_M)

            # Sample uniformly between bounds
            var dist = lower + rand * (upper - lower)
            distances[idx] = dist


fn smooth_bounds_triangle(
    lower: UnsafePointer[Float32],
    upper: UnsafePointer[Float32],
    num_atoms: Int
):
    """
    Smooth distance bounds using triangle inequality.

    Uses a Floyd-Warshall-like algorithm to propagate constraints.

    For all triples (i, j, k):
    - upper[i,k] <= upper[i,j] + upper[j,k]
    - lower[i,k] >= |lower[i,j] - upper[j,k]|
    """
    var n = num_atoms
    var changed = True
    var max_iterations = n * 2  # Limit iterations

    var iteration = 0
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        for k in range(n):
            for i in range(n):
                if i == k:
                    continue

                for j in range(i + 1, n):
                    if j == k:
                        continue

                    var idx_ij = get_pair_index(i, j, n)
                    var idx_ik = get_pair_index(i, k, n)
                    var idx_jk = get_pair_index(j, k, n)

                    # Update upper bound: d(i,j) <= d(i,k) + d(k,j)
                    var new_upper = upper[idx_ik] + upper[idx_jk]
                    if new_upper < upper[idx_ij]:
                        upper[idx_ij] = new_upper
                        changed = True

                    # Update lower bound: d(i,j) >= |d(i,k) - d(k,j)|
                    var diff1 = lower[idx_ik] - upper[idx_jk]
                    var diff2 = lower[idx_jk] - upper[idx_ik]
                    var new_lower = diff1 if diff1 > diff2 else diff2
                    if new_lower < Float32(0.0):
                        new_lower = Float32(0.0)

                    if new_lower > lower[idx_ij]:
                        lower[idx_ij] = new_lower
                        changed = True


fn check_bounds_validity(
    lower: UnsafePointer[Float32],
    upper: UnsafePointer[Float32],
    num_atoms: Int
) -> Bool:
    """
    Check if bounds are valid (lower <= upper for all pairs).
    """
    var n = num_atoms
    var matrix_size = n * (n - 1) // 2

    for idx in range(matrix_size):
        if lower[idx] > upper[idx] + Float32(0.001):  # Small tolerance
            return False

    return True


fn smooth_bounds_fast(
    lower: UnsafePointer[Float32],
    upper: UnsafePointer[Float32],
    num_atoms: Int,
    max_passes: Int
):
    """
    Fast triangle smoothing with limited passes.

    Better for GPU as it has fixed iteration count.
    """
    var n = num_atoms

    for _ in range(max_passes):
        # Forward pass
        for k in range(n):
            for i in range(n):
                if i == k:
                    continue
                for j in range(i + 1, n):
                    if j == k:
                        continue

                    var idx_ij = get_pair_index(i, j, n)
                    var idx_ik = get_pair_index(i, k, n)
                    var idx_jk = get_pair_index(j, k, n)

                    # Upper bound update
                    var new_upper = upper[idx_ik] + upper[idx_jk]
                    if new_upper < upper[idx_ij]:
                        upper[idx_ij] = new_upper

                    # Lower bound update
                    var diff1 = lower[idx_ik] - upper[idx_jk]
                    var diff2 = lower[idx_jk] - upper[idx_ik]
                    var new_lower = diff1 if diff1 > diff2 else diff2
                    if new_lower < Float32(0.0):
                        new_lower = Float32(0.0)
                    if new_lower > lower[idx_ij]:
                        lower[idx_ij] = new_lower

        # Backward pass
        for k in range(n - 1, -1, -1):
            for i in range(n - 1, -1, -1):
                if i == k:
                    continue
                for j in range(n - 1, i, -1):
                    if j == k:
                        continue

                    var idx_ij = get_pair_index(i, j, n)
                    var idx_ik = get_pair_index(i, k, n)
                    var idx_jk = get_pair_index(j, k, n)

                    var new_upper = upper[idx_ik] + upper[idx_jk]
                    if new_upper < upper[idx_ij]:
                        upper[idx_ij] = new_upper

                    var diff1 = lower[idx_ik] - upper[idx_jk]
                    var diff2 = lower[idx_jk] - upper[idx_ik]
                    var new_lower = diff1 if diff1 > diff2 else diff2
                    if new_lower < Float32(0.0):
                        new_lower = Float32(0.0)
                    if new_lower > lower[idx_ij]:
                        lower[idx_ij] = new_lower


fn distance_to_gram_matrix(
    distances: UnsafePointer[Float32],
    mut gram: UnsafePointer[Float32],
    num_atoms: Int
):
    """
    Convert distance matrix to Gram matrix (inner product matrix).

    Uses double centering:
    G = -0.5 * J * D^2 * J
    where J = I - (1/n) * 1*1^T (centering matrix)

    This is equivalent to:
    G[i,j] = -0.5 * (D[i,j]^2 - avg_row_i - avg_row_j + avg_total)
    """
    var n = num_atoms

    # Compute row averages of D^2
    var row_avg = UnsafePointer[Float32].alloc(n)
    var total_sum: Float32 = 0.0

    for i in range(n):
        var sum: Float32 = 0.0
        for j in range(n):
            var d_sq: Float32 = 0.0
            if i != j:
                var idx = get_pair_index(i, j, n)
                d_sq = distances[idx] * distances[idx]
            sum += d_sq
            total_sum += d_sq
        row_avg[i] = sum / Float32(n)

    var total_avg = total_sum / Float32(n * n)

    # Compute Gram matrix elements
    for i in range(n):
        for j in range(n):
            var d_sq: Float32 = 0.0
            if i != j:
                var idx = get_pair_index(i, j, n)
                d_sq = distances[idx] * distances[idx]

            # Double centering
            var g_ij = Float32(-0.5) * (d_sq - row_avg[i] - row_avg[j] + total_avg)
            gram[i * n + j] = g_ij

    row_avg.free()


fn power_iteration_3d(
    gram: UnsafePointer[Float32],
    mut x: UnsafePointer[Float32],
    mut y: UnsafePointer[Float32],
    mut z: UnsafePointer[Float32],
    num_atoms: Int,
    max_iterations: Int,
    tolerance: Float32
):
    """
    Extract top 3 eigenvectors of Gram matrix using power iteration.

    This is a simplified approach. For production, use proper
    eigendecomposition or randomized SVD.
    """
    var n = num_atoms

    # Allocate work vectors
    var v1 = UnsafePointer[Float32].alloc(n)
    var v2 = UnsafePointer[Float32].alloc(n)
    var v3 = UnsafePointer[Float32].alloc(n)
    var temp = UnsafePointer[Float32].alloc(n)

    # Initialize with random vectors
    var seed = 42
    for i in range(n):
        seed = (LCG_A * seed + LCG_C) & LCG_M
        v1[i] = Float32(seed) / Float32(LCG_M) - Float32(0.5)
        seed = (LCG_A * seed + LCG_C) & LCG_M
        v2[i] = Float32(seed) / Float32(LCG_M) - Float32(0.5)
        seed = (LCG_A * seed + LCG_C) & LCG_M
        v3[i] = Float32(seed) / Float32(LCG_M) - Float32(0.5)

    # Power iteration for first eigenvector
    var lambda1 = power_iterate(gram, v1, temp, n, max_iterations, tolerance)

    # Deflate matrix and get second eigenvector
    deflate_matrix(gram, v1, lambda1, n)
    var lambda2 = power_iterate(gram, v2, temp, n, max_iterations, tolerance)

    # Orthogonalize v2 against v1
    orthogonalize(v2, v1, n)
    normalize(v2, n)

    # Deflate and get third eigenvector
    deflate_matrix(gram, v2, lambda2, n)
    var lambda3 = power_iterate(gram, v3, temp, n, max_iterations, tolerance)

    # Orthogonalize v3 against v1 and v2
    orthogonalize(v3, v1, n)
    orthogonalize(v3, v2, n)
    normalize(v3, n)

    # Scale eigenvectors by sqrt of eigenvalues to get coordinates
    var scale1 = sqrt(lambda1) if lambda1 > Float32(0.0) else Float32(0.0)
    var scale2 = sqrt(lambda2) if lambda2 > Float32(0.0) else Float32(0.0)
    var scale3 = sqrt(lambda3) if lambda3 > Float32(0.0) else Float32(0.0)

    for i in range(n):
        x[i] = v1[i] * scale1
        y[i] = v2[i] * scale2
        z[i] = v3[i] * scale3

    # Free work vectors
    v1.free()
    v2.free()
    v3.free()
    temp.free()


fn power_iterate(
    gram: UnsafePointer[Float32],
    mut v: UnsafePointer[Float32],
    mut temp: UnsafePointer[Float32],
    n: Int,
    max_iterations: Int,
    tolerance: Float32
) -> Float32:
    """
    Power iteration to find dominant eigenvector.

    Returns the eigenvalue.
    """
    normalize(v, n)

    var eigenvalue: Float32 = 0.0

    for _ in range(max_iterations):
        # temp = G * v
        for i in range(n):
            var sum: Float32 = 0.0
            for j in range(n):
                sum += gram[i * n + j] * v[j]
            temp[i] = sum

        # Compute Rayleigh quotient (eigenvalue estimate)
        var num: Float32 = 0.0
        var denom: Float32 = 0.0
        for i in range(n):
            num += v[i] * temp[i]
            denom += v[i] * v[i]

        var new_eigenvalue = num / denom if denom > Float32(1e-10) else Float32(0.0)

        # Check convergence
        var diff = new_eigenvalue - eigenvalue
        if diff < Float32(0.0):
            diff = -diff
        if diff < tolerance:
            eigenvalue = new_eigenvalue
            break

        eigenvalue = new_eigenvalue

        # Normalize and swap
        normalize(temp, n)
        for i in range(n):
            v[i] = temp[i]

    return eigenvalue


fn deflate_matrix(
    mut gram: UnsafePointer[Float32],
    v: UnsafePointer[Float32],
    eigenvalue: Float32,
    n: Int
):
    """
    Deflate matrix: G = G - lambda * v * v^T
    """
    for i in range(n):
        for j in range(n):
            gram[i * n + j] = gram[i * n + j] - eigenvalue * v[i] * v[j]


fn orthogonalize(
    mut v: UnsafePointer[Float32],
    u: UnsafePointer[Float32],
    n: Int
):
    """
    Gram-Schmidt orthogonalization: v = v - (v.u)/(u.u) * u
    """
    var dot_vu: Float32 = 0.0
    var dot_uu: Float32 = 0.0
    for i in range(n):
        dot_vu += v[i] * u[i]
        dot_uu += u[i] * u[i]

    if dot_uu > Float32(1e-10):
        var scale = dot_vu / dot_uu
        for i in range(n):
            v[i] = v[i] - scale * u[i]


fn normalize(mut v: UnsafePointer[Float32], n: Int):
    """Normalize vector to unit length."""
    var norm: Float32 = 0.0
    for i in range(n):
        norm += v[i] * v[i]

    norm = sqrt(norm)
    if norm > Float32(1e-10):
        var inv_norm = Float32(1.0) / norm
        for i in range(n):
            v[i] = v[i] * inv_norm


fn embed_from_distances(
    distances: UnsafePointer[Float32],
    mut x: UnsafePointer[Float32],
    mut y: UnsafePointer[Float32],
    mut z: UnsafePointer[Float32],
    num_atoms: Int
):
    """
    Embed distance matrix into 3D coordinates.

    Main entry point for MDS embedding.
    """
    var n = num_atoms

    # Allocate Gram matrix
    var gram = UnsafePointer[Float32].alloc(n * n)

    # Convert distances to Gram matrix
    distance_to_gram_matrix(distances, gram, n)

    # Extract 3D coordinates via power iteration
    power_iteration_3d(gram, x, y, z, n, 100, Float32(1e-6))

    # Center coordinates at origin
    var cx: Float32 = 0.0
    var cy: Float32 = 0.0
    var cz: Float32 = 0.0
    for i in range(n):
        cx += x[i]
        cy += y[i]
        cz += z[i]
    cx /= Float32(n)
    cy /= Float32(n)
    cz /= Float32(n)

    for i in range(n):
        x[i] = x[i] - cx
        y[i] = y[i] - cy
        z[i] = z[i] - cz

    gram.free()


fn compute_distance_violations(
    x: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    z: UnsafePointer[Float32],
    lower: UnsafePointer[Float32],
    upper: UnsafePointer[Float32],
    num_atoms: Int
) -> Float32:
    """
    Compute sum of squared distance bound violations.

    Returns 0 if all distances are within bounds.
    """
    var violation: Float32 = 0.0
    var n = num_atoms

    for i in range(n):
        for j in range(i + 1, n):
            var dx = x[j] - x[i]
            var dy = y[j] - y[i]
            var dz = z[j] - z[i]
            var d = sqrt(dx * dx + dy * dy + dz * dz)

            var idx = get_pair_index(i, j, n)

            if d < lower[idx]:
                var diff = lower[idx] - d
                violation += diff * diff
            elif d > upper[idx]:
                var diff = d - upper[idx]
                violation += diff * diff

    return violation


fn refine_coordinates(
    mut x: UnsafePointer[Float32],
    mut y: UnsafePointer[Float32],
    mut z: UnsafePointer[Float32],
    lower: UnsafePointer[Float32],
    upper: UnsafePointer[Float32],
    num_atoms: Int,
    max_iterations: Int,
    tolerance: Float32
):
    """
    Refine coordinates to satisfy distance bounds.

    Uses a simple gradient descent on bound violations.
    """
    var n = num_atoms

    # Allocate gradients
    var grad_x = UnsafePointer[Float32].alloc(n)
    var grad_y = UnsafePointer[Float32].alloc(n)
    var grad_z = UnsafePointer[Float32].alloc(n)

    var step_size: Float32 = 0.1

    for iter in range(max_iterations):
        # Zero gradients
        for i in range(n):
            grad_x[i] = 0.0
            grad_y[i] = 0.0
            grad_z[i] = 0.0

        # Compute gradients from bound violations
        var total_violation: Float32 = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                var dx = x[j] - x[i]
                var dy = y[j] - y[i]
                var dz = z[j] - z[i]
                var d = sqrt(dx * dx + dy * dy + dz * dz)

                if d < Float32(1e-8):
                    continue

                var idx = get_pair_index(i, j, n)
                var l = lower[idx]
                var u = upper[idx]

                var force: Float32 = 0.0
                if d < l:
                    # Need to push apart
                    force = l - d
                    total_violation += force * force
                elif d > u:
                    # Need to pull together
                    force = u - d
                    total_violation += force * force

                if force != Float32(0.0):
                    # Force direction: unit vector from i to j
                    var inv_d = Float32(1.0) / d
                    var fx = force * dx * inv_d
                    var fy = force * dy * inv_d
                    var fz = force * dz * inv_d

                    # Apply force (opposite on i and j)
                    grad_x[i] = grad_x[i] - fx
                    grad_y[i] = grad_y[i] - fy
                    grad_z[i] = grad_z[i] - fz
                    grad_x[j] = grad_x[j] + fx
                    grad_y[j] = grad_y[j] + fy
                    grad_z[j] = grad_z[j] + fz

        # Check convergence
        if total_violation < tolerance:
            break

        # Apply gradients
        for i in range(n):
            x[i] = x[i] + step_size * grad_x[i]
            y[i] = y[i] + step_size * grad_y[i]
            z[i] = z[i] + step_size * grad_z[i]

        # Adaptive step size
        if iter > 0 and iter % 50 == 0:
            step_size *= Float32(0.9)

    grad_x.free()
    grad_y.free()
    grad_z.free()


fn chirality_check(
    x: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    z: UnsafePointer[Float32],
    chiral_centers: UnsafePointer[Int32],
    chiral_neighbors: UnsafePointer[Int32],
    expected_sign: UnsafePointer[Int32],
    num_chiral: Int
) -> Bool:
    """
    Check if chiral centers have correct stereochemistry.

    For each chiral center, computes signed volume of tetrahedron
    formed by the four neighbors.
    """
    for chiral_idx in range(num_chiral):
        var center = Int(chiral_centers[chiral_idx])
        var n0 = Int(chiral_neighbors[chiral_idx * 4 + 0])
        var n1 = Int(chiral_neighbors[chiral_idx * 4 + 1])
        var n2 = Int(chiral_neighbors[chiral_idx * 4 + 2])
        var n3 = Int(chiral_neighbors[chiral_idx * 4 + 3])

        # Vectors from n0 to n1, n2, n3
        var v1x = x[n1] - x[n0]
        var v1y = y[n1] - y[n0]
        var v1z = z[n1] - z[n0]

        var v2x = x[n2] - x[n0]
        var v2y = y[n2] - y[n0]
        var v2z = z[n2] - z[n0]

        var v3x = x[n3] - x[n0]
        var v3y = y[n3] - y[n0]
        var v3z = z[n3] - z[n0]

        # Signed volume = v1 . (v2 x v3)
        var cross_x = v2y * v3z - v2z * v3y
        var cross_y = v2z * v3x - v2x * v3z
        var cross_z = v2x * v3y - v2y * v3x

        var volume = v1x * cross_x + v1y * cross_y + v1z * cross_z

        var sign = 1 if volume > Float32(0.0) else -1
        if sign != Int(expected_sign[chiral_idx]):
            return False

    return True


fn invert_coordinates(
    mut x: UnsafePointer[Float32],
    mut y: UnsafePointer[Float32],
    mut z: UnsafePointer[Float32],
    num_atoms: Int
):
    """
    Invert coordinates through origin (mirror image).

    Used to fix chirality when needed.
    """
    for i in range(num_atoms):
        z[i] = -z[i]  # Reflect through xy plane


fn random_perturbation(
    mut x: UnsafePointer[Float32],
    mut y: UnsafePointer[Float32],
    mut z: UnsafePointer[Float32],
    num_atoms: Int,
    magnitude: Float32,
    seed: Int
):
    """
    Add random perturbation to coordinates.

    Useful for generating multiple conformers.
    """
    var state = seed

    for i in range(num_atoms):
        state = (LCG_A * state + LCG_C) & LCG_M
        var rx = (Float32(state) / Float32(LCG_M) - Float32(0.5)) * Float32(2.0) * magnitude

        state = (LCG_A * state + LCG_C) & LCG_M
        var ry = (Float32(state) / Float32(LCG_M) - Float32(0.5)) * Float32(2.0) * magnitude

        state = (LCG_A * state + LCG_C) & LCG_M
        var rz = (Float32(state) / Float32(LCG_M) - Float32(0.5)) * Float32(2.0) * magnitude

        x[i] = x[i] + rx
        y[i] = y[i] + ry
        z[i] = z[i] + rz
