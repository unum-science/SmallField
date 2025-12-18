# SmallField: Molecular Data Types
# Structure of Arrays (SoA) layout with 128-byte alignment for GPU coalescing
# Merged from: atom.mojo, bond.mojo, angle.mojo, torsion.mojo, molecule.mojo
# Updated for Mojo 0.25.7

from memory import alloc


# =============================================================================
# Constants
# =============================================================================

alias ALIGNMENT: Int = 128  # 128-byte alignment for GPU warp transactions
alias SIMD_WIDTH: Int = 8   # For AVX-512 with Float32

# Compile-time size classes for fixed-size molecule batches
alias SIZE_64: Int = 64      # Tiny fragments
alias SIZE_128: Int = 128    # Small molecules (median ~50 atoms with H)
alias SIZE_256: Int = 256    # Medium molecules
alias SIZE_512: Int = 512    # Large molecules
alias SIZE_1024: Int = 1024  # Very large molecules


# =============================================================================
# Utility Functions
# =============================================================================

fn align_capacity(n: Int) -> Int:
    """Round up capacity to nearest multiple of 32 for alignment."""
    return ((n + 31) // 32) * 32


fn select_size_class(num_atoms: Int) -> Int:
    """Select smallest size bucket that fits the given atom count."""
    if num_atoms <= 64:
        return SIZE_64
    if num_atoms <= 128:
        return SIZE_128
    if num_atoms <= 256:
        return SIZE_256
    if num_atoms <= 512:
        return SIZE_512
    return SIZE_1024


# =============================================================================
# AtomBuffers
# =============================================================================

struct AtomBuffers(Copyable, Movable):
    """
    SoA storage for atom positions and gradients.

    Memory layout optimized for:
    - GPU coalesced access (128-byte aligned)
    - CPU SIMD (AVX-512 friendly)
    - Batched molecule processing (CSR indexing)
    """
    # Positions (FP32 storage)
    var x: type_of(alloc[Float32](1))
    var y: type_of(alloc[Float32](1))
    var z: type_of(alloc[Float32](1))

    # Gradients (FP32 for accumulation stability)
    var grad_x: type_of(alloc[Float32](1))
    var grad_y: type_of(alloc[Float32](1))
    var grad_z: type_of(alloc[Float32](1))

    # Atom types for MMFF94 parameters
    var atom_type: type_of(alloc[Int32](1))

    # Partial charges
    var charge: type_of(alloc[Float32](1))

    # CSR indexing for batch access
    var atom_starts: type_of(alloc[Int32](1))  # [num_molecules + 1]

    # Sizes
    var capacity: Int
    var num_atoms: Int
    var num_molecules: Int

    fn __init__(out self, max_atoms: Int, max_molecules: Int):
        """Allocate buffers with aligned capacity."""
        var cap = align_capacity(max_atoms)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_atoms = 0
        self.num_molecules = 0

        # Allocate position buffers
        self.x = alloc[Float32](cap)
        self.y = alloc[Float32](cap)
        self.z = alloc[Float32](cap)

        # Allocate gradient buffers
        self.grad_x = alloc[Float32](cap)
        self.grad_y = alloc[Float32](cap)
        self.grad_z = alloc[Float32](cap)

        # Allocate type and charge buffers
        self.atom_type = alloc[Int32](cap)
        self.charge = alloc[Float32](cap)

        # Allocate CSR index
        self.atom_starts = alloc[Int32](mol_cap)

        # Initialize to zero
        for i in range(cap):
            self.x[i] = 0.0
            self.y[i] = 0.0
            self.z[i] = 0.0
            self.grad_x[i] = 0.0
            self.grad_y[i] = 0.0
            self.grad_z[i] = 0.0
            self.atom_type[i] = 0
            self.charge[i] = 0.0

        for i in range(mol_cap):
            self.atom_starts[i] = 0

    fn __copyinit__(out self, other: Self):
        self.capacity = other.capacity
        self.num_atoms = other.num_atoms
        self.num_molecules = other.num_molecules

        var cap = other.capacity
        var mol_cap = align_capacity(other.num_molecules + 1)

        self.x = alloc[Float32](cap)
        self.y = alloc[Float32](cap)
        self.z = alloc[Float32](cap)
        self.grad_x = alloc[Float32](cap)
        self.grad_y = alloc[Float32](cap)
        self.grad_z = alloc[Float32](cap)
        self.atom_type = alloc[Int32](cap)
        self.charge = alloc[Float32](cap)
        self.atom_starts = alloc[Int32](mol_cap)

        for i in range(cap):
            self.x[i] = other.x[i]
            self.y[i] = other.y[i]
            self.z[i] = other.z[i]
            self.grad_x[i] = other.grad_x[i]
            self.grad_y[i] = other.grad_y[i]
            self.grad_z[i] = other.grad_z[i]
            self.atom_type[i] = other.atom_type[i]
            self.charge[i] = other.charge[i]

        for i in range(mol_cap):
            self.atom_starts[i] = other.atom_starts[i]

    fn __moveinit__(out self, deinit other: Self):
        self.capacity = other.capacity
        self.num_atoms = other.num_atoms
        self.num_molecules = other.num_molecules
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.grad_x = other.grad_x
        self.grad_y = other.grad_y
        self.grad_z = other.grad_z
        self.atom_type = other.atom_type
        self.charge = other.charge
        self.atom_starts = other.atom_starts

    fn free(self):
        """Release all allocated memory."""
        self.x.free()
        self.y.free()
        self.z.free()
        self.grad_x.free()
        self.grad_y.free()
        self.grad_z.free()
        self.atom_type.free()
        self.charge.free()
        self.atom_starts.free()

    fn zero_gradients(self):
        """Reset all gradients to zero."""
        for i in range(self.num_atoms):
            self.grad_x[i] = 0.0
            self.grad_y[i] = 0.0
            self.grad_z[i] = 0.0

    fn set_atom(
        self,
        idx: Int,
        px: Float32, py: Float32, pz: Float32,
        atype: Int32,
        q: Float32
    ):
        """Set atom data at index."""
        self.x[idx] = px
        self.y[idx] = py
        self.z[idx] = pz
        self.atom_type[idx] = atype
        self.charge[idx] = q

    fn get_molecule_atom_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        """Get atom index range for a molecule: [start, end)."""
        var start = Int(self.atom_starts[mol_idx])
        var end = Int(self.atom_starts[mol_idx + 1])
        return (start, end)

    fn get_num_atoms_in_molecule(self, mol_idx: Int) -> Int:
        """Get number of atoms in a specific molecule."""
        return Int(self.atom_starts[mol_idx + 1]) - Int(self.atom_starts[mol_idx])


# =============================================================================
# BondBuffers
# =============================================================================

struct BondBuffers(Copyable, Movable):
    """
    SoA storage for bond topology and MMFF94 bond stretch parameters.

    Uses CSR-like indexing for efficient batch processing:
    - bond_starts[mol] to bond_starts[mol+1] gives bond range for molecule
    - Atom indices are local to the batch (offset by atom_starts)
    """
    # Bond topology (atom indices)
    var atom_i: type_of(alloc[Int32](1))
    var atom_j: type_of(alloc[Int32](1))

    # MMFF94 bond stretch parameters
    var force_constant: type_of(alloc[Float32](1))  # kb (mdyn/A)
    var rest_length: type_of(alloc[Float32](1))     # r0 (Angstroms)

    # Bond type for MMFF94 (0-5)
    var bond_type: type_of(alloc[Int32](1))

    # CSR indexing for batch access
    var bond_starts: type_of(alloc[Int32](1))  # [num_molecules + 1]

    # Sizes
    var capacity: Int
    var num_bonds: Int
    var num_molecules: Int

    fn __init__(out self, max_bonds: Int, max_molecules: Int):
        """Allocate buffers with aligned capacity."""
        var cap = align_capacity(max_bonds)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_bonds = 0
        self.num_molecules = 0

        # Allocate topology buffers
        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)

        # Allocate parameter buffers
        self.force_constant = alloc[Float32](cap)
        self.rest_length = alloc[Float32](cap)
        self.bond_type = alloc[Int32](cap)

        # Allocate CSR index
        self.bond_starts = alloc[Int32](mol_cap)

        # Initialize to zero
        for i in range(cap):
            self.atom_i[i] = 0
            self.atom_j[i] = 0
            self.force_constant[i] = 0.0
            self.rest_length[i] = 0.0
            self.bond_type[i] = 0

        for i in range(mol_cap):
            self.bond_starts[i] = 0

    fn __copyinit__(out self, other: Self):
        self.capacity = other.capacity
        self.num_bonds = other.num_bonds
        self.num_molecules = other.num_molecules

        var cap = other.capacity
        var mol_cap = align_capacity(other.num_molecules + 1)

        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)
        self.force_constant = alloc[Float32](cap)
        self.rest_length = alloc[Float32](cap)
        self.bond_type = alloc[Int32](cap)
        self.bond_starts = alloc[Int32](mol_cap)

        for i in range(cap):
            self.atom_i[i] = other.atom_i[i]
            self.atom_j[i] = other.atom_j[i]
            self.force_constant[i] = other.force_constant[i]
            self.rest_length[i] = other.rest_length[i]
            self.bond_type[i] = other.bond_type[i]

        for i in range(mol_cap):
            self.bond_starts[i] = other.bond_starts[i]

    fn __moveinit__(out self, deinit other: Self):
        self.capacity = other.capacity
        self.num_bonds = other.num_bonds
        self.num_molecules = other.num_molecules
        self.atom_i = other.atom_i
        self.atom_j = other.atom_j
        self.force_constant = other.force_constant
        self.rest_length = other.rest_length
        self.bond_type = other.bond_type
        self.bond_starts = other.bond_starts

    fn free(self):
        """Release all allocated memory."""
        self.atom_i.free()
        self.atom_j.free()
        self.force_constant.free()
        self.rest_length.free()
        self.bond_type.free()
        self.bond_starts.free()

    fn set_bond(
        self,
        idx: Int,
        i: Int32, j: Int32,
        kb: Float32, r0: Float32,
        btype: Int32
    ):
        """Set bond data at index."""
        self.atom_i[idx] = i
        self.atom_j[idx] = j
        self.force_constant[idx] = kb
        self.rest_length[idx] = r0
        self.bond_type[idx] = btype

    fn get_molecule_bond_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        """Get bond index range for a molecule: [start, end)."""
        var start = Int(self.bond_starts[mol_idx])
        var end = Int(self.bond_starts[mol_idx + 1])
        return (start, end)

    fn get_num_bonds_in_molecule(self, mol_idx: Int) -> Int:
        """Get number of bonds in a specific molecule."""
        return Int(self.bond_starts[mol_idx + 1]) - Int(self.bond_starts[mol_idx])


# =============================================================================
# AngleBuffers
# =============================================================================

struct AngleBuffers(Copyable, Movable):
    """
    SoA storage for angle topology and MMFF94 angle bend parameters.

    Angle i-j-k where j is the central atom.
    Uses CSR-like indexing for efficient batch processing.
    """
    # Angle topology (atom indices)
    var atom_i: type_of(alloc[Int32](1))  # First atom
    var atom_j: type_of(alloc[Int32](1))  # Central atom
    var atom_k: type_of(alloc[Int32](1))  # Third atom

    # MMFF94 angle bend parameters
    var force_constant: type_of(alloc[Float32](1))  # ka (mdyn*A/rad^2)
    var rest_angle: type_of(alloc[Float32](1))      # theta0 (radians)

    # Linear angle flag (for 180-degree angles)
    var is_linear: type_of(alloc[Int32](1))

    # CSR indexing for batch access
    var angle_starts: type_of(alloc[Int32](1))  # [num_molecules + 1]

    # Sizes
    var capacity: Int
    var num_angles: Int
    var num_molecules: Int

    fn __init__(out self, max_angles: Int, max_molecules: Int):
        """Allocate buffers with aligned capacity."""
        var cap = align_capacity(max_angles)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_angles = 0
        self.num_molecules = 0

        # Allocate topology buffers
        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)
        self.atom_k = alloc[Int32](cap)

        # Allocate parameter buffers
        self.force_constant = alloc[Float32](cap)
        self.rest_angle = alloc[Float32](cap)
        self.is_linear = alloc[Int32](cap)

        # Allocate CSR index
        self.angle_starts = alloc[Int32](mol_cap)

        # Initialize to zero
        for i in range(cap):
            self.atom_i[i] = 0
            self.atom_j[i] = 0
            self.atom_k[i] = 0
            self.force_constant[i] = 0.0
            self.rest_angle[i] = 0.0
            self.is_linear[i] = 0

        for i in range(mol_cap):
            self.angle_starts[i] = 0

    fn __copyinit__(out self, other: Self):
        self.capacity = other.capacity
        self.num_angles = other.num_angles
        self.num_molecules = other.num_molecules

        var cap = other.capacity
        var mol_cap = align_capacity(other.num_molecules + 1)

        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)
        self.atom_k = alloc[Int32](cap)
        self.force_constant = alloc[Float32](cap)
        self.rest_angle = alloc[Float32](cap)
        self.is_linear = alloc[Int32](cap)
        self.angle_starts = alloc[Int32](mol_cap)

        for i in range(cap):
            self.atom_i[i] = other.atom_i[i]
            self.atom_j[i] = other.atom_j[i]
            self.atom_k[i] = other.atom_k[i]
            self.force_constant[i] = other.force_constant[i]
            self.rest_angle[i] = other.rest_angle[i]
            self.is_linear[i] = other.is_linear[i]

        for i in range(mol_cap):
            self.angle_starts[i] = other.angle_starts[i]

    fn __moveinit__(out self, deinit other: Self):
        self.capacity = other.capacity
        self.num_angles = other.num_angles
        self.num_molecules = other.num_molecules
        self.atom_i = other.atom_i
        self.atom_j = other.atom_j
        self.atom_k = other.atom_k
        self.force_constant = other.force_constant
        self.rest_angle = other.rest_angle
        self.is_linear = other.is_linear
        self.angle_starts = other.angle_starts

    fn free(self):
        """Release all allocated memory."""
        self.atom_i.free()
        self.atom_j.free()
        self.atom_k.free()
        self.force_constant.free()
        self.rest_angle.free()
        self.is_linear.free()
        self.angle_starts.free()

    fn set_angle(
        self,
        idx: Int,
        i: Int32, j: Int32, k: Int32,
        ka: Float32, theta0: Float32,
        linear: Int32
    ):
        """Set angle data at index."""
        self.atom_i[idx] = i
        self.atom_j[idx] = j
        self.atom_k[idx] = k
        self.force_constant[idx] = ka
        self.rest_angle[idx] = theta0
        self.is_linear[idx] = linear

    fn get_molecule_angle_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        """Get angle index range for a molecule: [start, end)."""
        var start = Int(self.angle_starts[mol_idx])
        var end = Int(self.angle_starts[mol_idx + 1])
        return (start, end)

    fn get_num_angles_in_molecule(self, mol_idx: Int) -> Int:
        """Get number of angles in a specific molecule."""
        return Int(self.angle_starts[mol_idx + 1]) - Int(self.angle_starts[mol_idx])


# =============================================================================
# TorsionBuffers
# =============================================================================

struct TorsionBuffers(Copyable, Movable):
    """
    SoA storage for torsion topology and MMFF94 torsion parameters.

    Torsion i-j-k-l where j-k is the central bond.
    Uses CSR-like indexing for efficient batch processing.

    MMFF94 torsion energy:
    E = 0.5 * (V1*(1+cos(phi)) + V2*(1-cos(2*phi)) + V3*(1+cos(3*phi)))
    """
    # Torsion topology (atom indices)
    var atom_i: type_of(alloc[Int32](1))
    var atom_j: type_of(alloc[Int32](1))
    var atom_k: type_of(alloc[Int32](1))
    var atom_l: type_of(alloc[Int32](1))

    # MMFF94 torsion parameters (Fourier coefficients)
    var v1: type_of(alloc[Float32](1))  # kcal/mol
    var v2: type_of(alloc[Float32](1))  # kcal/mol
    var v3: type_of(alloc[Float32](1))  # kcal/mol

    # Torsion type
    var torsion_type: type_of(alloc[Int32](1))

    # CSR indexing for batch access
    var torsion_starts: type_of(alloc[Int32](1))  # [num_molecules + 1]

    # Sizes
    var capacity: Int
    var num_torsions: Int
    var num_molecules: Int

    fn __init__(out self, max_torsions: Int, max_molecules: Int):
        """Allocate buffers with aligned capacity."""
        var cap = align_capacity(max_torsions)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_torsions = 0
        self.num_molecules = 0

        # Allocate topology buffers
        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)
        self.atom_k = alloc[Int32](cap)
        self.atom_l = alloc[Int32](cap)

        # Allocate parameter buffers
        self.v1 = alloc[Float32](cap)
        self.v2 = alloc[Float32](cap)
        self.v3 = alloc[Float32](cap)
        self.torsion_type = alloc[Int32](cap)

        # Allocate CSR index
        self.torsion_starts = alloc[Int32](mol_cap)

        # Initialize to zero
        for i in range(cap):
            self.atom_i[i] = 0
            self.atom_j[i] = 0
            self.atom_k[i] = 0
            self.atom_l[i] = 0
            self.v1[i] = 0.0
            self.v2[i] = 0.0
            self.v3[i] = 0.0
            self.torsion_type[i] = 0

        for i in range(mol_cap):
            self.torsion_starts[i] = 0

    fn __copyinit__(out self, other: Self):
        self.capacity = other.capacity
        self.num_torsions = other.num_torsions
        self.num_molecules = other.num_molecules

        var cap = other.capacity
        var mol_cap = align_capacity(other.num_molecules + 1)

        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)
        self.atom_k = alloc[Int32](cap)
        self.atom_l = alloc[Int32](cap)
        self.v1 = alloc[Float32](cap)
        self.v2 = alloc[Float32](cap)
        self.v3 = alloc[Float32](cap)
        self.torsion_type = alloc[Int32](cap)
        self.torsion_starts = alloc[Int32](mol_cap)

        for i in range(cap):
            self.atom_i[i] = other.atom_i[i]
            self.atom_j[i] = other.atom_j[i]
            self.atom_k[i] = other.atom_k[i]
            self.atom_l[i] = other.atom_l[i]
            self.v1[i] = other.v1[i]
            self.v2[i] = other.v2[i]
            self.v3[i] = other.v3[i]
            self.torsion_type[i] = other.torsion_type[i]

        for i in range(mol_cap):
            self.torsion_starts[i] = other.torsion_starts[i]

    fn __moveinit__(out self, deinit other: Self):
        self.capacity = other.capacity
        self.num_torsions = other.num_torsions
        self.num_molecules = other.num_molecules
        self.atom_i = other.atom_i
        self.atom_j = other.atom_j
        self.atom_k = other.atom_k
        self.atom_l = other.atom_l
        self.v1 = other.v1
        self.v2 = other.v2
        self.v3 = other.v3
        self.torsion_type = other.torsion_type
        self.torsion_starts = other.torsion_starts

    fn free(self):
        """Release all allocated memory."""
        self.atom_i.free()
        self.atom_j.free()
        self.atom_k.free()
        self.atom_l.free()
        self.v1.free()
        self.v2.free()
        self.v3.free()
        self.torsion_type.free()
        self.torsion_starts.free()

    fn set_torsion(
        self,
        idx: Int,
        i: Int32, j: Int32, k: Int32, l: Int32,
        v1_val: Float32, v2_val: Float32, v3_val: Float32,
        ttype: Int32
    ):
        """Set torsion data at index."""
        self.atom_i[idx] = i
        self.atom_j[idx] = j
        self.atom_k[idx] = k
        self.atom_l[idx] = l
        self.v1[idx] = v1_val
        self.v2[idx] = v2_val
        self.v3[idx] = v3_val
        self.torsion_type[idx] = ttype

    fn get_molecule_torsion_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        """Get torsion index range for a molecule: [start, end)."""
        var start = Int(self.torsion_starts[mol_idx])
        var end = Int(self.torsion_starts[mol_idx + 1])
        return (start, end)

    fn get_num_torsions_in_molecule(self, mol_idx: Int) -> Int:
        """Get number of torsions in a specific molecule."""
        return Int(self.torsion_starts[mol_idx + 1]) - Int(self.torsion_starts[mol_idx])


# =============================================================================
# NonbondedPairs
# =============================================================================

struct NonbondedPairs(Copyable, Movable):
    """
    Storage for van der Waals and electrostatic interaction pairs.
    Excludes 1-2 (bonded) and 1-3 (angle) pairs.
    Includes 1-4 pairs with scaling factors.
    """
    var atom_i: type_of(alloc[Int32](1))
    var atom_j: type_of(alloc[Int32](1))
    var is_1_4: type_of(alloc[Int32](1))  # 1 if 1-4 pair, 0 otherwise

    # vdW parameters (combined for pair)
    var epsilon: type_of(alloc[Float32](1))  # kcal/mol
    var r_star: type_of(alloc[Float32](1))   # Angstroms

    # CSR indexing
    var pair_starts: type_of(alloc[Int32](1))

    var capacity: Int
    var num_pairs: Int
    var num_molecules: Int

    fn __init__(out self, max_pairs: Int, max_molecules: Int):
        var cap = align_capacity(max_pairs)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_pairs = 0
        self.num_molecules = 0

        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)
        self.is_1_4 = alloc[Int32](cap)
        self.epsilon = alloc[Float32](cap)
        self.r_star = alloc[Float32](cap)
        self.pair_starts = alloc[Int32](mol_cap)

        for i in range(cap):
            self.atom_i[i] = 0
            self.atom_j[i] = 0
            self.is_1_4[i] = 0
            self.epsilon[i] = 0.0
            self.r_star[i] = 0.0

        for i in range(mol_cap):
            self.pair_starts[i] = 0

    fn __copyinit__(out self, other: Self):
        self.capacity = other.capacity
        self.num_pairs = other.num_pairs
        self.num_molecules = other.num_molecules

        var cap = other.capacity
        var mol_cap = align_capacity(other.num_molecules + 1)

        self.atom_i = alloc[Int32](cap)
        self.atom_j = alloc[Int32](cap)
        self.is_1_4 = alloc[Int32](cap)
        self.epsilon = alloc[Float32](cap)
        self.r_star = alloc[Float32](cap)
        self.pair_starts = alloc[Int32](mol_cap)

        for i in range(cap):
            self.atom_i[i] = other.atom_i[i]
            self.atom_j[i] = other.atom_j[i]
            self.is_1_4[i] = other.is_1_4[i]
            self.epsilon[i] = other.epsilon[i]
            self.r_star[i] = other.r_star[i]

        for i in range(mol_cap):
            self.pair_starts[i] = other.pair_starts[i]

    fn __moveinit__(out self, deinit other: Self):
        self.capacity = other.capacity
        self.num_pairs = other.num_pairs
        self.num_molecules = other.num_molecules
        self.atom_i = other.atom_i
        self.atom_j = other.atom_j
        self.is_1_4 = other.is_1_4
        self.epsilon = other.epsilon
        self.r_star = other.r_star
        self.pair_starts = other.pair_starts

    fn free(self):
        self.atom_i.free()
        self.atom_j.free()
        self.is_1_4.free()
        self.epsilon.free()
        self.r_star.free()
        self.pair_starts.free()


# =============================================================================
# MoleculeBatch
# =============================================================================

struct MoleculeBatch(Copyable, Movable):
    """
    Container for a batch of molecules with all MMFF94 force field data.

    All data is stored in SoA format with CSR-like indexing for
    efficient GPU coalescing and CPU SIMD processing.
    """
    # Positions and gradients
    var x: type_of(alloc[Float32](1))
    var y: type_of(alloc[Float32](1))
    var z: type_of(alloc[Float32](1))
    var grad_x: type_of(alloc[Float32](1))
    var grad_y: type_of(alloc[Float32](1))
    var grad_z: type_of(alloc[Float32](1))
    var atom_type: type_of(alloc[Int32](1))
    var charge: type_of(alloc[Float32](1))
    var atom_starts: type_of(alloc[Int32](1))

    # Bonds
    var bond_i: type_of(alloc[Int32](1))
    var bond_j: type_of(alloc[Int32](1))
    var bond_kb: type_of(alloc[Float32](1))
    var bond_r0: type_of(alloc[Float32](1))
    var bond_starts: type_of(alloc[Int32](1))

    # Angles
    var angle_i: type_of(alloc[Int32](1))
    var angle_j: type_of(alloc[Int32](1))
    var angle_k: type_of(alloc[Int32](1))
    var angle_ka: type_of(alloc[Float32](1))
    var angle_theta0: type_of(alloc[Float32](1))
    var angle_starts: type_of(alloc[Int32](1))

    # Torsions
    var torsion_i: type_of(alloc[Int32](1))
    var torsion_j: type_of(alloc[Int32](1))
    var torsion_k: type_of(alloc[Int32](1))
    var torsion_l: type_of(alloc[Int32](1))
    var torsion_v1: type_of(alloc[Float32](1))
    var torsion_v2: type_of(alloc[Float32](1))
    var torsion_v3: type_of(alloc[Float32](1))
    var torsion_starts: type_of(alloc[Int32](1))

    # Nonbonded pairs
    var nb_i: type_of(alloc[Int32](1))
    var nb_j: type_of(alloc[Int32](1))
    var nb_is_1_4: type_of(alloc[Int32](1))
    var nb_epsilon: type_of(alloc[Float32](1))
    var nb_r_star: type_of(alloc[Float32](1))
    var nb_starts: type_of(alloc[Int32](1))

    # Capacities
    var atom_capacity: Int
    var bond_capacity: Int
    var angle_capacity: Int
    var torsion_capacity: Int
    var nb_capacity: Int
    var mol_capacity: Int

    # Current counts
    var num_atoms: Int
    var num_bonds: Int
    var num_angles: Int
    var num_torsions: Int
    var num_nb_pairs: Int
    var num_molecules: Int

    fn __init__(
        out self,
        max_atoms: Int,
        max_bonds: Int,
        max_angles: Int,
        max_torsions: Int,
        max_nb_pairs: Int,
        max_molecules: Int
    ):
        """Allocate all buffers with aligned capacity."""
        self.atom_capacity = align_capacity(max_atoms)
        self.bond_capacity = align_capacity(max_bonds)
        self.angle_capacity = align_capacity(max_angles)
        self.torsion_capacity = align_capacity(max_torsions)
        self.nb_capacity = align_capacity(max_nb_pairs)
        self.mol_capacity = align_capacity(max_molecules + 1)

        self.num_atoms = 0
        self.num_bonds = 0
        self.num_angles = 0
        self.num_torsions = 0
        self.num_nb_pairs = 0
        self.num_molecules = 0

        # Allocate atom buffers
        self.x = alloc[Float32](self.atom_capacity)
        self.y = alloc[Float32](self.atom_capacity)
        self.z = alloc[Float32](self.atom_capacity)
        self.grad_x = alloc[Float32](self.atom_capacity)
        self.grad_y = alloc[Float32](self.atom_capacity)
        self.grad_z = alloc[Float32](self.atom_capacity)
        self.atom_type = alloc[Int32](self.atom_capacity)
        self.charge = alloc[Float32](self.atom_capacity)
        self.atom_starts = alloc[Int32](self.mol_capacity)

        # Allocate bond buffers
        self.bond_i = alloc[Int32](self.bond_capacity)
        self.bond_j = alloc[Int32](self.bond_capacity)
        self.bond_kb = alloc[Float32](self.bond_capacity)
        self.bond_r0 = alloc[Float32](self.bond_capacity)
        self.bond_starts = alloc[Int32](self.mol_capacity)

        # Allocate angle buffers
        self.angle_i = alloc[Int32](self.angle_capacity)
        self.angle_j = alloc[Int32](self.angle_capacity)
        self.angle_k = alloc[Int32](self.angle_capacity)
        self.angle_ka = alloc[Float32](self.angle_capacity)
        self.angle_theta0 = alloc[Float32](self.angle_capacity)
        self.angle_starts = alloc[Int32](self.mol_capacity)

        # Allocate torsion buffers
        self.torsion_i = alloc[Int32](self.torsion_capacity)
        self.torsion_j = alloc[Int32](self.torsion_capacity)
        self.torsion_k = alloc[Int32](self.torsion_capacity)
        self.torsion_l = alloc[Int32](self.torsion_capacity)
        self.torsion_v1 = alloc[Float32](self.torsion_capacity)
        self.torsion_v2 = alloc[Float32](self.torsion_capacity)
        self.torsion_v3 = alloc[Float32](self.torsion_capacity)
        self.torsion_starts = alloc[Int32](self.mol_capacity)

        # Allocate nonbonded buffers
        self.nb_i = alloc[Int32](self.nb_capacity)
        self.nb_j = alloc[Int32](self.nb_capacity)
        self.nb_is_1_4 = alloc[Int32](self.nb_capacity)
        self.nb_epsilon = alloc[Float32](self.nb_capacity)
        self.nb_r_star = alloc[Float32](self.nb_capacity)
        self.nb_starts = alloc[Int32](self.mol_capacity)

        # Initialize CSR starts to 0
        for i in range(self.mol_capacity):
            self.atom_starts[i] = 0
            self.bond_starts[i] = 0
            self.angle_starts[i] = 0
            self.torsion_starts[i] = 0
            self.nb_starts[i] = 0

    fn __copyinit__(out self, other: Self):
        self.atom_capacity = other.atom_capacity
        self.bond_capacity = other.bond_capacity
        self.angle_capacity = other.angle_capacity
        self.torsion_capacity = other.torsion_capacity
        self.nb_capacity = other.nb_capacity
        self.mol_capacity = other.mol_capacity

        self.num_atoms = other.num_atoms
        self.num_bonds = other.num_bonds
        self.num_angles = other.num_angles
        self.num_torsions = other.num_torsions
        self.num_nb_pairs = other.num_nb_pairs
        self.num_molecules = other.num_molecules

        # Allocate and copy all buffers
        self.x = alloc[Float32](self.atom_capacity)
        self.y = alloc[Float32](self.atom_capacity)
        self.z = alloc[Float32](self.atom_capacity)
        self.grad_x = alloc[Float32](self.atom_capacity)
        self.grad_y = alloc[Float32](self.atom_capacity)
        self.grad_z = alloc[Float32](self.atom_capacity)
        self.atom_type = alloc[Int32](self.atom_capacity)
        self.charge = alloc[Float32](self.atom_capacity)
        self.atom_starts = alloc[Int32](self.mol_capacity)

        self.bond_i = alloc[Int32](self.bond_capacity)
        self.bond_j = alloc[Int32](self.bond_capacity)
        self.bond_kb = alloc[Float32](self.bond_capacity)
        self.bond_r0 = alloc[Float32](self.bond_capacity)
        self.bond_starts = alloc[Int32](self.mol_capacity)

        self.angle_i = alloc[Int32](self.angle_capacity)
        self.angle_j = alloc[Int32](self.angle_capacity)
        self.angle_k = alloc[Int32](self.angle_capacity)
        self.angle_ka = alloc[Float32](self.angle_capacity)
        self.angle_theta0 = alloc[Float32](self.angle_capacity)
        self.angle_starts = alloc[Int32](self.mol_capacity)

        self.torsion_i = alloc[Int32](self.torsion_capacity)
        self.torsion_j = alloc[Int32](self.torsion_capacity)
        self.torsion_k = alloc[Int32](self.torsion_capacity)
        self.torsion_l = alloc[Int32](self.torsion_capacity)
        self.torsion_v1 = alloc[Float32](self.torsion_capacity)
        self.torsion_v2 = alloc[Float32](self.torsion_capacity)
        self.torsion_v3 = alloc[Float32](self.torsion_capacity)
        self.torsion_starts = alloc[Int32](self.mol_capacity)

        self.nb_i = alloc[Int32](self.nb_capacity)
        self.nb_j = alloc[Int32](self.nb_capacity)
        self.nb_is_1_4 = alloc[Int32](self.nb_capacity)
        self.nb_epsilon = alloc[Float32](self.nb_capacity)
        self.nb_r_star = alloc[Float32](self.nb_capacity)
        self.nb_starts = alloc[Int32](self.mol_capacity)

        # Copy data
        for i in range(self.atom_capacity):
            self.x[i] = other.x[i]
            self.y[i] = other.y[i]
            self.z[i] = other.z[i]
            self.grad_x[i] = other.grad_x[i]
            self.grad_y[i] = other.grad_y[i]
            self.grad_z[i] = other.grad_z[i]
            self.atom_type[i] = other.atom_type[i]
            self.charge[i] = other.charge[i]

        for i in range(self.bond_capacity):
            self.bond_i[i] = other.bond_i[i]
            self.bond_j[i] = other.bond_j[i]
            self.bond_kb[i] = other.bond_kb[i]
            self.bond_r0[i] = other.bond_r0[i]

        for i in range(self.angle_capacity):
            self.angle_i[i] = other.angle_i[i]
            self.angle_j[i] = other.angle_j[i]
            self.angle_k[i] = other.angle_k[i]
            self.angle_ka[i] = other.angle_ka[i]
            self.angle_theta0[i] = other.angle_theta0[i]

        for i in range(self.torsion_capacity):
            self.torsion_i[i] = other.torsion_i[i]
            self.torsion_j[i] = other.torsion_j[i]
            self.torsion_k[i] = other.torsion_k[i]
            self.torsion_l[i] = other.torsion_l[i]
            self.torsion_v1[i] = other.torsion_v1[i]
            self.torsion_v2[i] = other.torsion_v2[i]
            self.torsion_v3[i] = other.torsion_v3[i]

        for i in range(self.nb_capacity):
            self.nb_i[i] = other.nb_i[i]
            self.nb_j[i] = other.nb_j[i]
            self.nb_is_1_4[i] = other.nb_is_1_4[i]
            self.nb_epsilon[i] = other.nb_epsilon[i]
            self.nb_r_star[i] = other.nb_r_star[i]

        for i in range(self.mol_capacity):
            self.atom_starts[i] = other.atom_starts[i]
            self.bond_starts[i] = other.bond_starts[i]
            self.angle_starts[i] = other.angle_starts[i]
            self.torsion_starts[i] = other.torsion_starts[i]
            self.nb_starts[i] = other.nb_starts[i]

    fn __moveinit__(out self, deinit other: Self):
        self.atom_capacity = other.atom_capacity
        self.bond_capacity = other.bond_capacity
        self.angle_capacity = other.angle_capacity
        self.torsion_capacity = other.torsion_capacity
        self.nb_capacity = other.nb_capacity
        self.mol_capacity = other.mol_capacity

        self.num_atoms = other.num_atoms
        self.num_bonds = other.num_bonds
        self.num_angles = other.num_angles
        self.num_torsions = other.num_torsions
        self.num_nb_pairs = other.num_nb_pairs
        self.num_molecules = other.num_molecules

        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.grad_x = other.grad_x
        self.grad_y = other.grad_y
        self.grad_z = other.grad_z
        self.atom_type = other.atom_type
        self.charge = other.charge
        self.atom_starts = other.atom_starts

        self.bond_i = other.bond_i
        self.bond_j = other.bond_j
        self.bond_kb = other.bond_kb
        self.bond_r0 = other.bond_r0
        self.bond_starts = other.bond_starts

        self.angle_i = other.angle_i
        self.angle_j = other.angle_j
        self.angle_k = other.angle_k
        self.angle_ka = other.angle_ka
        self.angle_theta0 = other.angle_theta0
        self.angle_starts = other.angle_starts

        self.torsion_i = other.torsion_i
        self.torsion_j = other.torsion_j
        self.torsion_k = other.torsion_k
        self.torsion_l = other.torsion_l
        self.torsion_v1 = other.torsion_v1
        self.torsion_v2 = other.torsion_v2
        self.torsion_v3 = other.torsion_v3
        self.torsion_starts = other.torsion_starts

        self.nb_i = other.nb_i
        self.nb_j = other.nb_j
        self.nb_is_1_4 = other.nb_is_1_4
        self.nb_epsilon = other.nb_epsilon
        self.nb_r_star = other.nb_r_star
        self.nb_starts = other.nb_starts

    fn free(self):
        """Release all allocated memory."""
        self.x.free()
        self.y.free()
        self.z.free()
        self.grad_x.free()
        self.grad_y.free()
        self.grad_z.free()
        self.atom_type.free()
        self.charge.free()
        self.atom_starts.free()

        self.bond_i.free()
        self.bond_j.free()
        self.bond_kb.free()
        self.bond_r0.free()
        self.bond_starts.free()

        self.angle_i.free()
        self.angle_j.free()
        self.angle_k.free()
        self.angle_ka.free()
        self.angle_theta0.free()
        self.angle_starts.free()

        self.torsion_i.free()
        self.torsion_j.free()
        self.torsion_k.free()
        self.torsion_l.free()
        self.torsion_v1.free()
        self.torsion_v2.free()
        self.torsion_v3.free()
        self.torsion_starts.free()

        self.nb_i.free()
        self.nb_j.free()
        self.nb_is_1_4.free()
        self.nb_epsilon.free()
        self.nb_r_star.free()
        self.nb_starts.free()

    fn zero_gradients(self):
        """Reset all gradients to zero."""
        for i in range(self.num_atoms):
            self.grad_x[i] = 0.0
            self.grad_y[i] = 0.0
            self.grad_z[i] = 0.0

    fn clear(mut self):
        """Reset batch to empty state."""
        self.num_atoms = 0
        self.num_bonds = 0
        self.num_angles = 0
        self.num_torsions = 0
        self.num_nb_pairs = 0
        self.num_molecules = 0
