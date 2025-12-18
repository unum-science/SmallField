# SmallField: Molecular Data Types
# Structure of Arrays (SoA) layout with 128-byte alignment for GPU coalescing
# Merged from: atom.mojo, bond.mojo, angle.mojo, torsion.mojo, molecule.mojo
# Updated for Mojo 0.25.7

from memory import UnsafePointer, alloc

# Constants
alias ALIGNMENT: Int = 128
alias SIMD_WIDTH: Int = 8

# Compile-time size classes for fixed-size molecule batches
# All are multiples of 64 for AVX-512 alignment
alias SIZE_64: Int = 64       # Tiny fragments
alias SIZE_128: Int = 128     # Small molecules
alias SIZE_192: Int = 192     # Medium-small (3×64)
alias SIZE_256: Int = 256     # Medium molecules
alias SIZE_320: Int = 320     # Medium-large (5×64)
alias SIZE_512: Int = 512     # Large molecules
alias SIZE_1024: Int = 1024   # Very large molecules

# Storage type aliases for mixed precision
alias StorageF16 = DType.float16
alias StorageBF16 = DType.bfloat16
alias StorageF32 = DType.float32

# Utility functions
fn align_capacity(n: Int) -> Int:
    """Round up capacity to nearest multiple of 32 for alignment."""
    return ((n + 31) // 32) * 32

fn select_size_class(num_atoms: Int) -> Int:
    """Select smallest size bucket that fits the given atom count."""
    if num_atoms <= 64:
        return SIZE_64
    if num_atoms <= 128:
        return SIZE_128
    if num_atoms <= 192:
        return SIZE_192
    if num_atoms <= 256:
        return SIZE_256
    if num_atoms <= 320:
        return SIZE_320
    if num_atoms <= 512:
        return SIZE_512
    return SIZE_1024

# AtomBuffers
struct AtomBuffers[StorageType: DType = DType.float32](Copyable, Movable):
    """
    SoA storage for atom positions and gradients.

    Memory layout optimized for GPU coalesced access and CPU SIMD.
    Positions can be FP16 or FP32, gradients always FP32 for stability.
    """
    var x: UnsafePointer[Scalar[StorageType]]
    var y: UnsafePointer[Scalar[StorageType]]
    var z: UnsafePointer[Scalar[StorageType]]

    var grad_x: UnsafePointer[Float32]
    var grad_y: UnsafePointer[Float32]
    var grad_z: UnsafePointer[Float32]

    var atom_type: UnsafePointer[Int32]
    var charge: UnsafePointer[Scalar[StorageType]]
    var atom_starts: UnsafePointer[Int32]

    var capacity: Int
    var num_atoms: Int
    var num_molecules: Int

    fn __init__(out self, max_atoms: Int, max_molecules: Int):
        var cap = align_capacity(max_atoms)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_atoms = 0
        self.num_molecules = 0

        self.x = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.y = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.z = UnsafePointer[Scalar[StorageType]].alloc(cap)

        self.grad_x = UnsafePointer[Float32].alloc(cap)
        self.grad_y = UnsafePointer[Float32].alloc(cap)
        self.grad_z = UnsafePointer[Float32].alloc(cap)

        self.atom_type = UnsafePointer[Int32].alloc(cap)
        self.charge = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.atom_starts = UnsafePointer[Int32].alloc(mol_cap)

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

        self.x = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.y = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.z = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.grad_x = UnsafePointer[Float32].alloc(cap)
        self.grad_y = UnsafePointer[Float32].alloc(cap)
        self.grad_z = UnsafePointer[Float32].alloc(cap)
        self.atom_type = UnsafePointer[Int32].alloc(cap)
        self.charge = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.atom_starts = UnsafePointer[Int32].alloc(mol_cap)

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
        for i in range(self.num_atoms):
            self.grad_x[i] = 0.0
            self.grad_y[i] = 0.0
            self.grad_z[i] = 0.0

    fn set_atom(
        self,
        idx: Int,
        px: Scalar[StorageType], py: Scalar[StorageType], pz: Scalar[StorageType],
        atype: Int32,
        q: Scalar[StorageType]
    ):
        self.x[idx] = px
        self.y[idx] = py
        self.z[idx] = pz
        self.atom_type[idx] = atype
        self.charge[idx] = q

    fn get_molecule_atom_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        var start = Int(self.atom_starts[mol_idx])
        var end = Int(self.atom_starts[mol_idx + 1])
        return (start, end)

    fn get_num_atoms_in_molecule(self, mol_idx: Int) -> Int:
        return Int(self.atom_starts[mol_idx + 1]) - Int(self.atom_starts[mol_idx])

    fn load_position_f32(self, idx: Int) -> Tuple[Float32, Float32, Float32]:
        """Load position as Float32, converting from storage type if needed."""
        @parameter
        if StorageType == DType.float16:
            return (
                Float32(self.x[idx]),
                Float32(self.y[idx]),
                Float32(self.z[idx])
            )
        else:
            return (
                self.x[idx].cast[DType.float32](),
                self.y[idx].cast[DType.float32](),
                self.z[idx].cast[DType.float32]()
            )

    fn load_charge_f32(self, idx: Int) -> Float32:
        """Load charge as Float32, converting from storage type if needed."""
        @parameter
        if StorageType == DType.float16:
            return Float32(self.charge[idx])
        else:
            return self.charge[idx].cast[DType.float32]()

# BondBuffers
struct BondBuffers[StorageType: DType = DType.float32](Copyable, Movable):
    """
    SoA storage for bond topology and MMFF94 bond stretch parameters.
    Uses CSR-like indexing for efficient batch processing.
    """
    var atom_i: UnsafePointer[Int32]
    var atom_j: UnsafePointer[Int32]

    var force_constant: UnsafePointer[Scalar[StorageType]]
    var rest_length: UnsafePointer[Scalar[StorageType]]
    var bond_type: UnsafePointer[Int32]
    var bond_starts: UnsafePointer[Int32]

    var capacity: Int
    var num_bonds: Int
    var num_molecules: Int

    fn __init__(out self, max_bonds: Int, max_molecules: Int):
        var cap = align_capacity(max_bonds)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_bonds = 0
        self.num_molecules = 0

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.force_constant = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.rest_length = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.bond_type = UnsafePointer[Int32].alloc(cap)
        self.bond_starts = UnsafePointer[Int32].alloc(mol_cap)

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

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.force_constant = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.rest_length = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.bond_type = UnsafePointer[Int32].alloc(cap)
        self.bond_starts = UnsafePointer[Int32].alloc(mol_cap)

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
        kb: Scalar[StorageType], r0: Scalar[StorageType],
        btype: Int32
    ):
        self.atom_i[idx] = i
        self.atom_j[idx] = j
        self.force_constant[idx] = kb
        self.rest_length[idx] = r0
        self.bond_type[idx] = btype

    fn get_molecule_bond_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        var start = Int(self.bond_starts[mol_idx])
        var end = Int(self.bond_starts[mol_idx + 1])
        return (start, end)

    fn get_num_bonds_in_molecule(self, mol_idx: Int) -> Int:
        return Int(self.bond_starts[mol_idx + 1]) - Int(self.bond_starts[mol_idx])

    fn load_params_f32(self, idx: Int) -> Tuple[Float32, Float32]:
        """Load bond parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.force_constant[idx]), Float32(self.rest_length[idx]))
        else:
            return (
                self.force_constant[idx].cast[DType.float32](),
                self.rest_length[idx].cast[DType.float32]()
            )

# AngleBuffers
struct AngleBuffers[StorageType: DType = DType.float32](Copyable, Movable):
    """
    SoA storage for angle topology and MMFF94 angle bend parameters.
    Angle i-j-k where j is the central atom.
    """
    var atom_i: UnsafePointer[Int32]
    var atom_j: UnsafePointer[Int32]
    var atom_k: UnsafePointer[Int32]

    var force_constant: UnsafePointer[Scalar[StorageType]]
    var rest_angle: UnsafePointer[Scalar[StorageType]]
    var is_linear: UnsafePointer[Int32]
    var angle_starts: UnsafePointer[Int32]

    var capacity: Int
    var num_angles: Int
    var num_molecules: Int

    fn __init__(out self, max_angles: Int, max_molecules: Int):
        var cap = align_capacity(max_angles)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_angles = 0
        self.num_molecules = 0

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.atom_k = UnsafePointer[Int32].alloc(cap)
        self.force_constant = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.rest_angle = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.is_linear = UnsafePointer[Int32].alloc(cap)
        self.angle_starts = UnsafePointer[Int32].alloc(mol_cap)

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

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.atom_k = UnsafePointer[Int32].alloc(cap)
        self.force_constant = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.rest_angle = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.is_linear = UnsafePointer[Int32].alloc(cap)
        self.angle_starts = UnsafePointer[Int32].alloc(mol_cap)

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
        ka: Scalar[StorageType], theta0: Scalar[StorageType],
        linear: Int32
    ):
        self.atom_i[idx] = i
        self.atom_j[idx] = j
        self.atom_k[idx] = k
        self.force_constant[idx] = ka
        self.rest_angle[idx] = theta0
        self.is_linear[idx] = linear

    fn get_molecule_angle_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        var start = Int(self.angle_starts[mol_idx])
        var end = Int(self.angle_starts[mol_idx + 1])
        return (start, end)

    fn get_num_angles_in_molecule(self, mol_idx: Int) -> Int:
        return Int(self.angle_starts[mol_idx + 1]) - Int(self.angle_starts[mol_idx])

    fn load_params_f32(self, idx: Int) -> Tuple[Float32, Float32]:
        """Load angle parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.force_constant[idx]), Float32(self.rest_angle[idx]))
        else:
            return (
                self.force_constant[idx].cast[DType.float32](),
                self.rest_angle[idx].cast[DType.float32]()
            )

# TorsionBuffers
struct TorsionBuffers[StorageType: DType = DType.float32](Copyable, Movable):
    """
    SoA storage for torsion topology and MMFF94 torsion parameters.
    Torsion i-j-k-l where j-k is the central bond.
    E = 0.5 * (V1*(1+cos(phi)) + V2*(1-cos(2*phi)) + V3*(1+cos(3*phi)))
    """
    var atom_i: UnsafePointer[Int32]
    var atom_j: UnsafePointer[Int32]
    var atom_k: UnsafePointer[Int32]
    var atom_l: UnsafePointer[Int32]

    var v1: UnsafePointer[Scalar[StorageType]]
    var v2: UnsafePointer[Scalar[StorageType]]
    var v3: UnsafePointer[Scalar[StorageType]]
    var torsion_type: UnsafePointer[Int32]
    var torsion_starts: UnsafePointer[Int32]

    var capacity: Int
    var num_torsions: Int
    var num_molecules: Int

    fn __init__(out self, max_torsions: Int, max_molecules: Int):
        var cap = align_capacity(max_torsions)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_torsions = 0
        self.num_molecules = 0

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.atom_k = UnsafePointer[Int32].alloc(cap)
        self.atom_l = UnsafePointer[Int32].alloc(cap)
        self.v1 = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.v2 = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.v3 = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.torsion_type = UnsafePointer[Int32].alloc(cap)
        self.torsion_starts = UnsafePointer[Int32].alloc(mol_cap)

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

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.atom_k = UnsafePointer[Int32].alloc(cap)
        self.atom_l = UnsafePointer[Int32].alloc(cap)
        self.v1 = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.v2 = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.v3 = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.torsion_type = UnsafePointer[Int32].alloc(cap)
        self.torsion_starts = UnsafePointer[Int32].alloc(mol_cap)

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
        v1_val: Scalar[StorageType], v2_val: Scalar[StorageType], v3_val: Scalar[StorageType],
        ttype: Int32
    ):
        self.atom_i[idx] = i
        self.atom_j[idx] = j
        self.atom_k[idx] = k
        self.atom_l[idx] = l
        self.v1[idx] = v1_val
        self.v2[idx] = v2_val
        self.v3[idx] = v3_val
        self.torsion_type[idx] = ttype

    fn get_molecule_torsion_range(self, mol_idx: Int) -> Tuple[Int, Int]:
        var start = Int(self.torsion_starts[mol_idx])
        var end = Int(self.torsion_starts[mol_idx + 1])
        return (start, end)

    fn get_num_torsions_in_molecule(self, mol_idx: Int) -> Int:
        return Int(self.torsion_starts[mol_idx + 1]) - Int(self.torsion_starts[mol_idx])

    fn load_params_f32(self, idx: Int) -> Tuple[Float32, Float32, Float32]:
        """Load torsion parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.v1[idx]), Float32(self.v2[idx]), Float32(self.v3[idx]))
        else:
            return (
                self.v1[idx].cast[DType.float32](),
                self.v2[idx].cast[DType.float32](),
                self.v3[idx].cast[DType.float32]()
            )

# NonbondedPairs
struct NonbondedPairs[StorageType: DType = DType.float32](Copyable, Movable):
    """
    Storage for van der Waals and electrostatic interaction pairs.
    Excludes 1-2 (bonded) and 1-3 (angle) pairs, includes 1-4 with scaling.
    """
    var atom_i: UnsafePointer[Int32]
    var atom_j: UnsafePointer[Int32]
    var is_1_4: UnsafePointer[Int32]

    var epsilon: UnsafePointer[Scalar[StorageType]]
    var r_star: UnsafePointer[Scalar[StorageType]]
    var pair_starts: UnsafePointer[Int32]

    var capacity: Int
    var num_pairs: Int
    var num_molecules: Int

    fn __init__(out self, max_pairs: Int, max_molecules: Int):
        var cap = align_capacity(max_pairs)
        var mol_cap = align_capacity(max_molecules + 1)

        self.capacity = cap
        self.num_pairs = 0
        self.num_molecules = 0

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.is_1_4 = UnsafePointer[Int32].alloc(cap)
        self.epsilon = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.r_star = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.pair_starts = UnsafePointer[Int32].alloc(mol_cap)

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

        self.atom_i = UnsafePointer[Int32].alloc(cap)
        self.atom_j = UnsafePointer[Int32].alloc(cap)
        self.is_1_4 = UnsafePointer[Int32].alloc(cap)
        self.epsilon = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.r_star = UnsafePointer[Scalar[StorageType]].alloc(cap)
        self.pair_starts = UnsafePointer[Int32].alloc(mol_cap)

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

    fn load_params_f32(self, idx: Int) -> Tuple[Float32, Float32]:
        """Load vdW parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.epsilon[idx]), Float32(self.r_star[idx]))
        else:
            return (
                self.epsilon[idx].cast[DType.float32](),
                self.r_star[idx].cast[DType.float32]()
            )

# MoleculeBatch
struct MoleculeBatch[StorageType: DType = DType.float32](Copyable, Movable):
    """
    Container for a batch of molecules with all MMFF94 force field data.
    All data is stored in SoA format with CSR-like indexing.
    Positions/parameters use StorageType, gradients always Float32.
    """
    var x: UnsafePointer[Scalar[StorageType]]
    var y: UnsafePointer[Scalar[StorageType]]
    var z: UnsafePointer[Scalar[StorageType]]
    var grad_x: UnsafePointer[Float32]
    var grad_y: UnsafePointer[Float32]
    var grad_z: UnsafePointer[Float32]
    var atom_type: UnsafePointer[Int32]
    var charge: UnsafePointer[Scalar[StorageType]]
    var atom_starts: UnsafePointer[Int32]

    var bond_i: UnsafePointer[Int32]
    var bond_j: UnsafePointer[Int32]
    var bond_kb: UnsafePointer[Scalar[StorageType]]
    var bond_r0: UnsafePointer[Scalar[StorageType]]
    var bond_starts: UnsafePointer[Int32]

    var angle_i: UnsafePointer[Int32]
    var angle_j: UnsafePointer[Int32]
    var angle_k: UnsafePointer[Int32]
    var angle_ka: UnsafePointer[Scalar[StorageType]]
    var angle_theta0: UnsafePointer[Scalar[StorageType]]
    var angle_starts: UnsafePointer[Int32]

    var torsion_i: UnsafePointer[Int32]
    var torsion_j: UnsafePointer[Int32]
    var torsion_k: UnsafePointer[Int32]
    var torsion_l: UnsafePointer[Int32]
    var torsion_v1: UnsafePointer[Scalar[StorageType]]
    var torsion_v2: UnsafePointer[Scalar[StorageType]]
    var torsion_v3: UnsafePointer[Scalar[StorageType]]
    var torsion_starts: UnsafePointer[Int32]

    var nb_i: UnsafePointer[Int32]
    var nb_j: UnsafePointer[Int32]
    var nb_is_1_4: UnsafePointer[Int32]
    var nb_epsilon: UnsafePointer[Scalar[StorageType]]
    var nb_r_star: UnsafePointer[Scalar[StorageType]]
    var nb_starts: UnsafePointer[Int32]

    var atom_capacity: Int
    var bond_capacity: Int
    var angle_capacity: Int
    var torsion_capacity: Int
    var nb_capacity: Int
    var mol_capacity: Int

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

        self.x = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.y = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.z = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.grad_x = UnsafePointer[Float32].alloc(self.atom_capacity)
        self.grad_y = UnsafePointer[Float32].alloc(self.atom_capacity)
        self.grad_z = UnsafePointer[Float32].alloc(self.atom_capacity)
        self.atom_type = UnsafePointer[Int32].alloc(self.atom_capacity)
        self.charge = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.atom_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.bond_i = UnsafePointer[Int32].alloc(self.bond_capacity)
        self.bond_j = UnsafePointer[Int32].alloc(self.bond_capacity)
        self.bond_kb = UnsafePointer[Scalar[StorageType]].alloc(self.bond_capacity)
        self.bond_r0 = UnsafePointer[Scalar[StorageType]].alloc(self.bond_capacity)
        self.bond_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.angle_i = UnsafePointer[Int32].alloc(self.angle_capacity)
        self.angle_j = UnsafePointer[Int32].alloc(self.angle_capacity)
        self.angle_k = UnsafePointer[Int32].alloc(self.angle_capacity)
        self.angle_ka = UnsafePointer[Scalar[StorageType]].alloc(self.angle_capacity)
        self.angle_theta0 = UnsafePointer[Scalar[StorageType]].alloc(self.angle_capacity)
        self.angle_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.torsion_i = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_j = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_k = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_l = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_v1 = UnsafePointer[Scalar[StorageType]].alloc(self.torsion_capacity)
        self.torsion_v2 = UnsafePointer[Scalar[StorageType]].alloc(self.torsion_capacity)
        self.torsion_v3 = UnsafePointer[Scalar[StorageType]].alloc(self.torsion_capacity)
        self.torsion_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.nb_i = UnsafePointer[Int32].alloc(self.nb_capacity)
        self.nb_j = UnsafePointer[Int32].alloc(self.nb_capacity)
        self.nb_is_1_4 = UnsafePointer[Int32].alloc(self.nb_capacity)
        self.nb_epsilon = UnsafePointer[Scalar[StorageType]].alloc(self.nb_capacity)
        self.nb_r_star = UnsafePointer[Scalar[StorageType]].alloc(self.nb_capacity)
        self.nb_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

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

        self.x = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.y = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.z = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.grad_x = UnsafePointer[Float32].alloc(self.atom_capacity)
        self.grad_y = UnsafePointer[Float32].alloc(self.atom_capacity)
        self.grad_z = UnsafePointer[Float32].alloc(self.atom_capacity)
        self.atom_type = UnsafePointer[Int32].alloc(self.atom_capacity)
        self.charge = UnsafePointer[Scalar[StorageType]].alloc(self.atom_capacity)
        self.atom_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.bond_i = UnsafePointer[Int32].alloc(self.bond_capacity)
        self.bond_j = UnsafePointer[Int32].alloc(self.bond_capacity)
        self.bond_kb = UnsafePointer[Scalar[StorageType]].alloc(self.bond_capacity)
        self.bond_r0 = UnsafePointer[Scalar[StorageType]].alloc(self.bond_capacity)
        self.bond_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.angle_i = UnsafePointer[Int32].alloc(self.angle_capacity)
        self.angle_j = UnsafePointer[Int32].alloc(self.angle_capacity)
        self.angle_k = UnsafePointer[Int32].alloc(self.angle_capacity)
        self.angle_ka = UnsafePointer[Scalar[StorageType]].alloc(self.angle_capacity)
        self.angle_theta0 = UnsafePointer[Scalar[StorageType]].alloc(self.angle_capacity)
        self.angle_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.torsion_i = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_j = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_k = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_l = UnsafePointer[Int32].alloc(self.torsion_capacity)
        self.torsion_v1 = UnsafePointer[Scalar[StorageType]].alloc(self.torsion_capacity)
        self.torsion_v2 = UnsafePointer[Scalar[StorageType]].alloc(self.torsion_capacity)
        self.torsion_v3 = UnsafePointer[Scalar[StorageType]].alloc(self.torsion_capacity)
        self.torsion_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

        self.nb_i = UnsafePointer[Int32].alloc(self.nb_capacity)
        self.nb_j = UnsafePointer[Int32].alloc(self.nb_capacity)
        self.nb_is_1_4 = UnsafePointer[Int32].alloc(self.nb_capacity)
        self.nb_epsilon = UnsafePointer[Scalar[StorageType]].alloc(self.nb_capacity)
        self.nb_r_star = UnsafePointer[Scalar[StorageType]].alloc(self.nb_capacity)
        self.nb_starts = UnsafePointer[Int32].alloc(self.mol_capacity)

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
        for i in range(self.num_atoms):
            self.grad_x[i] = 0.0
            self.grad_y[i] = 0.0
            self.grad_z[i] = 0.0

    fn clear(mut self):
        self.num_atoms = 0
        self.num_bonds = 0
        self.num_angles = 0
        self.num_torsions = 0
        self.num_nb_pairs = 0
        self.num_molecules = 0

    fn load_position_f32(self, idx: Int) -> Tuple[Float32, Float32, Float32]:
        """Load position as Float32, converting from storage type if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.x[idx]), Float32(self.y[idx]), Float32(self.z[idx]))
        else:
            return (
                self.x[idx].cast[DType.float32](),
                self.y[idx].cast[DType.float32](),
                self.z[idx].cast[DType.float32]()
            )

    fn load_charge_f32(self, idx: Int) -> Float32:
        """Load charge as Float32, converting from storage type if needed."""
        @parameter
        if StorageType == DType.float16:
            return Float32(self.charge[idx])
        else:
            return self.charge[idx].cast[DType.float32]()

    fn load_bond_params_f32(self, idx: Int) -> Tuple[Float32, Float32]:
        """Load bond parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.bond_kb[idx]), Float32(self.bond_r0[idx]))
        else:
            return (
                self.bond_kb[idx].cast[DType.float32](),
                self.bond_r0[idx].cast[DType.float32]()
            )

    fn load_angle_params_f32(self, idx: Int) -> Tuple[Float32, Float32]:
        """Load angle parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.angle_ka[idx]), Float32(self.angle_theta0[idx]))
        else:
            return (
                self.angle_ka[idx].cast[DType.float32](),
                self.angle_theta0[idx].cast[DType.float32]()
            )

    fn load_torsion_params_f32(self, idx: Int) -> Tuple[Float32, Float32, Float32]:
        """Load torsion parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (
                Float32(self.torsion_v1[idx]),
                Float32(self.torsion_v2[idx]),
                Float32(self.torsion_v3[idx])
            )
        else:
            return (
                self.torsion_v1[idx].cast[DType.float32](),
                self.torsion_v2[idx].cast[DType.float32](),
                self.torsion_v3[idx].cast[DType.float32]()
            )

    fn load_nb_params_f32(self, idx: Int) -> Tuple[Float32, Float32]:
        """Load nonbonded parameters as Float32, converting if needed."""
        @parameter
        if StorageType == DType.float16:
            return (Float32(self.nb_epsilon[idx]), Float32(self.nb_r_star[idx]))
        else:
            return (
                self.nb_epsilon[idx].cast[DType.float32](),
                self.nb_r_star[idx].cast[DType.float32]()
            )
