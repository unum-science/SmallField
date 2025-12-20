"""
MMFF94 Energy Kernel for NVIDIA H100 GPU

Architecture:
- One warp (32 threads) per molecule
- FP16 compute, FP32 accumulation
- Register-only design with warp shuffles
- No shared memory required

Supported molecule sizes:
- SLOT_COUNT=2:  up to 64 atoms  (2 shuffle rounds)
- SLOT_COUNT=4:  up to 128 atoms (4 shuffle rounds)
- SLOT_COUNT=8:  up to 256 atoms (8 shuffle rounds)
- SLOT_COUNT=16: up to 512 atoms (16 shuffle rounds)

TODO: Future Optimizations
========================
1. SHUFFLE REDUCTION:
   - Pack coordinates more efficiently (x,y,z interleaved)
   - Use predicated shuffles to skip empty slots
   - Explore warp-specialized roles (loader vs computer threads)

2. MATRIX OPERATIONS:
   - Reformulate distance computation as matrix multiply: D² = X·X^T + Y·Y^T + Z·Z^T
   - Batch multiple small molecules into tensor core friendly sizes
   - Investigate FP16 WMMA for coordinate transforms

3. MEMORY ACCESS:
   - Prefetch next molecule's pair data during computation
   - Use texture cache for read-only parameters
   - Coalesce pair data into 128-byte cache lines

4. PRECISION:
   - Kahan summation for FP16 accumulation
   - Mixed precision: FP16 distance, FP32 energy terms
   - Error analysis for FP16 vs FP32 baseline

5. OCCUPANCY:
   - Profile register usage, tune SLOT_COUNT per SM
   - Consider 2 molecules per warp for small molecules
   - Persistent kernel design for better scheduling
"""

from memory import UnsafePointer, bitcast
from sys.intrinsics import llvm_intrinsic
from math import sqrt
from collections import InlineArray

# =============================================================================
# Constants
# =============================================================================

alias WARP_SIZE: Int = 32
alias FULL_MASK: UInt32 = 0xFFFFFFFF

# Default slot count - can process molecules up to SLOT_COUNT * 32 atoms
alias DEFAULT_SLOT_COUNT: Int = 4  # 128 atoms max


# =============================================================================
# Warp Shuffle Intrinsics (PTX via NVVM LLVM intrinsics)
# =============================================================================

@always_inline
fn shfl_sync_i32(mask: UInt32, val: Int32, src_lane: Int32) -> Int32:
    """
    Shuffle 32-bit integer value from src_lane to calling thread.
    All threads in mask must participate.

    PTX: shfl.sync.idx.b32
    """
    return llvm_intrinsic[
        "llvm.nvvm.shfl.sync.idx.i32",
        Int32,
        UInt32,   # mask
        Int32,    # value
        Int32,    # src_lane
        Int32,    # width (must be 32 for full warp)
    ](mask, val, src_lane, Int32(32))


@always_inline
fn shfl_sync_f32(mask: UInt32, val: Float32, src_lane: Int32) -> Float32:
    """
    Shuffle FP32 value from src_lane to calling thread.

    PTX: shfl.sync.idx.f32
    """
    return llvm_intrinsic[
        "llvm.nvvm.shfl.sync.idx.f32",
        Float32,
        UInt32,
        Float32,
        Int32,
        Int32,
    ](mask, val, src_lane, Int32(32))


@always_inline
fn shfl_xor_f32(mask: UInt32, val: Float32, lane_mask: Int32) -> Float32:
    """
    Butterfly shuffle: exchange with lane (lane_id XOR lane_mask).
    Used for efficient tree reduction.

    PTX: shfl.sync.bfly.f32
    """
    return llvm_intrinsic[
        "llvm.nvvm.shfl.sync.bfly.f32",
        Float32,
        UInt32,
        Float32,
        Int32,
        Int32,
    ](mask, val, lane_mask, Int32(32))


@always_inline
fn shfl_down_f32(mask: UInt32, val: Float32, delta: Int32) -> Float32:
    """
    Shuffle down: receive from lane (lane_id + delta).
    Alternative reduction pattern.

    PTX: shfl.sync.down.f32
    """
    return llvm_intrinsic[
        "llvm.nvvm.shfl.sync.down.f32",
        Float32,
        UInt32,
        Float32,
        Int32,
        Int32,
    ](mask, val, delta, Int32(32))


# =============================================================================
# FP16 Math Functions
# =============================================================================
#
# On H100 GPU, these map to native PTX instructions:
#   h_rsqrt -> rsqrt.approx.f16 (llvm.nvvm.rsqrt.approx.f16)
#   h_rcp   -> rcp.approx.f16   (llvm.nvvm.rcp.approx.f16)
#   h_fma   -> fma.rn.f16       (llvm.fma.f16)
#
# For CPU testing, we use f32 operations with f16 conversion.
# When targeting GPU with `mojo build --target=nvptx64`, switch to intrinsics.

@always_inline
fn h_rsqrt(x: Float16) -> Float16:
    """Fast reciprocal square root for FP16."""
    # GPU: llvm.nvvm.rsqrt.approx.f16
    var x32 = Float32(x)
    if x32 <= 0.0:
        return Float16(0.0)
    return Float16(1.0 / sqrt(x32))


@always_inline
fn h_rcp(x: Float16) -> Float16:
    """Fast reciprocal for FP16."""
    # GPU: llvm.nvvm.rcp.approx.f16
    return Float16(1.0 / Float32(x))


@always_inline
fn h_sqrt(x: Float16) -> Float16:
    """FP16 square root."""
    if x <= Float16(0.0):
        return Float16(0.0)
    return Float16(sqrt(Float32(x)))


@always_inline
fn h_fma(a: Float16, b: Float16, c: Float16) -> Float16:
    """Fused multiply-add: a*b + c."""
    # GPU: llvm.fma.f16
    return Float16(Float32(a) * Float32(b) + Float32(c))


# =============================================================================
# FP16 Packing Utilities
# =============================================================================

@always_inline
fn f16_to_bits(x: Float16) -> UInt16:
    """Reinterpret FP16 as unsigned 16-bit integer."""
    return bitcast[DType.uint16](x)


@always_inline
fn bits_to_f16(x: UInt16) -> Float16:
    """Reinterpret unsigned 16-bit integer as FP16."""
    return bitcast[DType.float16](x)


@always_inline
fn pack_f16x2(lo: Float16, hi: Float16) -> Int32:
    """
    Pack two FP16 values into one 32-bit integer for shuffle.
    lo goes in bits [0:15], hi in bits [16:31].
    """
    var lo_bits = Int32(f16_to_bits(lo))
    var hi_bits = Int32(f16_to_bits(hi))
    return lo_bits | (hi_bits << 16)


@always_inline
fn unpack_lo_f16(packed: Int32) -> Float16:
    """Extract low FP16 from packed int32."""
    return bits_to_f16(UInt16(packed & 0xFFFF))


@always_inline
fn unpack_hi_f16(packed: Int32) -> Float16:
    """Extract high FP16 from packed int32."""
    return bits_to_f16(UInt16((packed >> 16) & 0xFFFF))


# =============================================================================
# Warp Reduction
# =============================================================================

@always_inline
fn warp_reduce_sum(val: Float32) -> Float32:
    """
    Butterfly reduction: sum across all 32 warp lanes.
    After reduction, ALL lanes have the total sum.

    Uses 5 shuffle operations (log2(32) = 5).

    TODO: Explore tree reduction vs sequential for different workloads.
    """
    var sum = val
    sum += shfl_xor_f32(FULL_MASK, sum, Int32(16))
    sum += shfl_xor_f32(FULL_MASK, sum, Int32(8))
    sum += shfl_xor_f32(FULL_MASK, sum, Int32(4))
    sum += shfl_xor_f32(FULL_MASK, sum, Int32(2))
    sum += shfl_xor_f32(FULL_MASK, sum, Int32(1))
    return sum


@always_inline
fn warp_reduce_sum_lane0(val: Float32) -> Float32:
    """
    Sequential reduction: only lane 0 gets the final sum.
    Slightly more efficient if only one lane needs result.

    Uses 5 shuffle operations.
    """
    var sum = val
    sum += shfl_down_f32(FULL_MASK, sum, Int32(16))
    sum += shfl_down_f32(FULL_MASK, sum, Int32(8))
    sum += shfl_down_f32(FULL_MASK, sum, Int32(4))
    sum += shfl_down_f32(FULL_MASK, sum, Int32(2))
    sum += shfl_down_f32(FULL_MASK, sum, Int32(1))
    return sum


# =============================================================================
# Coordinate Cache (Distributed across warp)
# =============================================================================

struct WarpCoordCache[SLOT_COUNT: Int]:
    """
    Distributed coordinate storage across 32 warp lanes.

    Each lane holds SLOT_COUNT atoms' coordinates.
    Max molecule size: SLOT_COUNT * 32 atoms.

    Memory layout per lane k (for SLOT_COUNT=4):
      Atoms owned: k, k+32, k+64, k+96
      Storage: x[0..3], y[0..3], z[0..3] as FP16

    Shuffle cost per coordinate fetch: SLOT_COUNT operations.

    TODO: Reduce shuffle count by:
      - Packing x,y,z together and using fewer wider shuffles
      - Predicated execution to skip empty slots
      - Warp-divergent paths for common small molecules
    """
    var x: InlineArray[Float16, SLOT_COUNT]
    var y: InlineArray[Float16, SLOT_COUNT]
    var z: InlineArray[Float16, SLOT_COUNT]

    fn __init__(out self):
        """Initialize all coordinates to zero."""
        self.x = InlineArray[Float16, SLOT_COUNT](Float16(0))
        self.y = InlineArray[Float16, SLOT_COUNT](Float16(0))
        self.z = InlineArray[Float16, SLOT_COUNT](Float16(0))

    @always_inline
    fn load_from_global(
        mut self,
        coord_x: UnsafePointer[Float16],
        coord_y: UnsafePointer[Float16],
        coord_z: UnsafePointer[Float16],
        atom_base: Int,
        n_atoms: Int,
        lane: Int
    ):
        """
        Collaborative load: each lane loads its subset of atoms.
        Coalesced memory access pattern.
        """
        @parameter
        for slot in range(SLOT_COUNT):
            var local_atom = lane + slot * WARP_SIZE
            if local_atom < n_atoms:
                var global_idx = atom_base + local_atom
                self.x[slot] = coord_x[global_idx]
                self.y[slot] = coord_y[global_idx]
                self.z[slot] = coord_z[global_idx]
            # else: already zero-initialized

    @always_inline
    fn get_atom_coords(
        self,
        atom_idx: Int,
        lane: Int
    ) -> Tuple[Float16, Float16, Float16]:
        """
        Fetch coordinates for any atom via warp shuffle.

        Cost: 3 * SLOT_COUNT shuffle operations.

        TODO: Optimize shuffle count:
          - Pack x,y into one i32, shuffle once, then z separately (2*SLOT_COUNT)
          - Use ballot to skip empty slots
          - Specialize for common molecule sizes
        """
        var owner = atom_idx & (WARP_SIZE - 1)   # atom_idx % 32
        var slot = atom_idx >> 5                  # atom_idx // 32

        var result_x: Float16 = Float16(0)
        var result_y: Float16 = Float16(0)
        var result_z: Float16 = Float16(0)

        # Shuffle from each slot, select matching one
        # TODO: This generates SLOT_COUNT shuffles; optimize with predication
        @parameter
        for s in range(SLOT_COUNT):
            # Pack FP16 into i32 for shuffle (i32 shuffle is universal)
            var packed_x = pack_f16x2(self.x[s], Float16(0))
            var packed_y = pack_f16x2(self.y[s], Float16(0))
            var packed_z = pack_f16x2(self.z[s], Float16(0))

            var shuffled_x = shfl_sync_i32(FULL_MASK, packed_x, Int32(owner))
            var shuffled_y = shfl_sync_i32(FULL_MASK, packed_y, Int32(owner))
            var shuffled_z = shfl_sync_i32(FULL_MASK, packed_z, Int32(owner))

            if slot == s:
                result_x = unpack_lo_f16(shuffled_x)
                result_y = unpack_lo_f16(shuffled_y)
                result_z = unpack_lo_f16(shuffled_z)

        return (result_x, result_y, result_z)


# =============================================================================
# Energy Computation Functions
# =============================================================================

@always_inline
fn compute_distance_f16(
    xi: Float16, yi: Float16, zi: Float16,
    xj: Float16, yj: Float16, zj: Float16
) -> Float16:
    """
    Euclidean distance in FP16.

    Uses FMA for better precision: dx² + dy² + dz².

    TODO: Consider returning r² and computing sqrt lazily.
    """
    var dx = xi - xj
    var dy = yi - yj
    var dz = zi - zj

    # r² = dx² + dy² + dz² using FMA chain
    var r2 = dx * dx
    r2 = h_fma(dy, dy, r2)
    r2 = h_fma(dz, dz, r2)

    return h_sqrt(r2)


@always_inline
fn compute_vdw_energy_f16(
    r: Float16,
    epsilon: Float16,
    r_star: Float16
) -> Float32:
    """
    MMFF94 Buffered 14-7 van der Waals energy.

    E = ε * (1.07*R*/(R+0.07*R*))^7 * ((1.12*R*^7/(R^7+0.12*R*^7)) - 2)

    Computation in FP16, final multiply in FP32 for accumulation precision.

    TODO: Precompute R*^7 and 0.07*R*, 0.12*R*^7 in data prep.
    """
    # Avoid division by zero
    var r_safe = r if r > Float16(0.001) else Float16(0.001)

    # ρ = R / R*
    var inv_rstar = h_rcp(r_star)
    var rho = r_safe * inv_rstar

    # ρ^7 via squaring: ρ² → ρ⁴ → ρ⁶ → ρ⁷
    var rho2 = rho * rho
    var rho4 = rho2 * rho2
    var rho6 = rho4 * rho2
    var rho7 = rho6 * rho

    # Buffered repulsive term: (1.07 / (ρ + 0.07))^7
    alias C_107 = Float16(1.07)
    alias C_007 = Float16(0.07)
    var t1 = C_107 * h_rcp(rho + C_007)
    var t1_2 = t1 * t1
    var t1_4 = t1_2 * t1_2
    var t1_6 = t1_4 * t1_2
    var t1_7 = t1_6 * t1

    # Attractive term: (1.12 / (ρ^7 + 0.12)) - 2
    alias C_112 = Float16(1.12)
    alias C_012 = Float16(0.12)
    alias C_2 = Float16(2.0)
    var t2 = C_112 * h_rcp(rho7 + C_012) - C_2

    # Final energy in FP32 for accumulation
    return Float32(epsilon) * Float32(t1_7) * Float32(t2)


@always_inline
fn compute_elec_energy_f16(
    r: Float16,
    qi: Float16,
    qj: Float16,
    is_1_4: Bool
) -> Float32:
    """
    MMFF94 Buffered Coulomb electrostatic energy.

    E = 332.0716 * qi * qj / (D*R + δ)

    Where D=1.0 (vacuum dielectric), δ=0.05 Å (buffer).
    For 1-4 interactions: scale by 0.75.

    TODO: Consider distance-dependent dielectric for solvation.
    """
    alias COULOMB_CONST = Float16(332.0716)
    alias DELTA = Float16(0.05)
    alias SCALE_1_4 = Float16(0.75)

    var r_buffered = r + DELTA
    var inv_r = h_rcp(r_buffered)

    var e = COULOMB_CONST * qi * qj * inv_r

    if is_1_4:
        e = e * SCALE_1_4

    return Float32(e)


# =============================================================================
# Single-Molecule Energy Computation (CPU simulation of warp logic)
# =============================================================================

fn compute_molecule_energy[
    SLOT_COUNT: Int = DEFAULT_SLOT_COUNT
](
    # Coordinates for this molecule (FP16)
    coord_x: UnsafePointer[Float16],
    coord_y: UnsafePointer[Float16],
    coord_z: UnsafePointer[Float16],
    n_atoms: Int,

    # Non-bonded pair data for this molecule
    nb_i: UnsafePointer[UInt16],
    nb_j: UnsafePointer[UInt16],
    nb_epsilon: UnsafePointer[Float16],
    nb_r_star: UnsafePointer[Float16],
    nb_qi: UnsafePointer[Float16],
    nb_qj: UnsafePointer[Float16],
    nb_is_1_4: UnsafePointer[UInt8],
    n_pairs: Int
) -> Tuple[Float32, Float32]:
    """
    Compute MMFF94 energy for a single molecule.

    This is a CPU simulation of the GPU warp kernel logic.
    On GPU, this would be executed by 32 threads in parallel.

    Returns: (vdw_energy, elec_energy)
    """
    var max_atoms = SLOT_COUNT * WARP_SIZE
    if n_atoms > max_atoms:
        return (Float32(0.0), Float32(0.0))

    var vdw_total: Float32 = 0.0
    var elec_total: Float32 = 0.0

    # Process all pairs sequentially (on GPU this is parallel across lanes)
    for p in range(n_pairs):
        var i = Int(nb_i[p])
        var j = Int(nb_j[p])
        var eps = nb_epsilon[p]
        var rstar = nb_r_star[p]
        var qi = nb_qi[p]
        var qj = nb_qj[p]
        var is_1_4 = nb_is_1_4[p] != 0

        # Get coordinates
        var xi = coord_x[i]
        var yi = coord_y[i]
        var zi = coord_z[i]
        var xj = coord_x[j]
        var yj = coord_y[j]
        var zj = coord_z[j]

        # Compute distance
        var r = compute_distance_f16(xi, yi, zi, xj, yj, zj)

        # Accumulate energies
        vdw_total += compute_vdw_energy_f16(r, eps, rstar)
        elec_total += compute_elec_energy_f16(r, qi, qj, is_1_4)

    return (vdw_total, elec_total)


# =============================================================================
# GPU Kernel (for actual GPU execution)
# =============================================================================
#
# NOTE: The actual GPU kernel would use:
#   - @gpu.kernel decorator
#   - gpu.thread_idx, gpu.block_idx for thread identification
#   - Warp shuffles for coordinate access (as shown in WarpCoordCache)
#   - Warp reduction for energy summation
#   - GPU memory stores for output
#
# The compute_molecule_energy function above simulates the per-molecule
# computation that would happen in parallel across warps on GPU.


# =============================================================================
# Kernel Launch Helpers
# =============================================================================

fn compute_grid_size(num_molecules: Int, threads_per_block: Int) -> Int:
    """Compute number of GPU blocks needed for given molecule count."""
    var threads_needed = num_molecules * WARP_SIZE
    return (threads_needed + threads_per_block - 1) // threads_per_block


fn select_slot_count(max_atoms: Int) -> Int:
    """Select SLOT_COUNT based on largest molecule size."""
    if max_atoms <= 64:
        return 2
    elif max_atoms <= 128:
        return 4
    elif max_atoms <= 256:
        return 8
    elif max_atoms <= 512:
        return 16
    else:
        return -1  # too large for register-only approach


# =============================================================================
# Batch Compute (CPU fallback - uses FP16 precision matching GPU algorithm)
# =============================================================================

fn compute_vdw_energy_h100(
    coord_x: UnsafePointer[Float16],
    coord_y: UnsafePointer[Float16],
    coord_z: UnsafePointer[Float16],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    epsilon: UnsafePointer[Float16],
    r_star: UnsafePointer[Float16],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int,
) -> Float32:
    """
    Compute MMFF94 vdW energy for a single molecule using FP16 precision.

    This uses the same FP16 algorithm as the GPU kernel (CPU fallback).
    """
    var vdw_sum: Float32 = 0.0

    for p in range(nb_start, nb_end):
        var i = Int(nb_i[p])
        var j = Int(nb_j[p])

        # Get coordinates
        var xi = coord_x[i]
        var yi = coord_y[i]
        var zi = coord_z[i]
        var xj = coord_x[j]
        var yj = coord_y[j]
        var zj = coord_z[j]

        # Compute distance
        var r = compute_distance_f16(xi, yi, zi, xj, yj, zj)

        # Compute vdW energy
        vdw_sum += compute_vdw_energy_f16(r, epsilon[p], r_star[p])

    return vdw_sum


fn compute_electrostatic_energy_h100(
    coord_x: UnsafePointer[Float16],
    coord_y: UnsafePointer[Float16],
    coord_z: UnsafePointer[Float16],
    nb_i: UnsafePointer[Int32],
    nb_j: UnsafePointer[Int32],
    qi: UnsafePointer[Float16],
    qj: UnsafePointer[Float16],
    is_1_4: UnsafePointer[Int32],
    atom_start: Int,
    nb_start: Int,
    nb_end: Int,
) -> Float32:
    """
    Compute MMFF94 electrostatic energy for a single molecule using FP16 precision.

    This uses the same FP16 algorithm as the GPU kernel (CPU fallback).
    """
    var elec_sum: Float32 = 0.0

    for p in range(nb_start, nb_end):
        var i = Int(nb_i[p])
        var j = Int(nb_j[p])

        # Get coordinates
        var xi = coord_x[i]
        var yi = coord_y[i]
        var zi = coord_z[i]
        var xj = coord_x[j]
        var yj = coord_y[j]
        var zj = coord_z[j]

        # Compute distance
        var r = compute_distance_f16(xi, yi, zi, xj, yj, zj)

        # Compute electrostatic energy
        var is14 = is_1_4[p] != 0
        elec_sum += compute_elec_energy_f16(r, qi[p], qj[p], is14)

    return elec_sum


# =============================================================================
# GPU Kernel Implementation (TODO: Requires LayoutTensor for proper GPU types)
# =============================================================================
#
# NOTE: Actual GPU execution requires restructuring to use LayoutTensor and
# DevicePassable types. The current implementation provides:
# 1. Complete warp shuffle intrinsics (lines 64-133)
# 2. FP16 math functions (lines 147-176)
# 3. Warp reduction (lines 222-256)
# 4. CPU simulation of the warp-parallel algorithm
#
# To enable GPU execution:
# 1. Use LayoutTensor instead of UnsafePointer for kernel parameters
# 2. Add @gpu.kernel decorator (when available)
# 3. Use DeviceContext with proper buffer types
#
# The compute_vdw_energy_h100 and compute_electrostatic_energy_h100 functions
# above use the same FP16 algorithm that would run on GPU.


# =============================================================================
# Test: Simple 2-atom molecule (H2-like)
# =============================================================================

fn main():
    """Test MMFF94 H100 kernel with a simple 2-atom system."""
    print("MMFF94 H100 Kernel Test")
    print("=" * 40)

    # Create a simple 2-atom molecule
    # Atoms at (0,0,0) and (1,0,0) - 1 Angstrom apart
    var n_atoms = 2
    var n_pairs = 1

    # Coordinates (FP16)
    var x = List[Float16](Float16(0.0), Float16(1.0))
    var y = List[Float16](Float16(0.0), Float16(0.0))
    var z = List[Float16](Float16(0.0), Float16(0.0))

    # NB pair: atoms 0 and 1
    var nb_i = List[UInt16](UInt16(0))
    var nb_j = List[UInt16](UInt16(1))

    # Typical MMFF94 parameters for C-C like interaction
    var epsilon = List[Float16](Float16(0.05))   # kcal/mol
    var r_star = List[Float16](Float16(3.8))     # Angstrom
    var qi = List[Float16](Float16(0.1))         # partial charge
    var qj = List[Float16](Float16(-0.1))        # partial charge
    var is_1_4 = List[UInt8](UInt8(0))           # not a 1-4 interaction

    # Compute energy
    var result = compute_molecule_energy[SLOT_COUNT=2](
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        z.unsafe_ptr(),
        n_atoms,
        nb_i.unsafe_ptr(),
        nb_j.unsafe_ptr(),
        epsilon.unsafe_ptr(),
        r_star.unsafe_ptr(),
        qi.unsafe_ptr(),
        qj.unsafe_ptr(),
        is_1_4.unsafe_ptr(),
        n_pairs
    )

    var vdw = result[0]
    var elec = result[1]

    print("Distance: 1.0 Angstrom")
    print("VdW epsilon:", 0.05, "kcal/mol")
    print("VdW R*:", 3.8, "Angstrom")
    print("Charges: +0.1, -0.1")
    print()
    print("Results:")
    print("  VdW energy:", vdw, "kcal/mol")
    print("  Elec energy:", elec, "kcal/mol")
    print("  Total:", vdw + elec, "kcal/mol")
    print()

    # Test distance calculation
    var r = compute_distance_f16(
        Float16(0.0), Float16(0.0), Float16(0.0),
        Float16(1.0), Float16(0.0), Float16(0.0)
    )
    print("Distance check:", r, "(expected: 1.0)")

    # Verify VdW is repulsive at short range (r < r_star)
    if vdw > 0:
        print("VdW is repulsive (correct for r < r_star)")
    else:
        print("WARNING: VdW should be repulsive at r=1.0 < r_star=3.8")

    # Verify electrostatic is attractive (opposite charges)
    if elec < 0:
        print("Elec is attractive (correct for opposite charges)")
    else:
        print("WARNING: Elec should be attractive for +/- charges")

    print()
    print("Test complete.")
