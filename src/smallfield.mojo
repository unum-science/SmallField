# SmallField: Python Extension Module
#
# Unified Python binding exposing all CPU-specific backends:
# - skylake: AVX-512 f32 (Intel Skylake-X+, AMD Zen4+)
# - simd: Generic SIMD f32 (ARM NEON, AVX2)
# - genoa: AVX-512 bf16 storage (AMD Genoa, Intel Cooper Lake+)
# - turin: AVX-512 f16 storage (AMD Turin, Intel Sapphire Rapids+)
# - h100: FP16 GPU kernel (H100 with warp reduction)
# - h100_cpu: FP16 precision CPU fallback
#
# Python API (batch-only, zero-copy via numpy buffer protocol):
#   compute_batch(n_mol, n_atoms, n_pairs, kernel,
#                 coords,      # (N, 3) f64 - x, y, z
#                 pair_idx,    # (P, 2) i64 - i, j indices
#                 vdw_params,  # (P, 2) f64 - epsilon, r_star
#                 charges,     # (P, 2) f64 - qi, qj
#                 is_1_4,      # (P,) i64 - 1-4 flags
#                 offsets,     # (M+1, 2) i64 - atom_off, nb_off
#                 vdw_out,     # (M,) f64 - output vdW energies
#                 elec_out     # (M,) f64 - output elec energies
#   ) -> None

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from memory import UnsafePointer
from memory.unsafe_pointer import alloc

# Import CPU backend implementations
from mmff94_skylake import (
    compute_vdw_energy_skylake,
    compute_electrostatic_energy_skylake,
)
from mmff94_simd import (
    compute_vdw_energy_simd,
    compute_electrostatic_energy_simd,
)
from mmff94_genoa import (
    compute_vdw_energy_genoa,
    compute_electrostatic_energy_genoa,
)
from mmff94_turin import (
    compute_vdw_energy_turin,
    compute_electrostatic_energy_turin,
)
from mmff94_h100 import (
    compute_vdw_energy_h100,
    compute_electrostatic_energy_h100,
    compute_batch_gpu,
)


@always_inline
fn round_up_32(n: Int) -> Int:
    """Round up to next multiple of 32 for SIMD alignment."""
    return ((n + 31) // 32) * 32


fn compute_batch_impl(
    py_self: PythonObject, args: PythonObject
) raises -> PythonObject:
    """Compute vdW and electrostatic energies for a batch of molecules.

    Args (12 positional):
        n_mol: Number of molecules
        n_atoms: Total atoms across all molecules
        n_pairs: Total nonbonded pairs
        kernel: "skylake" | "simd" | "genoa" | "turin"
        coords: (N, 3) f64 - atomic coordinates [x, y, z]
        pair_idx: (P, 2) i64 - pair indices [i, j]
        vdw_params: (P, 2) f64 - vdW parameters [epsilon, r_star]
        charges: (P, 2) f64 - partial charges [qi, qj]
        is_1_4: (P,) i64 - 1-4 interaction flags
        offsets: (M+1, 2) i64 - offsets [atom_off, nb_off]
        vdw_out: (M,) f64 - output vdW energies (mutable)
        elec_out: (M,) f64 - output electrostatic energies (mutable)

    Results written directly to vdw_out and elec_out arrays.
    """
    # Extract arguments from tuple
    var n_mol = Int(args[0])
    var n_atoms = Int(args[1])
    var n_pairs = Int(args[2])
    var kern = String(args[3])

    var coords = args[4]       # (N, 3) f64
    var pair_idx = args[5]     # (P, 2) i64
    var vdw_params = args[6]   # (P, 2) f64
    var charges = args[7]      # (P, 2) f64
    var is_1_4 = args[8]       # (P,) i64
    var offsets = args[9]      # (M+1, 2) i64
    var vdw_out = args[10]     # (M,) f64 output
    var elec_out = args[11]    # (M,) f64 output

    # Get direct pointers to numpy arrays (zero-copy!)
    # 2D arrays are row-major: coords[i, j] = ptr[i * ncols + j]
    var coords_ptr = coords.ctypes.data.unsafe_get_as_pointer[DType.float64]()
    var pair_idx_ptr = pair_idx.ctypes.data.unsafe_get_as_pointer[DType.int64]()
    var vdw_params_ptr = vdw_params.ctypes.data.unsafe_get_as_pointer[DType.float64]()
    var charges_ptr = charges.ctypes.data.unsafe_get_as_pointer[DType.float64]()
    var is14_ptr = is_1_4.ctypes.data.unsafe_get_as_pointer[DType.int64]()
    var offsets_ptr = offsets.ctypes.data.unsafe_get_as_pointer[DType.int64]()

    # Output arrays (mutable - we write results here)
    var vdw_out_ptr = vdw_out.ctypes.data.unsafe_get_as_pointer[DType.float64]()
    var elec_out_ptr = elec_out.ctypes.data.unsafe_get_as_pointer[DType.float64]()

    # Allocate aligned buffers for type conversion
    var alloc_atoms = round_up_32(n_atoms)
    var alloc_pairs = round_up_32(n_pairs)

    # Convert indices to Int32 (kernels expect i32)
    # pair_idx is (P, 2): row i has [nb_i, nb_j] at offsets [i*2, i*2+1]
    var buf_ni = alloc[Int32](alloc_pairs)
    var buf_nj = alloc[Int32](alloc_pairs)
    var buf_14 = alloc[Int32](alloc_pairs)

    for i in range(n_pairs):
        buf_ni[i] = Int32(Int(pair_idx_ptr[i * 2]))      # pair_idx[i, 0]
        buf_nj[i] = Int32(Int(pair_idx_ptr[i * 2 + 1]))  # pair_idx[i, 1]
        buf_14[i] = Int32(Int(is14_ptr[i]))

    if kern == "skylake":
        var buf_x = alloc[Float32](alloc_atoms)
        var buf_y = alloc[Float32](alloc_atoms)
        var buf_z = alloc[Float32](alloc_atoms)
        var buf_eps = alloc[Float32](alloc_pairs)
        var buf_rstar = alloc[Float32](alloc_pairs)
        var buf_qi = alloc[Float32](alloc_pairs)
        var buf_qj = alloc[Float32](alloc_pairs)

        # Convert f64 -> f32 using direct pointer access
        # coords is (N, 3): row i has [x, y, z] at offsets [i*3, i*3+1, i*3+2]
        for i in range(n_atoms):
            buf_x[i] = Float32(coords_ptr[i * 3])
            buf_y[i] = Float32(coords_ptr[i * 3 + 1])
            buf_z[i] = Float32(coords_ptr[i * 3 + 2])
        # vdw_params is (P, 2): [epsilon, r_star], charges is (P, 2): [qi, qj]
        for i in range(n_pairs):
            buf_eps[i] = Float32(vdw_params_ptr[i * 2])
            buf_rstar[i] = Float32(vdw_params_ptr[i * 2 + 1])
            buf_qi[i] = Float32(charges_ptr[i * 2])
            buf_qj[i] = Float32(charges_ptr[i * 2 + 1])

        # offsets is (M+1, 2): [atom_off, nb_off]
        for mol_idx in range(n_mol):
            var atom_start = Int(offsets_ptr[mol_idx * 2])
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])

            var vdw = compute_vdw_energy_skylake(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_eps, buf_rstar, atom_start, nb_start, nb_end
            )
            var elec = compute_electrostatic_energy_skylake(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_qi, buf_qj, buf_14, atom_start, nb_start, nb_end
            )
            # Write directly to output numpy arrays
            vdw_out_ptr[mol_idx] = Float64(vdw)
            elec_out_ptr[mol_idx] = Float64(elec)

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "simd":
        var buf_x = alloc[Float32](alloc_atoms)
        var buf_y = alloc[Float32](alloc_atoms)
        var buf_z = alloc[Float32](alloc_atoms)
        var buf_eps = alloc[Float32](alloc_pairs)
        var buf_rstar = alloc[Float32](alloc_pairs)
        var buf_qi = alloc[Float32](alloc_pairs)
        var buf_qj = alloc[Float32](alloc_pairs)

        # Convert f64 -> f32 from grouped matrices
        for i in range(n_atoms):
            buf_x[i] = Float32(coords_ptr[i * 3])
            buf_y[i] = Float32(coords_ptr[i * 3 + 1])
            buf_z[i] = Float32(coords_ptr[i * 3 + 2])
        for i in range(n_pairs):
            buf_eps[i] = Float32(vdw_params_ptr[i * 2])
            buf_rstar[i] = Float32(vdw_params_ptr[i * 2 + 1])
            buf_qi[i] = Float32(charges_ptr[i * 2])
            buf_qj[i] = Float32(charges_ptr[i * 2 + 1])

        for mol_idx in range(n_mol):
            var atom_start = Int(offsets_ptr[mol_idx * 2])
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])

            var vdw = compute_vdw_energy_simd[DType.float32, 8](
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_eps, buf_rstar, atom_start, nb_start, nb_end
            )
            var elec = compute_electrostatic_energy_simd[DType.float32, 8](
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_qi, buf_qj, buf_14, atom_start, nb_start, nb_end
            )
            # Write directly to output numpy arrays
            vdw_out_ptr[mol_idx] = Float64(vdw)
            elec_out_ptr[mol_idx] = Float64(elec)

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "simd_f64":
        # Full f64 precision SIMD kernel (baseline accuracy reference)
        var buf_x = alloc[Float64](alloc_atoms)
        var buf_y = alloc[Float64](alloc_atoms)
        var buf_z = alloc[Float64](alloc_atoms)
        var buf_eps = alloc[Float64](alloc_pairs)
        var buf_rstar = alloc[Float64](alloc_pairs)
        var buf_qi = alloc[Float64](alloc_pairs)
        var buf_qj = alloc[Float64](alloc_pairs)

        # Direct copy from numpy (no conversion needed - already f64)
        for i in range(n_atoms):
            buf_x[i] = coords_ptr[i * 3]
            buf_y[i] = coords_ptr[i * 3 + 1]
            buf_z[i] = coords_ptr[i * 3 + 2]
        for i in range(n_pairs):
            buf_eps[i] = vdw_params_ptr[i * 2]
            buf_rstar[i] = vdw_params_ptr[i * 2 + 1]
            buf_qi[i] = charges_ptr[i * 2]
            buf_qj[i] = charges_ptr[i * 2 + 1]

        for mol_idx in range(n_mol):
            var atom_start = Int(offsets_ptr[mol_idx * 2])
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])
            var vdw = compute_vdw_energy_simd[DType.float64, 4](
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_eps, buf_rstar, atom_start, nb_start, nb_end
            )
            var elec = compute_electrostatic_energy_simd[DType.float64, 4](
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_qi, buf_qj, buf_14, atom_start, nb_start, nb_end
            )
            # Results are already f64
            vdw_out_ptr[mol_idx] = Float64(vdw)
            elec_out_ptr[mol_idx] = Float64(elec)

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "genoa":
        var buf_x = alloc[BFloat16](alloc_atoms)
        var buf_y = alloc[BFloat16](alloc_atoms)
        var buf_z = alloc[BFloat16](alloc_atoms)
        var buf_eps = alloc[BFloat16](alloc_pairs)
        var buf_rstar = alloc[BFloat16](alloc_pairs)
        var buf_qi = alloc[BFloat16](alloc_pairs)
        var buf_qj = alloc[BFloat16](alloc_pairs)

        # Convert f64 -> bf16 from grouped matrices
        for i in range(n_atoms):
            buf_x[i] = BFloat16(coords_ptr[i * 3])
            buf_y[i] = BFloat16(coords_ptr[i * 3 + 1])
            buf_z[i] = BFloat16(coords_ptr[i * 3 + 2])
        for i in range(n_pairs):
            buf_eps[i] = BFloat16(vdw_params_ptr[i * 2])
            buf_rstar[i] = BFloat16(vdw_params_ptr[i * 2 + 1])
            buf_qi[i] = BFloat16(charges_ptr[i * 2])
            buf_qj[i] = BFloat16(charges_ptr[i * 2 + 1])

        for mol_idx in range(n_mol):
            var atom_start = Int(offsets_ptr[mol_idx * 2])
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])

            var vdw = compute_vdw_energy_genoa(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_eps, buf_rstar, atom_start, nb_start, nb_end
            )
            var elec = compute_electrostatic_energy_genoa(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_qi, buf_qj, buf_14, atom_start, nb_start, nb_end
            )
            # Write directly to output numpy arrays
            vdw_out_ptr[mol_idx] = Float64(vdw)
            elec_out_ptr[mol_idx] = Float64(elec)

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "turin":
        var buf_x = alloc[Float16](alloc_atoms)
        var buf_y = alloc[Float16](alloc_atoms)
        var buf_z = alloc[Float16](alloc_atoms)
        var buf_eps = alloc[Float16](alloc_pairs)
        var buf_rstar = alloc[Float16](alloc_pairs)
        var buf_qi = alloc[Float16](alloc_pairs)
        var buf_qj = alloc[Float16](alloc_pairs)

        # Convert f64 -> f16 from grouped matrices
        for i in range(n_atoms):
            buf_x[i] = Float16(coords_ptr[i * 3])
            buf_y[i] = Float16(coords_ptr[i * 3 + 1])
            buf_z[i] = Float16(coords_ptr[i * 3 + 2])
        for i in range(n_pairs):
            buf_eps[i] = Float16(vdw_params_ptr[i * 2])
            buf_rstar[i] = Float16(vdw_params_ptr[i * 2 + 1])
            buf_qi[i] = Float16(charges_ptr[i * 2])
            buf_qj[i] = Float16(charges_ptr[i * 2 + 1])

        for mol_idx in range(n_mol):
            var atom_start = Int(offsets_ptr[mol_idx * 2])
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])

            var vdw = compute_vdw_energy_turin(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_eps, buf_rstar, atom_start, nb_start, nb_end
            )
            var elec = compute_electrostatic_energy_turin(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_qi, buf_qj, buf_14, atom_start, nb_start, nb_end
            )
            # Write directly to output numpy arrays
            vdw_out_ptr[mol_idx] = Float64(vdw)
            elec_out_ptr[mol_idx] = Float64(elec)

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "h100_cpu":
        # H100 algorithm CPU fallback (uses FP16 precision)
        var buf_x = alloc[Float16](alloc_atoms)
        var buf_y = alloc[Float16](alloc_atoms)
        var buf_z = alloc[Float16](alloc_atoms)
        var buf_eps = alloc[Float16](alloc_pairs)
        var buf_rstar = alloc[Float16](alloc_pairs)
        var buf_qi = alloc[Float16](alloc_pairs)
        var buf_qj = alloc[Float16](alloc_pairs)

        # Convert f64 -> f16 from grouped matrices
        for i in range(n_atoms):
            buf_x[i] = Float16(coords_ptr[i * 3])
            buf_y[i] = Float16(coords_ptr[i * 3 + 1])
            buf_z[i] = Float16(coords_ptr[i * 3 + 2])
        for i in range(n_pairs):
            buf_eps[i] = Float16(vdw_params_ptr[i * 2])
            buf_rstar[i] = Float16(vdw_params_ptr[i * 2 + 1])
            buf_qi[i] = Float16(charges_ptr[i * 2])
            buf_qj[i] = Float16(charges_ptr[i * 2 + 1])

        for mol_idx in range(n_mol):
            var atom_start = Int(offsets_ptr[mol_idx * 2])
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])

            var vdw = compute_vdw_energy_h100(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_eps, buf_rstar, atom_start, nb_start, nb_end
            )
            var elec = compute_electrostatic_energy_h100(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_qi, buf_qj, buf_14, atom_start, nb_start, nb_end
            )
            # Write directly to output numpy arrays
            vdw_out_ptr[mol_idx] = Float64(vdw)
            elec_out_ptr[mol_idx] = Float64(elec)

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "h100":
        # H100 GPU kernel - warp-parallel FP16 with reduction
        var buf_x = alloc[Float16](alloc_atoms)
        var buf_y = alloc[Float16](alloc_atoms)
        var buf_z = alloc[Float16](alloc_atoms)
        var buf_eps = alloc[Float16](alloc_pairs)
        var buf_rstar = alloc[Float16](alloc_pairs)
        var buf_qi = alloc[Float16](alloc_pairs)
        var buf_qj = alloc[Float16](alloc_pairs)
        var buf_nb_start = alloc[Int32](n_mol)
        var buf_nb_end = alloc[Int32](n_mol)
        var buf_vdw_out = alloc[Float32](n_mol)
        var buf_elec_out = alloc[Float32](n_mol)

        # Convert f64 -> f16 from grouped matrices
        for i in range(n_atoms):
            buf_x[i] = Float16(coords_ptr[i * 3])
            buf_y[i] = Float16(coords_ptr[i * 3 + 1])
            buf_z[i] = Float16(coords_ptr[i * 3 + 2])
        for i in range(n_pairs):
            buf_eps[i] = Float16(vdw_params_ptr[i * 2])
            buf_rstar[i] = Float16(vdw_params_ptr[i * 2 + 1])
            buf_qi[i] = Float16(charges_ptr[i * 2])
            buf_qj[i] = Float16(charges_ptr[i * 2 + 1])

        # Compute nb_start and nb_end for each molecule
        for mol_idx in range(n_mol):
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])
            buf_nb_start[mol_idx] = Int32(nb_start)
            buf_nb_end[mol_idx] = Int32(nb_end)

        # Launch GPU kernel batch
        compute_batch_gpu(
            n_mol, n_atoms, n_pairs,
            buf_x, buf_y, buf_z,
            buf_ni, buf_nj,
            buf_eps, buf_rstar,
            buf_qi, buf_qj,
            buf_14,
            buf_nb_start, buf_nb_end,
            buf_vdw_out, buf_elec_out
        )

        # Copy results to output arrays
        for mol_idx in range(n_mol):
            vdw_out_ptr[mol_idx] = Float64(buf_vdw_out[mol_idx])
            elec_out_ptr[mol_idx] = Float64(buf_elec_out[mol_idx])

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()
        buf_nb_start.free()
        buf_nb_end.free()
        buf_vdw_out.free()
        buf_elec_out.free()

    else:
        # Default to skylake
        var buf_x = alloc[Float32](alloc_atoms)
        var buf_y = alloc[Float32](alloc_atoms)
        var buf_z = alloc[Float32](alloc_atoms)
        var buf_eps = alloc[Float32](alloc_pairs)
        var buf_rstar = alloc[Float32](alloc_pairs)
        var buf_qi = alloc[Float32](alloc_pairs)
        var buf_qj = alloc[Float32](alloc_pairs)

        # Convert f64 -> f32 from grouped matrices
        for i in range(n_atoms):
            buf_x[i] = Float32(coords_ptr[i * 3])
            buf_y[i] = Float32(coords_ptr[i * 3 + 1])
            buf_z[i] = Float32(coords_ptr[i * 3 + 2])
        for i in range(n_pairs):
            buf_eps[i] = Float32(vdw_params_ptr[i * 2])
            buf_rstar[i] = Float32(vdw_params_ptr[i * 2 + 1])
            buf_qi[i] = Float32(charges_ptr[i * 2])
            buf_qj[i] = Float32(charges_ptr[i * 2 + 1])

        for mol_idx in range(n_mol):
            var atom_start = Int(offsets_ptr[mol_idx * 2])
            var nb_start = Int(offsets_ptr[mol_idx * 2 + 1])
            var nb_end = Int(offsets_ptr[(mol_idx + 1) * 2 + 1])

            var vdw = compute_vdw_energy_skylake(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_eps, buf_rstar, atom_start, nb_start, nb_end
            )
            var elec = compute_electrostatic_energy_skylake(
                buf_x, buf_y, buf_z, buf_ni, buf_nj,
                buf_qi, buf_qj, buf_14, atom_start, nb_start, nb_end
            )
            # Write directly to output numpy arrays
            vdw_out_ptr[mol_idx] = Float64(vdw)
            elec_out_ptr[mol_idx] = Float64(elec)

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()
        buf_qi.free()
        buf_qj.free()

    buf_ni.free()
    buf_nj.free()
    buf_14.free()

    # Results already written to vdw_out and elec_out numpy arrays
    return Python.none()


@export
fn PyInit_smallfield() -> PythonObject:
    """Initialize the smallfield Python module."""
    try:
        var m = PythonModuleBuilder("smallfield")
        # Use def_py_function for functions with >6 args
        m.def_py_function[compute_batch_impl]("compute_batch")
        return m.finalize()
    except e:
        print("Error creating smallfield module:", e)
        return PythonObject()
