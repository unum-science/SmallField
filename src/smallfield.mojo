# SmallField: Python Extension Module
#
# Unified Python binding exposing all CPU-specific backends:
# - skylake: AVX-512 f32 (Intel Skylake-X+, AMD Zen4+)
# - simd: Generic SIMD f32 (ARM NEON, AVX2)
# - genoa: AVX-512 bf16 storage (AMD Genoa, Intel Cooper Lake+)
# - turin: AVX-512 f16 storage (AMD Turin, Intel Sapphire Rapids+)
#
# Python API:
#   vdw_energy(x, y, z, nb_i, nb_j, epsilon, r_star, n, kernel="skylake")
#   elec_energy(x, y, z, nb_i, nb_j, qi, qj, is_1_4, n, kernel="skylake")
#
# Input/Output: f64 numpy arrays (standard Python floats)
# Internal: Converts to f16/bf16/f32 based on kernel selection

from python import Python, PythonObject
from memory import UnsafePointer
from memory.unsafe_pointer import alloc

# =============================================================================
# SIMD-friendly allocation: Round up to multiples of 32
# =============================================================================

@always_inline
fn round_up_32(n: Int) -> Int:
    """Round up to next multiple of 32 for SIMD alignment."""
    return ((n + 31) // 32) * 32


# Import ALL backend implementations
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


# =============================================================================
# Van der Waals Energy - Unified Interface
# =============================================================================

fn vdw_energy_py(
    x: PythonObject,
    y: PythonObject,
    z: PythonObject,
    nb_i: PythonObject,
    nb_j: PythonObject,
    epsilon: PythonObject,
    r_star: PythonObject,
    num_pairs: PythonObject,
    num_atoms: PythonObject,
    kernel: PythonObject
) raises -> PythonObject:
    """Compute vdW energy with kernel dispatch."""
    var n_pairs = Int(num_pairs)
    var n_atoms = Int(num_atoms)
    var kern = String(kernel)

    # Round up to SIMD-friendly sizes
    var alloc_atoms = round_up_32(n_atoms)
    var alloc_pairs = round_up_32(n_pairs)

    var energy: Float32 = 0.0

    # Allocate index buffers (always i32)
    var buf_ni = alloc[Int32](alloc_pairs)
    var buf_nj = alloc[Int32](alloc_pairs)
    for i in range(n_pairs):
        buf_ni[i] = Int32(Int(nb_i[i]))
        buf_nj[i] = Int32(Int(nb_j[i]))

    if kern == "skylake" or kern == "simd":
        # f32 storage
        var buf_x = alloc[Float32](alloc_atoms)
        var buf_y = alloc[Float32](alloc_atoms)
        var buf_z = alloc[Float32](alloc_atoms)
        var buf_eps = alloc[Float32](alloc_pairs)
        var buf_rstar = alloc[Float32](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = Float32(Float64(x[i]))
            buf_y[i] = Float32(Float64(y[i]))
            buf_z[i] = Float32(Float64(z[i]))
        for i in range(n_pairs):
            buf_eps[i] = Float32(Float64(epsilon[i]))
            buf_rstar[i] = Float32(Float64(r_star[i]))

        if kern == "skylake":
            energy = compute_vdw_energy_skylake(
                buf_x, buf_y, buf_z,
                buf_ni, buf_nj,
                buf_eps, buf_rstar,
                0, 0, n_pairs
            )
        else:
            energy = compute_vdw_energy_simd[DType.float32, 8](
                buf_x, buf_y, buf_z,
                buf_ni, buf_nj,
                buf_eps, buf_rstar,
                0, 0, n_pairs
            )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()

    elif kern == "genoa":
        # bf16 storage
        var buf_x = alloc[BFloat16](alloc_atoms)
        var buf_y = alloc[BFloat16](alloc_atoms)
        var buf_z = alloc[BFloat16](alloc_atoms)
        var buf_eps = alloc[BFloat16](alloc_pairs)
        var buf_rstar = alloc[BFloat16](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = BFloat16(Float64(x[i]))
            buf_y[i] = BFloat16(Float64(y[i]))
            buf_z[i] = BFloat16(Float64(z[i]))
        for i in range(n_pairs):
            buf_eps[i] = BFloat16(Float64(epsilon[i]))
            buf_rstar[i] = BFloat16(Float64(r_star[i]))

        energy = compute_vdw_energy_genoa(
            buf_x, buf_y, buf_z,
            buf_ni, buf_nj,
            buf_eps, buf_rstar,
            0, 0, n_pairs
        )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()

    elif kern == "turin":
        # f16 storage
        var buf_x = alloc[Float16](alloc_atoms)
        var buf_y = alloc[Float16](alloc_atoms)
        var buf_z = alloc[Float16](alloc_atoms)
        var buf_eps = alloc[Float16](alloc_pairs)
        var buf_rstar = alloc[Float16](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = Float16(Float64(x[i]))
            buf_y[i] = Float16(Float64(y[i]))
            buf_z[i] = Float16(Float64(z[i]))
        for i in range(n_pairs):
            buf_eps[i] = Float16(Float64(epsilon[i]))
            buf_rstar[i] = Float16(Float64(r_star[i]))

        energy = compute_vdw_energy_turin(
            buf_x, buf_y, buf_z,
            buf_ni, buf_nj,
            buf_eps, buf_rstar,
            0, 0, n_pairs
        )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()

    else:
        # Default to skylake
        var buf_x = alloc[Float32](alloc_atoms)
        var buf_y = alloc[Float32](alloc_atoms)
        var buf_z = alloc[Float32](alloc_atoms)
        var buf_eps = alloc[Float32](alloc_pairs)
        var buf_rstar = alloc[Float32](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = Float32(Float64(x[i]))
            buf_y[i] = Float32(Float64(y[i]))
            buf_z[i] = Float32(Float64(z[i]))
        for i in range(n_pairs):
            buf_eps[i] = Float32(Float64(epsilon[i]))
            buf_rstar[i] = Float32(Float64(r_star[i]))

        energy = compute_vdw_energy_skylake(
            buf_x, buf_y, buf_z,
            buf_ni, buf_nj,
            buf_eps, buf_rstar,
            0, 0, n_pairs
        )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_eps.free()
        buf_rstar.free()

    buf_ni.free()
    buf_nj.free()

    # Return as Python float (f64)
    var builtins = Python.import_module("builtins")
    return builtins.float(Float64(energy))


# =============================================================================
# Electrostatic Energy - Unified Interface
# =============================================================================

fn elec_energy_py(
    x: PythonObject,
    y: PythonObject,
    z: PythonObject,
    nb_i: PythonObject,
    nb_j: PythonObject,
    qi: PythonObject,
    qj: PythonObject,
    is_1_4: PythonObject,
    num_pairs: PythonObject,
    num_atoms: PythonObject,
    kernel: PythonObject
) raises -> PythonObject:
    """Compute electrostatic energy with kernel dispatch."""
    var n_pairs = Int(num_pairs)
    var n_atoms = Int(num_atoms)
    var kern = String(kernel)

    # Round up to SIMD-friendly sizes
    var alloc_atoms = round_up_32(n_atoms)
    var alloc_pairs = round_up_32(n_pairs)

    var energy: Float32 = 0.0

    # Allocate index buffers (always i32)
    var buf_ni = alloc[Int32](alloc_pairs)
    var buf_nj = alloc[Int32](alloc_pairs)
    var buf_14 = alloc[Int32](alloc_pairs)
    for i in range(n_pairs):
        buf_ni[i] = Int32(Int(nb_i[i]))
        buf_nj[i] = Int32(Int(nb_j[i]))
        buf_14[i] = Int32(Int(is_1_4[i]))

    if kern == "skylake" or kern == "simd":
        # f32 storage
        var buf_x = alloc[Float32](alloc_atoms)
        var buf_y = alloc[Float32](alloc_atoms)
        var buf_z = alloc[Float32](alloc_atoms)
        var buf_qi = alloc[Float32](alloc_pairs)
        var buf_qj = alloc[Float32](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = Float32(Float64(x[i]))
            buf_y[i] = Float32(Float64(y[i]))
            buf_z[i] = Float32(Float64(z[i]))
        for i in range(n_pairs):
            buf_qi[i] = Float32(Float64(qi[i]))
            buf_qj[i] = Float32(Float64(qj[i]))

        if kern == "skylake":
            energy = compute_electrostatic_energy_skylake(
                buf_x, buf_y, buf_z,
                buf_ni, buf_nj,
                buf_qi, buf_qj,
                buf_14,
                0, 0, n_pairs
            )
        else:
            energy = compute_electrostatic_energy_simd[DType.float32, 8](
                buf_x, buf_y, buf_z,
                buf_ni, buf_nj,
                buf_qi, buf_qj,
                buf_14,
                0, 0, n_pairs
            )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "genoa":
        # bf16 storage
        var buf_x = alloc[BFloat16](alloc_atoms)
        var buf_y = alloc[BFloat16](alloc_atoms)
        var buf_z = alloc[BFloat16](alloc_atoms)
        var buf_qi = alloc[BFloat16](alloc_pairs)
        var buf_qj = alloc[BFloat16](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = BFloat16(Float64(x[i]))
            buf_y[i] = BFloat16(Float64(y[i]))
            buf_z[i] = BFloat16(Float64(z[i]))
        for i in range(n_pairs):
            buf_qi[i] = BFloat16(Float64(qi[i]))
            buf_qj[i] = BFloat16(Float64(qj[i]))

        energy = compute_electrostatic_energy_genoa(
            buf_x, buf_y, buf_z,
            buf_ni, buf_nj,
            buf_qi, buf_qj,
            buf_14,
            0, 0, n_pairs
        )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_qi.free()
        buf_qj.free()

    elif kern == "turin":
        # f16 storage
        var buf_x = alloc[Float16](alloc_atoms)
        var buf_y = alloc[Float16](alloc_atoms)
        var buf_z = alloc[Float16](alloc_atoms)
        var buf_qi = alloc[Float16](alloc_pairs)
        var buf_qj = alloc[Float16](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = Float16(Float64(x[i]))
            buf_y[i] = Float16(Float64(y[i]))
            buf_z[i] = Float16(Float64(z[i]))
        for i in range(n_pairs):
            buf_qi[i] = Float16(Float64(qi[i]))
            buf_qj[i] = Float16(Float64(qj[i]))

        energy = compute_electrostatic_energy_turin(
            buf_x, buf_y, buf_z,
            buf_ni, buf_nj,
            buf_qi, buf_qj,
            buf_14,
            0, 0, n_pairs
        )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_qi.free()
        buf_qj.free()

    else:
        # Default to skylake
        var buf_x = alloc[Float32](alloc_atoms)
        var buf_y = alloc[Float32](alloc_atoms)
        var buf_z = alloc[Float32](alloc_atoms)
        var buf_qi = alloc[Float32](alloc_pairs)
        var buf_qj = alloc[Float32](alloc_pairs)

        for i in range(n_atoms):
            buf_x[i] = Float32(Float64(x[i]))
            buf_y[i] = Float32(Float64(y[i]))
            buf_z[i] = Float32(Float64(z[i]))
        for i in range(n_pairs):
            buf_qi[i] = Float32(Float64(qi[i]))
            buf_qj[i] = Float32(Float64(qj[i]))

        energy = compute_electrostatic_energy_skylake(
            buf_x, buf_y, buf_z,
            buf_ni, buf_nj,
            buf_qi, buf_qj,
            buf_14,
            0, 0, n_pairs
        )

        buf_x.free()
        buf_y.free()
        buf_z.free()
        buf_qi.free()
        buf_qj.free()

    buf_ni.free()
    buf_nj.free()
    buf_14.free()

    # Return as Python float (f64)
    var builtins = Python.import_module("builtins")
    return builtins.float(Float64(energy))


# =============================================================================
# Simplified Wrappers for PythonModuleBuilder
# =============================================================================

fn vdw_wrapper(args: PythonObject) raises -> PythonObject:
    """Wrapper for vdw_energy_py that takes a dict of arguments."""
    return vdw_energy_py(
        args["x"], args["y"], args["z"],
        args["nb_i"], args["nb_j"],
        args["epsilon"], args["r_star"],
        args["num_pairs"], args["num_atoms"],
        args["kernel"]
    )


fn elec_wrapper(args: PythonObject) raises -> PythonObject:
    """Wrapper for elec_energy_py that takes a dict of arguments."""
    return elec_energy_py(
        args["x"], args["y"], args["z"],
        args["nb_i"], args["nb_j"],
        args["qi"], args["qj"],
        args["is_1_4"],
        args["num_pairs"], args["num_atoms"],
        args["kernel"]
    )


# =============================================================================
# Python Module Export (for direct import from Python)
# =============================================================================

from python.bindings import PythonModuleBuilder


@export
fn PyInit_smallfield() -> PythonObject:
    """Initialize the smallfield Python module."""
    try:
        var m = PythonModuleBuilder("smallfield")
        m.def_function[vdw_wrapper]("vdw_energy")
        m.def_function[elec_wrapper]("elec_energy")
        return m.finalize()
    except e:
        print("Error creating smallfield module:", e)
        return PythonObject()


# Build as Python module:
#   pixi run mojo build -O3 --emit shared-lib src/smallfield.mojo -o build/smallfield.so
#
# Then import from Python:
#   import smallfield
#   vdw = smallfield.vdw_energy({
#       'x': x, 'y': y, 'z': z,
#       'nb_i': nb_i, 'nb_j': nb_j,
#       'epsilon': eps, 'r_star': rstar,
#       'num_pairs': n, 'num_atoms': atoms,
#       'kernel': 'skylake'
#   })
