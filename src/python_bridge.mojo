# SmallField: Python Bridge
# Interface for calling SmallField from Python via RDKit
# Updated for Mojo 0.25.7

from memory import alloc
from math import sqrt


# Re-export MMFF94 constants
alias MMFF_BOND_PREFACTOR: Float32 = 143.9325
alias MMFF_CUBIC_STRETCH: Float32 = -2.0


fn bond_energy_single(r: Float32, r0: Float32, kb: Float32) -> Float32:
    """MMFF94 bond stretching energy for a single bond."""
    var dr = r - r0
    var dr2 = dr * dr
    var cs = MMFF_CUBIC_STRETCH
    var cs2 = cs * cs
    return Float32(0.5) * MMFF_BOND_PREFACTOR * kb * dr2 * (
        Float32(1.0) + cs * dr + Float32(7.0 / 12.0) * cs2 * dr2
    )


fn compute_distance(
    x1: Float32, y1: Float32, z1: Float32,
    x2: Float32, y2: Float32, z2: Float32
) -> Float32:
    """Compute Euclidean distance between two points."""
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    return sqrt(dx * dx + dy * dy + dz * dz)


fn main():
    """Test the Python bridge functionality."""
    print("SmallField Python Bridge")
    print("========================")
    print()

    # Test bond energy calculation
    var kb: Float32 = 4.258   # C-C single bond force constant
    var r0: Float32 = 1.508   # C-C equilibrium distance

    print("Bond Energy Test:")
    print("  Force constant (kb):", kb, "mdyn/A")
    print("  Equilibrium length (r0):", r0, "A")
    print()

    # Test at equilibrium
    var e_eq = bond_energy_single(r0, r0, kb)
    print("  At equilibrium (r =", r0, "A):")
    print("    Energy =", e_eq, "kcal/mol (expected: 0.0)")

    # Test stretched
    var r_stretched: Float32 = 1.6
    var e_stretched = bond_energy_single(r_stretched, r0, kb)
    print("  Stretched (r =", r_stretched, "A):")
    print("    Energy =", e_stretched, "kcal/mol")

    # Test compressed
    var r_compressed: Float32 = 1.4
    var e_compressed = bond_energy_single(r_compressed, r0, kb)
    print("  Compressed (r =", r_compressed, "A):")
    print("    Energy =", e_compressed, "kcal/mol")

    print()
    print("SmallField ready for Python integration!")
