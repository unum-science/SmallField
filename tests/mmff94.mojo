# SmallField: MMFF94 Unit Tests
# Validates energy calculations against known values
# Updated for Mojo 0.25.7

from math import sqrt
from memory import alloc


# MMFF94 constants
alias MMFF_BOND_PREFACTOR: Float32 = 143.9325
alias MMFF_CUBIC_STRETCH: Float32 = -2.0
alias PI: Float32 = 3.14159265358979323846


fn abs_f32(x: Float32) -> Float32:
    """Absolute value for Float32."""
    if x < Float32(0.0):
        return -x
    return x


fn bond_energy_single(r: Float32, r0: Float32, kb: Float32) -> Float32:
    var dr = r - r0
    var dr2 = dr * dr
    var cs = MMFF_CUBIC_STRETCH
    var cs2 = cs * cs
    return Float32(0.5) * MMFF_BOND_PREFACTOR * kb * dr2 * (
        Float32(1.0) + cs * dr + Float32(7.0 / 12.0) * cs2 * dr2
    )


fn assert_close(actual: Float32, expected: Float32, tol: Float32, msg: String) -> Bool:
    """Check if two values are close within tolerance."""
    var diff = actual - expected
    if diff < Float32(0.0):
        diff = -diff
    if diff <= tol:
        print("  PASS:", msg)
        return True
    else:
        print("  FAIL:", msg)
        print("    Expected:", expected)
        print("    Actual:", actual)
        print("    Diff:", diff)
        return False


fn test_bond_energy_at_equilibrium() -> Bool:
    """Bond energy should be zero at equilibrium."""
    print("Test: Bond energy at equilibrium")

    var kb: Float32 = 4.258
    var r0: Float32 = 1.508

    var energy = bond_energy_single(r0, r0, kb)

    return assert_close(energy, Float32(0.0), Float32(1e-6), "Energy at equilibrium")


fn test_bond_energy_positive_stretch() -> Bool:
    """Bond energy should be positive when stretched."""
    print("Test: Bond energy positive for stretched bond")

    var kb: Float32 = 4.258
    var r0: Float32 = 1.508
    var r: Float32 = 1.6  # Stretched

    var energy = bond_energy_single(r, r0, kb)

    if energy > Float32(0.0):
        print("  PASS: Energy =", energy, "> 0")
        return True
    else:
        print("  FAIL: Energy =", energy, "should be > 0")
        return False


fn test_bond_energy_positive_compress() -> Bool:
    """Bond energy should be positive when compressed."""
    print("Test: Bond energy positive for compressed bond")

    var kb: Float32 = 4.258
    var r0: Float32 = 1.508
    var r: Float32 = 1.4  # Compressed

    var energy = bond_energy_single(r, r0, kb)

    if energy > Float32(0.0):
        print("  PASS: Energy =", energy, "> 0")
        return True
    else:
        print("  FAIL: Energy =", energy, "should be > 0")
        return False


fn test_bond_energy_symmetry() -> Bool:
    """Bond energy should be symmetric around equilibrium (approximately)."""
    print("Test: Bond energy symmetry")

    var kb: Float32 = 4.258
    var r0: Float32 = 1.508
    var delta: Float32 = 0.1

    var e_stretched = bond_energy_single(r0 + delta, r0, kb)
    var e_compressed = bond_energy_single(r0 - delta, r0, kb)

    # Not exactly symmetric due to cubic term, but should be similar
    var ratio = e_stretched / e_compressed
    if ratio > Float32(0.5) and ratio < Float32(2.0):
        print("  PASS: Ratio =", ratio, "(approximately symmetric)")
        return True
    else:
        print("  FAIL: Ratio =", ratio, "(too asymmetric)")
        return False


fn test_bond_energy_scaling() -> Bool:
    """Bond energy should scale linearly with force constant."""
    print("Test: Bond energy scales with force constant")

    var kb1: Float32 = 4.0
    var kb2: Float32 = 8.0  # Double
    var r0: Float32 = 1.5
    var r: Float32 = 1.6

    var e1 = bond_energy_single(r, r0, kb1)
    var e2 = bond_energy_single(r, r0, kb2)

    var ratio = e2 / e1
    return assert_close(ratio, Float32(2.0), Float32(1e-4), "Energy ratio with 2x kb")


fn test_kahan_accuracy() -> Bool:
    """Test Kahan summation improves accuracy."""
    print("Test: Kahan summation accuracy")

    # Sum many small numbers that would lose precision with naive addition
    var n = 10000
    var small_val: Float32 = 0.0001

    # Naive sum
    var naive_sum: Float32 = 0.0
    for _ in range(n):
        naive_sum += small_val

    # Kahan sum
    var kahan_sum: Float32 = 0.0
    var compensation: Float32 = 0.0
    for _ in range(n):
        var y = small_val - compensation
        var t = kahan_sum + y
        compensation = (t - kahan_sum) - y
        kahan_sum = t

    # Expected value
    var expected = Float32(n) * small_val

    var naive_error = abs_f32(naive_sum - expected)
    var kahan_error = abs_f32(kahan_sum - expected)

    print("  Naive error:", naive_error)
    print("  Kahan error:", kahan_error)

    if kahan_error <= naive_error:
        print("  PASS: Kahan is at least as accurate")
        return True
    else:
        print("  FAIL: Kahan should be more accurate")
        return False


fn test_distance_calculation() -> Bool:
    """Test Euclidean distance calculation."""
    print("Test: Distance calculation")

    var x1: Float32 = 0.0
    var y1: Float32 = 0.0
    var z1: Float32 = 0.0
    var x2: Float32 = 3.0
    var y2: Float32 = 4.0
    var z2: Float32 = 0.0

    # Known: 3-4-5 right triangle
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var dist = sqrt(dx*dx + dy*dy + dz*dz)

    return assert_close(dist, Float32(5.0), Float32(1e-6), "3-4-5 distance")


fn main():
    print("SmallField MMFF94 Unit Tests")
    print("=" * 60)
    print()

    var passed = 0
    var failed = 0

    if test_bond_energy_at_equilibrium():
        passed += 1
    else:
        failed += 1
    print()

    if test_bond_energy_positive_stretch():
        passed += 1
    else:
        failed += 1
    print()

    if test_bond_energy_positive_compress():
        passed += 1
    else:
        failed += 1
    print()

    if test_bond_energy_symmetry():
        passed += 1
    else:
        failed += 1
    print()

    if test_bond_energy_scaling():
        passed += 1
    else:
        failed += 1
    print()

    if test_kahan_accuracy():
        passed += 1
    else:
        failed += 1
    print()

    if test_distance_calculation():
        passed += 1
    else:
        failed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "passed,", failed, "failed")

    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed!")
