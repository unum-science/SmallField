#!/usr/bin/env python3
"""
Test SmallField Python binding.

This script tests the Mojo MMFF94 implementations by compiling and running
the binding with test data.
"""

import subprocess
from pathlib import Path


def test_binding_compiles():
    """Test that the smallfield binding compiles."""
    src_dir = Path(__file__).parent.parent / "src"

    print("Testing SmallField Python Binding")
    print("=" * 40)

    # Test that the module compiles
    result = subprocess.run(
        ["pixi", "run", "mojo", "run", str(src_dir / "smallfield.mojo")],
        capture_output=True,
        text=True,
        cwd=str(src_dir.parent)
    )

    if result.returncode != 0:
        print("FAILED to compile smallfield.mojo:")
        print(result.stderr)
        return False

    print("Compilation: OK")
    # Show only the actual output, skip warnings
    for line in result.stdout.split('\n'):
        if not line.startswith('/') and not line.startswith('Included'):
            print(line)

    return True


def test_build_optimized():
    """Test that the binding can be built with -O3."""
    src_dir = Path(__file__).parent.parent / "src"
    build_dir = Path(__file__).parent.parent / "build"
    build_dir.mkdir(exist_ok=True)

    print("\nTesting optimized build (-O3)...")

    result = subprocess.run(
        ["pixi", "run", "mojo", "build", "-O3",
         str(src_dir / "smallfield.mojo"),
         "-o", str(build_dir / "smallfield")],
        capture_output=True,
        text=True,
        cwd=str(src_dir.parent)
    )

    if result.returncode != 0:
        print("FAILED to build smallfield.mojo with -O3:")
        print(result.stderr[:500] if result.stderr else "No error output")
        return False

    print("Optimized build: OK")
    return True


def show_api():
    """Show the intended API for the binding."""
    print("\nIntended Python API:")
    print("-" * 40)
    print("""
import smallfield
import numpy as np

# Coordinates (N atoms)
x = np.array([0.0, 1.54, -0.5], dtype=np.float64)
y = np.array([0.0, 0.0, 0.9], dtype=np.float64)
z = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Nonbonded pairs
nb_i = np.array([0, 0, 1], dtype=np.int64)
nb_j = np.array([1, 2, 2], dtype=np.int64)
epsilon = np.array([0.05, 0.02, 0.02], dtype=np.float64)
r_star = np.array([3.5, 3.0, 3.0], dtype=np.float64)

# Compute vdW energy with kernel selection
vdw = smallfield.vdw_energy_py(
    x, y, z,
    nb_i, nb_j,
    epsilon, r_star,
    len(nb_i),     # num_pairs
    len(x),        # num_atoms
    "skylake"      # kernel: skylake, simd, genoa, turin
)
print(f"vdW energy: {vdw:.4f} kcal/mol")
""")


if __name__ == "__main__":
    success = True

    success = test_binding_compiles() and success
    success = test_build_optimized() and success

    show_api()

    if success:
        print("\n[PASS] All binding tests passed")
    else:
        print("\n[FAIL] Some binding tests failed")
        exit(1)
