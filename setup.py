import os
import platform
import shutil
import subprocess
import sys
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc, get_python_lib

import pybind11
import numpy as np

is_macos = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"
nvcc_path = shutil.which("nvcc")
enable_openmp = not is_macos
enable_cuda = is_linux and nvcc_path is not None

# The CMake tree this links against, and where every intermediate object lands. `build_dev` is what
# the `dev` preset configures and stays the default; SMALLFIELD_BUILD_DIR points a parallel session
# at a private tree instead, so validating the binding does not write into a build directory
# something else is using.
smallfield_build_dir = os.environ.get("SMALLFIELD_BUILD_DIR") or "build_dev"

# RDKit ships no pkg-config, so probe the usual prefixes most-explicit first.
# RDKIT_PREFIX mirrors the CMake flag, CONDA_PREFIX covers an activated conda or
# micromamba environment, sys.prefix covers an install run from inside one, and the
# last entry is the in-tree environment `pixi install` creates. Same order as the
# resolution in CMakeLists.txt — the two must not drift.
rdkit_prefix_candidates = [
    os.environ.get("RDKIT_PREFIX"),
    os.environ.get("CONDA_PREFIX"),
    sys.prefix,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pixi", "envs", "default"),
]
rdkit_prefix = next(
    (
        candidate
        for candidate in rdkit_prefix_candidates
        if candidate and os.path.exists(os.path.join(candidate, "include", "rdkit", "GraphMol", "ROMol.h"))
    ),
    "",
)
if not rdkit_prefix:
    raise SystemExit(
        "SmallField needs the RDKit development headers to build.\n"
        "Run `pixi install`, or install them with `micromamba install -c conda-forge\n"
        "librdkit-dev` and either build from inside that environment or pass\n"
        "RDKIT_PREFIX=/path/to/environment."
    )

rdkit_include_root = os.path.join(rdkit_prefix, "include")
rdkit_include_dir = os.path.join(rdkit_include_root, "rdkit")
rdkit_lib_dir = os.path.join(rdkit_prefix, "lib")

# The toolkit is the one holding the `nvcc` on PATH, resolved through its symlinks, rather than a
# guess at where a toolkit is installed. `/usr/local/cuda` is an alternatives symlink that routinely
# names a different release than the `nvcc` that will run, and picking headers from one release while
# compiling under another is not a latent inconsistency: it is how `fatal error: cuda/std/array`
# arrives from a host compile that never saw the toolkit it was meant to build against. CUDA_HOME and
# CUDA_PATH still win, since naming a toolkit that is not first on PATH is what they are for.
cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
if not cuda_root and nvcc_path:
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
if not cuda_root:
    raise SystemExit(
        "SmallField needs a CUDA toolkit to build the Python binding: it includes libcu++\n"
        "through `smallfield/api/conformer_layout.hpp` whatever it ends up linking.\n"
        "Put `nvcc` on PATH, or pass CUDA_HOME=/path/to/toolkit."
    )

cuda_include_dirs = [
    os.path.join(cuda_root, "targets", "x86_64-linux", "include"),
    os.path.join(cuda_root, "include"),
    "/usr/include/cuda",
]
cuda_library_dirs = [
    os.path.join(cuda_root, "targets", "x86_64-linux", "lib"),
    os.path.join(cuda_root, "lib64"),
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/wsl/lib",
]

# CUDA 12 is what the binding builds against, and this is where that claim is stated: 13 relocated
# libcu++ from `include/cuda/std` to `include/cccl/cuda/std`, so the include paths above find nothing
# under it and the host compile of `_binding.cpp` has nowhere else to look. Probing exactly the
# directories that reach the compiler is what lets the refusal name the toolkit, rather than leaving
# a missing header to surface a minute into a compile.
if not any(os.path.exists(os.path.join(path, "cuda", "std", "array")) for path in cuda_include_dirs):
    raise SystemExit(
        f"The CUDA toolkit at {cuda_root} carries no libcu++ under include/cuda/std.\n"
        "CUDA 13 moved it to include/cccl, which this binding does not build against.\n"
        "Pass CUDA_HOME=/path/to/cuda-12.x, or put that toolkit's `nvcc` first on PATH."
    )

rdkit_libraries = [
    "RDKitSmilesParse",
    "RDKitGraphMol",
    "RDKitRDGeneral",
    "RDKitForceFieldHelpers",
    "RDKitForceField",
    "RDKitDistGeomHelpers",
    "RDKitDistGeometry",
    "RDKitDepictor",
    "RDKitSubstructMatch",
    "RDKitAlignment",
    "RDKitEigenSolvers",
    "RDKitMolTransforms",
    "RDKitChemTransforms",
    "RDKitDescriptors",
    "RDKitMolStandardize",
]

boost_libraries = ["boost_serialization", "boost_iostreams"]


class BuildExt(build_ext):
    """`build_ext` that will not compile the binding until the CMake libraries it links against exist."""

    def finalize_options(self):
        """Move intermediate objects beside the CMake tree they link against, out of setuptools' own ``build/``.

        A private build directory is then private for the whole build, not only for the library it resolves.
        """
        super().finalize_options()
        self.build_temp = os.path.join(smallfield_build_dir, os.path.basename(self.build_temp))

    def build_extensions(self):
        """Compile the extension once the libraries it links against are present.

        The extension is one translation unit against a library CMake has already built, so guaranteeing that
        library is the only thing this adds to the base class.
        """
        self.ensure_smallfield_libraries()
        super().build_extensions()

    def ensure_smallfield_libraries(self):
        """Configure and build the CMake targets the extension links, unless both artifacts are already present.

        Both files existing is the whole check, so a stale tree is reused rather than rebuilt, and the configure
        step is handed the same RDKit prefix this file resolved, so the library and the binding cannot end up
        compiled against two different RDKits.
        """
        static_library = os.path.join(smallfield_build_dir, "libsmallfield_lib.a")
        shared_library = os.path.join(smallfield_build_dir, "libsmallfield_python_lib.so")
        if os.path.exists(static_library) and os.path.exists(shared_library):
            return

        configure_cmd = ["cmake", "-S", ".", "-B", smallfield_build_dir]
        configure_cmd.append(f"-DRDKIT_PREFIX={rdkit_prefix}")
        subprocess.run(configure_cmd, check=True)
        subprocess.run(
            ["cmake", "--build", smallfield_build_dir, "--target", "smallfield_lib", "smallfield_python_lib", "-j"],
            check=True,
        )


# Get Python library path dynamically
python_lib_dir = get_python_lib(standard_lib=True)
python_lib_name = os.path.basename(python_lib_dir).replace(".so", "")

ext_modules = [
    Extension(
        "_smallfield",
        ["python/smallfield/_binding.cpp"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            get_python_inc(),
            "src",
            rdkit_include_root,
            rdkit_include_dir,
        ]
        + [path for path in cuda_include_dirs if path and os.path.exists(path)],
        library_dirs=[python_lib_dir, rdkit_lib_dir, smallfield_build_dir]
        + [path for path in cuda_library_dirs if path and os.path.exists(path)],
        extra_objects=[os.path.abspath(os.path.join(smallfield_build_dir, "libsmallfield_python_lib.so"))],
        libraries=[python_lib_name.replace(".a", "")]
        + rdkit_libraries
        + boost_libraries
        + (["cudart", "cuda", "curand", "cusolver"] if enable_cuda else [])
        + (["gomp"] if enable_openmp else []),
        # `-march=native` rather than a hardcoded AVX-512 list: the list named `-mavx512fp16`, which
        # only Sapphire Rapids and Turin carry, so the binding failed to build on any older host and
        # would have faulted on one that compiled it and ran elsewhere. Native matches what CMake
        # gives the library, and both make the result non-redistributable in the same way.
        extra_compile_args=["-std=c++20", "-fPIC", "-march=native"] + (["-fopenmp"] if enable_openmp else []),
        extra_link_args=[f"-Wl,-rpath,{python_lib_dir}"]
        + [f"-Wl,-rpath,{os.path.abspath(smallfield_build_dir)}"]
        + [f"-Wl,-rpath,{rdkit_lib_dir}"]
        + (["-fopenmp"] if enable_openmp else []),
        language="c++",
    ),
]

# Name, version, description, authors, URLs, dependencies and the Python floor are declared in
# `pyproject.toml` and read from there. Only what PEP 621 cannot express belongs here: the CUDA
# extension module and the build_ext subclass that drives CMake and nvcc. Restating a field in both
# places is how `python_requires` came to say 3.10 while `pyproject.toml` said 3.12.
setup(
    ext_modules=ext_modules,
    packages=["smallfield", "smallfield.tools"],
    package_dir={"smallfield": "python/smallfield"},
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
