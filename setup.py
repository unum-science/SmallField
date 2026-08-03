import os
import platform
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
is_nvcc_available = os.system("which nvcc > /dev/null 2>&1") == 0
enable_openmp = not is_macos
enable_cuda = is_linux and is_nvcc_available

# RDKit ships no pkg-config, so probe the usual prefixes most-explicit first.
# RDKIT_PREFIX mirrors the CMake flag, CONDA_PREFIX covers an active conda or
# pixi environment, and sys.prefix covers an install run from inside one.
rdkit_prefix_candidates = [
    os.environ.get("RDKIT_PREFIX"),
    os.environ.get("CONDA_PREFIX"),
    sys.prefix,
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
        "Install them with `micromamba install -c conda-forge librdkit-dev`, then either\n"
        "build from inside that environment or pass RDKIT_PREFIX=/path/to/conda/env."
    )

rdkit_include_root = os.path.join(rdkit_prefix, "include")
rdkit_include_dir = os.path.join(rdkit_include_root, "rdkit")
rdkit_lib_dir = os.path.join(rdkit_prefix, "lib")

cuda_root_candidates = [
    os.environ.get("CUDA_HOME"),
    os.environ.get("CUDA_PATH"),
    "/usr/local/cuda-13.0",
    "/usr/local/cuda",
    "/usr",
]
cuda_root = next((candidate for candidate in cuda_root_candidates if candidate and os.path.exists(candidate)), "")
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
    def build_extensions(self):
        self.ensure_smallfield_libraries()
        self.compiler.src_extensions.append(".cu")

        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                if is_nvcc_available:
                    self.build_cuda_extension(ext)
                else:
                    self.build_gcc_extension(ext)
            else:
                super().build_extension(ext)

    def ensure_smallfield_libraries(self):
        static_library = os.path.join("build", "libsmallfield_lib.a")
        shared_library = os.path.join("build", "libsmallfield_python_lib.so")
        if os.path.exists(static_library) and os.path.exists(shared_library):
            return

        configure_cmd = ["cmake", "-S", ".", "-B", "build"]
        configure_cmd.append(f"-DRDKIT_PREFIX={rdkit_prefix}")
        subprocess.run(configure_cmd, check=True)
        subprocess.run(
            ["cmake", "--build", "build", "--target", "smallfield_lib", "smallfield_python_lib", "-j"], check=True
        )

    def build_cuda_extension(self, ext):
        # Compile CUDA source files
        for source in ext.sources:
            if source.endswith(".cu"):
                self.compile_cuda(source)

        # Compile non-CUDA source files
        objects = []
        for source in ext.sources:
            if not source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++20",
                        "-fdiagnostics-color=always",
                        "-DSMALLFIELD_BUILD_PYTHON=1",
                    ]
                    + ext.extra_compile_args,
                    include_dirs=ext.include_dirs,
                )
                objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects + [os.path.join(self.build_temp, "smallfield.o")],
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def build_gcc_extension(self, ext):
        # Fallback for compiling on macOS or without CUDA
        objects = []
        for source in ext.sources:
            if source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_preargs=["-x", "c++"],
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++20",
                        "-fdiagnostics-color=always",
                        "-DSMALLFIELD_BUILD_PYTHON=1",
                    ]
                    + ext.extra_compile_args
                    + (["-fopenmp"] if enable_openmp else []),
                    include_dirs=ext.include_dirs,
                )
            else:
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++20",
                        "-fdiagnostics-color=always",
                        "-DSMALLFIELD_BUILD_PYTHON=1",
                    ]
                    + ext.extra_compile_args
                    + (["-fopenmp"] if enable_openmp else []),
                    include_dirs=ext.include_dirs,
                )
            objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=[lib for lib in ext.libraries if not lib.startswith("cu")],
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def compile_cuda(self, source):
        # Compile CUDA source file using NVCC
        ext = self.extensions[0]
        output_dir = self.build_temp
        os.makedirs(output_dir, exist_ok=True)
        include_dirs = self.compiler.include_dirs + ext.include_dirs
        include_dirs = " ".join(f"-I{dir}" for dir in include_dirs)
        output_file = os.path.join(output_dir, "smallfield.o")

        # Let's try inferring the compute capability from the GPU
        arch_code = "90"
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit

            device = cuda.Device(0)  # Get the default device
            major, minor = device.compute_capability()
            arch_code = f"{major}{minor}"
        except ImportError:
            # Fallback: try nvidia-smi to detect compute capability
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    cap = result.stdout.strip().split("\n")[0].strip()
                    arch_code = cap.replace(".", "")
            except Exception:
                pass

        # Pass AVX-512 flags to host compiler via -Xcompiler
        host_flags = (
            "-fPIC,-mavx512f,-mavx512bw,-mavx512dq,-mavx512vl,-mavx512bf16,-mavx512fp16"
        )
        if enable_openmp:
            host_flags += ",-fopenmp"
        cmd = (
            f"nvcc -c {source} -o {output_file} -std=c++20 "
            f"-gencode=arch=compute_{arch_code},code=sm_{arch_code} "
            f"-Xcompiler {host_flags} {include_dirs} -DSMALLFIELD_BUILD_PYTHON=1 -O3 -g"
        )
        if os.system(cmd) != 0:
            raise RuntimeError(f"nvcc compilation of {source} failed")


__version__ = open("VERSION", "r").read().strip()

long_description = ""
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

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
        library_dirs=[python_lib_dir, rdkit_lib_dir, "build"]
        + [path for path in cuda_library_dirs if path and os.path.exists(path)],
        extra_objects=[os.path.abspath(os.path.join("build", "libsmallfield_python_lib.so"))],
        libraries=[python_lib_name.replace(".a", "")]
        + rdkit_libraries
        + boost_libraries
        + (["cudart", "cuda", "curand", "cusolver"] if enable_cuda else [])
        + (["gomp"] if enable_openmp else []),
        extra_compile_args=[
            "-std=c++20",
            "-fPIC",
            "-mavx512f",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512bf16",  # Sapphire Rapids / Genoa BF16
            "-mavx512fp16",  # Sapphire Rapids / Turin FP16
        ]
        + (["-fopenmp"] if enable_openmp else []),
        extra_link_args=[f"-Wl,-rpath,{python_lib_dir}"]
        + [f"-Wl,-rpath,{os.path.abspath('build')}"]
        + [f"-Wl,-rpath,{rdkit_lib_dir}"]
        + (["-fopenmp"] if enable_openmp else []),
        language="c++",
    ),
]

setup(
    name="smallfield",
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/unum-science/SmallField",
    description="Fast MMFF94 force field implementation with CPU (AVX-512) and GPU (CUDA) support.",
    long_description=long_description,
    ext_modules=ext_modules,
    packages=["smallfield"],
    package_dir={"smallfield": "python/smallfield"},
    extras_require={
        "test": ["pytest", "pytest-benchmark", "rdkit"],
        "cpu": ["pytest", "pytest-benchmark", "rdkit"],
        "gpu": ["pytest", "pytest-benchmark", "rdkit", "cupy-cuda12x"],
    },
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.7",
)
