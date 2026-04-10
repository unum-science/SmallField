import os
import platform
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


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")

        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                if is_nvcc_available:
                    self.build_cuda_extension(ext)
                else:
                    self.build_gcc_extension(ext)
            else:
                super().build_extension(ext)

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
                        "-std=c++17",
                        "-fdiagnostics-color=always",
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
                        "-std=c++17",
                        "-fdiagnostics-color=always",
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
                        "-std=c++17",
                        "-fdiagnostics-color=always",
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
            f"nvcc -c {source} -o {output_file} -std=c++17 "
            f"-gencode=arch=compute_{arch_code},code=sm_{arch_code} "
            f"-Xcompiler {host_flags} {include_dirs} -O3 -g"
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
        ["src/smallfield.cu"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            get_python_inc(),
            "src",
            "/usr/local/cuda/include/",
            "/usr/include/cuda/",
        ],
        library_dirs=[
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/wsl/lib",
            python_lib_dir,
        ],
        libraries=[python_lib_name.replace(".a", "")]
        + (["cudart", "cuda", "curand", "cusolver"] if enable_cuda else [])
        + (["gomp"] if enable_openmp else []),
        extra_compile_args=[
            "-std=c++17",
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
