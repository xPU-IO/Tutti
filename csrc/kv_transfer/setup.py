# SPDX-License-Identifier: Apache-2.0
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Target arch: default to the local GPU (H20 -> sm_90); override via
# TORCH_CUDA_ARCH_LIST (torch's cpp_extension honours it, e.g. "9.0").
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"


def read_version():
    """版本唯一来源：仓库根 VERSION（与 CMake / 其余 Python 包共用）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "..", "VERSION")) as f:
        return f.read().strip()


setup(
    name="tutti-kv-transfer",
    version=read_version(),
    description=(
        "gather/scatter CUDA kernels for the Tutti vLLM connector "
        "(single_layer_kv_transfer ported from LMCache, Apache-2.0)"
    ),
    packages=["tutti_kv_transfer"],
    ext_modules=[
        CUDAExtension(
            "tutti_kv_transfer._native",
            sources=["src/mem_kernels.cu"],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
