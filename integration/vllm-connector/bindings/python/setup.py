# integration/vllm-connector/bindings/python/setup.py
#
# Builds the pybind11 extension tutti_runtime._core against an existing
# tutti C++ build tree. This script never builds tutti itself.
#
# Environment variables:
#   TUTTI_ROOT       repository root containing tutti/include (default:
#                    three levels up from this file, i.e. the repo root)
#   TUTTI_BUILD_DIR  existing tutti CMake build directory that contains
#                    tutti/presets/libtutti_presets.a (default: probe
#                    $TUTTI_ROOT/build)
#
# Prerequisite check (per task card): if libtutti_presets.a cannot be found,
# fail with a clear message instead of building tutti.

import glob
import os
import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(1)


def resolve_tutti_root():
    env = os.environ.get("TUTTI_ROOT")
    if env:
        return os.path.abspath(env)
    # integration/vllm-connector/bindings/python -> repo root is four levels up.
    return os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def resolve_build_dir(tutti_root):
    env = os.environ.get("TUTTI_BUILD_DIR")
    if env:
        return os.path.abspath(env)
    p = os.path.join(tutti_root, "build")
    if os.path.exists(os.path.join(p, "tutti", "presets", "libtutti_presets.a")):
        return p
    return None


def cmake_cache_value(build_dir, name):
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.exists(cache):
        return None
    prefix = name + ":"
    with open(cache) as f:
        for line in f:
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip()
    return None


def resolve_cuda_root(build_dir):
    """Prefer the exact toolkit used by the tutti build (from CMakeCache)."""
    toolkit_root = cmake_cache_value(build_dir, "CUDAToolkit_ROOT")
    if toolkit_root:
        return toolkit_root
    compiler = cmake_cache_value(build_dir, "CMAKE_CUDA_COMPILER")
    if compiler:
        return os.path.abspath(os.path.join(compiler, "..", ".."))
    return "/usr/local/cuda"


def resolve_accelerator_profile(build_dir):
    profile = cmake_cache_value(build_dir, "TUTTI_ACCELERATOR")
    if profile not in {"CUDA", "MUSA", "MACA"}:
        die(
            "unsupported or missing TUTTI_ACCELERATOR in %s/CMakeCache.txt: %r"
            % (build_dir, profile)
        )
    return profile


TUTTI_ROOT = resolve_tutti_root()
TUTTI_BUILD_DIR = resolve_build_dir(TUTTI_ROOT)

if TUTTI_BUILD_DIR is None:
    die(
        "libtutti_presets.a not found. This binding does not build tutti "
        "itself; point TUTTI_BUILD_DIR at an existing tutti CMake build "
        "directory (expected: $TUTTI_BUILD_DIR/tutti/presets/"
        "libtutti_presets.a), or configure/build tutti first."
    )

PRESETS_LIB = os.path.join(TUTTI_BUILD_DIR, "tutti", "presets", "libtutti_presets.a")
if not os.path.exists(PRESETS_LIB):
    die(
        "libtutti_presets not found under %s (task prerequisite failed; "
        "this binding never builds tutti itself)." % TUTTI_BUILD_DIR
    )

CUDA_ROOT = resolve_cuda_root(TUTTI_BUILD_DIR)
ACCELERATOR_PROFILE = resolve_accelerator_profile(TUTTI_BUILD_DIR)
TUTTI_INCLUDE = os.path.join(TUTTI_ROOT, "tutti", "include")

CCCL_DIRS = sorted(
    glob.glob(os.path.join(CUDA_ROOT, "targets", "*", "include", "cccl"))
)
LIBNVM_DIR = os.path.join(
    TUTTI_BUILD_DIR, "tutti", "device_manager", "nvme", "libnvm"
)

STATIC_LIBS = [
    PRESETS_LIB,
    os.path.join(
        TUTTI_BUILD_DIR, "tutti", "data_paths", "local_nvme",
        "libtutti_local_nvme_datapath.a",
    ),
    os.path.join(
        TUTTI_BUILD_DIR, "tutti", "data_paths", "striped_local_nvme",
        "libtutti_striped_local_nvme_datapath.a",
    ),
    os.path.join(
        TUTTI_BUILD_DIR, "tutti", "resolvers", "libtutti_resolver_factory.a",
    ),
]
for lib in STATIC_LIBS:
    if not os.path.exists(lib):
        die("required static library not found: %s" % lib)

RUNTIME_LIB_DIRS = []
if os.path.isdir(LIBNVM_DIR):
    RUNTIME_LIB_DIRS.append(LIBNVM_DIR)
if os.path.isdir(os.path.join(CUDA_ROOT, "lib64")):
    RUNTIME_LIB_DIRS.append(os.path.join(CUDA_ROOT, "lib64"))

ext = Pybind11Extension(
    "tutti_runtime._core",
    sources=["src/_core.cpp"],
    include_dirs=[TUTTI_INCLUDE, os.path.join(CUDA_ROOT, "include")] + CCCL_DIRS,
    extra_compile_args=[
        "-std=c++17",
        "-DTUTTI_USE_%s=1" % ACCELERATOR_PROFILE,
        '-DTUTTI_COMPILED_ACCELERATOR_PROFILE="%s"' % ACCELERATOR_PROFILE,
        "-DTUTTI_DEFAULT_ACCEL_ID=0",
    ],
    extra_objects=STATIC_LIBS,
    libraries=(["nvm"] if os.path.isdir(LIBNVM_DIR) else []) + ["cudart"],
    library_dirs=RUNTIME_LIB_DIRS,
    runtime_library_dirs=RUNTIME_LIB_DIRS,  # rpath so the built .so is loadable
    language="c++",
)

setup(ext_modules=[ext])
