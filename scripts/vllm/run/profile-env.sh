#!/usr/bin/env bash

# Machine-local environment for vLLM/Tutti profiling. Source this file.
#
# Every path below is verified to exist on this host. Previous revisions of this
# file pointed at /data2/ryeqiu, /data/home/ryeqiu/tutti-env and a vendored CUDA
# toolkit under third_pkgs -- none of which exist any more, so sourcing it
# silently produced a broken PATH.
#
# Scratch and profile output deliberately live on /mnt/nvme4 rather than the
# root filesystem: root is 20G and typically over 90% full, and nsys reports
# alone run to hundreds of megabytes.

_TUTTI_CUDA=/usr/local/cuda-13.0
_TUTTI_PROFILE_ROOT=/mnt/nvme4/tutti-profile
_TUTTI_VENV=/data/home/ryeqiu/vllm-env
_TUTTI_REPO=/data/home/ryeqiu/Tutti
# flashinfer JIT-compiles CUTLASS kernels at engine startup, and CUTLASS needs a
# modern C++ frontend. The system compiler here is GCC 8.5, whose libstdc++
# headers fail to instantiate cute::tuple ("incomplete type ... is not allowed"),
# which surfaces only as "Engine core initialization failed". This toolset must
# be ahead of /usr/bin on PATH.
_TUTTI_GCC=/opt/rh/gcc-toolset-13/root/usr

export TUTTI_PROFILE_ROOT="$_TUTTI_PROFILE_ROOT"
export TMPDIR="$_TUTTI_PROFILE_ROOT/tmp"
export XDG_CACHE_HOME="$_TUTTI_PROFILE_ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$VLLM_CACHE_ROOT/flashinfer_autotune_cache"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
export FLASHINFER_CUBIN_DIR="$XDG_CACHE_HOME/flashinfer-cubins"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"

# NVTX ranges are what make a Tutti nsys report readable: without them the
# storage work is an undifferentiated band of CUDA API calls.
export TUTTI_NVTX=1

# GPU selection is left to the caller. Exporting a fixed list here is what made
# the previous revision silently TP4-only.
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

export PATH="$_TUTTI_VENV/bin:$_TUTTI_GCC/bin:$_TUTTI_CUDA/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"

# nvcc picks its host compiler from CUDAHOSTCXX, not from PATH alone.
export CUDAHOSTCXX="$_TUTTI_GCC/bin/g++"
export CC="$_TUTTI_GCC/bin/gcc"
export CXX="$_TUTTI_GCC/bin/g++"

# The repo root provides the `tutti` package; csrc/python/src provides the
# compiled `tutti_runtime` extension.
export PYTHONPATH="$_TUTTI_REPO/csrc/python/src:$_TUTTI_REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$_TUTTI_GCC/lib64:$_TUTTI_CUDA/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export TUTTI_PYTHON="$_TUTTI_VENV/bin/python"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TUTTI_PROFILE_ROOT/logs" "$TUTTI_PROFILE_ROOT/reports"

unset _TUTTI_CUDA _TUTTI_PROFILE_ROOT _TUTTI_VENV _TUTTI_REPO _TUTTI_GCC
