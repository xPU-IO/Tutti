#!/usr/bin/env bash

# Machine-local environment for vLLM/Tutti profiling. Source this file.
_TUTTI_CUDA=/data/home/ryeqiu/Tutti/third_pkgs/nvidia/cuda-toolkit
_TUTTI_PROFILE_ROOT=/data2/ryeqiu/tutti-profile

export TUTTI_PROFILE_ROOT="$_TUTTI_PROFILE_ROOT"
export TMPDIR="$_TUTTI_PROFILE_ROOT/tmp"
export XDG_CACHE_HOME="$_TUTTI_PROFILE_ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$VLLM_CACHE_ROOT/flashinfer_autotune_cache"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
export FLASHINFER_CUBIN_DIR="$XDG_CACHE_HOME/flashinfer-cubins"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export TUTTI_NVTX=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH="/data/home/ryeqiu/tutti-env/bin:/data/home/ryeqiu/tutti-compilers/bin:$_TUTTI_CUDA/bin:/data/home/ryeqiu/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
export PYTHONPATH="/data/home/ryeqiu/Tutti/integration/vllm-connector/bindings/python/src:/data/home/ryeqiu/Tutti/integration/vllm-connector${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/data/home/ryeqiu/tutti-compilers/lib:$_TUTTI_CUDA/lib64:/usr/local/lib:/data/home/ryeqiu/Phoenix/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TUTTI_PROFILE_ROOT/logs" "$TUTTI_PROFILE_ROOT/reports"
unset _TUTTI_CUDA _TUTTI_PROFILE_ROOT
