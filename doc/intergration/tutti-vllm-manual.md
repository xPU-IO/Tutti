# Tutti × vLLM 操作手册

> 本机（TENCENT64）专用。权威架构：`integration/vllm-connector/ARCHITECTURE.md`；
> 设计复盘：`doc/intergration/tutti-vllm-integration.md`；编排：`ai-orch/`。

## 0. 命名与位置（统一，勿混用）

| 名称 | 路径 | 状态（2026-08-20） |
|---|---|---|
| Python 环境 | `/data/home/ryeqiu/tutti-env` | **就绪**（uv 建的 venv，python 3.11.16；torch 2.13.0+cu130、numpy、pytest 已装） |
| C/C++ 编译器 | `/data/home/ryeqiu/tutti-compilers` | **就绪**（conda 独立环境，gcc/g++ 12.4.0；vllm 要求 ≥11.3） |
| rust | `/data/home/ryeqiu/.rustup` + `.cargo` | **就绪**（stable 1.97.1，`rustup default stable` 已设） |
| 环境入口 | `/data/home/ryeqiu/env-tutti.sh` | 就绪（source 即用，见 §7） |
| tutti 源码 | `/data/home/ryeqiu/Tutti` | 主仓库 |
| tutti_runtime（pybind） | `Tutti/integration/vllm-connector/bindings/python` | **就绪**（T-101，editable 已装） |
| connector 适配层 | `Tutti/integration/vllm-connector/adapter/` | R1 进行中 |
| KV 引擎层 | `Tutti/integration/vllm-connector/engine/` | **就绪**（T-116：core/backend/memory_backend/chunk_index，59 tests 绿） |
| vLLM fork | `/data/home/ryeqiu/Tutti/third_pkgs/vllm` | **就绪**（0.1.dev20081+gb0e9cff5e，editable 已装） |
| vendored CUDA | `/data/home/ryeqiu/Tutti/third_pkgs/nvidia/cuda-toolkit` | 就绪（CUDA 13.3） |
| 模型 | `/data2/qwen`（Qwen2.5-14B、qwen2-5-7B）、`/data2/deepseek-ai`（DeepSeek-V3.1/V4-Flash）、`/data2/tencent`（Hy3、Hy3-FP8） | 冒烟候选：**qwen2-5-7B** |
| trace 负载 | `/data2/traces`（1.1G） | R3 bench 用 |
| 测试日志 | `/data/home/ryeqiu/log/`（vllm_build.log、pip_cuda.log） | |
| pip/tmp 缓存 | `/data/home/ryeqiu/pip-cache`、`/data/home/ryeqiu/tmp` | 磁盘纪律（系统盘曾满） |

## 1. 环境要求（已核实）

- 8×GPU（驱动 580.105.08）；8×NVMe（**NUMA1 四盘已 bring-up**，
  NUMA0 四盘待补，见 §3）
- 系统 python3.6 + gcc8.5：**不用**，一切进 tutti-env/tutti-compilers
- 磁盘纪律：TMPDIR/PIP_CACHE/RUSTUP/CARGO 一律指向 `/data/home/ryeqiu/`

## 2. 构建 Tutti C++（一次）

按 [`../getting-started.md`](../getting-started.md) 完成唯一默认硬件构建：

```bash
cd /data/home/ryeqiu/Tutti
cmake --preset default
cmake --build --preset default --parallel 8
```

## 3. snvme 内核模块 + tutti_daemon bring-up

严格三步（详见 `doc/tutti_daemon.md`）：

```bash
# 1) 内核模块（snvme.ko）：按 doc/getting-started.md 编译，并按站点流程签名/加载。
# 2) tutti_daemon（controller bring-up + mount）：
sudo env TUTTI_VERBOSE=1 nohup \
  /data/home/ryeqiu/Tutti/build/bin/tutti_daemon \
  --config config/local/tutti_daemon.yaml \
  > /data/home/ryeqiu/log/tutti_daemon.log 2>&1 &
# 3) 验证：
ls /dev/snvme* /dev/ssnvme*   # 设备节点出现
ls /mnt/nvme*/                # daemon 挂载点非空
# 真机测试 preset：config/local/preset_gpu0.json
#   TUTTI_NVME_PRESET=.../preset_gpu0.json \
#     python -m pytest tests/contract/test_local_store.py -k real -v
```

> 机器重启后需重跑（内核模块不持久）。

## 4. Python 环境 tutti-env（已完成，T-110）

实际构建方式（uv venv，非 conda）：

```bash
# 基座：uv 管理的 cpython-3.11
# /data/home/ryeqiu/uv-python/cpython-3.11-linux-x86_64-gnu/
# tutti-env 是其 venv；torch 2.13.0+cu130（cu130 wheel）已装

source /data/home/ryeqiu/tutti-env/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 5. 编译器环境 tutti-compilers（已完成）

vllm 要求 GCC≥11.3（PyTorch C++20 headers），系统 gcc8.5 不可用：

```bash
# 已用系统 conda 装独立编译器环境（不污染 tutti-env）：
# conda create -y -p /data/home/ryeqiu/tutti-compilers -c conda-forge \
#     gcc_linux-64=12 gxx_linux-64=12

/data/home/ryeqiu/tutti-compilers/bin/x86_64-conda-linux-gnu-gcc --version
# 12.4.0
```

## 6. 编译 vLLM（进行中，T-110）

```bash
source /data/home/ryeqiu/tutti-env/bin/activate
export CUDA_HOME=/data/home/ryeqiu/Tutti/third_pkgs/nvidia/cuda-toolkit
export PATH=$CUDA_HOME/bin:/data/home/ryeqiu/.cargo/bin:$PATH
export CC=/data/home/ryeqiu/tutti-compilers/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/home/ryeqiu/tutti-compilers/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX   # 必须！nvcc host 编译器，CC/CXX 不会传给 nvcc（见 §10）
export TMPDIR=/data/home/ryeqiu/tmp
export RUSTUP_HOME=/data/home/ryeqiu/.rustup CARGO_HOME=/data/home/ryeqiu/.cargo
cd /data/home/ryeqiu/Tutti/third_pkgs/vllm

pip install packaging setuptools-rust setuptools-scm
pip install -r requirements/common.txt -r requirements/cuda.txt   # 已完成

# 后台编译（断连不影响）：
nohup bash -c 'MAX_JOBS=48 pip install -e . --no-build-isolation' \
  > /data/home/ryeqiu/log/vllm_build.log 2>&1 &
tail -f /data/home/ryeqiu/log/vllm_build.log

# 验证（编译完成后）：
python -c "import vllm; print(vllm.__version__, vllm.__file__)"
# __file__ 必须指向 third_pkgs/vllm（editable 生效）
cd third_pkgs/vllm && git status --porcelain | grep "^ M" | head   # 必须无输出
```

**踩过的坑**（已解决）：① CUDA_HOME 未设置 → 报错
`AssertionError: CUDA_HOME is not set`；② GCC 8.5 →
`GCC >= 11.3 is required`；③ PEP517 元数据阶段挂起（pip -q 无输出假象，
改用 tee 实时日志）；④ 未设 `CUDAHOSTCXX` → nvcc 用系统 gcc 8.5 当 host，
`-std=c++20` 被静默忽略，CCCL 报 `CUB requires at least C++17`（详见 §10）。

## 7. 每次使用

```bash
source /data/home/ryeqiu/env-tutti.sh
```

`env-tutti.sh` 内容（已写好）：

```bash
source /data/home/ryeqiu/tutti-env/bin/activate
export TMPDIR=/data/home/ryeqiu/tmp PIP_CACHE_DIR=/data/home/ryeqiu/pip-cache
export CUDA_HOME=/data/home/ryeqiu/Tutti/third_pkgs/nvidia/cuda-toolkit
export PATH=$CUDA_HOME/bin:/data/home/ryeqiu/.cargo/bin:$PATH
export CC=/data/home/ryeqiu/tutti-compilers/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/home/ryeqiu/tutti-compilers/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX   # nvcc host 编译器（缺它 → CUB requires at least C++17）
export RUSTUP_HOME=/data/home/ryeqiu/.rustup CARGO_HOME=/data/home/ryeqiu/.cargo
export VLLM_SOURCE=/data/home/ryeqiu/Tutti/third_pkgs/vllm
```

## 8. vLLM 端到端测试（R2 起生效）

### 冒烟（T-202 产出 `start_tutti_vllm.sh`）

```bash
bash <scripts>/start_tutti_vllm.sh --detach     # qwen2-5-7B 起步
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2-5-7B","prompt":"Hello","max_tokens":16}'
```

配置核心（外部 connector，零 vLLM 侵入，D-005）：

```python
kv_transfer_config = {
    "kv_connector": "TuttiConnectorV1",
    "kv_connector_module_path": "adapter.connector",
    "kv_role": "kv_both",
    "extra_config": {"capacity_bytes": ...,   # 必需；硬件无关
                     "blocks_per_chunk": 0}}  # 0=自动（segment≥256KiB 对齐）
```

验收口径：同一 prompt 两遍，第二遍日志出现外部命中，输出 token 一致。

### 在环 bench（T-203）

LocalStoreBackend 真实建池（GB 级）→ GPU-direct 写 pattern → 读回校验 →
上报 GB/s（分级对照：单盘 ~7 / NUMA 4 盘 ~28 / 全机峰值 50+GB/s）。

### trace 负载（R3，`run_tutti_bench.sh`）

`/data2/traces` 持续发压（populate/interleave/stream），结果
`/data/home/ryeqiu/log/tutti_bench_*/`（TTFT p50/p90/p99 JSON）。

## 9. 更新后重建

| 更新了什么 | 操作 |
|---|---|
| tutti C++ | `cmake --build --preset default --parallel 8` → 重装 tutti_runtime（pip install -e 幂等） |
| connector 适配层 | 无需重编，直接重启 vllm |
| engine/kernels | 重装对应包（pip install -e） |
| vllm C++/rust | 重跑 §6（增量，rust 缓存有效） |
| 内核模块/daemon | 重启机器后重跑 §3 |

## 10. 常见问题

**Q: `No space left on device`（系统盘）**
确认 TMPDIR/PIP_CACHE_DIR/RUSTUP_HOME/CARGO_HOME 全部指向 /data
（env-tutti.sh 已封装）；`df -h /data`。

**Q: vllm 编译报 `CUDA_HOME is not set`**
`export CUDA_HOME=/data/home/ryeqiu/Tutti/third_pkgs/nvidia/cuda-toolkit`。

**Q: vllm 编译报 `GCC >= 11.3 is required (found 8.5.0)`**
用 tutti-compilers 的 gcc-12：`export CC/CXX`（env-tutti.sh 已含）。

**Q: vllm 编译报 `CUB/Thrust/libcu++ requires at least C++17`（大量 .cu 齐挂）**
根因：`CC/CXX` 只管 CMake 的 C++ 侧，**不会传给 nvcc**。未设
`CUDAHOSTCXX` 时 nvcc 落回 PATH 的系统 gcc 8.5 当 host 编译器，
`-std=c++20` 被静默忽略（日志中有一行
`nvcc warning : The -std=c++20 flag is not supported with the configured
host compiler. Flag will be ignored.`），等效按 C++14 编译；而 CUDA 13.0
自带的 CCCL 3.0 要求 ≥C++17，于是所有含 CUB/Thrust 头（torch stable 头
会拉入 thrust/complex.h）的 .cu 全部 `#error`。
修复：`export CUDAHOSTCXX=/data/home/ryeqiu/tutti-compilers/bin/x86_64-conda-linux-gnu-g++`
（env-tutti.sh 已含）后重跑 §6。验证：构建日志的 nvcc 命令行应含
`-ccbin .../x86_64-conda-linux-gnu-g++`。

**Q: pip 装依赖长时间无输出（似卡死）**
`-q` + `tail` 会吞掉下载进度；改用 `tee` 实时日志（如 pip_cuda.log）。

**Q: `ls /dev/snvme*` 为空 / 拓扑显示 N/A**
daemon 未 bring-up 或内核模块未加载：重跑 §3 三步；NUMA0 先用 NUMA1。

**Q: `import tutti_runtime` 失败**
TUTTI_BUILD_DIR 未指对，或 tutti C++ 未构建：先 §2 再 §5。

**Q: vllm 装成了 pip wheel 而非源码**
`vllm.__file__` 必须含 third_pkgs/vllm，否则 uninstall 后重跑 §6。

**Q: 子 session 的 worktree 里跑测试**
全部用 `/data/home/ryeqiu/tutti-env/bin/python`（3.11，含 pytest）；
T-116/T-117 零 vllm 依赖可直接跑，T-113 起等 T-110 编译完成后。
