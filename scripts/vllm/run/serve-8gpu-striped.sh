#!/usr/bin/env bash
#
# Tutti 在线服务：Hy3-FP8 × TP8 × 2 盘条带，与 8 卡离线基准同构。
#
# 与 vllm_smoke_server.sh 的区别：那份是 TP4 + 单盘 + 旧模块路径（adapter.connector，
# 已随包重构失效），这份对齐 scripts/vllm/run/bench-8gpu-striped.sh 的几何与
# 部署形态，可直接配合 vllm_online_longrun.py 做在线长跑。
#
# 用法：
#   RUN_DIR=/mnt/nvme4/tutti-profile/online/<tag> \
#     nohup bash scripts/vllm/run/serve-8gpu-striped.sh > $RUN_DIR/server.log 2>&1 &
#
# 容量推导（与 offline driver 同一规则，缺少它会踩两类坑）：
#   * initial_slots 必须覆盖单请求整波 chunk，否则 PoolResourceExhausted；
#   * high_watermark 必须覆盖整个工作集，否则请求中途扩容（实零写 + fsync）
#     会阻塞前向线程。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PORT="${PORT:-8192}"
# 工具调用（tool choice=auto 必需）：hy_v3 是 vLLM 为 HYV3ForCausalLM 内置的
# 解析器；换模型时按 vllm 的 --tool-call-parser 列表改（hermes/qwen3_coder…）。
# 这两个参数不影响 Tutti 的 key 命名空间，旧池数据继续可复用。
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hy_v3}"
MODEL="${MODEL:-/mnt/nvme4/models/Hy3-FP8}"
SERVED_NAME="${SERVED_NAME:-tutti}"
TP_SIZE="${TP_SIZE:-8}"
CHUNK_TOKENS="${CHUNK_TOKENS:-256}"
# 在线驱动只发不超过该长度的 prompt（脚本默认 --max-prompt-tokens 与之一致）。
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-20000}"
SAMPLES="${SAMPLES:-4}"
POOL_TAG="${POOL_TAG:-online1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_TOKENS + 16))}"

# HBM KV 池上限（blocks，block_size=64）。**隔离 Tutti 路径的关键旋钮**：
# 本机 vLLM 未注册 /reset_prefix_cache 端点（endpoint plugin 未注册），无法
# 清 HBM 前缀缓存，只能把池压到装不下测试集，迫使 HBM 淘汰、复用落到 Tutti。
# 本机默认 HBM 池是 1,289,081 tokens（约 20141 blocks），测试集 80k tokens
# 只占 6%——不压小则第二轮起全部命中 HBM，Tutti 完全不参与。
# 取值约束：
#   * ≥ 单个最长 prompt 所需 blocks（MAX_PROMPT_TOKENS/64 ≈ 313），否则调度
#     器无法容纳一个请求；
#   * < 测试集总 blocks（SAMPLES × 每请求 blocks），否则不会发生淘汰。
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-768}"

# 数据盘物理总量（方案 A）：直接写"这些盘一共占多少"，由 Python 层按条带
# 几何换算成槽位数（geometry.apply_capacity_bytes）。默认 4 TiB → 四盘各
# 约 1 TiB。用了它就不要同时给 num_chunks（两者互斥）。
CAPACITY_BYTES="${CAPACITY_BYTES:-$(( 4 * 1024 * 1024 * 1024 * 1024 ))}"
# 同步预热槽位数：open() 期只付这一份（首个请求 + 头几秒的工作集）。
# 其余容量由后台线程异步铺开（TUTTI_MATERIALIZE_*），写路径永不 create+fsync
# 槽位——10 TB 级的容量因此不再等于几十分钟启动或 44ms/槽的前向停顿。
HIGH_WATERMARK="${HIGH_WATERMARK:-4096}"
# 后台预建：线程数（0 = 关闭，退回写路径按需建槽）。口径见 TUTTI_PRECREATE_SCOPE。
# 余量 = 保持"分配前沿之前"多少个槽位已就绪；跟不上需求时写入被裁剪（不阻塞）。
export TUTTI_PRECREATE_THREADS="${TUTTI_PRECREATE_THREADS:-4}"
export TUTTI_PRECREATE_HEADROOM="${TUTTI_PRECREATE_HEADROOM:-4096}"
# 工作集（仅用于日志展示）
CHUNKS_PER_REQ=$(( (MAX_PROMPT_TOKENS + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))
WORKING_SET=$(( SAMPLES * CHUNKS_PER_REQ ))

echo "[serve] model=$MODEL tp=$TP_SIZE port=$PORT max_model_len=$MAX_MODEL_LEN"
echo "[serve] HBM KV 上限: ${NUM_GPU_BLOCKS_OVERRIDE} blocks" \
     "(${NUM_GPU_BLOCKS_OVERRIDE} × 64 = $(( NUM_GPU_BLOCKS_OVERRIDE * 64 )) tokens)"
echo "[serve] chunks_per_req=$CHUNKS_PER_REQ working_set=$WORKING_SET"
echo "[serve] pool: capacity_bytes=$CAPACITY_BYTES" \
     "($(( CAPACITY_BYTES / 1024 / 1024 / 1024 / 1024 )) TiB 总量)" \
     "prewarm_slots=$HIGH_WATERMARK"
echo "[serve] pool_tag=$POOL_TAG (换 tag = 全新冷池，脚本不删旧池)"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/profile-env.sh"
PYTHON="${TUTTI_PYTHON:?profile-env.sh did not set TUTTI_PYTHON}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Phoenix GDS 权重加载器（phxloader）：直接从 NVMe 读到显存，绕过 page cache。
# 冷缓存下标准 safetensors 加载每 shard 10s+（300GB 要 ~18 分钟），GDS 不受
# page cache 冷热影响。phxloader 未编译/未安装时自动回退 --load-format auto。
PHXLOADER_DIR="${PHXLOADER_DIR:-/data/home/ryeqiu/phoenix/adapters/vLLM/phxloader}"
if [ -f "$PHXLOADER_DIR/phxloader/__init__.py" ] &&
   ls "$PHXLOADER_DIR"/phxloader/_phxloader*.so >/dev/null 2>&1; then
    export PYTHONPATH="$PHXLOADER_DIR${PYTHONPATH:+:$PYTHONPATH}"
    LOAD_FORMAT="${LOAD_FORMAT:-phxsafetensors}"
else
    echo "[serve] 警告：phxloader 不可用（缺 _phxloader*.so）→ 回退标准加载"
    LOAD_FORMAT="${LOAD_FORMAT:-auto}"
fi
echo "[serve] load_format=$LOAD_FORMAT phxloader_dir=$PHXLOADER_DIR"

# rank0-3 -> 盘 {0,1}，rank4-7 -> 盘 {2,3}；每 rank 自己的池根（多 rank 共用
# 一个 root 会互相追加同名文件）。{LOCAL_RANK} 由 connector 的
# expand_placeholders 展开（大括号、无 $）。
read -r -d '' KV_CONFIG <<JSON || true
{
  "kv_connector": "TuttiConnectorV1",
  "kv_connector_module_path": "tutti.integration.vllm.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "chunk_tokens": $CHUNK_TOKENS,
    "max_chunks_per_wave": 512,
    "store": {
      "type": "tutti_nvme",
      "options": {
        "root": "/mnt/nvme0/tutti-kv-online-${POOL_TAG}-{LOCAL_RANK}",
        "capacity_bytes": $CAPACITY_BYTES,
        "high_watermark": $HIGH_WATERMARK,
        "io_stream": "auto",
        "layout": "striped",
        "stripe_unit": 65536,
        "preset": {
          "type": "striped",
          "daemon_config": "$REPO_ROOT/config/local/tutti_daemon.yaml",
          "gpu_id": "{LOCAL_RANK}",
          "device_groups": [[0, 1], [2, 3]],
          "stripe_unit": 65536,
          "num_queues": 8
        }
      }
    }
  }
}
JSON

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --block-size 64 \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --load-format "$LOAD_FORMAT" \
    --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE" \
    --enable-prefix-caching \
    --enable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --port "$PORT" \
    --no-enable-flashinfer-autotune \
    --kv-transfer-config "$KV_CONFIG"
