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
# 对外服务名：流量平台按服务组名 base_model_zh7 路由请求，必须挂成 vLLM
# 别名，否则 404 "model does not exist"。tutti 保留给本地脚本（driver 默认
# --model tutti）。vLLM 的 --served-model-name 接受空格分隔的多名字。
SERVED_NAME="${SERVED_NAME:-tutti base_model_zh7}"
TP_SIZE="${TP_SIZE:-8}"
CHUNK_TOKENS="${CHUNK_TOKENS:-256}"
# vLLM 的 KV block 大小（token 数）。决定 Tutti 侧的两条几何：
#   page_bytes       = BLOCK_SIZE × 512 B   （单条 IO 的大小）
#   blocks_per_chunk = CHUNK_TOKENS / BLOCK_SIZE（必须整除）
# 取 128：单条 IO 64 KiB，一个 chunk = 2 条 IO。
# 改这个值必须同步改 NUM_GPU_BLOCKS_OVERRIDE —— 后者的单位是 block 而非
# token，block 定义翻倍而该值不变会让 HBM KV 池占用翻倍（反之则缩水）。
BLOCK_SIZE="${BLOCK_SIZE:-128}"
# 在线驱动只发不超过该长度的 prompt（脚本默认 --max-prompt-tokens 与之一致）。
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-262144}"
SAMPLES="${SAMPLES:-4}"
POOL_TAG="${POOL_TAG:-online1}"
# 模型上限来自 config.json 的 max_position_embeddings；MAX_MODEL_LEN 允许比
# prompt 上限多 16（vLLM 惯例），但不得超过模型上限（264144 会直接启动失败）。
MODEL_MAX_LEN="${MODEL_MAX_LEN:-262144}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_TOKENS + 16 < MODEL_MAX_LEN ? MAX_PROMPT_TOKENS + 16 : MODEL_MAX_LEN ))}"

# HBM KV 池上限（单位：block）。默认**不传**：用 vLLM 自然池（本机 ~10077
# blocks = 1.29M tokens，TP8/bf16 下每 token 40 KiB，见 README 的几何推导）。
# 自然池可容纳 ~4.9 条 256K 或 ~9.8 条 128K 请求并发——生产服务用它。
#
# 只有"隔离 Tutti 路径做实验"时才需要压小它：本机 vLLM 未注册
# /reset_prefix_cache 端点，无法清 HBM 前缀缓存，把池压到装不下工作集才能
# 逼 HBM 淘汰、让复用落到 Tutti（离线测试的做法：1040 blocks = 1.01 并发
# 128K）。⚠️ 压小 = 并发被锁死（请求排队），线上服务不要压。
# 取值约束：≥ ceil(MAX_PROMPT_TOKENS / BLOCK_SIZE)（2048 for 256K），否则
# 调度器无法容纳一个请求。
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"

# 数据盘物理总量（方案 A）：**服务级总量**（8 卡加起来的总占用），默认 8 TiB。
# 注意 apply_capacity_bytes 的语义是"每个 rank 的容量"，而 8 个 rank 共享同一
# 组盘、各占一份 KV 分片，所以这里必须除以 TP_SIZE 再传给 store，否则每个
# rank 都按整份总量建槽 = TP 倍超发，盘会被预建线程铺满（2026-09-22 事故：
# 8TiB × 8 rank = 64TiB 需求铺满 4×5.8TB 盘，ENOSPC 后 KV 写全失败）。
# 换算示例：8 TiB 总量 → 每 rank 1 TiB → 每盘（8 rank / 4 盘）2 TiB。
TOTAL_CAPACITY_BYTES="${TOTAL_CAPACITY_BYTES:-$(( 8 * 1024 * 1024 * 1024 * 1024 ))}"
CAPACITY_BYTES=$(( TOTAL_CAPACITY_BYTES / TP_SIZE ))
# 同步预热槽位数：open() 期只付这一份（首个请求 + 头几秒的工作集）。
# 之后的增长由后台线程按需跟随（TUTTI_PRECREATE_*），写路径永不 create+fsync
# 槽位——容量因此不再等于几十分钟启动或 44ms/槽的前向停顿。
HIGH_WATERMARK="${HIGH_WATERMARK:-4096}"
# 后台预建：线程数（0 = 关闭，退回写路径按需建槽）。
# 余量 = 保持"分配前沿之前"多少个槽位已就绪；跟不上需求时写入被裁剪（不阻塞）。
# 预建只跟随分配前沿（盘上占用 ∝ 真实 KV），容量仅作硬上限——"铺满容量"的
# 口径已从代码里删除（2026-09-22：盘占用 ∝ 容量会把盘写满并与在线 KV 争带宽）。
export TUTTI_PRECREATE_THREADS="${TUTTI_PRECREATE_THREADS:-4}"
export TUTTI_PRECREATE_HEADROOM="${TUTTI_PRECREATE_HEADROOM:-4096}"
# 工作集（仅用于日志展示）
CHUNKS_PER_REQ=$(( (MAX_PROMPT_TOKENS + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))
WORKING_SET=$(( SAMPLES * CHUNKS_PER_REQ ))

echo "[serve] model=$MODEL tp=$TP_SIZE port=$PORT max_model_len=$MAX_MODEL_LEN (prompt≤$MAX_PROMPT_TOKENS)"
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    echo "[serve] HBM KV 池: ${NUM_GPU_BLOCKS_OVERRIDE} blocks（覆盖值）" \
         "(${NUM_GPU_BLOCKS_OVERRIDE} × ${BLOCK_SIZE} = $(( NUM_GPU_BLOCKS_OVERRIDE * BLOCK_SIZE )) tokens)"
else
    echo "[serve] HBM KV 池: vLLM 自然池（未覆盖；约 10077 blocks ≈ 1.29M tokens）"
fi
echo "[serve] chunks_per_req=$CHUNKS_PER_REQ working_set=$WORKING_SET"
echo "[serve] IO 几何: block_size=$BLOCK_SIZE tokens/block" \
     "blocks_per_chunk=$(( CHUNK_TOKENS / BLOCK_SIZE ))" \
     "device_groups=[[0,1,2,3]] (8 rank 共享 4 盘，槽位轮转)"
echo "[serve] pool: 服务级总量 $(( TOTAL_CAPACITY_BYTES / 1024 / 1024 / 1024 / 1024 )) TiB" \
     "(每 rank $(( CAPACITY_BYTES / 1024 / 1024 / 1024 )) GiB)" \
     "prewarm_slots=$HIGH_WATERMARK headroom=$TUTTI_PRECREATE_HEADROOM"
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
        "preset": {
          "type": "striped",
          "daemon_config": "$REPO_ROOT/config/local/tutti_daemon.yaml",
          "gpu_id": "{LOCAL_RANK}",
          "device_groups": [[0, 1, 2, 3]],
          "num_queues": 8
        }
      }
    }
  }
}
JSON

# HBM 池只在显式给了覆盖值时才传参（默认留空 = vLLM 自然池，生产口径）。
HBM_ARGS=()
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    HBM_ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    "${HBM_ARGS[@]}" \
    --model "$MODEL" \
    --served-model-name $SERVED_NAME \
    --tensor-parallel-size "$TP_SIZE" \
    --block-size "$BLOCK_SIZE" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --load-format "$LOAD_FORMAT" \
    --enable-prefix-caching \
    --enable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --port "$PORT" \
    --no-enable-flashinfer-autotune \
    --kv-transfer-config "$KV_CONFIG"
