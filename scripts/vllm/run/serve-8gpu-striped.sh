#!/usr/bin/env bash
#
# Tutti 在线服务。预置参数来自 Hy3-FP8 / TP8 双 RAID0；新机器须覆盖部署参数。
# 在线与离线 wrapper 的 block / KV dtype 默认不同，不能直接混用性能数据。
# 完整部署和 SSE 验收：doc/intergration/tutti-vllm-manual.md。
#
# 用法：先设置 MODEL、TUTTI_PROFILE_ROOT（NVMe 暂存目录）和本机盘组配置。
#   bash scripts/vllm/run/serve-8gpu-striped.sh [--dry-run]
# --dry-run 只打印 connector JSON 与启动命令，不启动服务。
#
# 数据文件位于 <mount>/r<rank>/segments，POOL_TAG 不隔离这些文件。
# 不得在同一 mount/rank 上并发运行多个服务或 benchmark。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
esac
[[ $# == 0 ]] || { echo "unknown argument: $1" >&2; exit 2; }

PORT="${PORT:-8192}"
# 工具调用（tool choice=auto 必需）。默认 hermes，与平台此前的 Mooncake 基线
# 一致：本服务只测 KV，不解析 Hy3 的工具调用。hermes 不认识 Hy3 的
# <tool_calls:opensource>，会把它当正文透传；hy_v3（Hy3 专用）会缓存它等参数，
# 在平台 max_tokens=1 的测法下正文为空、被判"答案为空"。需要结构化 tool_calls
# 时改用 TOOL_CALL_PARSER=hy_v3。不影响 Tutti 的 key 命名空间，旧池数据可复用。
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
: "${MODEL:?Set MODEL to a local model path or Hugging Face model ID}"
# 对外服务名：流量平台按服务组名 base_model_zh7 路由请求，必须挂成 vLLM
# 别名，否则 404 "model does not exist"。tutti 保留给本地脚本（driver 默认
# --model tutti）。vLLM 的 --served-model-name 接受空格分隔的多名字。
SERVED_NAME="${SERVED_NAME:-tutti base_model_zh7}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-1}"
# KV 精度。默认 fp8：Hy3-FP8 checkpoint 自带 KV scale（kv_cache_scheme: static），
# 每 token 字节减半 → HBM KV 容量约 2.4M token（bf16 时 1.23M），盘读量也减半。
# 设 auto 回到 bf16。KV 命名空间含 dtype，切换后旧池数据不会被误读（但也不能复用）。
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
# vLLM 的 KV block 大小（token 数）。决定 Tutti 侧的两条几何：
#   page_bytes       = BLOCK_SIZE × 每 token 字节（TP8：bf16 512 B / fp8 256 B，单条 IO 的大小）
#   blocks_per_chunk = CHUNK_TOKENS / BLOCK_SIZE（必须整除）
# 取 512：bf16 单条 IO 256 KiB、fp8 128 KiB，一个 chunk = 1 条 IO。
# 改这个值必须同步改 NUM_GPU_BLOCKS_OVERRIDE —— 后者的单位是 block 而非
# token，block 定义翻倍而该值不变会让 HBM KV 池占用翻倍（反之则缩水）。
BLOCK_SIZE="${BLOCK_SIZE:-512}"
# 在线驱动只发不超过该长度的 prompt（脚本默认 --max-prompt-tokens 与之一致）。
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-262144}"
POOL_TAG="${POOL_TAG:-online1}"
# 模型上限来自 config.json 的 max_position_embeddings；MAX_MODEL_LEN 允许比
# prompt 上限多 16（vLLM 惯例），但不得超过模型上限（264144 会直接启动失败）。
MODEL_MAX_LEN="${MODEL_MAX_LEN:-262144}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_TOKENS + 16 < MODEL_MAX_LEN ? MAX_PROMPT_TOKENS + 16 : MODEL_MAX_LEN ))}"
# Keep the online scheduler budget aligned with the Phoenix comparison unless
# a benchmark explicitly overrides it.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
FLASHINFER_AUTOTUNE="${FLASHINFER_AUTOTUNE:-1}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
REPLICA_IO="${REPLICA_IO:-0}"
case "$LANGUAGE_MODEL_ONLY" in
0|1) ;;
*) echo 'LANGUAGE_MODEL_ONLY must be 0 or 1' >&2; exit 2 ;;
esac
case "$REPLICA_IO" in
0|1) ;;
*) echo 'REPLICA_IO must be 0 or 1' >&2; exit 2 ;;
esac

# HBM KV 池上限（单位：block）。默认**不传**：用 vLLM 自然池。
# FP8 的可用 token 总量约 2.46M；512-token block 的块数由 vLLM 实测决定。
#
# 只有"隔离 Tutti 路径做实验"时才需要压小它：本机 vLLM 未注册
# /reset_prefix_cache 端点，无法清 HBM 前缀缓存，把池压到装不下工作集才能
# 逼 HBM 淘汰、让复用落到 Tutti（离线测试的做法：1040 blocks = 1.01 并发
# 128K）。⚠️ 压小 = 并发被锁死（请求排队），线上服务不要压。
# 取值约束：≥ ceil(MAX_PROMPT_TOKENS / BLOCK_SIZE)（512 for 256K/512），否则
# 调度器无法容纳一个请求。
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"

# 服务总payload预算；解析后除TP，不把整份预算传给每个rank。
TOTAL_CAPACITY="${TOTAL_CAPACITY:-8 TiB}"
if [[ -v TOTAL_CAPACITY_BYTES ]]; then
    echo "TOTAL_CAPACITY_BYTES 已删除，请用 TOTAL_CAPACITY，例如 8 TiB" >&2
    exit 2
fi

# 盘组部署（默认：两个 md RAID0，各由独立 daemon 管理）：
#   DAEMON_CONFIG  客户端设备清单 = 各 daemon 配置，逗号分隔（device_id 全局唯一）
#   DEVICE_GROUPS  按 rank 分组的 device_id，分号分组：rank 0-3 → kv0（NVMe 0-3，
#                  /mnt/tutti_md0），rank 4-7 → md0（NVMe 4-7，/mnt/nvme4），与 GPU
#                  所在 NUMA/PCIe switch 对应
#   KV_MOUNT       各 rank 池根（索引/检查点）所在挂载点
# 本机配置只是参考；新机器须覆盖这三个值。不要直接接管含主机数据的盘。
DAEMON_CONFIG="${DAEMON_CONFIG:-$REPO_ROOT/config/local/tutti_daemon_kv0.yaml,$REPO_ROOT/config/local/tutti_daemon_md0.yaml}"
DEVICE_GROUPS="${DEVICE_GROUPS:-0,1,2,3;4,5,6,7}"
KV_MOUNT="${KV_MOUNT:-/mnt/tutti_md0}"
NUM_QUEUES="${NUM_QUEUES:-8}"
if ! [[ "$NUM_QUEUES" =~ ^[1-9][0-9]*$ && "$TP_SIZE" =~ ^[1-9][0-9]*$ && "$DCP_SIZE" =~ ^[1-9][0-9]*$ && "$CHUNK_TOKENS" =~ ^[1-9][0-9]*$ &&
        "$BLOCK_SIZE" =~ ^[1-9][0-9]*$ && "$MAX_NUM_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_QUEUES, TP_SIZE, DCP_SIZE, CHUNK_TOKENS, BLOCK_SIZE, MAX_NUM_BATCHED_TOKENS must be positive integers" >&2
    exit 2
fi
if (( DCP_SIZE > TP_SIZE || TP_SIZE % DCP_SIZE != 0 )); then
    echo "DCP_SIZE must divide TP_SIZE and cannot exceed it" >&2
    exit 2
fi
case "$FLASHINFER_AUTOTUNE" in
1) FLASHINFER_ARGS=(--enable-flashinfer-autotune) ;;
0) FLASHINFER_ARGS=(--no-enable-flashinfer-autotune) ;;
*) echo 'FLASHINFER_AUTOTUNE must be 0 or 1' >&2; exit 2 ;;
esac
if (( CHUNK_TOKENS % (BLOCK_SIZE * DCP_SIZE) != 0 )); then
    echo "CHUNK_TOKENS must be divisible by BLOCK_SIZE * DCP_SIZE" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/profile-env.sh"
PYTHON="${TUTTI_PYTHON:?profile-env.sh did not set TUTTI_PYTHON}"
CAPACITY_BYTES="$("$PYTHON" - "$TOTAL_CAPACITY" "$TP_SIZE" <<'PY'
import sys
from tutti.common.utils import parse_capacity

try:
    size = parse_capacity(sys.argv[1]) // int(sys.argv[2])
    if size == 0:
        raise ValueError("TOTAL_CAPACITY must allocate at least one byte per rank")
except ValueError as exc:
    raise SystemExit(str(exc))
print(size)
PY
)"
DEVICE_GROUPS_JSON="[[${DEVICE_GROUPS//;/], [}]]"
# 工作集（仅用于日志展示）
CHUNKS_PER_REQ=$(( (MAX_PROMPT_TOKENS + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))
BLOCKS_PER_CHUNK=$(( CHUNK_TOKENS / BLOCK_SIZE / DCP_SIZE ))

echo "[serve] model=$MODEL tp=$TP_SIZE port=$PORT max_model_len=$MAX_MODEL_LEN (prompt≤$MAX_PROMPT_TOKENS) kv_cache_dtype=$KV_CACHE_DTYPE"
echo "[serve] decode_context_parallel_size=$DCP_SIZE"
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    echo "[serve] HBM KV 池: ${NUM_GPU_BLOCKS_OVERRIDE} blocks（覆盖值）" \
         "(${NUM_GPU_BLOCKS_OVERRIDE} × ${BLOCK_SIZE} = $(( NUM_GPU_BLOCKS_OVERRIDE * BLOCK_SIZE )) tokens)"
else
    echo "[serve] HBM KV 池: vLLM 自然池（未覆盖；实际容量见 vLLM 启动日志）"
fi
echo "[serve] chunks_per_req=$CHUNKS_PER_REQ"
echo "[serve] IO 几何: block_size=$BLOCK_SIZE tokens/block" \
     "blocks_per_chunk=$BLOCKS_PER_CHUNK" \
     "device_groups=$DEVICE_GROUPS_JSON kv_mount=$KV_MOUNT"
echo "[serve] daemon_config=$DAEMON_CONFIG num_queues_per_rank_per_device=$NUM_QUEUES"
echo "[serve] pool: 服务级payload预算 $TOTAL_CAPACITY (每rank $CAPACITY_BYTES B)"
echo "[serve] pool_tag=$POOL_TAG (换 tag = 全新索引；段文件按 rank 共享，脚本不删)"
echo "[serve] replica_io=$REPLICA_IO"


# Standard vLLM weight loading is independent of Tutti KV offload.
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
PHXLOADER_DIR="${PHXLOADER_DIR:-}"
if [[ -n "$PHXLOADER_DIR" ]]; then
    export PYTHONPATH="$PHXLOADER_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi
echo "[serve] load_format=$LOAD_FORMAT phxloader_dir=$PHXLOADER_DIR"

# json.dumps also handles spaces and quotes in paths. Validate device IDs
# against the daemon inventory before any model or KV allocation is started.
KV_CONFIG="$("$PYTHON" - "$DAEMON_CONFIG" "$DEVICE_GROUPS" "$KV_MOUNT" \
    "$POOL_TAG" "$TP_SIZE" "$CHUNK_TOKENS" "$CAPACITY_BYTES" "$NUM_QUEUES" <<'PY'
import json
import os
import sys
from pathlib import Path

import yaml

paths, group_text, mount, tag, tp, chunk_tokens, capacity, queues = sys.argv[1:]
paths = [item.strip() for item in paths.split(",") if item.strip()]
groups = [[int(item) for item in group.split(",")] for group in group_text.split(";")]
if not groups or int(tp) % len(groups):
    raise SystemExit("DEVICE_GROUPS group count must divide TP_SIZE")
devices = set()
for path in paths:
    config = yaml.safe_load(Path(path).read_text())
    for entry in config["nvmes"]:
        device_id = entry["device_id"]
        if device_id in devices:
            raise SystemExit(f"duplicate device_id: {device_id}")
        devices.add(device_id)
if not paths or any(device not in devices for group in groups for device in group):
    raise SystemExit("DEVICE_GROUPS contains devices missing from DAEMON_CONFIG")
if not Path(mount).is_absolute() or not tag or Path(tag).name != tag or tag in (".", ".."):
    raise SystemExit("KV_MOUNT must be absolute and POOL_TAG must be a directory name")
extra = {
    "chunk_tokens": int(chunk_tokens),
    "store": {"type": "tutti_nvme", "options": {
        "root": str(Path(mount) / f"tutti-kv-online-{tag}-{{LOCAL_RANK}}"),
        "capacity": f"{capacity} B",
        "preset": {
            "type": "raid0", "daemon_config": paths,
            "gpu_id": "{LOCAL_RANK}", "device_groups": groups,
            "num_queues": int(queues),
        },
    }},
}
if os.environ.get("REPLICA_IO", "0") == "1":
    extra["replica_io"] = True
print(json.dumps({
    "kv_connector": "TuttiConnectorV1",
    "kv_connector_module_path": "tutti.integration.vllm.connector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": extra,
}))
PY
)"
if (( ! DRY_RUN )) && ! mountpoint -q "$KV_MOUNT"; then
    echo "KV_MOUNT is not mounted: $KV_MOUNT (refusing to write metadata on the root disk)" >&2
    exit 1
fi

# HBM 池只在显式给了覆盖值时才传参（默认留空 = vLLM 自然池，生产口径）。
HBM_ARGS=()
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    HBM_ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi

# Piecewise compilation keeps the connector's attention callbacks eager
# between captured graph segments. ENFORCE_EAGER=0 opts into this path.
COMPUTE_ARGS=()
case "${ENFORCE_EAGER:-1}" in
1)
    COMPUTE_ARGS+=(--enforce-eager)
    ;;
0)
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    COMPUTE_ARGS+=(--compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}')
    ;;
*) echo 'ENFORCE_EAGER must be 0 or 1' >&2; exit 2 ;;
esac

read -r -a SERVED_NAMES <<< "$SERVED_NAME"
COMMAND=(
    "$PYTHON" -m vllm.entrypoints.openai.api_server
    "${HBM_ARGS[@]}"
    --model "$MODEL"
    --served-model-name "${SERVED_NAMES[@]}"
    --tensor-parallel-size "$TP_SIZE"
    --decode-context-parallel-size "$DCP_SIZE"
    --block-size "$BLOCK_SIZE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    "${COMPUTE_ARGS[@]}"
    --max-model-len "$MAX_MODEL_LEN"
    --load-format "$LOAD_FORMAT"
    --enable-prefix-caching
    --enable-log-requests
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    --port "$PORT"
    "${FLASHINFER_ARGS[@]}"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --kv-transfer-config "$KV_CONFIG"
)
if [[ "$LANGUAGE_MODEL_ONLY" == 1 ]]; then
    COMMAND+=(--language-model-only)
fi
if (( DRY_RUN )); then
    echo "[serve] kv_config=$KV_CONFIG"
    printf '[serve] command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
    exit 0
fi
exec "${COMMAND[@]}"
