#!/usr/bin/env bash
#
# Two-request KV-reuse benchmark with selectable TP/device groups and optional
# nsys capture. The historical filename does not require eight GPUs.
#
# The workload is a pair of requests per round: request A is cold, request B
# shares a prefix with A (--reuse-pct). What is being measured is how much of
# B's prefill is avoided by reusing A's KV.
#
#   --without-tutti   B reuses A's KV from vLLM's own HBM prefix cache. This is
#                     the baseline: no storage layer involved at all.
#   default           B reuses A's KV from NVMe through the Tutti connector.
#
# The HBM baseline is the reference for "is the storage path introducing
# bubbles": it shows what the same reuse looks like with zero storage work, so
# any extra gap in the Tutti timeline is attributable to storage.
#
# Usage:
#   scripts/vllm/run/bench-8gpu-striped.sh [--nsys] [--baseline] [options]
#
#   --nsys            capture an nsys report (adds real overhead; wall times
#                     from an nsys run are not comparable to a clean run)
#   --baseline        run with --without-tutti (HBM reuse only)
#   --tokens N        prompt length, default 10000
#   --reuse-pct N     shared prefix percentage, default 80
#   --rounds N        rounds of (A,B), default 2
#   --max-tokens N    output token limit, default 8; use 1 for prefill timing
#   --capacity SIZE   per-rank KV payload budget, e.g. "1 TiB"; required for Tutti
#   --block-size N    backend KV block tokens; MiniMax M3 uses 128
#   --kv-load-mode M  sync/async/auto; sync forces external-read validation
#   --load-format F   weight loader, default LOAD_FORMAT or auto
#   --enable-expert-parallel    use expert parallelism across TP ranks
#   --enable-ep-weight-filter   load only rank-local experts
#   --kv-cache-dtype D          main KV dtype, e.g. fp8
#   --indexer-kv-dtype D        sparse index KV dtype, e.g. bf16
#   --max-model-len N           override tokens + 8
#   --max-num-seqs N            override max(256, batch_size)
#   --max-num-batched-tokens N  override vLLM's scheduling token budget
#   --gpu-memory-utilization F  override vLLM's memory fraction
#   --language-model-only      skip multimodal components
#   --replica-io                enable cooperative Tutti reads (not baseline)
#   --payload-tag NAME          isolate segment payloads for this experiment
#   --tag NAME        label for output files
#   --pool-tag NAME   pool root suffix (checkpoint/metadata root). Segment
#                     files live under <mount>/r<rank>/segments and are shared
#                     across pool tags; a new tag only gives a fresh index.
#
# Set MODEL and TUTTI_PROFILE_ROOT (an NVMe scratch directory) before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

USE_NSYS=0
BASELINE=0
TOKENS=10000
REUSE_PCT=80
ROUNDS=2
MAX_TOKENS=8
TAG=""
POOL_TAG="${POOL_TAG:-v1}"
# 盘组部署，与 serve-8gpu-striped.sh 一致（默认：两个 RAID0，各一个 daemon）：
#   DAEMON_CONFIG  各 daemon 配置，逗号分隔；KV_MOUNT 池根挂载点；
#   DEVICE_GROUPS  按 rank 分组的 device_id（分号分组）。
DAEMON_CONFIG="${DAEMON_CONFIG:-$REPO_ROOT/config/local/tutti_daemon_kv0.yaml,$REPO_ROOT/config/local/tutti_daemon_md0.yaml}"
KV_MOUNT="${KV_MOUNT:-/mnt/tutti_md0}"
DEVICE_GROUPS="${DEVICE_GROUPS:-0,1,2,3;4,5,6,7}"
MODEL="${MODEL:-}"
TP_SIZE="${TP_SIZE:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
CHUNK_TOKENS="${CHUNK_TOKENS:-256}"
SHARED_TOKENS=""
KV_LOAD_MODE="${KV_LOAD_MODE:-async}"
NUM_QUEUES="${NUM_QUEUES:-8}"
[[ "$NUM_QUEUES" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_QUEUES must be a positive integer" >&2; exit 2; }
# 空值不传：connector按模型和请求加载模式推导in-flight；不是越大越快。
# arena显存还依赖max_batch_entries，不能复制另一个模型的固定MiB预算。
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-}"
# Standard vLLM loading requires no Phoenix installation. Alternative weight
# loaders may be selected explicitly via LOAD_FORMAT and PHXLOADER_DIR.
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
MODEL_ARGS=()
REPLICA_IO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsys)        USE_NSYS=1; shift ;;
        --baseline)    BASELINE=1; shift ;;
        --tokens)      TOKENS="$2"; shift 2 ;;
        --reuse-pct)   REUSE_PCT="$2"; shift 2 ;;
        --rounds)      ROUNDS="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --pool-tag)    POOL_TAG="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --capacity)    CAPACITY="$2"; shift 2 ;;
        --block-size)  BLOCK_SIZE="$2"; shift 2 ;;
        --chunk-tokens) CHUNK_TOKENS="$2"; shift 2 ;;
        --shared-tokens) SHARED_TOKENS="$2"; shift 2 ;;
        --kv-load-mode) KV_LOAD_MODE="$2"; shift 2 ;;
        --load-format) LOAD_FORMAT="$2"; shift 2 ;;
        --enable-expert-parallel|--enable-ep-weight-filter|--language-model-only)
            MODEL_ARGS+=( "$1" ); shift ;;
        --kv-cache-dtype|--indexer-kv-dtype|--max-model-len|--gpu-memory-utilization|--max-num-seqs|--max-num-batched-tokens)
            MODEL_ARGS+=( "$1" "$2" ); shift 2 ;;
        --replica-io)   REPLICA_IO=1; shift ;;
        --payload-tag) export TUTTI_E2E_PAYLOAD_TAG="$2"; shift 2 ;;
        -h|--help)     sed -n '2,46p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

: "${MODEL:?Set MODEL or pass --model to select the checkpoint}"
# GPU selection belongs to the caller.
# shellcheck source=/dev/null
source "$SCRIPT_DIR/profile-env.sh"

PYTHON="${TUTTI_PYTHON:?profile-env.sh did not set TUTTI_PYTHON}"
DRIVER="$REPO_ROOT/scripts/vllm/vllm_profile_offline.py"

# 可用调用者的 PYTHONPATH；显式 PHXLOADER_DIR 优先，driver 校验包能否导入。
if [[ "$LOAD_FORMAT" == "phxsafetensors" && -n "${PHXLOADER_DIR:-}" ]]; then
    [[ -f "$PHXLOADER_DIR/phxloader/__init__.py" ]] || {
        echo "phxloader missing: $PHXLOADER_DIR/phxloader (set PHXLOADER_DIR)" >&2; exit 1;
    }
    export PYTHONPATH="$PHXLOADER_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

command -v "$PYTHON" >/dev/null || { echo "interpreter missing: $PYTHON" >&2; exit 1; }
[[ -f "$DRIVER" ]] || { echo "driver missing: $DRIVER" >&2; exit 1; }
if (( ! BASELINE )) && ! mountpoint -q "$KV_MOUNT"; then
    echo "KV_MOUNT is not mounted: $KV_MOUNT" >&2
    exit 1
fi

MODE=$([[ "$BASELINE" == 1 ]] && echo hbm || echo tutti)
STAMP="$(date +%Y%m%d-%H%M%S)"
NAME="${TP_SIZE}gpu-${MODE}-${TOKENS}-reuse${REUSE_PCT}${TAG:+-$TAG}-$STAMP"
LOG="$TUTTI_PROFILE_ROOT/logs/$NAME.log"

# 每rank显式payload预算，随模型/批量/轮数检查工作集；不猜模型KV字节数。
if (( ! BASELINE )); then
    : "${CAPACITY:?Set CAPACITY or pass --capacity, for example 1 TiB per rank}"
fi
CHUNKS_PER_REQUEST=$(( (TOKENS + 255) / 256 ))

ARGS=(
    "$DRIVER"
    --model "$MODEL"
    --load-format "$LOAD_FORMAT"
    --tensor-parallel-size "$TP_SIZE"
    --block-size "$BLOCK_SIZE"
    --kv-load-mode "$KV_LOAD_MODE"
    --tokens "$TOKENS"
    --reuse-pct "$REUSE_PCT"
    --rounds "$ROUNDS"
    --max-tokens "$MAX_TOKENS"
    # 轮与轮之间重置：让下一轮的 A 仍是冷请求。
    --reset-local-prefix-between-rounds
    # The installed flashinfer predates the set_autotune_process_group symbol
    # this vLLM revision imports, so autotuning aborts every worker at warmup.
    # Skipping it costs some kernel-selection quality but is uniform across
    # baseline and Tutti runs, so comparisons stay valid.
    --disable-flashinfer-autotune
)
ARGS+=( --chunk-tokens "$CHUNK_TOKENS" )
if [[ -n "$SHARED_TOKENS" ]]; then
    ARGS+=( --shared-tokens "$SHARED_TOKENS" )
fi
ARGS+=( "${MODEL_ARGS[@]}" )

if [[ "$BASELINE" == 1 ]]; then
    ARGS+=( --without-tutti )
else
    # A 与 B 之间清掉 vLLM 自己的 HBM 前缀缓存，但保留 Tutti 缓存
    # （reset_prefix_cache(reset_connector=False)）。
    #
    # 这一项决定整个实验是否成立：不加它，B 的复用全部来自 HBM 前缀缓存，
    # Tutti 的读路径根本不会被触发——nsys 里表现为 io_kernel|op=read 完全缺失、
    # 读计划 0 次，而 B 的墙钟又很漂亮（因为它确实复用了，只是没经过 NVMe），
    # 于是极易误判为"直连读很快"。
    ARGS+=( --reset-local-prefix-between-requests )
    if (( REPLICA_IO )); then
        ARGS+=( --replica-io )
    fi
    # Groups partition TP ranks evenly. IDs are from daemon YAML, not nvmeN.
    # Syntax: comma-separated IDs per group, semicolon-separated groups.
    ARGS+=(
        --capacity "$CAPACITY"
        --device-groups "$DEVICE_GROUPS"
        # {LOCAL_RANK} 是 driver 的占位符（大括号、无 $）。写成 shell 的
        # ${LOCAL_RANK} 会被转义成字面量 "$0"，于是 8 个 rank 全写进同一个名为
        # "...-$0" 的目录——池按 rank 隔离的语义失效，且互相追加同名文件。
        --kv-root "${KV_MOUNT}/tutti-kv-8gpu-${POOL_TAG}-{LOCAL_RANK}"
        --daemon-config "$DAEMON_CONFIG"
        # Per rank, per device. Check ranks-per-device times queues against
        # the actual daemon grants; another host's queue pool is not a limit.
        --num-queues "$NUM_QUEUES"
        # MAX_IN_FLIGHT只在显式设置时覆盖模型准入推导。
        --kv-load-failure-policy fail
    )
fi

# 仅在显式指定时覆盖推导值——默认留空才能验证生产推导路径。
if [[ -n "${MAX_IN_FLIGHT:-}" ]]; then
    ARGS+=( --max-in-flight-operations "$MAX_IN_FLIGHT" )
    echo "[bench] 显式覆盖 max_in_flight_operations=$MAX_IN_FLIGHT"
fi

# No deletion or media reset here. A fresh POOL_TAG changes only the index;
# payload segment files remain shared at <mount>/r<rank>/segments.

echo "[bench] mode=$MODE tokens=$TOKENS reuse=${REUSE_PCT}% rounds=$ROUNDS tp=$TP_SIZE"
[[ "$BASELINE" == 1 ]] || echo "[bench] chunks_per_request=$CHUNKS_PER_REQUEST pool_tag=$POOL_TAG capacity_per_rank=$CAPACITY"
echo "[bench] log: $LOG"

cd "$REPO_ROOT"

if [[ "$USE_NSYS" == 1 ]]; then
    REPORT="$TUTTI_PROFILE_ROOT/reports/$NAME"
    echo "[bench] nsys report: $REPORT.nsys-rep"
    # 采集口径对齐历史可用报告（hy3-tp8-10k-striped2-nsys-multiround，约 210MB、
    # GPU 活动仅 15.8s），三项设置各有实测依据：
    #
    #  ① 只采集基准段，跳过引擎预热
    #     driver 在基准循环前推入固定名 NVTX 范围 "tutti.bench"（域 tutti），
    #     基线与 Tutti 两侧都会发出，因此两份报告同口径。
    #     --capture-range-end=none 让采集从该范围开始后持续到进程结束（其后只剩
    #     汇总输出，无 GPU 工作）。
    #
    #     注意本机 nsys 2025.3.2 的两个限制（已用最小程序逐一验证）：
    #     --nvtx-capture **不支持通配符**，且**必须写 @domain**（省略即匹配失败，
    #     报 "No reports were generated"）。所以不能用请求范围 'tutti.request*'
    #     作触发——请求名带轮次后缀（A-cold|r0 等）各不相同，无法用单个精确名
    #     覆盖，这也是引入 "tutti.bench" 的原因。
    #
    #     不这么做时预热会被整段采集：模型 299.9GB 的加载产生 61 万次 768KB 的
    #     H2D 拷贝（468GB）外加 110 万次 4 字节 H2D，在 166s 的报告里占 36-79s，
    #     而真正的基准段只有 179-194s——报告 488MB vs 历史 210MB、kernel 数 60 倍
    #     的差距主要来自这里，且与存储路径无关，纯属噪音。
    #
    #  ② 不采 osrt：它贡献 477 万条事件（历史同类报告仅 49 万），且 syscall 时间线
    #     对判断 GPU 气泡无用——需要看主机阻塞时 py-spy 更直接。
    #
    #  ③ 不用 --nvtx-capture 做内容过滤：早期用过的 '*@tutti.*' 会把 NVTX 从
    #     8.1 万条砍到 3.0 万条，连 tutti.striped_nvme.io_kernel|op=read 这类关键
    #     范围一起滤掉，正好丢掉最该看的部分。这里的 --nvtx-capture 只作为采集
    #     触发条件，不过滤内容。
    #  ④ 默认不采 cublas/cudnn，但可用 TUTTI_NSYS_TRACE 打开：请求开头的
    #     250ms 级 compute 气泡期间既无 kernel 也无 CUDA runtime 调用，时间就
    #     花在未采集的库调用（cuBLASLt 启发式搜索、cuDNN）里——不看这两类
    #     就永远只能看到"空白"。代价是事件数量显著增加。
    #  ⑤ GPU metrics（1 kHz）：PCIe 带宽利用率在报告的 GPU metrics (GPUm)
    #     表里——pcie_rx/pcie_tx 是每卡吞吐计数器，nvlink 域为 0（无 NVLink
    #     流量时）。8 卡全采（cuda-visible 跟随 CUDA_VISIBLE_DEVICES）。
    nsys profile \
        --output "$REPORT" \
        --force-overwrite true \
        --trace "${TUTTI_NSYS_TRACE:-cuda,nvtx}" \
        --capture-range=nvtx \
        --nvtx-capture='tutti.bench@tutti' \
        --capture-range-end=none \
        --cuda-memory-usage false \
        --sample none \
        --cpuctxsw none \
        --gpu-metrics-devices=cuda-visible \
        --gpu-metrics-frequency=1000 \
        "$PYTHON" "${ARGS[@]}" 2>&1 | tee "$LOG"
else
    "$PYTHON" "${ARGS[@]}" 2>&1 | tee "$LOG"
fi

RC="${PIPESTATUS[0]}"
echo
echo "[bench] exit=$RC"
grep -E '^\[SUMMARY\]|wall=' "$LOG" | tail -20 || true
exit "$RC"
