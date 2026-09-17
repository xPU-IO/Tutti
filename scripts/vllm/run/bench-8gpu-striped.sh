#!/usr/bin/env bash
#
# Two-request KV-reuse benchmark on 8 GPUs with 2-disk striping, with an
# optional nsys capture.
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
#   --tag NAME        label for output files
#   --pool-tag NAME   pool root suffix; use a new one for a guaranteed-cold
#                     pool instead of deleting the old one
#
# Everything lands under $TUTTI_PROFILE_ROOT (/mnt/nvme4/tutti-profile).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

USE_NSYS=0
BASELINE=0
TOKENS=10000
REUSE_PCT=80
ROUNDS=2
TAG=""
POOL_TAG="${POOL_TAG:-v1}"
MODEL="${MODEL:-/mnt/nvme4/models/Hy3-FP8}"
TP_SIZE=8
# 直连准入要求 >= 2 * num_layers（80 层模型 = 160），这是下限。
#
# 显存代价（实测三点、完全线性）：arena 槽位 = 2 x 本值，每槽位 576 KiB。
#   160 -> 320 槽位 -> 180 MiB      （下限，够用）
#   256 -> 512 槽位 -> 288 MiB
#   4096 -> 8192 槽位 -> 4608 MiB   （比下限多 4.4 GiB，实测峰值 90405->94833 MiB，
#                                     占 95.6 GiB 的 96.9%，逼近上限）
# 该内存与 KV cache 争同一块余量，故取"刚够"而非"越大越安全"。
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-}"   # 空 = 用生产侧按 2 x num_layers 的推导值

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsys)        USE_NSYS=1; shift ;;
        --baseline)    BASELINE=1; shift ;;
        --tokens)      TOKENS="$2"; shift 2 ;;
        --reuse-pct)   REUSE_PCT="$2"; shift 2 ;;
        --rounds)      ROUNDS="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --pool-tag)    POOL_TAG="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        -h|--help)     sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# 8 GPUs, not the 4 the old environment file hardcoded.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# shellcheck source=/dev/null
source "$SCRIPT_DIR/profile-env.sh"

PYTHON="${TUTTI_PYTHON:?profile-env.sh did not set TUTTI_PYTHON}"
DRIVER="$REPO_ROOT/scripts/vllm/vllm_profile_offline.py"

[[ -x "$PYTHON" ]] || { echo "interpreter missing: $PYTHON" >&2; exit 1; }
[[ -f "$DRIVER" ]]  || { echo "driver missing: $DRIVER" >&2; exit 1; }
[[ -d "$MODEL" ]]   || { echo "model missing: $MODEL" >&2; exit 1; }

MODE=$([[ "$BASELINE" == 1 ]] && echo hbm || echo tutti)
STAMP="$(date +%Y%m%d-%H%M%S)"
NAME="8gpu-${MODE}-${TOKENS}-reuse${REUSE_PCT}${TAG:+-$TAG}-$STAMP"
LOG="$TUTTI_PROFILE_ROOT/logs/$NAME.log"

# --- capacity sizing -------------------------------------------------------
# Pool capacity is NOT set here: the driver derives it itself
# (vllm_profile_offline.py:319 sizes num_chunks from tokens and rounds, with a
# floor of 10000, and separately raises the high watermark to cover the whole
# run). Passing a capacity from outside would fight that logic.
#
# The sizing rule it implements matters, so recording it here: the pool must
# hold every distinct chunk the whole run touches. Sizing to one request's worth
# is the classic mistake -- the pool then grows mid-request, and growth writes
# real zeros plus fsync, which blocks the forward thread.
CHUNKS_PER_REQUEST=$(( (TOKENS + 255) / 256 ))

ARGS=(
    "$DRIVER"
    --model "$MODEL"
    --tensor-parallel-size "$TP_SIZE"
    --tokens "$TOKENS"
    --reuse-pct "$REUSE_PCT"
    --rounds "$ROUNDS"
    --max-tokens 8
    # 轮与轮之间重置：让下一轮的 A 仍是冷请求。
    --reset-local-prefix-between-rounds
    # The installed flashinfer predates the set_autotune_process_group symbol
    # this vLLM revision imports, so autotuning aborts every worker at warmup.
    # Skipping it costs some kernel-selection quality but is uniform across
    # baseline and Tutti runs, so comparisons stay valid.
    --disable-flashinfer-autotune
)

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
    # rank0-3 -> disks {0,1}, rank4-7 -> disks {2,3}. Each rank gets its own
    # pool root: several ranks sharing one root would append to the same files.
    #
    # --kv-layout striped is mandatory here, not cosmetic: without it the driver
    # takes its single-device branch and derives device_id={LOCAL_RANK}, so rank
    # 5 asks the daemon for NVMe device 5 on a 4-disk host and every worker dies
    # with "daemon 配置无 device_id=5 的 NVMe 条目".
    #
    # --device-groups is semicolon-between-groups, comma-within-group. A JSON
    # spelling like '[[0,1],[2,3]]' parses to nonsense and silently falls back to
    # the same failure.
    ARGS+=(
        --kv-layout striped
        --device-groups '0,1;2,3'
        # {LOCAL_RANK} 是 driver 的占位符（大括号、无 $）。写成 shell 的
        # ${LOCAL_RANK} 会被转义成字面量 "$0"，于是 8 个 rank 全写进同一个名为
        # "...-$0" 的目录——池按 rank 隔离的语义失效，且互相追加同名文件。
        --kv-root "/mnt/nvme0/tutti-kv-8gpu-${POOL_TAG}-{LOCAL_RANK}"
        # 64 KiB. A TP8 layer segment is 128 KiB, so a layer spans exactly two
        # stripes -- one per disk -- without cutting an entry in half.
        --stripe-unit 65536
        # Per device. The default of 32 would have 4 ranks x 32 = 128 queues on a
        # disk whose usable pool is 119, which fails with EAGAIN.
        --num-queues 8
        # 默认不传 --max-in-flight-operations：生产侧（factory）已按
        # 2 x num_layers 推导出下限，此处保持默认即是在验证那条路径。
        # MAX_IN_FLIGHT=N 可显式覆盖，用于实验更大配额。
        # 不再需要 --direct-transfer-strict：staged 暂存路径已退役，直连准入
        # 失败一律抛出，strict 成为恒真行为（该键仍被接受但不再读取）。
        --kv-load-failure-policy fail
    )
fi

# 仅在显式指定时覆盖推导值——默认留空才能验证生产推导路径。
if [[ -n "${MAX_IN_FLIGHT:-}" ]]; then
    ARGS+=( --max-in-flight-operations "$MAX_IN_FLIGHT" )
    echo "[bench] 显式覆盖 max_in_flight_operations=$MAX_IN_FLIGHT"
fi

# No deletion path on purpose. A populated pool holds ~10k slot files per rank,
# and bulk removal is intercepted by a safe-delete shim -- which silently killed
# this script on its first run, leaving the pools intact and no log behind.
# Switching POOL_TAG gives a guaranteed-cold pool without deleting anything, and
# keeps the previous run's data available for comparison.

echo "[bench] mode=$MODE tokens=$TOKENS reuse=${REUSE_PCT}% rounds=$ROUNDS tp=$TP_SIZE"
[[ "$BASELINE" == 1 ]] || echo "[bench] chunks_per_request=$CHUNKS_PER_REQUEST pool_tag=$POOL_TAG (pool sized by driver)"
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
    nsys profile \
        --output "$REPORT" \
        --force-overwrite true \
        --trace cuda,nvtx \
        --capture-range=nvtx \
        --nvtx-capture='tutti.bench@tutti' \
        --capture-range-end=none \
        --cuda-memory-usage false \
        --sample none \
        --cpuctxsw none \
        "$PYTHON" "${ARGS[@]}" 2>&1 | tee "$LOG"
else
    "$PYTHON" "${ARGS[@]}" 2>&1 | tee "$LOG"
fi

RC="${PIPESTATUS[0]}"
echo
echo "[bench] exit=$RC"
grep -E '^\[SUMMARY\]|wall=' "$LOG" | tail -20 || true
exit "$RC"
