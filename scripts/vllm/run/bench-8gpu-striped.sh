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
    # Round-to-round isolation: without this, round 2's request A would hit
    # vLLM's HBM prefix cache from round 1 and stop measuring a cold request.
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
        # 直连路径要求在飞操作配额 >= 2 * num_layers（读写各一份，80 层 = 160）。
        # 不传时 runtime 只报 4，于是 select_transfer 判定
        # "direct operation capacity is insufficient: configured=4, required=160"
        # 并静默回退到 staged 暂存路径——这不是报错，只是一条 warning，很容易
        # 误以为在测直连。256 留出余量。
        --max-in-flight-operations 256
        # 直连不可用时直接失败而非回退：本次实验要测的就是直连，静默走 staged
        # 会让整组数字失去意义。
        --direct-transfer-strict
        --kv-load-failure-policy fail
    )
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
    # --nvtx-capture limits ranges to Tutti's own domain; capturing everything
    # buries the storage work in framework noise.
    nsys profile \
        --output "$REPORT" \
        --force-overwrite true \
        --trace cuda,nvtx,osrt \
        --nvtx-capture='*@tutti.*' \
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
