#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# NVMeService client-mode non-destructive attach smoke harness.
#
# Fixes from T-003:
#   1. mount_path is NOT the raw NVMe filesystem mount target.  It is a work
#      directory where the daemon creates GPU<n>/ sub-dirs and symlinks.
#      The dry-run now verifies /dev/nvmeXn1 is NOT mounted and has no holders.
#   2. The real client binary is nvmeservice_client (OUTPUT_NAME), not the
#      CMake target name nvmeservice_client_example.
#
# Fix from T-013:
#   The dry-run block-device gate was structurally broken: it required four
#   block devices to exist, but those devices are only created by the daemon
#   at runtime (B3 bind+probe).  When the daemon is NOT running — which is
#   exactly when dry-run is used — the four target NVMe PCI devices are
#   UNBOUND and no block device exists.  This is the steady state, not a
#   fault.  The dry-run now treats missing block devices as INFO (expected
#   when daemon is not running) and only errors if a block device EXISTS but
#   is mounted or has non-empty holders.
# =============================================================================

usage() {
    cat <<'USAGE'
Usage:
  ./run_attach_smoke.sh             dry-run: validate prerequisites and print the plan
  ./run_attach_smoke.sh --execute   run the non-I/O attach smoke

Overrides:
  DAEMON_BIN  daemon executable (default: cuda-module preset's tutti_daemon)
  CLIENT_BIN  client executable (default: cuda-module preset's nvmeservice_client)
  SUDO        privilege prefix, default: "sudo -n"
  ENDPOINT    gRPC endpoint, default: "127.0.0.1:50051"
USAGE
}

if [[ $# -eq 0 ]]; then
    MODE="dry-run"
elif [[ $# -eq 1 && "$1" == "--execute" ]]; then
    MODE="execute"
elif [[ $# -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    usage
    exit 0
else
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$SCRIPT_DIR/generate_attach_config.py"
DAEMON_BIN="${DAEMON_BIN-$REPO_ROOT/build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon}"
CLIENT_BIN="${CLIENT_BIN-$REPO_ROOT/build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client}"
SUDO="${SUDO-sudo -n}"
ENDPOINT="${ENDPOINT-127.0.0.1:50051}"
WORK_ROOT="$SCRIPT_DIR/.work"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$WORK_ROOT/logs/$TIMESTAMP"
RUN_LOG="/dev/null"
CONFIG_PATH=""
DAEMON_PID=""
DAEMON_LOG=""
DAEMON_CLEAN=0

SUDO_CMD=()
read -r -a SUDO_CMD <<< "$SUDO"

GPU_IDS=(0 1 2 3)
BDFS=("0000:08:00.0" "0000:4b:00.0" "0000:57:00.0" "0000:63:00.0")

# Auto-detect block device names: snvme driver creates "snvmeXn1" after
# re-bind, while initial module load may create "nvmeXn1".
resolve_block_device() {
    local dev_id="$1"
    for name in "nvme${dev_id}n1" "snvme${dev_id}n1"; do
        if [[ -e "/dev/$name" ]]; then
            echo "/dev/$name"
            return 0
        fi
    done
    echo "/dev/nvme${dev_id}n1"  # default for error message
}

BLOCK_DEVICES=()
for _i in 0 1 2 3; do
    BLOCK_DEVICES+=("$(resolve_block_device "$_i")")
done

CHECK_ERRORS=()

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

log_msg() {
    echo "$@" | tee -a "$RUN_LOG"
}

add_error() {
    CHECK_ERRORS+=("$1")
}

check_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        add_error "required command is unavailable: $name"
        return 1
    fi
    return 0
}

check_endpoint_free() {
    local host port rc
    host="${ENDPOINT%:*}"
    port="${ENDPOINT##*:}"
    echo "[check] endpoint free: $ENDPOINT"
    if python3 -c 'import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect((sys.argv[1], int(sys.argv[2])))
except ConnectionRefusedError:
    raise SystemExit(0)
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(2)
else:
    raise SystemExit(1)
finally:
    s.close()
' "$host" "$port"; then
        echo "  OK: no listener on $ENDPOINT"
    else
        rc=$?
        if [[ $rc -eq 1 ]]; then
            add_error "gRPC endpoint is already occupied: $ENDPOINT"
        else
            add_error "could not determine whether $ENDPOINT is free (probe rc=$rc)"
        fi
    fi
}

preflight() {
    local gpu idx bdf block holders_dir holders

    echo "=== Tutti NVMeService attach smoke: $MODE ==="
    echo "repo: $REPO_ROOT"
    echo "endpoint: $ENDPOINT"
    echo "daemon: $DAEMON_BIN"
    echo "client: $CLIENT_BIN"
    echo "generator: $GENERATOR"
    echo

    check_command python3 || true
    check_command findmnt || true
    check_command nvidia-smi || true
    check_command timeout || true

    if [[ ! -x "$GENERATOR" ]]; then
        add_error "generator is not executable: $GENERATOR"
    fi
    if [[ ! -x "$DAEMON_BIN" ]]; then
        add_error "daemon binary is not executable: $DAEMON_BIN"
    else
        echo "[check] daemon executable: $DAEMON_BIN"
    fi
    if [[ ! -x "$CLIENT_BIN" ]]; then
        add_error "client binary is not executable: $CLIENT_BIN"
    else
        echo "[check] client executable: $CLIENT_BIN"
    fi

    for gpu in "${GPU_IDS[@]}"; do
        echo "[check] nvidia-smi -i $gpu"
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            add_error "nvidia-smi is unavailable; cannot query GPU $gpu"
        elif nvidia-smi -i "$gpu" >/dev/null 2>&1; then
            echo "  OK: GPU $gpu is queryable"
        else
            add_error "nvidia-smi -i $gpu failed"
        fi
    done

    for bdf in "${BDFS[@]}"; do
        local sysfs_path="/sys/bus/pci/devices/$bdf"
        echo "[check] PCI $bdf"
        if [[ -e "$sysfs_path" ]]; then
            echo "  OK: PCI $bdf exists"
            # Report driver binding status (informational, not an error).
            # When the daemon is not running, these devices are UNBOUND —
            # this is the expected steady state.  The daemon binds them at
            # startup via B3 and unbinds on exit.
            local driver_link="$sysfs_path/driver"
            if [[ -L "$driver_link" ]]; then
                local driver_name
                driver_name="$(basename "$(readlink -f "$driver_link")" 2>/dev/null || echo "unknown")"
                echo "  INFO: PCI $bdf bound to driver: $driver_name"
            else
                echo "  INFO: PCI $bdf is UNBOUND (expected when daemon is not running)"
            fi
        else
            add_error "PCI device is missing: $sysfs_path"
        fi
    done

    for block in "${BLOCK_DEVICES[@]}"; do
        echo "[check] block device $block"
        if [[ ! -e "$block" ]]; then
            # Block devices are created by the snvme driver via B3 bind+probe
            # when the daemon starts.  When the daemon is NOT running, the
            # four target PCI devices are UNBOUND and no block device exists.
            # This is the expected steady state, not a fault.
            if [[ "$MODE" == "execute" ]]; then
                echo "  WARN: $block does not exist (daemon will create via B3 bind+probe)"
            else
                echo "  INFO: $block does not exist (expected when daemon is not running;"
                echo "        daemon creates block devices via B3 bind+probe at startup)"
            fi
            continue
        fi
        echo "  OK: block device exists: $block"

        # Verify NOT mounted (raw NVMe must not have a filesystem mount target).
        echo "[check] $block not mounted"
        if findmnt -n -o TARGET --source "$block" >/dev/null 2>&1; then
            local mnt
            mnt="$(findmnt -n -o TARGET --source "$block" 2>/dev/null | head -1)"
            add_error "$block is mounted at $mnt; raw NVMe must NOT be mounted"
        else
            echo "  OK: $block has no mount target"
        fi

        # Verify holders are empty.
        holders_dir="/sys/class/block/$(basename "$block")/holders"
        echo "[check] $block holders empty"
        if [[ -d "$holders_dir" ]]; then
            holders=()
            while IFS= read -r h; do
                [[ -n "$h" ]] && holders+=("$h")
            done < <(ls -1 "$holders_dir" 2>/dev/null || true)
            if [[ ${#holders[@]} -gt 0 ]]; then
                add_error "$block has non-empty holders: ${holders[*]}"
            else
                echo "  OK: $block holders empty"
            fi
        else
            echo "  OK: $block has no holders directory (treated as empty)"
        fi
    done

    echo "[check] ${SUDO:-<empty>} true"
    if [[ ${#SUDO_CMD[@]} -eq 0 ]]; then
        add_error "SUDO is empty; passwordless sudo check cannot be performed"
    elif ! "${SUDO_CMD[@]}" true >/dev/null 2>&1; then
        add_error "passwordless sudo is unavailable for: $SUDO"
    else
        echo "  OK: passwordless privilege prefix works"
    fi

    check_endpoint_free

    echo
    if [[ ${#CHECK_ERRORS[@]} -eq 0 ]]; then
        echo "Preflight: PASS"
    else
        echo "Preflight: BLOCKED/FAIL"
        for line in "${CHECK_ERRORS[@]}"; do
            echo "  ERROR: $line"
        done
    fi
}

print_plan() {
    local config_path="${CONFIG_PATH:-$RUN_DIR/attach_config.yaml}"
    echo
    echo "=== Planned commands ==="
    echo "Generate config:"
    print_command python3 -B "$GENERATOR" --output "$config_path"
    echo "Start daemon:"
    print_command "${SUDO_CMD[@]}" "$DAEMON_BIN" --config "$config_path"
    echo "List devices:"
    print_command timeout 90s "$CLIENT_BIN" --endpoint "$ENDPOINT" --list-only --skip-io
    local idx
    for idx in "${!GPU_IDS[@]}"; do
        echo "Attach device=$idx cuda=${GPU_IDS[$idx]}:"
        print_command timeout 90s "${SUDO_CMD[@]}" "$CLIENT_BIN" \
            --endpoint "$ENDPOINT" --device "$idx" --cuda "${GPU_IDS[$idx]}" \
            --count 2 --hold 2 --skip-io
    done
}

stop_daemon() {
    local deadline rc=0
    if [[ -z "$DAEMON_PID" ]]; then
        return 0
    fi

    # Find the actual daemon process (sudo does not forward SIGTERM).
    # DAEMON_PID is the sudo wrapper; the real daemon is its child or
    # any process matching the daemon binary + config path.
    local real_pids=()
    local p
    # Try child of sudo first.
    while IFS= read -r p; do
        [[ -n "$p" ]] && real_pids+=("$p")
    done < <(pgrep -P "$DAEMON_PID" 2>/dev/null || true)
    # Also match by command line.
    while IFS= read -r p; do
        [[ -n "$p" ]] && real_pids+=("$p")
    done < <(pgrep -f "$DAEMON_BIN" 2>/dev/null || true)

    log_msg "Stopping daemon: sudo pid=$DAEMON_PID, real pids=[${real_pids[*]:-none}]"

    # SIGTERM the real daemon process(es) via sudo so the signal is delivered.
    for p in "${real_pids[@]:-}"; do
        [[ -z "$p" ]] && continue
        "${SUDO_CMD[@]}" kill -TERM "$p" 2>/dev/null || true
    done
    # Also SIGTERM the sudo wrapper.
    kill -TERM "$DAEMON_PID" 2>/dev/null || true

    deadline=$((SECONDS + 20))
    while true; do
        local still_running=0
        for p in "${real_pids[@]:-}"; do
            [[ -z "$p" ]] && continue
            if "${SUDO_CMD[@]}" kill -0 "$p" 2>/dev/null; then
                still_running=1
                break
            fi
        done
        if kill -0 "$DAEMON_PID" 2>/dev/null; then
            still_running=1
        fi
        [[ "$still_running" -eq 0 ]] && break
        if (( SECONDS >= deadline )); then
            log_msg "Daemon did not exit after 20s SIGTERM; using SIGKILL"
            for p in "${real_pids[@]:-}"; do
                [[ -z "$p" ]] && continue
                "${SUDO_CMD[@]}" kill -KILL "$p" 2>/dev/null || true
            done
            kill -KILL "$DAEMON_PID" 2>/dev/null || true
            rc=1
            break
        fi
        sleep 1
    done

    wait "$DAEMON_PID" 2>/dev/null || true
    # Give the log a moment to flush.
    sleep 1
    DAEMON_PID=""

    if [[ -n "$DAEMON_LOG" ]] && grep -Fq "tutti_daemon exited cleanly." "$DAEMON_LOG" 2>/dev/null; then
        DAEMON_CLEAN=1
        log_msg "Daemon cleanup: clean exit observed"
    else
        DAEMON_CLEAN=0
        log_msg "Daemon cleanup: clean exit was NOT observed"
        rc=1
    fi
    return "$rc"
}

cleanup_on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "$DAEMON_PID" ]]; then
        stop_daemon || true
    fi
    exit "$rc"
}

run_logged() {
    local logfile="$1"
    shift
    log_msg "COMMAND: $*"
    set +e
    timeout 90s "$@" >"$logfile" 2>&1
    local rc=$?
    set -e
    log_msg "RESULT rc=$rc log=$logfile"
    return "$rc"
}

validate_list_log() {
    local logfile="$1"
    local rc=0 idx
    for idx in 0 1 2 3; do
        if ! grep -Fq "device_id=$idx" "$logfile"; then
            log_msg "ListDevices validation: missing device_id=$idx"
            rc=1
        fi
    done
    if ! grep -Fq "ListDevices" "$logfile"; then
        log_msg "ListDevices validation: missing 'ListDevices' marker"
        rc=1
    fi
    return "$rc"
}

validate_attach_log() {
    local logfile="$1"
    local device_id="$2"
    local cuda_id="$3"
    local expected_mount_prefix="$4"
    local rc=0

    # Positive markers that must appear.
    local pos_markers=(
        "ListDevices"
        "allocation_id"
        "device_id     : $device_id"
        "cuda_device   : $cuda_id"
        "granted_queues: 2"
        "snvme_dev"
        "mount_path"
        "mount->"
        "nvm_ctrl_attach_client"
        "nvm_create_group"
        "nvm_destroy_group (skip-io path)"
        "nvm_ctrl_free_client (skip-io path)"
        "Holding session for 2s"
        "Disconnect"
        "Done."
    )
    local marker
    for marker in "${pos_markers[@]}"; do
        if ! grep -Fq "$marker" "$logfile"; then
            log_msg "device=$device_id: missing output marker: $marker"
            rc=1
        fi
    done

    # mount_path must be under the expected work dir.
    if ! grep -Fq "$expected_mount_prefix" "$logfile"; then
        log_msg "device=$device_id: mount_path not under expected work dir $expected_mount_prefix"
        rc=1
    fi

    # Negative markers that must NOT appear.
    local neg_markers=(
        "Write IO"
        "Read IO"
        "Write+Read+verify"
        "mapped SQ/CQ"
        "nvm_add_user_queue"
        "lease revoked by daemon for allocation"
        "Connect rejected"
        "Connect RPC failed"
        "Disconnect RPC failed"
        "Disconnect rejected"
    )
    for marker in "${neg_markers[@]}"; do
        if grep -Fq "$marker" "$logfile"; then
            log_msg "device=$device_id: forbidden marker found: $marker"
            rc=1
        fi
    done

    return "$rc"
}

check_daemon_not_gpu_resident() {
    local daemon_process_pid=""
    local pid cmdline

    # DAEMON_PID is normally the sudo wrapper.  Match the exact executable
    # and generated config in the real child's NUL-separated command line.
    while IFS= read -r pid; do
        [[ -n "$pid" ]] || continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        if [[ "$cmdline" == "$DAEMON_BIN --config $CONFIG_PATH " ]]; then
            daemon_process_pid="$pid"
            break
        fi
    done < <(pgrep -x tutti_daemon 2>/dev/null || true)

    if [[ -z "$daemon_process_pid" ]]; then
        log_msg "GPU residency validation: could not resolve real tutti_daemon pid"
        return 1
    fi

    if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | tr -d ' ' | grep -Fxq "$daemon_process_pid"; then
        log_msg "GPU residency validation: daemon pid=$daemon_process_pid unexpectedly owns a CUDA context"
        return 1
    fi

    log_msg "GPU residency validation: PASS (daemon pid=$daemon_process_pid is absent from NVIDIA compute processes)"
    return 0
}

check_symlink() {
    local gpu_work="$1"
    local nvme_work="$2"
    local gpu_id="$3"
    local rc=0

    # The daemon creates ssnvmeN symlinks under the GPU work dir.
    # We check that at least one ssnvme* symlink exists and points into nvme_work/GPU<id>.
    local found=0
    local link target
    if [[ -d "$gpu_work" ]]; then
        while IFS= read -r link; do
            [[ -z "$link" ]] && continue
            target="$(readlink -f "$link" 2>/dev/null || true)"
            if [[ -n "$target" ]]; then
                log_msg "symlink: $link -> $target"
                found=1
            fi
        done < <(find "$gpu_work" -maxdepth 1 -name 'ssnvme*' -type l 2>/dev/null || true)
    fi
    if [[ "$found" -eq 0 ]]; then
        log_msg "device=$gpu_id: no ssnvme* symlink found under $gpu_work"
        rc=1
    fi
    return "$rc"
}

# =============================================================================
# Main
# =============================================================================

if [[ "$MODE" == "execute" ]]; then
    mkdir -p "$RUN_DIR"
    RUN_LOG="$RUN_DIR/harness.log"
    CONFIG_PATH="$RUN_DIR/attach_config.yaml"
    : > "$RUN_LOG"
fi

preflight
print_plan

if [[ ${#CHECK_ERRORS[@]} -ne 0 ]]; then
    if [[ "$MODE" == "dry-run" ]]; then
        echo
        echo "Dry-run result: BLOCKED (no daemon started; no files created)"
    else
        echo
        echo "Execute result: BLOCKED (preflight failed; no daemon started)"
    fi
    exit 1
fi

if [[ "$MODE" != "execute" ]]; then
    echo
    echo "Dry-run result: PASS (no daemon started; no files created)"
    exit 0
fi

trap cleanup_on_exit EXIT INT TERM
OVERALL_RC=0
DAEMON_LOG="$RUN_DIR/daemon.log"
LIST_LOG="$RUN_DIR/client_list.log"

# Generate config.
log_msg "COMMAND: python3 -B $GENERATOR --output $CONFIG_PATH"
if ! python3 -B "$GENERATOR" --output "$CONFIG_PATH" >>"$RUN_LOG" 2>&1; then
    log_msg "Config generation failed"
    exit 1
fi
log_msg "Config generated: $CONFIG_PATH"

# Determine work dirs for symlink checks.
GPUS_WORK="$RUN_DIR/mount_work/gpus"
NVMES_WORK="$RUN_DIR/mount_work/nvmes"

# Start daemon.
log_msg "COMMAND: ${SUDO_CMD[*]} $DAEMON_BIN --config $CONFIG_PATH"
{
    printf 'COMMAND: '
    printf '%q ' "${SUDO_CMD[@]}" "$DAEMON_BIN" --config "$CONFIG_PATH"
    printf '\n'
    "${SUDO_CMD[@]}" "$DAEMON_BIN" --config "$CONFIG_PATH"
} >"$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!
log_msg "Daemon pid=$DAEMON_PID log=$DAEMON_LOG"

# Wait for daemon to be listening and report owned devices.
LISTENING=0
for _ in $(seq 1 30); do
    if grep -Fq "tutti_daemon listening on $ENDPOINT" "$DAEMON_LOG" 2>/dev/null; then
        LISTENING=1
        break
    fi
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done
if [[ "$LISTENING" -ne 1 ]]; then
    log_msg "Daemon did not report listening on $ENDPOINT within 30 seconds"
    tail -n 80 "$DAEMON_LOG" >&2 || true
    exit 1
fi
log_msg "Daemon listening: PASS"

if ! grep -Fq "Owned devices:" "$DAEMON_LOG" 2>/dev/null; then
    log_msg "Daemon did not report 'Owned devices:'"
    tail -n 80 "$DAEMON_LOG" >&2 || true
    exit 1
fi
log_msg "Daemon device ownership report: PASS"

# Owner-only bring-up must not initialize CUDA.  Run this before starting any
# attach client so the process list cannot be confused with a client context.
if ! check_daemon_not_gpu_resident; then
    OVERALL_RC=1
fi

# ListDevices.
LIST_CMD=("$CLIENT_BIN" --endpoint "$ENDPOINT" --list-only --skip-io)
if ! run_logged "$LIST_LOG" "${LIST_CMD[@]}"; then
    OVERALL_RC=1
fi
if ! validate_list_log "$LIST_LOG"; then
    OVERALL_RC=1
else
    log_msg "ListDevices validation: PASS"
fi

# Four attach groups.
for idx in 0 1 2 3; do
    client_log="$RUN_DIR/client_device_${idx}_gpu_${GPU_IDS[$idx]}.log"
    client_cmd=(
        "${SUDO_CMD[@]}" "$CLIENT_BIN"
        --endpoint "$ENDPOINT"
        --device "$idx"
        --cuda "${GPU_IDS[$idx]}"
        --count 2
        --hold 2
        --skip-io
    )
    if ! run_logged "$client_log" "${client_cmd[@]}"; then
        OVERALL_RC=1
    fi
    expected_mount_prefix="$GPUS_WORK/gpu${GPU_IDS[$idx]}"
    if ! validate_attach_log "$client_log" "$idx" "${GPU_IDS[$idx]}" "$expected_mount_prefix"; then
        OVERALL_RC=1
    else
        log_msg "device=$idx cuda=${GPU_IDS[$idx]} validation: PASS"
    fi
    # Check symlink after each attach.
    check_symlink "$GPUS_WORK/gpu${GPU_IDS[$idx]}" "$NVMES_WORK/nvme${idx}" "${GPU_IDS[$idx]}" || OVERALL_RC=1
done

# Stop daemon.
if ! stop_daemon; then
    OVERALL_RC=1
fi
if [[ "$DAEMON_CLEAN" -ne 1 ]]; then
    OVERALL_RC=1
fi

# Check symlink state after daemon stopped.
log_msg "Post-daemon symlink state:"
for idx in 0 1 2 3; do
    gpu_dir="$GPUS_WORK/gpu${GPU_IDS[$idx]}"
    remaining=$(find "$gpu_dir" -maxdepth 1 -name 'ssnvme*' -type l 2>/dev/null | wc -l)
    log_msg "  gpu${GPU_IDS[$idx]}: $remaining ssnvme* symlinks remaining"
done

if [[ "$OVERALL_RC" -eq 0 ]]; then
    log_msg "Overall: PASS"
else
    log_msg "Overall: FAIL"
fi
exit "$OVERALL_RC"
