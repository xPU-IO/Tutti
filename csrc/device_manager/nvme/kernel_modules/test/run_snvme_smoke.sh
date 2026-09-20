#!/usr/bin/env bash
# run_snvme_smoke.sh -- one-shot wrapper around snvme_smoke / snvme_smoke_gpu.
#
# What it does, in order:
#   1. (Re)builds the test binaries via the local Makefile.
#   2. Verifies that /dev/snvm_control exists; if not, refuses to run.
#   3. Picks a PCI BDF: from $PCI_ADDR if set, else from the first NVMe
#      device the kernel knows about.
#   4. Runs the chosen smoke test:
#        --gpu       : ./snvme_smoke_gpu (additionally exercises GPU paths)
#        otherwise   : ./snvme_smoke      (UAPI-only, libc, no CUDA needed)
#      Add --bind to also exercise the destructive bring-up.
#
# Usage:
#   ./run_snvme_smoke.sh                          # host UAPI smoke (safe)
#   ./run_snvme_smoke.sh --gpu                    # + GPU map paths (safe)
#   ./run_snvme_smoke.sh --gpu --bind             # GPU + full bring-up (destructive)
#   ./run_snvme_smoke.sh --gpu --gpu-id 1         # use cuda device 1
#   PCI_ADDR=0000:50:00.0 ./run_snvme_smoke.sh
#   ./run_snvme_smoke.sh 0000:50:00.0 --gpu --bind

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- Parse args ---
BIND_FLAG=""
GPU_MODE=""
GPU_ID=""
ARG_BDF=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bind)    BIND_FLAG="--bind"; shift ;;
        --gpu)     GPU_MODE="1"; shift ;;
        --gpu-id)  GPU_ID="$2"; GPU_MODE="1"; shift 2 ;;
        --help|-h)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)         ARG_BDF="$1"; shift ;;
    esac
done

# --- Build ---
echo "[+] building snvme smoke tests ..."
if ! make -s; then
    echo "[!] build failed." >&2
    echo "    Toolchain snapshot:" >&2
    command -v nvcc >/dev/null 2>&1 \
        && nvcc --version 2>&1 | sed -e 's/^/      /' >&2 \
        || echo "      nvcc: NOT FOUND on PATH" >&2
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "    Host GPU compute capability:" >&2
        nvidia-smi --query-gpu=name,compute_cap --format=csv \
            2>&1 | sed -e 's/^/      /' >&2
    fi
    echo "    Hint: if the failure is 'Unsupported gpu architecture',"  >&2
    echo "    the nvcc you have does not support this Makefile's"        >&2
    echo "    auto-detected CUDA_ARCH.  Override it explicitly, e.g.:"   >&2
    echo "      make CUDA_ARCH=sm_80    # A100 / lowest CUDA-13 default" >&2
    echo "      make CUDA_ARCH=sm_90    # H100 / H200 / H20"             >&2
    exit 1
fi

# --- Pre-flight ---
if [[ ! -e /dev/snvm_control ]]; then
    echo "[!] /dev/snvm_control not found." >&2
    echo "    Load the kernel module first, e.g.:" >&2
    echo "      sudo insmod /path/to/snvme-core.ko" >&2
    echo "      sudo insmod /path/to/snvme.ko" >&2
    exit 1
fi

# --- Choose binary ---
if [[ -n "$GPU_MODE" ]]; then
    BIN="./snvme_smoke_gpu"
    if [[ ! -x "$BIN" ]]; then
        echo "[!] $BIN not built. Make sure 'nvcc' is on PATH and re-run." >&2
        exit 1
    fi
else
    BIN="./snvme_smoke"
fi

# --- Pick BDF ---
BDF="${ARG_BDF:-${PCI_ADDR:-}}"
if [[ -z "$BDF" ]]; then
    BDF="$(lspci -D -d ::0108 2>/dev/null | awk 'NR==1 {print $1}')"
fi
if [[ -z "$BDF" ]]; then
    echo "[!] No NVMe BDF found. Set PCI_ADDR=DDDD:BB:DD.F or pass it as arg." >&2
    exit 1
fi
echo "[+] using PCI BDF: $BDF"
if [[ -n "$BIND_FLAG" ]]; then
    echo "[!] --bind requested: this will detach the in-tree nvme driver"
    echo "    from $BDF for the duration of the test."
fi

# --- Run ---
EXTRA=()
[[ -n "$BIND_FLAG" ]] && EXTRA+=("$BIND_FLAG")
[[ -n "$GPU_ID"    ]] && EXTRA+=("--gpu" "$GPU_ID")

# Cannot `exec` here: we need the post-mortem hook below to run on
# GPU failures so that "system not yet initialized" gets diagnosed
# instead of dropping the user back at the prompt with no context.
set +e
sudo "$BIN" "${EXTRA[@]}" "$BDF"
rc=$?
set -e

# CUDA-init post-mortem.  Triggered only when the GPU binary
# exited non-zero AND we have nvidia tooling available -- 99% of
# the failures we see here are environment problems on the host,
# not bugs in snvme_smoke_gpu itself.
if [[ "$rc" -ne 0 && -n "$GPU_MODE" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo                                                                    >&2
    echo "[!] $BIN exited with status $rc."                                 >&2
    echo "    CUDA-init post-mortem (most common failure modes):"           >&2

    # 1. NVSwitch hosts: fabricmanager must be active.
    if ls /dev/nvidia-nvswitch* >/dev/null 2>&1; then
        if systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
            echo "      [ ok ] nvidia-fabricmanager is active"              >&2
        else
            echo "      [FAIL] NVSwitch host but nvidia-fabricmanager"      >&2
            echo "             is NOT active.  Without it the CUDA"         >&2
            echo "             runtime returns 802 / 'system not yet"       >&2
            echo "             initialized'.  Fix:"                         >&2
            echo "               sudo systemctl enable --now \\"            >&2
            echo "                   nvidia-fabricmanager"                  >&2
        fi
    fi

    # 2. Fabric registration state per GPU.  Authoritative check
    #    on NVLink-fabric hosts (HGX H20 / H100 8-way): every GPU's
    #    "Fabric" block in `nvidia-smi -q` must report
    #         State  : Completed
    #         Status : Success
    #    If Fabric is fully Completed, cuInit() should succeed
    #    regardless of persistence-mode setting (verified 2026-05-19
    #    on HGX H20 + driver 580.65.06: PM=Disabled across all 8
    #    GPUs, Fabric State all Completed, CUDA programs run fine
    #    after reboot).
    #
    #    "GPU Fabric GUID" can legitimately stay "N/A" even on a
    #    healthy fabric -- do NOT use GUID presence as the test
    #    (an earlier revision of this script did and produced
    #    false positives).
    fab_total=0
    fab_done=0
    if ls /dev/nvidia-nvswitch* >/dev/null 2>&1; then
        fab_total="$(nvidia-smi -q 2>/dev/null \
                     | awk '/^    Fabric$/,/^$/' \
                     | grep -c '^        State' || true)"
        fab_done="$(nvidia-smi -q 2>/dev/null \
                    | awk '/^    Fabric$/,/^$/' \
                    | grep -E '^        State' | grep -c 'Completed' || true)"
        if [[ "${fab_total:-0}" -gt 0 ]]; then
            if [[ "${fab_done:-0}" -lt "${fab_total}" ]]; then
                echo "      [FAIL] only ${fab_done}/${fab_total} GPU(s) report"        >&2
                echo "             Fabric State = Completed.  fabricmanager has"      >&2
                echo "             configured NVSwitch routing but at least one"       >&2
                echo "             GPU has not registered yet."                        >&2
                echo "             Inspect with:"                                       >&2
                echo "               nvidia-smi -q | grep -A2 Fabric"                  >&2
                echo "               cat /sys/module/nvidia/refcnt"                    >&2
                echo "             Recovery: see PORTING.md section 8.1 (or, in"       >&2
                echo "             extreme cases, reinstall the NVIDIA driver after"   >&2
                echo "             killing every CUDA process and rmmod-ing nvidia*)." >&2
            else
                echo "      [ ok ] Fabric State = Completed on all ${fab_total} GPU(s)" >&2
            fi
        fi
    fi

    # 2b. Persistence mode -- only flagged as a *secondary* hint when
    #     Fabric registration is incomplete.  PM=Disabled is NOT
    #     itself a failure: a freshly rebooted HGX H20 has every GPU
    #     PM=Disabled and CUDA still works, because fabricmanager
    #     completes the Inband handshake without a resident driver
    #     context as long as nothing has poisoned the kernel module
    #     state.  PM=1 only matters as a *recovery* knob when the
    #     kernel module is in a half-initialised state (e.g. after a
    #     killed CUDA process leaked a ref into nvidia-uvm) -- in
    #     that case re-enabling PM keeps the driver context resident
    #     long enough for fabricmanager's retry to succeed.
    #
    #     So: print the PM hint *only* if Fabric is incomplete,
    #     because that's the only scenario where toggling PM is the
    #     right next step.  Everywhere else, PM=Disabled is normal
    #     and the previous "[FAIL] PM=Disabled" message was a false
    #     alarm (verified 2026-05-19 against the rebooted HGX H20).
    if [[ "${fab_total:-0}" -gt 0 && "${fab_done:-0}" -lt "${fab_total}" ]]; then
        pm_off="$(nvidia-smi --query-gpu=persistence_mode \
                  --format=csv,noheader 2>/dev/null \
                  | grep -ci '^Disabled' || true)"
        if [[ "${pm_off:-0}" -gt 0 ]]; then
            echo "      [hint] $pm_off GPU(s) currently have"                          >&2
            echo "             Persistence-Mode = Disabled.  Combined with"            >&2
            echo "             the incomplete Fabric state above, this often"          >&2
            echo "             unsticks NVLink registration:"                           >&2
            echo "               sudo nvidia-smi -pm 1"                                 >&2
            echo "               sudo systemctl restart nvidia-fabricmanager"           >&2
            echo "             To persist across reboot, install"                       >&2
            echo "             nvidia-persistenced (see PORTING.md section 8.1)."       >&2
        fi
    fi

    # 3. nvidia.ko refcount sanity.  Normal: < ~10.  Triple-digit
    #    refcount almost always means a monitoring loop (Tencent
    #    titan-agent's deviceQuery / getGPUInfo is the most common
    #    offender on internal hosts) is leaking one ref per failed
    #    cuInit and dragging the driver into a poisoned half-state.
    if [[ -r /sys/module/nvidia/refcnt ]]; then
        rc_nv="$(cat /sys/module/nvidia/refcnt 2>/dev/null)"
        if [[ -n "$rc_nv" && "$rc_nv" -gt 30 ]]; then
            echo "      [warn] /sys/module/nvidia/refcnt = $rc_nv (>30)"        >&2
            echo "             likely cuInit-loop leak.  Suspect monitoring:"  >&2
            echo "               pgrep -fa deviceQuery"                        >&2
            echo "               pgrep -fa getGPUInfo"                         >&2
        fi
    fi

    # 4. libcuda sanity: just record the version, don't compare via
    #    `strings`.  The integers exposed via `strings libcuda.so.1`
    #    are an internal compatibility table (for older client
    #    drivers), NOT the library's own version (a 580.65.06
    #    libcuda legitimately shows '575.57.07' as its highest
    #    match).  An earlier revision of this script wrongly used
    #    that comparison and was corrected on 2026-05-18.
    krn="$(awk '/NVRM version:/ {print $8}' /proc/driver/nvidia/version 2>/dev/null)"
    libcuda="$(readlink -f /lib64/libcuda.so.1 2>/dev/null \
               || readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null \
               || true)"
    if [[ -n "$krn" && -n "$libcuda" && -r "$libcuda" ]]; then
        echo "      [info] kernel NVRM = $krn,  $libcuda"                       >&2
    fi
fi
exit "$rc"
