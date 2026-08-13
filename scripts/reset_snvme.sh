#!/bin/bash
#
# reset_snvme.sh -- attempt a clean snvme.ko / snvme_core.ko reload.
#
# Behaviour change vs. earlier revision: this script is now strictly
# fail-fast.  Any rmmod failure aborts the script *before* attempting
# insmod, because re-inserting a half-unloaded module is the exact
# failure mode that produced the "sysfs cannot create duplicate
# filename '/devices/virtual/libsnvm helper'" wedge that requires a
# reboot to recover from.
#
# When rmmod fails this script will:
#   1. Print which processes (if any) are holding /dev/snvm_control
#      or /dev/ssnvme* file descriptors.
#   2. Print the kernel module reference count so the operator can
#      tell whether the leak is in userspace (kill the holder) or
#      kernel-space (reboot is the only safe option).
#   3. Exit non-zero without touching insmod.
#
# Pass --force-cleanup to additionally terminate anything holding
# /dev/snvm* fds before rmmod.  Use sparingly; if a NVMeService
# daemon is running this WILL kill it.  Termination is graceful-first
# (SIGTERM, up to a 10s wait for the daemon's own chrdev/controller
# teardown -- see NVMeService.md's "clean shutdown" sequence) and
# only escalates to SIGKILL if the fd is still held afterwards.
# SIGKILL-ing the daemon mid-teardown (or any -9 that bypasses this
# script) can leave the kernel module's controller/chrdev state
# leaked, which then makes rmmod fail with "Module snvme is in use"
# and requires a host reboot to clear -- do not retry insmod in that
# state (see the insmod failure message below).
#
# NOTE: a holder whose /proc/<pid>/exe shows "(deleted)" is NOT
# necessarily stale -- that is the normal, harmless result of
# rebuilding the daemon binary (cmake --build) while the daemon keeps
# running against its old inode.  Verify liveness (ps -p <pid>, tail
# its log for recent reaper/RPC activity) before assuming a holder is
# safe to kill.
#
# IMPORTANT (kernel-refcount root cause, confirmed by reading
# snvm_fops/snvm_dev_fops in the kernel_modules sources): the Linux
# cdev framework auto-pairs try_module_get()/module_put() on every
# open()/fd-close of /dev/snvm_control and /dev/ssnvme* regardless of
# whether the driver defines its own .release -- so a SIGKILL'd
# daemon does NOT by itself leak snvme's module refcount.  The two
# refs that actually DO outlive the daemon (and that
# SNVM_DEVICE_UNBIND/SNVM_CHRDEV_REMOVE never touch) are:
#   1. the ext4 block device /dev/snvme<N>n<Y> being mount(2)'d
#      (nvme_storage's HostFsBackedNvmeStorage owns this mount -- see
#      mount_if_needed_locked/umount_locked); only umount(2) drops it.
#   2. the RAW admin chardev /dev/snvme<N> (no leading 's') being open
#      by some other process (nvme-cli, another daemon, ...).
# This script now checks + surfaces both BEFORE rmmod (step 2b) --
# check that output first; if it's empty and rmmod still says
# "in use", THAT is the case with no known non-reboot recovery.
#
# Pass --no-insmod to only do unbind+rmmod (rebuild flow: edit code,
# rmmod, build, insmod manually).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SNVME_BUILD_DIR:-$SCRIPT_DIR/../build/cuda-module/module}"

FORCE_CLEANUP=0
DO_INSMOD=1
for arg in "$@"; do
    case "$arg" in
        --force-cleanup) FORCE_CLEANUP=1 ;;
        --no-insmod)     DO_INSMOD=0 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--force-cleanup] [--no-insmod]

  --force-cleanup   SIGKILL any process holding /dev/snvm* before rmmod.
                    Will kill running NVMeService daemons; use with care.
  --no-insmod       Stop after rmmod; do not reload the module.
EOF
            exit 0
            ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "=== snvme reset (force_cleanup=$FORCE_CLEANUP, do_insmod=$DO_INSMOD) ==="

# ----------------------------------------------------------------
# Step 1: Unbind any controllers currently owned by snvme so the
# in-tree nvme driver can take them back over.  unbind.sh is
# idempotent and safe to run when no controllers are bound.
# ----------------------------------------------------------------
echo "[1/4] Unbinding all snvme-owned controllers..."
bash "$SCRIPT_DIR/unbind.sh"

# ----------------------------------------------------------------
# Step 2: If anything is still holding /dev/snvm* fds, surface it.
# ----------------------------------------------------------------
echo "[2/4] Checking for fd holders..."
HOLDERS=$(sudo lsof /dev/snvm_control /dev/ssnvme* 2>/dev/null || true)
if [ -n "$HOLDERS" ]; then
    echo "  Currently held by:"
    echo "$HOLDERS" | sed 's/^/    /'
    echo "  (a '(deleted)' /proc/<pid>/exe here is a normal rebuild artifact,"
    echo "  NOT a sign the process is stale -- verify liveness before killing.)"
    if [ "$FORCE_CLEANUP" = "1" ]; then
        # Skip the lsof header by stripping the first line; column 2 is PID.
        PIDS=$(echo "$HOLDERS" | tail -n +2 | awk '{print $2}' | sort -u)
        echo "  --force-cleanup set: sending SIGTERM first (graceful shutdown --"
        echo "  gives the daemon a chance to run its own chrdev/controller"
        echo "  teardown instead of leaking kernel state)..."
        echo "$PIDS" | xargs -r sudo kill -TERM
        for _ in $(seq 1 10); do
            sleep 1
            STILL=$(sudo lsof /dev/snvm_control /dev/ssnvme* 2>/dev/null \
                | tail -n +2 | awk '{print $2}' | sort -u || true)
            [ -z "$STILL" ] && break
        done
        STILL=$(sudo lsof /dev/snvm_control /dev/ssnvme* 2>/dev/null \
            | tail -n +2 | awk '{print $2}' | sort -u || true)
        if [ -n "$STILL" ]; then
            echo "  Still held after 10s graceful wait -- escalating to SIGKILL."
            echo "  WARNING: this can leak kernel-side controller/chrdev state"
            echo "  (rmmod 'Module snvme is in use', requiring a reboot) if the"
            echo "  holder was killed mid-teardown."
            echo "$STILL" | xargs -r sudo kill -9
            sleep 1
        fi
    else
        echo "  Refusing to rmmod with live fd holders; rerun with --force-cleanup"
        echo "  (SIGTERM first, up to a 10s grace period, then SIGKILL only if"
        echo "  still held), or stop the holders manually first." >&2
        exit 1
    fi
else
    echo "  No fd holders."
fi

# ----------------------------------------------------------------
# Step 2b: the two refcount holders that SNVM_DEVICE_UNBIND /
# SNVM_CHRDEV_REMOVE never touch (see the IMPORTANT note above) --
# check these BEFORE blaming "kernel-side leak, reboot only".
# ----------------------------------------------------------------
echo "[2b/4] Checking for mounted /dev/snvme*n* and open raw admin chardevs..."
MOUNTED=$(mount | grep -E '^/dev/snvme[0-9]+n[0-9]+ ' || true)
RAW_HOLDERS=$(sudo lsof /dev/snvme[0-9]* 2>/dev/null | grep -v ssnvme || true)

if [ -n "$MOUNTED" ]; then
    echo "  Mounted (this is the #1 cause of a refcount that survives"
    echo "  a SIGKILL'd daemon -- umount(2) is the only thing that drops it):"
    echo "$MOUNTED" | sed 's/^/    /'
    if [ "$FORCE_CLEANUP" = "1" ]; then
        echo "  --force-cleanup set: unmounting..."
        echo "$MOUNTED" | awk '{print $3}' | while read -r mp; do
            sudo umount "$mp" || echo "  umount $mp failed (errno $?)"
        done
    else
        echo "  Refusing to rmmod with a mounted snvme block device; rerun"
        echo "  with --force-cleanup to umount it, or umount manually first." >&2
        exit 1
    fi
else
    echo "  No mounted /dev/snvme*n* block devices."
fi

if [ -n "$RAW_HOLDERS" ]; then
    echo "  Raw /dev/snvme<N> admin chardev still open by:"
    echo "$RAW_HOLDERS" | sed 's/^/    /'
    echo "  SNVM_DEVICE_UNBIND/SNVM_CHRDEV_REMOVE never close this fd for"
    echo "  you -- stop/kill the holder above manually, then re-run."
else
    echo "  No open raw /dev/snvme<N> admin chardevs."
fi

# ----------------------------------------------------------------
# Step 3: rmmod, fail-fast on error.
# ----------------------------------------------------------------
echo "[3/4] Removing kernel module..."

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory $BUILD_DIR does not exist" >&2
    exit 1
fi

# Skip rmmod entirely if the module isn't loaded -- saves
# time on first boot and avoids a noisy "module not found".
if ! lsmod | grep -qE '^snvme(\s|$)' && ! lsmod | grep -qE '^snvme_core(\s|$)'; then
    echo "  Module not loaded, skipping rmmod."
else
    cd "$BUILD_DIR"
    if ! make rmmod; then
        echo
        echo "rmmod failed.  Refcount snapshot:"
        lsmod | grep -E '^(snvme|snvme_core)\s' | sed 's/^/    /' || true
        echo
        echo "Common causes (roughly in order of likelihood -- see the"
        echo "IMPORTANT note at the top of this script):"
        echo "  * Step 2b above found a mounted /dev/snvme*n* or an open"
        echo "    raw /dev/snvme<N> -- go fix that first, it's the usual"
        echo "    cause and does NOT require a reboot."
        echo "  * A user-space process still has /dev/ssnvme* open"
        echo "    (lsof check above missed it; race).  Re-run with"
        echo "    --force-cleanup to kill all holders, then retry."
        echo "  * A controller is still bound to snvme.  Check"
        echo "    'lspci -k -d ::0108' for any 'Kernel driver in use:"
        echo "    snvme' lines."
        echo "  * Genuine kernel-side leak inside the module (rare, and"
        echo "    NOT the same as \"daemon got SIGKILL'd\" -- module"
        echo "    refcount is auto-paired by the cdev framework on every"
        echo "    fd close regardless of signal).  In this case rmmod"
        echo "    will keep failing until reboot." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------
# Step 4: insmod (unless --no-insmod).
# ----------------------------------------------------------------
if [ "$DO_INSMOD" = "0" ]; then
    echo "[4/4] --no-insmod set, stopping here.  Module is unloaded."
    echo "snvme reset (rmmod-only) completed!"
    exit 0
fi

echo "[4/4] Re-inserting kernel module..."
cd "$BUILD_DIR"
if ! make insmod; then
    echo
    echo "insmod failed.  See dmesg for the kernel-side reason." >&2
    echo "If dmesg says 'cannot create duplicate filename" >&2
    echo "/devices/virtual/libsnvm helper' or 'failed to create" >&2
    echo "/dev/snvm_control: -17', the module's last unload was" >&2
    echo "incomplete and a reboot is required.  Do NOT keep" >&2
    echo "retrying insmod -- repeated attempts have been observed" >&2
    echo "to crash the host." >&2
    exit 1
fi

echo "Kernel module reloaded successfully."
echo "snvme reset completed!"
