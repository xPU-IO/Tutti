#!/usr/bin/env bash
# 准备本机 Tutti 运行环境：加载内核模块 -> 启动 tutti_daemon -> 校验设备与挂载。
#
# 依赖顺序不可交换（/dev/snvmeN* 由 daemon bring-up 后才出现，先 mount 会找不到设备）：
#   1. phoenixfs.ko            （可选，Phoenix GDS 路径需要）
#   2. snvme-core.ko + snvme.ko
#   3. tutti_daemon            （建块设备节点 + auto_mount + gRPC :50051）
#
# 子命令：
#   up        （默认）依次确保上述三层就绪，幂等
#   status    只读检查，不做任何改动，退出码非 0 表示环境不完整
#   down      停止 daemon（SIGTERM，触发其 umount 流程）；不卸载模块
#   restart   down 后 up
#
# 选项：
#   --dry-run     打印将要执行的命令，不实际执行
#   --no-phoenix  跳过 phoenixfs（RDMA 场景需要卸载它，见 doc/）
#
# 环境变量覆盖：
#   TUTTI_ROOT / TUTTI_BUILD / TUTTI_DAEMON_CONFIG / TUTTI_LOG_DIR / TUTTI_DAEMON_PORT
#   PHOENIX_BUILD
#
# 本脚本永不编译、永不签名模块。缺产物时给出最小构建命令后退出。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TUTTI_ROOT="${TUTTI_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
readonly TUTTI_BUILD="${TUTTI_BUILD:-${TUTTI_ROOT}/build}"
readonly DAEMON_CONFIG="${TUTTI_DAEMON_CONFIG:-${TUTTI_ROOT}/config/local/tutti_daemon.yaml}"
readonly PHOENIX_BUILD="${PHOENIX_BUILD:-/data/home/ryeqiu/phoenix/build}"
readonly PHOENIX_KO="${PHOENIX_BUILD}/module/phoenixfs.ko"
readonly LOG_DIR="${TUTTI_LOG_DIR:-/data/home/ryeqiu/log}"
readonly DAEMON_LOG="${LOG_DIR}/tutti_daemon.log"
readonly DAEMON_PID_FILE="${LOG_DIR}/tutti_daemon.pid"
readonly DAEMON_PORT="${TUTTI_DAEMON_PORT:-50051}"
readonly SNVME_CORE_KO="${TUTTI_BUILD}/module/snvme-core.ko"
readonly SNVME_KO="${TUTTI_BUILD}/module/snvme.ko"
readonly DAEMON_BIN="${TUTTI_BUILD}/bin/tutti_daemon"

DRY_RUN=0
WANT_PHOENIX=1
SUDO_READY=0

note() { printf '[tutti] %s\n' "$*"; }
warn() { printf '[tutti] WARN: %s\n' "$*" >&2; }
die()  { printf '[tutti] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

module_loaded() { [[ -d "/sys/module/$1" ]]; }

ensure_sudo() {
    (( SUDO_READY || DRY_RUN )) && return 0
    if (( EUID == 0 )); then SUDO_READY=1; return 0; fi
    command -v sudo >/dev/null 2>&1 || die '需要 root 或 sudo 来加载模块 / 启动 daemon。'
    sudo -v || die '需要 sudo 权限。'
    SUDO_READY=1
}

print_tutti_build_hint() {
    cat >&2 <<EOF
本脚本不编译。缺失的产物请先手动构建：
  cd ${TUTTI_ROOT}
  cmake --preset default
  cmake --build --preset default --parallel 16
EOF
}

print_signature_hint() {
    cat >&2 <<EOF
$1 加载失败。本脚本不编译也不签名模块。
若 dmesg 出现 "Required key not available" / module signature / key rejection，
请按站点签名流程签名以下已构建模块后重跑：
EOF
    shift
    printf '  %s\n' "$@" >&2
}

daemon_pid() { pgrep -n -x tutti_daemon 2>/dev/null || true; }

daemon_port_listening() {
    command -v ss >/dev/null 2>&1 || return 0   # 无 ss 时不作为判据
    ss -ltnH 2>/dev/null | awk -v p="$DAEMON_PORT" '$4 ~ (":" p "$"){f=1} END{exit !f}'
}

# ---- 只读检查 ---------------------------------------------------------------

# 返回不健康项个数；打印每项状态。--quiet 只返回计数。
check_all() {
    local quiet="${1:-}"
    local bad=0
    local mounted expected

    _row() {  # $1=名称 $2=ok/skip/bad $3=详情
        [[ "$quiet" == "--quiet" ]] && return 0
        local mark
        case "$2" in
            ok)   mark='  OK  ' ;;
            skip) mark=' SKIP ' ;;
            *)    mark=' FAIL ' ;;
        esac
        printf '[%s] %-22s %s\n' "$mark" "$1" "$3"
    }

    if module_loaded phoenixfs; then
        _row phoenixfs ok '已加载'
    elif (( WANT_PHOENIX )); then
        _row phoenixfs skip '未加载（仅 Phoenix GDS 路径需要）'
    else
        _row phoenixfs skip '按 --no-phoenix 跳过'
    fi

    if module_loaded snvme_core && module_loaded snvme; then
        _row snvme ok 'snvme-core + snvme 已加载'
    else
        _row snvme bad "snvme_core=$(module_loaded snvme_core && echo Y || echo N) snvme=$(module_loaded snvme && echo Y || echo N)"
        bad=$((bad + 1))
    fi

    if [[ -e /dev/snvm_control ]]; then
        _row /dev/snvm_control ok '存在'
    else
        _row /dev/snvm_control bad '缺失（模块未正确初始化）'
        bad=$((bad + 1))
    fi

    local pid
    pid="$(daemon_pid)"
    if [[ -n "$pid" ]]; then
        _row tutti_daemon ok "运行中 PID=${pid}"
    else
        _row tutti_daemon bad '未运行'
        bad=$((bad + 1))
    fi

    if [[ -n "$pid" ]] && daemon_port_listening; then
        _row "gRPC :${DAEMON_PORT}" ok '监听中'
    elif [[ -n "$pid" ]]; then
        _row "gRPC :${DAEMON_PORT}" bad '进程在但端口未监听'
        bad=$((bad + 1))
    else
        _row "gRPC :${DAEMON_PORT}" bad '未监听'
        bad=$((bad + 1))
    fi

    # 块设备节点由 daemon bring-up 创建，是"daemon 真的干完活了"的证据
    local nodes
    nodes="$(ls /dev/snvme*n1 2>/dev/null | wc -l)"
    if (( nodes > 0 )); then
        _row '块设备节点' ok "${nodes} 个 /dev/snvme*n1"
    else
        _row '块设备节点' bad '无 /dev/snvme*n1（daemon 未完成 bring-up）'
        bad=$((bad + 1))
    fi

    # auto_mount: true 的挂载点必须真的挂上，否则连接器写 KV 会落到根盘
    expected="$(grep -c 'auto_mount: true' "$DAEMON_CONFIG" 2>/dev/null || echo 0)"
    mounted="$(mount | grep -c '^/dev/snvme[0-9]*n1 on ' || true)"
    if (( expected > 0 && mounted >= expected )); then
        _row 'auto_mount 挂载' ok "${mounted}/${expected} 已挂载"
    elif (( expected > 0 )); then
        _row 'auto_mount 挂载' bad "${mounted}/${expected} 已挂载（配置期望 ${expected}）"
        bad=$((bad + 1))
    else
        _row 'auto_mount 挂载' skip '配置未启用 auto_mount'
    fi

    return "$bad"
}

# ---- 变更动作 ---------------------------------------------------------------

ensure_phoenix() {
    (( WANT_PHOENIX )) || { note '按 --no-phoenix 跳过 phoenixfs。'; return 0; }
    module_loaded phoenixfs && { note 'phoenixfs 已加载。'; return 0; }
    if [[ ! -f "$PHOENIX_KO" ]]; then
        warn "缺少 ${PHOENIX_KO}，跳过 phoenixfs（仅影响 Phoenix GDS 路径）。"
        warn "需要它请先手动构建： cd ${PHOENIX_BUILD} && make modules"
        return 0
    fi
    ensure_sudo
    if (( DRY_RUN )); then note "dry-run: (cd ${PHOENIX_BUILD} && make insmod)"; return 0; fi
    note '加载 phoenixfs。'
    if (( EUID == 0 )); then
        insmod "$PHOENIX_KO" || { print_signature_hint 'Phoenix 模块' "$PHOENIX_KO"; die 'phoenixfs 加载失败。'; }
    else
        ( cd "$PHOENIX_BUILD" && make insmod ) \
            || { print_signature_hint 'Phoenix 模块' "$PHOENIX_KO"; die 'phoenixfs 加载失败。'; }
    fi
    module_loaded phoenixfs || die 'insmod 返回成功但 /sys/module/phoenixfs 不存在。'
}

ensure_snvme() {
    local core=0 pci=0
    module_loaded snvme_core && core=1
    module_loaded snvme && pci=1
    (( core && pci )) && { note 'snvme-core 和 snvme 已加载。'; return 0; }
    (( core || pci )) && die '检测到半加载的 SNVMe 模块状态；请先手动 rmmod 清理后重跑。'

    [[ -f "$SNVME_CORE_KO" && -f "$SNVME_KO" ]] || {
        print_tutti_build_hint
        die "缺少内核模块：${SNVME_CORE_KO} / ${SNVME_KO}"
    }
    ensure_sudo
    if (( DRY_RUN )); then
        note "dry-run: (cd ${TUTTI_ROOT} && cmake --build --preset default --target insmod)"
        return 0
    fi
    note '加载 snvme-core 和 snvme。'
    # insmod target（csrc/cmake/SNVMeModule.cmake）只转发 `make insmod`，不构建 .ko。
    ( cd "$TUTTI_ROOT" && cmake --build --preset default --target insmod ) \
        || { print_signature_hint 'SNVMe 模块' "$SNVME_CORE_KO" "$SNVME_KO"; die 'SNVMe 模块加载失败。'; }
    module_loaded snvme_core || die 'snvme-core 未加载。'
    module_loaded snvme || die 'snvme 未加载。'
    [[ -e /dev/snvm_control ]] || die '模块已加载但 /dev/snvm_control 不存在。'
}

start_daemon() {
    local pid launcher attempt
    if pid="$(daemon_pid)"; [[ -n "$pid" ]]; then
        mkdir -p "$LOG_DIR"; printf '%s\n' "$pid" > "$DAEMON_PID_FILE"
        note "tutti_daemon 已运行，PID=${pid}。"
        return 0
    fi
    [[ -x "$DAEMON_BIN" ]] || { print_tutti_build_hint; die "缺少 ${DAEMON_BIN}"; }
    [[ -f "$DAEMON_CONFIG" ]] || die "缺少 daemon 配置：${DAEMON_CONFIG}"
    mkdir -p "$LOG_DIR"
    ensure_sudo
    if (( DRY_RUN )); then
        note "dry-run: nohup sudo env TUTTI_VERBOSE=1 ${DAEMON_BIN} --config ${DAEMON_CONFIG} >> ${DAEMON_LOG} 2>&1 &"
        return 0
    fi
    printf '\n===== %s: starting tutti_daemon =====\n' "$(date --iso-8601=seconds)" >> "$DAEMON_LOG"
    note "启动 tutti_daemon，日志：${DAEMON_LOG}"
    if (( EUID == 0 )); then
        nohup env TUTTI_VERBOSE=1 "$DAEMON_BIN" --config "$DAEMON_CONFIG" >> "$DAEMON_LOG" 2>&1 < /dev/null &
    else
        nohup sudo env TUTTI_VERBOSE=1 "$DAEMON_BIN" --config "$DAEMON_CONFIG" >> "$DAEMON_LOG" 2>&1 < /dev/null &
    fi
    launcher=$!
    printf '%s\n' "$launcher" > "$DAEMON_PID_FILE"
    for attempt in $(seq 1 30); do
        pid="$(daemon_pid)"
        if [[ -n "$pid" ]] && daemon_port_listening; then
            printf '%s\n' "$pid" > "$DAEMON_PID_FILE"
            note "tutti_daemon 就绪，PID=${pid}，端口=${DAEMON_PORT}。"
            return 0
        fi
        if ! kill -0 "$launcher" 2>/dev/null && [[ -z "$pid" ]]; then
            tail -n 80 "$DAEMON_LOG" >&2 || true
            die "tutti_daemon 启动期间退出；完整日志：${DAEMON_LOG}"
        fi
        sleep 1
    done
    tail -n 80 "$DAEMON_LOG" >&2 || true
    die "tutti_daemon 在 30 秒内未监听 ${DAEMON_PORT}；完整日志：${DAEMON_LOG}"
}

stop_daemon() {
    local pid
    pid="$(daemon_pid)"
    [[ -z "$pid" ]] && { note 'tutti_daemon 未运行。'; return 0; }
    ensure_sudo
    if (( DRY_RUN )); then note "dry-run: sudo kill -TERM ${pid}"; return 0; fi
    # daemon 自带 SIGTERM 处理：drain gRPC -> umount（EBUSY 时会在日志里报持有者）
    note "停止 tutti_daemon PID=${pid}（SIGTERM，触发其 umount 流程）。"
    if (( EUID == 0 )); then kill -TERM "$pid"; else sudo kill -TERM "$pid"; fi
    for _ in $(seq 1 30); do
        [[ -z "$(daemon_pid)" ]] && { note 'tutti_daemon 已停止。'; rm -f "$DAEMON_PID_FILE"; return 0; }
        sleep 1
    done
    warn "tutti_daemon 30 秒内未退出（可能有进程占用挂载点）；见 ${DAEMON_LOG}"
    return 1
}

cmd_up() {
    note "Tutti 根目录：${TUTTI_ROOT}"
    note "build：${TUTTI_BUILD}"
    ensure_phoenix
    ensure_snvme
    start_daemon
    if (( DRY_RUN )); then note 'dry-run 结束。'; return 0; fi
    echo
    if check_all; then
        note '环境就绪。'
    else
        die '环境仍不完整（见上表）。'
    fi
}

main() {
    local subcmd=""
    while (( $# )); do
        case "$1" in
            up|status|down|restart) subcmd="$1" ;;
            --dry-run)    DRY_RUN=1 ;;
            --no-phoenix) WANT_PHOENIX=0 ;;
            -h|--help)    usage; return 0 ;;
            *) usage >&2; die "未知参数：$1" ;;
        esac
        shift
    done
    case "${subcmd:-up}" in
        up)      cmd_up ;;
        status)  check_all || die "环境不完整（${?} 项异常）" ;;
        down)    stop_daemon ;;
        restart) stop_daemon || true; cmd_up ;;
    esac
}

main "$@"
