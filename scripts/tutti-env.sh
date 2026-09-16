#!/usr/bin/env bash
# Tutti 环境管理：构建、内核模块、daemon、测试的单一入口。
#
# 子命令
#   status                  只读全面检查（构建产物 / Python 扩展 / 内核模块 /
#                           daemon / 挂载）。退出码非 0 表示环境不完整。
#   bootstrap               从零到可运行：build -> build-ext -> up
#   build [--clean]         cmake configure + 构建全部 C++ 目标（含内核模块）
#   build-ext               构建两个 Python 扩展（tutti_runtime、tutti_kv_transfer）
#   modules load|unload|rebuild
#   daemon start|stop|restart
#   up                      modules load + daemon start + status（**不编译**）
#   down                    停止 daemon（不卸载模块）
#   test py|cpp|all         跑测试（内置正确的 PYTHONPATH 与前置检查）
#   env                     打印供 shell eval 的环境变量（PYTHONPATH 等）
#
# 选项
#   --dry-run       只打印将执行的命令
#   --no-phoenix    跳过 phoenixfs（RDMA 场景需卸载它）
#   -j N            构建并行度（默认 nproc 上限 32）
#   --clean         仅 build：先删 build/ 再全新 configure
#
# 环境变量覆盖
#   TUTTI_ROOT TUTTI_BUILD TUTTI_DAEMON_CONFIG TUTTI_LOG_DIR TUTTI_DAEMON_PORT
#   PHOENIX_BUILD PYTHON_BIN
#
# 依赖顺序不可交换：/dev/snvme*n1 由 daemon bring-up 后才出现，先 mount 会失败。
#   内核模块 -> tutti_daemon（建块设备节点 + auto_mount + gRPC）-> 负载
#
# 本脚本不签名内核模块。若加载失败且 dmesg 出现 "Required key not available"，
# 请按站点签名流程签名 build/module/*.ko 后重跑。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TUTTI_ROOT="${TUTTI_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
readonly TUTTI_BUILD="${TUTTI_BUILD:-${TUTTI_ROOT}/build}"
readonly DAEMON_CONFIG="${TUTTI_DAEMON_CONFIG:-${TUTTI_ROOT}/config/local/tutti_daemon.yaml}"
readonly PHOENIX_BUILD="${PHOENIX_BUILD:-/data/home/ryeqiu/phoenix/build}"
readonly PHOENIX_KO="${PHOENIX_BUILD}/module/phoenixfs.ko"
readonly LOG_DIR="${TUTTI_LOG_DIR:-${TUTTI_ROOT}/.local/log}"
readonly DAEMON_LOG="${LOG_DIR}/tutti_daemon.log"
readonly DAEMON_PID_FILE="${LOG_DIR}/tutti_daemon.pid"
readonly DAEMON_PORT="${TUTTI_DAEMON_PORT:-50051}"
readonly SNVME_CORE_KO="${TUTTI_BUILD}/module/snvme-core.ko"
readonly SNVME_KO="${TUTTI_BUILD}/module/snvme.ko"
readonly DAEMON_BIN="${TUTTI_BUILD}/bin/tutti_daemon"
readonly LIBNVM_SO="${TUTTI_BUILD}/csrc/device_manager/nvme/libnvm/libnvm.so"
readonly PYBIND_DIR="${TUTTI_ROOT}/csrc/python"
readonly KVTRANSFER_DIR="${TUTTI_ROOT}/csrc/kv_transfer"

# 解释器必须与扩展的 ABI 标签一致（.so 名形如 _core.cpython-311-*.so）。
# 系统 python3 常常过旧（本机 /usr/bin/python3 是 3.6.8），故按此优先级发现：
#   PYTHON_BIN 显式指定 > 已激活的 venv > python3。
discover_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then printf '%s' "$PYTHON_BIN"; return; fi
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        printf '%s' "${VIRTUAL_ENV}/bin/python"; return
    fi
    printf 'python3'
}
readonly PYTHON_BIN="$(discover_python)"

DRY_RUN=0
WANT_PHOENIX=1
DO_CLEAN=0
JOBS=""
SUDO_READY=0

note() { printf '[tutti] %s\n' "$*"; }
warn() { printf '[tutti] WARN: %s\n' "$*" >&2; }
die()  { printf '[tutti] ERROR: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

jobs_arg() { printf '%s' "${JOBS:-$(nproc 2>/dev/null | awk '{print ($1>32)?32:$1}')}"; }
module_loaded() { [[ -d "/sys/module/$1" ]]; }
daemon_pid() { pgrep -n -x tutti_daemon 2>/dev/null || true; }

daemon_port_listening() {
    command -v ss >/dev/null 2>&1 || return 0   # 无 ss 时不作为判据
    ss -ltnH 2>/dev/null | awk -v p="$DAEMON_PORT" '$4 ~ (":" p "$"){f=1} END{exit !f}'
}

ensure_sudo() {
    (( SUDO_READY || DRY_RUN )) && return 0
    if (( EUID == 0 )); then SUDO_READY=1; return 0; fi
    command -v sudo >/dev/null 2>&1 || die '需要 root 或 sudo 来加载模块 / 启动 daemon。'
    sudo -v || die '需要 sudo 权限。'
    SUDO_READY=1
}

run() {  # dry-run 感知的执行器
    if (( DRY_RUN )); then note "dry-run: $*"; return 0; fi
    "$@"
}

# 供 pytest / 手动运行使用：仓库根 + 两个扩展的 in-place 产物目录
tutti_pythonpath() {
    printf '%s:%s/src:%s' "$TUTTI_ROOT" "$PYBIND_DIR" "$KVTRANSFER_DIR"
}

python_can_import() {
    PYTHONPATH="$(tutti_pythonpath)" "$PYTHON_BIN" -c "import $1" 2>/dev/null
}

python_version() {
    "$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?"
}

# 区分"扩展未构建"与"解释器与扩展 ABI 不匹配"——后者最容易浪费时间。
# $1=模块名 $2=搜索 .so 的目录
extension_state() {
    local module="$1" dir="$2" so tag
    if python_can_import "$module"; then printf 'ok'; return; fi
    so="$(find "$dir" -name '*.so' 2>/dev/null | head -1)"
    if [[ -z "$so" ]]; then printf 'missing'; return; fi
    tag="$(basename "$so" | sed -n 's/.*cpython-\([0-9]*\).*/\1/p')"
    if [[ -n "$tag" ]]; then
        local want="${tag:0:1}.${tag:1}"
        [[ "$want" != "$(python_version)" ]] && { printf 'abi:%s' "$want"; return; }
    fi
    printf 'broken'
}

# ---- 只读检查 ---------------------------------------------------------------

check_all() {
    local quiet="${1:-}" bad=0 pid nodes mounted expected runpath

    _row() {
        [[ "$quiet" == "--quiet" ]] && return 0
        local mark
        case "$2" in
            ok) mark='  OK  ' ;; skip) mark=' SKIP ' ;; *) mark=' FAIL ' ;;
        esac
        printf '[%s] %-24s %s\n' "$mark" "$1" "$3"
    }

    # --- 构建产物 ---
    if [[ -x "$DAEMON_BIN" ]]; then
        _row 'tutti_daemon 二进制' ok "$DAEMON_BIN"
    else
        _row 'tutti_daemon 二进制' bad "缺失（跑 $0 build）"; bad=$((bad + 1))
    fi
    if [[ -f "$SNVME_CORE_KO" && -f "$SNVME_KO" ]]; then
        _row '内核模块 .ko' ok 'snvme-core.ko + snvme.ko 已构建'
    else
        _row '内核模块 .ko' bad "缺失（跑 $0 build）"; bad=$((bad + 1))
    fi
    if [[ -f "$LIBNVM_SO" ]]; then
        _row libnvm.so ok "$LIBNVM_SO"
    else
        _row libnvm.so bad '缺失：Python 扩展的 RUNPATH 指向此处，缺则 import 失败'
        bad=$((bad + 1))
    fi

    # --- Python 解释器与扩展 ---
    local pyver state
    pyver="$(python_version)"
    if [[ "$pyver" == "?" ]]; then
        _row 'Python 解释器' bad "${PYTHON_BIN} 不可用"; bad=$((bad + 1))
    elif [[ "${pyver%%.*}" -ge 3 && "${pyver#*.}" -ge 10 ]]; then
        _row 'Python 解释器' ok "${PYTHON_BIN} (${pyver})"
    else
        _row 'Python 解释器' bad "${PYTHON_BIN} 是 ${pyver}，项目要求 >= 3.10（用 PYTHON_BIN= 指定或激活 venv）"
        bad=$((bad + 1))
    fi

    _ext_row() {  # $1=显示名 $2=模块 $3=目录
        local st; st="$(extension_state "$2" "$3")"
        case "$st" in
            ok)      _row "$1" ok '可导入' ;;
            missing) _row "$1" bad "未构建（跑 $0 build-ext）"; bad=$((bad + 1)) ;;
            abi:*)   _row "$1" bad ".so 是 cpython-${st#abi:} 而当前解释器 ${pyver}，ABI 不匹配"
                     bad=$((bad + 1)) ;;
            *)       _row "$1" bad "已构建但导入失败（多为依赖缺失，见 $0 build-ext 输出）"
                     bad=$((bad + 1)) ;;
        esac
    }
    _ext_row 'tutti_runtime 扩展' tutti_runtime "${PYBIND_DIR}/src"
    _ext_row 'tutti_kv_transfer 扩展' tutti_kv_transfer "$KVTRANSFER_DIR"
    # RUNPATH 与实际 .so 位置是否一致（构建树被删/迁移后最常见的坑）
    local core_so
    core_so="$(find "${PYBIND_DIR}/src" -name '_core*.so' 2>/dev/null | head -1)"
    if [[ -n "$core_so" ]] && command -v readelf >/dev/null 2>&1; then
        runpath="$(readelf -d "$core_so" 2>/dev/null |
                   sed -n 's/.*RUNPATH.*\[\(.*\)\]/\1/p' | tr ':' '\n' |
                   grep -m1 libnvm || true)"
        if [[ -z "$runpath" ]]; then
            _row 'RUNPATH -> libnvm' skip '未在 RUNPATH 中声明'
        elif [[ -f "${runpath}/libnvm.so" ]]; then
            _row 'RUNPATH -> libnvm' ok "$runpath"
        else
            _row 'RUNPATH -> libnvm' bad "指向 ${runpath} 但该处无 libnvm.so（构建树被删/迁移过）"
            bad=$((bad + 1))
        fi
    fi

    # --- 内核模块 ---
    if module_loaded phoenixfs; then
        _row phoenixfs ok '已加载'
    elif (( WANT_PHOENIX )); then
        _row phoenixfs skip '未加载（仅 Phoenix GDS 路径需要）'
    else
        _row phoenixfs skip '按 --no-phoenix 跳过'
    fi
    if module_loaded snvme_core && module_loaded snvme; then
        _row 'snvme 模块' ok 'snvme-core + snvme 已加载'
    else
        _row 'snvme 模块' bad "core=$(module_loaded snvme_core && echo Y || echo N) pci=$(module_loaded snvme && echo Y || echo N)"
        bad=$((bad + 1))
    fi
    if [[ -e /dev/snvm_control ]]; then
        _row /dev/snvm_control ok '存在'
    else
        _row /dev/snvm_control bad '缺失（模块未正确初始化）'; bad=$((bad + 1))
    fi

    # --- daemon ---
    pid="$(daemon_pid)"
    if [[ -n "$pid" ]]; then
        _row tutti_daemon ok "运行中 PID=${pid}"
    else
        _row tutti_daemon bad "未运行（跑 $0 daemon start）"; bad=$((bad + 1))
    fi
    if [[ -n "$pid" ]] && daemon_port_listening; then
        _row "gRPC :${DAEMON_PORT}" ok '监听中'
    elif [[ -n "$pid" ]]; then
        _row "gRPC :${DAEMON_PORT}" bad '进程在但端口未监听'; bad=$((bad + 1))
    else
        _row "gRPC :${DAEMON_PORT}" bad '未监听'; bad=$((bad + 1))
    fi

    # 块设备节点由 daemon bring-up 创建，是 "daemon 真的干完活了" 的证据
    nodes="$(ls /dev/snvme*n1 2>/dev/null | wc -l)"
    if (( nodes > 0 )); then
        _row '块设备节点' ok "${nodes} 个 /dev/snvme*n1"
    else
        _row '块设备节点' bad '无 /dev/snvme*n1（daemon 未完成 bring-up）'; bad=$((bad + 1))
    fi

    # auto_mount 未生效时 KV 会落到根盘，现象只是"慢"，极难察觉
    expected="$(grep -c 'auto_mount: true' "$DAEMON_CONFIG" 2>/dev/null || echo 0)"
    mounted="$(mount | grep -c '^/dev/snvme[0-9]*n1 on ' || true)"
    if (( expected > 0 && mounted >= expected )); then
        _row 'auto_mount 挂载' ok "${mounted}/${expected} 已挂载"
    elif (( expected > 0 )); then
        _row 'auto_mount 挂载' bad "${mounted}/${expected}（配置期望 ${expected}）"; bad=$((bad + 1))
    else
        _row 'auto_mount 挂载' skip '配置未启用 auto_mount'
    fi

    return "$bad"
}

# ---- 构建 -------------------------------------------------------------------

cmd_build() {
    command -v cmake >/dev/null 2>&1 || die '未找到 cmake。'
    if (( DO_CLEAN )); then
        note "删除 ${TUTTI_BUILD} 后全新配置。"
        run rm -rf "$TUTTI_BUILD"
    fi
    note "配置（preset default：RelWithDebInfo + 内核模块 + 硬件测试）。"
    ( cd "$TUTTI_ROOT" && run cmake --preset default ) || die 'cmake 配置失败。'
    note "构建全部目标，并行度 $(jobs_arg)。modules 是 ALL 目标，一并构建。"
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --parallel "$(jobs_arg)" ) \
        || die 'C++ 构建失败。'
    (( DRY_RUN )) && return 0
    [[ -x "$DAEMON_BIN" ]] || warn "构建结束但缺 ${DAEMON_BIN}"
    [[ -f "$LIBNVM_SO" ]] || warn "构建结束但缺 ${LIBNVM_SO}"
    note '构建完成。'
}

cmd_build_ext() {
    [[ -f "$LIBNVM_SO" ]] || die "缺 ${LIBNVM_SO}；先跑 $0 build。"
    # 必须清干净再建：build_ext 会复用旧对象文件，导致 RUNPATH 仍指向旧构建树。
    note '构建 pybind 扩展 tutti_runtime。'
    run rm -rf "${PYBIND_DIR}/build"
    if (( ! DRY_RUN )); then find "${PYBIND_DIR}/src" -name '*.so' -delete; fi
    ( cd "$PYBIND_DIR" && run env TUTTI_ROOT="$TUTTI_ROOT" TUTTI_BUILD_DIR="$TUTTI_BUILD" \
        "$PYTHON_BIN" setup.py build_ext --inplace ) || die 'tutti_runtime 构建失败。'
    note '构建 CUDA 扩展 tutti_kv_transfer。'
    run rm -rf "${KVTRANSFER_DIR}/build"
    if (( ! DRY_RUN )); then find "$KVTRANSFER_DIR" -name '*.so' -delete; fi
    ( cd "$KVTRANSFER_DIR" && run "$PYTHON_BIN" setup.py build_ext --inplace ) \
        || die 'tutti_kv_transfer 构建失败（需要 torch + nvcc）。'
    (( DRY_RUN )) && return 0
    python_can_import tutti_runtime || die 'tutti_runtime 构建后仍不可导入。'
    python_can_import tutti_kv_transfer || die 'tutti_kv_transfer 构建后仍不可导入。'
    note '两个 Python 扩展就绪。'
}

# ---- 内核模块 ---------------------------------------------------------------

modules_rebuild() {
    note '重建内核模块（clean_modules + modules）。'
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target clean_modules ) || true
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target modules \
        --parallel "$(jobs_arg)" ) || die '内核模块构建失败。'
    note '内核模块已重建；load 前请确认已按站点流程签名。'
}

modules_load() {
    local core=0 pci=0
    module_loaded snvme_core && core=1
    module_loaded snvme && pci=1
    (( core && pci )) && { note 'snvme-core 和 snvme 已加载。'; return 0; }
    (( core || pci )) && die '检测到半加载状态；请先 modules unload 清理。'
    [[ -f "$SNVME_CORE_KO" && -f "$SNVME_KO" ]] || die "缺内核模块；先跑 $0 build。"
    ensure_sudo
    note '加载 snvme-core 和 snvme（insmod target 只转发 make insmod，不构建）。'
    if ! ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target insmod ); then
        warn '加载失败。若 dmesg 出现 "Required key not available" / signature，'
        warn "请签名以下模块后重试： ${SNVME_CORE_KO} ${SNVME_KO}"
        die 'SNVMe 模块加载失败。'
    fi
    (( DRY_RUN )) && return 0
    module_loaded snvme_core || die 'snvme-core 未加载。'
    module_loaded snvme || die 'snvme 未加载。'
    [[ -e /dev/snvm_control ]] || die '模块已加载但 /dev/snvm_control 不存在。'
}

modules_unload() {
    module_loaded snvme || module_loaded snvme_core || { note 'snvme 模块未加载。'; return 0; }
    # daemon 持有模块引用；不先停它 rmmod 必然 EBUSY
    [[ -n "$(daemon_pid)" ]] && { warn 'tutti_daemon 在运行，会持有模块引用。'; stop_daemon; }
    ensure_sudo
    note '卸载 snvme 与 snvme-core。卸载后 /dev/snvme* 与 /dev/ssnvme* 全部消失。'
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target rmmod ) \
        || die '卸载失败（可能仍有进程占用）。'
}

ensure_phoenix() {
    (( WANT_PHOENIX )) || { note '按 --no-phoenix 跳过 phoenixfs。'; return 0; }
    module_loaded phoenixfs && { note 'phoenixfs 已加载。'; return 0; }
    if [[ ! -f "$PHOENIX_KO" ]]; then
        warn "缺 ${PHOENIX_KO}，跳过（仅影响 Phoenix GDS 路径）。"
        warn "需要它请先手动构建： cd ${PHOENIX_BUILD} && make modules"
        return 0
    fi
    ensure_sudo
    note '加载 phoenixfs。'
    if (( EUID == 0 )); then
        run insmod "$PHOENIX_KO" || { warn 'phoenixfs 加载失败（可能需签名）。'; return 0; }
    else
        ( cd "$PHOENIX_BUILD" && run make insmod ) \
            || { warn 'phoenixfs 加载失败（可能需签名）。'; return 0; }
    fi
}

# ---- daemon -----------------------------------------------------------------

start_daemon() {
    local pid launcher attempt
    if pid="$(daemon_pid)"; [[ -n "$pid" ]]; then
        mkdir -p "$LOG_DIR"; printf '%s\n' "$pid" > "$DAEMON_PID_FILE"
        note "tutti_daemon 已运行，PID=${pid}。"
        return 0
    fi
    [[ -x "$DAEMON_BIN" ]] || die "缺 ${DAEMON_BIN}；先跑 $0 build。"
    [[ -f "$DAEMON_CONFIG" ]] || die "缺配置 ${DAEMON_CONFIG}"
    module_loaded snvme || die "snvme 未加载；先跑 $0 modules load。"
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
            die "tutti_daemon 启动期间退出；日志：${DAEMON_LOG}"
        fi
        sleep 1
    done
    tail -n 80 "$DAEMON_LOG" >&2 || true
    die "tutti_daemon 30 秒内未监听 ${DAEMON_PORT}；日志：${DAEMON_LOG}"
}

stop_daemon() {
    local pid
    pid="$(daemon_pid)"
    [[ -z "$pid" ]] && { note 'tutti_daemon 未运行。'; return 0; }
    ensure_sudo
    if (( DRY_RUN )); then note "dry-run: sudo kill -TERM ${pid}"; return 0; fi
    # daemon 自带 SIGTERM 处理：drain gRPC -> umount（EBUSY 时日志里报持有者）
    note "停止 tutti_daemon PID=${pid}（SIGTERM，触发其 umount 流程）。"
    if (( EUID == 0 )); then kill -TERM "$pid"; else sudo kill -TERM "$pid"; fi
    for _ in $(seq 1 30); do
        [[ -z "$(daemon_pid)" ]] && { note 'tutti_daemon 已停止。'; rm -f "$DAEMON_PID_FILE"; return 0; }
        sleep 1
    done
    warn "30 秒内未退出（可能有进程占用挂载点）；见 ${DAEMON_LOG}"
    return 1
}

# ---- 测试 -------------------------------------------------------------------

cmd_test() {
    local what="${1:-all}" rc=0
    case "$what" in
        py|all)
            note '跑 Python 测试（tests/python）。'
            ( cd "$TUTTI_ROOT" && run env -u PYTHONPATH PYTHONPATH="$(tutti_pythonpath)" \
                "$PYTHON_BIN" -m pytest tests/python -q ) || rc=1
            ;;
    esac
    case "$what" in
        cpp|all)
            note '跑 C++ 测试（ctest）。硬件契约测试需要环境就绪。'
            if ! check_all --quiet; then
                warn "环境不完整，硬件契约测试会失败。先跑 $0 status 查看。"
            fi
            ( cd "$TUTTI_BUILD" && run ctest --output-on-failure ) || rc=1
            ;;
    esac
    case "$what" in
        py|cpp|all) ;;
        *) die "test 的参数须为 py|cpp|all，收到：${what}" ;;
    esac
    return "$rc"
}

# ---- 组合动作 ---------------------------------------------------------------

cmd_up() {
    note "Tutti 根目录：${TUTTI_ROOT}"
    ensure_phoenix
    modules_load
    start_daemon
    (( DRY_RUN )) && { note 'dry-run 结束。'; return 0; }
    echo
    if check_all; then note '环境就绪。'; else die '环境仍不完整（见上表）。'; fi
}

cmd_bootstrap() {
    note '从零 bootstrap：build -> build-ext -> up'
    cmd_build
    cmd_build_ext
    cmd_up
}

cmd_env() {
    printf 'export PYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}"\n' "$(tutti_pythonpath)"
    printf 'export TUTTI_ROOT="%s"\n' "$TUTTI_ROOT"
    printf 'export TUTTI_BUILD_DIR="%s"\n' "$TUTTI_BUILD"
}

main() {
    local subcmd="" arg=""
    while (( $# )); do
        case "$1" in
            status|bootstrap|build|build-ext|modules|daemon|up|down|test|env)
                if [[ -z "$subcmd" ]]; then subcmd="$1"; else arg="$1"; fi ;;
            load|unload|rebuild|start|stop|restart|py|cpp|all) arg="$1" ;;
            --dry-run)    DRY_RUN=1 ;;
            --no-phoenix) WANT_PHOENIX=0 ;;
            --clean)      DO_CLEAN=1 ;;
            -j)           shift; JOBS="${1:?-j 需要数值}" ;;
            -j*)          JOBS="${1#-j}" ;;
            -h|--help)    usage; return 0 ;;
            *) usage >&2; die "未知参数：$1" ;;
        esac
        shift
    done
    case "${subcmd:-status}" in
        status)    check_all || die "环境不完整（$? 项异常）" ;;
        bootstrap) cmd_bootstrap ;;
        build)     cmd_build ;;
        build-ext) cmd_build_ext ;;
        modules)
            case "${arg:-}" in
                load) modules_load ;; unload) modules_unload ;; rebuild) modules_rebuild ;;
                *) die 'modules 须带 load|unload|rebuild' ;;
            esac ;;
        daemon)
            case "${arg:-}" in
                start) start_daemon ;; stop) stop_daemon ;;
                restart) stop_daemon || true; start_daemon ;;
                *) die 'daemon 须带 start|stop|restart' ;;
            esac ;;
        up)   cmd_up ;;
        down) stop_daemon ;;
        test) cmd_test "${arg:-all}" ;;
        env)  cmd_env ;;
    esac
}

main "$@"
