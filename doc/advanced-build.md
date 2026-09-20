# 高级构建与驱动维护

普通开发只使用 [`getting-started.md`](getting-started.md)：

```bash
cmake --preset default
cmake --build --preset default --parallel 8
```

本文件只面向修改 `snvme` 内核模块、诊断 Kbuild、处理不受支持内核或移植其他 GPU
厂商的维护者。不要将这里的命令用于日常 NVIDIA 构建。

## 驱动代码修改后重编译与 reload

模块产物固定在 `build/module/`。修改驱动代码后：

```bash
cmake --build --preset default --target modules
```

修改模块后必须 reload；否则用户态 ioctl 编号和已加载模块可能不一致：

```bash
sudo bash scripts/unbind.sh
lsof /dev/snvm_control /dev/ssnvme* /dev/snvme*n* 2>/dev/null
cmake --build --preset default --target rmmod
cmake --build --preset default --target insmod
```

`rmmod` 前必须停止 `tutti_daemon`，卸载相关文件系统，并确保没有进程持有
`/dev/snvm*`。

## 直接 Kbuild

CMake 生成的 Kbuild 输出目录为 `build/module/`。仅在诊断模块编译问题时直接调用：

```bash
make -C build/module
```

日常重编译仍优先使用：

```bash
cmake --build --preset default --target modules
```

## Kernel baseline

统一模块树位于 `tutti/device_manager/nvme/kernel_modules/snvme/`。其 Kbuild 根据运行
内核自动选择 baseline：

| 运行内核 | baseline |
| --- | --- |
| 5.4.x | `5.4-tlinux4` |
| 5.10.x | `5.10` |
| 5.11–5.19 | `5.15` |
| 6.x | `6.8` |

查看可用 baseline：

```bash
find tutti/device_manager/nvme/kernel_modules/snvme/baseline \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
```

`SNVME_KERNEL_VERSION` 是历史版本树参数，不用于统一模块树。仅在维护 Kbuild baseline
时，才可使用诊断覆盖：

```bash
make -C build/module SNVME_BASELINE=<baseline>
```

不要用 baseline 覆盖绕过不受支持的运行内核。

## MUSA / MACA 厂商构建

共享 `default` preset 固定启用完整硬件栈，但不固定 accelerator、SDK 路径或 P2P
backend：根 CMake 默认 `CUDA`，所以 NVIDIA 用户无需额外参数。厂商机器也不新增共享
preset；使用同一个 `build/`，在**全新 build 目录**上通过 CMake cache 参数覆盖。

```bash
rm -rf build
```

### MUSA

`MUSA.cmake` 的默认 SDK 路径是 `/usr/local/musa/include` 与
`/usr/local/musa/lib`。非标准安装时明确传入路径；SNVMe 的 Metax P2P 头也必须传入：

```bash
cmake --preset default \
  -DTUTTI_ACCELERATOR=MUSA \
  -DMUSA_INCLUDE_DIR=/path/to/musa/include \
  -DMUSA_LIB_DIR=/path/to/musa/lib \
  -DSNVME_P2P_INCLUDE_DIR=/path/to/metax/p2p/include
cmake --build --preset default --parallel 8
```

当前顶层仍以 CUDA language 初始化设备代码。因此 MUSA 机器还必须由厂商 SDK 提供
兼容的 CUDA-language compiler，或在第一条命令中追加厂商要求的
`-DCMAKE_CUDA_COMPILER=<compiler>`。该路径尚未做硬件验收。

### MACA

`MACA.cmake` 读取 `MACA_HOME`，默认回退 `/opt/maca`，并优先使用 `lib64`。应使用
厂商的 `cmake_maca` wrapper 以设置其 CUDA-language toolchain：

```bash
rm -rf build
export MACA_HOME=/opt/maca
cmake_maca --preset default \
  -DTUTTI_ACCELERATOR=MACA \
  -DMACA_ROOT="$MACA_HOME" \
  -DSNVME_P2P_INCLUDE_DIR=/path/to/metax/p2p/include
cmake --build build --parallel 8
```

MACA profile 目前在源码中仍标记为 template，实际 compiler、runtime 库和 P2P 头路径
必须由厂商确认；不能把以上命令视为已完成硬件认证。

### preset 与依赖来源

`CMakePresets.json` 是仓库共享的基线，应随源码提交；它只定义一个 `default`。当前
`prepare_env.sh` **不生成 preset**，只安装或确认依赖。CMake 先查找系统包；仅当
`yaml-cpp`、gRPC 或 Protobuf 缺失且仓库已有 `third_pkgs/vcpkg` 安装时，才自动把它加入
`CMAKE_PREFIX_PATH` 作为后备。因此普通机器有完整系统开发包时无需 vcpkg，也无需
`CMAKE_TOOLCHAIN_FILE`。

`CMakeUserPresets.json` 是被忽略的本机文件，适合 IDE 或个人快捷命令；不要用它新增
第二套共享构建矩阵。切换 accelerator、compiler 或 SDK 后必须删除 `build/` 再重新配置。

## 相关文档

- [`extending_tutti.md` 的 SNVMe driver smoke tests](extending_tutti.md#snvme-driver-smoke-tests)：模块 smoke test 和恢复流程。
- [`tutti_daemon.md`](tutti_daemon.md)：模块加载后的 daemon 配置与部署。
- [`design/kernel-portability.md`](design/kernel-portability.md)：跨内核和 GPU vendor 的实现设计。
