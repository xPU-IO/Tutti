# Tutti 手动构建指南

> 本文是 Tutti 的**手动编译唯一入口**。它只使用仓库根目录的 CMake、现有
> `third_pkgs/vcpkg` 安装结果和显式的 `cmake -S/-B` 参数，因此不会安装依赖、
> 不会修改 `third_pkgs/vcpkg`，也不会加载内核模块。
>
> 成功标准：完成对应 profile 的 configure、build 与测试；可选的 kernel-module
> 路径还应生成 `snvme.ko` 和 `snvme-core.ko`。加载模块、接管 NVMe、格式化磁盘和
> 启动 daemon 不属于本文范围。

## 1. 文档边界

仓库中有历史 preset、部署和测试文档。为避免混用，按下表分工：

| 文档 | 负责内容 |
| --- | --- |
| **本文** | 手动 configure、build、非硬件测试，以及 kernel-module 的编译前置条件 |
| [`build_and_test.md`](build_and_test.md) | 模块已经编译后的 SNVMe smoke-test 阶梯；其中的 preset 示例是可选 bootstrap 路径，不与本文的构建目录混用 |
| [`tutti_daemon.md`](tutti_daemon.md) | 已编译模块后的磁盘准备、daemon 与挂载运维；部署配置以 `config/local_nvme_config.yaml` 的 canonical schema 为准 |
| [`../examples/layerwise_kv_overlap/README.md`](../examples/layerwise_kv_overlap/README.md) | 已启动 daemon 后，`layerwise_kv_overlap` 的参数和运行方式 |

根目录是唯一支持的 CMake source directory；不要把 `tutti/` 子目录单独传给
`cmake -S`。

## 2. 为什么不用 preset

`CMakePresets.json` 是被忽略的机器本地文件，可能由
`scripts/prepare_env.sh` 生成，也可能保留历史内容。不同机器上的 `host`、`cuda`
和 `cuda-module` 名称并不保证存在或含有相同参数。

本指南故意不用 `cmake --preset`，而是固定使用显式的构建目录：

- `build/manual-host`：不需要 CUDA 的 API/SPI 契约测试；
- `build/manual-cuda`：CUDA 用户态、daemon 和非硬件测试；
- `build/manual-cuda-module`：在前两者基础上构建 `snvme` 内核模块。

新机器尚未安装依赖时，才考虑运行 `scripts/prepare_env.sh`。该脚本会调用系统包
管理器，并可能通过 vcpkg 安装包和重写本地 preset；当前已具备 vcpkg 依赖的环境
不需要运行它。

## 3. 一次性预检

在仓库根目录执行。手动路径最低要求为 CMake 3.18；若要使用生成的 preset，才需要
CMake 3.21 以上。CUDA profile 需要 CUDA Toolkit 12.6 以上。

下面假定当前已下载的 vcpkg triplet 是本仓库使用的 `x64-linux`。若构建其他 CPU
架构，先准备该架构对应的 vcpkg 安装结果，再把检查路径和 `VCPKG_TARGET_TRIPLET`
一并换成实际值。

```bash
cd /path/to/Tutti

export ROOT="$PWD"
export TUTTI_VCPKG_TOOLCHAIN="$ROOT/third_pkgs/vcpkg/scripts/buildsystems/vcpkg.cmake"
export JOBS="${JOBS:-$(nproc)}"

for required in \
  "$TUTTI_VCPKG_TOOLCHAIN" \
  "$ROOT/third_pkgs/vcpkg/installed/x64-linux/share/grpc/gRPCConfig.cmake" \
  "$ROOT/third_pkgs/vcpkg/installed/x64-linux/share/yaml-cpp/yaml-cpp-config.cmake"; do
  test -f "$required" || {
    printf '缺少已安装依赖：%s\n' "$required" >&2
    exit 1
  }
done

command -v cmake >/dev/null || { echo '未找到 cmake' >&2; exit 1; }
command -v c++ >/dev/null || { echo '未找到 C++ 编译器' >&2; exit 1; }
cmake --version
c++ --version
```

CUDA 构建额外检查：

```bash
command -v nvcc >/dev/null || { echo '未找到 nvcc' >&2; exit 1; }
nvcc --version

# 已加载 NVIDIA 驱动时，可由此读取计算能力；仅编译用户态时不强制要求 nvidia-smi。
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
else
  echo '未找到 nvidia-smi；请按目标 GPU 的计算能力手动设置 CUDA_ARCH。' >&2
fi
```

`CMAKE_CUDA_ARCHITECTURES` 使用不带小数点的计算能力；例如上面命令输出 `9.0` 时
填写 `90`。本文以 `90` 为例，必须按实际 GPU 修改。

> `CMAKE_TOOLCHAIN_FILE`、编译器和 accelerator profile 都在第一次 configure 时写入
> build cache。要切换其中任何一项，请使用新的构建目录，或删除对应的
> `build/manual-*` 目录后重新 configure。

## 4. 先验证 HOST profile

HOST profile 不编译 CUDA、daemon 或 kernel module，适合先确认公共 API、配置解析和
基础构建环境。这是排查环境问题的第一步。

```bash
export HOST_BUILD="$ROOT/build/manual-host"

cmake -S "$ROOT" -B "$HOST_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_TOOLCHAIN_FILE="$TUTTI_VCPKG_TOOLCHAIN" \
  -DTUTTI_ACCELERATOR=HOST \
  -DBUILD_TESTING=ON \
  -DTUTTI_BUILD_HARDWARE_TESTS=OFF \
  -DTUTTI_BUILD_KERNEL_MODULE=OFF

cmake --build "$HOST_BUILD" --parallel "$JOBS"
ctest --test-dir "$HOST_BUILD" --output-on-failure
```

成功后，`$HOST_BUILD/bin/` 内含契约测试二进制；这一步不需要 GPU、NVMe、root 权限
或已加载的模块。

## 5. 构建 CUDA 用户态

此 profile 构建 CUDA runtime、`libnvm`、gRPC NVMe service、`tutti_daemon` 和非硬件
测试，但**不**构建或加载内核模块。它是日常用户态开发和回归测试的默认选择。

```bash
export CUDA_ARCH=90                  # 按本机 GPU 修改
export CUDA_BUILD="$ROOT/build/manual-cuda"

cmake -S "$ROOT" -B "$CUDA_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_TOOLCHAIN_FILE="$TUTTI_VCPKG_TOOLCHAIN" \
  -DTUTTI_ACCELERATOR=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DTUTTI_BUILD_HARDWARE_STACK=ON \
  -DTUTTI_FEATURE_LOCAL_NVME=ON \
  -DBUILD_TESTING=ON \
  -DTUTTI_BUILD_HARDWARE_TESTS=OFF \
  -DTUTTI_BUILD_KERNEL_MODULE=OFF

cmake --build "$CUDA_BUILD" --parallel "$JOBS"
ctest --test-dir "$CUDA_BUILD" --output-on-failure -LE hardware
```

常用产物位于 `$CUDA_BUILD/bin/`：

| 产物 | 用途 |
| --- | --- |
| `tutti_daemon` | NVMe owner / gRPC daemon；运行需要另行完成模块和磁盘部署 |
| `nvmeservice_client` | 查询 daemon 资源和 accelerator view 的诊断客户端 |
| `tutti_runtime_example` | TuttiRuntime 的普通示例 |

`TUTTI_BUILD_HARDWARE_TESTS=OFF` 时不会构建真实 NVMe 工作负载；这样普通 build/test
不会触碰 GPU、设备节点或磁盘。

## 6. 可选：编译 SNVMe 内核模块

### 6.1 前置条件

模块编译与 vcpkg 无关，另外需要：

1. 当前正在运行内核的 build tree：`/lib/modules/$(uname -r)/build`；
2. NVIDIA 驱动提供的 `nv-p2p.h`；
3. CUDA profile 和本节的 vcpkg C++ 依赖。

先显式检查，找不到 P2P 头文件时应停止在此处；CUDA 用户态构建仍可正常使用。

```bash
export KERNEL_BUILD="/lib/modules/$(uname -r)/build"
export P2P_HEADER=""

if test -f "$ROOT/third_pkgs/nvidia/nv-p2p.h"; then
  P2P_HEADER="$ROOT/third_pkgs/nvidia/nv-p2p.h"
else
  P2P_HEADER="$(find /usr/src -type f -name nv-p2p.h -print -quit 2>/dev/null)"
fi

test -d "$KERNEL_BUILD" || {
  echo "缺少当前内核的 build tree: $KERNEL_BUILD" >&2
  exit 1
}
test -n "$P2P_HEADER" || {
  echo "未找到 nv-p2p.h；先安装匹配 NVIDIA 驱动的 kernel source/devel 包，或指定其目录。" >&2
  exit 1
}

export SNVME_P2P_INCLUDE_DIR="$(dirname "$P2P_HEADER")"
```

这与 CMake 的默认搜索次序一致：优先仓库外部依赖目录中的
`third_pkgs/nvidia/nv-p2p.h`，再搜索 `/usr/src/nvidia-*`。显式传入
`SNVME_P2P_INCLUDE_DIR` 可避免本机驱动头文件位置不一致。

统一的 `snvme` 源码树按当前 `uname -r` 自动选择 `5.4-tlinux4`、`5.10`、`5.15` 或
`6.8` baseline。正常本机构建**不要设置**历史变量 `SNVME_KERNEL_VERSION`；它不负责
选择统一树内实际编译的 baseline。

### 6.2 只编译，不加载

```bash
export MODULE_BUILD="$ROOT/build/manual-cuda-module"

cmake -S "$ROOT" -B "$MODULE_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_TOOLCHAIN_FILE="$TUTTI_VCPKG_TOOLCHAIN" \
  -DTUTTI_ACCELERATOR=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DTUTTI_BUILD_HARDWARE_STACK=ON \
  -DTUTTI_FEATURE_LOCAL_NVME=ON \
  -DBUILD_TESTING=ON \
  -DTUTTI_BUILD_HARDWARE_TESTS=OFF \
  -DTUTTI_BUILD_KERNEL_MODULE=ON \
  -DTUTTI_P2P_BACKEND=nvidia \
  -DSNVME_P2P_INCLUDE_DIR="$SNVME_P2P_INCLUDE_DIR"

cmake --build "$MODULE_BUILD" \
  --target modules tutti_daemon \
  --parallel "$JOBS"

ls -lh "$MODULE_BUILD/module/snvme-core.ko" "$MODULE_BUILD/module/snvme.ko"
```

这一步只生成：

```text
$MODULE_BUILD/module/snvme-core.ko
$MODULE_BUILD/module/snvme.ko
$MODULE_BUILD/bin/tutti_daemon
```

不要把 `cmake --build ... --target insmod` 当作编译步骤；该 target 会调用 `sudo` 并
改变正在运行的内核驱动状态。模块签名、加载、NVMe 接管、ext4 准备和 daemon 启动请
按 [`tutti_daemon.md`](tutti_daemon.md) 执行，并只使用确认可清空的测试盘。

## 7. 构建真实 KV workload（可选）

`layerwise_kv_overlap` 是需要 GPU、已加载模块、运行中的 daemon 和 daemon 发布的
view 目录的**手动工作负载**。它不再作为无参数 CTest 测试注册，因为程序必须接收
至少一个 `--directory`。

若需要它，在第 6 节 configure 时把：

```text
-DTUTTI_BUILD_HARDWARE_TESTS=OFF
```

改为：

```text
-DTUTTI_BUILD_HARDWARE_TESTS=ON
```

然后构建目标：

```bash
cmake --build "$MODULE_BUILD" \
  --target tutti_layerwise_kv_overlap \
  --parallel "$JOBS"
```

后续从 daemon 的 `nvmeservice_client --list-only` 或启动日志取得真实 view 目录，
再按 [`../examples/layerwise_kv_overlap/README.md`](../examples/layerwise_kv_overlap/README.md)
运行。不要猜测 `/dev/ssnvmeN`、`/dev/snvmeNn1` 或 `/mnt/gpuN/ssnvmeN` 的编号。

## 8. 常见故障与处理

| 现象 | 处理 |
| --- | --- |
| `Could not find gRPC` 或 `yaml-cpp` | 先执行第 3 节的三个 `test -f`；确认使用的是同一份 `$TUTTI_VCPKG_TOOLCHAIN`。更换 toolchain 后删除对应 build directory 再 configure。 |
| CMake 找错 CUDA | 在**首次** configure 时补充 `-DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCUDAToolkit_ROOT=/path/to/cuda`；不要在已有 cache 中切换。 |
| CUDA 版本低于 12.6 | 升级 CUDA Toolkit；CUDA profile 会在 configure 阶段拒绝。 |
| 找不到 `nv-p2p.h` | 仅阻塞 kernel module；安装匹配 NVIDIA 驱动的开发头文件，或显式设置 `SNVME_P2P_INCLUDE_DIR`。不要伪造该头文件。 |
| 找不到 `/lib/modules/$(uname -r)/build` | 安装与当前 `uname -r` 精确匹配的 kernel headers/devel 包；这不是 vcpkg 依赖。 |
| `ctest` 误触硬件测试 | 普通 CUDA build 保持 `TUTTI_BUILD_HARDWARE_TESTS=OFF`，并使用 `ctest --test-dir "$CUDA_BUILD" -LE hardware`。 |
| 想改变 GPU 架构、编译器或 toolchain | 新建 `build/manual-*` 目录或删除旧目录后重新 configure；不要复用不兼容的 CMake cache。 |

## 9. 清理和下一步

只删除本文创建的构建产物：

```bash
rm -rf "$ROOT/build/manual-host" \
       "$ROOT/build/manual-cuda" \
       "$ROOT/build/manual-cuda-module"
```

下一步按需求选择：

- 修改 API、resolver、binding 或 DataPath：从第 4 或第 5 节开始；
- 修改内核模块：第 6 节编译后阅读 [`build_and_test.md`](build_and_test.md) 的 smoke-test
  阶梯；
- 在真实盘上部署：阅读 [`tutti_daemon.md`](tutti_daemon.md)，以
  `config/local_nvme_config.yaml` 为模板；
- 运行 KV workload：阅读
  [`../examples/layerwise_kv_overlap/README.md`](../examples/layerwise_kv_overlap/README.md)。
