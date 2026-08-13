# `public/main` 合并实施记录（2026-08-13）

## 合并范围

本次将 `public/main` 合并到 `fix/multi-gpu`，只处理代码与编译层面的兼容，不运行
测试、示例程序、daemon，也不加载或卸载内核模块。

- 当前分支合并前提交：`349592a18a09b67fcd6f4ea02d6d59668b5e1475`。
- 上游提交：`d67a8be755d2d2f1739e54761ab26951fdd00a50`。
- 合并提交：`e684348391c984ce601b615d2a7b09cb206b369c`。
- 合并方式：`git merge --no-commit --no-ff public/main`，解决冲突并完成编译检查后
  创建 merge commit。

上游相对共同基线的主要内容包括 Linux 6.8 SNVMe 支持、daemon 不创建 CUDA
资源、NVMe 测试可移植性、MetaX/MACA 适配和 README 更新。当前分支的主要约束是
保留 canonical 多加速器配置、显式 accelerator/resource 身份、RPC allocation 和
queue reservation，以及 `nvmeservice_state.cu` 向普通 C++ 源文件的迁移。

## 合并原则

1. 通用 Runtime、NVMeService 和 libnvm 路径以当前分支已经建立的多加速器与资源
   管理契约为基础，吸收上游新增 API、内核兼容和可移植性修改。
2. 同一能力在两条分支中使用不同 API 名称时，采用上游公开名称，同时保留当前分支
   确实需要的扩展接口，不保留含义重复的两套 API。
3. daemon 保持 host-only owner 模型，不重新引入 CUDA 源编译或由 daemon 创建 GPU
   runtime 资源的行为。
4. MetaX/MACA 专属实现按要求完全采用 `public/main`，不基于本机 NVIDIA 环境推断或
   修改其实现，也不把缺少 MetaX 环境视为合并阻塞。
5. 除 Git 报告的文本冲突外，额外审查双方共同修改但被 Git 自动合并的文件，并以
   编译结果发现声明重复、接口错配和源文件迁移遗漏。

## Git 显式冲突及处理

本次共有 17 个显式冲突文件，其中 `nvmeservice_state.cu` 是 modify/delete 冲突。

### 配置、文档和 contract test

| 文件 | 冲突原因 | 合并方案 |
| --- | --- | --- |
| `config/local_nvme_config.yaml` | 上游调整 NVMe 示例配置，当前分支已迁移到 canonical accelerator/resource schema | 保留当前分支 canonical 配置和显式 ID，不恢复旧 schema |
| `doc/local_nvme_contract_tests.md` | 上游测试可移植性说明与当前分支多 GPU、双盘流程同时修改 | 合入通用可移植性说明，保留当前分支实际配置、资源 ID 和多 GPU 流程 |
| `tutti/device_manager/README.md` | daemon/NVMe 使用说明在两边均更新 | 保留当前 RPC 和资源模型，吸收上游不依赖 daemon CUDA 资源的说明 |
| `tutti/device_manager/nvme/nvmeservice/NVMeService.md` | service 生命周期和调用方式同时变化 | 文档与最终 host-only owner、RPC allocation 和 queue reservation 行为对齐 |
| `tests/striped_local_nvme_contract/striped_local_nvme_contract_test.cpp` | 上游增强测试参数可移植性，当前分支依赖多资源 RPC metadata | 保留多资源 contract，合入不硬编码环境信息的上游处理 |

### Local NVMe 数据路径

| 文件 | 冲突原因 | 合并方案 |
| --- | --- | --- |
| `tutti/data_paths/local_nvme/io/nvme_queue_group.cu` | 上游 controller 初始化 API 调整与当前分支 accelerator 绑定、queue group 逻辑重叠 | 调用上游正式 GPU owner API，同时保留当前分支 accelerator 归属和 queue reservation 契约 |

### libnvm controller 与 queue API

| 文件 | 冲突原因 | 合并方案 |
| --- | --- | --- |
| `tutti/device_manager/nvme/libnvm/include/ctrl.h` | C++ wrapper 的 controller 初始化入口和资源生命周期同时修改 | wrapper 使用 `nvm_controller_init_gpu`，保留当前分支 queue/resource 生命周期约束 |
| `tutti/device_manager/nvme/libnvm/include/nvm_ctrl.h` | 双方新增的 owner/GPU 初始化接口名称和返回信息不同 | 采用 `nvm_controller_init_owner`、`nvm_controller_init_gpu`；增加 `nvm_controller_init_owner_with_result` 承载 daemon 所需设备路径返回值 |
| `tutti/device_manager/nvme/libnvm/include/queue.h` | queue 建立方式与当前分支 reservation 逻辑同时修改 | 吸收上游 API 变化，保留已预留 queue 的数量和所有权语义 |
| `tutti/device_manager/nvme/libnvm/src/ctrl.cpp` | controller 清理路径与 owner/GPU 初始化拆分重叠 | 与最终两类初始化入口配套，保持 host-only daemon 与 GPU owner 的清理职责 |
| `tutti/device_manager/nvme/libnvm/src/linux/device.cpp` | Linux controller 初始化实现被双方大幅修改 | 共用内部实现，导出三个最终入口；daemon 使用带结果的 owner 入口，数据路径使用 GPU 入口 |

最终 API 分工如下：

- `nvm_controller_init_owner`：上游正式的 daemon owner 初始化入口。
- `nvm_controller_init_gpu`：上游正式的 standalone GPU-owner 初始化入口。
- `nvm_controller_init_owner_with_result`：保留当前分支需要的扩展，除 owner 初始化外
  返回 chrdev、block device 等 bring-up 结果，供 NVMeService 校验和发布资源。

### NVMeService 构建和状态管理

| 文件 | 冲突原因 | 合并方案 |
| --- | --- | --- |
| `tutti/device_manager/nvme/nvmeservice/CMakeLists.txt` | 上游移除 daemon CUDA 编译依赖，当前分支已迁移 state 源文件并扩展 RPC | 保持 daemon 为普通 C++ 目标，使用 `.cpp` state 实现并保留当前 RPC 源集 |
| `tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon.cpp` | daemon 启动参数和 accelerator/resource 配置同时修改 | 保留 canonical 配置解析和 host-only daemon 行为 |
| `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_config.h` | 上游配置校验声明与当前 canonical schema 声明重叠 | 合并校验接口，保持 accelerator 与 NVMe resource ID 的既有含义 |
| `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_state.h` | 上游 owner 初始化变化与当前 RPC allocation 状态同时修改 | 使用最终 owner API，保留 allocation、queue reservation、进程回收和资源发布状态 |
| `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_state.cu` | 上游修改旧 `.cu`，当前分支已删除并迁移到 `.cpp` | 保持删除；将需要的上游变化合并到 `nvmeservice_state.cpp`，避免 daemon 被作为 CUDA 源编译 |

### MetaX/MACA

| 文件 | 冲突原因 | 合并方案 |
| --- | --- | --- |
| `tutti/cmake/accelerators/MACA.cmake` | 当前分支的通用 profile 定义与上游 MetaX SDK、cu-bridge、runtime library 配置重叠 | 最终文件采用 `public/main` 版本；不在 NVIDIA 环境中推断 MetaX 构建参数 |

同时核对了 `tutti/include/tutti/cuda_like.h`、
`tutti/include/tutti/gpu_vendor/maca.h` 以及 5.4、5.15、6.8 SNVMe 的
`peer_memory/metax.c`。这些 MetaX/MACA 专属文件均保持与 `public/main` 一致。

## 自动合并文件的语义审查

Git 无冲突地完成文本合并并不保证代码语义正确。本次对双方共同修改的 56 个文件
进行了额外检查，重点检查以下交叉区域：

- 同名函数、类型、宏和 CMake target 是否被双方各新增一次。
- 声明、定义和调用点是否统一使用最终 controller API。
- `.cu` 到 `.cpp` 迁移后，CMake 源列表和 include/链接依赖是否仍引用旧文件。
- daemon host-only owner 与 Local NVMe GPU owner 是否被错误合并为同一路径。
- queue capacity、reserved queues 和 accelerator/resource ID 是否在自动合并后发生
  含义漂移。

审查发现一处实际语义冲突：

- `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_config.cpp` 中，两条分支都新增
  了 `validate_uniform_block_size`。Git 将两个定义都保留下来，首次编译报告 duplicate
  definition。解决方式是删除重复实现，只保留与当前 canonical resource 模型一致的
  单一定义，并保持头文件声明不变。

完成修复后，声明、定义与调用点均唯一；未再发现由自动合并造成的编译错误或残留
冲突标记。

## 编译与静态检查

执行了以下只涉及配置、编译和文本/语法检查的命令：

```bash
cmake --preset cuda-module
cmake --build --preset cuda-module -j16
git diff --cached --check
bash -n scripts/prepare_env.sh tests/service_client/run_attach_smoke.sh
rg -n '^(<{7} .+|={7}|>{7} .+)$' . \
  --glob '!build/**' --glob '!third_pkgs/**' --glob '!.git/**'
```

CUDA/NVIDIA 构建成功，覆盖 libnvm、NVMeService、TuttiRuntime、Local/Striped NVMe
DataPath、编译型 tests/examples，以及 Linux 5.15 NVIDIA SNVMe `modules` 目标。

Linux 6.8 SNVMe 源码已经合入，但本机没有对应 6.8 kernel headers，因此未编译该
版本模块。MetaX/MACA 按约定信任 `public/main`，没有 MetaX 环境，也未进行配置或
编译验证；这两项都不是本次合并的阻塞条件。

本次没有执行 `ctest`、任何测试或示例二进制、daemon、`insmod` 或 `rmmod`。

## 结果

- `public/main` 已成为 `fix/multi-gpu` 的祖先，合并提交保留两个父节点。
- 17 个 Git 显式冲突全部解决，无未合并文件或冲突标记。
- 自动合并审查发现并修复 1 处重复函数定义。
- 通用 CUDA/NVIDIA 代码和本机可用的 5.15 SNVMe 模块编译通过。
- MetaX/MACA 专属改动与 `public/main` 一致，留待具备相应环境的人员验证。
