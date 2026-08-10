# 多加速器 Runtime 阶段 4 接力说明

## 接力起点

- 工作分支：`fix/multi-gpu`。
- 阶段 0 提交：`807b49e`。
- 阶段 1 提交：`1eed707`（`runtime: establish accelerator identity contract`）。
- 阶段 2 提交：`61af360`（`runtime: implement multi-accelerator phase 2 device guards`）。
- 阶段 3 提交：`1cee2b7`（`daemon: implement multi-accelerator phase 3 allocation`）。
- 阶段 3 实施与验证记录：`doc/impl/multi-accelerator-phase-3.md`。
- 总体设计：`doc/design/multi-accelerator-runtime.md`，阶段 4 对应第 12.5 节；loader
  生命周期还应同时阅读第 4.4、7、10、11、12.0、13.2--13.4 节。

新会话开始后先运行 `git status --short --branch` 和 `git log -1 --oneline`。工作树
应当干净。保留本文，不要修改或重做已经完成的阶段 0--3。

## 已完成契约

阶段 0--3 已经固定并实现以下前置契约：

- accelerator identity 统一为 `accel_id`；daemon 的 `device_id` 只表示 NVMe
  resource。
- 一个 `StorageRuntime` 固定一个 `accel_id`；Local/Striped DataPath 固定绑定同一
  accelerator，创建前执行 binding preflight。
- Runtime 和 DataPath 的 accelerator API 边界使用 backend-neutral guard，设置目标
  device 并恢复 caller current device。
- pointer、stream、memory、context 和 Runtime accelerator 的归属冲突会在 DMA map、
  kernel launch 或 doorbell 之前失败。
- daemon canonical schema 为 `accelerators[].accel_id/view_root`、
  `nvmes[].device_id/backing_mount_path/allowed_accel_ids`；完整 legacy-only schema
  仍在迁移窗口内可读，但新旧字段混用失败。当前实现包含 canonical 与 legacy 兼容
  两套入口，后续可以在调用方迁移完成后考虑清理 legacy。
- daemon 提供 `ListAccelerators`、`ListNvmeResources`、`AcquireNvmeSlices`、
  `Release`，并保留旧 `ListDevices/Connect/Disconnect/Heartbeat` wire/source
  compatibility。
- daemon allocation 支持 `allowed`、`explicit`、`striped`；一次 striped allocation
  对外只有一个 `allocation_id`，内部包含多个有序 slice，并通过统一 Release/reaper
  回收全部 queue reservation。
- daemon 不创建 accelerator runtime context；`available` 表示 bring-up/mount/view/
  queue admission 可用，不等于已经完成真实 I/O 的 `validated_available`。

阶段 3 留下的明确下一步是：实现 allocation-driven loader、owned Runtime bundle 和
`tutti/config` 对新 RPC metadata 的消费。本阶段只做这一层装配，不进入阶段 5/6 的
完整实机闭环和双 Runtime 并发矩阵。

## 当前实现差距

当前 `tutti/config` loader 仍是 pre-phase-4 形态：

- `TuttiRuntime` 只 owned `StorageRuntime`、Local DataPath vector 和 Local resolver
  vector；没有 daemon client、allocation RAII、Striped 组件或显式 shutdown/Release
  顺序。
- `DeviceSpec` 仍使用 `cuda_device`、`snvme_dev`，并从 local NVMe YAML 的
  `allowed_gpus` 和数组顺序推导 `/dev/ssnvmeN`。
- `load_tutti_config()` 不连接 daemon，不调用 `ListAccelerators` 或
  `AcquireNvmeSlices`，没有使用 owner-returned `chrdev_path`、`block_path`、
  `backing_mount_path`、`view_path` 和 queue metadata。
- 多个 `DeviceSpec` 会重复注册固定的 `file` scheme 和 `local-nvme-ext4` DataPath
  key；这不是 `StorageRuntime` 核心限制，而是当前 loader 未按单组顶层组件装配。
- fallback 仍会使用 `/dev/ssnvme0` 和从 `cuda_device` 拼出的 `/dev/snvme...`。
  阶段 4 必须删除这种产品 loader 路径上的 path 推导。

## 阶段 4 目标

只实施设计第 12.5 节“allocation 驱动的单组装配 loader”：

1. 应用配置解析 `accelerator.profile`、`runtime.accel_id`、daemon endpoint、
   `nvme.selection`、`nvme.device_ids`、`nvme.queues_per_controller` 和 striped
   所需的 `stripe_unit`。
2. loader 查询编译 backend device count，确认请求的 `accel_id` 存在；HOST profile
   下不创建 accelerator Runtime，也不向 daemon 申请 NVMe slice。
3. loader 使用 `ListAccelerators` 校验 daemon 部署映射，再用
   `AcquireNvmeSlices` 获取一个逻辑 allocation。loader 不再读取 daemon YAML，不再
   从数组序号、accelerator ordinal 或 `device_id` 合成 `/dev` 路径。
4. 将 gRPC/client DTO 转换为 Tutti 内部只读 slice metadata；DataPath、resolver 和
   Runtime SPI 不直接依赖 protobuf 生成类型。
5. `allowed`/`explicit` 返回一个 slice 时，构造唯一的
   `LocalFileResolver + LocalNvmeDataPath`。
6. `striped` 返回两个或更多 slice 时，构造唯一的
   `StripedResolver + StripedDataPath`；多个 shard resolver 和 device descriptor 只
   存在于 striped 组件内部。
7. loader 对 Runtime 只注册一个 resolver scheme 和一个 DataPath key。这是产品
   loader/factory 的装配约束，不改变 `StorageRuntime` 已有的多 scheme/multi-key
   核心路由能力。
8. 修正 owned bundle 生命周期：创建失败事务性回滚；正常关闭时先 drain/close/
   unregister，调用一次 `StorageRuntime::shutdown()`，销毁 Runtime 和组件，然后
   `Release(allocation_id)`，最后销毁 daemon client。
9. 提供可注入的 daemon/resource client seam，使 loader 单测不需要 root、gRPC
   daemon、NVMe 或 GPU。

## 明确非目标

阶段 4 不实施以下内容：

- 不删除 daemon legacy YAML/RPC compatibility。可以在文档和 TODO 中标记后续清理，
  但本阶段只消费 canonical 新 RPC。
- 不扩展 `StorageRuntime` 核心路由为同 scheme 多 resolver 或动态 resolver 选择。
- 不让产品 loader 同时装配多个独立的顶层 resolver/DataPath 组合。
- 不实现双 Runtime、双 accelerator 并发 E2E、跨 accelerator submit、P2P、MANAGED
  memory 迁移、自动故障转移、NUMA 调度、probe worker 或 `validated_available`。
- 不把 daemon RPC 放进 `StorageRuntime` 公共 API；daemon allocation 生命周期属于
  loader-owned bundle。
- 不修复与本阶段无关的 Local DataPath `ops_` 并发风险，除非阶段 4 代码直接触发
  并且无法绕开。

## 冻结的 loader 配置语义

在保持既有 `tutti_config.yaml` parse-only 能力的基础上，引入最小的新配置语义。
字段命名应以 canonical 术语为准，旧 `gpu/cuda_device/allowed_gpus` 不应进入新
loader path：

```yaml
accelerator:
  profile: "cuda"              # 必须与编译 profile 一致

runtime:
  accel_id: 0                  # accelerator Runtime 的 backend ordinal

nvme_service:
  endpoint: "127.0.0.1:50051"

nvme:
  selection: "explicit"        # allowed | explicit | striped
  device_ids: [0]              # allowed 必须为空；explicit 恰好 1 个；striped 至少 2 个
  queues_per_controller: 4     # 0 使用 daemon default
  stripe_unit: 65536           # striped 使用；单盘可忽略或保留默认
```

规则：

- `runtime.accel_id >= 0` 创建 accelerator Runtime；HOST profile 的 `accel_id=-1`
  不连接 daemon。
- `accelerator.profile` 只做与编译结果的一致性检查，不触发运行时后端切换。
- `selection` 字符串必须 fail-closed；不能把未知值默认为 `allowed`。
- `allowed` 配置中出现 `device_ids` 必须失败；`explicit` 数量不是 1 必须失败；
  `striped` 数量少于 2 或重复必须失败。
- `queues_per_controller < 0` 必须失败；0 表示使用 daemon default。
- loader 必须用 `ListAccelerators` 确认 daemon 认识该 `accel_id`，并在本地用
  backend device count 确认 ordinal 存在。两者任一失败都不能执行 Acquire。
- Acquire 返回的每个 slice 的 `accel_id` 必须等于请求 Runtime 的 `accel_id`，每个
  slice 的 `allowed_accel_ids` 必须包含该 ID。
- striped 返回 slice 数量、顺序和请求 `device_ids` 不一致时必须 Release 并失败。
- 所有 path、BAR0、namespace、block size、MDTS 和 queue grant 都来自 allocation
  metadata，不从配置或 resource ID 推导。

如果需要保留旧 `local_nvme_config` 字段用于 parse-only 或兼容测试，可以保留，但
`load_tutti_config()` 的新 daemon path 不应再通过该字段派生设备。旧派生函数应被
标注为 legacy，并由测试约束不会在新配置中被调用。

## 冻结的 metadata 到组件映射

### 单 slice

一个 slice 构造一个顶层 Local 组合：

- `LocalNvmeDataPath` 使用 `slice.chrdev_path`、`slice.bar0_size`、
  `slice.accel_id`、`slice.granted_queues`、`slice.namespace_id`、
  `slice.logical_block_size` 和 `slice.max_data_size`。
- `LocalFileResolver` 使用 `slice.pci_bdf`、`slice.namespace_id`、
  `slice.logical_block_size`，以及 `slice.block_path`/`slice.backing_mount_path`
  中与现有 resolver constructor 匹配的 backing device config。不要再硬编码
  `0000:08:00.0` 或拼 `/dev/snvme...`。
- Runtime 只注册一个 scheme，例如 `file`，和一个 DataPath key，例如
  `local-nvme-ext4`。

### 多 slice

两个或更多 slice 构造一个顶层 Striped 组合：

- 为每个 slice 构造一个 shard `LocalFileResolver`，其 backing facts 来自该 slice。
- `StripedResolver` 持有 shard resolver vector 和配置的 `stripe_unit`。
- `StripedDataPath` 的 `DeviceDescriptor` vector 按 allocation slice 顺序构造；
  每项使用对应 slice 的 `chrdev_path`、`bar0_size`、`namespace_id`、
  `accel_id`、`granted_queues`、`logical_block_size`。
- 所有 slice 的 `accel_id` 必须相同且等于 Runtime `accel_id`；logical block size
  必须一致。若 daemon 已经保证一致，loader 仍应 assert/fail-closed 防止 fake 或
  旧服务返回不一致 metadata。
- Runtime 只注册一个 scheme，例如 `striped`，和一个 DataPath key，例如
  `striped-local-nvme`。

## Owned Runtime bundle 生命周期

新增或重构 `TuttiRuntime` 时不要依赖成员声明的隐式逆序析构来表达业务顺序。目标
语义应显式可测试：

```text
create:
  parse config
  construct daemon client
  ListAccelerators/ListNvmeResources preflight
  AcquireNvmeSlices
  construct resolver/DataPath objects
  StorageRuntime::create()
  return owned bundle

failure after Acquire:
  destroy partially constructed Runtime/components
  Release(allocation_id)
  stop heartbeat / destroy daemon client

shutdown/destructor:
  if runtime exists:
    runtime->shutdown()
    destroy runtime
  destroy resolver/DataPath objects
  Release(allocation_id) exactly once
  destroy daemon client
```

阶段 4 可以先只调用 `StorageRuntime::shutdown()`，不额外发明 drain API。不要在
Runtime 已经 shutdown 后再次直接 shutdown DataPath；DataPath lifecycle 由 Runtime
管理。

Release 行为必须满足：

- Acquire 未成功时不调用 Release。
- Acquire 成功但任一后续步骤失败时调用一次 Release。
- `StorageRuntime::create()` 失败时调用一次 Release。
- 正常 `shutdown()` 或 bundle 析构调用一次 Release。
- 显式 `shutdown()` 后析构不能再次 Release。
- daemon client 不可达、Acquire 被拒绝或 Release 返回 already-released 都有确定
  状态和诊断。

## 主要代码落点

- config schema、parse-only 类型、loader entry：
  `tutti/config/tutti_config.{h,cpp}`、`tutti/config/tutti_config_parse.cpp`。
- loader fake seam 和 tests：新增或扩展 `tests/config_loader/`，必要时新增
  `tests/runtime_bundle_loader_contract/`。
- daemon client DTO：`tutti/device_manager/nvme/nvmeservice/src/nvmeservice_client.h`。
  如果 `tutti/config` 不能直接依赖 nvmeservice target，需要定义接口 seam 并在
  local-NVMe build block 中装配具体实现。
- Local component mapping：
  `tutti/data_paths/local_nvme/local_nvme_data_path.h`、
  `tutti/resolvers/local_file/resolver.h`。
- Striped component mapping：
  `tutti/resolvers/striped_file/resolver.h`、
  `tutti/data_paths/striped_local_nvme/striped_data_path.h`、
  `tutti/bindings/striped_local_nvme/binding.h`。
- CMake：`tutti/config/CMakeLists.txt`、`tutti/CMakeLists.txt`、相关 tests
  `CMakeLists.txt`。注意 HOST build 不应因 nvmeservice/gRPC 不存在而失败。
- 用户文档：`doc/design/multi-accelerator-runtime.md`、
  `doc/tutti_daemon.md`、`doc/local_nvme_contract_tests.md`、
  `tutti/device_manager/nvme/nvmeservice/NVMeService.md`、`tests/service_client/README.md`。
- 阶段实施记录：新增 `doc/impl/multi-accelerator-phase-4.md`。

## 建议实施顺序

1. 先定义 loader 内部 slice/resource client seam 和 fake client contract test。此时
   不连接真实 daemon，只验证配置解析、selection 规则和 metadata 校验。
2. 扩展 `ParsedConfig`/`parse_tutti_config()`，加入新字段和 fail-closed validation。
   保留旧字段 parse-only 回归，但新 daemon path 不应使用 legacy local YAML 派生。
3. 重构 `TuttiRuntime` owned bundle，显式建模 allocation handle、daemon client、
   resolver/DataPath ownership 和 shutdown/Release 顺序。
4. 实现单 slice 映射，使用 fake metadata 构造一个 Local resolver/DataPath，并保持
   Runtime 只收到一个 resolver binding 和一个 DataPath binding。
5. 实现 striped 映射，使用 fake metadata 构造一个 Striped resolver/DataPath，同样
   只向 Runtime 注入一组顶层组件。
6. 接入真实 `NvmeServiceClient`，把 fake seam 的接口映射到
   `ListAccelerators`、`ListNvmeResources`、`AcquireNvmeSlices`、`Release`。
7. 删除 loader 产品路径上的 `/dev/ssnvmeN`、`/dev/snvmeNn1` 和 daemon YAML 数组顺序
   推导；用 `rg` 确认只剩 legacy helper、文档或历史记录。
8. 更新文档和阶段 4 实施记录，运行无硬件回归；必要时执行最小 daemon integration，
   但不要把阶段 5/6 硬件矩阵塞进本阶段。

每一步都保持可编译；不要先改 config schema 再留下 loader、tests 和 CMake 长时间不
一致。

## 无硬件验证要求

新增 contract test 至少覆盖：

- 新配置字段 parse 成功；unknown selection、负队列、profile 不匹配、无效
  `accel_id`、错误 device ID 数量均 fail-closed。
- fake `ListAccelerators` 缺少目标 `accel_id` 时不调用 Acquire。
- fake Acquire 返回 1 个 slice 时，loader 创建一个顶层 resolver 和一个顶层 DataPath。
- fake Acquire 返回 2 个 slice 时，loader 创建一个顶层 Striped resolver 和一个顶层
  Striped DataPath；slice 顺序保持。
- Runtime 核心已有 multi scheme/multi-key contract 继续通过，以证明“loader 选择一
  组”不是“核心只能一组”。
- 返回 slice 的 `accel_id`、ACL、logical block size 或数量/顺序与请求不一致时，
  loader Release 后失败。
- Acquire 成功后分别在 resolver 构造、DataPath 构造、DataPath initialize、
  `StorageRuntime::create()` 和 bundle shutdown 路径注入失败，断言 Release 次数、
  顺序和 allocation_id 正确。
- daemon 不可达或 Acquire 被拒绝时 Runtime 创建失败，不 fallback 到 `/dev/ssnvme0`。
- 显式 shutdown 后析构不重复 shutdown 或 Release。

随后运行既有回归：

```bash
cmake --build build/host --parallel 8
ctest --test-dir build/host --output-on-failure -j 8

cmake --build build/cuda --parallel 8
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8

cmake --build build/cuda-module --parallel 8
ctest --test-dir build/cuda-module -LE hardware --output-on-failure -j 8
```

新增 test target 名称确定后，将其显式加入构建命令并在实施记录中写出实际命令和
通过数量。若 build tree 不包含新 target，先重新 configure，不要把“未发现测试”
当成通过。

## 可选最小实机验证

阶段 4 的主要门槛是 fake-client loader contract。若环境允许，可在阶段末尾做最小
daemon integration，但不要扩大到阶段 5/6 的完整矩阵：

1. 启动 phase 3 已验证过的 `tutti_daemon` 和 canonical daemon YAML。
2. 用 `load_tutti_config()` 创建一个单 slice Runtime bundle，打开一个测试 target，
   注册 memory，执行最小真实 I/O，shutdown 后确认 daemon allocation 已 Release。
3. 用 `striped` selection 创建一个双 slice Runtime bundle，执行最小真实 I/O，确认
   一次 shutdown/Release 归还两个 controller reservation。
4. 记录 daemon RPC metadata、slice path、queue grant、scratch path 和清理状态。

真实 I/O 只操作 contract test 的唯一 scratch file/区域，不得对裸 namespace 做破坏性
覆盖。`--skip-io` 只能作为 attach 诊断，不能作为最终 correctness 证据。

## 完成规则

阶段 4 完成前必须满足：

1. allocation-driven loader、config schema、fake resource-client seam、Local/Striped
   单组顶层装配、owned bundle shutdown/Release 顺序和对应测试全部完成。
2. loader 不再从 daemon YAML、数组序号、`accel_id` 或 `device_id` 推导数据路径；
   所有 path 和 controller metadata 都来自 allocation slice。
3. 无硬件 contract 和既有 HOST/CUDA/CUDA-module 非硬件回归全绿；如执行实机验证，
   记录实际命令、路径、queue metadata、Release 后状态和残余风险。
4. 新增 `doc/impl/multi-accelerator-phase-4.md`，记录基线、改动、命令/结果、清理
   状态和未进入阶段 5/6 的事项。
5. 运行 `git diff --check`，确认没有生成物、日志、密码、临时 YAML 或无关改动。
6. 将阶段 4 实现、测试、文档和实施记录作为独立 git commit 提交；建议提交说明为
   `runtime: implement multi-accelerator phase 4 loader`。
7. 报告 commit hash、验证结果和残余风险，然后停止，不实施阶段 5。

阶段 4 的 exit gate 是：配置 loader 能从 daemon allocation metadata 事务性创建一个
单盘或 striped Runtime bundle，并保证创建失败与 shutdown 都准确释放 allocation；
Runtime 核心仍保持既有多 scheme/multi-key 能力，产品 loader 只选择注入一组顶层
组件。
