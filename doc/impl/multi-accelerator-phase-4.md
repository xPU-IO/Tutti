# 多加速器 Runtime 阶段 4 实施记录

## 基线与范围

- 分支：`fix/multi-gpu`。
- 实施起点：`7e5fe4f`（`docs: add multi-accelerator phase 4 handoff`），阶段 3
  基线为 `1cee2b7`。
- 阶段 4 实施提交：`a493603`（`runtime: implement phase 4 allocation loader`）。
- 本记录只覆盖阶段 4 的 allocation-driven loader、owned Runtime bundle、daemon
  resource-client seam、单组顶层 Local/Striped 装配和 loader 级验证；没有进入阶段 5
  的真实 `load_tutti_config()` I/O 闭环，也没有实现双 Runtime 并发矩阵。
- 阶段 0--3 的 daemon canonical/legacy compatibility 保持不变；本阶段只让产品
  loader 消费 canonical 新 RPC metadata。

## 实施内容

- `tutti/config` 新增 canonical 应用配置语义：
  `accelerator.profile`、`runtime.accel_id`、`nvme_service.endpoint`、
  `nvme.selection`、`nvme.device_ids`、`nvme.queues_per_controller` 和
  `nvme.stripe_unit`。`selection` fail-closed，不再把未知值默认为 `allowed`。
- `load_tutti_config()` 不再读取 daemon YAML，也不再从 legacy `local_nvme_config`、
  NVMe 数组顺序、accelerator ordinal 或 `device_id` 推导 `/dev/ssnvmeN` /
  `/dev/snvme...`。legacy `derive_local_nvme_devices()` 仅保留为 parse-only/兼容测试
  helper，并在头文件中标注不会被新 loader path 使用。
- 新增 loader 内部只读 DTO：
  `RuntimeAcceleratorInfo`、`RuntimeNvmeResource`、`RuntimeNvmeSlice` 和
  `RuntimeNvmeAllocation`。DataPath、resolver 和 Runtime SPI 不依赖 protobuf 生成
  类型。
- 新增 `RuntimeResourceClient` seam，生产实现包装现有 `NvmeServiceClient` 的
  `ListAccelerators`、`ListNvmeResources`、`AcquireNvmeSlices` 和 `Release`；测试
  通过 fake client 注入，不需要 root、gRPC daemon、NVMe 或 GPU。
- loader 先校验 `accelerator.profile` 与编译 profile 一致，再用 backend device
  count 校验 `runtime.accel_id` 存在；HOST profile 下 `accel_id=-1` 不创建 daemon
  client、不执行 Acquire，只创建 host-only `StorageRuntime`。
- accelerator Runtime path 中，loader 先用 `ListAccelerators` 确认 daemon 认识目标
  `accel_id`，再用 `ListNvmeResources` 校验显式请求资源的 ACL/availability，随后
  执行 `AcquireNvmeSlices`。
- Acquire 返回后，loader 对每个 slice 做 fail-closed 校验：slice `accel_id` 必须
  等于 Runtime `accel_id`，`allowed_accel_ids` 必须包含该 ID，`chrdev_path`、
  `block_path`、`pci_bdf`、namespace、logical block size 和 BAR0 metadata 必须由
  allocation 提供。
- 单 slice 的 `allowed/explicit` allocation 构造唯一顶层
  `LocalFileResolver + LocalNvmeDataPath`，Runtime 只注册 `file` scheme 和
  `local-nvme-ext4` DataPath key。
- 两个及以上 slice 的 `striped` allocation 构造唯一顶层
  `StripedResolver + StripedDataPath`，shard `LocalFileResolver` 和
  `DeviceDescriptor` vector 仅存在于 striped 组件内部。loader 校验返回 slice 数量、
  顺序、`accel_id` 和 logical block size 一致性。
- `TuttiRuntime` 改为显式 owned bundle：共同持有 `StorageRuntime`、顶层
  DataPath/resolver、resource client、allocation ID 和 slice metadata；`shutdown()`
  显式调用一次 `StorageRuntime::shutdown()`，销毁 Runtime/组件后调用一次
  `Release(allocation_id)`，析构只做兜底且不会二次 Release。
- `config/tutti_config.yaml` 示例迁移到 canonical loader 语义；保留
  `local_nvme_config` 注释为 legacy parse-only compatibility key。
- CMake 中 `tutti_config` 链接 Local/Striped DataPath 与 resolver；有 `nvmeservice`
  target 时启用真实 gRPC resource client。Local/Striped DataPath 静态库启用
  `CUDA_RESOLVE_DEVICE_SYMBOLS`，使 loader 测试可链接拥有 CUDA RDC object 的
  DataPath 库。

## 接口与行为变化摘要

### 接口变化

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 应用配置 accelerator | 旧 loader 主要解析 `gpu.vendor`，产品路径使用 `cuda_device` 语义 | 新 loader 解析 `accelerator.profile` 与 `runtime.accel_id`，并按编译 profile/设备数量校验 |
| daemon endpoint | loader 不连接 daemon | 新增 `nvme_service.endpoint`，accelerator Runtime path 通过 resource client 连接 daemon |
| NVMe 选择 | 从 `local_nvme_config` 和 legacy YAML 推导设备 map | 新增 `nvme.selection/device_ids/queues_per_controller/stripe_unit`，只驱动 `AcquireNvmeSlices` |
| loader resource seam | 无可注入 seam，测试需要绕开完整 loader | 新增 `RuntimeResourceClient`、backend device count 和 runtime factory 注入点 |
| allocation metadata | loader 不消费新 RPC slice metadata | 新增内部 DTO，并把 `chrdev_path`、`block_path`、`pci_bdf`、queue grant、BAR0、namespace、MDTS 映射到组件 |
| owned bundle | 只 owned Runtime、Local DataPath vector 和 Local resolver vector | owned Runtime bundle 同时 owned resource client、allocation ID、slice metadata、Local/Striped 顶层组件，并提供显式 `shutdown()` |
| Runtime bindings | 多个 `DeviceSpec` 会重复注册 `file` 和 `local-nvme-ext4` | 产品 loader 始终只注册一个 resolver scheme 和一个 DataPath key；多 shard 只在 striped 组件内部展开 |

### 行为变化

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| path 来源 | fallback 到 `/dev/ssnvme0`，或由 CUDA ordinal 拼 `/dev/snvme...` | 所有 path 来自 daemon allocation metadata；loader 不再合成产品路径 |
| `allowed` 配置 | 旧配置没有明确 fail-closed 规则 | `allowed` 出现 `device_ids` 直接失败 |
| `explicit` 配置 | 可由旧 device map 推导多个 top-level 组件 | 必须且只能配置一个 `device_id`，Acquire 必须返回一个 slice |
| `striped` 配置 | loader 不支持 striped allocation 装配 | 必须配置至少两个唯一 `device_id`，Acquire 返回 slice 数量和顺序必须与请求一致 |
| daemon preflight | loader 不执行 `ListAccelerators`/`ListNvmeResources` | Acquire 前必须确认 daemon 认识 `accel_id`，显式资源 ACL 包含该 accelerator 且可用 |
| HOST profile | 旧 loader 仍可能构造 local NVMe 默认路径 | `runtime.accel_id=-1` 时不连接 daemon、不申请 NVMe slice，只创建 host-only Runtime |
| 创建失败回滚 | Acquire 后失败没有 loader-owned Release 语义 | Acquire 成功后任一校验、组件构造或 Runtime create 失败，bundle 析构路径 Release 一次 |
| shutdown | 依赖成员逆序析构表达业务顺序 | 显式 shutdown 顺序：Runtime shutdown/reset，组件销毁，Release allocation，销毁 client |

## 文档同步

除本文外，本阶段触及的文档及其作用如下：

| 文档 | 修改描述 |
| --- | --- |
| `config/tutti_config.yaml` | 示例配置迁移到 `accelerator/runtime/nvme_service/nvme` canonical loader path，删除旧 `/dev` fallback 说明 |
| `doc/impl/multi-accelerator-phase-4-handoff.md` | 阶段 4 输入说明，未修改；作为本实施记录的来源 |
| `doc/design/multi-accelerator-runtime.md` | 阶段 4 继续遵守第 4.4、7、8、10、11、12.5、13.2--13.4 节约束，未在本阶段改写 |

## 无硬件验证

实际命令与结果：

```text
cmake --preset cuda
  configure OK; gRPC/nvmeservice enabled

cmake --build --preset cuda --parallel 8 --target tutti_config_loader_test
build/cuda/bin/tutti_config_loader_test
  RESULT: PASS (72 checks)

cmake --build --preset cuda --parallel 8 --target \
  tutti_config_loader_test \
  tutti_storage_runtime_contract_test \
  nvmeservice_phase3_contract_test
ctest --test-dir build/cuda --output-on-failure -R \
  'tutti_config_loader_test|tutti_storage_runtime_contract_test|nvmeservice_phase3_contract'
  3/3 passed
```

`tutti_config_loader_test` 覆盖：

- canonical 字段 parse、priority chain、RDMA/bad YAML/missing file fail-closed。
- `allowed` 带 device IDs、`explicit` 零个/多个 IDs、`striped` 少于两个 IDs、重复
  IDs、未知 selection 均 fail-closed。
- fake client 单 slice 返回时，loader 只发布一个顶层 `file` resolver 和一个
  `local-nvme-ext4` DataPath key。
- fake client 双 slice 返回时，loader 只发布一个顶层 `striped` resolver 和一个
  `striped-local-nvme` DataPath key。
- striped 返回 slice 顺序不一致、slice `accel_id` 不一致、Runtime create 失败都会
  Release 一次。
- daemon accelerator ordinal 不存在时不执行 Acquire；Acquire 自身失败时不执行
  Release。
- 显式 `shutdown()` 后析构不会重复 Release。

`tutti_storage_runtime_contract_test` 继续覆盖核心 Runtime 多 scheme/multi-key 能力，
证明阶段 4 的“单组顶层装配”只是产品 loader 约束，不改变 `StorageRuntime` 核心路由
能力。

## 双盘 daemon 验证

按阶段 4 handoff 要求，用 `config/local/daemon_2disk.yaml` 启动 daemon：

```text
sudo -S env TUTTI_VERBOSE=1 \
  build/cuda/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2disk.yaml < ~/.passwd/1
```

daemon owner/RPC 事实：

| device_id | pci_bdf | chrdev | block | backing mount | capacity | block/page | queue depth | BAR0 |
|---:|---|---|---|---|---:|---|---:|---:|
| 0 | `0000:41:00.0` | `/dev/ssnvme0` | `/dev/snvme0n1` | `/mnt/snvme/nvme1` | 96 | 4096/4096 | 1024 | 32768 |
| 1 | `0000:44:00.0` | `/dev/ssnvme1` | `/dev/snvme1n1` | `/mnt/snvme/nvme2` | 96 | 4096/4096 | 1024 | 32768 |

资源发现：

```text
build/cuda/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --list-only
  ListAccelerators: accel_id 0/1
  ListNvmeResources: device_id 0/1 available=true, reserved=0, available_queues=96
```

非破坏性 allocation smoke：

```text
build/cuda/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --accel 0 \
  --selection explicit --device 0 --count 4 --skip-io
  PASS: one slice, attach/create/destroy/free path OK

build/cuda/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --accel 0 \
  --selection striped --device 0 --device 1 --count 4 --skip-io
  PASS: one allocation, two ordered slices, attach/create/destroy/free path OK
```

本验证只确认 daemon resource discovery、allocation metadata、attach/create/destroy 和
Release 路径可用；按阶段边界没有执行真实 file resolver + DataPath I/O 闭环。daemon
Ctrl-C 后 clean exit，并卸载 `/mnt/snvme/nvme1` 和 `/mnt/snvme/nvme2`。

## 残余风险

- 阶段 5 仍需实现并验证单 Runtime 的真实 `load_tutti_config()` 单盘与 striped I/O
  闭环，包括通过 loader 打开真实 target、注册内存、submit/wait、shutdown 和 Release。
- 阶段 4 fake-client 测试不初始化真实 Local/Striped DataPath；构造 metadata 映射已
  覆盖，但真实 resolver/DataPath 与 allocation metadata 的端到端配合留给阶段 5。
- 当前生产 `GrpcRuntimeResourceClient` 通过持有 `NvmeServiceClient::Allocation`
  RAII handle 间接 Release；`RuntimeResourceClient::release()` 返回 OK 表示本地
  release path 已执行，但底层 RPC 的 already-released/错误细节仍由现有 client 打印到
  stderr，后续可考虑把 Release 结果结构化上抛。
- `StorageRuntime::shutdown(0)` 仍是阶段 4 bundle 使用的唯一 drain/close/unregister
  入口；没有新增专用 drain API。
- 阶段 2/3 已记录的 Local DataPath `ops_` 并发风险未在本阶段修复；阶段 4 loader
  不直接触发该路径。
