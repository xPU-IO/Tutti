# 多加速器 Runtime 阶段 3 接力说明

## 接力起点

- 工作分支：`fix/multi-gpu`。
- 阶段 0 提交：`807b49e`。
- 阶段 1 提交：`1eed707`（`runtime: establish accelerator identity contract`）。
- 阶段 2 提交：`61af360`（`runtime: implement multi-accelerator phase 2 device guards`）。
- 阶段 2 实施与验证记录：`doc/impl/multi-accelerator-phase-2.md`。
- 总体设计：`doc/design/multi-accelerator-runtime.md`，阶段 3 对应第 12.4 节；资源、
  RPC 和验收语义还应同时阅读第 3、7、10、13--15 节。

本 handoff 创建时的实现基线 HEAD 是 `61af360`。新会话开始后先运行
`git status --short --branch` 和 `git log -1 --oneline`；除本 handoff 可能尚未提交
外，工作树应当干净。保留本文，不要修改或重做已经完成的阶段 0/1/2。

## 已完成契约

阶段 0--2 已经固定并实现以下前置契约：

- accelerator identity 直接使用编译 backend ordinal，通用 API 统一使用
  `accel_id`/`expected_accel_id`；daemon 的 `device_id` 只表示 NVMe resource。
- 一个 `StorageRuntime` 固定一个 `accel_id`；Local/Striped DataPath 固定绑定同一
  accelerator，创建前执行 binding preflight。
- backend-neutral `DeviceGuard` 已按 PyTorch guard 的职责形状实现，但没有引入
  libtorch/C10 依赖。Runtime 和 Local/Striped DataPath 在 accelerator API 边界设置
  目标 device，并在成功、错误和清理路径恢复 caller current device。
- pointer、stream、memory、context 和 Runtime accelerator 的归属冲突会在 DMA map、
  kernel launch 或 doorbell 之前失败。
- HOST/CUDA 非硬件测试已分别通过 17/17、18/18；阶段 2 双盘实机 Local、Runtime
  Local、Striped contract 分别通过 824/824、156/156、57/57。

阶段 2 没有修改 daemon YAML、protobuf/gRPC、NVMeService client 或 queue policy。
这些仍是 legacy baseline，不能作为阶段 3 新接口的命名依据。

阶段 2 留有一个与本阶段正交的残余风险：首轮 Local hardware contract 在同一
`LocalNvmeDataPath` 双线程 `submit/query/release` 时发生过 SIGSEGV，栈指向
`progress_impl_` 遍历共享 `ops_`；行缓冲复跑完整通过。除非阶段 3 的 daemon/client
改动确实触及这条路径，不要顺手扩展为 DataPath 并发模型重构。将该风险继续记录在
阶段 3 实施报告中即可。

## 阶段 3 目标

只实施设计第 12.4 节“daemon resource 与原子 slice allocation”：

1. 将 daemon canonical YAML、parser 和内部 resource model 迁移为
   `accelerators[].accel_id/view_root`、`nvmes[].device_id`、
   `nvmes[].backing_mount_path` 和 `nvmes[].allowed_accel_ids`。
2. 保留一个版本的 legacy-only YAML 兼容读取，发出可测试的 deprecation warning；
   canonical/legacy 字段混用必须失败。
3. 新增或演进 accelerator-neutral 的 `ListAccelerators`、`ListNvmeResources`、
   `AcquireNvmeSlices` 和 `Release`，同时保留旧
   `ListDevices/Connect/Heartbeat/Disconnect` wire/source compatibility。
4. 让 libnvm owner bring-up 向 daemon 直接返回实际创建的 chrdev path/minor，并把
   ioctl `disk_name` 对应的 block path 作为 bring-up 事实向上传递。
5. 在 daemon 中实现 per-controller queue reservation ledger；单盘和 striped
   acquisition 都在一个临界区内完成校验、全量预留和 allocation 创建。
6. 让 Release、旧 Disconnect、heartbeat timeout 和 PID/starttime reaper 走同一条
   allocation 回收路径，一次归还 allocation 的全部 slice 预算。
7. 同步迁移 checked-in 配置、示例 client、测试生成器和用户文档；新 CLI 使用
   `--accel`，`--cuda` 只留给旧 client 兼容入口。
8. 证明 daemon 的 boot、list 和 acquire 都不会调用 accelerator runtime 或创建
   accelerator compute context。

## 明确非目标

阶段 3 不实施以下内容：

- 不实现阶段 4 的 allocation-driven loader、owned Runtime bundle 或 Local/Striped
  顶层组件装配，也不修改 `tutti/config` 来消费新 RPC。
- 不改变 `StorageRuntime` 的多 scheme/multi-key 核心路由能力，不修改 resolver SPI。
- 不实现跨 accelerator submit、P2P、MANAGED memory 协议、自动故障转移、NUMA
  调度、优先级或独立 `validated_available` probe worker。
- 不根据 daemon 配置创建 CUDA/MUSA/MACA context，不用 accelerator API 验证物理
  device count。daemon 只校验 `accel_id` 非负、可表示且存在于配置集合；backend
  ordinal 的实际存在性由 client/后续 loader 校验。
- 不修复与本阶段无关的 Local DataPath `ops_` 并发风险。
- 完成后不进入阶段 4。

## 冻结的 schema 与兼容规则

仓库内目标配置必须使用 canonical schema，例如本机双盘配置应迁移为：

```yaml
accelerators:
  - accel_id: 0
    view_root: "/mnt/snvme/gpu0"
  - accel_id: 1
    view_root: "/mnt/snvme/gpu1"

nvmes:
  - device_id: 0
    pci_addr: "0000:41:00.0"
    backing_mount_path: "/mnt/snvme/nvme1"
    namespace_id: 1
    kernel_ioq_cap: 32
    allowed_accel_ids: [0, 1]
    auto_mount: true
  - device_id: 1
    pci_addr: "0000:44:00.0"
    backing_mount_path: "/mnt/snvme/nvme2"
    namespace_id: 1
    kernel_ioq_cap: 32
    allowed_accel_ids: [0, 1]
    auto_mount: true
```

迁移规则必须保持确定性：

- canonical `accelerators[].accel_id`、`nvmes[].device_id` 必须非负且唯一；
  `allowed_accel_ids` 中每项必须引用已配置 accelerator，重复引用失败。
- canonical `nvmes[].device_id` 缺失或重复必须失败。NVMe 数组顺序不是身份，反转
  顺序后 list/acquire 的 resource identity 和 selection 结果不变。
- 空或缺失的 canonical `allowed_accel_ids` 继续表示全部已配置 accelerator，但进入
  内部模型和 RPC snapshot 前必须展开为确定、有序的集合。
- 一个完全 legacy 的文件可在本过渡版本中读取：`gpus[].id/mount_path`、
  `nvmes[].mount_path/allowed_gpus` 被规范化到新模型；legacy NVMe 缺少
  `device_id` 时可以暂按数组下标补齐，但必须告警。
- 任何 canonical/legacy 混用都失败，包括只在某个 NVMe entry 中混用，不能按字段
  静默择优。canonical 顶层中缺失 `device_id` 不能使用 legacy 下标补齐。
- `pci_addr` 保持 YAML 字段名，并规范化为 PCI BDF；新 RPC 使用 `pci_bdf`。
- warning 应通过 parser diagnostics 或可注入 sink 进行断言，不要让测试依赖不可控
  的 stderr 文本。可保留旧 `parse_config_file()` 作为兼容 wrapper。

checked-in 的 `config/local/daemon_2disk.yaml`、根目录配置样例、测试 fixture 和生成
脚本必须在同一阶段迁移到 canonical schema；不能因为 parser 支持 legacy 而继续
提交新的 legacy 配置。

## 冻结的 resource 与 RPC 语义

新内部 model 和新 RPC 只能用 `accel_id`/`allowed_accel_ids` 表示 accelerator。
`cuda_device`/`allowed_gpus` 只允许出现在旧 protobuf、旧 client 参数或 adapter 的
最外层；进入 allocator 前立即转换，返回旧 RPC 前再转换。不得给同一个内部字段起
两套别名。

### Selection

- `allowed`：请求不携带 `device_id`，从 ACL 允许、状态为 `available` 且队列预算
  足够的资源中，按数值 `device_id` 升序选择第一个，返回一个 slice。
- `explicit`：请求必须且只能携带一个 `device_id`，返回一个 slice；不存在、重复、
  ACL 或预算不满足都明确失败，不回退到其他盘。
- `striped`：请求按给定顺序携带两个或更多互不重复的 `device_id`，返回相同顺序的
  slice；任一资源失败时整个请求失败，不产生 allocation 或 reservation。

每次成功 acquisition 返回一个逻辑 `allocation_id`。striped 可以在内部记录多个
slice reservation，但 client 只 Release 这一个 allocation。不得把多个 slice 拆成
彼此独立的外部 lease。

### Metadata

`ListNvmeResources` 和 `NvmeSlice` 所需事实必须来自配置与本次实际 bring-up：

- daemon NVMe `device_id`；`NvmeSlice` 还携带本次请求的 `accel_id`；
- 规范化 `pci_bdf`、实际 `chrdev_path`/minor、ioctl `disk_name` 对应的
  `block_path`；
- namespace id、page size、logical block size、queue depth、DSTRD、BAR0 size 和
  max data size；
- `backing_mount_path`、请求 accelerator 对应的 `view_path`；
- 展开并排序后的 `allowed_accel_ids`；
- controller queue capacity 和当前可分配状态；`NvmeSlice` 还携带本次 granted
  queue quota；
- heartbeat interval 和 lease timeout。

`device_id` 查找必须使用显式 ID map，不能再把它当 `devices_` vector 下标。
`publish_gpu_views()`、list、acquire、legacy Connect 和日志都遵守该规则。

libnvm 的 `nvm_controller_init_b3_common()` 已在 `SNVM_CHRDEV_CREATE` 后直接取得
minor 并打开 `/dev/ssnvme<minor>`，但当前 owner API 没有把该事实返回，上层又从
`disk.disk_name` 反推 chrdev。阶段 3 应扩展 owner bring-up result/out-param，在保留
必要兼容 wrapper 的同时直接返回 chrdev minor/path 和 disk name/block path。允许从
ioctl 返回的 `disk_name` 形成 `/dev/<disk_name>`，但不得从 YAML 数组位置或 daemon
`device_id` 拼 `/dev/ssnvmeN`、`/dev/snvmeNn1`。

bring-up 后应 fail-closed 校验 BDF、chrdev 和 block device 的关联及 path 存在性。
stale minor、空 disk name、BDF/chrdev/block 不一致、mount 或目标 accelerator view
发布失败时，resource 可以带诊断出现在 list 中，但不能进入 `available` 或被 acquire。
`view_path` 必须是对应 `accel_id` 的 configured `view_root` 下实际发布的路径；后续
resolver 只能消费返回值，不能使用 `backing_mount_path` 或自行拼路径。

### Queue ledger 与回收

- 每个 controller 的 admission capacity 以 bring-up 时内核报告的 user queue 能力为
  上限，不从 YAML 或 `device_id` 推导。请求为 0 时使用 default policy；正数继续受
  daemon per-client policy 和 kernel `max_queues_per_group` 限制。
- 对一次 acquisition 先计算每个 controller 的有效 grant，再在同一个
  `state_mtx_` 临界区内校验 accelerator membership、selection、ACL、resource
  available、全部 queue budget 和 allocation ID 唯一性，最后一次性扣减并插入。
- 不允许先扣第一个 striped slice 再校验第二个。任何错误、异常或 allocation 插入
  失败都必须保持所有 controller 的 reservation 不变。
- daemon ledger 是 admission control，kernel user-QID pool 仍是实际建队列的最终
  权威；reservation 不等于 queue 已创建。client attach/add-queue 失败必须 Release。
- 新 Release、旧 Disconnect、heartbeat timeout、dead/reused PID reaper 都调用同一
  个锁内回收 helper，一次归还 allocation 记录中的全部 slice。重复 Release 必须
  选择并测试一种确定行为：幂等成功，或返回专门的 already-released/not-found 状态；
  不得二次增加 available budget。
- 旧 Connect 必须作为“单个 explicit slice”进入同一个 allocator 和 ledger，旧
  Disconnect 映射到同一个 Release。兼容入口不能绕过新 RPC 的 ACL 或预算。
- 并发 Acquire 后任一 controller 的 reserved 总数不得超过 capacity；Release/reaper
  后必须精确回到 acquire 前数值。

保留旧 proto service method 和已有字段编号，新增 method/message 使用新字段。旧
`ListDevices` 从 canonical snapshot 适配，旧 `Connect`/`Disconnect` 使用上述统一
allocator；不要保留第二套 legacy state machine。新 client CLI 的 `--accel` 走新
RPC，`--cuda` 只作为旧路径 alias；两者同时提供且冲突时明确失败。

## 主要代码落点

- schema、parser、validation：
  `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_config.{h,cpp}`。
- canonical resource、availability、allocation、reservation、reaper：
  `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_state.{h,cu}`。如将纯控制面
  实现改为 `.cpp`，需保持构建和安装目标兼容。
- protobuf 和 compatibility adapter：
  `nvmeservice.proto`、`nvmeservice_server.{h,cpp}`、
  `nvmeservice_client.{h,cpp}`。
- owner bring-up metadata：`tutti/device_manager/nvme/libnvm/include/nvm_ctrl.h`、
  `tutti/device_manager/nvme/libnvm/src/linux/device.cpp`，以及必要的 libnvm result
  类型定义。
- daemon/client CLI：
  `tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon.cpp`、
  `nvmeservice_daemon.cpp`、`nvmeservice_client.cpp` 和 client I/O bridge。
- canonical checked-in 配置：`config/local/daemon_2disk.yaml`、`sys_config.yaml` 及
  其他 `rg` 找到的 daemon YAML fixture。
- 测试与脚本：新增 daemon config/allocator/RPC contract test；同步更新
  `tests/service_client/generate_attach_config.py`、`run_attach_smoke.sh` 和 README。
- 用户文档：`doc/tutti_daemon.md`、`doc/local_nvme_contract_tests.md`、
  `tutti/device_manager/nvme/nvmeservice/NVMeService.md`。
- 阶段实施记录：新增 `doc/impl/multi-accelerator-phase-3.md`。

当前 `ServiceState` 构造函数会立即进行真实 owner bring-up，不适合作为纯状态单测
入口。应提取最小的可注入 resource snapshot/bring-up seam 或独立 allocator state，
让 schema、selection、ledger 和 reaper 测试不需要 root、GPU、NVMe 或 gRPC 端口；
不要为此建立超出阶段 3 的通用调度框架。

## 建议实施顺序

1. 先为 canonical/legacy parser 和显式 `device_id` 建立无硬件 contract test，再迁移
   config model 与 checked-in YAML。此时内部代码编译通过，但不要保留双命名 model。
2. 扩展 libnvm owner bring-up result，建立可注入 metadata 校验；将 ServiceState 从
   vector-index identity 改为显式 `device_id` lookup，并实现 resource availability。
3. 实现纯状态 selection、per-controller reservation ledger、统一 Release/reaper，
   先通过原子性和并发测试。
4. 扩展 proto/server/client，新 RPC 直接使用 canonical model；最后把旧 RPC 改成
   compatibility adapter，并验证旧字段编号和旧 client 行为。
5. 更新 CLI、生成脚本和全部用户文档，使用 `rg` 确认 legacy 名称只剩兼容边界、
   迁移说明和历史记录。
6. 运行全量无硬件回归，再执行双盘 daemon integration 和最小真实 I/O；完成清理
   后记录 `doc/impl/multi-accelerator-phase-3.md`。

每一步都保持可编译；不要先改完 proto 再留下 parser/state/client 长时间不一致。

## 无硬件验证要求

新增 CTest contract，至少覆盖：

- canonical schema 成功；legacy-only 告警并规范化；任意新旧字段混用失败。
- canonical 缺失/重复/负数 `device_id`，重复/负数/溢出 `accel_id`，重复 BDF/path，
  未知或重复 `allowed_accel_ids` 均失败。
- legacy 缺失 `device_id` 仅在完整 legacy 模式按数组顺序补齐并告警。
- `allowed`、`explicit`、`striped` 三种 selection 的数量、顺序和错误语义。
- ACL 拒绝、unknown device、重复 striped device、not-available resource、queue 预算
  不足。
- striped 任一 slice 失败或 allocation insert 注入失败后无部分 reservation。
- Release、旧 Disconnect、heartbeat timeout、dead PID/starttime-reuse reaper 全量退款；
  重复 Release 不重复退款。
- 多线程 Acquire/Release 竞争，任何时刻 reservation 不超 capacity，最终 ledger 归零。
- 反转 resource/YAML 数组顺序后，按 `device_id` 的 list、allowed selection 和
  explicit selection 不变；striped 返回请求顺序。
- stale minor、空 disk name、BDF/chrdev/block mismatch 时不进入 `available`。
- `view_path` 属于目标 accelerator 的 `view_root`；不存在的 accelerator 或 view
  发布失败不能 acquire。
- 新 gRPC metadata 字段完整且与 state snapshot 相同；一个 allocation 包含全部
  striped slice；旧 List/Connect/Heartbeat/Disconnect 仍可用且共享同一 ledger。
- daemon list/acquire path 使用 fake backend 时没有 accelerator API 调用。

随后运行既有回归：

```bash
cmake --build build/host --parallel 8
ctest --test-dir build/host --output-on-failure -j 8

cmake --build build/cuda --parallel 8
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8

cmake --preset cuda-module
cmake --build --preset cuda-module --parallel 8 --target \
  nvmeservice tutti_daemon nvmeservice_client_example
ctest --test-dir build/cuda-module -LE hardware --output-on-failure -j 8
```

新增 test target 名称确定后，将其显式加入构建命令并在实施记录中写出实际命令和
通过数量。若现有 build tree 不包含新 target，先重新 configure，不要把“未发现测试”
当成通过。

## 双盘实机验证

本机硬件事实：

| resource | ordinal/BDF | 实际阶段 2 路径 | view/mount |
| --- | --- | --- | --- |
| GPU 0 | `accel_id=0`, `00000000:4B:00.0` | - | `/mnt/snvme/gpu0` view root |
| GPU 1 | `accel_id=1`, `00000000:4C:00.0` | - | `/mnt/snvme/gpu1` view root |
| NVMe 0 | `device_id=0`, `0000:41:00.0` | `/dev/ssnvme0`, `/dev/snvme0n1` | `/mnt/snvme/nvme1` |
| NVMe 1 | `device_id=1`, `0000:44:00.0` | `/dev/ssnvme1`, `/dev/snvme1n1` | `/mnt/snvme/nvme2` |

上述 `/dev` 路径只是阶段 2 一次运行的观测值，不是阶段 3 的期望常量。每次启动都
必须从 owner bring-up/RPC 重新取得并验证。两个 namespace 当时均报告 block size
4096、BAR0 32768、`user_io_qps=96`、`max_queues_per_group=16`。

GPU 1 上有既有 Python compute processes。测试可以使用 GPU 1，但禁止终止、暂停或
干预这些进程；验证前后记录 compute-process 列表，并确认 daemon PID 从未出现。

硬件验证按以下顺序执行：

1. 用迁移后的 `config/local/daemon_2disk.yaml` 启动 daemon，先执行
   `ListAccelerators` 和 `ListNvmeResources`，核对两个显式 `device_id`、真实
   chrdev/block/BDF、mount/view、ACL、block/page/BAR0 和 queue metadata。
2. 对 accelerator 0/1 分别执行 allowed 和 explicit acquisition，attach client、创建
   queue group，并执行 client-side 最小真实 I/O 后 Release。`--skip-io` 只能用于
   attach 诊断，不能作为最终 `validated_available` 证据。
3. 执行双盘 striped acquisition，确认一个 allocation 返回两个有序 slice；对两个
   slice 完成最小真实 I/O，然后一次 Release 并确认两个 controller 同时退款。
4. 并发执行两个 acquisition：先分盘，再让两个 accelerator 共享同一个 NVMe；确认
   ledger 隔离、预算不足可复现、释放一方不影响另一方。
5. 使用短 lease 测试 client 正常断开、异常退出、heartbeat timeout 和 PID/starttime
   reaper，确认 allocation 与全部 reservation 最终归还。异常 client 只结束本测试
   启动的进程。
6. 比较 daemon 启动、list、acquire 前后的 accelerator compute-process 列表，证明
   daemon 没有创建 context。

启动时可以通过 stdin 使用 `~/.passwd/1`，禁止显示、复制到命令行参数或写入日志：

```bash
sudo -S env TUTTI_VERBOSE=1 \
  build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2disk.yaml < ~/.passwd/1
```

真实 I/O 只操作 contract test 的唯一 scratch file/区域，并记录具体路径、LBA 范围和
数据校验结果。不得仅凭 daemon `available` 宣称 `validated_available`。完整
Runtime A1--A4、B1--B3 场景依赖阶段 4 loader；阶段 3 只验证它们所需的 daemon
allocation、metadata、共享与回收前提，不提前实现 loader。

daemon 必须通过单次 SIGINT 或 SIGTERM 正常退出，禁止 SIGKILL。结束后确认：

- `/mnt/snvme/nvme1`、`/mnt/snvme/nvme2` 已卸载，GPU view 已移除；
- 本次 RPC 返回的 chrdev/block path 已消失，`:50051` 不再监听；
- `0000:41:00.0`、`0000:44:00.0` 恢复为 unbound；
- 无 allocation、queue reservation、client queue/map 或测试进程残留；
- GPU 1 上原有 compute processes 未被干预。

## 完成规则

阶段 3 完成前必须满足：

1. canonical schema、真实 bring-up path、四个新 RPC、compatibility adapter、原子
   ledger/reaper、配置/脚本/文档和对应测试全部完成。
2. 无硬件 contract 和既有 HOST/CUDA 回归全绿；双盘 RPC metadata、单盘/striped
   allocation、并发共享、回收、daemon no-context 和 client-side 最小真实 I/O 均有
   可复查结果。
3. 新增 `doc/impl/multi-accelerator-phase-3.md`，记录基线、改动、所有命令/结果、
   实际路径与 queue metadata、清理状态和残余风险。
4. 运行 `git diff --check`，确认没有生成物、日志、密码、临时 YAML 或无关改动。
5. 将阶段 3 实现、测试、canonical 配置、同步文档和实施记录作为独立 git commit
   提交；建议提交说明为 `daemon: implement multi-accelerator phase 3 allocation`。
6. 报告 commit hash、验证结果和残余风险，然后停止，不实施阶段 4。

阶段 3 的 exit gate 是：Runtime 构造未来所需的 NVMe 事实都能从一个原子 allocation
中取得；任何 daemon/client compatibility 路径都不再从 accelerator ordinal、NVMe
`device_id` 或数组下标推导数据路径，也不能绕过 ACL 和 queue reservation ledger。
