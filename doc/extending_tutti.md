# Extending Tutti: Adding a Resolver + Binding + DataPath

This guide walks through adding a new storage backend as a community
contributor. It uses the **memfs** sample (`tutti/bindings/memfs/`) as a
concrete example. The entire sample was added with **zero core changes** —
no `tutti/include/tutti/**` or Runtime source file was modified.

## What you need to create

| Component | Location | Purpose |
|-----------|----------|---------|
| **Binding** | `tutti/bindings/<name>/binding.h` | Payload type, identity constants, pairing helpers (`make_resolved_target` / `view_payload`) |
| **DataPath** | `tutti/bindings/<name>/<name>_data_path.h` | Implements `tutti::DataPath` SPI (open/close/register/submit/progress/query/release) |
| **Resolver** | `tutti/resolvers/<name>/resolver.h` | Implements `tutti::StorageTargetResolver` SPI (parses URI, produces `ResolvedTarget`) |
| **CMakeLists.txt** | `tutti/bindings/<name>/CMakeLists.txt` | INTERFACE library + optional test registration |
| **Contract test** | `tests/<name>_sample_contract/` | URI parsing, E2E, boundary, lifecycle |

## What you must NOT modify

- `tutti/include/tutti/**` — public/SPI headers (frozen)
- `tutti/storage_runtime.h` — Runtime implementation (frozen)
- Any existing resolver/binding/DataPath package

If you find you must change a core file, stop and record it as a gap.

## A second example: striped (multi-device, fused-kernel submission)

`tutti/data_paths/striped_local_nvme/` (+ `tutti/resolvers/striped_file/` +
`tutti/bindings/striped_local_nvme/`) is a second, more advanced community
extension: it fans a single logical `striped://name?devs=<m1,m2,...>&unit=<bytes>`
target out across N local NVMe devices with unit-granularity round-robin
striping, submitted through exactly **one** `cudaLaunchKernel` per
`rt.submit()` call (a device table of N `DeviceTargetHandle*` lets one fused
kernel dispatch entries to whichever device each stripe unit landed on).
Like memfs, it was added with **zero core changes** — callers see a plain
`TargetHandle` from `rt.open("striped://...")` and never reference a
`Striped*` type; see `tests/striped_local_nvme_contract/` (tests 87/90) for
the "zero striped-awareness at the call site" proof and the fault/partial-
commit contract. It shares the `nvme_submit_primitives.cuh` device-side
primitives with `tutti/data_paths/local_nvme/` (extracted once, unchanged)
rather than reimplementing `resolve_lba`/doorbell/CQ-poll logic. See the
package's own header comments (`striped_data_path.h`,
`resolvers/striped_file/resolver.h`, `bindings/striped_local_nvme/binding.h`)
for the full design.

## Step-by-step (memfs example)

### 1. Define the payload (`binding.h`)

The payload is **pair-private**: it lives only in your binding package. No
core header references it.

```cpp
namespace tutti::binding::memfs {

inline constexpr std::string_view kPayloadTypeId = "memfs-payload-v1";
inline constexpr std::uint32_t kPayloadApiVersion = 1;
inline constexpr std::string_view kRecommendedDataPathKey = "memfs";

class MemfsPayload { /* ... immutable, factory-created ... */ };

// Resolver packs payload + lease into ResolvedTarget:
Result<ResolvedTarget> make_resolved_target(uint64_t size,
    shared_ptr<const MemfsPayload> payload, ...);

// DataPath extracts payload with type-id + version check:
Result<const MemfsPayload*> view_payload(const ResolvedTarget& target);

} // namespace tutti::binding::memfs
```

Key points:
- `kPayloadTypeId` + `kPayloadApiVersion` are the **only** place these
  strings appear. Both `make_resolved_target` and `view_payload` use them,
  so resolver and DataPath physically cannot diverge.
- `kRecommendedDataPathKey` tells the Runtime which DataPath to use.

### 2. Implement the DataPath (`<name>_data_path.h`)

Implement `tutti::DataPath` (from `<tutti/spi/data_path.h>`):

```cpp
class MemfsDataPath : public tutti::DataPath {
    const DataPathCapabilities& capabilities() const override;
    Status initialize(const DataPathConfig&, ResourceProvider&) override;
    Status shutdown(uint64_t) override;
    Result<DataPathTarget> open(const ResolvedTarget&) override;
    Status close(DataPathTarget) override;
    Result<RegistrationDomainKey> registration_domain(DataPathTarget) const override;
    Result<DataPathMemory> register_memory(const DataPathMemoryView&, ...) override;
    Status unregister_memory(DataPathMemory) override;
    SubmitOutcome submit(const DataPathRequest*, size_t, const HostSubmitContext&) override;
    Result<ProgressResult> progress(ProgressBudget) override;
    Result<DataPathSnapshot> query(DataPathOp) const override;
    Status release(DataPathOp) override;
};
```

Use `detail::SpiIdentityMint::mint<...>(token, generation)` to mint opaque
identities for targets/memory/ops (see `tutti/testing/mock_data_path.h`
for a complete reference implementation).

### 3. Implement the resolver (`resolver.h`)

Implement `tutti::StorageTargetResolver` (from
`<tutti/spi/storage_target_resolver.h>`):

```cpp
class MemfsResolver : public StorageTargetResolver {
    Result<ResolvedTarget> resolve(string_view uri,
                                    const ResolveOptions&) override {
        // 1. Parse URI
        // 2. Create payload (backing resource)
        // 3. Return make_resolved_target(size, payload)
    }
};
```

### 4. Create CMakeLists.txt

```cmake
add_library(tutti_memfs_binding INTERFACE)
target_include_directories(tutti_memfs_binding INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)
target_link_libraries(tutti_memfs_binding INTERFACE tutti_spi)

if(BUILD_TESTING)
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/memfs_sample_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_memfs_sample_contract")
endif()
```

### 5. Register with one line

Add **one line** to `tutti/CMakeLists.txt`, inside the `if(BUILD_TESTING)`
block immediately after `include(CTest)`:

```cmake
if(BUILD_TESTING)
    include(CTest)

    add_subdirectory(bindings/memfs)   # <-- the one line
```

That's it — the library and test are now built.

**Placement matters**: `add_test()` only registers in directories processed
*after* `include(CTest)` has enabled testing. Putting the line earlier
(e.g. next to the production `add_subdirectory(bindings/...)` calls) will
build the test binary but silently leave it out of `ctest`.

### 6. Write contract tests

Create `tests/<name>_sample_contract/` with:
- URI parsing (valid + invalid)
- E2E via `StorageRuntime`: open → register → submit(WRITE) → wait →
  submit(READ) → wait → verify data → release → close → shutdown
- Boundary rejection (offset + length > size)
- Lease lifecycle (target invalid after close)

## How the Runtime wires it together

```
User: rt.open("memfs://4096", {"memfs"})
  → Runtime extracts scheme "memfs"
  → Finds MemfsResolver registered for scheme "memfs"
  → resolver.resolve("memfs://4096") → ResolvedTarget
  → ResolvedTarget.recommended_data_path_key() == "memfs"
  → Finds MemfsDataPath registered for key "memfs"
  → data_path.open(resolved_target) → DataPathTarget
  → TargetHandle returned to user
```

## Checklist

- [ ] Payload type defined only in `binding.h` (grep: no references in
      `tutti/include/tutti/**`)
- [ ] Identity constants (type id, API version, DataPath key) in one place
- [ ] DataPath implements all SPI virtuals
- [ ] Resolver parses URI and produces `ResolvedTarget` via
      `make_resolved_target`
- [ ] CMakeLists.txt defines INTERFACE library + test under BUILD_TESTING
- [ ] Exactly one `add_subdirectory` line added to `tutti/CMakeLists.txt`
- [ ] No core files modified (`git diff` shows only new files + one line)
- [ ] Contract tests pass
- [ ] Existing tests still pass (no regression)

## SNVMe driver smoke tests

本节只说明 SNVMe 驱动 smoke test。标准构建、模块产物和 daemon 启动见
[`getting-started.md`](getting-started.md)。驱动代码修改后的重编译、reload 与 baseline
诊断见 [`advanced-build.md`](advanced-build.md)。

> 文中的 PCI BDF、磁盘和挂载路径均为示例。运行 bind 或写盘测试前，必须确认目标是
> 可清空的测试 NVMe。

### 1. 前置条件

先按 [`getting-started.md`](getting-started.md) 完成默认构建并加载驱动，然后检查：

```bash
lsmod | grep snvme
ls -l /dev/snvm_control
```

### 2. 编译 smoke test

```bash
make -C tutti/device_manager/nvme/kernel_modules/test
make -C tutti/device_manager/nvme/kernel_modules/test gpu
```

测试二进制：

| 二进制 | bind | 写盘 | 用途 |
| --- | --- | --- | --- |
| `snvme_smoke` | 否 | 否 | UAPI、host memory map 与 BAR0 mmap |
| `snvme_smoke_qgroup` | 否 | 否 | queue-group 生命周期 |
| `snvme_smoke_gpu` | 否/是 | 否 | GPU P2P memory map 路径 |
| `snvme_smoke_addq` | 是 | 否 | B3 `NVM_ADD_USER_QUEUE` |
| `snvme_smoke_io` | 是 | 是 | CPU 端到端读写和逐字节校验 |
| `snvme_ubind` | — | 否 | owner-side unbind/reset helper |

按顺序运行：先安全测试，再运行 bind 测试，最后才运行写盘测试。

### 3. 选择测试设备

```bash
sudo bash scripts/pci_topology_check.sh
export TGT=0000:e3:00.0  # 替换为可清空的测试 NVMe PCI BDF
```

> `snvme_smoke_io` 会写入磁盘。确认目标盘不含重要数据后再继续。

### 4. 安全测试

不 bind 控制器，不会接管内核 `nvme` 驱动：

```bash
cd tutti/device_manager/nvme/kernel_modules/test
sudo ./snvme_smoke "$TGT"
sudo ./snvme_smoke_qgroup "$TGT"
sudo ./snvme_smoke_gpu --gpu 0 "$TGT"
```

每项成功时退出码为 `0`。

### 5. bind 与写盘测试

以下测试会接管目标控制器；`snvme_smoke_io` 会写入 LBA：

```bash
# bind + user queue，不写 LBA
sudo ./snvme_smoke_addq "$TGT"

# CPU 端到端读写校验，会写盘
sudo ./snvme_smoke_io "$TGT"

# GPU P2P 完整轮次，会 bind，但不写 LBA
sudo ./snvme_smoke_gpu --gpu 0 --rounds 4 "$TGT"
```

若出现 `no room for user queues`，将内核 IO queue cap 降低后重试：

```bash
SNVME_TEST_KERNEL_IOQ_CAP=16 sudo ./snvme_smoke_addq "$TGT"
SNVME_TEST_KERNEL_IOQ_CAP=16 sudo ./snvme_smoke_io "$TGT"
```

### 6. 清理与恢复

测试异常退出后可执行：

```bash
sudo ./snvme_ubind "$TGT"
echo "$TGT" | sudo tee /sys/bus/pci/drivers/nvme/bind
```

完整驱动 reset、直接 Kbuild 重编译和 kernel baseline 选择见
[`advanced-build.md`](advanced-build.md)。
