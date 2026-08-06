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
