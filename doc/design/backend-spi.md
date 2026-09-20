# Backend SPI — the DataPath / Resolver / Binding contracts

> The normative contract for plugging a new storage backend into Tutti.
> For a step-by-step contributor walkthrough see
> [../extending_tutti.md](../extending_tutti.md); this document is the
> semantic contract those implementations are held to.

## 1. Design rules

1. **The public API never names an implementation.** `StorageRuntime`
   deals in opaque `TargetHandle` / `MemoryHandle` / `IOHandle`. Concrete
   `LocalNvmeDataPath` / `StripedDataPath` types are private to
   `tutti/data_paths/`.
2. **SPI implementations never see public request internals beyond the
   contract types** in `tutti/include/tutti/spi/` + `io_types.h` /
   `memory_types.h` / `status.h`.
3. **Adding a backend requires zero core changes.** New resolver +
   binding + DataPath packages plug in without touching
   `tutti/include/tutti/**` or the Runtime (proven twice: the memfs
   sample and the striped multi-device DataPath).
4. **Single-threaded access per instance.** The Runtime serializes all
   SPI entry points (`registry_mutex_`, `datapath_open_mutex_`);
   implementations may keep plain maps/vectors without internal locking.

## 2. The three SPI surfaces

### 2.1 `tutti::StorageTargetResolver` (`spi/storage_target_resolver.h`)

```cpp
Result<ResolvedTarget> resolve(std::string_view uri, const ResolveOptions&);
```

Parses a URI, walks whatever host-side metadata the storage needs
(FIEMAP for local files), and returns an immutable `ResolvedTarget`:
logical size + pair-private payload + lease. Resolution is host-side IO
intensive and therefore parallelized by `open_batch()` — resolvers must
be safe for concurrent `resolve()` calls (read-only shared state).

Fail-closed: nonexistent files, non-regular files, malformed URIs, and
path-component attacks (`.`/`..`/NUL) are rejected at resolve time with
`INVALID_ARGUMENT` / `NOT_FOUND`, never silently sanitized.

### 2.2 `tutti::DataPath` (`spi/data_path.h`)

The full IO engine contract:

```cpp
const DataPathCapabilities& capabilities() const;
Status initialize(const DataPathConfig&, ResourceProvider&);
Status shutdown(uint64_t timeout_ms);
Result<DataPathTarget>  open(const ResolvedTarget&);
Status                  close(DataPathTarget);
Result<RegistrationDomainKey> registration_domain(DataPathTarget) const;
Result<DataPathMemory>  register_memory(const DataPathMemoryView&, ...);
Status                  unregister_memory(DataPathMemory);
SubmitOutcome           submit(const DataPathRequest*, size_t,
                               const HostSubmitContext&);
Result<ProgressResult>  progress(ProgressBudget);
Result<DataPathSnapshot> query(DataPathOp) const;
Status                  release(DataPathOp);
```

Key semantics every implementation must honor:

- **Partial-commit submit.** `SubmitOutcome.io` carrying a value does
  *not* imply all requests were accepted. Rejected requests (e.g.
  backpressure from the in-flight quota) are reported in
  `initial_states`; they were never executed. Callers must walk
  `initial_states` and resubmit rejections — skipping this loses data
  silently (`wait()` returns normally for accepted requests only).
- **In-flight protection.** `unregister_memory` must refuse (or defer)
  while ops on that memory are in flight — the DataPath tracks accepted
  ops' memory tokens for exactly this check.
- **Opaque identities.** Target/memory/op identities are minted via
  `detail::SpiIdentityMint::mint<T>(token, generation)`; generations
  make stale-handle use detectable instead of aliasing a recycled slot.

### 2.3 Bindings — the pair-private payload contract

A binding package (`tutti/bindings/<name>/binding.h`) is the *only* place
where a resolver's output format and a DataPath's input expectation meet:

- `kPayloadTypeId` + `kPayloadApiVersion` + `kRecommendedDataPathKey`
  are each defined exactly once; both packing (`make_resolved_target`)
  and unpacking (`view_payload`, with type-id + version check) go through
  the binding, so the pair cannot drift apart.
- The Runtime routes purely on these constants: scheme → resolver,
  `recommended_data_path_key` → DataPath.

## 3. Runtime routing & grouping

```text
rt.open(uri)
  → scheme(uri) → resolver.resolve(uri)          (parallel under open_batch)
  → payload.recommended_data_path_key → DataPath.open(resolved)
  → TargetHandle                                   (opaque, generation-tagged)

rt.submit(reqs, n)
  → group consecutive requests by DataPath
  → one DataPath::submit() per group             (one kernel launch)
  → SubmitOutcome{io, initial_states[]}          (per-request status)
```

## 4. Capability & capacity knobs

`DataPathCapabilities` advertises alignment requirements
(`memory_alignment_bytes`, `target_alignment_bytes`), execution mode
(`DEVICE_EXECUTION`), and queue/depth facts. Capacity knobs (handle cache
L1/L2 sizes, PRP cache capacity, in-flight quota, batch entry caps) are
constructor-injected today and slated to move into a user-facing config
file (programmatic override > config file > defaults).

## 5. What a backend gets for free

By implementing the contracts above, a backend inherits: the stable
public API, `open_batch` parallelism, submit grouping, per-request
fail-closed status, opaque identity handling, and the contract test
kit (`tutti/testing/mock_data_path.h` is a complete reference
implementation used by the hardware-independent suites).

## 6. Optional paged-memory path

`DataPath::register_paged_memory()` and `submit_paged()` are an explicit
optional extension for a backend that can resolve a block table in the
device-facing IO path. They are deliberately separate from
`register_memory()` / `submit()` and from `supports_direct`: direct
contiguous device IO does not imply paged-cache support.

A backend must set `DataPathCapabilities::supports_paged_memory` and override
all paged hooks before the extension is selectable. The base `DataPath`
implementation returns `UNSUPPORTED`, rejects every paged request, and mints
no operation. Existing staged and contiguous backends therefore retain their
current behavior.

`DataPathPagedMemoryView` describes per-layer physical block pools without
exposing an NVMe descriptor or file-format field. `DataPathPagedRequest`
contains a copied logical block-id vector, layer index, token interval, and
ordinary target byte range. Implementations must copy the block-id metadata
before returning from an asynchronous submit; callers are not required to
keep that vector alive until completion. No implementation may advertise
paged support while internally flattening the cache through a staging buffer:
that remains a valid staged fallback, but it must leave the capability false
and return `UNSUPPORTED` from the paged hooks.
