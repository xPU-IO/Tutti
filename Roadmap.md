# Tutti Unified Storage Runtime Roadmap

## Status

- Current active version: `v0.1`
- This document describes the `v0.1` architecture baseline and the active roadmap.
- Future version number changes are maintainer-driven and must not be advanced automatically.
- Historical roadmap snapshots are archived under [`doc/history/`](doc/history/).
- Every version roadmap must preserve:
  - a `Feature Snapshot`
  - a `Known Bugs Snapshot`

## v0.1 Positioning

`Tutti` (Italian for "all instruments together") is a `CPU/GPU companion
storage software stack`: a unified storage runtime in which the CPU and GPU
paths cooperate on top of a shared memory subsystem and a pluggable backend
SPI. `v0.1` positions the codebase as a `Unified Storage Runtime` rather than
only a GPU file abstraction.

Naming note:

- `Tutti` is the project name used by the architecture, public APIs, and the
  `tutti::` C++ namespace.
- New code, headers, and documents should use the `Tutti` name.

The runtime is intended to provide:

- Stable upper-layer APIs for systems such as `LMCache` and `Mooncake`
- Both `CPU-side` and `GPU-side` read/write paths
- A standalone memory subsystem for host/device allocation and registration
- A backend SPI that can evolve toward `Local NVMe`, `GDS`, `RDMA`, and vendor-specific backends
- Clear separation between `device manager`, `IO engine`, and `memory model`

`v0.1` explicitly does **not** define `cooperative submit`.

Current assumption:

- CPU and GPU do not normally access the same data region at the same time
- Submission mode is either `CPU_SUBMIT` or `GPU_SUBMIT`
- Any future CPU/GPU hardware-cooperative path will be introduced only after an explicit version decision

## v0.1 Feature Snapshot

The `v0.1` version baseline is intended to include the following core features:

- reposition the project as a `Unified Storage Runtime`
- define stable runtime-facing abstractions instead of exposing file-system-only internals
- provide both `CPU-side` and `GPU-side` read/write paths
- support exactly two submission modes:
  - `CPU_SUBMIT`
  - `GPU_SUBMIT`
- treat memory management as a first-class subsystem with host/device allocation and registration semantics
- define a backend SPI suitable for future `Local NVMe`, `GDS`, and `RDMA` evolution
- treat the modified NVMe Linux kernel module as part of the `local_nvme` backend baseline
- support both direct GPU-owned bootstrap and `NVMeService`-owned bootstrap
- keep upper-layer adaptation for systems such as `LMCache` and `Mooncake` outside the core runtime

Note:

- this snapshot records the `v0.1` feature baseline and roadmap intent
- it does not imply that every item is already fully implemented in the current codebase

## v0.1 Known Bugs Snapshot

Known bugs and unstable areas currently tracked for `v0.1`:

- `GPU file persistence` has known correctness/stability issues and must not yet be treated as a stable persistence contract

## Current Repository Problems

The current codebase does not yet match the `v0.1` target architecture. Main issues:

- Initialization is over-coupled: controller discovery, GPU setup, NVMe setup, file management, and runtime bootstrap are mixed together
- Configuration is split across incompatible formats and parsing paths
- Public interfaces leak backend details such as controller internals, CUDA details, and file-layout assumptions
- Memory registration, mapping, and allocation are spread across storage-facing classes instead of a dedicated subsystem
- The current file-oriented model is too narrow for upper-layer cache/block/object runtimes
- The modified NVMe kernel module lifecycle is not yet treated as a formal deployment and compatibility concern

## v0.1 Target Architecture

### Layering

`v0.1` uses the following logical layers:

1. `API Layer`
   - Stable runtime API exposed to applications and adapters
   - Provides CPU read/write, GPU read/write, batch IO, and lifecycle management

2. `Adapter Layer`
   - Integration adapters for frameworks such as `LMCache` and `Mooncake`
   - Converts framework-specific concepts into runtime-neutral requests

3. `Core Runtime Layer`
   - In-process request lifecycle: accept `BatchRequest`, lower to SPI batches, drain completions
   - Owns the runtime-visible *noun* types: `Device`, `Lease`, `IOBuffer`, `BatchRequest`, `StorageTarget`, `CapabilitySet`
   - Must not depend on a concrete backend implementation
   - Must not contain cross-process daemon logic (that belongs in the Device Manager Layer)

4. `Memory Layer` *(independent — parallel to Device Manager)*
   - Manages host/device allocation, registration, deregistration, and region metadata
   - Owns the memory model used by both the IO engine and upper-layer integrations
   - Has no dependency on the Device Manager

5. `Device Manager Layer` *(independent — parallel to Memory Layer)*
   - Cross-process device fleet management
   - Owns the *service* interfaces that produce runtime nouns: `IDeviceRegistry` (produces `Device`), `ILeaseManager` (produces `Lease`)
   - Device discovery, topology, capability advertisement
   - Queue/resource lease lifecycle (issue, heartbeat, release, reap)
   - Process attach metadata and daemon/client wire protocols
   - Has no dependency on the Memory Layer

6. `Filesystem Layer` *(independent — parallel to Backend Implementations)*
   - Resolves namespaces (file paths, object keys, DFS handles) into `StorageTarget` values that backends can consume
   - Implementations: ext4 + FIEMAP, custom on-device layout, distributed FS local clients (3FS / JuiceFS / DAOS / ...), object-store clients
   - Independent of the data-path backend it composes with — the only contract is `StorageTarget`
   - Has no dependency on a specific backend implementation

7. `IO Engine Layer` *(depends on Memory Layer and Device Manager)*
   - Read/write submission
   - mapping and buffer preparation
   - completion handling
   - batch execution
   - CPU_SUBMIT and GPU_SUBMIT execution paths

8. `Backend SPI Layer`
   - Formal backend extension interface (`IBackendProvider`, `IQueueProvider`)
   - Supports pluggable backends without changing upper-layer APIs

9. `Backend Implementations`
   - Data-path implementations grouped by transport, not by filesystem
   - `local_nvme` as the first reference backend (libnvm + snvme kernel module + NVMeService daemon)
   - Future candidates: `local_rdma`, `gds`, and hybrid backends
   - Each backend composes orthogonally with any compatible filesystem from the Filesystem Layer

### Kernel Module Baseline

The current `backend/kernel_modules` area contains a modified Linux NVMe kernel module lineage used to support CPU-side and GPU-side access to NVMe queues.

This must be treated as a formal architecture dependency rather than an implementation detail.

Rules for `v0.1`:

- the kernel module is part of the `local_nvme` backend baseline
- queue-level CPU/GPU simultaneous access capability is a backend/driver capability, not a public API promise of cooperative submit
- runtime semantics must still assume explicit ownership and clear submission mode boundaries
- kernel-facing logic must be isolated enough that future Linux version support can evolve without rewriting upper-layer APIs

### Driver Lifecycle and Bootstrap Model

The system must support two runtime bootstrap paths:

- `GPU-owned bootstrap`
  - a GPU process initializes the data path directly when deployment chooses a process-local ownership model

- `Service-owned bootstrap`
  - `NVMeService` initializes and manages shared controller/queue resources
  - GPU processes attach later without owning low-level initialization

`v0.1` should treat these as deployment modes over the same backend/device-manager model, not as separate architectures.

### Deployment and Compatibility Constraints

For the modified NVMe kernel module, the roadmap must account for:

- Linux version compatibility strategy
  - the module will need an explicit support matrix for targeted kernel versions
  - internal adaptation points should be isolated for kernel API drift

- installation timing
  - the module is expected to be installed before runtime use
  - preferred deployment model is system startup installation rather than ad hoc runtime build/load

- operational packaging
  - deployment should consider package-based install, DKMS-style rebuild strategy, or another explicit lifecycle model
  - startup integration should consider `systemd`, boot-time module loading, and service ordering

- runtime prerequisites
  - service startup, GPU attach flow, permissions, device nodes, and module readiness must be checkable before data-path use

- failure handling
  - deployment design must define what happens when the module is missing, kernel ABI is incompatible, or service bootstrap fails

These constraints are part of architecture planning because they directly affect portability, operability, and contributor usability.

### Runtime Object Model

`v0.1` should converge on the following core objects:

- `Runtime`
- `RuntimeConfig`
- `Device`
- `StorageTarget`
- `MemoryRegion`
- `IOBuffer`
- `IORequest`
- `BatchRequest`
- `Completion`
- `Lease`
- `CapabilitySet`

Rules:

- Upper layers must depend on these abstract objects instead of controller/file implementation details
- Backend implementations may extend internals, but not the public object model
- File is treated as one storage object form, not the only storage abstraction

### Submission Model

`v0.1` supports exactly two submission modes:

- `CPU_SUBMIT`
  - CPU prepares and submits IO
  - GPU may consume or produce the data buffer, but submission ownership remains on CPU

- `GPU_SUBMIT`
  - GPU prepares and initiates the IO path defined by the backend
  - CPU may assist with control or completion plumbing, but not as a cooperative execution model

Non-goals in `v0.1`:

- No `COOPERATIVE_SUBMIT`
- No requirement that CPU and GPU simultaneously operate on the same logical region
- No implicit concurrency semantics beyond explicit API contracts

Note:

- backend driver support for CPU/GPU queue access does not by itself justify exposing a cooperative runtime submission model
- queue-sharing capability and runtime ownership semantics must remain distinct concepts

### Memory Model

The memory subsystem is a first-class part of `v0.1`.

Supported memory categories:

- `HOST`
- `PINNED_HOST`
- `DEVICE`
- `MANAGED`
- `EXTERNAL`

Memory operations must be separated by semantics:

- Allocation
  - `allocate_host`
  - `allocate_pinned_host`
  - `allocate_device`
  - `free`

- Registration
  - `register_host`
  - `register_device`
  - `unregister`

- Query
  - `query_region`
  - `query_capabilities`

- Exchange
  - reserved for future import/export and cross-process sharing

Every `MemoryRegion` should describe at least:

- address
- size
- alignment
- location
- ownership
- registration state
- access capabilities
- backend-visible attributes

### API Direction for Upper Layers

`v0.1` runtime APIs should serve both general applications and cache/object systems.

Required API classes:

- Runtime lifecycle API
- Capability and topology query API
- CPU read/write API
- GPU read/write API
- Batch IO API
- Memory allocation and registration API
- Queue/resource lease API

API constraints:

- Do not expose backend-private types in the public API
- Do not bind public APIs to a specific file-layout implementation
- Do not hardcode `LMCache` or `Mooncake` structures into the core runtime
- Framework-specific adaptation belongs in dedicated adapters

## v0.1 Recommended Directory Direction

This is a design target, not a completed repository state.

The layout separates **two extension axes** so they can grow
independently:

- *transport / data-path* (how bytes physically move) lives under
  `backends/`
- *namespace / metadata* (how a name is resolved to an address) lives
  under `filesystems/`

The two meet only through the `StorageTarget` value type defined in
`runtime/`. Adding one new transport does not require touching any
filesystem code, and vice versa — replacing an n × m combinatorial
explosion with an n + m matrix.

```text
Tutti/
├── api/                # public runtime API the application links against
├── runtime/            # in-process request lifecycle + runtime-visible nouns
├── memory/             # IMemorySubsystem, MemoryRegion, registration
├── device_manager/     # cross-process device fleet management & leases
├── io_engine/          # IBackendProvider SPI + submission/completion/batching
├── filesystems/        # namespace -> StorageTarget (FS / object / DFS client)
│   ├── include/        # IFilesystem / INamespaceResolver SPI
│   ├── ext4_fiemap/    # ext4 + FIEMAP -> (file_id, LBA range)
│   ├── tutti_layout/   # custom on-device GPU-file layout (legacy libgeminifs)
│   ├── dfs_client/     # distributed FS local clients (3FS, JuiceFS, DAOS, ...)
│   └── object_store/   # S3-shape namespaces (future)
├── backends/           # data-path backends -- implement IBackendProvider
│   ├── include/        # cross-backend helpers (BufferDescriptor builders, ...)
│   ├── local_nvme/     # libnvm + snvme kernel module + NVMeService daemon
│   ├── local_rdma/     # ibverbs + RDMA QP pool + (future) RDMAService daemon
│   └── gds/            # NVIDIA GDS adapter
├── adapters/           # LMCache, Mooncake, and other framework integrations
└── doc/
    ├── architecture/   # architecture descriptions
    ├── design/         # design contracts (backend SPI, ...)
    ├── rfcs/           # design RFCs
    ├── ai/             # AI-facing subsystem docs
    └── history/        # archived roadmap snapshots
```

### Layer Responsibility Split

The boundary between `runtime/` and `device_manager/` is a frequent
source of confusion; v0.1 fixes it with two rules.

**Rule 1: process scope.**

- `device_manager/` owns everything that crosses processes — daemons,
  wire protocols, leases, PID-based reaper logic, device discovery
  that has to query a service.
- `runtime/` owns everything that stays inside one process — accepting
  a `BatchRequest`, lowering it to SPI batches, dispatching through
  `IBackendProvider`, draining `IOCompletion` to `ICompletionSink`.

**Rule 2: noun vs service.**

- *Nouns* (the value types users hold and pass around) live in
  `runtime/`: `Device`, `Lease`, `IOBuffer`, `BatchRequest`,
  `StorageTarget`, `CapabilitySet`.
- *Services* (the lifecycle interfaces those nouns are produced by)
  live in `device_manager/`: `IDeviceRegistry` (produces `Device`),
  `ILeaseManager` (produces `Lease`).

Backends *register* into device_manager (publishing devices and
implementing the lease/registry services) and *implement* the
io_engine SPI (`IBackendProvider`). The runtime never talks to
backends directly — it goes through device_manager for fleet info and
through the SPI for IO.

### Filesystem vs Backend Composition

A `Device` registered into the runtime carries both a backend
provider (data-path) and an associated filesystem resolver
(namespace). They are paired at config time, not built in. This makes
new combinations cheap:

| Use case | filesystems/ | backends/ |
|---|---|---|
| Local NVMe + on-device file layout | `tutti_layout/` | `local_nvme/` |
| Local NVMe + ext4 files | `ext4_fiemap/` | `local_nvme/` |
| Local RDMA + distributed FS client | `dfs_client/<x>/` | `local_rdma/` |
| GDS + ext4 files | `ext4_fiemap/` | `gds/` |
| Local NVMe raw (no FS) | none (passthrough `BLOCK_RANGE`) | `local_nvme/` |

The filesystem layer's only job is to produce a `StorageTarget`; the
backend's only job is to consume one. Neither layer includes the
other's private headers.

## Active Roadmap

### Phase 0: Freeze the Architectural Baseline

Goals:

- Define `Unified Storage Runtime` as the official product direction
- Stop extending the old monolithic initialization path
- Freeze core concepts, naming, and boundaries before moving directories

Deliverables:

- `v0.1` architecture document in this roadmap
- stable terminology for runtime, memory, device manager, IO engine, and backend SPI
- explicit rejection of `cooperative submit` in this version
- naming transition requirement recorded so future APIs are not forced to retain the `GeminiFS` label

### Phase 1: Define Stable Core Interfaces

Goals:

- Define the public runtime API surface before rewriting internals
- Remove direct exposure of controller/file-system internals from future public headers

Deliverables:

- public object model
- request/response model
- error model
- lifecycle model
- capability query model

### Phase 2: Extract the Memory Subsystem

Goals:

- Separate memory ownership and memory registration from storage logic
- Make host and device memory first-class runtime resources

Deliverables:

- `MemoryRegion` model
- host/device allocation APIs
- host/device registration APIs
- clear ownership and teardown semantics

### Phase 3: Split Device Manager and IO Engine

Goals:

- Move discovery, topology, leases, and shared-resource metadata into the device manager
- Keep read/write execution in the IO engine
- Separate service-owned bootstrap and process-owned bootstrap from upper-layer APIs

Deliverables:

- device manager responsibilities and interfaces
- IO engine responsibilities and interfaces
- runtime bootstrap mode definition
- attach/init boundary for GPU process and `NVMeService`
- explicit attach path between device manager and IO engine

### Phase 4: Introduce Backend SPI

Goals:

- Make backend replacement and extension possible without rewriting upper layers
- Ensure future `RDMA` and `GDS` work is additive rather than invasive

Design contract: [`doc/design/backend-spi.md`](doc/design/backend-spi.md)

Deliverables:

- `IBackendProvider` interface (`prepare_descriptors`, `acquire_queue`, `release_queue`, `launch_io_kernel`)
- `BufferDescriptor` tagged union (NVMe + RDMA placeholder)
- `BackendRegistry` wiring
- `local_nvme` refactored to implement `IBackendProvider`

### Phase 5: Land the First Reference Backend

Goals:

- Provide one complete backend that validates the architecture
- Use `local_nvme` as the reference implementation

Deliverables:

- reference local backend
- CPU_SUBMIT path
- GPU_SUBMIT path
- capability reporting
- explicit kernel-module dependency contract
- bootstrap support for direct init and service-managed init

### Phase 6: Build Upper-Layer Adapters

Goals:

- Make the runtime consumable by external application stacks
- Keep framework-specific logic outside the core runtime

Deliverables:

- `LMCache` adapter plan
- `Mooncake` adapter plan
- adapter boundary rules

### Phase 7: Documentation and Governance

Goals:

- Make the architecture maintainable by both humans and AI contributors
- Ensure later backends can be implemented independently

Deliverables:

- architecture docs
- AI-facing subsystem docs
- RFC templates and review rules
- backend extension guidance
- deployment guide for module install, boot ordering, and service startup
- kernel compatibility policy for supported Linux versions

## Versioning Rules

- `Roadmap.md` is always the active roadmap for the current version selected by the maintainer
- Version changes are not made automatically
- Every active and archived version roadmap must retain a per-version `Feature Snapshot` and `Known Bugs Snapshot`
- When a new version is opened, the previous active roadmap snapshot should be copied into [`doc/history/`](doc/history/)
- Archive file naming should follow:
  - `roadmap-v0.1.md`
  - `roadmap-v0.2.md`
  - `roadmap-v1.0.md`

## Out of Scope for v0.1

- cooperative CPU/GPU submit model
- simultaneous CPU/GPU access optimization for the same logical region
- committing to a single remote transport design before the backend SPI is stabilized
- binding core runtime APIs directly to one framework's internal data structures
