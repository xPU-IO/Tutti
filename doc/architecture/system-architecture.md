# GeminiFS System Architecture

> **Version**: v0.1 (Refactoring Phase)  
> **Last Updated**: 2026-04-09  
> **Status**: Active — reflects current code with v0.1 target gaps annotated

---

## 1. Overview

GeminiFS is a GPU-oriented Unified Storage Runtime that enables GPU kernels to submit NVMe I/O directly without CPU involvement in the data path. The runtime manages:

- GPU memory registration and DMA mapping
- NVMe controller lifecycle and queue management
- GPU-side file abstraction and batched I/O submission
- Persistent metadata for file and tensor mappings

The current implementation is a **monolithic library** (`libgeminifs`) being refactored into the 8-layer architecture defined in `Roadmap.md`.

---

## 2. Target Architecture

Memory Layer and Device Manager are **independent parallel layers** — memory registration has no dependency on storage topology management, and vice versa. Both feed into IO Engine, which needs registered DMA buffers (from Memory Layer) and acquired NVMe queues (from Device Manager) to submit I/O.

```mermaid
graph TD
    A["🔷 API Layer\npublic runtime APIs\n(GeminiFS class)"]
    B["🔷 Adapter Layer\nLMCache · Mooncake\n❌ Not yet implemented"]
    C["🔷 Core Runtime Layer\nobject model · lifecycle\n⚠️ Mixed into GeminiFS class"]
    D["🔷 Memory Layer\nDMA registration · PRP/SGL mapping · region model\n⚠️ Partially in GPUController"]
    E["🔷 Device Manager Layer\ndiscovery · topology · queue leasing\n⚠️ Mixed with IO engine"]
    F["🔷 IO Engine Layer\nsubmission · completion · batching\n✅ Well-isolated on device side"]
    G["🔷 Backend SPI Layer\nabstract backend interface\n❌ Not yet defined"]
    H["🔷 Backend Implementations\nlocal_nvme (partial)\nGDS · RDMA (future)"]

    A --> B
    B --> C
    C --> D
    C --> E
    D --> F
    E --> F
    F --> G
    G --> H

    style A fill:#1a3a5c,color:#fff,stroke:#4a9eff
    style B fill:#5c1a1a,color:#fff,stroke:#ff4a4a
    style C fill:#2d4a1a,color:#fff,stroke:#6aff4a
    style D fill:#2d4a1a,color:#fff,stroke:#6aff4a
    style E fill:#4a3a1a,color:#fff,stroke:#ffaa4a
    style F fill:#1a3a5c,color:#fff,stroke:#4a9eff
    style G fill:#5c1a1a,color:#fff,stroke:#ff4a4a
    style H fill:#1a5c2d,color:#fff,stroke:#4aff8a
```

---

## 3. Current Component Map (As-Is)

### 3.1 Class Hierarchy and Ownership

```mermaid
classDiagram
    class GeminiFS {
        -GPUFileManager* gpu_file_manager_
        -vector~GPUControllerPtr~ gpu_controllers_
        -vector~nvme_ctrl_param~ nvme_params_
        +init(config_path, num_files, shape, reset)
        +cleanup()
        +geminifs_batched_read()
        +geminifs_batched_write()
        +geminifs_register_tensor_with_gpu()
        +geminifs_gpu_open_file()
    }

    class GPUController {
        -int device_id_
        -string mount_base_path_
        -map~ptr, geminifs_dma*~ dma_contexts_
        -vector~NVMeControllerPtr~ nvme_controllers_
        -unique_ptr~GPUMemoryMapper~ memory_mapper_
        +registerTensorMemory(tensor, granularity)
        +unregisterTensorMemory(ptr)
        +getDMAContext(ptr) geminifs_dma*
        +addNVMeController(ctrl)
        +initialize()
    }

    class GPUMemoryMapper {
        -unordered_map~ptr, GPUHashEntry*~ host_mappings_
        -int device_id_
        +initialize()
        +addBatchMappings(ptr, size, mappings)
        +lookupMappings(ptr) GPUHashEntry*
        +removeAllMappings(ptr)
    }

    class GPUHashEntry {
        +uint64_t dev_ptr
        +uint64_t tensor_size
        +PRPMappingEntrySpan mappings
        +GPUHashEntry* next
    }

    class NVMeController {
        +ControllerPtr controller
        +unique_ptr~FileManager~ file_manager
        +string mount_path
        +uint64_t maxIOsize
        +QueueAcquireHelper* d_queue_acquire_helper
        +host_file_create_managed()
        +device_file_create_managed()
        +device_file_open_managed()
        +g_open()
    }

    class GPUFileManager {
        -string log_file_path_
        -map~GPUFileId, GPUFileDesc~ file_id_to_desc_map_
        -GPUControllerPtr gpu_controller_
        -unique_ptr~BatchIoPool~ io_ctx_pool_
        +createGPUFiles(file_size, count, shape)
        +deleteGPUFile(id)
        +getNVMeFilesSpanById(id)
        +allocateIoContexts()
        +releaseIoContexts()
        +forcePersist()
    }

    class FileManager {
        -string log_file_path_
        -map~uint32, NVMeFileDesc~ nvme_file_id_to_file_map_
        -vector~OpenFileHandle~ open_files_
        +createFile(block_size, virt_size) NVMeFileId
        +deleteFile(id)
        +getFileById(id) NVMeFileDesc
        +closeAllOpenFiles()
        +forcePersist()
    }

    class geminifs_dma {
        +uint64_t* ioaddrs
        +DmaPtr dma_ptr
        +uint64_t slice_granularity
        +vector~GranularitySliceGroup~ granularity_groups
        +DmaPtr type2_prp_dma_ptr
    }

    class GranularitySliceGroup {
        +uint64_t gpu_tensor_ptr
        +size_t granularity_offset
        +size_t granularity_size
        +vector~SubSliceInfo~ sub_slices
        +vector~PRPMappingEntry~ prp_mappings
    }

    class PRPMappingEntry {
        +uint32_t transfer_type
        +uint32_t data_length
        +uint64_t prp1
        +uint64_t prp2
        +uint64_t tensor_offset
    }

    GeminiFS "1" --> "1" GPUFileManager
    GeminiFS "1" --> "1..*" GPUController
    GPUController "1" --> "0..*" NVMeController
    GPUController "1" --> "1" GPUMemoryMapper
    GPUController "1" --> "0..*" geminifs_dma
    GPUMemoryMapper "1" --> "0..*" GPUHashEntry
    NVMeController "1" --> "1" FileManager
    GPUFileManager "1" --> "1" GPUController
    geminifs_dma "1" --> "0..*" GranularitySliceGroup
    GranularitySliceGroup "1" --> "0..*" PRPMappingEntry
```

### 3.2 Device-Side (GPU Kernel) Components

```mermaid
classDiagram
    class NVMe_File {
        +Controller* ctrl
        +geminiFS_hdr* hdr
        +QueueAcquireHelper* queue_acquire_helper
        +size_t nvme_page_size
        +read_in(prp1, prp2, offset, nbytes)
        +write_out(prp1, prp2, offset, nbytes)
        -get_nvme_offset(va) uint64_t
        -nvme_xfer(offset, prp1, prp2, nbytes, is_read)
    }

    class QueueAcquireHelper {
        +acquire_queue() int
        +issue_nvme_cmd(qp, prp1, prp2, ...)
        +issue_nvme_cmd_sgl(qp, sgl_addr, ...)
        +poll(qp, cid)
    }

    class QueuePair {
        +nvm_queue_t sq
        +nvm_queue_t cq
        +uint32_t nvmNamespace
    }

    class BatchIoEntry {
        +GPUIoContext* d_ioctxs
        +uint32_t capacity
        +uint32_t used
    }

    class GPUIoContext {
        +NVMeFilesSpan nvme_files
        +PRPMappingEntry* prp_entry
        +size_t file_offset
        +uint32_t prp_idx
    }

    NVMe_File --> QueueAcquireHelper : acquires queue
    QueueAcquireHelper --> QueuePair : submits cmd
    BatchIoEntry "1" --> "0..512" GPUIoContext
    GPUIoContext --> NVMe_File : references
    GPUIoContext --> PRPMappingEntry : references
```

---

## 4. Data Flow: Tensor Registration

```mermaid
sequenceDiagram
    participant App as Application
    participant GFS as GeminiFS
    participant GC as GPUController
    participant Mapper as GPUMemoryMapper
    participant NVMe as NVMeController (first)

    App->>GFS: geminifs_register_tensor_with_gpu(tensor, granularity)
    GFS->>GC: registerTensorMemory(tensor, granularity)
    GC->>GC: validateTensor(tensor)
    GC->>GC: createDMAContext(tensor, granularity)
    note over GC: calls libnvm to map GPU pages → DMA bus addresses
    GC->>NVMe: getDeviceDma(ctrl, gpu_ptr, size, device_id)
    NVMe-->>GC: DmaPtr (ioaddrs[])
    GC->>GC: performDMASlicing(dma_ctx, tensor_size, granularity)
    note over GC: Creates GranularitySliceGroup[] with SubSliceInfo[]
    GC->>GC: initializePRPEntries(dma_ctx)
    note over GC: Builds PRPMappingEntry per sub-slice (SINGLE/DUAL/LIST)
    GC->>GC: initializePRPList(dma_ctxs)
    note over GC: cudaMalloc PRP list pages; cudaMemcpy host→device
    GC->>Mapper: addBatchMappings(tensor_ptr, size, prp_entries, gpu_buf)
    note over Mapper: Stores GPUHashEntry with PRPMappingEntrySpan (device pointer)
    Mapper-->>GC: true
    GC-->>GFS: true
    GFS-->>App: success
```

---

## 5. Data Flow: Batched I/O Submission

```mermaid
sequenceDiagram
    participant App as Application
    participant GFS as GeminiFS
    participant GFM as GPUFileManager
    participant Mapper as GPUMemoryMapper
    participant Pool as BatchIoPool
    participant Kern as GPU Kernel

    App->>GFS: geminifs_batched_read(k_caches, file_ids, layer_idx, stream)
    GFS->>GFM: getNVMeFilesSpanById(file_id)
    GFM-->>GFS: NVMeFilesSpan (device ptr to NVMe_File*)
    GFS->>Mapper: lookupMappings(tensor_ptr)
    Mapper-->>GFS: GPUHashEntry → PRPMappingEntrySpan
    GFS->>GFM: allocateIoContexts(count)
    GFM->>Pool: acquire(count)
    Pool-->>GFM: BatchIoEntry*
    GFM-->>GFS: BatchIoEntry*
    GFS->>GFS: fill_ctx() → GPUIoContext[] per tensor per PRP entry
    GFS->>GFS: cudaMemcpyAsync(BatchIoEntry::d_ioctxs, ctx_array, stream)
    GFS->>Kern: nvme_batch_xfer_kernel<<<blocks, 32, stream>>>(entry, size, count, is_read)
    note over Kern: Each thread handles one GPUIoContext
    Kern->>Kern: ctx = d_ioctxs[tid]
    Kern->>Kern: nvme_file = ctx.nvme_files[0]
    Kern->>Kern: prp = ctx.prp_entry[ctx.prp_idx]
    Kern->>Kern: nvme_file->nvme_xfer(ctx.file_offset, prp, is_read)
    note over Kern: QueueAcquireHelper::acquire_queue() → issue_nvme_cmd() → poll()
    GFS->>GFS: cudaLaunchHostFunc(stream, release_ioctx, entry_list)
    note over GFS: async callback returns BatchIoEntry to pool after kernel completes
    GFS-->>App: cudaStream_t (async)
```

---

## 6. Initialization Flow

```mermaid
flowchart TD
    A([GeminiFS::init]) --> B[Phase 1: System Config\nauto_configure_fd_limits\ncudaGetDevice]
    B --> C[Phase 2: Config Parse\nparse_system_config\nparse_and_setup_controllers]
    C --> D[Create GPUController\nGPUMemoryMapper::initialize]
    D --> E[For each NVMe param:\nNVMeController constructor]
    E --> F[open_single_controller\nPCI bind via ioctl]
    F --> G[cudaMalloc QueueAcquireHelper\ninit_queue_acquire_helper_kernel]
    G --> H[FileManager constructor\nload or create log file]
    H --> I[gpu_controller→addNVMeController]
    I --> J[Phase 3: GPUFileManager\ninitializeLogFile\nloadFromFile\nBatchIoPool alloc 64K entries]
    J --> K{reset flag?}
    K -->|Yes| L[deleteGPUFile for all\nFileManager::deleteFile]
    K -->|No| M[Phase 4: Check existing files\nif files_to_create > 0]
    L --> M
    M --> N[createGPUFiles parallel per NVMe:\nhost_create_geminifs_file\nhost_refine_nvmeofst fiemap ioctl\ndevice_file_create_managed\ncudaMalloc NVMe_File on GPU]
    N --> O[Assemble GPUFileDesc\nwrite to log slot]
    O --> P[Phase 5: Open all GPU files\ninitGPUFile per id]
    P --> Q([Ready])

    style A fill:#1a3a5c,color:#fff
    style Q fill:#1a5c2d,color:#fff
    style F fill:#4a3a1a,color:#fff
    style G fill:#4a3a1a,color:#fff
```

---

## 7. PRP/SGL Memory Mapping Model

```mermaid
graph LR
    T[GPU Tensor\n2GB device memory]

    subgraph DMA["geminifs_dma"]
        IO[ioaddrs[]\nDMA bus addresses\nper 4KB page]
        G0[GranularitySliceGroup 0\ngranularity=256MB]
        G1[GranularitySliceGroup 1\ngranularity=256MB]
        Gn[...]
    end

    subgraph Slices["Sub-slices (per granularity)"]
        S00[SubSlice 0: offset=0, size=128KB]
        S01[SubSlice 1: offset=128KB, size=128KB]
        S0n[...]
    end

    subgraph PRP["PRPMappingEntry (per sub-slice)"]
        P1["SINGLE PAGE\ntransfer_type=0\nprp1=ioaddr[i]\nprp2=0\ndata_length≤4KB"]
        P2["DUAL PAGE\ntransfer_type=1\nprp1=ioaddr[i]\nprp2=ioaddr[i+1]\ndata_length≤8KB"]
        P3["PRP LIST\ntransfer_type=2\nprp1=ioaddr[i]\nprp2=list_page_ioaddr\ndata_length>8KB"]
    end

    subgraph PRPPage["PRP List Page (GPU memory, DMA-mapped)"]
        L1[ioaddr[i+1]]
        L2[ioaddr[i+2]]
        Ln[...]
        LT[type=PRP_TYPE_LIST]
    end

    T --> DMA
    DMA --> Slices
    G0 --> S00
    G0 --> S01
    G0 --> S0n
    S00 --> P1
    S01 --> P2
    S0n --> P3
    P3 --> PRPPage

    style T fill:#1a3a5c,color:#fff
    style PRP fill:#2d4a1a,color:#fff
    style PRPPage fill:#4a2d1a,color:#fff
```

---

## 8. Component Dependency Graph

Memory Layer and Device Manager are independent. The IO Engine depends on both: it needs DMA-mapped PRP entries (Memory Layer) and acquired NVMe queues (Device Manager) to issue I/O.

```mermaid
graph TD
    App([Application / Framework])

    subgraph API["API Layer"]
        GFS[GeminiFS]
        CFG[parse_system_config]
    end

    subgraph Runtime["Core Runtime"]
        GCR[GPUControllerRegistry\nSingleton]
    end

    subgraph MemLayer["Memory Layer (independent)"]
        GC_MEM[GPUController\nmemory side]
        GMap[GPUMemoryMapper]
        GDMA[geminifs_dma\n+ GranularitySliceGroup]
        PRP[PRPMappingEntry]
    end

    subgraph CtrlPlane["Device Manager (independent)"]
        GC_CTRL[GPUController\nstorage side]
        NC[NVMeController]
        FM[FileManager\nper NVMe]
        GFM[GPUFileManager]
    end

    subgraph DataPlane["IO Engine (Device-side)"]
        KER["nvme_*_kernel\nGPU kernels"]
        NF[NVMe_File\ndevice class]
        QH[QueueAcquireHelper\ndevice class]
    end

    subgraph Backend["Backend: local_nvme"]
        LIBNVM[libnvm\nuser-space NVMe lib]
        SNVME[snvme\nkernel module]
    end

    App --> GFS
    GFS --> CFG
    GFS --> GCR
    GFS --> GC_MEM
    GFS --> GC_CTRL
    GFS --> GFM

    GC_MEM --> GMap
    GC_MEM --> GDMA
    GDMA --> PRP
    GMap --> PRP

    GC_CTRL --> NC
    NC --> FM
    NC --> LIBNVM
    GFM --> GC_CTRL
    GFM --> FM

    GC_MEM --> KER
    GC_CTRL --> KER
    GFM --> KER
    KER --> NF
    NF --> QH
    QH --> LIBNVM
    LIBNVM --> SNVME

    style API fill:#1a3a5c,color:#fff,stroke:#4a9eff
    style Runtime fill:#2d2d1a,color:#fff,stroke:#aaaa4a
    style MemLayer fill:#1a4a2d,color:#fff,stroke:#4aff9a
    style CtrlPlane fill:#2d1a4a,color:#fff,stroke:#9a4aff
    style DataPlane fill:#1a2d4a,color:#fff,stroke:#4a7aff
    style Backend fill:#3a1a1a,color:#fff,stroke:#ff4a4a
```

---

## 9. Refactoring Gap Analysis

```mermaid
graph LR
    subgraph Current["Current State"]
        direction TB
        C1[GeminiFS\nmonolithic orchestrator\n⚠️ init混合所有阶段]
        C2[GPUController\nMemory + Storage混合\n⚠️ 应拆分为两个类]
        C3[NVMeController\nControl + Data混合\n⚠️ device文件操作应独立]
        C4[GPUFileManager\nFile管理 + IO池\n⚠️ BatchIoPool应独立]
        C5[No Backend SPI\n❌ 直接调用libnvm]
        C6[No Adapter Layer\n❌ 无LMCache/Mooncake]
        C7[Split Config\n⚠️ YAML vs INI未统一]
    end

    subgraph Target["v0.1 Target"]
        direction TB
        T1[API Layer\n稳定公共接口]
        T2[MemorySubsystem\n独立分配/注册/Region模型]
        T3[ControlPlane\n专注拓扑/发现/租约]
        T4[DataPlane\n纯提交/完成语义]
        T5[BackendSPI\nBackendProvider接口]
        T6[local_nvme Backend\nPRP/SGL实现]
        T7[Adapters\nLMCache · Mooncake]
        T8[Unified Config\n单一YAML格式]
    end

    C2 -.->|Phase 2: Extract| T2
    C1 -.->|Phase 1: Define interfaces| T1
    C3 -.->|Phase 3: Split| T3
    C3 -.->|Phase 3: Split| T4
    C5 -.->|Phase 4: Define SPI| T5
    C5 -.->|Phase 5: Implement| T6
    C6 -.->|Phase 6: Build| T7
    C7 -.->|Phase 1 blocker| T8
```

---

## 10. Known Constraints and Non-Goals (v0.1)

```mermaid
mindmap
  root((v0.1 Constraints))
    NO Cooperative Submit
      CPU and GPU must not access same queue simultaneously
      Queue-sharing capability ≠ runtime ownership semantics
      Future co-submit needs explicit version decision
    Hardware Requirements
      Linux 5.15.0 only
      NVIDIA H100+ (sm_90)
      CUDA 12.6+
      BIOS: Above4G=ON, IOMMU=OFF
    Kernel Module
      snvme must be installed at system startup
      Not ad-hoc runtime build
      Formal support matrix required
    Config
      YAML (NVMeService) vs INI (libgeminifs)
      Must unify before modularization
    Naming
      Do not hard-code "GeminiFS" in new abstractions
      Runtime name may change in future versions
    GPU File Persistence
      NOT stable in v0.1
      Must not be treated as reliable persistence contract
```

---

## 11. Directory Structure (Current → Target)

```
Current (Historical Boundaries)          Target (v0.1 Direction)
────────────────────────────────         ────────────────────────────────
GeminiFS/                                GeminiFS/
├── backends/local/                      ├── api/                    ← New
│   ├── nvme/libnvm/        ✅           ├── runtime/               ← New
│   ├── kernel_modules/snvme-5.15.0/ ✅   ├── memory/                ← Extract from libgeminifs
│   └── NVMeService/        ✅           ├── device_manager/         ← New
├── filesystems/ext4/                    ├── io_engine/            ← New
│   └── libgeminifs/        ⚠️ Monolith  ├── backends/
│       ├── include/                     │   └── local_nvme/        ← Move from backends/local
│       ├── gpu_controller.cu            ├── adapters/              ← New (LMCache, Mooncake)
│       ├── nvme_controller.cu           ├── doc/architecture/      ← This file
│       ├── gpu_file_manager.cu          ├── examples/
│       ├── geminifs.cu                  └── ...
│       └── ...
├── examples/               ✅
├── doc/
│   └── architecture/       ← This file
└── ...
```

---

*Generated from codebase analysis. Update this document when interfaces change.*
