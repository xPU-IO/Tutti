#pragma once

// tutti/data_paths/local_nvme/local_nvme_data_path.h
//
// First DataPath: LocalNvmeDataPath.
//
// Moves the existing NVMe backend's target-lifecycle structure onto the
// frozen DataPath SPI.  lifecycle, capabilities, target open/close,
// registration_domain, memory registration, and IO submission
// (submit/progress/query/release) have real behaviour.
//
// Memory registration uses libnvm's nvm_dma_map_data_device /
// nvm_dma_map_data_host, ported from the client-attach + DMA map
// usage in nvmeservice_client_io.cu.  The controller is attached
// in client-only mode (nvm_ctrl_attach_client), which opens
// /dev/ssnvme<N> and mmaps BAR0 without touching bind/chrdev ioctls.
//
// Deferred (subsequent tasks):
//   - Device Manager / vdevice roster integration

#include <tutti/status.h>
#include <tutti/io_types.h>
#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/bindings/ext4_local_nvme/binding.h>

#include "tutti/data_paths/local_nvme/metadata/metadata_arena.h"
#include "tutti/data_paths/local_nvme/metadata/handle_workspace_cache.h"
#include "tutti/data_paths/local_nvme/metadata/prp_page_cache.h"
#include "tutti/data_paths/local_nvme/metadata/desc_pool.h"
#include "tutti/data_paths/local_nvme/metadata/prp_buf_pool.h"
#include "tutti/data_paths/local_nvme/io/prp_builder.h"  // AddressDescriptor (Round 16 S6 test accessor)

#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tutti::data_paths::local_nvme {

// Forward declarations for private io/ types.
class NvmeQueueGroup;
struct DeviceTargetHandle;
struct DeviceSubmitEntry;
struct EntryCompletionStatus;

// -------------------------------------------------------------------------
// LbaExtent -- block-unit extent (converted from binding's byte-unit Extent)
//
// Mirrors the old NvmeFileDeviceHandle's LbaExtent semantics:
//   start_lba      = device_offset / block_size
//   length_blocks  = length / block_size
// -------------------------------------------------------------------------
struct LbaExtent {
    std::uint64_t start_lba      = 0;
    std::uint64_t length_blocks  = 0;
    std::uint64_t logical_offset_bytes = 0;  // preserved from payload for
                                             // future file-relative mapping
};

// -------------------------------------------------------------------------
// LocalNvmeTargetState -- host-side target state
//
// In the production backend this corresponds to the data stored in
// NvmeFileDeviceHandle, but on the host side and without CUDA or libnvm
// dependencies.  The future device-resident handle allocation
// (cudaMalloc + extent H2D + d_qps) is deferred to the IO submission task.
// -------------------------------------------------------------------------
struct LocalNvmeTargetState {
    binding::ext4_local_nvme::NamespaceIdentity ns;
    std::uint64_t file_size_bytes = 0;
    std::uint32_t block_size_log  = 0;  // log2(block_size) for fast conversion
    std::vector<LbaExtent> lba_extents;

    // Identity bookkeeping (matches the minted DataPathTarget)
    std::uint64_t token      = 0;
    std::uint64_t generation = 0;

    // Device-resident handle (GPU memory). nullptr if not yet built
    // (skeleton mode or open() before queue group exists).
    DeviceTargetHandle* dev_handle = nullptr;
    void* dev_overflow = nullptr;  // overflow extents GPU buffer (or nullptr)

    // Handle cache entry (non-null when handle is cache-owned).
    // When non-null, dev_handle/dev_overflow are borrowed from the cache
    // and must NOT be freed by close() — the cache owns their lifetime.
    HandleWorkspaceCache::Entry* cache_entry = nullptr;
    std::uint64_t cache_key = 0;
};

// -------------------------------------------------------------------------
// LocalNvmeDataPath -- concrete DataPath for local NVMe targets
// -------------------------------------------------------------------------
class LocalNvmeDataPath : public DataPath {
public:
    // Skeleton constructor (Round 7): no controller, no queue group.
    LocalNvmeDataPath(std::string snvme_dev_path = "",
                      std::uint32_t bar0_size = 0);

    // Production constructor: includes queue group parameters.
    // snvme_dev_path: e.g. "/dev/ssnvme0"
    // bar0_size: BAR0 size in bytes
    // cuda_device: primary GPU for d_qps + per-queue rings
    // num_user_queues: count of user QPs to create
    // (SQ/CQ ring depth is NOT a parameter: it is fixed by the kernel
    //  module's io_queue_depth and obtained via NVM_GET_DEV_INFO at
    //  bring-up — the kernel sizes user IOQ rings with dev->q_depth
    //  unconditionally, so a userspace-specified value can only desync)
    // namespace_id: NVMe namespace ID
    // block_size: NVMe block size (typ. 4096)
    // mdts_bytes: maximum data transfer size per NVMe command (0 = default 128KiB)
    // max_batch_entries: max sub-IO entries per op (0 = default 256)
    // cq_poll_budget: max CQ poll iterations before timeout (0 = default 10000000)
    // handle_cache_capacity: handle cache slots (0 = disabled)
    // prp_cache_capacity: PRP cache slots (0 = disabled)
    // threads_per_block: submit kernel block size (1..1024, default 16).
    //                    Must not exceed the queue group's actual queue count.
    // max_in_flight_operations: cap on concurrent IN_FLIGHT ops (0 = default 16)
    // max_batch_requests: cap on input request count per submit() call
    //                     (0 = default: follow max_batch_entries, i.e. the
    //                     pre-Round-15-S4 behavior where both caps were the
    //                     same knob)
    // max_request_bytes_override: cap on bytes per single request, i.e.
    //                     caps_.max_single_io_bytes/max_batch_bytes (0 =
    //                     default: computed in initialize() as
    //                     max_batch_entries * effective_mdts, the
    //                     pre-Round-15-S4 formula)
    //
    // Memory account (Round 15 S4): the MetadataArena pre-allocates
    //   arena_slots = 2 * max_in_flight_operations
    //   bytes/slot  = max_batch_entries *
    //                 (sizeof(DeviceSubmitEntry) [48B]
    //                  + sizeof(EntryCompletionStatus) [8B]
    //                  + page_size [PRP-list pool, typ. 4096B])
    //   total       = arena_slots * bytes/slot
    // Defaults (16, 256): 32 slots * 256 * 4152B ~= 32.4 MiB.
    // Example (in-flight=8, batch_entries=4096): 16 slots * 4096 * 4152B
    //   ~= 259.5 MiB. Callers requesting large batch_entries must budget
    //   GPU memory accordingly.
    LocalNvmeDataPath(std::string snvme_dev_path,
                      std::uint32_t bar0_size,
                      std::uint32_t cuda_device,
                      std::uint32_t num_user_queues,
                      std::uint32_t namespace_id,
                      std::uint32_t block_size,
                      std::uint64_t mdts_bytes = 0,
                      std::uint32_t max_batch_entries = 0,
                      std::uint32_t cq_poll_budget = 0,
                      std::uint32_t handle_cache_capacity = 0,
                      std::uint32_t prp_cache_capacity = 0,
                      std::uint64_t max_in_flight_operations = 0,
                      std::uint64_t max_batch_requests = 0,
                      std::uint64_t max_request_bytes_override = 0,
                      // Round 16 S6b: L2 (host-pinned content) tier for the
                      // handle cache.  0 = default 4×L1 when L1 enabled.
                      std::uint32_t handle_cache_l2_capacity = 0,
                      std::string controller_pci_addr = {},
                      std::uint32_t threads_per_block = 16);

    ~LocalNvmeDataPath() override;

    // ---- identity / capabilities ----
    const DataPathCapabilities& capabilities() const override;

    // ---- lifecycle ----
    Status initialize(const DataPathConfig& config,
                      ResourceProvider& resources) override;
    Status shutdown(std::uint64_t timeout_ns) override;

    // ---- target lifecycle ----
    Result<DataPathTarget> open(const ResolvedTarget& target) override;
    Status close(DataPathTarget target) override;
    Result<RegistrationDomainKey> registration_domain(
        DataPathTarget target) const override;

    // ---- memory registration ----
    Result<DataPathMemory> register_memory(
        const DataPathMemoryView& view,
        const RegistrationDomainKey& domain) override;
    Status unregister_memory(DataPathMemory memory) override;

    // ---- submit / progress / query / release ----
    SubmitOutcome submit(const DataPathRequest* requests,
                         std::size_t count,
                         const HostSubmitContext& ctx) override;
    Result<ProgressResult> progress(ProgressBudget budget) override;
    Result<DataPathSnapshot> query(DataPathOp op) const override;
    Status release(DataPathOp op) override;

    // ---- test-only accessor ----
    // Returns nullptr if the target identity is invalid or not found.
    const LocalNvmeTargetState* test_target_state(
        DataPathTarget target) const;

    // ---- test-only: get the raw nvm_dma_t* for a registered memory ----
    // Returns nullptr if the memory identity is invalid or not found.
    // The returned pointer borrows from the registration table; caller
    // must ensure the DataPath outlives the pointer.
    const nvm_dma_t* test_dma_handle(DataPathMemory memory) const;

    // ---- test-only: queue group accessors ----
    // Returns the queue group's group_id (0 if not created).
    std::uint32_t test_queue_group_id() const;
    // Returns the queue group's d_qps (nullptr if not created).
    const void* test_d_qps() const;
    // Returns the queue group's n_qps (0 if not created).
    std::uint32_t test_n_qps() const;
    // Returns the device target handle GPU pointer for a target (nullptr if none).
    const void* test_dev_handle(DataPathTarget target) const;

    // ---- test-only: MDTS / capacity / capability accessors ----
    std::uint64_t test_hardware_mdts() const;
    std::uint64_t test_effective_mdts() const;
    std::uint64_t test_prp_list_page_capacity() const;
    std::uint32_t test_in_flight_count() const;
    std::uint32_t test_threads_per_block() const {
        return threads_per_block_;
    }
    // Returns true if the op's arena lease is still held (not yet released).
    bool test_op_has_resources(DataPathOp op) const;

    // ---- test-only: submit failure injection seams ----
    // Simulates a pre-launch failure after resource reservation, proving
    // op=nullopt + zero-issued when no kernel was issued.
    void test_set_inject_launch_failure(bool v);
    // Simulates cudaEventRecord failure after a successful kernel launch,
    // proving the issued IO remains observable through a terminal op.
    void test_set_inject_event_record_failure(bool v);

    // ---- test-only: per-op entry / PRP-list observability ----
    // These accessors let a private contract test inspect the fan-out entries
    // and PRP-list DMA of a submitted op without touching the DataPath private
    // map. They D2H-copy from the op's device entry array; the test never sees
    // a raw device pointer. Only in local_nvme_data_path.h (private SPI),
    // never in tutti/include/tutti/**.

    // Number of device entries fan-out produced for this op (0 if unknown).
    std::uint32_t test_entry_count(DataPathOp op) const;

    // D2H-copy entry[index] of the op's device entry array into `out`.
    // Returns false if op not found, index out of range, or D2H failed.
    bool test_copy_entry(DataPathOp op, std::uint32_t index,
                        DeviceSubmitEntry& out) const;
    // Round 16 S6 (REQUIRED 0): copy a single entry's AddressDescriptor
    // (prp1/prp2/data_length) from GPU to host for test observability.
    // The entry itself no longer carries these fields inline.
    bool test_copy_entry_desc(DataPathOp op, std::uint32_t index,
                              AddressDescriptor& out) const;

    // True if this op owns a PRP-list DMA (i.e. at least one LIST sub-IO).
    bool test_op_has_prp_list_dma(DataPathOp op) const;

    // DMA IOVA of PRP-list page `list_idx` for this op, or 0 if the op has no
    // PRP-list DMA or list_idx is out of range. A LIST entry's prp2 must equal
    // this value (it is a DMA IOVA, never a CUDA virtual pointer).
    std::uint64_t test_prp_list_ioaddr(DataPathOp op,
                                       std::uint32_t list_idx) const;

    // Number of PRP-list pages (ioaddrs) this op owns, or 0 if none.
    std::uint32_t test_prp_list_page_count(DataPathOp op) const;

    // ---- test-only: per-entry completion status observability ----
    // D2H-copies the per-entry completion status array for this op.
    // Returns false if op not found, status array is null, or D2H failed.
    // The out vector is resized to entry_count.
    bool test_copy_completion_status(
        DataPathOp op,
        std::vector<std::uint32_t>& out_results) const;

    // Returns the CQ poll budget configured for this DataPath.
    std::uint32_t test_cq_poll_budget() const;

    // ---- test-only: submit()/kernel-launch call counters (Round 15 S4) ----
    // test_submit_call_count_ increments once per submit() invocation
    // (regardless of outcome). test_kernel_launch_count_ increments once
    // per successful cudaLaunchKernel issued from submit() (i.e. excludes
    // calls that were rejected before reaching the launch step, and the
    // injected-launch-failure path). Together these let a contract test
    // prove "one rt->submit -> one DataPath::submit -> one kernel launch"
    // for a large multi-target batch.
    std::uint64_t test_submit_call_count() const;
    std::uint64_t test_kernel_launch_count() const;
    void test_reset_submit_counters();

    // ---- test-only: completion error injection seams ----
    // When set, the device kernel's resolve_lba is forced to fail for
    // entries whose target_offset matches the injection offset.
    // This proves resolve_lba failure -> FAILED, not silent success.
    void test_set_inject_resolve_lba_failure(bool v);
    bool test_get_inject_resolve_lba_failure() const;

    // ---- test-only: NVMe CQ error injection seam (FIX 2) ----
    // When set, the device kernel synthesizes an NVMe CQ error (dword3
    // SCT/SC bit) on otherwise-normal completion, proving result=3 -> FAILED.
    void test_set_inject_nvme_error(bool v);
    bool test_get_inject_nvme_error() const;

    // ---- test-only: progress() query-error injection seam (FIX 3) ----
    // When set, progress() forces cudaEventQuery/cudaStreamQuery to behave as
    // a persistent CUDA error (not NotReady), proving op -> FAILED instead of
    // stuck IN_FLIGHT.  Runtime S2 wait/shutdown can then observe the terminal.
    void test_set_inject_query_error(bool v);
    bool test_get_inject_query_error() const;

    // ---- test-only: has_timeout observability (FIX 4) ----
    // True if any per-entry result==2 (CQ timeout) was observed for this op.
    // Such ops retain their PRP-list DMA mapping after release() (conservative
    // leak) because the timed-out command may still be in the controller queue.
    bool test_op_has_timeout(DataPathOp op) const;

    // ---- test-only: MetadataArena accessors ----
    // Returns the arena's total slot capacity.
    std::uint32_t test_arena_capacity() const;
    // Returns the arena's currently available (free) slot count.
    std::uint32_t test_arena_available() const;
    // Returns the arena's allocation counters (test seam for zero-alloc proof).
    MetadataArena::AllocCounts test_arena_alloc_counts() const;
    // Resets the arena's allocation counters (call after init, before submit cycles).
    void test_arena_reset_alloc_counts();

    // ---- test-only: HandleWorkspaceCache accessors ----
    bool test_handle_cache_enabled() const;
    HandleWorkspaceCache::Stats test_handle_cache_stats() const;

    // ---- test-only: PrpPageCache accessors ----
    bool test_prp_cache_enabled() const;
    PrpPageCache::Stats test_prp_cache_stats() const;

private:
    // Public SPI entry points are thin device-guarded wrappers.  The impl
    // methods deliberately contain the existing resource/error paths without
    // duplicating current-device plumbing at every early return.
    Status initialize_impl_(const DataPathConfig& config,
                            ResourceProvider& resources);
    Status shutdown_impl_(std::uint64_t timeout_ns);
    Result<DataPathTarget> open_impl_(const ResolvedTarget& target);
    Status close_impl_(DataPathTarget target);
    Result<RegistrationDomainKey> registration_domain_impl_(
        DataPathTarget target) const;
    Result<DataPathMemory> register_memory_impl_(
        const DataPathMemoryView& view,
        const RegistrationDomainKey& domain);
    Status unregister_memory_impl_(DataPathMemory memory);
    SubmitOutcome submit_impl_(const DataPathRequest* requests,
                               std::size_t count,
                               const HostSubmitContext& ctx);
    Result<ProgressResult> progress_impl_(ProgressBudget budget);
    Result<DataPathSnapshot> query_impl_(DataPathOp op) const;
    Status release_impl_(DataPathOp op);

    DataPathCapabilities caps_{};
    bool initialized_ = false;
    std::uint64_t next_token_ = 1;

    // token -> target state.  Removal from this map on close() fully
    // invalidates the identity: any subsequent lookup by the old
    // token+generation fails.  This avoids the P0-8 dangling-pointer
    // issue in the old backend (where close freed the handle but left
    // a stale cache entry).
    std::unordered_map<std::uint64_t, LocalNvmeTargetState> targets_;

    // Look up a target by token+generation.  Returns nullptr if not found
    // or generation mismatch (e.g. already closed).
    const LocalNvmeTargetState* find_(DataPathTarget target) const;

    // Look up a memory registration by token+generation.
    struct MemReg {
        nvm_dma_t* dma = nullptr;
        void* base = nullptr;
        std::uint64_t size_bytes = 0;
        DataPathMemoryKind kind = DataPathMemoryKind::HOST;
        std::int32_t accel_id = -1;
        std::uint64_t generation = 0;
        bool unregistered = false;

        // Round 16 S5 (V3): registration-time pre-built descriptors
        // (legacy build_io_slice_table 9-stage path).  When io_granularity
        // > 0 at register_memory time, these are populated and submit uses
        // pointer arithmetic (e.prp_entry = d_descs + sub) instead of
        // per-submit PRP computation + H2D.
        struct PrebuiltDesc {
            // GPU-resident AddressDescriptor[] (24 bytes each: prp1, prp2, data_length)
            void* d_descs = nullptr;       // cudaMalloc'd
            std::uint64_t num_descs = 0;   // total sub-IO descriptors
            std::uint64_t bytes_per_slice = 0;  // granularity used (= min(io_granularity, MDTS))
            std::uint64_t ios_per_slice = 0;    // sub-IOs per slice
            // PRP-list DMA mapping (host-pinned, DMA-mapped via nvm_dma_map_data_host).
            // prp2 in each AddressDescriptor points into the pool segment's ioaddrs[].
            // The NVMe controller reads PRP lists from these IOVAs via PCIe DMA.
            // R19 S3: sub-page packing — 16 slices share one 4KiB page
            // (each slice's list uses ≤31 entries = 248B, packed at 256B slots).
            // R19 S3b: pool-managed (PrpBufPool). Ownership is in the pool;
            // freed on DataPath shutdown. No per-unregister nvm_dma_unmap.
            PrpBufRef prp_buf_ref;          // pool sub-allocation reference
            std::uint64_t num_prp_pages = 0;    // pages allocated (≤ num_slices/16)
            bool valid = false;
        };
        PrebuiltDesc prebuilt;
    };
    const MemReg* find_mem_(DataPathMemory memory) const;
    MemReg* find_mem_(DataPathMemory memory);

    // Round 16 S5 (V3): registration-time pre-build (legacy 9-stage).
    // Returns true on success, false on failure (error in status_msg).
    bool build_prebuilt_descriptors_(MemReg& reg,
                                     std::uint64_t io_granularity,
                                     std::string& status_msg);
    void destroy_prebuilt_descriptors_(MemReg& reg);
    // Each submitted op leases workspace from the MetadataArena (event,
    // d_entries, d_status, PRP-list pages).  The op holds references to
    // target and memory identities; close and unregister must reject if
    // any IN_FLIGHT op references them.

    // Completion mode (private DataPath metadata, not public/SPI).
    // EVENT: normal path -- cudaEventRecord on caller stream after kernel.
    // STREAM_QUERY: fallback when event record fails after launch --
    //   progress() uses cudaStreamQuery on the borrowed stream.
    enum class CompletionMode { EVENT, STREAM_QUERY };

    struct OpEntry {
        OpState state = OpState::IN_FLIGHT;
        Status status;
        std::uint64_t bytes_transferred = 0;
        std::uint64_t total_bytes = 0;       // target bytes for completion verification

        // Arena lease: slot_index identifies the borrowed arena slot.
        // UINT32_MAX = no lease (op not yet submitted or already released).
        std::uint32_t arena_slot = UINT32_MAX;

        // Per-op GPU workspace (borrowed from arena -- NOT cudaMalloc'd per-op).
        DeviceSubmitEntry* d_entries = nullptr;  // arena slot's entry array
        EntryCompletionStatus* d_status = nullptr; // arena slot's status array
        std::uint32_t entry_count = 0;
        // Round 16 S6 (REQUIRED 0): entry lengths (was inline in
        // DeviceSubmitEntry::length; now in descriptor on GPU, but
        // aggregate_completion_status_ needs them host-side).
        std::vector<std::uint64_t> entry_lengths;
        void* event = nullptr;          // arena slot's pre-created cudaEvent_t
        void* stream = nullptr;         // borrowed cudaStream_t
        CompletionMode completion_mode = CompletionMode::EVENT;

        // PRP-list workspace (borrowed from arena's pre-allocated DMA-mapped pool).
        // prp_list_dma is the arena's shared DMA mapping; per-slot IOVAs
        // start at prp_ioaddrs_base.  prp_pages_devptr is this slot's
        // PRP page GPU base.  prp_list_page_count = pages actually used.
        nvm_dma_t* prp_list_dma = nullptr;   // arena DMA mapping (shared, borrowed)
        std::uint32_t prp_ioaddrs_base = 0;  // this slot's first IOVA index
        void* prp_pages_devptr = nullptr;    // this slot's PRP page GPU base
        std::uint32_t prp_list_page_count = 0; // pages used (LIST sub-IOs, 0 if none)

        // PRP cache references (when PrpPageCache is enabled).
        // Each cached PRP-list page is pinned for the op's lifetime.
        struct PrpCacheRef {
            PrpPageCache::Entry* entry = nullptr;
            std::uint64_t ioaddr = 0;
        };
        std::vector<PrpCacheRef> prp_cache_refs;  // empty if cache disabled

        // Handle cache entries pinned by this op (for unpin on release).
        std::vector<HandleWorkspaceCache::Entry*> handle_cache_refs;

        // FIX 4: set by aggregate_completion_status_ when any entry result==2.
        // When true, release()/shutdown() call arena_.release_with_timeout_leak()
        // instead of arena_.release(): the timed-out NVMe command may still be
        // in the controller SQ/CQ and could DMA into the PRP-list pages after
        // they are reused.  The slot is permanently consumed (bounded leak).
        // The CID is also not returned to the SQ, so that queue slot is
        // degraded until an abort/reset (future work).  event/d_entries/
        // d_status are still returned to the arena -- the kernel has returned.
        bool has_timeout = false;

        // Target/memory identities referenced by this op (for in-flight tracking).
        std::vector<std::uint64_t> target_tokens;
        std::vector<std::uint64_t> memory_tokens;

        std::uint64_t op_token = 0;
        std::uint64_t op_generation = 0;
    };
    std::unordered_map<std::uint64_t, OpEntry> ops_;
    std::uint64_t next_op_token_ = 1;
    const OpEntry* find_op_(DataPathOp op) const;
    OpEntry* find_op_(DataPathOp op);
    bool target_has_inflight_ops_(std::uint64_t token) const;
    bool memory_has_inflight_ops_(std::uint64_t token) const;

    // Aggregate per-entry completion status: D2H the status array,
    // check each entry's result, set op.state/status/bytes_transferred.
    // Called when the stream event signals (kernel finished).
    void aggregate_completion_status_(OpEntry& op);

    // Controller connection (client-only attach).
    std::string snvme_dev_path_;
    std::uint32_t bar0_size_ = 0;
    nvm_ctrl_t* ctrl_ = nullptr;

    // Queue group parameters (production mode).
    std::uint32_t cuda_device_ = 0;
    std::uint32_t num_user_queues_ = 0;
    std::uint32_t queue_depth_ = 0;
    std::uint32_t namespace_id_ = 0;
    std::uint32_t block_size_ = 0;
    std::string controller_pci_addr_;

    // IO limits.
    std::uint64_t mdts_bytes_ = 0;        // configured override (0 = use hardware)
    std::uint64_t hardware_mdts_bytes_ = 0; // from dev_info.max_data_size
    std::uint64_t effective_mdts_bytes_ = 0; // min(override, hardware) or hardware
    std::uint32_t max_batch_entries_ = 0; // max sub-IO entries per op (private limit)
    std::uint64_t max_batch_requests_ = 0; // max input request count
    std::uint64_t max_in_flight_operations_ = 16; // enforced cap on ops_.size()
    std::uint64_t max_request_bytes_ = 0;   // max bytes per single request (with fan-out)
    // Constructor override for max_request_bytes_ (0 = compute in initialize()
    // as max_batch_entries_ * effective_mdts_bytes_, the pre-S4 formula).
    std::uint64_t max_request_bytes_override_ = 0;
    std::uint32_t threads_per_block_ = 16;
    std::uint64_t prp_list_page_capacity_ = 0; // max data pages per single PRP-list page

    // Test-only submit failure injection seams.
    bool test_inject_launch_failure_ = false;
    bool test_inject_event_record_failure_ = false;
    bool test_inject_resolve_lba_failure_ = false;
    // FIX 2: synthesize NVMe CQ error (dword3 SCT/SC bit) on normal completion.
    bool test_inject_nvme_error_ = false;
    // FIX 3: force progress() query to return a persistent CUDA error.
    bool test_inject_query_error_ = false;

    // CQ poll budget: max iterations before a per-entry CQ timeout.
    // 0 means use the default (set in constructor).
    std::uint32_t cq_poll_budget_ = 0;

    // Round 15 S4 test-only counters (see test_submit_call_count() above).
    std::uint64_t test_submit_call_count_ = 0;
    std::uint64_t test_kernel_launch_count_ = 0;

    // Queue group (created in initialize(), destroyed before ctrl free).
    std::unique_ptr<NvmeQueueGroup> queue_group_;

    // MetadataArena: per-device, bounded pool of per-op workspace.
    // Pre-allocates all events, entry/status arrays, and PRP-list pages
    // at initialize() time.  submit() leases from the arena (zero cudaMalloc);
    // release() returns the slot.  Timeout ops leak the slot (bounded).
    MetadataArena arena_;

    // HandleWorkspaceCache: caches device target handles by file extent signature.
    // Capacity 0 = disabled (open/close build/free per call, current behavior).
    HandleWorkspaceCache handle_cache_;
    std::uint32_t handle_cache_capacity_ = 0;
    std::uint32_t handle_cache_l2_capacity_ = 0;  // Round 16 S6b: 0 = 4×L1

    // PrpPageCache: caches PRP-list pages by {memory_token, start_page, pages_in_io}.
    // Capacity 0 = disabled (arena PRP pool used per submit, current behavior).
    PrpPageCache prp_cache_;
    std::uint32_t prp_cache_capacity_ = 0;

    // Memory registration table: token -> MemReg.
    std::unordered_map<std::uint64_t, MemReg> mem_regs_;
    std::uint64_t next_mem_token_ = 1;

    // R19 S3 REQUIRED 3: GPU descriptor pool — replaces per-registration
    // cudaMalloc for pre-built AddressDescriptor[] arrays.
    DescPool desc_pool_;

    // R19 S3b REQUIRED 1: host-pinned PRP-list buffer pool — replaces
    // per-registration nvm_dma_map_data_host for pre-built PRP pages.
    PrpBufPool prp_buf_pool_;
};

} // namespace tutti::data_paths::local_nvme
