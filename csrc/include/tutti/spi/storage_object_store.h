#pragma once

// tutti/spi/storage_object_store.h -- In-repo storage-object-layer SPI.
//
// Object lifecycle, space accounting, validity determination and crash
// recovery for durable KV objects. This is the layer that owns every
// filesystem-level concern: it exists so that upper layers (KV index,
// orchestration, framework adapters) never issue a POSIX call.
//
// Relationship to the neighbouring SPIs:
//
//   StorageTargetResolver  name -> ResolvedTarget          (read-only)
//   StorageObjectStore     key  -> ObjectPlacement         (this header)
//   StorageRuntime         uri  -> ticket -> IO
//
// StorageTargetResolver requires the file to already exist with physical
// blocks allocated; it cannot create, size, or invalidate anything. This SPI
// supplies the missing verbs. It deliberately does NOT resolve: the runtime
// owns that, and doing it here would resolve every object twice. See
// ObjectPlacement.
//
// Correctness boundary: this layer guarantees METADATA self-consistency, not
// payload correctness. Payload is never checksummed -- doing so would route
// data through the host CPU and defeat the GPU-direct premise. See the
// commentary on ObjectHeaderLayout.
//
// No transport-private, device-private, or kernel-private types appear here.
// Allowed includes: <tutti/status.h>, <tutti/spi/object_digest.h>, and the
// C++17 standard library.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <tutti/status.h>
#include <tutti/spi/object_digest.h>

namespace tutti {

// -------------------------------------------------------------------------
// ObjectKey
//
// The store's only notion of object identity: an opaque byte string. The
// store never interprets the contents -- it compares and hashes them. All
// semantics (chunk hash, layer, rank, model fingerprint) are encoded into
// these bytes by the caller.
//
// identity() is a stable 64-bit digest written into the object header so a
// slot can prove what it holds. A digest collision cannot yield wrong data:
// recovery also compares key_crc32 and payload_bytes (see ObjectHeaderLayout).
// -------------------------------------------------------------------------
struct ObjectKey {
    std::vector<std::uint8_t> bytes;

    // Stable 64-bit digest of `bytes`, persisted in object headers. Must be
    // deterministic across processes, builds and hosts -- see object_digest.h
    // for why std::hash cannot be used here.
    std::uint64_t identity() const noexcept {
        return object_identity(bytes.data(), bytes.size());
    }

    // CRC-32C over the full key bytes. Recorded alongside identity() so a
    // 64-bit digest collision cannot cause a wrong object to be accepted.
    std::uint32_t key_crc32() const noexcept {
        return crc32c(bytes.data(), bytes.size());
    }

    bool operator==(const ObjectKey& other) const noexcept {
        return bytes == other.bytes;
    }
    bool operator!=(const ObjectKey& other) const noexcept {
        return !(*this == other);
    }
};

// -------------------------------------------------------------------------
// ObjectLayout -- object geometry, declared once at open().
//
// An object is one contiguous payload divided into fixed-size segments. KV
// layers map onto segments, but this layer does not know what a "layer" is.
// All segments of one object live inside that one object: a chunk's entire
// per-layer payload is a single file, not one file per layer.
// -------------------------------------------------------------------------
struct ObjectLayout {
    std::uint64_t segment_bytes = 0;
    std::uint32_t segment_count = 0;

    std::uint64_t payload_bytes() const noexcept {
        return segment_bytes * static_cast<std::uint64_t>(segment_count);
    }

    bool operator==(const ObjectLayout& other) const noexcept {
        return segment_bytes == other.segment_bytes &&
               segment_count == other.segment_count;
    }
};

// -------------------------------------------------------------------------
// StoreDevice -- one backing device.
//
// mount_path is what this layer uses: it decides where slot files live. The
// remaining fields are deployment facts the RUNTIME's resolver needs -- the
// controller identity lets it prove a file's FIEMAP extents belong to the
// namespace it was configured for, so a file cannot be mapped onto the wrong
// device. They travel here because the caller configures one store, not two
// subsystems, but this layer only reads mount_path.
// -------------------------------------------------------------------------
struct StoreDevice {
    // Directory holding this device's slot files. Required.
    std::string mount_path;

    // NVMe namespace identity, for the runtime's resolver.
    std::string controller_pci_addr;
    std::uint32_t namespace_id = 0;
    std::uint32_t block_size = 0;

    // Block device whose extents back mount_path (e.g. /dev/snvme0n1), and the
    // namespace's byte offset within it.
    std::string backing_device_path;
    std::uint64_t namespace_base_bytes = 0;
};

// -------------------------------------------------------------------------
// StoreConfig
//
// capacity_bytes / capacity_slots is a CEILING declaration. Media is
// materialised one file group at a time (one fixed-size segment file per
// device): open() makes the first group ready, precreate_step() adds later
// groups. Materialisation writes real zeros (FIEMAP-backed DMA cannot use
// sparse or preallocated holes), so the ceiling never implies an up-front
// write of the whole capacity.
// -------------------------------------------------------------------------
struct StoreConfig {
    // Namespace root: where this store's metadata (checkpoint, residency
    // bitmaps) lives. Slot data lives under each device's mount_path, which for
    // a single-device deployment is usually this same directory.
    std::string uri;

    // Total capacity ceiling for this store. 0 means "whatever the backend
    // reports as usable". Implementations clamp to the real backend capacity.
    //
    // Slot-count form of the same knob. A caller that thinks in objects (chunks)
    // should size the cache in slots: only the store knows the object geometry,
    // so a byte figure derived by the caller can disagree with the real per-slot
    // space and silently produce fewer slots than asked for. Non-zero wins.
    std::uint64_t capacity_slots = 0;
    std::uint64_t capacity_bytes = 0;

    ObjectLayout layout;

    // Fixed-size shared file geometry; a file holds segment_file_slots slots.
    // These must be nonzero. A change to either value requires a new namespace.
    std::uint64_t segment_file_slots = 0;

    // Reserved prefix within each slot. The object header remains in its first
    // 4096 bytes; the payload starts at this 4096-aligned offset.
    std::uint64_t segment_header_bytes = 0;

    // Geometry/model fingerprint. A mismatch against persisted state is
    // fail-closed (open() returns INVALID_ARGUMENT and preserves the data).
    std::vector<std::uint8_t> namespace_fingerprint;

    // Backing devices. Slots rotate over them by slot number; each device holds
    // this rank's fixed-size segment files under <mount_path>/r<rank>/segments.
    std::vector<StoreDevice> devices;

    // Background space reclamation. When false, reclamation runs
    // synchronously on the calling thread (single-threaded tests).
    bool background_reclaim = true;

    // ---- Cross-rank residency bitmap ----

    std::uint32_t rank_id = 0;

    // 1 means single-rank: no bitmap is created and
    // contains_prefix_all_ranks() degenerates to contains_prefix().
    std::uint32_t rank_count = 1;

    // Read-only view of a pool owned by another process (e.g. the scheduler's
    // residency index reading a worker's namespace). open() then loads the
    // checkpoint and maps the residency bitmaps WITHOUT creating directories,
    // precreating slots or warming up, and every mutating call fails with
    // UNSUPPORTED instead of being silently ignored -- a misconfigured writer
    // must fail loudly, not lose data.
    bool read_only = false;

    // msync period for the residency bitmap. 0 = never actively sync (rely
    // on kernel writeback alone). Purely a performance knob: the bitmap is an
    // accelerating index, never authoritative, and its skew direction is
    // pinned to under-reporting. See contains_prefix_all_ranks().
    std::uint32_t residency_sync_interval_ms = 1000;
};

// -------------------------------------------------------------------------
// ObjectPlacement -- where an object's payload lives.
//
// Carries a URI plus an offset rather than a resolved target, because
// RESOLUTION BELONGS TO THE RUNTIME. StorageRuntime::open_batch() takes URIs,
// looks up a resolver by scheme, and mints the ticket that the data path
// actually uses. If this layer resolved as well, every object would be resolved
// twice -- and resolution is open + fstat + fsync + FIEMAP, plus a
// peer-memory DMA mapping serialised by a global driver lock. Doing it twice is
// not a minor waste; that path is historically where a first write stalled for
// tens of seconds.
//
// So the division is: this layer decides WHICH slot an object occupies and
// whether it is valid; the runtime decides how that slot's bytes reach the GPU.
//
// Callers form per-segment IO as:
//     ticket        = runtime.open(placement.uri)      // cached per slot
//     target_offset = placement.offset + segment_index * layout.segment_bytes
// -------------------------------------------------------------------------
struct ObjectPlacement {
    // Slot URI in the scheme the configured backend expects. A pure function of
    // the slot number, so a caller may cache tickets keyed by this string.
    std::string uri;

    // Payload start within the object's logical address space: already past
    // the object header, so segment 0 begins exactly here.
    std::uint64_t offset = 0;
    std::uint64_t payload_bytes = 0;

    // Slot number. Exposed because the residency bitmap is indexed by it and
    // because a caller may want to key its own ticket cache on it rather than
    // on the URI string.
    std::uint64_t slot = 0;

    // Space generation, bumped whenever the underlying space is recycled.
    // Detects ABA reuse: a stale placement carries a stale generation.
    std::uint64_t generation = 0;

    bool valid() const noexcept { return !uri.empty(); }
};

// -------------------------------------------------------------------------
// StoreUsage -- a queryable ratio, not an alarm.
//
// A cache that never reaches its ceiling is over-provisioned, so "full" is a
// steady state rather than a fault. This layer therefore reports usage and
// lets the caller decide; it never warns on high occupancy.
// -------------------------------------------------------------------------
struct StoreUsage {
    std::uint64_t capacity_bytes = 0;    // effective, already clamped
    std::uint64_t committed_bytes = 0;   // durable, valid objects
    std::uint64_t reserved_bytes = 0;    // reserved, not yet committed
    std::uint64_t reclaiming_bytes = 0;  // being reclaimed, not yet allocatable
    std::uint64_t usable_bytes = 0;      // allocatable right now
    std::uint64_t free_slots = 0;        // materialised slots ready to reserve
    std::uint64_t reclaiming_slots = 0;  // slots awaiting physical zeroing
    std::uint64_t reclaimed_total = 0;   // successful scrubs since open()
    std::uint64_t reclaim_failures = 0;  // scrub attempts that were requeued

    // Per backing device occupancy, in the backend's device order. Striped
    // layouts are not required to be balanced -- the constraint is only that
    // no device exceeds its own available space. The apportioning rule is
    // implementation-private.
    std::vector<std::uint64_t> per_device_bytes;
};

// -------------------------------------------------------------------------
// ReserveOutcome -- partial acceptance is a legal result, not an error.
//
// When capacity is exhausted, reserve() takes what it can and reports the
// remainder in rejected_count with status OK. It never blocks waiting for
// space and never raises: a cache miss must not fail the request that
// produced it.
//
// accepted[i] corresponds to accepted_keys[i].
// -------------------------------------------------------------------------
struct ReserveOutcome {
    std::vector<ObjectPlacement> accepted;
    std::vector<ObjectKey> accepted_keys;
    std::uint64_t rejected_count = 0;
};

// -------------------------------------------------------------------------
// ObjectHeaderLayout -- on-media object header, replacing all marker files.
//
// Every object is preceded by a header of kHeaderBytes (4096, satisfying
// O_DIRECT alignment). A valid header IS the statement that the payload is
// valid; there is no external marker file.
//
// commit() writes this header and fsyncs it. That is the single commit action
// per object -- not one per segment. Callers may write segments in any order,
// or only some of them: before commit, the object is uniformly invalid. This
// is what makes "if the write did not finish, treat it as never written" the
// structural default rather than a convention.
//
// CRC coverage is METADATA ONLY:
//   header_crc32 covers the header's own leading bytes.
//   key_crc32    covers ObjectKey::bytes, guarding against identity()
//                digest collisions.
// The payload is never checksummed. At ~10 MiB per object and terabyte
// capacities, checksumming on the read path would pull data through the host
// CPU and destroy the GPU-direct premise. Payload integrity rests on the
// device's own end-to-end protection plus commit ordering (payload fsync
// before header fsync). A valid header proves the payload was durable at
// commit time; it does not prove the media has not since degraded.
// -------------------------------------------------------------------------
struct ObjectHeaderLayout {
    static constexpr std::uint64_t kHeaderBytes = 4096;
    static constexpr char kMagic[8] = {'T', 'U', 'T', 'T', 'I', 'O', 'B', 'J'};
    static constexpr std::uint32_t kVersion = 1;

    // Field offsets within the header.
    static constexpr std::size_t kMagicOffset        = 0;   // 8 bytes
    static constexpr std::size_t kVersionOffset      = 8;   // 4
    static constexpr std::size_t kIdentityOffset     = 16;  // 8
    static constexpr std::size_t kPayloadBytesOffset = 24;  // 8
    static constexpr std::size_t kGenerationOffset   = 32;  // 8
    static constexpr std::size_t kKeyLenOffset       = 40;  // 4
    static constexpr std::size_t kKeyCrc32Offset     = 44;  // 4
    static constexpr std::size_t kCommitSeqOffset    = 48;  // 8
    static constexpr std::size_t kHeaderCrc32Offset  = 56;  // 4

    // header_crc32 covers [0, kHeaderCrc32Offset).
    static constexpr std::size_t kCrcCoveredBytes = kHeaderCrc32Offset;
};

// -------------------------------------------------------------------------
// CheckpointLayout -- metadata checkpoint region.
//
// Mirrored containers written alternately, each carrying a monotonic sequence
// number and a CRC over its body. Recovery picks the highest sequence that
// passes CRC. Atomicity comes from never overwriting the container currently
// holding the newest valid state, so no rename or temp file is needed.
//
// The checkpoint is an ACCELERATOR, not the truth: losing it entirely is
// survivable by scanning object headers (at the cost of reading every
// header). This differs deliberately from designs where the manifest is the
// sole record of what exists -- there, losing it makes all data unusable.
// -------------------------------------------------------------------------
struct CheckpointLayout {
    static constexpr char kMagic[8] = {'T', 'U', 'T', 'T', 'I', 'I', 'D', 'X'};
    static constexpr std::uint32_t kVersion = 1;
    static constexpr std::uint32_t kContainerCount = 2;
    static constexpr std::uint64_t kContainerHeaderBytes = 4096;

    static constexpr std::size_t kMagicOffset      = 0;   // 8 bytes
    static constexpr std::size_t kVersionOffset    = 8;   // 4
    static constexpr std::size_t kSequenceOffset   = 12;  // 8
    static constexpr std::size_t kBodyBytesOffset  = 20;  // 8
    static constexpr std::size_t kBodyCrc32Offset  = 28;  // 4
};

// -------------------------------------------------------------------------
// ResidencyBitmapLayout -- cross-rank residency, mmap-backed.
//
// One file per rank ("residency/r<rank>.bitmap"), each mmapped by its owner
// for writing. Bit s means "slot s is committed on this rank". commit() sets,
// release()/abort() clear -- all in memory. Durability comes from kernel
// writeback plus the periodic msync configured by
// StoreConfig::residency_sync_interval_ms.
//
// One file per rank (rather than one shared file) means writers never
// conflict, there is no cross-process false sharing, and no atomic bit
// operations are required. A reader mmaps all rank_count files and ANDs them
// to obtain "committed on every rank".
//
// THE BITMAP IS NEVER AUTHORITATIVE. Dirty mmap pages are written back on the
// kernel's schedule, so after a crash it is unknowable which pages landed;
// "the bitmap says present" is therefore untrustworthy by construction. Two
// mechanisms pin the skew direction to under-reporting only:
//
//   1. During recovery the bitmap supplies candidates only. Authority remains
//      the object-header cross-check, and the bitmap is rebuilt from the
//      verified set rather than trusted as found.
//   2. If the bitmap is unusable (missing, bad magic/fingerprint, slot_count
//      mismatch), contains_prefix_all_ranks() degenerates to this rank's own
//      contains_prefix().
//
// Under-reporting costs a recomputation; over-reporting would load truncated
// KV and corrupt results silently. Only the former is permitted.
// -------------------------------------------------------------------------
struct ResidencyBitmapLayout {
    static constexpr char kMagic[8] = {'T', 'U', 'T', 'T', 'I', 'R', 'E', 'S'};
    static constexpr std::uint32_t kVersion = 1;
    static constexpr std::uint64_t kHeaderBytes = 4096;

    static constexpr std::size_t kMagicOffset           = 0;   // 8 bytes
    static constexpr std::size_t kVersionOffset         = 8;   // 4
    static constexpr std::size_t kRankIdOffset          = 12;  // 4
    static constexpr std::size_t kRankCountOffset       = 16;  // 4
    static constexpr std::size_t kSlotCountOffset       = 20;  // 8
    static constexpr std::size_t kFingerprintOffset     = 28;  // 8 (digest)
};

// -------------------------------------------------------------------------
// StorageObjectStore
//
// Lifecycle: open() -> [reserve/commit/abort/lookup/release/pin/unpin]
//            -> close()
//
// reserve/commit are separate because space occupancy and data validity are
// distinct events with an IO window between them. A crash inside that window
// leaves the reservation invalid, which is exactly the desired semantics.
//
// Thread safety: all methods are safe for concurrent use. reserve, commit and
// release never block on IO -- space materialisation happens on the
// background reclaimer when enabled.
// -------------------------------------------------------------------------
class StorageObjectStore {
public:
    virtual ~StorageObjectStore() = default;

    // ---- Lifecycle ----

    // Open the namespace. If durable state exists, validate
    // namespace_fingerprint and layout: on match, recover; on mismatch,
    // return INVALID_ARGUMENT and preserve the existing data.
    //
    // Fail-closed by design, and there is deliberately no purge() entry
    // point: destroying a populated cache is an operational action, not a
    // runtime capability.
    virtual Status open(const StoreConfig& config) = 0;
    virtual Status close() = 0;

    // ---- Queries (side-effect free) ----

    virtual bool contains(const ObjectKey& key) const = 0;

    // Length of the longest prefix of `keys` that is committed in this store.
    // Batched so prefix matching does not cross the language boundary once
    // per key.
    virtual std::uint64_t contains_prefix(const ObjectKey* keys,
                                          std::size_t count) const = 0;

    // Length of the longest prefix committed on EVERY participating rank,
    // answered from the residency bitmap.
    //
    // May under-report (see ResidencyBitmapLayout); it must never
    // over-report. Degenerates to contains_prefix() when the bitmap is
    // unavailable or rank_count == 1.
    virtual std::uint64_t contains_prefix_all_ranks(
        const ObjectKey* keys, std::size_t count) const = 0;

    // Placement of a committed object. NOT_FOUND if absent or uncommitted.
    virtual Result<ObjectPlacement> lookup(const ObjectKey& key) const = 0;

    virtual StoreUsage usage() const = 0;

    // ---- Bind-time warm-up ----
    //
    // Slots whose backing files exist on media right now: the ready file-group
    // prefix, [0, ready_slots()). A caller that wants
    // to warm the runtime's target and peer-memory registration caches before
    // serving traffic needs real URIs to open: the first IO on a slot otherwise
    // pays resolve() plus registration -- hundreds of milliseconds per device,
    // serialised behind the runtime's registry lock -- on the request path
    // where it stalls the forward thread.
    virtual std::uint64_t ready_slots() const = 0;

    // URI for a slot in [0, ready_slots()). Its format is the placement's, not
    // the caller's: it must be exactly what the resolver parses. Empty when the
    // slot is not ready.
    virtual std::string slot_uri(std::uint64_t slot) const = 0;

    // Generation of a slot, so a warmed-up cache entry can be validated later
    // without reopening. Zero when the slot is out of range.
    virtual std::uint64_t slot_generation(std::uint64_t slot) const = 0;

    // ---- Write path ----

    // Reserve space. Writes no payload and asserts no validity.
    //
    // On insufficient capacity, returns partial acceptance with status OK and
    // rejected_count > 0. Keys already committed are not re-reserved: they
    // appear in the outcome carrying their existing placement.
    virtual Result<ReserveOutcome> reserve(const ObjectKey* keys,
                                            std::size_t count) = 0;

    // Declare payload valid. The caller guarantees payload IO is durable.
    //
    // Atomicity is the implementation's responsibility (object header plus
    // checkpoint). A crash before commit leaves the reservation invalid.
    virtual Status commit(const ObjectKey* keys, std::size_t count) = 0;

    // Discard uncommitted reservations; their space returns to reclamation.
    // Committed keys are ignored so this can never delete live data.
    virtual Status abort(const ObjectKey* keys, std::size_t count) = 0;

    // ---- Release and read protection ----

    // Release committed objects. Pinned objects are left alone. Returns the
    // number actually released.
    virtual Result<std::uint64_t> release(const ObjectKey* keys,
                                          std::size_t count) = 0;

    // Read protection: while pinned, release() will not reclaim the object.
    // Reentrant (reference counted).
    virtual Status pin(const ObjectKey* keys, std::size_t count) = 0;
    virtual Status unpin(const ObjectKey* keys, std::size_t count) = 0;

    // ---- Recovery ----

    // Enumerate objects confirmed valid after open(). Called once before
    // steady state; steady-state residency is owned by the caller's
    // in-memory index, not by repeated scans.
    virtual Result<std::vector<ObjectKey>> recover() = 0;

    // Force a metadata checkpoint. Implementations checkpoint on their own
    // schedule; this exists for shutdown and for tests.
    virtual Status checkpoint() = 0;

    // ---- Growth: file-group precreate ----
    //
    // The unit of precreation is a file group: one fixed-size segment file per
    // device, i.e. segment_file_slots x devices consecutive slots. A group is
    // published only after every one of its files is zero-filled, fsynced and
    // FIEMAP-verified, so [0, precreated_slots()) is always backed by media.
    //
    //   * open() adopts every group already on media and, for a writer,
    //     materialises the first group before returning. Service startup
    //     therefore waits for the first files once; nothing is created lazily.
    //   * reserve() never creates media. A slot beyond precreated_slots() is
    //     rejected like capacity exhaustion (the caller trims its write).
    //   * precreate_step() is the only growth path and runs on a background
    //     thread.

    // Materialises the next file group when fewer than `headroom` ready slots
    // remain past the allocation frontier. Returns the number of slots
    // published (0 = nothing to do, or another call is already building).
    // No store lock is held during the file IO.
    virtual Result<std::uint64_t> precreate_step(std::uint64_t headroom) {
        (void)headroom;
        return Result<std::uint64_t>::Failure(
            Status(StatusCode::UNSUPPORTED, "precreate_step unsupported"));
    }

    // Slots [0, precreated_slots()) are ready. 0 means "unknown/none".
    virtual std::uint64_t precreated_slots() const { return 0; }

    // Target computed by the last precreate_step(): growth has caught up when
    // precreated_slots() reaches it.
    virtual std::uint64_t precreate_target() const { return 0; }
};

// -------------------------------------------------------------------------
// Factory -- selects an implementation by URI scheme, mirroring
// resolver_factory. Returns UNSUPPORTED for an unknown scheme.
// -------------------------------------------------------------------------
Result<std::unique_ptr<StorageObjectStore>> create_storage_object_store(
    std::string_view scheme);

} // namespace tutti
