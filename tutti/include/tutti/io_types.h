#pragma once

// tutti/io_types.h -- Frozen Phase 1 public IO nouns.
//
// Header-only value types describing memory/target identities and byte-range
// IO intent. No runtime, registry, data-path or validation logic lives here.
// Accelerator contract types (e.g. cudaStream_t) are obtained solely via the
// unified <tutti/cuda_like.h> selector.
//
// Allowed includes: <cstdint>, <tutti/cuda_like.h>.

#include <cstdint>
#include <tutti/cuda_like.h>

namespace tutti {

// The future runtime that owns valid handle minting. Forward-declared only;
// not implemented in this header (no registry, no heap allocation).
class StorageRuntime;

namespace detail {

// Phantom-tagged opaque handle. Each public handle type is a distinct
// instantiation; the Tag prevents cross-type conversion. Internal identity is
// (runtime_id, slot, generation); generation == 0 denotes an invalid
// (never-minted) handle. The binary layout of these fields is NOT part of the
// public ABI and may change; only the public contract (default-invalid,
// valid(), equality, copy/move) is stable.
//
// Valid handles can only be minted through the private constructor, which is
// accessible exclusively to the future StorageRuntime. No public factory is
// exposed, so ordinary application code cannot forge a valid handle.
template <typename Tag>
class OpaqueHandle {
public:
    constexpr OpaqueHandle() noexcept = default;

    constexpr bool valid() const noexcept { return generation_ != 0; }

    constexpr bool operator==(const OpaqueHandle& other) const noexcept {
        return runtime_id_ == other.runtime_id_
            && slot_ == other.slot_
            && generation_ == other.generation_;
    }
    constexpr bool operator!=(const OpaqueHandle& other) const noexcept {
        return !(*this == other);
    }

private:
    friend class ::tutti::StorageRuntime;
    constexpr OpaqueHandle(std::uint32_t runtime_id, std::uint32_t slot,
                           std::uint64_t generation) noexcept
        : runtime_id_(runtime_id), slot_(slot), generation_(generation) {}

    std::uint32_t runtime_id_ = 0;
    std::uint32_t slot_       = 0;
    std::uint64_t generation_ = 0;
};

struct MemoryHandleTag {};
struct TargetHandleTag {};
struct IoHandleTag {};

} // namespace detail

using MemoryHandle = detail::OpaqueHandle<detail::MemoryHandleTag>;
using TargetHandle = detail::OpaqueHandle<detail::TargetHandleTag>;
using IoHandle     = detail::OpaqueHandle<detail::IoHandleTag>;

// Direction of a byte-range transfer.
enum class IoDirection {
    READ,   // target -> memory
    WRITE,  // memory -> target
};

// Backend-neutral byte-range IO intent. Exactly these six fields; both offsets
// and length are in bytes. No stream, no backend/data-path pointer, no tensor
// shape, no storage-private descriptor. Bounds/alignment/capability
// validation is the runtime's responsibility, not this value type's.
struct IoRequest {
    IoDirection   direction;
    MemoryHandle  memory;
    std::uint64_t memory_offset;
    TargetHandle  target;
    std::uint64_t target_offset;
    std::uint64_t length;
};

// Where submitted data-path work executes, from the host caller's perspective.
enum class ExecutionDomain {
    HOST_EXECUTION,
    DEVICE_EXECUTION,
};

// Host-side submission context. cudaStream_t comes only from
// <tutti/cuda_like.h>. HOST_EXECUTION may use a null stream; for
// DEVICE_EXECUTION the runtime will require a non-null stream. This value type
// performs no CUDA API calls and no runtime validation.
struct HostSubmitContext {
    ExecutionDomain execution_domain;
    // -1 means "use the owning Runtime's accelerator".  This is an
    // accelerator identity, not a daemon NVMe device_id.
    std::int32_t    accel_id;
    cudaStream_t    stream;
};

} // namespace tutti
