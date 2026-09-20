#pragma once

// tutti/memory_types.h -- Phase 1 public memory view types.
//
// Header-only C++17 value types describing memory position, allocation
// owner, and expected identity for runtime validation.  No transport,
// no vendor runtime dependency.
//
// Position (MemoryKind) and allocation owner are
// orthogonal:
//   - DEVICE + CALLER_OWNED  : caller-allocated device memory
//   - HOST   + RUNTIME_OWNED : runtime-allocated host buffer
//   - etc.
//
// EXTERNAL is no longer a MemoryKind; caller-provided memory is expressed
// as CALLER_OWNED (any kind).  No IMPORTED allocation owner exists.
//
// MemoryView is a public aggregate value type, NOT a transport
// registration record.  It carries no transport-private descriptor.
// Actual pointer validation is the future StorageRuntime's job.
//
// "Optional" expected_* fields use explicit sentinels (not std::optional):
//   - expected_accel_id < 0        : accelerator not specified
//   - expected_profile.empty()     : profile not specified
//   - expected_kind                : always set (no sentinel; caller picks)
//
// Allowed includes: <cstdint>, <string>.

#include <cstdint>
#include <string>

namespace tutti {

// -------------------------------------------------------------------------
// MemoryKind -- where memory physically resides.
//
// Exactly four values; deliberately no EXTERNAL, no IMPORTED.
// -------------------------------------------------------------------------
enum class MemoryKind {
    HOST,          // ordinary pageable host memory
    PINNED_HOST,   // page-locked host memory
    DEVICE,        // device-local memory
    MANAGED,       // unified/managed memory
};

// -------------------------------------------------------------------------
// Allocation owner enum.
//
// Exactly two values; deliberately no IMPORTED.
// -------------------------------------------------------------------------
enum class MemoryOwnership {
    RUNTIME_OWNED,  // allocated and freed by the Tutti runtime
    CALLER_OWNED,   // allocated and freed by the caller
};

// -------------------------------------------------------------------------
// MemoryView -- public memory view descriptor.
//
// Six frozen fields, in this order:
//   address            : starting address of the view
//   size               : size in bytes
//   expected_kind      : caller-expected memory position (always set)
//   allocation owner   : who allocated this memory
//   expected_accel_id  : accelerator id (< 0 = unspecified)
//   expected_profile   : expected profile name ("" = unspecified)
//
// This value type performs no pointer query or validation.
// -------------------------------------------------------------------------
struct MemoryView {
    void*           address;
    std::uint64_t   size;
    MemoryKind      expected_kind;
    MemoryOwnership ownership;
    std::int32_t    expected_accel_id;   // < 0 = unspecified
    std::string     expected_profile;    // empty = unspecified
    // Round 16 S5 (V3): io_granularity > 0 enables registration-time PRP
    // pre-build (legacy build_io_slice_table 9-stage path).  0 = dynamic
    // path (current behavior, retains generality).  When > 0, the DataPath
    // pre-builds AddressDescriptor[] at register_memory time and submit
    // uses pointer arithmetic (e.prp_entry = d_descs + sub) instead of
    // per-submit PRP computation + H2D.
    std::uint64_t   io_granularity = 0;
};

} // namespace tutti
