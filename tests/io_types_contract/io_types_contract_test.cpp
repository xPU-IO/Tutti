// tests/io_types_contract/io_types_contract_test.cpp
//
// Hardware-free contract test for <tutti/io_types.h>.
// Plain C++17 executable; no GTest, no CUDA SDK, no hardware, no IO.
// Tutti public types are obtained ONLY via <tutti/io_types.h>.

#include <tutti/io_types.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <utility>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

} // namespace

// =========================================================================
// Compile-time type properties (static_assert)
// =========================================================================

// 1. The three handles are distinct types.
static_assert(!std::is_same_v<tutti::MemoryHandle, tutti::TargetHandle>,
              "MemoryHandle and TargetHandle must be distinct");
static_assert(!std::is_same_v<tutti::MemoryHandle, tutti::IoHandle>,
              "MemoryHandle and IoHandle must be distinct");
static_assert(!std::is_same_v<tutti::TargetHandle, tutti::IoHandle>,
              "TargetHandle and IoHandle must be distinct");

// 2. No implicit conversion between handle types.
static_assert(!std::is_convertible_v<tutti::MemoryHandle, tutti::TargetHandle>);
static_assert(!std::is_convertible_v<tutti::MemoryHandle, tutti::IoHandle>);
static_assert(!std::is_convertible_v<tutti::TargetHandle, tutti::MemoryHandle>);
static_assert(!std::is_convertible_v<tutti::TargetHandle, tutti::IoHandle>);
static_assert(!std::is_convertible_v<tutti::IoHandle, tutti::MemoryHandle>);
static_assert(!std::is_convertible_v<tutti::IoHandle, tutti::TargetHandle>);

// 3. Cannot implicitly construct from void* or integers.
static_assert(!std::is_convertible_v<void*, tutti::MemoryHandle>);
static_assert(!std::is_convertible_v<void*, tutti::TargetHandle>);
static_assert(!std::is_convertible_v<void*, tutti::IoHandle>);
static_assert(!std::is_convertible_v<int, tutti::MemoryHandle>);
static_assert(!std::is_convertible_v<int, tutti::TargetHandle>);
static_assert(!std::is_convertible_v<int, tutti::IoHandle>);
static_assert(!std::is_constructible_v<tutti::MemoryHandle, void*>);
static_assert(!std::is_constructible_v<tutti::TargetHandle, void*>);
static_assert(!std::is_constructible_v<tutti::IoHandle, void*>);
static_assert(!std::is_constructible_v<tutti::MemoryHandle, int>);
static_assert(!std::is_constructible_v<tutti::TargetHandle, int>);
static_assert(!std::is_constructible_v<tutti::IoHandle, int>);

// 5. Default-constructible, copy/move constructible and assignable.
static_assert(std::is_default_constructible_v<tutti::MemoryHandle>);
static_assert(std::is_default_constructible_v<tutti::TargetHandle>);
static_assert(std::is_default_constructible_v<tutti::IoHandle>);
static_assert(std::is_copy_constructible_v<tutti::MemoryHandle>);
static_assert(std::is_copy_constructible_v<tutti::TargetHandle>);
static_assert(std::is_copy_constructible_v<tutti::IoHandle>);
static_assert(std::is_move_constructible_v<tutti::MemoryHandle>);
static_assert(std::is_move_constructible_v<tutti::TargetHandle>);
static_assert(std::is_move_constructible_v<tutti::IoHandle>);
static_assert(std::is_copy_assignable_v<tutti::MemoryHandle>);
static_assert(std::is_copy_assignable_v<tutti::TargetHandle>);
static_assert(std::is_copy_assignable_v<tutti::IoHandle>);
static_assert(std::is_move_assignable_v<tutti::MemoryHandle>);
static_assert(std::is_move_assignable_v<tutti::TargetHandle>);
static_assert(std::is_move_assignable_v<tutti::IoHandle>);

// 7. The test deliberately does NOT pin sizeof/offsetof/internal layout. Only
//    assert the handles are complete object types; nothing else about their
//    representation is relied upon.
static_assert(std::is_object_v<tutti::MemoryHandle>);
static_assert(std::is_object_v<tutti::TargetHandle>);
static_assert(std::is_object_v<tutti::IoHandle>);

// 9/10. IoRequest fields have the frozen types; both offsets are std::uint64_t
//       (independent fields, not a single shared offset).
static_assert(std::is_same_v<decltype(tutti::IoRequest::direction), tutti::IoDirection>);
static_assert(std::is_same_v<decltype(tutti::IoRequest::memory), tutti::MemoryHandle>);
static_assert(std::is_same_v<decltype(tutti::IoRequest::memory_offset), std::uint64_t>);
static_assert(std::is_same_v<decltype(tutti::IoRequest::target), tutti::TargetHandle>);
static_assert(std::is_same_v<decltype(tutti::IoRequest::target_offset), std::uint64_t>);
static_assert(std::is_same_v<decltype(tutti::IoRequest::length), std::uint64_t>);

// 11. ExecutionDomain has exactly the two required enumerators, distinct.
static_assert(tutti::ExecutionDomain::HOST_EXECUTION !=
              tutti::ExecutionDomain::DEVICE_EXECUTION);

// 12. HostSubmitContext field types; cudaStream_t comes via <tutti/cuda_like.h>.
static_assert(std::is_same_v<decltype(tutti::HostSubmitContext::execution_domain),
                             tutti::ExecutionDomain>);
static_assert(std::is_same_v<decltype(tutti::HostSubmitContext::accel_id),
                             std::int32_t>);
static_assert(std::is_same_v<decltype(tutti::HostSubmitContext::stream),
                             cudaStream_t>);

// =========================================================================
// Runtime semantics
// =========================================================================

int main() {
    // 4. Default-constructed handles are invalid.
    CHECK(!tutti::MemoryHandle{}.valid());
    CHECK(!tutti::TargetHandle{}.valid());
    CHECK(!tutti::IoHandle{}.valid());

    {
        tutti::MemoryHandle mh;
        tutti::TargetHandle th;
        tutti::IoHandle     ih;
        CHECK(!mh.valid());
        CHECK(!th.valid());
        CHECK(!ih.valid());
    }

    // 6. Same-type invalid handle equality.
    CHECK(tutti::MemoryHandle{} == tutti::MemoryHandle{});
    CHECK(tutti::TargetHandle{} == tutti::TargetHandle{});
    CHECK(tutti::IoHandle{} == tutti::IoHandle{});
    CHECK(!(tutti::MemoryHandle{} != tutti::MemoryHandle{}));

    // 5. Copy/move construct and assign (lightweight value type).
    {
        tutti::MemoryHandle a;
        tutti::MemoryHandle b(a);             // copy ctor
        tutti::MemoryHandle c(std::move(b));  // move ctor
        tutti::MemoryHandle d;
        d = a;                                // copy assign
        tutti::MemoryHandle e;
        e = std::move(c);                     // move assign
        CHECK(a == d);
        CHECK(a == e);
        CHECK(!a.valid());
    }

    // 8. IoDirection enumerators.
    {
        tutti::IoDirection r = tutti::IoDirection::READ;
        tutti::IoDirection w = tutti::IoDirection::WRITE;
        CHECK(r == tutti::IoDirection::READ);
        CHECK(w == tutti::IoDirection::WRITE);
        CHECK(r != w);
    }

    // 9/10. IoRequest: all six fields assignable/readable; offsets independent.
    {
        tutti::IoRequest req{};
        req.direction     = tutti::IoDirection::WRITE;
        req.memory        = tutti::MemoryHandle{};
        req.memory_offset = 4096;
        req.target        = tutti::TargetHandle{};
        req.target_offset = 1048576;
        req.length        = 65536;

        CHECK(req.direction == tutti::IoDirection::WRITE);
        CHECK(!req.memory.valid());
        CHECK(req.memory_offset == 4096u);
        CHECK(!req.target.valid());
        CHECK(req.target_offset == 1048576u);
        CHECK(req.length == 65536u);

        // The two offsets are independent fields.
        req.memory_offset = 10;
        req.target_offset = 20;
        CHECK(req.memory_offset == 10u);
        CHECK(req.target_offset == 20u);
        CHECK(req.memory_offset != req.target_offset);
    }

    // 12. HostSubmitContext under HOST profile; null stream expresses HOST exec.
    {
        tutti::HostSubmitContext ctx{
            tutti::ExecutionDomain::HOST_EXECUTION,
            0,
            nullptr};
        CHECK(ctx.execution_domain == tutti::ExecutionDomain::HOST_EXECUTION);
        CHECK(ctx.accel_id == 0);
        CHECK(ctx.stream == nullptr);

        ctx.execution_domain = tutti::ExecutionDomain::DEVICE_EXECUTION;
        ctx.accel_id = 1;
        CHECK(ctx.execution_domain == tutti::ExecutionDomain::DEVICE_EXECUTION);
        CHECK(ctx.accel_id == 1);
    }

    if (g_failures == 0) {
        std::printf("tutti_io_types_contract_test: all checks passed\n");
        return 0;
    }
    std::printf("tutti_io_types_contract_test: %d failure(s)\n", g_failures);
    return 1;
}
