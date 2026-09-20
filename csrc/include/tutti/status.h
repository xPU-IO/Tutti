#pragma once

// tutti/status.h -- Phase 1 hardware-free error contract.
//
// Header-only C++17 value types: StatusCode, Status, Result<T>.
// Pure standard library; no hardware, vendor, or transport dependency.

#include <optional>
#include <string>
#include <utility>

namespace tutti {

// -------------------------------------------------------------------------
// StatusCode
//
// Strong-typed, hardware-free error classification.
// OK denotes success; every other value is an error category.
// No bare int, no bool, no vendor-private codes.
// -------------------------------------------------------------------------
enum class StatusCode {
    OK,
    INVALID_ARGUMENT,
    OUT_OF_RANGE,
    NOT_FOUND,
    UNSUPPORTED,
    NOT_READY,
    BUSY,
    RESOURCE_EXHAUSTED,
    TIMEOUT,
    DEVICE_ERROR,
    DATA_LOSS,
    INTERNAL,
};

// -------------------------------------------------------------------------
// Status
//
// Either success (OK) or an error carrying a human-readable message.
// Default-constructed Status is OK.
// Copy/move are trivial value semantics.
// -------------------------------------------------------------------------
class Status {
public:
    // Default: OK with empty message.
    Status() = default;

    // Explicit success factory.
    static Status Ok() { return Status(); }

    // Construct an error status.
    // (Using OK here is technically legal but semantically odd;
    //  callers should prefer Status::Ok() for success.)
    Status(StatusCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    bool ok() const noexcept { return code_ == StatusCode::OK; }
    StatusCode code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }

private:
    StatusCode code_ = StatusCode::OK;
    std::string message_;
};

// -------------------------------------------------------------------------
// Result<T>
//
// Exactly one of:
//   success: a T value + OK status
//   failure: a non-OK status, no T value
//
// The illegal combination "OK status but no value" is deterministically
// normalized to a non-OK INTERNAL error at construction time.
//
// Value access via value() is a precondition: has_value() must be true.
// Calling value() on a failure result is a contract violation.
// -------------------------------------------------------------------------
template <typename T>
class Result {
public:
    // ---- Success path ----

    // Implicit construction from T (success).
    Result(T value)
        : value_(std::move(value)) {}

    // Named success factory.
    static Result Success(T value) {
        return Result(std::move(value));
    }

    // ---- Failure path ----

    // Explicit construction from Status (failure).
    // If status is OK, normalizes to INTERNAL to avoid the illegal
    // "OK-without-value" state.
    explicit Result(Status status)
        : status_(normalize_failure_(std::move(status))) {}

    // Named failure factory.
    static Result Failure(Status status) {
        return Result(std::move(status));
    }

    // ---- Queries ----

    bool ok() const noexcept { return status_.ok(); }
    bool has_value() const noexcept { return value_.has_value(); }
    const Status& status() const noexcept { return status_; }

    // ---- Value access (precondition: has_value()) ----

    T& value() & { return *value_; }
    const T& value() const& { return *value_; }
    T&& value() && { return std::move(*value_); }

private:
    static Status normalize_failure_(Status status) {
        if (status.ok()) {
            return Status(StatusCode::INTERNAL,
                "Result<T> constructed with OK status but no value; "
                "normalized to INTERNAL");
        }
        return status;
    }

    std::optional<T> value_;
    Status status_;  // default OK for success path
};

} // namespace tutti
