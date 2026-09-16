#pragma once

// tutti/accelerator_device_guard.h -- thread-local current-device guard.
//
// The guard is deliberately expressed in terms of the CUDA-like profile
// abstraction.  CUDA, MUSA and MACA use their vendor runtime through that
// abstraction; HOST has a deterministic, observable no-op implementation.
//
// Like c10::DeviceGuard, this type only owns thread-local current-device
// state.  It does not own a device/context and it never introduces a vendor
// framework dependency.  A target of -1 is the equivalent of an absent
// OptionalDeviceGuard and is therefore a no-op on every backend.

#include <tutti/cuda_like.h>
#include <tutti/status.h>

#include <cstdint>
#include <string>

namespace tutti {

namespace testing {

// Test seam modelled after PyTorch's DeviceGuardImplInterface: production
// guards use the cuda-like backend below, while contract tests can inject
// deterministic get/set failures without linking C10 or libtorch.
struct DeviceGuardBackend {
    void* context = nullptr;
    Status (*get_device)(void* context, std::int32_t& device) = nullptr;
    Status (*set_device)(void* context, std::int32_t device) = nullptr;
};

struct DeviceGuardTestAccess;

} // namespace testing

namespace detail {

inline Status device_guard_get_current(void*, std::int32_t& device) {
#if defined(TUTTI_USE_HOST)
    (void)device;
    return Status(StatusCode::UNSUPPORTED,
                  "HOST profile has no current accelerator");
#else
    int current = -1;
    const cudaError_t error = cudaGetDevice(&current);
    if (error != cudaSuccess) {
        return Status(StatusCode::DEVICE_ERROR,
                      "cuda-like current-device query failed: " +
                      std::string(cudaGetErrorString(error)));
    }
    device = static_cast<std::int32_t>(current);
    return Status::Ok();
#endif
}

inline Status device_guard_set_current(void*, std::int32_t device) {
#if defined(TUTTI_USE_HOST)
    (void)device;
    return Status(StatusCode::UNSUPPORTED,
                  "HOST profile has no current accelerator");
#else
    const cudaError_t error = cudaSetDevice(static_cast<int>(device));
    if (error != cudaSuccess) {
        return Status(StatusCode::DEVICE_ERROR,
                      "cuda-like current-device selection failed: " +
                      std::string(cudaGetErrorString(error)));
    }
    return Status::Ok();
#endif
}

inline const testing::DeviceGuardBackend& device_guard_backend() {
    static const testing::DeviceGuardBackend backend{
        nullptr, &device_guard_get_current, &device_guard_set_current};
    return backend;
}

} // namespace detail

enum class DeviceGuardState {
    INACTIVE,
    ACTIVE,
    ENTER_FAILED,
    RESTORED,
    RESTORE_FAILED,
};

class DeviceGuard {
public:
    explicit DeviceGuard(std::int32_t target_accel_id) noexcept
        : DeviceGuard(target_accel_id, detail::device_guard_backend()) {}

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    ~DeviceGuard() noexcept {
        // Explicit restore is the normal path.  The destructor is only a
        // no-throw best-effort fallback, so a restore error is retained in
        // state/diagnostic but cannot escape during stack unwinding.
        if (state_ == DeviceGuardState::ACTIVE) {
            (void)restore();
        } else if (state_ == DeviceGuardState::RESTORE_FAILED &&
                   previous_accel_id_ >= 0) {
            // Preserve the explicit restore error for the caller, but make
            // one final no-throw attempt before the guard disappears.
            (void)backend_->set_device(backend_->context, previous_accel_id_);
        }
    }

    bool ok() const noexcept {
        return state_ == DeviceGuardState::ACTIVE ||
               state_ == DeviceGuardState::RESTORED;
    }

    bool host_noop() const noexcept {
#if defined(TUTTI_USE_HOST)
        return true;
#else
        return false;
#endif
    }

    DeviceGuardState state() const noexcept { return state_; }
    std::int32_t target_accel_id() const noexcept { return target_accel_id_; }
    std::int32_t previous_accel_id() const noexcept { return previous_accel_id_; }
    const Status& status() const noexcept { return status_; }
    const std::string& diagnostic() const noexcept { return status_.message(); }

    // Restore the caller's current device.  This is idempotent after a
    // successful restore and returns the first deterministic error otherwise.
    Status restore() noexcept {
        if (state_ == DeviceGuardState::RESTORED) {
            return Status::Ok();
        }
        if (state_ != DeviceGuardState::ACTIVE) {
            return status_;
        }

#if defined(TUTTI_USE_HOST)
        state_ = DeviceGuardState::RESTORED;
        status_ = Status::Ok();
        return status_;
#else
        if (previous_accel_id_ < 0) {
            state_ = DeviceGuardState::RESTORED;
            status_ = Status::Ok();
            return status_;
        }
        Status restore_status =
            backend_->set_device(backend_->context, previous_accel_id_);
        if (!restore_status.ok()) {
            state_ = DeviceGuardState::RESTORE_FAILED;
            status_ = Status(
                restore_status.code(),
                "device guard: restore current accelerator " +
                std::to_string(previous_accel_id_) + " failed: " +
                restore_status.message());
            return status_;
        }
        state_ = DeviceGuardState::RESTORED;
        status_ = Status::Ok();
        return status_;
#endif
    }

private:
    friend struct testing::DeviceGuardTestAccess;

    DeviceGuard(std::int32_t target_accel_id,
                const testing::DeviceGuardBackend& backend) noexcept
        : target_accel_id_(target_accel_id), backend_(&backend) {
        enter_();
    }

    void enter_() noexcept {
        if (target_accel_id_ < -1) {
            state_ = DeviceGuardState::ENTER_FAILED;
            status_ = Status(StatusCode::INVALID_ARGUMENT,
                              "device guard: accel_id must be -1 or non-negative");
            return;
        }

        // -1 is the public sentinel for a host-only/optional accelerator
        // scope.  In particular, never pass it to cudaSetDevice().
        if (target_accel_id_ == -1) {
            state_ = DeviceGuardState::ACTIVE;
            status_ = Status::Ok();
            return;
        }

#if defined(TUTTI_USE_HOST)
        // HOST has no current accelerator.  Do not call the shim's synthetic
        // cudaSetDevice: the no-op must never perturb caller-visible state.
        state_ = DeviceGuardState::ACTIVE;
        status_ = Status::Ok();
#else
        Status get_status =
            backend_->get_device(backend_->context, previous_accel_id_);
        if (!get_status.ok()) {
            state_ = DeviceGuardState::ENTER_FAILED;
            status_ = Status(
                get_status.code(),
                "device guard: query current accelerator failed: " +
                get_status.message());
            return;
        }
        if (target_accel_id_ != previous_accel_id_) {
            Status set_status =
                backend_->set_device(backend_->context, target_accel_id_);
            if (!set_status.ok()) {
                state_ = DeviceGuardState::ENTER_FAILED;
                status_ = Status(
                    set_status.code(),
                    "device guard: select accelerator " +
                    std::to_string(target_accel_id_) + " failed: " +
                    set_status.message());
                return;
            }
        }
        state_ = DeviceGuardState::ACTIVE;
        status_ = Status::Ok();
#endif
    }

    std::int32_t target_accel_id_ = -1;
    std::int32_t previous_accel_id_ = -1;
    DeviceGuardState state_ = DeviceGuardState::INACTIVE;
    Status status_;
    const testing::DeviceGuardBackend* backend_ = nullptr;
};

using AcceleratorDeviceGuard = DeviceGuard;

namespace testing {

struct DeviceGuardTestAccess {
    static DeviceGuard create(std::int32_t target_accel_id,
                              const DeviceGuardBackend& backend) {
        return DeviceGuard(target_accel_id, backend);
    }
};

} // namespace testing

} // namespace tutti
