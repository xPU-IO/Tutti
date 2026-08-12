#pragma once

// tutti/include/tutti/presets/local_nvme.h
//
// Preset assembly layer for Local NVMe + Striped NVMe DataPaths.
//
// This header is part of the public API: it exposes factory functions
// that construct a fully-assembled StorageRuntime with the appropriate
// DataPath(s) + resolver(s) injected. The factory .cpp includes private
// headers (LocalNvmeDataPath, StripedDataPath, LocalFileResolver,
// StripedResolver) but the returned types are public (StorageRuntime +
// RuntimeTelemetry).
//
// Consumers (examples, tests, applications) call make_*_runtime() and
// receive a ready-to-use StorageRuntime without referencing any private
// data-path or resolver types.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tutti/storage_runtime.h>

namespace tutti::presets {

// Telemetry counters for submit/launch instrumentation.
// The example uses these to assert "one submit() = one kernel launch"
// on the normal path.
struct RuntimeTelemetry {
    std::function<std::uint64_t()> submit_call_count;
    std::function<std::uint64_t()> kernel_launch_count;
    std::function<void()>          reset_counters;
};

// Configuration for a single NVMe device.
struct NvmeDeviceConfig {
    std::string ssnvme_path;     // e.g. "/dev/ssnvme0"
    std::string pci_bdf;         // e.g. "0000:08:00.0"
    std::string backing_device;  // e.g. "/dev/snvme0n1"
    std::string mount_path;      // e.g. "/mnt/nvme0"
    std::uint32_t namespace_id = 1;
    std::uint32_t block_size = 4096;
    static constexpr std::uint64_t kDefaultBar0 = 16384;
    std::uint64_t bar0_size = kDefaultBar0;
};

// Configuration for single-disk LocalNvmeDataPath.
struct LocalNvmePreset {
    NvmeDeviceConfig device;
    std::int32_t gpu_id = 0;
    std::uint32_t num_queues = 16;
    // Capacity knobs (0 = use defaults)
    std::uint32_t max_batch_entries = 4096;
    std::uint32_t max_in_flight_operations = 4;
    std::uint32_t threads_per_block = 16;
    std::uint32_t handle_cache_capacity = 4096;
    std::uint32_t prp_cache_capacity = 4096;
};

// Configuration for 4-disk striped mode.
struct StripedNvmePreset {
    std::vector<NvmeDeviceConfig> devices;  // typically 4
    std::int32_t gpu_id = 0;
    std::uint32_t num_queues = 32;           // per-device
    std::uint64_t stripe_unit = 524288;     // 512 KiB (tensor-aligned)
    std::uint32_t max_batch_entries = 8192;
    std::uint32_t max_in_flight_operations = 4;
    std::uint32_t threads_per_block = 16;
    std::uint32_t prp_cache_capacity = 4096;
};

// Factory: create a single-disk LocalNvmeDataPath runtime.
// Returns {runtime, telemetry}. telemetry.submit_call_count etc. are
// pre-bound to the internal DataPath.
struct RuntimeWithTelemetry {
    std::unique_ptr<StorageRuntime> runtime;
    RuntimeTelemetry telemetry;
};

RuntimeWithTelemetry make_local_nvme_runtime(const LocalNvmePreset& preset);

// Factory: create a 4-disk striped runtime.
RuntimeWithTelemetry make_striped_nvme_runtime(const StripedNvmePreset& preset);

// Helper: fill a GPU buffer with a byte pattern (DMA-visible).
// Implemented in the preset library (CUDA kernel internally).
void launch_fill_pattern(void* buf, unsigned char val, std::uint64_t n, void* stream);

// Helper: run a simple SGEMM on the GPU (for compute-IO overlap simulation).
// C = A * B, where A/B/C are n×n float matrices on the GPU.
void launch_sgemm(const float* A, const float* B, float* C, int n, int iters, void* stream);

} // namespace tutti::presets
