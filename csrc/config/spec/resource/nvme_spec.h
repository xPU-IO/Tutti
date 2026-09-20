#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tutti::config {

enum class NvmeSelection {
    Allowed,
    Explicit,
    Striped,
};

struct NvmeProviderSpec {
    std::string type;
    std::string endpoint;
};

struct NvmeAllocationSpec {
    NvmeSelection selection = NvmeSelection::Allowed;
    // NVMe 设备号（daemon 视角），不是加速器序号。配置键仍是 "device_ids"
    // （兼容既有 YAML），C++ 侧统一为 nvme_device_ids 以消除与 accel_id
    // 的歧义（命名统一 N2）。
    std::vector<std::int32_t> nvme_device_ids;
    std::int32_t queues_per_controller = 0;
};

struct NvmeResourceConfig {
    NvmeProviderSpec provider;
    NvmeAllocationSpec allocation;
};

} // namespace tutti::config
