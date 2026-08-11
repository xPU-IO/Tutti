#include <tutti/config/storage_config.h>

#include <algorithm>
#include <array>
#include <limits>

#include <tutti/bindings/ext4_local_nvme/binding.h>
#include <tutti/bindings/striped_local_nvme/binding.h>
#include "tutti/bindings/memfs/binding.h"

namespace tutti::config {
namespace {

namespace ext4_binding = tutti::binding::ext4_local_nvme;
namespace memfs_binding = tutti::binding::memfs;
namespace striped_binding = tutti::binding::striped_local_nvme;

constexpr std::array<StorageContract, 3> kStorageContracts{{
    {
        "ext4-local-nvme",
        "local-file",
        "file",
        "local-nvme",
        "nvme",
        1,
        1,
        ext4_binding::kResolverTypeId,
        ext4_binding::kPayloadTypeId,
        ext4_binding::kPayloadApiVersion,
        ext4_binding::kRecommendedDataPathKey,
        true,
    },
    {
        "striped-local-nvme",
        "striped-file",
        "striped",
        "striped-local-nvme",
        "nvme",
        2,
        std::numeric_limits<std::size_t>::max(),
        striped_binding::kResolverTypeId,
        striped_binding::kPayloadTypeId,
        striped_binding::kPayloadApiVersion,
        striped_binding::kRecommendedDataPathKey,
        true,
    },
    {
        "memfs",
        "memfs",
        "memfs",
        "memfs",
        "memory",
        1,
        1,
        memfs_binding::kResolverTypeId,
        memfs_binding::kPayloadTypeId,
        memfs_binding::kPayloadApiVersion,
        memfs_binding::kRecommendedDataPathKey,
        false,
    },
}};

} // namespace

const StorageContract* find_storage_contract(std::string_view name) {
    const auto found = std::find_if(
        kStorageContracts.begin(), kStorageContracts.end(),
        [&](const StorageContract& contract) { return contract.name == name; });
    return found == kStorageContracts.end() ? nullptr : &*found;
}

} // namespace tutti::config
