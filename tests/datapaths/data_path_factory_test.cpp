#include "tutti/data_paths/data_path_factory.h"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "tutti/resource/memory/memory_resource.h"
#include "tutti/resource/nvme/nvme_resource.h"
#if defined(TUTTI_TEST_HAS_LOCAL_NVME)
#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include "tutti/data_paths/striped_local_nvme/striped_data_path.h"
#endif

namespace {

int failures = 0;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::fprintf(stderr, "FAIL: %s\n", message);                    \
            ++failures;                                                       \
        }                                                                     \
    } while (false)

class ViewResource final : public tutti::Resource {
public:
    ViewResource(std::string id,
                 std::string type,
                 std::unique_ptr<const tutti::ResourceView> datapath_view)
        : id_(std::move(id)),
          type_(std::move(type)),
          datapath_view_(std::move(datapath_view)) {}

    const tutti::ResourceCapabilities& capabilities() const override {
        capabilities_.resource_type = type_;
        capabilities_.provides_datapath_view = true;
        return capabilities_;
    }

    tutti::Status initialize() override { return tutti::Status::Ok(); }
    tutti::Status shutdown() override { return tutti::Status::Ok(); }
    tutti::ResourceInfo info() const override {
        return {id_, type_, tutti::ResourceState::INITIALIZED};
    }

    tutti::Result<std::unique_ptr<const tutti::ResourceView>>
    get_resolver_view() const override {
        return tutti::Result<std::unique_ptr<const tutti::ResourceView>>::Failure(
            tutti::Status(tutti::StatusCode::UNSUPPORTED,
                          "unused resolver view"));
    }

    tutti::Result<std::unique_ptr<const tutti::ResourceView>>
    get_datapath_view() const override {
        if (const auto* nvme = dynamic_cast<const
                tutti::resources::nvme::NvmeDataPathResourceView*>(
                    datapath_view_.get())) {
            return std::unique_ptr<const tutti::ResourceView>(
                new tutti::resources::nvme::NvmeDataPathResourceView(*nvme));
        }
        if (const auto* memory = dynamic_cast<const
                tutti::resources::memory::MemoryResourceView*>(
                    datapath_view_.get())) {
            return std::unique_ptr<const tutti::ResourceView>(
                new tutti::resources::memory::MemoryResourceView(*memory));
        }
        return tutti::Result<std::unique_ptr<const tutti::ResourceView>>::Failure(
            tutti::Status(tutti::StatusCode::NOT_READY,
                          "DataPath view is unavailable"));
    }

private:
    std::string id_;
    std::string type_;
    mutable tutti::ResourceCapabilities capabilities_;
    std::unique_ptr<const tutti::ResourceView> datapath_view_;
};

tutti::config::BackendSpec relation(std::string contract,
                                    std::string datapath,
                                    std::string resource) {
    tutti::config::BackendSpec result;
    result.id = "backend0";
    result.contract = std::move(contract);
    result.resolver = "resolver0";
    result.datapath = std::move(datapath);
    result.resource = std::move(resource);
    if (result.contract == "ext4-local-nvme") {
        result.config = tutti::config::Ext4LocalNvmeBackendConfig{};
    } else if (result.contract == "striped-local-nvme") {
        result.config = tutti::config::StripedLocalNvmeBackendConfig{};
    } else {
        result.config = tutti::config::MemfsBackendConfig{};
    }
    return result;
}

std::unique_ptr<const tutti::ResourceView> nvme_view(
    std::size_t count, std::uint32_t granted_queues = 4) {
    auto view = std::make_unique<
        tutti::resources::nvme::NvmeDataPathResourceView>();
    for (std::size_t index = 0; index < count; ++index) {
        view->slices.push_back({
            static_cast<std::int32_t>(index),
            "0000:0" + std::to_string(index) + ":00.0",
            0,
            "/dev/ssnvme" + std::to_string(index),
            1,
            4096,
            16384,
            128 * 1024,
            granted_queues,
        });
    }
    return view;
}

void test_memfs_creation() {
    auto view = std::make_unique<
        tutti::resources::memory::MemoryResourceView>();
    view->capacity_bytes = 4096;
    ViewResource resource("memory0", "memory", std::move(view));
    tutti::config::DataPathSpec spec{
        "configured-memfs-dp", "memfs",
        tutti::config::MemfsDataPathConfig{}};
    auto backend = relation("memfs", "configured-memfs-dp", "memory0");

    auto created = tutti::data_paths::create_data_path(
        spec, {resource, backend, -1});
    CHECK(created.ok(), "memfs DataPath factory should accept memory view");
    CHECK(created.ok() && created.value().instance != nullptr,
          "memfs DataPath factory should return an instance");
    CHECK(created.ok() &&
              created.value().instance->capabilities().name == "memfs",
          "memfs DataPath factory should construct MemfsDataPath");
    CHECK(created.ok() &&
              created.value().initialize_config.name == "memfs",
          "memfs DataPath factory should return initialize config");
}

void test_local_nvme_creation_boundary() {
    ViewResource resource("nvme0", "nvme", nvme_view(1, 64));
    tutti::config::LocalNvmeDataPathConfig config;
    config.max_batch_entries = 64;
    config.max_in_flight_operations = 4;
    config.handle_cache_capacity = 8;
    config.prp_cache_capacity = 16;
    config.handle_cache_l2_capacity = 32;
    config.threads_per_block = 64;
    tutti::config::DataPathSpec spec{
        "configured-local-dp", "local-nvme", config};
    auto backend = relation("ext4-local-nvme", "configured-local-dp", "nvme0");

    auto created = tutti::data_paths::create_data_path(
        spec, {resource, backend, 0});
#if defined(TUTTI_TEST_HAS_LOCAL_NVME)
    CHECK(created.ok(), "local-NVMe factory should construct at creation boundary");
    CHECK(created.ok() &&
              created.value().instance->capabilities().name == "local_nvme",
          "local-NVMe factory should construct LocalNvmeDataPath");
    CHECK(created.ok() &&
              created.value().initialize_config.name == "local_nvme",
          "local-NVMe factory should return initialize config");
    const auto* local = dynamic_cast<const
        tutti::data_paths::local_nvme::LocalNvmeDataPath*>(
            created.value().instance.get());
    CHECK(local != nullptr && local->test_threads_per_block() == 64,
          "local-NVMe factory should pass threads_per_block from spec");
#else
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::UNSUPPORTED,
          "local-NVMe factory should be unavailable without hardware stack");
#endif
}

void test_local_nvme_rejects_unsafe_block_size() {
    ViewResource resource("nvme0", "nvme", nvme_view(1));
    tutti::config::LocalNvmeDataPathConfig config;
    config.threads_per_block = 8;
    tutti::config::DataPathSpec spec{
        "configured-local-dp", "local-nvme", config};
    auto backend = relation("ext4-local-nvme", "configured-local-dp", "nvme0");

    auto created = tutti::data_paths::create_data_path(
        spec, {resource, backend, 0});
#if defined(TUTTI_TEST_HAS_LOCAL_NVME)
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::INVALID_ARGUMENT,
          "local-NVMe factory should reject block sizes above queue grant");
#else
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::UNSUPPORTED,
          "local-NVMe factory should be unavailable without hardware stack");
#endif
}

void test_striped_nvme_creation_boundary() {
    ViewResource resource("nvme0", "nvme", nvme_view(2, 128));
    tutti::config::StripedLocalNvmeDataPathConfig config;
    config.max_batch_entries = 64;
    config.max_in_flight_operations = 4;
    config.handle_cache_capacity = 8;
    config.prp_cache_capacity = 16;
    config.threads_per_block = 128;
    tutti::config::DataPathSpec spec{
        "configured-striped-dp", "striped-local-nvme", config};
    auto backend = relation("striped-local-nvme",
                            "configured-striped-dp", "nvme0");

    auto created = tutti::data_paths::create_data_path(
        spec, {resource, backend, 0});
#if defined(TUTTI_TEST_HAS_LOCAL_NVME)
    CHECK(created.ok(), "striped factory should construct at creation boundary");
    CHECK(created.ok() &&
              created.value().instance->capabilities().name ==
                  "striped-local-nvme",
          "striped factory should construct StripedDataPath");
    CHECK(created.ok() &&
              created.value().initialize_config.name ==
                  "striped-local-nvme",
          "striped factory should return initialize config");
    const auto* striped = dynamic_cast<const
        tutti::data_paths::striped_local_nvme::StripedDataPath*>(
            created.value().instance.get());
    CHECK(striped != nullptr && striped->test_threads_per_block() == 128,
          "striped factory should pass threads_per_block from spec");
#else
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::UNSUPPORTED,
          "striped factory should be unavailable without hardware stack");
#endif
}

void test_striped_nvme_rejects_unsafe_block_size() {
    ViewResource resource("nvme0", "nvme", nvme_view(2));
    tutti::config::StripedLocalNvmeDataPathConfig config;
    config.threads_per_block = 8;
    tutti::config::DataPathSpec spec{
        "configured-striped-dp", "striped-local-nvme", config};
    auto backend = relation("striped-local-nvme",
                            "configured-striped-dp", "nvme0");

    auto created = tutti::data_paths::create_data_path(
        spec, {resource, backend, 0});
#if defined(TUTTI_TEST_HAS_LOCAL_NVME)
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::INVALID_ARGUMENT,
          "striped factory should reject block sizes above any queue grant");
#else
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::UNSUPPORTED,
          "striped factory should be unavailable without hardware stack");
#endif
}

void test_relation_mismatch_rejected() {
    auto view = std::make_unique<
        tutti::resources::memory::MemoryResourceView>();
    view->capacity_bytes = 4096;
    ViewResource resource("memory0", "memory", std::move(view));
    tutti::config::DataPathSpec spec{
        "memfs-dp", "memfs", tutti::config::MemfsDataPathConfig{}};
    auto backend = relation("memfs", "another-dp", "memory0");

    auto created = tutti::data_paths::create_data_path(
        spec, {resource, backend, -1});
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::INVALID_ARGUMENT,
          "DataPath factory should reject relation mismatch");
}

} // namespace

int main() {
    test_memfs_creation();
    test_local_nvme_creation_boundary();
    test_local_nvme_rejects_unsafe_block_size();
    test_striped_nvme_creation_boundary();
    test_striped_nvme_rejects_unsafe_block_size();
    test_relation_mismatch_rejected();
    if (failures == 0) {
        std::puts("DataPath factory tests passed");
        return 0;
    }
    std::fprintf(stderr, "%d DataPath factory test(s) failed\n", failures);
    return 1;
}
