#include <tutti/config/tutti_runtime_spec.h>

#include <cstdio>

namespace {

int failures = 0;
#define CHECK(condition)                                                       \
    do { if (!(condition)) {                                                   \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                     #condition); ++failures;                                  \
    } } while (false)

tutti::config::TuttiRuntimeSpec local_spec() {
    using namespace tutti::config;
    TuttiRuntimeSpec spec;
    spec.accelerator.profile = "CUDA";
    spec.runtime.accel_id = 0;
    NvmeResourceConfig resource;
    resource.provider = {"nvme-service", "endpoint"};
    resource.allocation.selection = NvmeSelection::Explicit;
    resource.allocation.device_ids = {0};
    resource.allocation.queues_per_controller = 4;
    spec.storage.resources.push_back({"resource", "nvme", resource});
    spec.storage.resolvers.push_back(
        {"resolver", "local-file", "file", LocalFileResolverConfig{}});
    spec.storage.datapaths.push_back(
        {"datapath", "local-nvme", LocalNvmeDataPathConfig{}});
    spec.storage.backends.push_back(
        {"backend", "ext4-local-nvme", "resolver", "datapath", "resource",
         Ext4LocalNvmeBackendConfig{}});
    return spec;
}

} // namespace

int main() {
    using namespace tutti::config;
    CHECK(local_spec().validate().ok());
    {
        auto spec = local_spec();
        auto& config = std::get<LocalNvmeDataPathConfig>(
            spec.storage.datapaths.front().config);
        config.threads_per_block = 0;
        const auto status = spec.validate();
        CHECK(!status.ok());
        CHECK(status.message().find("threads_per_block must be in [1, 1024]") !=
              std::string::npos);
    }
    {
        auto spec = local_spec();
        auto& config = std::get<LocalNvmeDataPathConfig>(
            spec.storage.datapaths.front().config);
        config.threads_per_block = 1025;
        CHECK(!spec.validate().ok());
    }
    {
        auto spec = local_spec();
        spec.accelerator.profile = "cuda";
        CHECK(spec.validate().ok());
    }

    {
        auto spec = local_spec();
        auto backend = spec.storage.backends.front();
        backend.id = "backend-2";
        spec.storage.backends.push_back(std::move(backend));
        const auto status = spec.validate();
        CHECK(!status.ok());
        CHECK(status.message().find("exactly one backend") != std::string::npos);
    }
    {
        auto spec = local_spec();
        spec.storage.backends.front().resource = "missing";
        const auto status = spec.validate();
        CHECK(!status.ok());
        CHECK(status.message().find(".resource references unknown id missing") !=
              std::string::npos);
    }
    {
        auto spec = local_spec();
        spec.storage.datapaths.push_back(
            {"datapath-2", "local-nvme", LocalNvmeDataPathConfig{}});
        auto backend = spec.storage.backends.front();
        backend.id = "backend-2";
        backend.datapath = "datapath-2";
        spec.storage.backends.push_back(std::move(backend));
        const auto status = spec.validate();
        CHECK(!status.ok());
        CHECK(status.message().find("consumed by a different datapath") !=
              std::string::npos);
    }
    {
        auto spec = local_spec();
        spec.storage.backends.front().config = MemfsBackendConfig{};
        const auto status = spec.validate();
        CHECK(!status.ok());
        CHECK(status.message().find("config does not match contract") !=
              std::string::npos);
    }
    {
        auto spec = local_spec();
        auto& resource = std::get<NvmeResourceConfig>(
            spec.storage.resources.front().config);
        resource.allocation.selection = NvmeSelection::Striped;
        resource.allocation.device_ids = {0};
        spec.storage.resolvers.front() =
            {"resolver", "striped-file", "striped",
             StripedFileResolverConfig{}};
        spec.storage.datapaths.front() =
            {"datapath", "striped-local-nvme",
             StripedLocalNvmeDataPathConfig{}};
        spec.storage.backends.front().contract = "striped-local-nvme";
        spec.storage.backends.front().config =
            StripedLocalNvmeBackendConfig{};
        const auto status = spec.validate();
        CHECK(!status.ok());
        CHECK(status.message().find("at least two device_ids") !=
              std::string::npos);
    }
    {
        TuttiRuntimeSpec spec;
        spec.accelerator.profile = "HOST";
        spec.runtime.accel_id = -1;
        spec.storage.resources.push_back(
            {"memory", "memory", MemoryResourceConfig{4096}});
        spec.storage.resolvers.push_back(
            {"resolver", "memfs", "memfs", MemfsResolverConfig{}});
        spec.storage.datapaths.push_back(
            {"datapath", "memfs", MemfsDataPathConfig{}});
        spec.storage.backends.push_back(
            {"backend", "memfs", "resolver", "datapath", "memory",
             MemfsBackendConfig{}});
        CHECK(spec.validate().ok());
    }
    std::printf("spec tests: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
