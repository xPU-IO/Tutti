#include "tutti/resolvers/resolver_factory.h"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tutti/resolvers/local_file/resolver.h>
#include <tutti/resolvers/striped_file/resolver.h>

#include "tutti/resolvers/memfs/resolver.h"
#include "tutti/resource/memory/memory_resource.h"
#include "tutti/resource/nvme/nvme_resource.h"

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
                 std::unique_ptr<const tutti::ResourceView> resolver_view)
        : id_(std::move(id)),
          type_(std::move(type)),
          resolver_view_(std::move(resolver_view)) {}

    const tutti::ResourceCapabilities& capabilities() const override {
        capabilities_.resource_type = type_;
        capabilities_.provides_resolver_view = true;
        return capabilities_;
    }

    tutti::Status initialize() override { return tutti::Status::Ok(); }
    tutti::Status shutdown() override { return tutti::Status::Ok(); }
    tutti::ResourceInfo info() const override {
        return {id_, type_, tutti::ResourceState::INITIALIZED};
    }

    tutti::Result<std::unique_ptr<const tutti::ResourceView>>
    get_resolver_view() const override {
        if (const auto* nvme = dynamic_cast<const
                tutti::resources::nvme::NvmeResolverResourceView*>(
                    resolver_view_.get())) {
            return std::unique_ptr<const tutti::ResourceView>(
                new tutti::resources::nvme::NvmeResolverResourceView(*nvme));
        }
        if (const auto* memory = dynamic_cast<const
                tutti::resources::memory::MemoryResourceView*>(
                    resolver_view_.get())) {
            return std::unique_ptr<const tutti::ResourceView>(
                new tutti::resources::memory::MemoryResourceView(*memory));
        }
        return tutti::Result<std::unique_ptr<const tutti::ResourceView>>::Failure(
            tutti::Status(tutti::StatusCode::NOT_READY,
                          "resolver view is unavailable"));
    }

    tutti::Result<std::unique_ptr<const tutti::ResourceView>>
    get_datapath_view() const override {
        return tutti::Result<std::unique_ptr<const tutti::ResourceView>>::Failure(
            tutti::Status(tutti::StatusCode::UNSUPPORTED,
                          "unused DataPath view"));
    }

private:
    std::string id_;
    std::string type_;
    mutable tutti::ResourceCapabilities capabilities_;
    std::unique_ptr<const tutti::ResourceView> resolver_view_;
};

tutti::config::BackendSpec relation(std::string contract,
                                    std::string resolver,
                                    std::string datapath,
                                    std::string resource) {
    tutti::config::BackendSpec result;
    result.id = "backend0";
    result.contract = std::move(contract);
    result.resolver = std::move(resolver);
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

std::unique_ptr<const tutti::ResourceView> nvme_view(std::size_t count) {
    auto view = std::make_unique<
        tutti::resources::nvme::NvmeResolverResourceView>();
    for (std::size_t index = 0; index < count; ++index) {
        view->slices.push_back({
            static_cast<std::int32_t>(index),
            "0000:0" + std::to_string(index) + ":00.0",
            "/dev/snvme" + std::to_string(index) + "n1",
            "/mnt/nvme" + std::to_string(index),
            "/mnt/gpu0/ssnvme" + std::to_string(index),
            1,
            4096,
        });
    }
    return view;
}

void test_local_file_creation() {
    ViewResource resource("nvme0", "nvme", nvme_view(1));
    tutti::config::ResolverSpec spec{
        "resolver0", "local-file", "file",
        tutti::config::LocalFileResolverConfig{}};
    auto backend = relation("ext4-local-nvme", "resolver0",
                            "configured-local-dp", "nvme0");

    auto created = tutti::resolvers::create_resolver(
        spec, {resource, backend, "configured-local-dp"});
    CHECK(created.ok(), "local-file factory should accept one NVMe slice");
    CHECK(created.ok() && dynamic_cast<
              tutti::resolvers::local_file::LocalFileResolver*>(
                  created.value().get()) != nullptr,
          "local-file factory should construct LocalFileResolver");
}

void test_striped_file_creation() {
    ViewResource resource("nvme0", "nvme", nvme_view(2));
    tutti::config::ResolverSpec spec{
        "resolver0", "striped-file", "striped",
        tutti::config::StripedFileResolverConfig{}};
    auto backend = relation("striped-local-nvme", "resolver0",
                            "configured-striped-dp", "nvme0");

    auto created = tutti::resolvers::create_resolver(
        spec, {resource, backend, "configured-striped-dp"});
    CHECK(created.ok(), "striped-file factory should accept two NVMe slices");
    CHECK(created.ok() && dynamic_cast<
              tutti::resolvers::striped_file::StripedResolver*>(
                  created.value().get()) != nullptr,
          "striped-file factory should construct StripedResolver");
}

void test_memfs_creation_and_instance_key() {
    auto view = std::make_unique<
        tutti::resources::memory::MemoryResourceView>();
    view->capacity_bytes = 4096;
    ViewResource resource("memory0", "memory", std::move(view));
    tutti::config::ResolverSpec spec{
        "resolver0", "memfs", "memfs",
        tutti::config::MemfsResolverConfig{}};
    auto backend = relation("memfs", "resolver0",
                            "configured-memfs-dp", "memory0");

    auto created = tutti::resolvers::create_resolver(
        spec, {resource, backend, "configured-memfs-dp"});
    CHECK(created.ok(), "memfs factory should accept memory Resource view");
    if (!created.ok()) return;

    tutti::ResolveOptions options;
    options.scheme = "memfs";
    auto resolved = created.value()->resolve("memfs://1024", options);
    CHECK(resolved.ok(), "created memfs resolver should resolve within capacity");
    CHECK(resolved.ok() &&
              resolved.value().recommended_data_path_key() ==
                  "configured-memfs-dp",
          "resolver should emit configured DataPath instance key");

    auto too_large = created.value()->resolve("memfs://8192", options);
    CHECK(!too_large.ok() &&
              too_large.status().code() ==
                  tutti::StatusCode::RESOURCE_EXHAUSTED,
          "memfs resolver should enforce memory Resource capacity");
}

void test_wrong_view_rejected() {
    auto view = std::make_unique<
        tutti::resources::memory::MemoryResourceView>();
    view->capacity_bytes = 4096;
    ViewResource resource("memory0", "memory", std::move(view));
    tutti::config::ResolverSpec spec{
        "resolver0", "local-file", "file",
        tutti::config::LocalFileResolverConfig{}};
    auto backend = relation("ext4-local-nvme", "resolver0",
                            "local-dp", "memory0");

    auto created = tutti::resolvers::create_resolver(
        spec, {resource, backend, "local-dp"});
    CHECK(!created.ok() &&
              created.status().code() == tutti::StatusCode::INVALID_ARGUMENT,
          "local-file factory should reject memory Resource view");
}

} // namespace

int main() {
    test_local_file_creation();
    test_striped_file_creation();
    test_memfs_creation_and_instance_key();
    test_wrong_view_rejected();
    if (failures == 0) {
        std::puts("resolver factory tests passed");
        return 0;
    }
    std::fprintf(stderr, "%d resolver factory test(s) failed\n", failures);
    return 1;
}
