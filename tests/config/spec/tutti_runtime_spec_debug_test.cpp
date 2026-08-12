#include <tutti/config/tutti_runtime_spec.h>

#include <cstdio>
#include <string>

int main() {
    using namespace tutti::config;
    TuttiRuntimeSpec spec;
    spec.accelerator.profile = "CUDA";
    spec.runtime.accel_id = 0;
    NvmeResourceConfig resource;
    resource.provider = {"nvme-service", "endpoint"};
    resource.allocation.selection = NvmeSelection::Striped;
    resource.allocation.device_ids = {0, 1};
    resource.allocation.queues_per_controller = 4;
    spec.storage.resources.push_back({"nvme", "nvme", resource});
    spec.storage.resolvers.push_back(
        {"resolver", "striped-file", "striped", StripedFileResolverConfig{}});
    spec.storage.datapaths.push_back(
        {"datapath", "striped-local-nvme", StripedLocalNvmeDataPathConfig{}});
    spec.storage.backends.push_back(
        {"backend", "striped-local-nvme", "resolver", "datapath", "nvme",
         StripedLocalNvmeBackendConfig{}});

    auto debug = spec.to_debug_string();
    if (!debug.ok()) return 1;
    const std::string expected =
        "accelerator.profile = \"CUDA\"\n"
        "runtime.accel_id = 0\n"
        "storage.resources[0].id = \"nvme\"\n"
        "storage.resources[0].type = \"nvme\"\n"
        "storage.resources[0].provider.type = \"nvme-service\"\n"
        "storage.resources[0].provider.endpoint = \"endpoint\"\n"
        "storage.resources[0].allocation.selection = \"striped\"\n"
        "storage.resources[0].allocation.device_ids = [0, 1]\n"
        "storage.resources[0].allocation.queues_per_controller = 4\n"
        "storage.resolvers[0].id = \"resolver\"\n"
        "storage.resolvers[0].type = \"striped-file\"\n"
        "storage.resolvers[0].scheme = \"striped\"\n"
        "storage.datapaths[0].id = \"datapath\"\n"
        "storage.datapaths[0].type = \"striped-local-nvme\"\n"
        "storage.datapaths[0].config.handle_cache_capacity = 0\n"
        "storage.datapaths[0].config.prp_cache_capacity = 0\n"
        "storage.datapaths[0].config.handle_cache_l2_capacity = 0\n"
        "storage.datapaths[0].config.threads_per_block = 16\n"
        "storage.datapaths[0].config.max_in_flight_operations = 0\n"
        "storage.datapaths[0].config.max_batch_entries = 0\n"
        "storage.datapaths[0].config.io_granularity = 0\n"
        "storage.backends[0].id = \"backend\"\n"
        "storage.backends[0].contract = \"striped-local-nvme\"\n"
        "storage.backends[0].resolver = \"resolver\"\n"
        "storage.backends[0].datapath = \"datapath\"\n"
        "storage.backends[0].resource = \"nvme\"\n"
        "storage.backends[0].config.stripe_unit = 524288\n";
    if (debug.value() != expected) {
        std::fprintf(stderr, "unexpected debug output:\n%s", debug.value().c_str());
        return 1;
    }
    spec.storage.backends.clear();
    if (spec.to_debug_string().ok()) return 1;
    std::puts("spec debug tests: PASS");
    return 0;
}
