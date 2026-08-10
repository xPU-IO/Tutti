/**
 * nvmeservice_client.cpp -- gRPC half of the NVMeService client smoke.
 *
 * Canonical mode lists accelerators/resources, acquires one logical
 * allocation (possibly striped), and dispatches each returned slice into the
 * libnvm + accelerator I/O bridge. The legacy --cuda option remains a thin
 * compatibility path through Connect/Disconnect.
 */

#include "nvmeservice_client.h"
#include "nvmeservice_client_io.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

void print_usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s [options]\n\n"
        "  --endpoint <host:port>\n"
        "  --accel <id>             canonical accelerator ordinal\n"
        "  --selection <mode>       allowed|explicit|striped\n"
        "  --device <id>            repeat for striped selection\n"
        "  --count <n>              queues per controller (0=default)\n"
        "  --cuda <id>              legacy Connect compatibility path\n"
        "  --hold <sec>             keep allocation alive after I/O\n"
        "  --list-only              list canonical resources and exit\n"
        "  --skip-io                attach/create/destroy only\n",
        program);
}

int run_io(int32_t accel_id, const nvmeservice::ClientNvmeSlice& slice,
           bool skip_io) {
    nvmeservice_client_io_args arguments {};
    arguments.cuda_dev = accel_id;
    arguments.snvme_dev_path = slice.chrdev_path.c_str();
    arguments.bar0_size = slice.bar0_size;
    arguments.namespace_id = slice.namespace_id;
    arguments.blk_size = slice.logical_block_size;
    arguments.queue_depth = slice.queue_depth;
    arguments.granted_queues = static_cast<int>(slice.granted_queues);
    arguments.skip_io = skip_io;
    return run_nvmeservice_client_io(&arguments);
}

} // namespace

int main(int argc, char** argv) {
    std::string endpoint = "127.0.0.1:50051";
    int32_t accel_id = -1;
    int32_t legacy_cuda_id = -1;
    int32_t queues = 0;
    int hold_seconds = 0;
    bool accel_specified = false;
    bool cuda_specified = false;
    bool list_only = false;
    bool skip_io = false;
    std::string selection_name = "allowed";
    std::vector<int32_t> device_ids;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&](const char* option) {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", option);
                std::exit(1);
            }
            return argv[++index];
        };
        if (argument == "--endpoint") endpoint = next("--endpoint");
        else if (argument == "--accel") {
            accel_id = std::atoi(next("--accel"));
            accel_specified = true;
        } else if (argument == "--cuda") {
            legacy_cuda_id = std::atoi(next("--cuda"));
            cuda_specified = true;
        } else if (argument == "--selection") {
            selection_name = next("--selection");
        } else if (argument == "--device") {
            device_ids.push_back(std::atoi(next("--device")));
        } else if (argument == "--count") {
            queues = std::atoi(next("--count"));
        } else if (argument == "--hold") {
            hold_seconds = std::atoi(next("--hold"));
        } else if (argument == "--list-only") list_only = true;
        else if (argument == "--skip-io") skip_io = true;
        else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argument.c_str());
            return 1;
        }
    }

    if (accel_specified && cuda_specified) {
        std::fprintf(stderr, "--accel and legacy --cuda cannot be combined\n");
        return 1;
    }

    nvmeservice::NvmeServiceClient client(endpoint);
    const auto accelerators = client.list_accelerators();
    const auto resources = client.list_nvme_resources();
    if (accelerators.empty() || resources.empty()) {
        std::fprintf(stderr, "daemon returned no accelerators or NVMe resources\n");
        return 1;
    }

    std::cout << "=== ListAccelerators ===\n";
    for (const auto& accelerator : accelerators) {
        std::cout << "accel_id=" << accelerator.accel_id
                  << " view_root=" << accelerator.view_root << '\n';
    }
    std::cout << "=== ListNvmeResources ===\n";
    for (const auto& resource : resources) {
        std::cout << "device_id=" << resource.device_id
                  << " pci_bdf=" << resource.pci_bdf
                  << " chrdev=" << resource.chrdev_path
                  << " block=" << resource.block_path
                  << " backing=" << resource.backing_mount_path
                  << " block_size=" << resource.logical_block_size
                  << " page_size=" << resource.page_size
                  << " queue_depth=" << resource.queue_depth
                  << " bar0=" << resource.bar0_size
                  << " capacity=" << resource.controller_queue_capacity
                  << " reserved=" << resource.reserved_queues
                  << " available_queues=" << resource.available_queues
                  << " available=" << (resource.available ? "true" : "false")
                  << " diagnostic=" << resource.diagnostic << '\n';
    }
    if (list_only) return 0;

    if (cuda_specified) {
        const int32_t device_id = device_ids.empty() ? 0 : device_ids.front();
        auto session = client.connect(device_id, legacy_cuda_id, queues);
        if (!session) return 1;
        nvmeservice::ClientNvmeSlice slice;
        slice.chrdev_path = session->snvme_dev_path;
        slice.bar0_size = session->bar0_size;
        slice.namespace_id = session->namespace_id;
        slice.logical_block_size = session->blk_size;
        slice.queue_depth = session->queue_depth;
        slice.granted_queues = static_cast<uint32_t>(session->granted_queues);
        const int status = run_io(legacy_cuda_id, slice, skip_io);
        if (status != 0) return status;
        if (hold_seconds > 0) {
            std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
        }
        return 0;
    }

    if (!accel_specified) accel_id = accelerators.front().accel_id;
    nvmeservice::ClientSelectionMode selection;
    if (selection_name == "allowed") {
        selection = nvmeservice::ClientSelectionMode::Allowed;
    } else if (selection_name == "explicit") {
        selection = nvmeservice::ClientSelectionMode::Explicit;
    } else if (selection_name == "striped") {
        selection = nvmeservice::ClientSelectionMode::Striped;
    } else {
        std::fprintf(stderr, "unknown selection: %s\n", selection_name.c_str());
        return 1;
    }

    auto allocation = client.acquire_nvme_slices(
        accel_id, selection, device_ids, queues);
    if (!allocation) return 1;
    std::cout << "allocation_id=" << allocation->allocation_id
              << " slices=" << allocation->slices.size() << '\n';
    for (const auto& slice : allocation->slices) {
        std::cout << "slice device_id=" << slice.device_id
                  << " accel_id=" << slice.accel_id
                  << " chrdev=" << slice.chrdev_path
                  << " block=" << slice.block_path
                  << " view=" << slice.view_path
                  << " granted_queues=" << slice.granted_queues << '\n';
        const int status = run_io(accel_id, slice, skip_io);
        if (status != 0) return status;
    }
    if (hold_seconds > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
    }
    allocation.reset();
    return 0;
}
