/**
 * nvmeservice_client.cpp -- gRPC half of the NVMeService client smoke.
 *
 * Calls Connect, prints the grant, then dispatches into
 * run_nvmeservice_client_io() (compiled in nvmeservice_client_io.cu)
 * to drive the actual libnvm + GPU IO path.  After IO it holds the
 * session for `--hold` seconds (heartbeat thread runs in background)
 * before letting the Session dtor send Disconnect.
 */

#include "nvmeservice_client.h"
#include "nvmeservice_client_io.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --endpoint <host:port>   gRPC endpoint (default 127.0.0.1:50051)\n"
        "  --device   <id>          device_id to allocate on (default 0)\n"
        "  --cuda     <id>          target cuda_device (default: first allowed)\n"
        "  --count    <n>           queues to request (0 = daemon default)\n"
        "  --hold     <sec>         hold session AFTER IO smoke (default 0)\n"
        "  --list-only              list devices and exit (no Connect)\n"
        "  --skip-io                Connect + attach + create + destroy, no IO\n"
        "  -h, --help               show this message\n",
        prog);
}

int main(int argc, char** argv) {
    std::string endpoint = "127.0.0.1:50051";
    int32_t device_id   = 0;
    int32_t cuda_device = -1;
    int32_t num_queues  = 0;
    int     hold_seconds = 0;
    bool    list_only   = false;
    bool    skip_io     = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--endpoint")       endpoint    = next("--endpoint");
        else if (a == "--device")    device_id   = std::atoi(next("--device"));
        else if (a == "--cuda")      cuda_device = std::atoi(next("--cuda"));
        else if (a == "--count")     num_queues  = std::atoi(next("--count"));
        else if (a == "--hold")      hold_seconds = std::atoi(next("--hold"));
        else if (a == "--list-only") list_only   = true;
        else if (a == "--skip-io")   skip_io     = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "Connecting to " << endpoint << " ...\n";
    nvmeservice::NvmeServiceClient client(endpoint);

    std::cout << "\n=== ListDevices ===\n";
    auto devs = client.list_devices();
    if (devs.empty()) {
        std::fprintf(stderr, "No devices returned. Is the daemon running?\n");
        return 1;
    }
    for (const auto& d : devs) {
        std::cout << "  device_id=" << d.device_id
                  << " pci=" << d.pci_addr
                  << " snvme=" << d.snvme_dev_path
                  << " ns=" << d.namespace_id
                  << " page=" << d.page_size
                  << " blk=" << d.blk_size
                  << " qdepth=" << d.queue_depth
                  << " bar0=" << d.bar0_size
                  << " max_user_qid=" << d.max_user_qid
                  << " max_q/grp=" << d.max_queues_per_group
                  << "\n";
        for (const auto& a : d.allowed_gpus) {
            std::cout << "      allowed: cuda_device=" << a.cuda_device
                      << " mount=" << (a.mount_path.empty() ? "(none)" : a.mount_path)
                      << "\n";
        }
    }

    if (list_only) return 0;

    if (cuda_device < 0) {
        for (const auto& d : devs) {
            if (d.device_id != device_id) continue;
            if (!d.allowed_gpus.empty()) cuda_device = d.allowed_gpus.front().cuda_device;
            break;
        }
        if (cuda_device < 0) {
            std::fprintf(stderr, "Could not auto-pick cuda_device on device %d\n",
                         device_id);
            return 1;
        }
    }

    std::cout << "\n=== Connect device=" << device_id
              << " cuda_device=" << cuda_device
              << " count=" << (num_queues == 0 ? "(default)"
                                                : std::to_string(num_queues))
              << " ===\n";
    auto sess = client.connect(device_id, cuda_device, num_queues);
    if (!sess) {
        std::fprintf(stderr, "connect() failed\n");
        return 1;
    }

    std::cout << "  allocation_id : " << sess->allocation_id << "\n";
    std::cout << "  device_id     : " << sess->device_id << "\n";
    std::cout << "  cuda_device   : " << sess->cuda_device << "\n";
    std::cout << "  granted_queues: " << sess->granted_queues << "\n";
    std::cout << "  snvme_dev     : " << sess->snvme_dev_path << "\n";
    std::cout << "  bar0_size     : 0x" << std::hex << sess->bar0_size << std::dec << "\n";
    std::cout << "  ns_id         : " << sess->namespace_id << "\n";
    std::cout << "  blk_size      : " << sess->blk_size << "\n";
    std::cout << "  queue_depth   : " << sess->queue_depth << "\n";
    std::cout << "  mount_path    : "
              << (sess->mount_path.empty() ? "(empty)" : sess->mount_path)
              << "\n";
    std::cout << "  heartbeat     : " << sess->heartbeat_interval_sec << "s\n";
    std::cout << "  lease timeout : " << sess->lease_timeout_sec << "s\n";

    if (!sess->mount_path.empty()) {
        std::error_code ec;
        const auto resolved = std::filesystem::read_symlink(sess->mount_path, ec);
        if (!ec) {
            std::cout << "  mount->        : " << resolved.string() << "\n";
        }
    }

    std::cout << "\n=== libnvm bring-up + GPU IO smoke ===\n";
    nvmeservice_client_io_args io_args;
    io_args.cuda_dev       = cuda_device;
    io_args.snvme_dev_path = sess->snvme_dev_path.c_str();
    io_args.bar0_size      = sess->bar0_size;
    io_args.namespace_id   = sess->namespace_id;
    io_args.blk_size       = sess->blk_size;
    io_args.queue_depth    = sess->queue_depth;
    io_args.granted_queues = sess->granted_queues;
    io_args.skip_io        = skip_io;

    int rc = run_nvmeservice_client_io(&io_args);
    if (rc != 0) {
        std::fprintf(stderr, "IO smoke failed rc=%d\n", rc);
        return rc;
    }

    if (hold_seconds > 0) {
        std::cout << "\n=== Holding session for " << hold_seconds
                  << "s (heartbeat thread running) ===\n";
        for (int i = 0; i < hold_seconds; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if ((i + 1) % 5 == 0 || i + 1 == hold_seconds) {
                std::cout << "  " << (i + 1) << "s elapsed\n";
                std::cout.flush();
            }
        }
    }

    std::cout << "\n=== Disconnect (Session dtor) ===\n";
    sess.reset();

    std::cout << "\nDone.\n";
    return 0;
}
