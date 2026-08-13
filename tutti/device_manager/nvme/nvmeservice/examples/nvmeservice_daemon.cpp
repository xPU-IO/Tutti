/**
 * nvmeservice_daemon.cpp -- NVMeService daemon entry point.
 *
 * Reads sys_config.yaml, brings up every NVMe controller described
 * there as the *owner* (libnvm: chrdev_create + cap + bind + probe),
 * publishes per-GPU view symlinks for filesystems already mounted by
 * the operator, starts the gRPC server, and runs until SIGINT/SIGTERM.
 * No quota ledger is maintained -- the kernel owns user QID accounting.
 */

#include "nvmeservice_config.h"
#include "mount_manager.h"
#include "nvmeservice_server.h"
#include "nvmeservice_state.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

/*
 * Async-signal-safe shutdown handshake.
 *
 * Calling grpc::Server::Shutdown() (or anything that touches an
 * absl::Mutex) directly from a signal handler is unsafe: the handler
 * may interrupt the main thread mid-mutex-acquire, and abseil's
 * RAW_CHECK then aborts with
 *
 *   RAW: Check ... failed: detected illegal recursion into Mutex code
 *
 * Instead we just flip an atomic flag in the handler -- write(2) to
 * an atomic<int> is async-signal-safe -- and let the main thread
 * poll it and run the actual Shutdown() / stop_reaper() teardown.
 */
static std::atomic<int> g_stop{0};

static void on_signal(int /*sig*/) {
    g_stop.store(1, std::memory_order_relaxed);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --config <sys_config.yaml>\n"
        "\n"
        "Reads the given config and runs the NVMeService daemon until\n"
        "a signal is received (SIGINT/SIGTERM).\n",
        prog);
}

int main(int argc, char** argv) {
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty()) {
        std::fprintf(stderr, "Missing --config\n");
        print_usage(argv[0]);
        return 1;
    }

    std::string parse_err;
    auto cfg_opt = nvmeservice::parse_config_file(config_path, &parse_err);
    if (!cfg_opt.has_value()) {
        std::fprintf(stderr, "Config parse failed: %s\n", parse_err.c_str());
        return 1;
    }
    const auto& cfg = cfg_opt.value();

    for (const auto& n : cfg.nvmes) {
        std::cout << "Parsed NVMe config: pci=" << n.pci_addr
                  << " backing_mount=" << n.backing_mount_path
                  << " ns=" << n.namespace_id
                  << " kernel_ioq_cap=" << n.kernel_ioq_cap
                  << " allowed_accel_ids=" << n.allowed_accel_ids.size()
                  << "\n";
    }

    {
        std::shared_ptr<nvmeservice::ServiceState> state;
        try {
            state = std::make_shared<nvmeservice::ServiceState>(cfg);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ServiceState init failed: %s\n", e.what());
            return 1;
        }

        // This example does not own the mount lifecycle. Publish only views
        // backed by an operator-prepared mount; never create ACCEL<n> on the
        // host filesystem as a substitute for a missing mount.
        for (const auto& nvme : cfg.nvmes) {
            if (!nvmeservice::MountManager::is_mounted(nvme.backing_mount_path)) {
                std::fprintf(stderr,
                    "warning: %s is not mounted; accelerator views for device_id=%d "
                    "will not be published\n",
                    nvme.backing_mount_path.c_str(), nvme.device_id);
                continue;
            }
            state->publish_accelerator_views(nvme.device_id);
        }

        state->start_reaper();

        nvmeservice::NvmeServiceImpl svc(state);

        grpc::ServerBuilder builder;
        int bound_port = 0;
        builder.AddListeningPort(cfg.grpc.endpoint,
                                grpc::InsecureServerCredentials(),
                                &bound_port);
        builder.RegisterService(&svc);

        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
        if (!server) {
            std::fprintf(stderr, "Failed to start gRPC server on %s\n",
                        cfg.grpc.endpoint.c_str());
            state->stop_reaper();
            return 1;
        }

        std::signal(SIGINT,  on_signal);
        std::signal(SIGTERM, on_signal);
        std::signal(SIGPIPE, SIG_IGN);

        std::cout << "NVMeService daemon listening on "
                << cfg.grpc.endpoint << " (port " << bound_port << ")\n";
        std::cout << "Registered devices:\n";
        for (const auto& d : state->list_nvme_resources()) {
            std::cout << "  device_id=" << d.device_id
                    << " pci=" << d.pci_bdf
                    << " chrdev=" << d.chrdev_path
                    << " block=" << d.block_path
                    << " ns=" << d.namespace_id
                    << " page=" << d.page_size
                    << " blk=" << d.logical_block_size
                    << " qdepth=" << d.queue_depth
                    << " dstrd=" << d.dstrd
                    << " bar0=" << d.bar0_size
                    << " max_user_qid=" << d.max_user_qid
                    << " max_q/grp=" << d.max_queues_per_group
                    << "\n";
            for (const auto accel_id : d.allowed_accel_ids) {
                const auto view = d.view_paths.find(accel_id);
                std::cout << "      allowed: accel_id=" << accel_id
                          << " view=" << (view == d.view_paths.end()
                              ? "(none)" : view->second) << "\n";
            }
        }
        std::cout << "lease: heartbeat=" << cfg.lease.heartbeat_interval_sec
                << "s timeout=" << cfg.lease.timeout_sec << "s\n";
        std::cout << "queue_pool: default=" << cfg.queue_pool.default_per_client
                << " max=" << cfg.queue_pool.max_per_client << "\n";
        std::cout.flush();

        /* Poll the signal flag from the main thread and call
         * Shutdown() here (NOT in the handler).  100 ms granularity
         * is plenty for an interactive daemon; tighten if you care
         * about SIGTERM-to-exit latency in tests. */
        while (g_stop.load(std::memory_order_relaxed) == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "Shutting down...\n";
        std::cout.flush();
        server->Shutdown();
        server->Wait();

        state->stop_reaper();
        state->unpublish_gpu_views();
        std::cout << "Daemon exited cleanly.\n";
    }

    return 0;
}
