/**
 * tutti_daemon.cpp -- Tutti NVMe service daemon entry.
 *
 * The owner half of the SERVICE_CLIENT data plane.  This process is
 * the sole owner of every NVMe controller named in sys_config.yaml:
 * it does the libnvm owner bring-up (chrdev_create + cap + bind + probe),
 * mounts each block device, publishes per-GPU view symlinks, and serves
 * gRPC so that Coordinator
 * instances running in SERVICE_CLIENT mode (see
 * coordinator/include/coordinator_config.h) can attach as libnvm
 * clients and build their own user queue groups.
 *
 * No quota ledger is maintained here -- the kernel owns user QID
 * accounting; the daemon only brokers the chrdev/bind lease.  This is
 * a thin wrapper over nvmeservice::ServiceState (R9.q1 lean: reuse the
 * SessionBroker, wrap only the entry point), giving the daemon a
 * single tutti-branded entry that lives alongside the rest of the
 * examples/ tree.
 *
 * Round 17 S1: auto-mount/unmount lifecycle.
 *   - On startup, for every NVMe with auto_mount=true, the daemon
 *     mounts the block device at mount_path (ext4) via mount(2).
 *   - On SIGTERM/SIGINT, the daemon drains gRPC, then umounts every
 *     device it mounted.  On EBUSY, it scans /proc to report holders
 *     (PID + comm + fd/maps/cwd) and retries up to unmount_retry.max
 *     times with unmount_retry.interval_ms sleep.
 *   - A second SIGTERM/SIGINT during the retry loop forces an
 *     immediate exit (mounts left in place, reported to the operator).
 *
 * Pairing:
 *   1. Start this daemon on the NVMe-owning host:
 *        sudo ./tutti_daemon --config sys_config.yaml
 *   2. Point a SERVICE_CLIENT Coordinator / e2e_smoke at it:
 *        sudo ./e2e_smoke --cuda 0 --service 127.0.0.1:50051 \
 *                         --dev-id 0 --dev-id 1
 */

#include "nvmeservice_config.h"
#include "nvmeservice_server.h"
#include "nvmeservice_state.h"
#include "mount_manager.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "tutti_verbose.h"

/*
 * Async-signal-safe shutdown handshake.  Calling grpc::Server::
 * Shutdown() (which touches an absl::Mutex) directly from a signal
 * handler is unsafe -- abseil's RAW_CHECK aborts on "illegal
 * recursion into Mutex code".  We just flip an atomic flag in the
 * handler (async-signal-safe) and let the main thread poll it and run
 * the real teardown.
 *
 * Round 17 S1: g_stop counts signals.  The FIRST signal triggers
 * graceful shutdown (gRPC drain + unmount with retry).  A SECOND
 * signal during the unmount retry loop flips g_force_exit, which
 * MountManager::unmount_all() checks at the top of each retry
 * iteration to break out immediately.
 */
static std::atomic<int> g_stop{0};           // 0=running, 1=stop requested
static std::atomic<int> g_force_exit{0};     // R17 S1: second signal
static nvmeservice::MountManager* g_mount_mgr = nullptr;  // R17 S1

static void on_signal(int /*sig*/) {
    int prev = g_stop.fetch_add(1, std::memory_order_relaxed);
    if (prev >= 1) {
        // Second or subsequent signal — force exit.
        g_force_exit.store(1, std::memory_order_relaxed);
        if (g_mount_mgr) g_mount_mgr->request_force_exit();
    }
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --config <sys_config.yaml>\n"
        "\n"
        "Brings up every NVMe named in the config as the owner and runs\n"
        "the tutti NVMe service daemon until SIGINT/SIGTERM.\n"
        "\n"
        "Round 17 S1: auto-mounts every NVMe with auto_mount=true at\n"
        "startup, and auto-unmounts on shutdown.  On EBUSY, reports\n"
        "holders (PID/comm/type) and retries up to unmount_retry.max\n"
        "times.  A second SIGTERM forces immediate exit.\n",
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
        // Lookup order: config/local_nvme_config.yaml (canonical) >
        // config/sys_config.yaml > ./sys_config.yaml (both legacy names
        // emit a deprecation warning).
        std::FILE* f = std::fopen("config/local_nvme_config.yaml", "r");
        if (f) {
            std::fclose(f);
            config_path = "config/local_nvme_config.yaml";
        } else {
            f = std::fopen("config/sys_config.yaml", "r");
            if (f) {
                std::fclose(f);
                config_path = "config/sys_config.yaml";
                std::fprintf(stderr,
                    "WARNING: config/sys_config.yaml has been renamed to "
                    "config/local_nvme_config.yaml (deprecated name in use). "
                    "Rename it or pass --config explicitly.\n");
            } else {
                f = std::fopen("sys_config.yaml", "r");
                if (f) {
                    std::fclose(f);
                    config_path = "sys_config.yaml";
                    std::fprintf(stderr,
                        "WARNING: starting daemon with sys_config.yaml from "
                        "the repository root (deprecated).  Move it to "
                        "config/local_nvme_config.yaml or pass --config "
                        "explicitly.\n");
                }
            }
        }
    }

    if (config_path.empty()) {
        std::fprintf(stderr, "Missing --config (also tried "
            "config/local_nvme_config.yaml, config/sys_config.yaml and "
            "./sys_config.yaml)\n");
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

    std::shared_ptr<nvmeservice::ServiceState> state;
    try {
        state = std::make_shared<nvmeservice::ServiceState>(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ServiceState init failed: %s\n", e.what());
        return 1;
    }

    // The per-namespace logical block size reported by the controller is the
    // source of truth.  Uniformity is enforced by ServiceState before any
    // mount or gRPC setup; warn when the uniform value differs from the
    // current 4 KiB striped assumption.
    const auto startup_devices = state->list_nvme_resources();
    const uint32_t startup_block_size = startup_devices.empty()
        ? 0 : startup_devices.front().logical_block_size;
    for (const auto& d : startup_devices) {
        if (d.logical_block_size != nvmeservice::kExpectedNvmeBlockSize) {
            std::fprintf(stderr,
                         "WARNING: NVMe device_id=%d pci=%s reports "
                         "block_size=%u bytes; expected %u bytes for "
                         "striped operation\n",
                         d.device_id, d.pci_bdf.c_str(), d.logical_block_size,
                         nvmeservice::kExpectedNvmeBlockSize);
        }
    }

    // Post-bring-up mount: the namespace block device only exists after the
    // owner has bound and probed the controller. Consume the owner-returned
    // block_path; device_id and YAML array order do not determine /dev names.
    nvmeservice::MountManager mount_mgr(cfg.unmount_retry);
    g_mount_mgr = &mount_mgr;
    for (const auto& nvme : cfg.nvmes) {
        if (!nvme.auto_mount) continue;
        const auto found = std::find_if(startup_devices.begin(), startup_devices.end(),
            [&](const auto& resource) { return resource.device_id == nvme.device_id; });
        if (found == startup_devices.end() || found->block_path.empty()) {
            std::fprintf(stderr, "warning: no bring-up block path for device_id=%d\n",
                         nvme.device_id);
            continue;
        }
        auto mr = mount_mgr.mount_one(found->block_path, nvme.backing_mount_path);
        if (!mr.error.empty()) {
            std::fprintf(stderr,
                         "warning: auto-mount %s at %s failed: %s "
                         "(continuing; device will not be auto-unmounted)\n",
                         found->block_path.c_str(), nvme.backing_mount_path.c_str(),
                         mr.error.c_str());
        }
    }

    // Publish accelerator views only after the backing mount is visible.
    // Creating ACCEL<n> before mount(2) would place it on the host filesystem
    // and the later mount would hide it.
    for (const auto& nvme : cfg.nvmes) {
        if (!nvmeservice::MountManager::is_mounted(nvme.backing_mount_path)) {
            std::fprintf(stderr,
                         "warning: %s is not mounted; accelerator views for "
                         "device_id=%d will not be published\n",
                         nvme.backing_mount_path.c_str(), nvme.device_id);
            continue;
        }
        if (!state->publish_accelerator_views(nvme.device_id)) {
            std::fprintf(stderr,
                         "warning: one or more accelerator views for device_id=%d "
                         "could not be published\n",
                         nvme.device_id);
        }
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
        state->unpublish_gpu_views();
        mount_mgr.unmount_all();
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);

    // "listening on" is always printed — the operator needs to know the
    // daemon is ready and on which port.  Owned-devices listing and
    // shutdown messages are info-level (gated by TUTTI_VERBOSE).
    std::cout << "tutti_daemon listening on " << cfg.grpc.endpoint
              << " (port " << bound_port << ")"
              << " block_size=" << startup_block_size << "\n";
    if (tutti_verbose()) {
        std::cout << "Owned devices:\n";
        for (const auto& d : state->list_nvme_resources()) {
            std::cout << "  device_id=" << d.device_id
                      << " pci="  << d.pci_bdf
                      << " chrdev=" << d.chrdev_path
                      << " block=" << d.block_path
                      << " ns="   << d.namespace_id
                      << " block_size=" << d.logical_block_size
                      << " io_qp_limit=" << d.max_user_qid
                      << " kernel_io_qps=" << d.kernel_io_qps
                      << " user_io_qps=" << d.controller_queue_capacity
                      << " max_user_qid=" << d.max_user_qid
                      << " max_q_per_grp=" << d.max_queues_per_group
                      << "\n";
        }
        std::cout.flush();
    }

    /* Poll the signal flag from the main thread (NOT the handler). */
    while (g_stop.load(std::memory_order_relaxed) == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (tutti_verbose()) {
        std::cout << "Shutting down...\n";
        std::cout.flush();
    }
    server->Shutdown();
    server->Wait();
    state->stop_reaper();

    // Symlinks and empty ACCEL<n> directories belong to the mounted
    // filesystem, so remove them before mount_manager hides that filesystem
    // again.  The destructor repeats this defensively; the operation is
    // idempotent.
    state->unpublish_gpu_views();

    // Round 17 S1: auto-unmount with EBUSY diagnostics + retry.
    if (g_force_exit.load(std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "tutti_daemon: second signal received — force-exit; "
                     "mounts left in place\n");
    } else {
        int remaining = mount_mgr.unmount_all();
        if (remaining > 0) {
            std::fprintf(stderr,
                         "tutti_daemon: %d mount(s) still busy after retries; "
                         "leaving mounted\n",
                         remaining);
        }
    }

    if (tutti_verbose()) {
        std::cout << "tutti_daemon exited cleanly.\n";
    }

    return 0;
}
