// tests/mount_manager_contract/mount_manager_contract_test.cpp
// Round 17 S1: host-side unit tests for MountManager.
//
// These tests do NOT touch the running daemon or real NVMe devices.
// They verify:
//   1. config parsing (auto_mount field, unmount_retry block, defaults)
//   2. controller-reported block-size uniformity policy
//   3. is_mounted() against /proc/self/mountinfo
//   4. scan_holders() finds the test process itself when it holds an fd
//   5. mount_one() recursively prepares a configured mount point
//   6. mount_one() + unmount_all() lifecycle using a loopback tmpfs
//   7. force_exit_requested() short-circuits the retry loop
//   8. path_is_prefix() edge cases (not directly exposed, but tested
//      indirectly via scan_holders with paths that share prefixes)
//   9. scan_holders() finds a child process cwd

#include "mount_manager.h"
#include "nvmeservice_config.h"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <string>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while(0)

// =====================================================================
// Test 1: config parsing — auto_mount field + unmount_retry
// =====================================================================
static void test_config_parsing() {
    std::fprintf(stderr, "=== Test 1: config parsing ===\n");

    // Write a minimal YAML to a temp file.
    const char* yaml =
        "grpc:\n"
        "  endpoint: \"127.0.0.1:50051\"\n"
        "gpus:\n"
        "  - id: 0\n"
        "    mount_path: \"/mnt/gpu0\"\n"
        "nvmes:\n"
        "  - pci_addr: \"0000:08:00.0\"\n"
        "    mount_path: \"/mnt/nvme1\"\n"
        "    namespace_id: 1\n"
        "    auto_mount: true\n"
        "  - pci_addr: \"0000:4b:00.0\"\n"
        "    mount_path: \"/mnt/nvme2\"\n"
        "    namespace_id: 1\n"
        "    auto_mount: false\n"
        "queue_pool:\n"
        "  default_per_client: 4\n"
        "  max_per_client: 16\n"
        "lease:\n"
        "  heartbeat_interval_sec: 10\n"
        "  timeout_sec: 30\n"
        "unmount_retry:\n"
        "  interval_ms: 500\n"
        "  max: 10\n";
    char tmpl[] = "/tmp/tutti_cfgXXXXXX";
    int fd = ::mkstemp(tmpl);
    CHECK(fd >= 0, "mkstemp");
    ::write(fd, yaml, std::strlen(yaml));
    ::close(fd);

    std::string err;
    auto cfg = nvmeservice::parse_config_file(tmpl, &err);
    ::unlink(tmpl);

    CHECK(cfg.has_value(), "config parse should succeed");
    if (!cfg) return;

    CHECK(cfg->nvmes.size() == 2, "two nvmes");
    CHECK(cfg->nvmes[0].auto_mount == true, "nvme0 auto_mount=true");
    CHECK(cfg->nvmes[1].auto_mount == false, "nvme1 auto_mount=false");
    CHECK(cfg->unmount_retry.interval_ms == 500, "unmount_retry.interval_ms=500");
    CHECK(cfg->unmount_retry.max == 10, "unmount_retry.max=10");
}

// =====================================================================
// Test 2: config defaults — auto_mount defaults to true when absent
// =====================================================================
static void test_config_defaults() {
    std::fprintf(stderr, "=== Test 2: config defaults ===\n");

    const char* yaml =
        "grpc:\n"
        "  endpoint: \"127.0.0.1:50051\"\n"
        "gpus:\n"
        "  - id: 0\n"
        "    mount_path: \"/mnt/gpu0\"\n"
        "nvmes:\n"
        "  - pci_addr: \"0000:08:00.0\"\n"
        "    mount_path: \"/mnt/nvme1\"\n"
        "    namespace_id: 1\n"
        "lease:\n"
        "  heartbeat_interval_sec: 10\n"
        "  timeout_sec: 30\n";
    char tmpl[] = "/tmp/tutti_cfg2XXXXXX";
    int fd = ::mkstemp(tmpl);
    ::write(fd, yaml, std::strlen(yaml));
    ::close(fd);

    std::string err;
    auto cfg = nvmeservice::parse_config_file(tmpl, &err);
    ::unlink(tmpl);

    CHECK(cfg.has_value(), "config parse should succeed");
    if (!cfg) return;

    // auto_mount absent → default true
    CHECK(cfg->nvmes[0].auto_mount == true, "auto_mount defaults to true");
    // unmount_retry absent → defaults
    CHECK(cfg->unmount_retry.interval_ms == 1000, "default interval_ms=1000");
    CHECK(cfg->unmount_retry.max == 30, "default max=30");
    CHECK(cfg->queue_pool.default_per_client == 4,
          "default queue_pool.default_per_client=4");
    CHECK(cfg->queue_pool.max_per_client == 32,
          "default queue_pool.max_per_client=32");
}

// =====================================================================
// Test 3: block-size policy — uniformity is required, 4 KiB is a warning
// =====================================================================
static void test_block_size_policy() {
    std::fprintf(stderr, "=== Test 3: block-size policy ===\n");

    std::string error;
    CHECK(nvmeservice::validate_uniform_block_size({4096, 4096}, &error),
          "uniform 4 KiB block sizes are accepted");
    CHECK(nvmeservice::validate_uniform_block_size({512, 512}, &error),
          "uniform non-4 KiB block sizes are accepted for warning-only policy");

    error.clear();
    CHECK(!nvmeservice::validate_uniform_block_size({4096, 512}, &error),
          "mixed block sizes are rejected");
    CHECK(error.find("not uniform") != std::string::npos,
          "mixed block size error identifies uniformity violation");

    error.clear();
    CHECK(!nvmeservice::validate_uniform_block_size({}, &error),
          "missing block sizes are rejected");
}

// =====================================================================
// Test 4: is_mounted — /proc/self/mountinfo should list /proc, /sys, etc.
// =====================================================================
static void test_is_mounted() {
    std::fprintf(stderr, "=== Test 4: is_mounted ===\n");

    // /proc is always mounted on Linux.
    CHECK(nvmeservice::MountManager::is_mounted("/proc"), "/proc is mounted");
    CHECK(nvmeservice::MountManager::is_mounted("/sys"), "/sys is mounted");
    CHECK(!nvmeservice::MountManager::is_mounted("/nonexistent_mount_xyz"),
          "nonexistent path is not mounted");
    // Edge case: path with similar prefix should not match.
    // /proc exists; /pro shouldn't match.
    CHECK(!nvmeservice::MountManager::is_mounted("/pro"), "/pro is not mounted");
}

// =====================================================================
// Test 5: scan_holders — the test process holds an fd on /tmp
// =====================================================================
static void test_scan_holders_self() {
    std::fprintf(stderr, "=== Test 5: scan_holders (self) ===\n");

    // Open a file under /tmp and keep the fd open while scanning.
    char tmpl[] = "/tmp/tutti_holderXXXXXX";
    int fd = ::mkstemp(tmpl);
    CHECK(fd >= 0, "mkstemp for holder test");

    // Scan /tmp for holders — our own process should show up.
    auto holders = nvmeservice::MountManager::scan_holders("/tmp");

    bool found_self = false;
    for (const auto& h : holders) {
        if (h.pid == static_cast<uint32_t>(::getpid())) {
            found_self = true;
            CHECK(h.holder_type == "fd", "self holder type is fd");
            break;
        }
    }
    CHECK(found_self, "scan_holders finds self holding /tmp fd");

    ::close(fd);
    ::unlink(tmpl);
}

// =====================================================================
// Test 6: mount_one owns recursive mount-point preparation
// =====================================================================
static void test_recursive_mount_point_creation() {
    std::fprintf(stderr, "=== Test 6: recursive mount-point creation ===\n");

    char base_template[] = "/tmp/tutti_mount_pathXXXXXX";
    char* base_dir = ::mkdtemp(base_template);
    CHECK(base_dir != nullptr, "mkdtemp for recursive mount point");
    if (!base_dir) return;

    const fs::path mount_path = fs::path(base_dir) / "nested" / "nvme0";
    nvmeservice::UnmountRetryConfig rc;
    nvmeservice::MountManager mgr(rc);

    // The device intentionally does not exist.  mount(2) must fail, but the
    // complete configured mount-point hierarchy must already have been made.
    auto mr = mgr.mount_one("/dev/tutti_missing_block_device", mount_path.string());
    CHECK(!mr.error.empty(), "missing block device fails mount");
    CHECK(fs::is_directory(mount_path), "mount_one creates nested mount point");
    CHECK(!mr.mounted_by_daemon, "failed mount is not daemon-owned");

    std::error_code ec;
    fs::remove_all(base_dir, ec);
    CHECK(!ec, "remove recursive mount-point fixture");
}

// =====================================================================
// Test 7: mount_one + unmount_all lifecycle with tmpfs
// =====================================================================
static void test_mount_lifecycle() {
    std::fprintf(stderr, "=== Test 7: mount lifecycle (tmpfs) ===\n");

    // Create a directory to use as mount point.
    char mnt_template[] = "/tmp/tutti_mntXXXXXX";
    char* mnt_dir = ::mkdtemp(mnt_template);
    CHECK(mnt_dir != nullptr, "mkdtemp for mount point");
    if (!mnt_dir) return;

    // Mount a tmpfs (not ext4, but tests the lifecycle without a real block dev).
    nvmeservice::UnmountRetryConfig rc;
    rc.interval_ms = 100;
    rc.max = 3;
    nvmeservice::MountManager mgr(rc);

    // We can't use mount_one() because it hardcodes "ext4"; instead
    // mount a tmpfs directly and record it as "owned" by using the
    // internal mechanism.  But mount_one() is the public API...
    // Workaround: mount tmpfs directly, then simulate "ownership" by
    // calling unmount_all() which should cleanly umount it (since
    // we're the only one holding it).
    //
    // Actually, mount_one() records ownership only if mount(2) succeeds.
    // Let's use a loop device or just test the unmount path with tmpfs:
    int mrc = ::mount("tmpfs", mnt_dir, "tmpfs", 0, nullptr);
    if (mrc != 0 && (errno == EPERM || errno == EACCES)) {
        std::fprintf(stderr,
                     "SKIP: tmpfs mount requires CAP_SYS_ADMIN in this environment\n");
        ::rmdir(mnt_dir);
        return;
    }
    CHECK(mrc == 0, "mount tmpfs");
    if (mrc != 0) { ::rmdir(mnt_dir); return; }

    // mount_one() must REFUSE this path, not adopt it.
    //
    // The path holds a tmpfs while we were asked to mount ext4, so it is the
    // exact shape of the bug this asserts against: adopting any pre-existing
    // mount means the daemon then publishes accelerator views on a filesystem
    // that is not the target device.
    //
    // /dev/null is passed as the device so stat() succeeds and the *filesystem
    // type* is what rejects the mount; with a nonexistent path the check would
    // stop earlier, at "cannot verify", and prove less.
    auto mr = mgr.mount_one("/dev/null", mnt_dir);
    CHECK(!mr.already_mounted, "mount_one does NOT adopt a wrong-fstype mount");
    CHECK(!mr.mounted_by_daemon, "mount_one takes no ownership of it");
    CHECK(!mr.error.empty(), "mount_one reports why it refused");
    CHECK(mr.error.find("tmpfs") != std::string::npos,
          "refusal names the actual filesystem type");

    // unmount_all() should do nothing (no owned mounts).
    int remaining = mgr.unmount_all();
    CHECK(remaining == 0, "unmount_all returns 0 (no owned mounts)");

    // Clean up the tmpfs ourselves.
    CHECK(::umount2(mnt_dir, 0) == 0, "manual umount tmpfs");
    ::rmdir(mnt_dir);
}

// =====================================================================
// Test 8: force_exit short-circuits retry loop
// =====================================================================
static void test_force_exit() {
    std::fprintf(stderr, "=== Test 8: force_exit ===\n");

    nvmeservice::UnmountRetryConfig rc;
    rc.interval_ms = 10000;  // long sleep so force_exit triggers during wait
    rc.max = 100;
    nvmeservice::MountManager mgr(rc);

    // Without any owned mounts, unmount_all() returns 0 immediately.
    // To test force_exit properly we'd need a busy mount, which requires
    // a child process.  Instead, just verify the flag mechanism.
    CHECK(!mgr.force_exit_requested(), "force_exit initially false");
    mgr.request_force_exit();
    CHECK(mgr.force_exit_requested(), "force_exit true after request");
}

// =====================================================================
// Test 9: scan_holders with cwd — child process cd's into a mount
// =====================================================================
static void test_scan_holders_cwd() {
    std::fprintf(stderr, "=== Test 9: scan_holders (cwd) ===\n");

    // Fork a child that cd's into /tmp and sleeps.
    pid_t pid = ::fork();
    if (pid == 0) {
        // Child: cd to /tmp and sleep.
        ::chdir("/tmp");
        ::sleep(5);
        ::_exit(0);
    }

    // Parent: wait a moment for child to cd, then scan.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto holders = nvmeservice::MountManager::scan_holders("/tmp");
    bool found_child = false;
    for (const auto& h : holders) {
        if (h.pid == static_cast<uint32_t>(pid) && h.holder_type == "cwd") {
            found_child = true;
            break;
        }
    }
    CHECK(found_child, "scan_holders finds child process cwd");

    // Kill the child.
    ::kill(pid, SIGKILL);
    ::waitpid(pid, nullptr, 0);
}

// =====================================================================
// Test 10: existing_mount_acceptable — the adoption decision, in isolation.
//
// Pure, hence no privileges needed: the point is that a tmpfs or a foreign
// block device must never be adopted, and proving that should not require
// CAP_SYS_ADMIN to actually mount one. Test 7 covers the integrated path when
// the environment permits it.
// =====================================================================
static void test_existing_mount_decision() {
    std::fprintf(stderr, "=== Test 10: existing-mount decision ===\n");

    const auto make = [](const char* fs_type, const char* source,
                         unsigned long major, unsigned long minor) {
        nvmeservice::MountEntry e;
        e.mount_point = "/mnt/nvme0";
        e.fs_type = fs_type;
        e.source = source;
        e.major = major;
        e.minor = minor;
        return e;
    };

    // The intended mount: ext4 from the very device we asked for.
    std::string reason;
    CHECK(nvmeservice::existing_mount_acceptable(
              make("ext4", "/dev/snvme0n1", 259, 1), "ext4", 259, 1, &reason),
          "matching ext4 mount is accepted");

    // Wrong filesystem type — tmpfs, overlay, a leftover xfs, whatever.
    reason.clear();
    CHECK(!nvmeservice::existing_mount_acceptable(
              make("tmpfs", "tmpfs", 0, 55), "ext4", 259, 1, &reason),
          "tmpfs is refused");
    CHECK(reason.find("tmpfs") != std::string::npos,
          "refusal names the filesystem type it found");

    // Right filesystem, WRONG device: this is the case a path-only check cannot
    // see, and the reason the comparison uses major:minor rather than the
    // source string. A stale mount from a previous disk layout looks perfectly
    // healthy by path.
    reason.clear();
    CHECK(!nvmeservice::existing_mount_acceptable(
              make("ext4", "/dev/snvme1n1", 259, 2), "ext4", 259, 1, &reason),
          "ext4 from a different device is refused");
    CHECK(reason.find("259:2") != std::string::npos,
          "refusal names the device it found");

    // Same device numbers but a different source path spelling: accepted, since
    // device identity is what the kernel mounted. Guards against "fixing" the
    // check back to string comparison.
    CHECK(nvmeservice::existing_mount_acceptable(
              make("ext4", "/dev/disk/by-id/whatever", 259, 1), "ext4", 259, 1,
              nullptr),
          "device identity decides, not the source string");

    // reason is optional.
    CHECK(!nvmeservice::existing_mount_acceptable(
              make("tmpfs", "tmpfs", 0, 1), "ext4", 259, 1, nullptr),
          "refusal works without a reason out-param");
}

// =====================================================================
// Test 11: lookup_mount parses real /proc/self/mountinfo
//
// Exercises the tokenising parser against real kernel output rather than
// fixtures: /proc is always mounted, and procfs is a case where the fs_type
// follows the variable-length optional-fields section, which is what the old
// fixed-offset scan could not have handled.
// =====================================================================
static void test_lookup_mount_real_mountinfo() {
    std::fprintf(stderr, "=== Test 11: lookup_mount on real mountinfo ===\n");

    nvmeservice::MountEntry entry;
    CHECK(nvmeservice::MountManager::lookup_mount("/proc", &entry),
          "lookup_mount finds /proc");
    CHECK(entry.mount_point == "/proc", "mount_point parsed correctly");
    CHECK(entry.fs_type == "proc", "fs_type parsed correctly");
    CHECK(entry.source == "proc", "source parsed correctly");

    // Device numbers are present (0:3 for proc in a typical namespace) and must
    // not both be unset, which is what a failed parse would leave behind.
    CHECK(entry.major != 0 || entry.minor != 0,
          "device numbers were parsed, not left at zero");

    // Agrees with is_mounted, since the latter is built on it now.
    CHECK(nvmeservice::MountManager::is_mounted("/proc"),
          "is_mounted agrees with lookup_mount");

    // A path that exists but is not a mount point.
    CHECK(!nvmeservice::MountManager::lookup_mount("/proc/self", &entry),
          "/proc/self is inside a mount, not a mount point");

    // null out-param is allowed (is_mounted relies on it).
    CHECK(nvmeservice::MountManager::lookup_mount("/proc", nullptr),
          "lookup_mount works without an out-param");
}

int main() {
    test_config_parsing();
    test_config_defaults();
    test_block_size_policy();
    test_is_mounted();
    test_scan_holders_self();
    test_recursive_mount_point_creation();
    test_mount_lifecycle();
    test_force_exit();
    test_scan_holders_cwd();
    test_existing_mount_decision();
    test_lookup_mount_real_mountinfo();

    std::fprintf(stderr, "\n=== Summary: %d passed, %d failed ===\n",
                 g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
