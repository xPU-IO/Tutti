#include "mount_manager.h"

#include "tutti_verbose.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mntent.h>
#include <sstream>
#include <string>
#include <sys/mount.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace nvmeservice {

namespace {

// Read /proc/<pid>/comm (process name, max 15 chars + NUL).
std::string read_comm(uint32_t pid) {
    std::string path = "/proc/" + std::to_string(pid) + "/comm";
    std::ifstream f(path);
    std::string comm;
    if (f.is_open()) {
        std::getline(f, comm);
    }
    if (comm.empty()) comm = "?";
    return comm;
}

// Check if a string starts with a prefix.
bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

// Check if a path is a prefix of another path (mount_path prefix match).
// e.g. "/mnt/nvme1" is a prefix of "/mnt/nvme1/foo" but NOT of "/mnt/nvme10".
bool path_is_prefix(const std::string& path, const std::string& prefix) {
    if (!starts_with(path, prefix)) return false;
    // Ensure the char after the prefix is '/' or end-of-string, so
    // "/mnt/nvme1" doesn't match "/mnt/nvme10".
    if (path.size() == prefix.size()) return true;
    return path[prefix.size()] == '/';
}

} // namespace

MountManager::MountManager(const UnmountRetryConfig& retry_cfg)
    : retry_cfg_(retry_cfg) {}

bool MountManager::is_mounted(const std::string& mount_path) {
    // Round 17 S1: /proc/self/mountinfo format differs from /etc/mtab:
    //   mount_id parent_id major:minor root mount_point ...
    // We need field 5 (mount_point), which may contain spaces
    // encoded as \040.  Parse line-by-line.
    std::ifstream f("/proc/self/mountinfo");
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        // Skip fields 1-4 (mount_id parent_id major:minor root),
        // then field 5 is the mount point.
        // Fields are space-separated; mount point is the 5th.
        size_t pos = 0;
        for (int field = 0; field < 4; ++field) {
            pos = line.find(' ', pos);
            if (pos == std::string::npos) break;
            pos += 1;  // skip the space
        }
        if (pos == std::string::npos) continue;
        // Now extract the mount point (up to the next space).
        size_t end = line.find(' ', pos);
        std::string mp = (end == std::string::npos)
                         ? line.substr(pos)
                         : line.substr(pos, end - pos);
        // Decode octal escapes (\040 = space, etc.)
        std::string decoded;
        for (size_t i = 0; i < mp.size(); ++i) {
            if (mp[i] == '\\' && i + 3 < mp.size() &&
                mp[i+1] == '0' && mp[i+2] == '4' && mp[i+3] == '0') {
                decoded += ' ';
                i += 3;
            } else {
                decoded += mp[i];
            }
        }
        if (decoded == mount_path) return true;
    }
    return false;
}

MountResult MountManager::mount_one(const std::string& block_device,
                                    const std::string& mount_path) {
    MountResult res;
    res.block_device = block_device;

    // 1. Create the mount point, including configured parent directories.
    // ServiceState no longer creates <mount_path>/GPU<n> before mount(2), so
    // mount-point preparation belongs entirely to MountManager.
    std::error_code ec;
    std::filesystem::create_directories(mount_path, ec);
    if (ec) {
        res.error = "mkdir " + mount_path + " failed: " + ec.message();
        return res;
    }

    struct stat st;
    if (::stat(mount_path.c_str(), &st) != 0) {
        res.error = "stat " + mount_path + " failed: " +
                    std::strerror(errno);
        return res;
    }
    if (!S_ISDIR(st.st_mode)) {
        res.error = mount_path + " is not a directory";
        return res;
    }

    // 2. Check if already mounted (by a previous operator or daemon).
    if (is_mounted(mount_path)) {
        res.already_mounted = true;
        TUTTI_INFO("mount_manager: %s already mounted at %s (not taking ownership)\n",
                   block_device.c_str(), mount_path.c_str());
        return res;
    }

    // 3. mount(2) — ext4, default options.
    int rc = ::mount(block_device.c_str(), mount_path.c_str(), "ext4", 0, nullptr);
    if (rc != 0) {
        res.error = "mount(" + block_device + ", " + mount_path +
                    ", ext4) failed: " + std::strerror(errno);
        TUTTI_INFO("mount_manager: %s (continuing without mount)\n", res.error.c_str());
        return res;
    }

    // 4. Record ownership.
    res.mounted_by_daemon = true;
    owned_mounts_.push_back({block_device, mount_path});
    TUTTI_INFO("mount_manager: mounted %s at %s (owned)\n",
               block_device.c_str(), mount_path.c_str());
    return res;
}

int MountManager::try_umount_(const std::string& mount_path) {
    // MNT_FORCE would corrupt data; never use it here.  Use plain umount2.
    // If the caller wants a lazy unmount (MNT_DETACH), they can add a flag
    // later; for now plain umount2(0).
    int rc = ::umount2(mount_path.c_str(), 0);
    if (rc == 0) return 0;
    return errno;
}

std::vector<MountHolder> MountManager::scan_holders(const std::string& mount_path) {
    std::vector<MountHolder> holders;
    DIR* proc = ::opendir("/proc");
    if (!proc) return holders;

    struct dirent* de;
    while ((de = ::readdir(proc)) != nullptr) {
        // Only numeric entries are PIDs.
        bool is_pid = true;
        for (const char* p = de->d_name; *p; ++p) {
            if (*p < '0' || *p > '9') { is_pid = false; break; }
        }
        if (!is_pid) continue;

        uint32_t pid = static_cast<uint32_t>(std::strtoul(de->d_name, nullptr, 10));
        if (pid == 0) continue;

        std::string comm = read_comm(pid);

        // --- Check /proc/<pid>/fd/ for open files on the mount ---
        {
            std::string fd_dir = "/proc/" + std::to_string(pid) + "/fd";
            DIR* fdp = ::opendir(fd_dir.c_str());
            if (fdp) {
                struct dirent* fde;
                while ((fde = ::readdir(fdp)) != nullptr) {
                    if (fde->d_name[0] == '.') continue;
                    std::string link_path = fd_dir + "/" + fde->d_name;
                    char buf[4096];
                    ssize_t n = ::readlink(link_path.c_str(), buf, sizeof(buf) - 1);
                    if (n > 0) {
                        buf[n] = '\0';
                        std::string target(buf);
                        if (path_is_prefix(target, mount_path)) {
                            holders.push_back({pid, comm, "fd",
                                               "fd=" + std::string(fde->d_name) +
                                               " -> " + target});
                        }
                    }
                }
                ::closedir(fdp);
            }
        }

        // --- Check /proc/<pid>/cwd ---
        {
            std::string cwd_link = "/proc/" + std::to_string(pid) + "/cwd";
            char buf[4096];
            ssize_t n = ::readlink(cwd_link.c_str(), buf, sizeof(buf) - 1);
            if (n > 0) {
                buf[n] = '\0';
                std::string target(buf);
                if (path_is_prefix(target, mount_path)) {
                    holders.push_back({pid, comm, "cwd", target});
                }
            }
        }

        // --- Check /proc/<pid>/maps for mmap'd files on the mount ---
        {
            std::string maps_path = "/proc/" + std::to_string(pid) + "/maps";
            std::ifstream mf(maps_path);
            std::string line;
            while (std::getline(mf, line)) {
                // maps lines: addr perms offset dev inode pathname
                // The pathname is the last whitespace-separated field
                // if it exists.  Check if it starts with mount_path.
                size_t last_space = line.find_last_of(' ');
                if (last_space == std::string::npos) continue;
                std::string pathname = line.substr(last_space + 1);
                if (path_is_prefix(pathname, mount_path)) {
                    holders.push_back({pid, comm, "maps", pathname});
                    break;  // one entry per pid for maps is enough
                }
            }
        }
    }
    ::closedir(proc);
    return holders;
}

void MountManager::report_holders_(const std::string& mount_path,
                                    const std::vector<MountHolder>& holders) {
    if (holders.empty()) {
        std::fprintf(stderr,
                     "mount_manager: %s busy but no holders found in /proc\n",
                     mount_path.c_str());
        return;
    }
    std::fprintf(stderr,
                 "mount_manager: %s busy — %zu holder(s):\n",
                 mount_path.c_str(), holders.size());
    for (const auto& h : holders) {
        std::fprintf(stderr,
                     "  PID=%-8u  comm=%-16s  type=%-5s  %s\n",
                     h.pid, h.comm.c_str(), h.holder_type.c_str(),
                     h.detail.c_str());
    }
    std::fprintf(stderr,
                 "  (Close these processes or their files, then the daemon "
                 "will retry. Send SIGTERM again to force-exit without unmounting.)\n");
    std::fflush(stderr);
}

int MountManager::unmount_all() {
    int remaining = 0;

    for (const auto& m : owned_mounts_) {
        bool unmounted = false;
        for (uint32_t attempt = 0; attempt < retry_cfg_.max; ++attempt) {
            if (force_exit_requested()) {
                std::fprintf(stderr,
                             "mount_manager: force-exit requested; "
                             "leaving %s mounted\n",
                             m.mount_path.c_str());
                ++remaining;
                unmounted = true;  // stop retrying this one
                break;
            }

            int err = try_umount_(m.mount_path);
            if (err == 0) {
                TUTTI_INFO("mount_manager: unmounted %s\n", m.mount_path.c_str());
                unmounted = true;
                break;
            }

            if (err == EINVAL) {
                // Not a mount point anymore (already gone or never was).
                TUTTI_INFO("mount_manager: %s not a mount point (EINVAL), skipping\n",
                           m.mount_path.c_str());
                unmounted = true;
                break;
            }

            if (err == EBUSY) {
                auto holders = scan_holders(m.mount_path);
                report_holders_(m.mount_path, holders);
                if (attempt + 1 < retry_cfg_.max) {
                    std::fprintf(stderr,
                                 "mount_manager: retrying %s in %u ms "
                                 "(attempt %u/%u)\n",
                                 m.mount_path.c_str(), retry_cfg_.interval_ms,
                                 attempt + 2, retry_cfg_.max);
                    std::fflush(stderr);
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(retry_cfg_.interval_ms));
                }
            } else {
                // Other errno — log and skip.
                std::fprintf(stderr,
                             "mount_manager: umount %s failed: %s\n",
                             m.mount_path.c_str(), std::strerror(err));
                unmounted = true;  // can't do anything about it
                break;
            }
        }

        if (!unmounted) {
            std::fprintf(stderr,
                         "mount_manager: %s still busy after %u retries; "
                         "leaving mounted\n",
                         m.mount_path.c_str(), retry_cfg_.max);
            ++remaining;
        }
    }

    return remaining;
}

} // namespace nvmeservice
