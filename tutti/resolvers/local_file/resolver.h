#pragma once

// tutti/resolvers/local_file/resolver.h
//
// Fail-closed StorageTargetResolver for controlled file targets backed by
// a local NVMe namespace.
//
// This resolver ONLY returns a ResolvedTarget when it can positively prove:
//   1. The file is a regular file (fstat S_ISREG).
//   2. The file resides on the configured backing block device (st_dev match).
//   3. All FIEMAP extents are complete, contiguous, block-aligned, and carry
//      none of the unsafe flags (UNKNOWN, DELALLOC, UNWRITTEN, ENCODED,
//      NOT_ALIGNED, SHARED, DATA_ENCRYPTED, DATA_INLINE, DATA_TAIL).
//   4. The extents fully cover [0, file_size) with no holes or overlaps.
//   5. The device offset (fe_physical + configured namespace_base) does not
//      overflow and is block-aligned.
//
// If any condition cannot be proven, resolve() returns a structured failure
// (INVALID_ARGUMENT, UNSUPPORTED, NOT_READY, DATA_LOSS, or OUT_OF_RANGE).
// There is no silent fallback to "fe_physical == namespace offset".
//
// Deployment contract:
//   The caller (deployment/operator) MUST ensure that during the handle
//   lifetime the file's layout is not mutated by truncate, hole-punch,
//   reflink, COW, defrag, or any other mechanism.  The resolver holds an
//   fd lease (cooperative only) but does NOT claim that advisory flock or
//   fd-open prevents external layout mutation.  The deployment must enforce
//   layout stability through exclusive management.
//
// The resolver does NOT submit IO, construct PRP/LBA descriptors, or
// understand NVMe queue internals.  It produces a ResolvedTarget carrying
// an Ext4LocalNvmePayload via the ext4_local_nvme binding.

#include <tutti/status.h>
#include <tutti/spi/storage_target_resolver.h>

#include <tutti/bindings/ext4_local_nvme/binding.h>

#include <linux/fiemap.h>
#include <linux/fs.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace tutti::resolvers::local_file {

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

inline constexpr std::string_view kScheme = "file";
inline constexpr std::uint32_t kFiemapMaxExtentsPerCall = 256;
inline constexpr std::uint32_t kMaxTotalExtents = 124;

// -------------------------------------------------------------------------
// BackingDeviceConfig
//
// Explicitly declares the verified backing block device and the namespace
// byte base for this resolver instance.
//
//   backing_device_path:  Required path to the block device (e.g.
//                         "/dev/snvme0n1") used to verify st_dev identity.
//                         An empty path is rejected; it would leave the
//                         FIEMAP physical-offset mapping unproven.
//
//   namespace_base_bytes: The byte offset within the NVMe namespace where
//                         the backing block device starts.  For a whole-disk
//                         ext4 filesystem on /dev/snvme0n1 with no partition,
//                         this is 0.  For a partition, this is the partition
//                         start byte.  Must be block-aligned.
//
// The resolver adds namespace_base_bytes to each extent's fe_physical to
// produce the device_offset in the payload.  This is the ONLY place where
// fe_physical is mapped to namespace offset.
// -------------------------------------------------------------------------
struct BackingDeviceConfig {
    std::string backing_device_path;
    std::uint64_t namespace_base_bytes = 0;
};

// -------------------------------------------------------------------------
// FileDescriptorLease
//
// Owner lease: holds the open fd.  Destroyed when the last shared_ptr
// reference is released.  The destructor calls close().
//
// NOTE: This fd lease is COOPERATIVE ONLY.  It does not prevent other
// processes from truncate/hole-punch/reflink/COW.  Layout stability is
// the deployment's responsibility, not the resolver's.
// -------------------------------------------------------------------------
struct FileDescriptorLease {
    int fd = -1;

    explicit FileDescriptorLease(int f) : fd(f) {}
    ~FileDescriptorLease() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    FileDescriptorLease(const FileDescriptorLease&) = delete;
    FileDescriptorLease& operator=(const FileDescriptorLease&) = delete;
    FileDescriptorLease(FileDescriptorLease&& other) noexcept : fd(other.fd) {
        other.fd = -1;
    }
    FileDescriptorLease& operator=(FileDescriptorLease&& other) noexcept {
        if (this != &other) {
            if (fd >= 0) ::close(fd);
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }
};

// -------------------------------------------------------------------------
// LocalFileResolver
//
// Resolves "file://<absolute-path>" URIs by opening the file, verifying
// backing device identity, collecting FIEMAP extents (fail-closed on
// unsafe flags/states), and producing a ResolvedTarget with device offsets
// that include the configured namespace base.
//
// NamespaceIdentity (controller_pci_addr / namespace_id / block_size)
// and BackingDeviceConfig (backing_device_path / namespace_base_bytes)
// are injected at construction.  resolve() does NOT probe hardware.
// -------------------------------------------------------------------------
class LocalFileResolver : public StorageTargetResolver {
public:
    // Production constructor: takes namespace identity + backing device config.
    LocalFileResolver(std::string controller_pci_addr,
                      std::uint32_t namespace_id,
                      std::uint32_t block_size,
                      BackingDeviceConfig backing_config,
                      std::uint32_t exts_per_call = kFiemapMaxExtentsPerCall,
                      std::string data_path_key =
                          std::string(binding::ext4_local_nvme::
                                          kRecommendedDataPathKey))
        : ns_{
              std::move(controller_pci_addr),
              namespace_id,
              block_size},
          backing_config_(std::move(backing_config)),
          exts_per_call_(exts_per_call == 0
                          ? kFiemapMaxExtentsPerCall
                          : exts_per_call),
          data_path_key_(std::move(data_path_key)) {

        // Validate namespace_base alignment at construction.
        // If misaligned, resolve() will reject all files.
    }

    Result<ResolvedTarget> resolve(
        std::string_view uri,
        const ResolveOptions& options) override {

        // 1. Scheme check.
        if (options.scheme != kScheme) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::UNSUPPORTED,
                       "scheme '" + options.scheme +
                       "' != '" + std::string(kScheme) + "'"));
        }

        // 2. Parse URI.
        const std::string_view prefix = "file://";
        if (uri.size() < prefix.size() ||
            uri.substr(0, prefix.size()) != prefix) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "uri must start with 'file://': " +
                       std::string(uri)));
        }
        std::string_view path = uri.substr(prefix.size());
        if (path.empty() || path[0] != '/') {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "uri path must be absolute: " + std::string(uri)));
        }

        // 3. block_size == 0 check.
        if (ns_.block_size == 0) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "block_size == 0"));
        }

        // 4. A backing device is mandatory: without it the resolver cannot
        // prove that FIEMAP physical offsets belong to this namespace.
        if (backing_config_.backing_device_path.empty()) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "backing_device_path must be configured"));
        }

        // 5. namespace_base alignment check.
        if (backing_config_.namespace_base_bytes % ns_.block_size != 0) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "namespace_base_bytes (" +
                       std::to_string(backing_config_.namespace_base_bytes) +
                       ") not aligned to block_size (" +
                       std::to_string(ns_.block_size) + ")"));
        }

        // 5.5 Pre-stat: reject non-regular files with INVALID_ARGUMENT before
        // the O_DIRECT open. (open(dir, O_DIRECT) fails EINVAL on ext4, which
        // would surface as DEVICE_ERROR and mask the intended semantics.)
        {
            struct stat pre_st{};
            if (::stat(path.data(), &pre_st) == 0 && !S_ISREG(pre_st.st_mode)) {
                return Result<ResolvedTarget>::Failure(
                    Status(StatusCode::INVALID_ARGUMENT,
                           "not a regular file: " + std::string(path)));
            }
        }

        // 6. Open the file read-only. Project policy: ALL file opens carry
        // O_DIRECT (no page-cache pollution; harmless for FIEMAP-only use).
        int fd = ::open(std::string(path).c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) {
            int saved_errno = errno;
            if (saved_errno == ENOENT) {
                return Result<ResolvedTarget>::Failure(
                    Status(StatusCode::NOT_FOUND,
                           "file not found: " + std::string(path)));
            }
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::DEVICE_ERROR,
                       "open(" + std::string(path) + ") failed: errno " +
                       std::to_string(saved_errno) + " (" +
                       std::strerror(saved_errno) + ")"));
        }

        // 7. fstat: verify regular file + backing device identity.
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            int saved_errno = errno;
            ::close(fd);
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::DEVICE_ERROR,
                       "fstat(" + std::string(path) + ") failed: errno " +
                       std::to_string(saved_errno) + " (" +
                       std::strerror(saved_errno) + ")"));
        }

        // Must be a regular file.
        if (!S_ISREG(st.st_mode)) {
            ::close(fd);
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "not a regular file: " + std::string(path)));
        }

        // Verify backing device identity if configured.
        if (!backing_config_.backing_device_path.empty()) {
            struct stat dev_st{};
            if (::stat(backing_config_.backing_device_path.c_str(),
                       &dev_st) != 0) {
                int saved_errno = errno;
                ::close(fd);
                return Result<ResolvedTarget>::Failure(
                    Status(StatusCode::NOT_READY,
                           "cannot stat backing device " +
                           backing_config_.backing_device_path +
                           ": errno " + std::to_string(saved_errno)));
            }
            if (!S_ISBLK(dev_st.st_mode)) {
                ::close(fd);
                return Result<ResolvedTarget>::Failure(
                    Status(StatusCode::INVALID_ARGUMENT,
                           "configured backing path is not a block device: " +
                           backing_config_.backing_device_path));
            }
            if (st.st_dev != dev_st.st_rdev) {
                ::close(fd);
                return Result<ResolvedTarget>::Failure(
                    Status(StatusCode::INVALID_ARGUMENT,
                           "file's filesystem device (st_dev=" +
                           std::to_string(st.st_dev) +
                           ") does not match configured backing device (st_rdev=" +
                           std::to_string(dev_st.st_rdev) + "): " +
                           backing_config_.backing_device_path));
            }
        }

        // 8. Collect extents via FIEMAP (fail-closed).
        std::vector<binding::ext4_local_nvme::Extent> extents;
        std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);
        Status fiemap_status = collect_fiemap_(
            fd, path, file_size, extents);

        if (!fiemap_status.ok()) {
            ::close(fd);
            return Result<ResolvedTarget>::Failure(std::move(fiemap_status));
        }

        // 9. Create the immutable payload (create() runs validate()).
        auto payload_result = binding::ext4_local_nvme::Ext4LocalNvmePayload::
            create(ns_, std::move(extents), file_size);

        if (!payload_result.ok()) {
            ::close(fd);
            return Result<ResolvedTarget>::Failure(
                payload_result.status());
        }

        // 10. Create the fd lease and produce ResolvedTarget.
        auto lease = std::make_shared<FileDescriptorLease>(fd);

        return binding::ext4_local_nvme::make_resolved_target(
            std::string(binding::ext4_local_nvme::kResolverTypeId),
            file_size,
            std::move(payload_result).value(),
            std::move(lease),
            data_path_key_);
    }

private:
    // -----------------------------------------------------------------
    // collect_fiemap_ — fail-closed FIEMAP extent collection.
    //
    // Rejects extents with ANY of these flags:
    //   UNKNOWN, DELALLOC, UNWRITTEN, ENCODED, NOT_ALIGNED, SHARED,
    //   DATA_ENCRYPTED, DATA_INLINE, DATA_TAIL
    //
    // LAST is the only flag allowed (marks end of file).
    //
    // device_offset = fe_physical + namespace_base_bytes
    // -----------------------------------------------------------------
    Status collect_fiemap_(
        int fd,
        std::string_view path,
        std::uint64_t file_size,
        std::vector<binding::ext4_local_nvme::Extent>& out_extents) const {

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            int saved_errno = errno;
            return Status(StatusCode::DEVICE_ERROR,
                          "fstat(" + std::string(path) + ") failed: errno " +
                          std::to_string(saved_errno) + " (" +
                          std::strerror(saved_errno) + ")");
        }

        std::uint32_t fs_block_size = static_cast<std::uint32_t>(st.st_blksize);
        if (fs_block_size == 0) {
            return Status(StatusCode::DEVICE_ERROR,
                          "fstat returned st_blksize == 0 for " +
                          std::string(path));
        }

        if (fs_block_size % ns_.block_size != 0) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "fs block size " + std::to_string(fs_block_size) +
                          " not multiple of block_size " +
                          std::to_string(ns_.block_size));
        }

        // fsync before fiemap.
        if (::fsync(fd) != 0) {
            int saved_errno = errno;
            return Status(StatusCode::DEVICE_ERROR,
                          "fsync before fiemap: errno " +
                          std::to_string(saved_errno) + " (" +
                          std::strerror(saved_errno) + ")");
        }

        // Allocate FIEMAP buffer.
        const std::size_t buf_bytes =
            sizeof(struct fiemap) +
            sizeof(struct fiemap_extent) * exts_per_call_;
        std::vector<unsigned char> buf(buf_bytes, 0);
        auto* fm = reinterpret_cast<struct fiemap*>(buf.data());

        std::uint64_t logical_cursor = 0;
        bool done = false;

        while (!done) {
            std::memset(buf.data(), 0, buf_bytes);
            fm->fm_start        = logical_cursor;
            fm->fm_length       = ~static_cast<std::uint64_t>(0) - logical_cursor;
            fm->fm_flags        = FIEMAP_FLAG_SYNC;
            fm->fm_extent_count = exts_per_call_;

            if (::ioctl(fd, FS_IOC_FIEMAP, fm) != 0) {
                int saved_errno = errno;
                return Status(StatusCode::DEVICE_ERROR,
                              "FS_IOC_FIEMAP: errno " +
                              std::to_string(saved_errno) + " (" +
                              std::strerror(saved_errno) + ")");
            }

            if (fm->fm_mapped_extents == 0) {
                break;
            }

            for (std::uint32_t i = 0; i < fm->fm_mapped_extents; ++i) {
                const struct fiemap_extent& ex = fm->fm_extents[i];

                // --- Fail-closed flag mask ---
                // Reject ALL unsafe flags.  UNWRITTEN is now rejected
                // (fallocate-only files are NOT accepted).
                // LAST is allowed (end-of-file marker only).
                const std::uint32_t bad_flags =
                    FIEMAP_EXTENT_UNKNOWN |
                    FIEMAP_EXTENT_DELALLOC |
                    FIEMAP_EXTENT_DATA_ENCRYPTED |
                    FIEMAP_EXTENT_NOT_ALIGNED |
                    FIEMAP_EXTENT_DATA_INLINE |
                    FIEMAP_EXTENT_DATA_TAIL |
                    FIEMAP_EXTENT_UNWRITTEN |
                    FIEMAP_EXTENT_ENCODED |
                    FIEMAP_EXTENT_SHARED;

                if (ex.fe_flags & bad_flags) {
                    char tmp[256];
                    std::snprintf(tmp, sizeof(tmp),
                        "extent has unsafe flags 0x%x "
                        "(logical=%llu phys=%llu len=%llu)",
                        static_cast<unsigned>(ex.fe_flags),
                        static_cast<unsigned long long>(ex.fe_logical),
                        static_cast<unsigned long long>(ex.fe_physical),
                        static_cast<unsigned long long>(ex.fe_length));
                    return Status(StatusCode::DATA_LOSS, tmp);
                }

                // Alignment check: fe_physical and fe_length must be
                // block-aligned.
                if (ex.fe_physical % ns_.block_size != 0 ||
                    ex.fe_length % ns_.block_size != 0) {
                    char tmp[256];
                    std::snprintf(tmp, sizeof(tmp),
                        "extent not block-aligned: phys=%llu len=%llu "
                        "block_size=%u",
                        static_cast<unsigned long long>(ex.fe_physical),
                        static_cast<unsigned long long>(ex.fe_length),
                        ns_.block_size);
                    return Status(StatusCode::INVALID_ARGUMENT, tmp);
                }

                // Zero-length extent.
                if (ex.fe_length == 0) {
                    return Status(StatusCode::DATA_LOSS,
                                  "zero-length extent at logical " +
                                  std::to_string(ex.fe_logical));
                }

                // Non-monotonic extent (logical cursor went backwards).
                if (ex.fe_logical < logical_cursor) {
                    return Status(StatusCode::DATA_LOSS,
                                  "non-monotonic extent: logical " +
                                  std::to_string(ex.fe_logical) +
                                  " < cursor " + std::to_string(logical_cursor));
                }

                // --- Compute device_offset = fe_physical + namespace_base ---
                // Check for overflow.
                if (ex.fe_physical >
                    UINT64_MAX - backing_config_.namespace_base_bytes) {
                    return Status(StatusCode::OUT_OF_RANGE,
                                  "device_offset overflow: fe_physical=" +
                                  std::to_string(ex.fe_physical) +
                                  " + namespace_base=" +
                                  std::to_string(backing_config_.namespace_base_bytes));
                }
                std::uint64_t device_offset =
                    ex.fe_physical + backing_config_.namespace_base_bytes;

                // Check device_offset alignment (namespace_base is already
                // verified block-aligned, but guard anyway).
                if (device_offset % ns_.block_size != 0) {
                    return Status(StatusCode::INVALID_ARGUMENT,
                                  "device_offset not block-aligned: " +
                                  std::to_string(device_offset));
                }

                // Build the binding Extent (byte units).
                binding::ext4_local_nvme::Extent out{};
                out.logical_offset = ex.fe_logical;
                out.device_offset  = device_offset;
                out.length         = ex.fe_length;
                out_extents.push_back(out);

                // Total extent cap.
                if (out_extents.size() > kMaxTotalExtents) {
                    return Status(StatusCode::RESOURCE_EXHAUSTED,
                                  "too many extents (>" +
                                  std::to_string(kMaxTotalExtents) +
                                  "); file too fragmented");
                }

                if (ex.fe_flags & FIEMAP_EXTENT_LAST) {
                    done = true;
                }
                logical_cursor = ex.fe_logical + ex.fe_length;
            }

            if (!done && fm->fm_mapped_extents < exts_per_call_) {
                break;
            }
        }

        // 0 extent check.
        if (out_extents.empty()) {
            return Status(StatusCode::DATA_LOSS,
                          "fiemap returned 0 extents "
                          "(file is empty / sparse?)");
        }

        // File coverage check: extents must cover [0, file_size).
        // validate() in Ext4LocalNvmePayload also checks this, but we
        // verify here to give a more specific error.
        std::uint64_t covered = 0;
        for (const auto& e : out_extents) {
            covered += e.length;
        }
        if (covered != file_size) {
            return Status(StatusCode::DATA_LOSS,
                          "extents cover " + std::to_string(covered) +
                          " bytes but file_size is " +
                          std::to_string(file_size) +
                          " (incomplete coverage or holes)");
        }

        return Status::Ok();
    }

    binding::ext4_local_nvme::NamespaceIdentity ns_;
    BackingDeviceConfig backing_config_;
    std::uint32_t exts_per_call_;
    std::string data_path_key_;
};

} // namespace tutti::resolvers::local_file
