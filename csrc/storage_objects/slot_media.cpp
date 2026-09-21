// csrc/storage_objects/slot_media.cpp

#include "csrc/storage_objects/slot_media.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace tutti::storage_objects {
namespace {

constexpr std::size_t kAlignment = 4096;

// Chunk size for materialisation. Large enough that per-write syscall overhead
// is irrelevant, small enough that the pinned aligned buffer stays modest.
constexpr std::size_t kZeroChunkBytes = 4u * 1024 * 1024;

Status errno_status(const char* what, const std::string& path) {
    std::string message = what;
    message += " (";
    message += path;
    message += "): ";
    message += std::strerror(errno);
    return Status(StatusCode::INTERNAL, message);
}

bool is_aligned(std::uint64_t value) noexcept {
    return value % kAlignment == 0;
}

// RAII descriptor. Deliberately minimal: an open fd escaping on an error path
// would leak a descriptor per failed slot, which at pool scale exhausts the
// process limit and turns a transient error into a hard stop.
class Fd {
public:
    explicit Fd(int fd) : fd_(fd) {}
    ~Fd() { if (fd_ >= 0) ::close(fd_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int get() const noexcept { return fd_; }
    bool valid() const noexcept { return fd_ >= 0; }
private:
    int fd_;
};

// Write `bytes` of zeros at `offset` using O_DIRECT-aligned chunks.
Status write_zeros(int fd, const std::string& path, std::uint64_t offset,
                   std::uint64_t bytes) {
    if (bytes == 0) return {};

    AlignedBuffer buffer(kZeroChunkBytes);
    if (!buffer.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    buffer.zero();

    std::uint64_t written = 0;
    while (written < bytes) {
        const std::uint64_t remaining = bytes - written;
        const std::size_t chunk = remaining < kZeroChunkBytes
                                      ? static_cast<std::size_t>(remaining)
                                      : kZeroChunkBytes;
        // O_DIRECT requires the length to be aligned too. All slot sizes are
        // whole 4096 multiples by construction, so a short tail can only mean a
        // caller passed an unaligned size -- reject rather than silently fall
        // back to buffered IO.
        if (!is_aligned(chunk)) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "zero length must be 4096-aligned for O_DIRECT");
        }
        const ssize_t n = ::pwrite(fd, buffer.data(), chunk,
                                   static_cast<off_t>(offset + written));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pwrite zeros", path);
        }
        if (n == 0) {
            return Status(StatusCode::INTERNAL, "pwrite returned 0: " + path);
        }
        written += static_cast<std::uint64_t>(n);
    }
    return {};
}

std::string parent_of(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

} // namespace

// -------------------------------------------------------------------------
// AlignedBuffer
// -------------------------------------------------------------------------

AlignedBuffer::AlignedBuffer(std::size_t bytes) {
    if (bytes == 0) return;
    // Round up so the allocation itself is a whole number of blocks; O_DIRECT
    // cares about the address and the length actually used, and rounding here
    // means callers never have to.
    const std::size_t rounded = ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
    void* raw = nullptr;
    if (::posix_memalign(&raw, kAlignment, rounded) != 0) return;
    data_ = static_cast<std::uint8_t*>(raw);
    size_ = rounded;
}

AlignedBuffer::~AlignedBuffer() {
    if (data_ != nullptr) std::free(data_);
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) std::free(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void AlignedBuffer::zero() noexcept {
    if (data_ != nullptr) std::memset(data_, 0, size_);
}

// -------------------------------------------------------------------------
// Directory helpers
// -------------------------------------------------------------------------

Status ensure_directory(const std::string& path) {
    if (path.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "empty directory path");
    }
    // Create parents first. Walking forward and ignoring EEXIST is both simpler
    // and race-tolerant: several ranks bring up the same namespace root
    // concurrently, so "already exists" is the normal case, not an error.
    std::string partial;
    partial.reserve(path.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
        partial.push_back(path[i]);
        const bool last = (i + 1 == path.size());
        if (path[i] != '/' && !last) continue;
        if (partial == "/" || partial.empty()) continue;
        std::string dir = partial;
        if (dir.size() > 1 && dir.back() == '/') dir.pop_back();
        if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
            return errno_status("mkdir", dir);
        }
    }
    return {};
}

Status sync_directory(const std::string& path) {
    const Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECTORY));
    if (!fd.valid()) return errno_status("open directory", path);
    if (::fsync(fd.get()) != 0) return errno_status("fsync directory", path);
    return {};
}

// -------------------------------------------------------------------------
// Materialisation
// -------------------------------------------------------------------------

Status materialise_slot(const std::vector<std::string>& paths,
                        std::uint64_t bytes_per_shard) {
    if (paths.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "no shard paths");
    }
    if (bytes_per_shard == 0 || !is_aligned(bytes_per_shard)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "shard size must be nonzero and 4096-aligned");
    }

    for (const std::string& path : paths) {
        const std::string dir = parent_of(path);
        const Status dir_status = ensure_directory(dir);
        if (!dir_status.ok()) return dir_status;

        const Fd fd(::open(path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644));
        if (!fd.valid()) return errno_status("open slot for materialise", path);

        struct stat st{};
        if (::fstat(fd.get(), &st) != 0) {
            return errno_status("fstat slot", path);
        }

        // Idempotent: an existing file already at the right size is assumed
        // materialised. Rewriting it on every restart would make bringing up an
        // existing terabyte pool as expensive as creating it.
        if (static_cast<std::uint64_t>(st.st_size) == bytes_per_shard) {
            continue;
        }

        if (::ftruncate(fd.get(), static_cast<off_t>(bytes_per_shard)) != 0) {
            return errno_status("ftruncate slot", path);
        }
        // ftruncate only sets the size; the extents are still holes. Writing
        // real zeros is what allocates them, which FIEMAP-based resolution
        // requires.
        const Status zeroed = write_zeros(fd.get(), path, 0, bytes_per_shard);
        if (!zeroed.ok()) return zeroed;
        if (::fsync(fd.get()) != 0) return errno_status("fsync slot", path);

        // The file's own fsync does not make its directory entry durable.
        const Status dir_synced = sync_directory(dir);
        if (!dir_synced.ok()) return dir_synced;
    }
    return {};
}

Status slot_is_precreated(const std::vector<std::string>& paths,
                            std::uint64_t bytes_per_shard, bool* out) {
    if (out == nullptr) {
        return Status(StatusCode::INVALID_ARGUMENT, "no output flag");
    }
    *out = false;
    if (paths.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "no shard paths");
    }
    for (const std::string& path : paths) {
        struct stat st{};
        if (::stat(path.c_str(), &st) != 0) {
            if (errno == ENOENT || errno == ENOTDIR) return {};
            return errno_status("stat slot", path);
        }
        if (!S_ISREG(st.st_mode) ||
            static_cast<std::uint64_t>(st.st_size) != bytes_per_shard) {
            return {};
        }
    }
    *out = true;
    return {};
}

Status zero_slot(const std::vector<std::string>& paths,
                 std::uint64_t bytes_per_shard) {
    if (paths.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "no shard paths");
    }
    if (bytes_per_shard == 0 || !is_aligned(bytes_per_shard)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "shard size must be nonzero and 4096-aligned");
    }

    for (const std::string& path : paths) {
        const Fd fd(::open(path.c_str(), O_RDWR | O_DIRECT));
        if (!fd.valid()) return errno_status("open slot for zeroing", path);
        const Status zeroed = write_zeros(fd.get(), path, 0, bytes_per_shard);
        if (!zeroed.ok()) return zeroed;
        if (::fsync(fd.get()) != 0) return errno_status("fsync zeroed slot", path);
    }
    return {};
}

// -------------------------------------------------------------------------
// Header IO
// -------------------------------------------------------------------------

Status write_object_header(const std::string& path, std::uint64_t offset,
                           const std::uint8_t* header,
                           std::size_t header_bytes) {
    if (header == nullptr || header_bytes == 0 || !is_aligned(header_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header offset must be 4096-aligned");
    }

    // The caller's buffer may not be aligned, so stage through one that is.
    AlignedBuffer staged(header_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    std::memcpy(staged.data(), header, header_bytes);

    const Fd fd(::open(path.c_str(), O_RDWR | O_DIRECT));
    if (!fd.valid()) return errno_status("open slot for header write", path);

    std::size_t written = 0;
    while (written < header_bytes) {
        const ssize_t n = ::pwrite(fd.get(), staged.data() + written,
                                   header_bytes - written,
                                   static_cast<off_t>(offset + written));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pwrite header", path);
        }
        if (n == 0) {
            return Status(StatusCode::INTERNAL, "pwrite header returned 0: " + path);
        }
        written += static_cast<std::size_t>(n);
    }

    // THE commit point. The payload must already be durable: "payload fsync then
    // header fsync" is precisely what makes a valid header imply durable data.
    if (::fsync(fd.get()) != 0) return errno_status("fsync header", path);
    return {};
}

Status read_object_header(const std::string& path, std::uint64_t offset,
                          std::uint8_t* out, std::size_t out_bytes) {
    if (out == nullptr || out_bytes == 0 || !is_aligned(out_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header buffer must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header offset must be 4096-aligned");
    }

    const Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) {
        if (errno == ENOENT) {
            return Status(StatusCode::NOT_FOUND, "slot file absent: " + path);
        }
        return errno_status("open slot for header read", path);
    }

    AlignedBuffer staged(out_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    staged.zero();

    std::size_t read_total = 0;
    while (read_total < out_bytes) {
        const ssize_t n = ::pread(fd.get(), staged.data() + read_total,
                                  out_bytes - read_total,
                                  static_cast<off_t>(offset + read_total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pread header", path);
        }
        if (n == 0) {
            // Short at EOF: a file smaller than its header simply holds no
            // object. That is an absence, not a failure.
            if (read_total == 0) {
                return Status(StatusCode::NOT_FOUND,
                              "slot shorter than header: " + path);
            }
            break;
        }
        read_total += static_cast<std::size_t>(n);
    }

    std::memcpy(out, staged.data(), out_bytes);
    return {};
}

// -------------------------------------------------------------------------
// Checkpoint IO
// -------------------------------------------------------------------------

Status write_checkpoint_container(const std::string& path, std::uint64_t offset,
                                  const std::uint8_t* image,
                                  std::size_t image_bytes) {
    if (image == nullptr || image_bytes == 0 || !is_aligned(image_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint image must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint offset must be 4096-aligned");
    }

    AlignedBuffer staged(image_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    std::memcpy(staged.data(), image, image_bytes);

    const Fd fd(::open(path.c_str(), O_RDWR | O_DIRECT));
    if (!fd.valid()) return errno_status("open checkpoint", path);

    std::size_t written = 0;
    while (written < image_bytes) {
        const ssize_t n = ::pwrite(fd.get(), staged.data() + written,
                                   image_bytes - written,
                                   static_cast<off_t>(offset + written));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pwrite checkpoint", path);
        }
        if (n == 0) {
            return Status(StatusCode::INTERNAL, "pwrite checkpoint returned 0");
        }
        written += static_cast<std::size_t>(n);
    }
    if (::fsync(fd.get()) != 0) return errno_status("fsync checkpoint", path);
    return {};
}

Status read_checkpoint_container(const std::string& path, std::uint64_t offset,
                                 std::uint8_t* out, std::size_t out_bytes) {
    if (out == nullptr || out_bytes == 0 || !is_aligned(out_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint buffer must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint offset must be 4096-aligned");
    }

    const Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) {
        if (errno == ENOENT) {
            return Status(StatusCode::NOT_FOUND, "checkpoint absent: " + path);
        }
        return errno_status("open checkpoint for read", path);
    }

    AlignedBuffer staged(out_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    staged.zero();

    std::size_t read_total = 0;
    while (read_total < out_bytes) {
        const ssize_t n = ::pread(fd.get(), staged.data() + read_total,
                                  out_bytes - read_total,
                                  static_cast<off_t>(offset + read_total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pread checkpoint", path);
        }
        if (n == 0) break;  // short at EOF: remaining bytes stay zero
        read_total += static_cast<std::size_t>(n);
    }

    std::memcpy(out, staged.data(), out_bytes);
    return {};
}

Status ensure_metadata_file(const std::string& path, std::uint64_t bytes) {
    if (bytes == 0 || !is_aligned(bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "metadata file size must be nonzero and 4096-aligned");
    }
    const std::string dir = parent_of(path);
    const Status dir_status = ensure_directory(dir);
    if (!dir_status.ok()) return dir_status;

    const Fd fd(::open(path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644));
    if (!fd.valid()) return errno_status("open metadata file", path);

    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) return errno_status("fstat metadata", path);
    if (static_cast<std::uint64_t>(st.st_size) >= bytes) return {};

    if (::ftruncate(fd.get(), static_cast<off_t>(bytes)) != 0) {
        return errno_status("ftruncate metadata", path);
    }
    const Status zeroed = write_zeros(fd.get(), path, 0, bytes);
    if (!zeroed.ok()) return zeroed;
    if (::fsync(fd.get()) != 0) return errno_status("fsync metadata", path);
    return sync_directory(dir);
}

} // namespace tutti::storage_objects
