// csrc/storage_objects/residency_bitmap.cpp

#include "csrc/storage_objects/residency_bitmap.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

namespace tutti::storage_objects {
namespace {

constexpr std::uint64_t kAlignment = 4096;

void store_u32(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}

void store_u64(std::uint8_t* p, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
}

std::uint32_t load_u32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t load_u64(const std::uint8_t* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

std::uint64_t round_up(std::uint64_t value, std::uint64_t multiple) noexcept {
    return ((value + multiple - 1) / multiple) * multiple;
}

Status errno_status(const char* what) {
    std::string message = what;
    message += ": ";
    message += std::strerror(errno);
    return Status(StatusCode::INTERNAL, message);
}

} // namespace

std::uint64_t residency_file_bytes(std::uint64_t slot_count) noexcept {
    const std::uint64_t body = (slot_count + 7) / 8;
    return round_up(ResidencyBitmapLayout::kHeaderBytes + body, kAlignment);
}

std::uint64_t residency_bit_offset(std::uint64_t slot) noexcept {
    return ResidencyBitmapLayout::kHeaderBytes + slot / 8;
}

std::uint32_t residency_bit_shift(std::uint64_t slot) noexcept {
    return static_cast<std::uint32_t>(slot % 8);
}

void encode_residency_header(std::uint8_t* out, std::uint32_t rank_id,
                            std::uint32_t rank_count, std::uint64_t slot_count,
                            std::uint64_t fingerprint_digest) noexcept {
    std::memset(out, 0, ResidencyBitmapLayout::kHeaderBytes);
    std::memcpy(out + ResidencyBitmapLayout::kMagicOffset,
                ResidencyBitmapLayout::kMagic,
                sizeof(ResidencyBitmapLayout::kMagic));
    store_u32(out + ResidencyBitmapLayout::kVersionOffset,
              ResidencyBitmapLayout::kVersion);
    store_u32(out + ResidencyBitmapLayout::kRankIdOffset, rank_id);
    store_u32(out + ResidencyBitmapLayout::kRankCountOffset, rank_count);
    store_u64(out + ResidencyBitmapLayout::kSlotCountOffset, slot_count);
    store_u64(out + ResidencyBitmapLayout::kFingerprintOffset,
              fingerprint_digest);
}

const char* to_string(ResidencyRejection rejection) noexcept {
    switch (rejection) {
        case ResidencyRejection::kNone: return "none";
        case ResidencyRejection::kEmptyFile: return "empty_file";
        case ResidencyRejection::kBadMagic: return "bad_magic";
        case ResidencyRejection::kUnsupportedVersion: return "unsupported_version";
        case ResidencyRejection::kTruncatedBuffer: return "truncated_buffer";
        case ResidencyRejection::kSlotCountMismatch: return "slot_count_mismatch";
        case ResidencyRejection::kFingerprintMismatch: return "fingerprint_mismatch";
        case ResidencyRejection::kRankCountMismatch: return "rank_count_mismatch";
    }
    return "unknown";
}

ResidencyRejection validate_residency_header(
    const std::uint8_t* data, std::size_t data_bytes, std::uint32_t rank_count,
    std::uint64_t slot_count, std::uint64_t fingerprint_digest) noexcept {
    if (data == nullptr || data_bytes < ResidencyBitmapLayout::kHeaderBytes) {
        return ResidencyRejection::kTruncatedBuffer;
    }

    bool all_zero = true;
    for (std::size_t i = 0; i < sizeof(ResidencyBitmapLayout::kMagic); ++i) {
        if (data[ResidencyBitmapLayout::kMagicOffset + i] != 0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) return ResidencyRejection::kEmptyFile;

    if (std::memcmp(data + ResidencyBitmapLayout::kMagicOffset,
                    ResidencyBitmapLayout::kMagic,
                    sizeof(ResidencyBitmapLayout::kMagic)) != 0) {
        return ResidencyRejection::kBadMagic;
    }
    if (load_u32(data + ResidencyBitmapLayout::kVersionOffset) !=
        ResidencyBitmapLayout::kVersion) {
        return ResidencyRejection::kUnsupportedVersion;
    }
    if (load_u32(data + ResidencyBitmapLayout::kRankCountOffset) != rank_count) {
        return ResidencyRejection::kRankCountMismatch;
    }
    if (load_u64(data + ResidencyBitmapLayout::kSlotCountOffset) != slot_count) {
        return ResidencyRejection::kSlotCountMismatch;
    }
    if (load_u64(data + ResidencyBitmapLayout::kFingerprintOffset) !=
        fingerprint_digest) {
        return ResidencyRejection::kFingerprintMismatch;
    }
    return ResidencyRejection::kNone;
}

ResidencyBitmap::~ResidencyBitmap() { close(); }

ResidencyBitmap::ResidencyBitmap(ResidencyBitmap&& other) noexcept
    : base_(other.base_),
      mapped_bytes_(other.mapped_bytes_),
      slot_count_(other.slot_count_),
      writable_(other.writable_) {
    other.base_ = nullptr;
    other.mapped_bytes_ = 0;
    other.slot_count_ = 0;
    other.writable_ = false;
}

ResidencyBitmap& ResidencyBitmap::operator=(ResidencyBitmap&& other) noexcept {
    if (this != &other) {
        close();
        base_ = other.base_;
        mapped_bytes_ = other.mapped_bytes_;
        slot_count_ = other.slot_count_;
        writable_ = other.writable_;
        other.base_ = nullptr;
        other.mapped_bytes_ = 0;
        other.slot_count_ = 0;
        other.writable_ = false;
    }
    return *this;
}

Status ResidencyBitmap::open_writable(const std::string& path,
                                      std::uint32_t rank_id,
                                      std::uint32_t rank_count,
                                      std::uint64_t slot_count,
                                      std::uint64_t fingerprint_digest) {
    close();
    if (slot_count == 0) {
        return Status(StatusCode::INVALID_ARGUMENT, "slot_count must be nonzero");
    }
    const std::uint64_t want_bytes = residency_file_bytes(slot_count);

    // Not O_DIRECT: this file is mmapped, and O_DIRECT is meaningless for a
    // mapping. It is also pure metadata that never carries KV payload, so the
    // project-wide O_DIRECT rule for data files does not apply.
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) return errno_status("open residency bitmap");

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        const Status s = errno_status("fstat residency bitmap");
        ::close(fd);
        return s;
    }

    const bool fresh = (st.st_size == 0);
    if (fresh) {
        if (::ftruncate(fd, static_cast<off_t>(want_bytes)) != 0) {
            const Status s = errno_status("size residency bitmap");
            ::close(fd);
            return s;
        }
    } else if (static_cast<std::uint64_t>(st.st_size) != want_bytes) {
        // Geometry disagreement. Refuse rather than resize: the file may belong
        // to a still-running peer, and truncating it would destroy information
        // that peer needs. The caller degrades to the single-rank path.
        ::close(fd);
        return Status(StatusCode::INVALID_ARGUMENT,
                      "residency bitmap size mismatch");
    }

    void* mapping = ::mmap(nullptr, static_cast<std::size_t>(want_bytes),
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    // The descriptor is not needed once mapped; the mapping keeps the file
    // alive, and holding fds open across a large rank count is pure cost.
    ::close(fd);
    if (mapping == MAP_FAILED) return errno_status("mmap residency bitmap");

    auto* bytes = static_cast<std::uint8_t*>(mapping);
    if (fresh) {
        encode_residency_header(bytes, rank_id, rank_count, slot_count,
                                fingerprint_digest);
    } else {
        const ResidencyRejection rejection = validate_residency_header(
            bytes, static_cast<std::size_t>(want_bytes), rank_count, slot_count,
            fingerprint_digest);
        if (rejection == ResidencyRejection::kEmptyFile) {
            // Right size but never initialised: finish initialisation.
            encode_residency_header(bytes, rank_id, rank_count, slot_count,
                                    fingerprint_digest);
        } else if (rejection != ResidencyRejection::kNone) {
            ::munmap(mapping, static_cast<std::size_t>(want_bytes));
            std::string message = "residency bitmap rejected: ";
            message += to_string(rejection);
            return Status(StatusCode::INVALID_ARGUMENT, message);
        }
    }

    base_ = mapping;
    mapped_bytes_ = static_cast<std::size_t>(want_bytes);
    slot_count_ = slot_count;
    writable_ = true;
    return {};
}

Status ResidencyBitmap::open_readonly(const std::string& path,
                                      std::uint32_t rank_count,
                                      std::uint64_t slot_count,
                                      std::uint64_t fingerprint_digest) {
    close();
    if (slot_count == 0) {
        return Status(StatusCode::INVALID_ARGUMENT, "slot_count must be nonzero");
    }
    const std::uint64_t want_bytes = residency_file_bytes(slot_count);

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return errno_status("open residency bitmap (readonly)");

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        const Status s = errno_status("fstat residency bitmap (readonly)");
        ::close(fd);
        return s;
    }
    if (static_cast<std::uint64_t>(st.st_size) != want_bytes) {
        ::close(fd);
        return Status(StatusCode::INVALID_ARGUMENT,
                      "residency bitmap size mismatch");
    }

    void* mapping = ::mmap(nullptr, static_cast<std::size_t>(want_bytes),
                           PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapping == MAP_FAILED) {
        return errno_status("mmap residency bitmap (readonly)");
    }

    const ResidencyRejection rejection = validate_residency_header(
        static_cast<const std::uint8_t*>(mapping),
        static_cast<std::size_t>(want_bytes), rank_count, slot_count,
        fingerprint_digest);
    if (rejection != ResidencyRejection::kNone) {
        ::munmap(mapping, static_cast<std::size_t>(want_bytes));
        std::string message = "residency bitmap rejected: ";
        message += to_string(rejection);
        return Status(StatusCode::INVALID_ARGUMENT, message);
    }

    base_ = mapping;
    mapped_bytes_ = static_cast<std::size_t>(want_bytes);
    slot_count_ = slot_count;
    writable_ = false;
    return {};
}

void ResidencyBitmap::close() {
    if (base_ != nullptr) {
        // No msync here: durability must be requested explicitly so its cost
        // stays visible at the call site.
        ::munmap(base_, mapped_bytes_);
        base_ = nullptr;
    }
    mapped_bytes_ = 0;
    slot_count_ = 0;
    writable_ = false;
}

bool ResidencyBitmap::test(std::uint64_t slot) const noexcept {
    // Unusable or out-of-range reads as absent: the under-reporting direction.
    if (base_ == nullptr || slot >= slot_count_) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(base_);
    const std::uint8_t byte = bytes[residency_bit_offset(slot)];
    return (byte >> residency_bit_shift(slot)) & 1u;
}

void ResidencyBitmap::set(std::uint64_t slot) noexcept {
    if (base_ == nullptr || !writable_ || slot >= slot_count_) return;
    auto* bytes = static_cast<std::uint8_t*>(base_);
    bytes[residency_bit_offset(slot)] |=
        static_cast<std::uint8_t>(1u << residency_bit_shift(slot));
}

void ResidencyBitmap::clear(std::uint64_t slot) noexcept {
    if (base_ == nullptr || !writable_ || slot >= slot_count_) return;
    auto* bytes = static_cast<std::uint8_t*>(base_);
    bytes[residency_bit_offset(slot)] &=
        static_cast<std::uint8_t>(~(1u << residency_bit_shift(slot)));
}

void ResidencyBitmap::clear_all() noexcept {
    if (base_ == nullptr || !writable_) return;
    auto* bytes = static_cast<std::uint8_t*>(base_);
    std::memset(bytes + ResidencyBitmapLayout::kHeaderBytes, 0,
                mapped_bytes_ - ResidencyBitmapLayout::kHeaderBytes);
}

Status ResidencyBitmap::sync() {
    if (base_ == nullptr || !writable_) return {};
    if (::msync(base_, mapped_bytes_, MS_ASYNC) != 0) {
        return errno_status("msync residency bitmap");
    }
    return {};
}

bool all_ranks_committed(const std::vector<const ResidencyBitmap*>& ranks,
                        std::uint64_t slot) noexcept {
    if (ranks.empty()) return false;
    for (const ResidencyBitmap* rank : ranks) {
        // A missing or unusable rank means we cannot assert cross-rank
        // residency. Answering false is the only safe answer.
        if (rank == nullptr || !rank->usable()) return false;
        if (!rank->test(slot)) return false;
    }
    return true;
}

} // namespace tutti::storage_objects
