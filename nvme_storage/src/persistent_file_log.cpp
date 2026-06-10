/**
 * persistent_file_log.cpp -- on-disk file directory for one Device.
 */

#include "persistent_file_log.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tutti {

namespace {

// 'TUTITLOG' little-endian.
constexpr uint64_t kLogMagic   = 0x474F4C544954555455ULL & 0xFFFFFFFFFFFFFFFFULL;
constexpr uint64_t kLogMagicV1 = 0x474F4C5449545554ULL;     // "TUTITLOG"
constexpr uint32_t kLogVersion = 1;

constexpr size_t   kNameMax    = 128;

#pragma pack(push, 1)
struct OnDiskHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t entry_count;
    uint64_t next_file_id;
    uint64_t reserved[2];
};
struct OnDiskEntry {
    uint64_t file_id;
    uint64_t size_bytes;
    uint32_t num_extents;
    uint32_t reserved;
    char     name[kNameMax];
    // followed by num_extents * sizeof(LbaExtent)
};
#pragma pack(pop)
static_assert(sizeof(OnDiskHeader) == 8 + 4 + 4 + 8 + 16,
              "OnDiskHeader layout drift");
static_assert(sizeof(OnDiskEntry)  == 8 + 8 + 4 + 4 + kNameMax,
              "OnDiskEntry layout drift");

bool write_full(int fd, const void* buf, size_t n) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::write(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}

bool read_full(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}

} // namespace

PersistentFileLog::PersistentFileLog() = default;
PersistentFileLog::~PersistentFileLog() = default;

bool PersistentFileLog::load_or_init(std::string log_path) {
    log_path_ = std::move(log_path);
    entries_.clear();
    name_to_index_.clear();
    next_file_id_ = 1;
    return load_locked();
}

bool PersistentFileLog::load_locked() {
    int fd = ::open(log_path_.c_str(), O_RDONLY);
    if (fd < 0) {
        if (errno == ENOENT) {
            // Fresh log; persist an empty header so future load
            // sees us.
            return persist_locked();
        }
        std::fprintf(stderr,
            "[persistent_file_log] open(%s) for read: errno %d\n",
            log_path_.c_str(), errno);
        return false;
    }

    OnDiskHeader hdr{};
    if (!read_full(fd, &hdr, sizeof(hdr))) {
        ::close(fd);
        std::fprintf(stderr,
            "[persistent_file_log] short read on header: errno %d\n", errno);
        return false;
    }
    if (hdr.magic != kLogMagicV1) {
        ::close(fd);
        std::fprintf(stderr,
            "[persistent_file_log] bad magic 0x%llx at %s\n",
            (unsigned long long)hdr.magic, log_path_.c_str());
        return false;
    }
    if (hdr.version != kLogVersion) {
        ::close(fd);
        std::fprintf(stderr,
            "[persistent_file_log] unsupported version %u\n", hdr.version);
        return false;
    }

    next_file_id_ = (hdr.next_file_id == 0) ? 1 : hdr.next_file_id;

    entries_.reserve(hdr.entry_count);
    for (uint32_t i = 0; i < hdr.entry_count; ++i) {
        OnDiskEntry oe{};
        if (!read_full(fd, &oe, sizeof(oe))) {
            ::close(fd);
            std::fprintf(stderr,
                "[persistent_file_log] short read on entry #%u\n", i);
            return false;
        }
        Entry e{};
        e.file_id    = oe.file_id;
        e.size_bytes = oe.size_bytes;
        e.name.assign(oe.name, ::strnlen(oe.name, kNameMax));

        e.extents.resize(oe.num_extents);
        if (oe.num_extents > 0) {
            if (!read_full(fd, e.extents.data(),
                           oe.num_extents * sizeof(LbaExtent))) {
                ::close(fd);
                std::fprintf(stderr,
                    "[persistent_file_log] short read on extents of #%u\n", i);
                return false;
            }
        }

        const std::size_t idx = entries_.size();
        name_to_index_[e.name] = idx;
        entries_.push_back(std::move(e));
    }

    ::close(fd);
    return true;
}

const PersistentFileLog::Entry*
PersistentFileLog::find_by_name(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) return nullptr;
    return &entries_[it->second];
}

uint64_t PersistentFileLog::next_file_id() {
    return next_file_id_++;
}

bool PersistentFileLog::add(Entry e) {
    if (e.name.size() >= kNameMax) {
        std::fprintf(stderr,
            "[persistent_file_log] name too long (%zu >= %zu)\n",
            e.name.size(), kNameMax);
        return false;
    }
    if (name_to_index_.count(e.name) != 0) return false;
    const std::size_t idx = entries_.size();
    name_to_index_[e.name] = idx;
    if (e.file_id >= next_file_id_) next_file_id_ = e.file_id + 1;
    entries_.push_back(std::move(e));
    return true;
}

bool PersistentFileLog::remove(uint64_t file_id) {
    // Linear scan to find the entry by file_id, then swap-and-pop
    // to avoid the O(N) std::vector::erase + index rebuild that
    // the original implementation incurred on every delete (which
    // turned cleanup of N files into O(N^2) memory traffic).
    //
    // Stable insertion order is NOT a public invariant of this
    // log -- on-disk recovery (load_or_init) and find_by_name
    // both ignore order -- so swap-and-pop is safe here.
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].file_id == file_id) {
            name_to_index_.erase(entries_[i].name);
            const std::size_t last = entries_.size() - 1;
            if (i != last) {
                entries_[i] = std::move(entries_[last]);
                name_to_index_[entries_[i].name] = i;
            }
            entries_.pop_back();
            return true;
        }
    }
    return false;
}

bool PersistentFileLog::persist() {
    return persist_locked();
}

bool PersistentFileLog::persist_locked() {
    // Atomic rename: write to .tmp then rename onto final path.
    std::string tmp_path = log_path_ + ".tmp";

    int fd = ::open(tmp_path.c_str(),
                     O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        std::fprintf(stderr,
            "[persistent_file_log] open(%s) for write: errno %d\n",
            tmp_path.c_str(), errno);
        return false;
    }

    OnDiskHeader hdr{};
    hdr.magic        = kLogMagicV1;
    hdr.version      = kLogVersion;
    hdr.entry_count  = (uint32_t)entries_.size();
    hdr.next_file_id = next_file_id_;
    if (!write_full(fd, &hdr, sizeof(hdr))) {
        ::close(fd); ::unlink(tmp_path.c_str());
        return false;
    }

    for (const auto& e : entries_) {
        OnDiskEntry oe{};
        oe.file_id     = e.file_id;
        oe.size_bytes  = e.size_bytes;
        oe.num_extents = (uint32_t)e.extents.size();
        std::memset(oe.name, 0, kNameMax);
        std::memcpy(oe.name, e.name.data(),
                    std::min(e.name.size(), (size_t)(kNameMax - 1)));
        if (!write_full(fd, &oe, sizeof(oe))) {
            ::close(fd); ::unlink(tmp_path.c_str());
            return false;
        }
        if (!e.extents.empty()) {
            if (!write_full(fd, e.extents.data(),
                            e.extents.size() * sizeof(LbaExtent))) {
                ::close(fd); ::unlink(tmp_path.c_str());
                return false;
            }
        }
    }

    if (::fsync(fd) != 0) {
        std::fprintf(stderr,
            "[persistent_file_log] fsync(%s): errno %d\n",
            tmp_path.c_str(), errno);
        ::close(fd); ::unlink(tmp_path.c_str());
        return false;
    }
    ::close(fd);

    if (::rename(tmp_path.c_str(), log_path_.c_str()) != 0) {
        std::fprintf(stderr,
            "[persistent_file_log] rename(%s -> %s): errno %d\n",
            tmp_path.c_str(), log_path_.c_str(), errno);
        ::unlink(tmp_path.c_str());
        return false;
    }
    return true;
}

} // namespace tutti
