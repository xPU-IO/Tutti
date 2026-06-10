#include "persistent_gpu_file_log.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <cstdio>
#include <cstring>
#include <utility>

namespace tutti {

namespace {

// 'TUTTGPU\0' little-endian = 0x0055504754545554
constexpr uint64_t kMagic   = 0x0055504754545554ULL;
constexpr uint32_t kVersion = 1u;

#pragma pack(push, 1)
struct OnDiskHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t entry_count;
    uint64_t generation;
    uint32_t next_file_id;
    uint32_t reserved[5];
};
static_assert(sizeof(OnDiskHeader) == 8 + 4 + 4 + 8 + 4 + 5 * 4,
              "OnDiskHeader layout drift");

struct OnDiskEntry {
    uint32_t file_id;
    uint32_t num_shards;
    uint64_t total_size;
    uint32_t tensor_shape[3];
    uint32_t reserved0;
    char     name[128];
    int32_t  shard_device_ids[kGpuFileMaxShards];   // 4
    uint32_t reserved1[4];
};
static_assert(sizeof(OnDiskEntry) ==
              4 + 4 + 8 + 12 + 4 + 128 + (int)kGpuFileMaxShards * 4 + 16,
              "OnDiskEntry layout drift");
#pragma pack(pop)

bool write_full(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        p += w;
        n -= (size_t)w;
    }
    return true;
}

bool read_full(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    while (n > 0) {
        ssize_t r = ::read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;     // unexpected EOF
        p += r;
        n -= (size_t)r;
    }
    return true;
}

}  // namespace

PersistentGpuFileLog::PersistentGpuFileLog()  = default;
PersistentGpuFileLog::~PersistentGpuFileLog() = default;

bool PersistentGpuFileLog::load_or_init(std::string log_path) {
    log_path_ = std::move(log_path);
    return load_locked();
}

bool PersistentGpuFileLog::load_locked() {
    entries_.clear();
    name_to_index_.clear();
    next_file_id_ = 1;
    generation_   = 0;

    int fd = ::open(log_path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (errno == ENOENT) {
            return true;            // fresh log
        }
        std::fprintf(stderr,
            "[block_storage] open(%s) for read: errno %d (%s)\n",
            log_path_.c_str(), errno, std::strerror(errno));
        return false;
    }

    OnDiskHeader hdr;
    if (!read_full(fd, &hdr, sizeof(hdr))) {
        std::fprintf(stderr,
            "[block_storage] short read on %s header\n", log_path_.c_str());
        ::close(fd);
        return false;
    }
    if (hdr.magic != kMagic || hdr.version != kVersion) {
        std::fprintf(stderr,
            "[block_storage] %s: bad magic 0x%lx or version %u\n",
            log_path_.c_str(),
            (unsigned long)hdr.magic, (unsigned)hdr.version);
        ::close(fd);
        return false;
    }

    next_file_id_ = hdr.next_file_id ? hdr.next_file_id : 1u;
    generation_   = hdr.generation;

    entries_.reserve(hdr.entry_count);
    for (uint32_t i = 0; i < hdr.entry_count; ++i) {
        OnDiskEntry de;
        if (!read_full(fd, &de, sizeof(de))) {
            std::fprintf(stderr,
                "[block_storage] short read on %s entry %u\n",
                log_path_.c_str(), i);
            ::close(fd);
            entries_.clear();
            name_to_index_.clear();
            return false;
        }
        if (de.num_shards == 0 || de.num_shards > kGpuFileMaxShards) {
            std::fprintf(stderr,
                "[block_storage] %s entry %u: bogus num_shards=%u\n",
                log_path_.c_str(), i, de.num_shards);
            ::close(fd);
            entries_.clear();
            name_to_index_.clear();
            return false;
        }
        Entry e;
        e.file_id          = de.file_id;
        e.name.assign(de.name,
                      strnlen(de.name, sizeof(de.name)));
        e.total_size       = de.total_size;
        e.tensor_shape[0]  = de.tensor_shape[0];
        e.tensor_shape[1]  = de.tensor_shape[1];
        e.tensor_shape[2]  = de.tensor_shape[2];
        e.num_shards       = de.num_shards;
        e.shard_device_ids.assign(de.shard_device_ids,
                                  de.shard_device_ids + de.num_shards);
        name_to_index_[e.name] = entries_.size();
        entries_.push_back(std::move(e));
    }
    ::close(fd);
    return true;
}

const PersistentGpuFileLog::Entry*
PersistentGpuFileLog::find_by_name(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) return nullptr;
    return &entries_[it->second];
}

uint32_t PersistentGpuFileLog::next_file_id() {
    return next_file_id_++;
}

bool PersistentGpuFileLog::add(Entry e) {
    if (name_to_index_.count(e.name)) return false;
    if (e.num_shards == 0 || e.num_shards > kGpuFileMaxShards) return false;
    if (e.shard_device_ids.size() != e.num_shards)             return false;
    name_to_index_[e.name] = entries_.size();
    entries_.push_back(std::move(e));
    return true;
}

bool PersistentGpuFileLog::remove(uint32_t file_id) {
    // Linear scan + swap-and-pop, mirroring PersistentFileLog::remove.
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

void PersistentGpuFileLog::overwrite_from(const PersistentGpuFileLog& other) {
    entries_       = other.entries_;
    name_to_index_ = other.name_to_index_;
    next_file_id_  = other.next_file_id_;
    generation_    = other.generation_;
    // log_path_ stays ours -- this is "I have stale data, accept peer's".
}

bool PersistentGpuFileLog::persist() {
    return persist_locked();
}

bool PersistentGpuFileLog::persist_locked() {
    std::string tmp = log_path_ + ".tmp";
    int fd = ::open(tmp.c_str(),
                    O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::fprintf(stderr,
            "[block_storage] open(%s) for write: errno %d\n",
            tmp.c_str(), errno);
        return false;
    }

    OnDiskHeader hdr{};
    hdr.magic        = kMagic;
    hdr.version      = kVersion;
    hdr.entry_count  = (uint32_t)entries_.size();
    hdr.generation   = generation_ + 1;          // bump at persist time
    hdr.next_file_id = next_file_id_;

    if (!write_full(fd, &hdr, sizeof(hdr))) goto fail;

    for (const auto& e : entries_) {
        OnDiskEntry de{};
        de.file_id          = e.file_id;
        de.num_shards       = e.num_shards;
        de.total_size       = e.total_size;
        de.tensor_shape[0]  = e.tensor_shape[0];
        de.tensor_shape[1]  = e.tensor_shape[1];
        de.tensor_shape[2]  = e.tensor_shape[2];
        std::strncpy(de.name, e.name.c_str(), sizeof(de.name) - 1);
        for (uint32_t i = 0; i < e.num_shards; ++i) {
            de.shard_device_ids[i] = e.shard_device_ids[i];
        }
        if (!write_full(fd, &de, sizeof(de))) goto fail;
    }

    if (::fsync(fd) != 0) {
        std::fprintf(stderr,
            "[block_storage] fsync(%s): errno %d\n", tmp.c_str(), errno);
        goto fail;
    }
    ::close(fd);

    if (::rename(tmp.c_str(), log_path_.c_str()) != 0) {
        std::fprintf(stderr,
            "[block_storage] rename(%s -> %s): errno %d\n",
            tmp.c_str(), log_path_.c_str(), errno);
        ::unlink(tmp.c_str());
        return false;
    }

    generation_ += 1;
    return true;

fail:
    ::close(fd);
    ::unlink(tmp.c_str());
    return false;
}

}  // namespace tutti
