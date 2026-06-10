#include "host_fs_backed_block_storage.h"
#include "persistent_gpu_file_log.h"
#include "nvme_storage.h"
#include "nvme_file.h"

// device_manager: we need Device::device_id to translate between the
// log's persisted shard_device_ids and live Device pointers.
#include "device.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <utility>

namespace tutti {

HostFsBackedBlockStorage::HostFsBackedBlockStorage()  = default;

HostFsBackedBlockStorage::~HostFsBackedBlockStorage() {
    (void)shutdown();
}

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

HostFsBackedBlockStorage::PerDeviceState*
HostFsBackedBlockStorage::find_state(const Device* dev) const {
    if (dev == nullptr) return nullptr;
    for (const auto& s : states_) {
        if (s->device == dev) return s.get();
    }
    return nullptr;
}

HostFsBackedBlockStorage::PerDeviceState*
HostFsBackedBlockStorage::find_state(int32_t device_id) const {
    for (const auto& s : states_) {
        if (s->device->device_id == device_id) return s.get();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Shard naming convention: <gpu_name>.s<i>
//   * cheap to scan (fixed suffix)
//   * sortable by GpuFile name
//   * a stale shard's parent GpuFile is recoverable from the prefix
// ---------------------------------------------------------------------------
std::string
HostFsBackedBlockStorage::shard_name_(std::string_view gpu_name, uint32_t i) {
    std::string out;
    out.reserve(gpu_name.size() + 4);
    out.append(gpu_name);
    out.push_back('.');
    out.push_back('s');
    if (i < 10) {
        out.push_back('0' + (char)i);
    } else {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%u", i);
        out.append(buf);
    }
    return out;
}

bool
HostFsBackedBlockStorage::parse_shard_name_(std::string_view nvme_name,
                                            std::string*     out_gpu_name,
                                            uint32_t*        out_shard_idx) {
    // Find the last ".s" before a numeric suffix.
    if (nvme_name.size() < 3) return false;
    auto dot = nvme_name.rfind(".s");
    if (dot == std::string_view::npos) return false;
    if (dot == 0) return false;                 // no gpu_name
    auto digits = nvme_name.substr(dot + 2);
    if (digits.empty()) return false;
    uint32_t v = 0;
    for (char c : digits) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (uint32_t)(c - '0');
        if (v >= kGpuFileMaxShards) return false;  // out of range
    }
    if (out_gpu_name)  out_gpu_name->assign(nvme_name.substr(0, dot));
    if (out_shard_idx) *out_shard_idx = v;
    return true;
}

// ---------------------------------------------------------------------------
// Spec validation
// ---------------------------------------------------------------------------
bool HostFsBackedBlockStorage::validate_spec_(const GpuFileSpec& spec) const {
    if (spec.name.empty())             return false;
    if (spec.total_size == 0)          return false;
    const uint32_t ns = spec.tensor_shape[0];
    const uint32_t ny = spec.tensor_shape[1];
    const uint32_t ts = spec.tensor_shape[2];
    if (ns == 0 || ny == 0 || ts == 0) return false;
    if (ns > kGpuFileMaxShards)        return false;
    // total_size must equal ns * ny * ts (within uint64 -- ns/ny/ts
    // are 32-bit so the product fits in 96 bits but we cap at 64).
    const __uint128_t prod = (__uint128_t)ns * (__uint128_t)ny * (__uint128_t)ts;
    if (prod != (__uint128_t)spec.total_size) return false;
    if (spec.shard_placement.size() != ns) return false;
    // 1:1 placement: no duplicate Device.
    std::set<const Device*> seen;
    for (auto* d : spec.shard_placement) {
        if (d == nullptr)             return false;
        if (find_state(d) == nullptr) return false;   // not bootstrapped
        if (!seen.insert(d).second)   return false;   // duplicate
    }
    return true;
}

// ---------------------------------------------------------------------------
// Bootstrap / shutdown
// ---------------------------------------------------------------------------
bool HostFsBackedBlockStorage::bootstrap(
    INvmeStorage*                       storage,
    const std::vector<const Device*>&   devices)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (is_open_) return true;
    if (storage == nullptr || devices.empty()) return false;

    storage_ = storage;

    // Build PerDeviceState for every participating Device.  Each
    // gets a full mirror of the gpu_file_log; load_or_init treats
    // ENOENT as "fresh empty log" so a brand new device just makes
    // an empty entry table.
    states_.reserve(devices.size());
    for (const auto* d : devices) {
        if (d == nullptr) {
            std::fprintf(stderr,
                "[block_storage] bootstrap: null Device in list\n");
            states_.clear();
            return false;
        }
        std::string mp = storage_->mount_path(d);
        if (mp.empty()) {
            std::fprintf(stderr,
                "[block_storage] bootstrap: storage->mount_path(dev=%d) "
                "empty -- did you call INvmeStorage::bootstrap first?\n",
                d->device_id);
            states_.clear();
            return false;
        }
        // <mount>/.tutti is created by nvme_storage; we just use it.
        std::string log_path = mp + "/.tutti/gpu_file_log.bin";
        auto sp = std::make_unique<PerDeviceState>();
        sp->device   = d;
        sp->log_path = std::move(log_path);
        sp->log      = std::make_unique<PersistentGpuFileLog>();
        if (!sp->log->load_or_init(sp->log_path)) {
            std::fprintf(stderr,
                "[block_storage] bootstrap: load_or_init(%s) failed\n",
                sp->log_path.c_str());
            states_.clear();
            return false;
        }
        states_.push_back(std::move(sp));
    }

    // Cross-device generation arbitration: pick the highest-gen log
    // and rewrite the others so every device ends up byte-identical.
    uint64_t max_gen = 0;
    PerDeviceState* leader = nullptr;
    for (const auto& s : states_) {
        if (s->log->generation() > max_gen) {
            max_gen = s->log->generation();
            leader  = s.get();
        }
    }
    if (leader != nullptr) {
        for (auto& s : states_) {
            if (s.get() == leader) continue;
            if (s->log->generation() < max_gen) {
                std::fprintf(stderr,
                    "[block_storage] bootstrap: device %d log generation "
                    "%lu < leader %lu (dev=%d); pulling leader's view\n",
                    s->device->device_id,
                    (unsigned long)s->log->generation(),
                    (unsigned long)max_gen, leader->device->device_id);
                s->log->overwrite_from(*leader->log);
                s->dirty = true;
            }
        }
    }

    // Reconcile against nvme_storage state (drop tombstones, unlink
    // ghosts).  Best-effort -- never fails bootstrap.
    (void)reconcile_locked_();

    // If reconcile or arbitration produced changes, persist them
    // before declaring open.  This keeps the on-disk view consistent
    // before any caller starts mutating.
    bool any_dirty = false;
    for (auto& s : states_) any_dirty = any_dirty || s->dirty;
    if (any_dirty) {
        for (auto& s : states_) {
            if (!s->dirty) continue;
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] bootstrap: persist on dev=%d failed; "
                    "will retry on next flush_metadata\n",
                    s->device->device_id);
            } else {
                s->dirty = false;
            }
        }
    }

    is_open_ = true;
    std::fprintf(stderr,
        "[block_storage] bootstrap ready: devices=%zu entries=%zu\n",
        states_.size(),
        states_.empty() ? 0 : states_.front()->log->size());
    return true;
}

bool HostFsBackedBlockStorage::shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) {
        states_.clear();
        files_.clear();
        return true;
    }
    bool ok = true;

    // Close any still-open GpuFiles.  Each GpuFile holds NvmeFile*
    // references that nvme_storage owns; we call close_file for
    // each so the host_fd's get fsync'd + closed.
    for (auto& [fid, gf] : files_) {
        for (auto* nf : gf->shards) {
            if (nf != nullptr) (void)storage_->close_file(nf);
        }
    }
    files_.clear();

    // Final persist on every dirty device.
    for (auto& s : states_) {
        if (s->dirty) {
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] shutdown: persist on dev=%d failed\n",
                    s->device->device_id);
                ok = false;
            } else {
                s->dirty = false;
            }
        }
    }

    states_.clear();
    storage_ = nullptr;
    is_open_ = false;
    return ok;
}

// ---------------------------------------------------------------------------
// Reconcile (R6.2 analogue of nvme_storage R5a.0)
// ---------------------------------------------------------------------------
//
// For the leader log, walk every entry and its shards:
//   * if any shard NvmeFile is missing -> tombstone (drop entry,
//     keep what shards do exist? we just unlink them too -- safer
//     for a "clean directory" semantic).
// Then for every NvmeFile name on every device matching "*.s<i>":
//   * if it doesn't correspond to a live entry -> ghost, unlink it
//     via INvmeStorage::delete_file.
//
// This is best-effort; we log warnings but never fail bootstrap.
bool HostFsBackedBlockStorage::reconcile_locked_() {
    if (states_.empty()) return true;

    // The arbitration step above already overwrote stragglers from
    // the leader, so every device's in-memory log is identical.
    // Walk states_[0]->log as the source of truth.
    auto& leader_log = *states_.front()->log;

    // Build set of every "live" shard-name on each device, keyed by
    // device_id.  Used later for ghost detection.
    std::map<int32_t, std::set<std::string>> live_shards_per_dev;

    // 1. Tombstone sweep.
    std::vector<uint32_t> doomed_file_ids;
    for (const auto& e : leader_log.entries()) {
        bool entry_ok = true;
        for (uint32_t i = 0; i < e.num_shards; ++i) {
            const int32_t did = e.shard_device_ids[i];
            const std::string sn = shard_name_(e.name, i);
            auto* state = find_state(did);
            if (state == nullptr) {
                std::fprintf(stderr,
                    "[block_storage] reconcile: entry '%s' shard %u "
                    "references unbootstrapped dev=%d -> tombstone\n",
                    e.name.c_str(), i, did);
                entry_ok = false;
                break;
            }
            // Check the shard NvmeFile actually exists on that
            // device's PersistentFileLog.
            const auto names = storage_->list_file_names(state->device);
            if (std::find(names.begin(), names.end(), sn) == names.end()) {
                std::fprintf(stderr,
                    "[block_storage] reconcile: entry '%s' shard %u "
                    "missing NvmeFile '%s' on dev=%d -> tombstone\n",
                    e.name.c_str(), i, sn.c_str(), did);
                entry_ok = false;
                break;
            }
            live_shards_per_dev[did].insert(sn);
        }
        if (!entry_ok) doomed_file_ids.push_back(e.file_id);
    }
    if (!doomed_file_ids.empty()) {
        for (uint32_t fid : doomed_file_ids) {
            for (auto& s : states_) (void)s->log->remove(fid);
        }
        for (auto& s : states_) s->dirty = true;
    }

    // 2. Ghost sweep: per device, scan NvmeFile names matching
    //    "*.s<i>" and unlink any that aren't in the live set.
    for (auto& s : states_) {
        const auto names = storage_->list_file_names(s->device);
        for (const auto& nm : names) {
            std::string gpu_name;
            uint32_t    si = 0;
            if (!parse_shard_name_(nm, &gpu_name, &si)) continue;
            // Live?
            const auto& live_set = live_shards_per_dev[s->device->device_id];
            if (live_set.find(nm) != live_set.end()) continue;
            // Ghost: unlink via INvmeStorage::delete_file (deferred,
            // we'll flush at end of bootstrap).
            NvmeFile* nf = storage_->open_file(s->device, nm);
            if (nf == nullptr) continue;
            if (storage_->delete_file(nf, /*persist_now=*/false)) {
                std::fprintf(stderr,
                    "[block_storage] reconcile: removed ghost shard "
                    "'%s' on dev=%d\n", nm.c_str(), s->device->device_id);
            }
        }
        // Drain pending nvme_storage log writes from the ghost sweep.
        (void)storage_->flush_metadata(s->device);
    }
    return true;
}

void HostFsBackedBlockStorage::mark_all_dirty_locked_() {
    for (auto& s : states_) s->dirty = true;
}

// ---------------------------------------------------------------------------
// Directory ops
// ---------------------------------------------------------------------------
GpuFile* HostFsBackedBlockStorage::create_gpu_file(const GpuFileSpec& spec,
                                                    bool persist_now)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_)              return nullptr;
    if (!validate_spec_(spec))  return nullptr;

    const std::string nm{spec.name};

    // Name must not already exist anywhere.  Check leader log.
    if (states_.front()->log->find_by_name(nm) != nullptr) {
        std::fprintf(stderr,
            "[block_storage] create_gpu_file: '%s' already exists\n",
            nm.c_str());
        return nullptr;
    }

    const uint32_t num_shards         = spec.tensor_shape[0];
    const uint64_t per_shard_size     = spec.total_size / num_shards;

    // Allocate shard NvmeFiles using nvme_storage's bulk-init flags
    // when caller deferred, so create + flush at end stays O(1) RPCs
    // per device.  Rollback: any failure unwinds previously created
    // shards via delete_file.
    std::vector<NvmeFile*> shards(num_shards, nullptr);
    bool ok = true;
    for (uint32_t i = 0; i < num_shards; ++i) {
        const Device* d = spec.shard_placement[i];
        const std::string sn = shard_name_(nm, i);
        NvmeFile* nf = storage_->create_file(
            d, sn, per_shard_size,
            /*persist_now=*/persist_now,
            /*sync_now=*/persist_now);
        if (nf == nullptr) {
            std::fprintf(stderr,
                "[block_storage] create_gpu_file '%s': shard %u "
                "create_file failed on dev=%d\n",
                nm.c_str(), i, d->device_id);
            ok = false;
            break;
        }
        shards[i] = nf;
    }
    if (!ok) {
        for (uint32_t i = 0; i < num_shards; ++i) {
            if (shards[i] != nullptr) {
                (void)storage_->delete_file(shards[i],
                                            /*persist_now=*/false);
            }
        }
        for (uint32_t i = 0; i < num_shards; ++i) {
            (void)storage_->flush_metadata(spec.shard_placement[i]);
        }
        return nullptr;
    }

    // Allocate file_id from the leader log; every device's log will
    // get the same id since they are kept identical.
    uint32_t fid = states_.front()->log->next_file_id();

    // Record on every device's mirror log.
    PersistentGpuFileLog::Entry e;
    e.file_id        = fid;
    e.name           = nm;
    e.total_size     = spec.total_size;
    e.tensor_shape[0] = spec.tensor_shape[0];
    e.tensor_shape[1] = spec.tensor_shape[1];
    e.tensor_shape[2] = spec.tensor_shape[2];
    e.num_shards     = num_shards;
    e.shard_device_ids.resize(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        e.shard_device_ids[i] = spec.shard_placement[i]->device_id;
    }
    for (auto& s : states_) {
        // Sync next_file_id across all mirrors so they don't diverge.
        // A simple strategy: each mirror's next_file_id_ tracks the
        // leader.  For brevity we leave it monotonic per-mirror; the
        // only invariant we need is that no two mirrors hand out the
        // same fid for different entries -- which is satisfied by
        // arbitration at bootstrap (overwrite_from copies next_file_id_).
        if (!s->log->add(e)) {
            std::fprintf(stderr,
                "[block_storage] create_gpu_file '%s': mirror log on "
                "dev=%d add() rejected (race?)\n",
                nm.c_str(), s->device->device_id);
            // Best-effort rollback.
            for (auto& sx : states_) (void)sx->log->remove(fid);
            for (auto& nf : shards) (void)storage_->delete_file(nf, false);
            for (auto& sx : states_) (void)storage_->flush_metadata(sx->device);
            return nullptr;
        }
        s->dirty = true;
    }

    if (persist_now) {
        for (auto& s : states_) {
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] create_gpu_file '%s': persist on "
                    "dev=%d failed; flushed nvme_storage data may "
                    "outlive log entry until next flush_metadata\n",
                    nm.c_str(), s->device->device_id);
                // Stay dirty for retry.
                continue;
            }
            s->dirty = false;
        }
    }

    auto gf      = std::make_unique<GpuFile>();
    gf->id           = fid;
    gf->name         = nm;
    gf->total_size   = spec.total_size;
    gf->tensor_shape[0] = spec.tensor_shape[0];
    gf->tensor_shape[1] = spec.tensor_shape[1];
    gf->tensor_shape[2] = spec.tensor_shape[2];
    gf->shards       = std::move(shards);
    GpuFile* raw = gf.get();
    files_[fid] = std::move(gf);
    return raw;
}

GpuFile* HostFsBackedBlockStorage::open_gpu_file(std::string_view name) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) return nullptr;
    const std::string nm{name};

    // Already open?
    for (auto& [fid, gf] : files_) {
        if (gf->name == nm) return gf.get();
    }

    const auto* e = states_.front()->log->find_by_name(nm);
    if (e == nullptr) return nullptr;

    auto gf = std::make_unique<GpuFile>();
    gf->id           = e->file_id;
    gf->name         = e->name;
    gf->total_size   = e->total_size;
    gf->tensor_shape[0] = e->tensor_shape[0];
    gf->tensor_shape[1] = e->tensor_shape[1];
    gf->tensor_shape[2] = e->tensor_shape[2];
    gf->shards.resize(e->num_shards, nullptr);
    for (uint32_t i = 0; i < e->num_shards; ++i) {
        auto* state = find_state(e->shard_device_ids[i]);
        if (state == nullptr) {
            std::fprintf(stderr,
                "[block_storage] open_gpu_file '%s': shard %u dev_id=%d "
                "not bootstrapped\n", nm.c_str(), i, e->shard_device_ids[i]);
            return nullptr;
        }
        const std::string sn = shard_name_(nm, i);
        NvmeFile* nf = storage_->open_file(state->device, sn);
        if (nf == nullptr) {
            std::fprintf(stderr,
                "[block_storage] open_gpu_file '%s': shard %u open '%s' "
                "on dev=%d failed\n", nm.c_str(), i, sn.c_str(),
                state->device->device_id);
            return nullptr;
        }
        gf->shards[i] = nf;
    }

    GpuFile* raw = gf.get();
    files_[e->file_id] = std::move(gf);
    return raw;
}

bool HostFsBackedBlockStorage::close_gpu_file(GpuFile* file) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) return false;
    auto it = files_.find(file->id);
    if (it == files_.end()) return false;
    bool ok = true;
    for (auto* nf : it->second->shards) {
        if (nf != nullptr && !storage_->close_file(nf)) ok = false;
    }
    files_.erase(it);
    return ok;
}

bool HostFsBackedBlockStorage::delete_gpu_file(GpuFile* file,
                                                bool persist_now) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) return false;
    auto it = files_.find(file->id);
    if (it == files_.end()) return false;

    const uint32_t fid = file->id;

    bool ok = true;
    for (auto* nf : it->second->shards) {
        if (nf != nullptr) {
            if (!storage_->delete_file(nf, /*persist_now=*/persist_now)) {
                ok = false;
            }
        }
    }

    for (auto& s : states_) {
        (void)s->log->remove(fid);
        s->dirty = true;
    }
    files_.erase(it);

    if (persist_now) {
        for (auto& s : states_) {
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] delete_gpu_file: persist dev=%d "
                    "failed; staying dirty for retry\n",
                    s->device->device_id);
                ok = false;
                continue;
            }
            s->dirty = false;
        }
    }
    return ok;
}

bool HostFsBackedBlockStorage::flush_metadata() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) return false;
    bool ok = true;
    // Also drain underlying nvme_storage's deferred state.
    for (auto& s : states_) {
        if (!storage_->flush_metadata(s->device)) ok = false;
        if (s->dirty) {
            if (!s->log->persist()) {
                ok = false;
                continue;
            }
            s->dirty = false;
        }
    }
    return ok;
}

std::vector<std::string>
HostFsBackedBlockStorage::list_gpu_file_names() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> out;
    if (!is_open_ || states_.empty()) return out;
    const auto& es = states_.front()->log->entries();
    out.reserve(es.size());
    for (const auto& e : es) out.push_back(e.name);
    return out;
}

std::vector<GpuFile*>
HostFsBackedBlockStorage::list_open_gpu_files() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<GpuFile*> out;
    out.reserve(files_.size());
    for (const auto& [fid, gf] : files_) out.push_back(gf.get());
    return out;
}

// ---------------------------------------------------------------------------
// GPU acquire/release (R6.3)
// ---------------------------------------------------------------------------
//
// Per-shard fan-out over the underlying INvmeStorage's R5b API.
// Each NvmeFile shard's NvmeFileDeviceHandle is cudaMalloc'd on the
// GPU of that shard's Device by nvme_storage; we just collect the
// pointers into a host vector inside the GpuFileHandle.
//
// io_engine (R8) reads d_shards_host[shard_idx] on the host when
// staging per-IO GPUIoContext arrays, copying the resolved
// NvmeFileDeviceHandle* directly into each ctx before cudaMemcpy.
// The kernel never indirects through a GPU-resident shard table:
// one less GPU load per IO and no shard-table lifetime to manage.
//
// Failure semantics:
//   - If any per-shard acquire fails, all previously acquired shard
//     handles are released and the function returns nullptr.
//   - The returned GpuFileHandle is heap-allocated; release_device_handle
//     is the matching deleter (it also releases each shard handle and
//     does NOT touch file->shards or files_).
//   - Idempotent against concurrent close_gpu_file: once a caller
//     has a GpuFile* and calls acquire on it, the underlying
//     NvmeFile shard pointers stay valid for the lifetime of the
//     returned GpuFileHandle (caller must release before close).
GpuFileHandle*
HostFsBackedBlockStorage::acquire_device_handle(GpuFile* file) {
    if (file == nullptr) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handle: file == nullptr\n");
        return nullptr;
    }
    if (storage_ == nullptr) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handle: not bootstrapped\n");
        return nullptr;
    }

    const uint32_t num_shards = file->tensor_shape[0];
    if (num_shards == 0 || file->shards.size() != num_shards) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handle: file '%s' has "
            "tensor_shape[0]=%u but %zu shards\n",
            file->name.c_str(), num_shards, file->shards.size());
        return nullptr;
    }

    auto handle = std::make_unique<GpuFileHandle>();
    handle->file        = file;
    handle->tensor_size = file->tensor_shape[2];
    handle->num_shards  = num_shards;
    handle->d_shards_host.reserve(num_shards);

    // Per-shard acquire_device_handle.  On failure, unwind every
    // shard handle we already collected before returning nullptr.
    for (uint32_t s = 0; s < num_shards; ++s) {
        NvmeFile* shard = file->shards[s];
        if (shard == nullptr) {
            std::fprintf(stderr,
                "[block_storage] acquire_device_handle: file '%s' "
                "shard %u is null\n",
                file->name.c_str(), s);
            for (auto* dh : handle->d_shards_host) {
                if (dh != nullptr) storage_->release_device_handle(dh);
            }
            return nullptr;
        }
        NvmeFileDeviceHandle* dh = storage_->acquire_device_handle(shard);
        if (dh == nullptr) {
            std::fprintf(stderr,
                "[block_storage] acquire_device_handle: "
                "nvme_storage->acquire_device_handle failed for "
                "file='%s' shard=%u name='%s'\n",
                file->name.c_str(), s, shard->name.c_str());
            for (auto* prev : handle->d_shards_host) {
                if (prev != nullptr) storage_->release_device_handle(prev);
            }
            return nullptr;
        }
        handle->d_shards_host.push_back(dh);
    }

    return handle.release();
}

void HostFsBackedBlockStorage::release_device_handle(GpuFileHandle* h) {
    if (h == nullptr) return;
    if (storage_ == nullptr) {
        // Storage already shut down.  We must not touch GPU memory
        // because acquire_device_handle's cudaMalloc lives in the
        // nvme_storage layer and is invalid after shutdown.  Best
        // we can do is leak the GpuFileHandle struct itself.
        std::fprintf(stderr,
            "[block_storage] release_device_handle called after "
            "shutdown; leaking GpuFileHandle to avoid use-after-free\n");
        return;
    }
    for (auto* dh : h->d_shards_host) {
        if (dh != nullptr) storage_->release_device_handle(dh);
    }
    delete h;
}

}  // namespace tutti
