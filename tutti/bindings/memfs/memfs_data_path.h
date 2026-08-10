#pragma once

// tutti/bindings/memfs/memfs_data_path.h
//
// SAMPLE-ONLY DataPath for the memfs binding.  Implements the full
// DataPath SPI lifecycle against an in-memory backing buffer.
//
// This is a community-extension sample, NOT a production backend.  It
// proves that a new DataPath can be added without modifying core Runtime
// or SPI headers.
//
// Header-only C++17.  Depends only on public/SPI headers, the memfs
// binding header, and the standard library.

#include <tutti/spi/data_path.h>
#include <tutti/status.h>
#include <tutti/io_types.h>
#include "tutti/bindings/memfs/binding.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tutti::binding::memfs {

// -------------------------------------------------------------------------
// MemfsDataPath — in-memory DataPath for the memfs sample.
//
// Lifecycle:
//   open()         — extracts MemfsPayload, stores backing buffer ref
//   register_memory() — stores caller memory address
//   submit()       — synchronous memcpy, auto-completes immediately
//   progress()     — no-op (all ops are synchronous)
//   query()        — returns COMPLETED
//   release()      — erases op record
//   close()        — releases backing buffer ref
//
// Synchronous completion means submit() creates an op that is already
// COMPLETED.  progress() has nothing to do.  This matches the semantics
// of an in-memory "device" with zero latency.
// -------------------------------------------------------------------------

class MemfsDataPath : public tutti::DataPath {
public:
    MemfsDataPath() {
        caps_.name = "memfs";
        caps_.source_api_version = 1;
        caps_.supports_host_execution = true;
        caps_.supports_device_execution = false;
        caps_.supports_host_memory = true;
        caps_.supports_device_memory = false;
        caps_.supports_direct = true;
        caps_.supports_staged = false;
        caps_.supports_read = true;
        caps_.supports_write = true;
        caps_.target_alignment_bytes = 1;
        caps_.memory_alignment_bytes = 1;
        caps_.length_alignment_bytes = 1;
        caps_.max_single_io_bytes = 1ull << 30;  // 1 GiB
        caps_.max_batch_requests = 64;
        caps_.max_batch_bytes = 1ull << 30;
        caps_.max_in_flight_operations = 256;
        caps_.supports_scatter_gather = false;
        caps_.registration_scope = RegistrationScope::PER_TARGET;
        caps_.progress_model = ProgressModel::HOST_POLL;
        caps_.supports_multi_stream = false;
        caps_.max_concurrent_streams = 1;
        caps_.max_concurrent_operations = 256;
        caps_.bound_accel_id = -1;
    }

    ~MemfsDataPath() override = default;

    // ---- DataPath SPI ----

    const DataPathCapabilities& capabilities() const override {
        return caps_;
    }

    Status initialize(const DataPathConfig& config,
                      ResourceProvider& /*resources*/) override {
        (void)config;
        return Status::Ok();
    }

    Status shutdown(std::uint64_t /*timeout_ns*/) override {
        std::lock_guard<std::mutex> lock(mtx_);
        targets_.clear();
        memory_.clear();
        ops_.clear();
        return Status::Ok();
    }

    Result<DataPathTarget> open(const ResolvedTarget& target) override {
        auto payload = view_payload(target);
        if (!payload.ok()) {
            return Result<DataPathTarget>::Failure(payload.status());
        }
        // Raw pointer is safe: the Runtime keeps the ResolvedTarget (and
        // its payload) alive from open() until close() is called.  The
        // inflight_count mechanism prevents close() while submit() is
        // active, so the pointer is never used after the payload is freed.
        MemfsPayload* p = const_cast<MemfsPayload*>(payload.value());

        std::uint64_t tok = next_target_++;
        std::lock_guard<std::mutex> lock(mtx_);
        targets_[tok] = TargetRecord{
            p,
            target.logical_size(),
            "memfs-domain-" + std::to_string(tok),
        };
        return detail::SpiIdentityMint::mint<detail::DataPathTargetTag>(tok, 1);
    }

    Status close(DataPathTarget target) override {
        std::lock_guard<std::mutex> lock(mtx_);
        targets_.erase(target.token());
        return Status::Ok();
    }

    Result<RegistrationDomainKey> registration_domain(
        DataPathTarget target) const override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = targets_.find(target.token());
        if (it == targets_.end()) {
            return Result<RegistrationDomainKey>::Failure(
                Status(StatusCode::NOT_FOUND, "unknown target"));
        }
        return RegistrationDomainKey{it->second.domain_key};
    }

    Result<DataPathMemory> register_memory(
        const DataPathMemoryView& view,
        const RegistrationDomainKey& /*domain*/) override {
        if (view.base == nullptr || view.size_bytes == 0) {
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memory view must be non-null/non-zero"));
        }
        std::uint64_t tok = next_memory_++;
        std::lock_guard<std::mutex> lock(mtx_);
        memory_[tok] = MemoryRecord{view.base, view.size_bytes};
        return detail::SpiIdentityMint::mint<detail::DataPathMemoryTag>(tok, 1);
    }

    Status unregister_memory(DataPathMemory mem) override {
        std::lock_guard<std::mutex> lock(mtx_);
        memory_.erase(mem.token());
        return Status::Ok();
    }

    SubmitOutcome submit(const DataPathRequest* requests,
                         std::size_t count,
                         const HostSubmitContext& /*ctx*/) override {
        SubmitOutcome out;
        out.initial_states.resize(count);

        if (count == 0) {
            out.status = Status::Ok();
            return out;
        }
        if (requests == nullptr) {
            out.status = Status(StatusCode::INVALID_ARGUMENT, "null requests");
            return out;
        }

        std::lock_guard<std::mutex> lock(mtx_);

        // Validate and execute each request synchronously.
        std::size_t accepted = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const DataPathRequest& req = requests[i];

            // Look up target.
            auto tit = targets_.find(req.target.token());
            if (tit == targets_.end()) {
                out.initial_states[i].state = RequestState::REJECTED;
                out.initial_states[i].status = Status(
                    StatusCode::NOT_FOUND, "unknown target in request");
                continue;
            }

            // Look up memory.
            auto mit = memory_.find(req.memory.token());
            if (mit == memory_.end()) {
                out.initial_states[i].state = RequestState::REJECTED;
                out.initial_states[i].status = Status(
                    StatusCode::NOT_FOUND, "unknown memory in request");
                continue;
            }

            const TargetRecord& tr = tit->second;
            const MemoryRecord& mr = mit->second;

            // Bounds checks.
            if (req.intent.target_offset > tr.logical_size ||
                req.intent.length > tr.logical_size - req.intent.target_offset) {
                out.initial_states[i].state = RequestState::REJECTED;
                out.initial_states[i].status = Status(
                    StatusCode::OUT_OF_RANGE, "target bounds exceeded");
                continue;
            }
            if (req.intent.memory_offset > mr.size ||
                req.intent.length > mr.size - req.intent.memory_offset) {
                out.initial_states[i].state = RequestState::REJECTED;
                out.initial_states[i].status = Status(
                    StatusCode::OUT_OF_RANGE, "memory bounds exceeded");
                continue;
            }

            // Execute the transfer.
            std::vector<std::uint8_t>& backing = tr.payload->mutable_backing();
            if (req.intent.direction == IoDirection::READ) {
                std::memcpy(
                    static_cast<std::uint8_t*>(mr.address) + req.intent.memory_offset,
                    backing.data() + req.intent.target_offset,
                    static_cast<std::size_t>(req.intent.length));
            } else {
                std::memcpy(
                    backing.data() + req.intent.target_offset,
                    static_cast<const std::uint8_t*>(mr.address) + req.intent.memory_offset,
                    static_cast<std::size_t>(req.intent.length));
            }

            out.initial_states[i].state = RequestState::ACCEPTED;
            out.initial_states[i].status = Status::Ok();
            ++accepted;
        }

        if (accepted == 0) {
            out.status = Status(StatusCode::INVALID_ARGUMENT,
                                "all requests rejected");
            return out;
        }

        // Mint an op that is already COMPLETED (synchronous).
        std::uint64_t op_tok = next_op_++;
        ops_[op_tok] = OpRecord{
            OpState::COMPLETED,
            Status::Ok(),
            0,  // bytes_transferred filled below
        };
        // Count total bytes for accepted requests.
        for (std::size_t i = 0; i < count; ++i) {
            if (out.initial_states[i].state == RequestState::ACCEPTED) {
                ops_[op_tok].bytes_transferred += requests[i].intent.length;
            }
        }

        out.op = detail::SpiIdentityMint::mint<detail::DataPathOpTag>(op_tok, 1);
        out.status = (accepted == count)
            ? Status::Ok()
            : Status(StatusCode::INVALID_ARGUMENT,
                     "partial submit: some requests rejected");
        return out;
    }

    Result<ProgressResult> progress(ProgressBudget /*budget*/) override {
        // All ops are synchronous (COMPLETED at submit time).
        ProgressResult result;
        result.more_work_likely = false;
        return result;
    }

    Result<DataPathSnapshot> query(DataPathOp op) const override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = ops_.find(op.token());
        if (it == ops_.end()) {
            return Result<DataPathSnapshot>::Failure(
                Status(StatusCode::NOT_FOUND, "unknown op"));
        }
        const OpRecord& rec = it->second;
        return DataPathSnapshot{rec.state, rec.terminal_status,
                                rec.bytes_transferred};
    }

    Status release(DataPathOp op) override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = ops_.find(op.token());
        if (it == ops_.end()) {
            return Status(StatusCode::NOT_FOUND, "unknown op");
        }
        if (it->second.state == OpState::IN_FLIGHT) {
            return Status(StatusCode::BUSY, "op not terminal");
        }
        ops_.erase(it);
        return Status::Ok();
    }

private:
    struct TargetRecord {
        MemfsPayload* payload = nullptr;  // raw; valid open()→close()
        std::uint64_t logical_size = 0;
        std::string domain_key;
    };
    struct MemoryRecord {
        void* address = nullptr;
        std::uint64_t size = 0;
    };
    struct OpRecord {
        OpState state = OpState::IN_FLIGHT;
        Status terminal_status;
        std::uint64_t bytes_transferred = 0;
    };

    DataPathCapabilities caps_;

    mutable std::mutex mtx_;
    std::uint64_t next_target_ = 1;
    std::uint64_t next_memory_ = 1;
    std::uint64_t next_op_ = 1;
    std::unordered_map<std::uint64_t, TargetRecord> targets_;
    std::unordered_map<std::uint64_t, MemoryRecord> memory_;
    std::unordered_map<std::uint64_t, OpRecord> ops_;
};

} // namespace tutti::binding::memfs
