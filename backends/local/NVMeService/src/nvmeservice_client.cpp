#include "nvmeservice_client.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <unistd.h>

namespace nvmeservice {

namespace {

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

NvmeServiceClient::Session::~Session() {
    if (owner != nullptr) {
        owner->release_session(this);
    }
}

// ---------------------------------------------------------------------------
// NvmeServiceClient
// ---------------------------------------------------------------------------

NvmeServiceClient::NvmeServiceClient(const std::string& endpoint)
    : endpoint_(endpoint),
      channel_(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials())),
      stub_(NvmeService::NewStub(channel_))
{}

NvmeServiceClient::~NvmeServiceClient() {
    stop_heartbeat();
}

std::vector<ClientDeviceInfo> NvmeServiceClient::list_devices() {
    std::vector<ClientDeviceInfo> out;

    grpc::ClientContext ctx;
    Empty req;
    DeviceListResponse resp;

    auto status = stub_->ListDevices(&ctx, req, &resp);
    if (!status.ok()) {
        std::fprintf(stderr, "list_devices RPC failed: %s\n",
                     status.error_message().c_str());
        return out;
    }

    out.reserve(resp.devices_size());
    for (const auto& d : resp.devices()) {
        ClientDeviceInfo info;
        info.device_id            = d.device_id();
        info.pci_addr             = d.pci_addr();
        info.snvme_dev_path       = d.snvme_dev_path();
        info.namespace_id         = d.namespace_id();
        info.page_size            = d.page_size();
        info.blk_size             = d.blk_size();
        info.blk_size_log         = d.blk_size_log();
        info.queue_depth          = d.queue_depth();
        info.dstrd                = d.dstrd();
        info.bar0_size            = d.bar0_size();
        info.max_user_qid         = d.max_user_qid();
        info.max_queues_per_group = d.max_queues_per_group();
        info.allowed_gpus.reserve(d.allowed_gpus_size());
        for (const auto& a : d.allowed_gpus()) {
            ClientAllowedGpu row;
            row.cuda_device = a.cuda_device();
            row.mount_path  = a.mount_path();
            info.allowed_gpus.push_back(std::move(row));
        }
        out.push_back(std::move(info));
    }
    return out;
}

std::unique_ptr<NvmeServiceClient::Session>
NvmeServiceClient::connect(int32_t device_id,
                            int32_t cuda_device,
                            int32_t num_queues) {
    grpc::ClientContext ctx;
    ConnectRequest req;
    ConnectResponse resp;

    req.set_device_id(device_id);
    req.set_cuda_device(cuda_device);
    req.set_num_queues(num_queues);
    req.set_client_pid(static_cast<uint32_t>(::getpid()));

    auto status = stub_->Connect(&ctx, req, &resp);
    if (!status.ok()) {
        std::fprintf(stderr, "Connect RPC failed: %s\n",
                     status.error_message().c_str());
        return nullptr;
    }
    if (!resp.error_message().empty()) {
        std::fprintf(stderr, "Connect rejected: %s\n",
                     resp.error_message().c_str());
        return nullptr;
    }

    auto sess = std::make_unique<Session>();
    sess->allocation_id          = resp.allocation_id();
    sess->device_id              = device_id;
    sess->cuda_device            = cuda_device;
    sess->granted_queues         = resp.granted_queues();
    sess->pci_addr               = resp.pci_addr();
    sess->snvme_dev_path         = resp.snvme_dev_path();
    sess->bar0_size              = resp.bar0_size();
    sess->dstrd                  = resp.dstrd();
    sess->namespace_id           = resp.namespace_id();
    sess->page_size              = resp.page_size();
    sess->blk_size               = resp.blk_size();
    sess->blk_size_log           = resp.blk_size_log();
    sess->queue_depth            = resp.queue_depth();
    sess->mount_path             = resp.mount_path();
    sess->heartbeat_interval_sec = resp.heartbeat_interval_sec();
    sess->lease_timeout_sec      = resp.lease_timeout_sec();
    sess->client_pid             = static_cast<uint32_t>(::getpid());
    sess->owner                  = this;

    {
        std::lock_guard<std::mutex> lock(live_mtx_);
        LiveSession ls;
        ls.allocation_id          = sess->allocation_id;
        ls.heartbeat_interval_sec = sess->heartbeat_interval_sec;
        live_sessions_.emplace(sess->allocation_id, std::move(ls));
    }
    ensure_heartbeat_started();

    return sess;
}

void NvmeServiceClient::release_session(Session* sess) {
    if (!sess || sess->allocation_id.empty()) return;

    {
        std::lock_guard<std::mutex> lock(live_mtx_);
        live_sessions_.erase(sess->allocation_id);
    }

    grpc::ClientContext ctx;
    DisconnectRequest req;
    DisconnectResponse resp;
    req.set_allocation_id(sess->allocation_id);
    req.set_client_pid(sess->client_pid);

    auto status = stub_->Disconnect(&ctx, req, &resp);
    if (!status.ok()) {
        std::fprintf(stderr, "Disconnect RPC failed: %s\n",
                     status.error_message().c_str());
    } else if (!resp.success()) {
        std::fprintf(stderr, "Disconnect rejected: %s\n",
                     resp.error_message().c_str());
    }
}

// ---------------------------------------------------------------------------
// Heartbeat
// ---------------------------------------------------------------------------

void NvmeServiceClient::ensure_heartbeat_started() {
    if (hb_running_.exchange(true)) return;
    hb_thread_ = std::thread(&NvmeServiceClient::heartbeat_loop, this);
}

void NvmeServiceClient::stop_heartbeat() {
    if (!hb_running_.exchange(false)) return;
    if (hb_thread_.joinable()) hb_thread_.join();
}

void NvmeServiceClient::heartbeat_loop() {
    while (hb_running_.load()) {
        std::vector<std::string> ids;
        uint32_t interval = 10;
        {
            std::lock_guard<std::mutex> lock(live_mtx_);
            if (live_sessions_.empty()) {
                hb_running_ = false;
                break;
            }
            ids.reserve(live_sessions_.size());
            for (const auto& kv : live_sessions_) {
                ids.push_back(kv.first);
                interval = std::min(interval, kv.second.heartbeat_interval_sec);
            }
        }

        // Open one bidi stream per tick.  Wasteful but trivially
        // correct for low frequencies (default 10s).
        {
            grpc::ClientContext ctx;
            auto stream = stub_->Heartbeat(&ctx);

            for (const auto& aid : ids) {
                HeartbeatMsg msg;
                msg.set_allocation_id(aid);
                msg.set_timestamp_ns(now_ns());
                if (!stream->Write(msg)) break;
            }
            stream->WritesDone();

            HeartbeatMsg resp;
            while (stream->Read(&resp)) {
                if (resp.has_notice() &&
                    resp.notice().kind() == AdminNotice::LEASE_REVOKED) {
                    std::fprintf(stderr,
                        "lease revoked by daemon for allocation %s: %s\n",
                        resp.allocation_id().c_str(),
                        resp.notice().message().c_str());
                    std::lock_guard<std::mutex> lock(live_mtx_);
                    live_sessions_.erase(resp.allocation_id());
                }
            }
            stream->Finish();
        }

        for (uint32_t i = 0; i < interval && hb_running_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

} // namespace nvmeservice
