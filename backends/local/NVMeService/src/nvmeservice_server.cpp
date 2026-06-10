#include "nvmeservice_server.h"

namespace nvmeservice {

NvmeServiceImpl::NvmeServiceImpl(std::shared_ptr<ServiceState> state)
    : state_(std::move(state)) {}

// ---------------------------------------------------------------------------
// ListDevices
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::ListDevices(grpc::ServerContext* /*ctx*/,
                                           const Empty* /*request*/,
                                           DeviceListResponse* response) {
    const auto snaps = state_->list_devices();
    for (const auto& s : snaps) {
        DeviceInfo* di = response->add_devices();
        di->set_device_id(s.device_id);
        di->set_pci_addr(s.pci_addr);
        di->set_snvme_dev_path(s.snvme_dev_path);
        di->set_namespace_id(s.namespace_id);
        di->set_page_size(s.page_size);
        di->set_blk_size(s.blk_size);
        di->set_blk_size_log(s.blk_size_log);
        di->set_queue_depth(s.queue_depth);
        di->set_dstrd(s.dstrd);
        di->set_bar0_size(s.bar0_size);
        di->set_max_user_qid(s.max_user_qid);
        di->set_max_queues_per_group(s.max_queues_per_group);
        for (const auto& a : s.allowed_gpus) {
            AllowedGpu* row = di->add_allowed_gpus();
            row->set_cuda_device(a.cuda_device);
            row->set_mount_path(a.mount_path);
        }
    }
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::Connect(grpc::ServerContext* /*ctx*/,
                                       const ConnectRequest* request,
                                       ConnectResponse* response) {
    auto result = state_->connect(request->device_id(),
                                   request->cuda_device(),
                                   request->num_queues(),
                                   request->client_pid());

    if (!result.success) {
        response->set_error_message(result.error);
        return grpc::Status::OK;
    }

    const auto& g = result.grant;
    response->set_allocation_id(g.allocation_id);
    response->set_pci_addr(g.pci_addr);
    response->set_snvme_dev_path(g.snvme_dev_path);
    response->set_bar0_size(g.bar0_size);
    response->set_granted_queues(g.granted_queues);
    response->set_namespace_id(g.namespace_id);
    response->set_page_size(g.page_size);
    response->set_blk_size(g.blk_size);
    response->set_blk_size_log(g.blk_size_log);
    response->set_queue_depth(g.queue_depth);
    response->set_dstrd(g.dstrd);
    response->set_heartbeat_interval_sec(g.heartbeat_interval_sec);
    response->set_lease_timeout_sec(g.lease_timeout_sec);
    response->set_mount_path(g.mount_path);

    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Disconnect
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::Disconnect(grpc::ServerContext* /*ctx*/,
                                          const DisconnectRequest* request,
                                          DisconnectResponse* response) {
    std::string err;
    bool ok = state_->disconnect(request->allocation_id(),
                                  request->client_pid(),
                                  &err);
    response->set_success(ok);
    if (!ok) response->set_error_message(err);
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Heartbeat (bidi stream)
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::Heartbeat(grpc::ServerContext* /*ctx*/,
                                         grpc::ServerReaderWriter<HeartbeatMsg, HeartbeatMsg>* stream) {
    HeartbeatMsg in;
    while (stream->Read(&in)) {
        std::string err;
        bool ok = state_->update_heartbeat(in.allocation_id(), &err);

        if (!ok) {
            HeartbeatMsg out;
            out.set_allocation_id(in.allocation_id());
            out.set_timestamp_ns(in.timestamp_ns());
            auto* notice = out.mutable_notice();
            notice->set_kind(AdminNotice::LEASE_REVOKED);
            notice->set_message(err);
            stream->Write(out);
            return grpc::Status::OK;
        }

        HeartbeatMsg out;
        out.set_allocation_id(in.allocation_id());
        out.set_timestamp_ns(in.timestamp_ns());
        stream->Write(out);
    }
    return grpc::Status::OK;
}

} // namespace nvmeservice
