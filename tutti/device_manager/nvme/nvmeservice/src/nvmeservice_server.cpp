#include "nvmeservice_server.h"

namespace nvmeservice {
namespace {

// Copy the proto-free canonical snapshot without exposing ServiceState internals.
void fill_resource(const NvmeResourceSnapshot& source, NvmeResource* target) {
    target->set_device_id(source.device_id);
    target->set_pci_bdf(source.pci_bdf);
    target->set_chrdev_minor(source.chrdev_minor);
    target->set_chrdev_path(source.chrdev_path);
    target->set_block_path(source.block_path);
    target->set_backing_mount_path(source.backing_mount_path);
    target->set_namespace_id(source.namespace_id);
    target->set_page_size(source.page_size);
    target->set_logical_block_size(source.logical_block_size);
    target->set_logical_block_size_log(source.logical_block_size_log);
    target->set_queue_depth(source.queue_depth);
    target->set_dstrd(source.dstrd);
    target->set_bar0_size(source.bar0_size);
    target->set_max_data_size(source.max_data_size);
    target->set_max_user_qid(source.max_user_qid);
    target->set_kernel_io_qps(source.kernel_io_qps);
    target->set_controller_queue_capacity(source.controller_queue_capacity);
    target->set_max_queues_per_group(source.max_queues_per_group);
    target->set_reserved_queues(source.reserved_queues);
    target->set_available_queues(source.available_queues);
    for (int32_t accel_id : source.allowed_accel_ids) {
        target->add_allowed_accel_ids(accel_id);
    }
    target->set_available(source.available);
    target->set_diagnostic(source.diagnostic);
    target->set_heartbeat_interval_sec(source.heartbeat_interval_sec);
    target->set_lease_timeout_sec(source.lease_timeout_sec);
}

// A slice carries the complete owner/path/queue facts needed by a client
// attach. Striped slices remain children of one external allocation ID.
void fill_slice_proto(const NvmeSliceGrant& source, NvmeSlice* target) {
    target->set_device_id(source.device_id);
    target->set_accel_id(source.accel_id);
    target->set_pci_bdf(source.pci_bdf);
    target->set_chrdev_minor(source.chrdev_minor);
    target->set_chrdev_path(source.chrdev_path);
    target->set_block_path(source.block_path);
    target->set_backing_mount_path(source.backing_mount_path);
    target->set_view_path(source.view_path);
    target->set_namespace_id(source.namespace_id);
    target->set_page_size(source.page_size);
    target->set_logical_block_size(source.logical_block_size);
    target->set_logical_block_size_log(source.logical_block_size_log);
    target->set_queue_depth(source.queue_depth);
    target->set_dstrd(source.dstrd);
    target->set_bar0_size(source.bar0_size);
    target->set_max_data_size(source.max_data_size);
    target->set_controller_queue_capacity(source.controller_queue_capacity);
    target->set_granted_queues(source.granted_queues);
    target->set_max_queues_per_group(source.max_queues_per_group);
    for (int32_t accel_id : source.allowed_accel_ids) {
        target->add_allowed_accel_ids(accel_id);
    }
    target->set_heartbeat_interval_sec(source.heartbeat_interval_sec);
    target->set_lease_timeout_sec(source.lease_timeout_sec);
}

} // namespace

NvmeServiceImpl::NvmeServiceImpl(std::shared_ptr<ServiceState> state)
    : state_(std::move(state)) {}

// ---------------------------------------------------------------------------
// Canonical accelerator/resource listing and allocation RPCs
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::ListAccelerators(
    grpc::ServerContext*, const Empty*, AcceleratorListResponse* response) {
    for (const auto& accelerator : state_->list_accelerators()) {
        auto* row = response->add_accelerators();
        row->set_accel_id(accelerator.accel_id);
        row->set_view_root(accelerator.view_root);
    }
    return grpc::Status::OK;
}

grpc::Status NvmeServiceImpl::ListNvmeResources(
    grpc::ServerContext*, const Empty*, NvmeResourceListResponse* response) {
    for (const auto& resource : state_->list_nvme_resources()) {
        fill_resource(resource, response->add_resources());
    }
    return grpc::Status::OK;
}

grpc::Status NvmeServiceImpl::AcquireNvmeSlices(
    grpc::ServerContext*, const AcquireNvmeSlicesRequest* request,
    AcquireNvmeSlicesResponse* response) {
    AcquireRequest state_request;
    state_request.accel_id = request->accel_id();
    switch (request->selection()) {
        case NVME_SELECTION_ALLOWED:
            state_request.selection = SelectionMode::Allowed;
            break;
        case NVME_SELECTION_EXPLICIT:
            state_request.selection = SelectionMode::Explicit;
            break;
        case NVME_SELECTION_STRIPED:
            state_request.selection = SelectionMode::Striped;
            break;
        default:
            response->set_error_message("unknown selection mode");
            return grpc::Status::OK;
    }
    state_request.device_ids.assign(request->device_ids().begin(),
                                    request->device_ids().end());
    state_request.queues_per_controller = request->queues_per_controller();
    state_request.client_pid = request->client_pid();
    const auto result = state_->acquire(state_request);
    if (!result.success) {
        response->set_error_message(result.error);
        return grpc::Status::OK;
    }
    response->set_allocation_id(result.grant.allocation_id);
    for (const auto& slice : result.grant.slices) {
        fill_slice_proto(slice, response->add_slices());
    }
    return grpc::Status::OK;
}

grpc::Status NvmeServiceImpl::Release(grpc::ServerContext*,
                                      const ReleaseRequest* request,
                                      ReleaseResponse* response) {
    const auto result = state_->release(request->allocation_id());
    response->set_success(result.success);
    response->set_already_released(result.already_released);
    response->set_error_message(result.error);
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Legacy wire-compatible adapters
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::ListDevices(grpc::ServerContext*, const Empty*,
                                          DeviceListResponse* response) {
    for (const auto& resource : state_->list_nvme_resources()) {
        auto* device = response->add_devices();
        device->set_device_id(resource.device_id);
        device->set_pci_addr(resource.pci_bdf);
        device->set_snvme_dev_path(resource.chrdev_path);
        device->set_namespace_id(resource.namespace_id);
        device->set_page_size(resource.page_size);
        device->set_blk_size(resource.logical_block_size);
        device->set_blk_size_log(resource.logical_block_size_log);
        device->set_queue_depth(resource.queue_depth);
        device->set_dstrd(resource.dstrd);
        device->set_bar0_size(resource.bar0_size);
        device->set_max_user_qid(resource.max_user_qid);
        device->set_max_queues_per_group(resource.max_queues_per_group);
        for (int32_t accel_id : resource.allowed_accel_ids) {
            auto* allowed = device->add_allowed_gpus();
            allowed->set_cuda_device(accel_id);
            const auto view = resource.view_paths.find(accel_id);
            if (view != resource.view_paths.end()) {
                allowed->set_mount_path(view->second);
            }
        }
    }
    return grpc::Status::OK;
}

grpc::Status NvmeServiceImpl::Connect(grpc::ServerContext*,
                                      const ConnectRequest* request,
                                      ConnectResponse* response) {
    const auto result = state_->connect(
        request->device_id(), request->cuda_device(), request->num_queues(),
        request->client_pid());
    if (!result.success) {
        response->set_error_message(result.error);
        return grpc::Status::OK;
    }
    const auto& grant = result.grant;
    response->set_allocation_id(grant.allocation_id);
    response->set_pci_addr(grant.pci_addr);
    response->set_snvme_dev_path(grant.snvme_dev_path);
    response->set_bar0_size(grant.bar0_size);
    response->set_granted_queues(grant.granted_queues);
    response->set_namespace_id(grant.namespace_id);
    response->set_page_size(grant.page_size);
    response->set_blk_size(grant.blk_size);
    response->set_blk_size_log(grant.blk_size_log);
    response->set_queue_depth(grant.queue_depth);
    response->set_dstrd(grant.dstrd);
    response->set_max_data_size(grant.max_data_size);
    response->set_heartbeat_interval_sec(grant.heartbeat_interval_sec);
    response->set_lease_timeout_sec(grant.lease_timeout_sec);
    response->set_mount_path(grant.mount_path);
    return grpc::Status::OK;
}

grpc::Status NvmeServiceImpl::Disconnect(grpc::ServerContext*,
                                         const DisconnectRequest* request,
                                         DisconnectResponse* response) {
    std::string error;
    const bool success = state_->disconnect(
        request->allocation_id(), request->client_pid(), &error);
    response->set_success(success);
    response->set_error_message(error);
    return grpc::Status::OK;
}

// Heartbeat updates the same allocation lease that Release and the reaper
// operate on; a revoked allocation is reported through the existing notice.
grpc::Status NvmeServiceImpl::Heartbeat(
    grpc::ServerContext*,
    grpc::ServerReaderWriter<HeartbeatMsg, HeartbeatMsg>* stream) {
    HeartbeatMsg input;
    while (stream->Read(&input)) {
        std::string error;
        const bool success = state_->update_heartbeat(input.allocation_id(),
                                                       &error);
        HeartbeatMsg output;
        output.set_allocation_id(input.allocation_id());
        output.set_timestamp_ns(input.timestamp_ns());
        if (!success) {
            output.mutable_notice()->set_kind(AdminNotice::LEASE_REVOKED);
            output.mutable_notice()->set_message(error);
        }
        stream->Write(output);
        if (!success) break;
    }
    return grpc::Status::OK;
}

} // namespace nvmeservice
