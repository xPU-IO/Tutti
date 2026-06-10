#ifndef __NVMESERVICE_SERVER_H__
#define __NVMESERVICE_SERVER_H__

/**
 * nvmeservice_server.h -- gRPC service implementation.
 *
 * Thin translation layer: converts between proto messages and the
 * protobuf-free ServiceState API.  All mutable state lives in
 * ServiceState.
 */

#include <memory>

#include <grpcpp/grpcpp.h>
#include "nvmeservice.grpc.pb.h"

#include "nvmeservice_state.h"

namespace nvmeservice {

class NvmeServiceImpl final : public NvmeService::Service {
public:
    explicit NvmeServiceImpl(std::shared_ptr<ServiceState> state);

    grpc::Status ListDevices(grpc::ServerContext* ctx,
                              const Empty* request,
                              DeviceListResponse* response) override;

    grpc::Status Connect(grpc::ServerContext* ctx,
                          const ConnectRequest* request,
                          ConnectResponse* response) override;

    grpc::Status Disconnect(grpc::ServerContext* ctx,
                             const DisconnectRequest* request,
                             DisconnectResponse* response) override;

    grpc::Status Heartbeat(grpc::ServerContext* ctx,
                            grpc::ServerReaderWriter<HeartbeatMsg, HeartbeatMsg>* stream) override;

private:
    std::shared_ptr<ServiceState> state_;
};

} // namespace nvmeservice

#endif // __NVMESERVICE_SERVER_H__
