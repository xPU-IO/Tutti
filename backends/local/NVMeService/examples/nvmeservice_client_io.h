#ifndef NVMESERVICE_CLIENT_IO_H
#define NVMESERVICE_CLIENT_IO_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pure-CUDA payload for the NVMeService client IO smoke.
 *
 * The .cu side has zero gRPC / protobuf includes (nvcc chokes on
 * protobuf's C++17 inline-static-variable traits), so we hand it
 * just the post-Connect metadata it needs.
 */
struct nvmeservice_client_io_args {
    int          cuda_dev;
    const char*  snvme_dev_path;
    uint64_t     bar0_size;
    uint32_t     namespace_id;
    uint32_t     blk_size;
    uint32_t     queue_depth;
    int          granted_queues;  /* informational; smoke uses 1 queue */
    bool         skip_io;
};

/**
 * Returns 0 on success, errno-like on failure.  Diagnostics go to
 * stderr.  Caller has already done the gRPC Connect; this routine
 * runs nvm_ctrl_attach_client + nvm_create_group + map + add_queue +
 * GPU IO (or just attach + create + destroy when skip_io=true).
 */
int run_nvmeservice_client_io(const struct nvmeservice_client_io_args* args);

#ifdef __cplusplus
}
#endif

#endif /* NVMESERVICE_CLIENT_IO_H */
