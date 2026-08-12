// tests/header_hygiene/header_hygiene_test.cpp
//
// Phase 3 header-hygiene consumer test.
//
// This translation unit links ONLY tutti_api (the public usage-requirements
// target).  It proves two things:
//
//   POSITIVE: public headers are reachable and usable — <tutti/status.h>,
//             <tutti/io_types.h>, <tutti/memory_types.h>,
//             <tutti/cuda_like.h>, <tutti/spi/...>, <tutti/storage_runtime.h>.
//
//   NEGATIVE: local-NVMe private headers are NOT reachable via __has_include.
//             If any private header is reachable, a compile-time #error fires.
//
// The test adds no include paths of its own; it relies solely on the
// INTERFACE_INCLUDE_DIRECTORIES propagated by tutti_api -> tutti_cuda_like.

// ---------------------------------------------------------------------------
// POSITIVE: public headers must be reachable
// ---------------------------------------------------------------------------

#include <tutti/status.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/cuda_like.h>
#include <tutti/config/tutti_runtime_config_parser.h>
#include <tutti/config/tutti_runtime_spec.h>
#include <tutti/resource.h>
#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>

#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// NEGATIVE: local-NVMe private headers must NOT be reachable
//
// __has_include is the C++17 standard way to probe whether a header is on the
// include search path.  If any of these resolve, it means a private include
// directory leaked into the public target's INTERFACE_INCLUDE_DIRECTORIES
// (or was inherited from a parent directory scope).
// ---------------------------------------------------------------------------

#if defined(__has_include)

#  if __has_include(<tutti/tutti_runtime/tutti_runtime_internal.h>)
#    error "HYGIENE VIOLATION: Runtime test seam is publicly reachable"
#  endif
#  if __has_include(<tutti/resource/nvme/nvme_resource.h>)
#    error "HYGIENE VIOLATION: NVMe Resource implementation is publicly reachable"
#  endif

// --- libnvm private headers (tutti/device_manager/nvme/libnvm/include/) ---
#  if __has_include(<nvm_types.h>)
#    error "HYGIENE VIOLATION: <nvm_types.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_dma.h>)
#    error "HYGIENE VIOLATION: <nvm_dma.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_queue.h>)
#    error "HYGIENE VIOLATION: <nvm_queue.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_ctrl.h>)
#    error "HYGIENE VIOLATION: <nvm_ctrl.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_io.h>)
#    error "HYGIENE VIOLATION: <nvm_io.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_error.h>)
#    error "HYGIENE VIOLATION: <nvm_error.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_util.h>)
#    error "HYGIENE VIOLATION: <nvm_util.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_admin.h>)
#    error "HYGIENE VIOLATION: <nvm_admin.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_cmd.h>)
#    error "HYGIENE VIOLATION: <nvm_cmd.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_aq.h>)
#    error "HYGIENE VIOLATION: <nvm_aq.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_rpc.h>)
#    error "HYGIENE VIOLATION: <nvm_rpc.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<nvm_parallel_queue.h>)
#    error "HYGIENE VIOLATION: <nvm_parallel_queue.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<ctrl.h>)
#    error "HYGIENE VIOLATION: <ctrl.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<queue.h>)
#    error "HYGIENE VIOLATION: <queue.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<buffer.h>)
#    error "HYGIENE VIOLATION: <buffer.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<ioctl.h>)
#    error "HYGIENE VIOLATION: <ioctl.h> (libnvm private) is reachable from public include path"
#  endif
#  if __has_include(<map.h>)
#    error "HYGIENE VIOLATION: <map.h> (libnvm/snvm private) is reachable from public include path"
#  endif

// --- libnvm internal headers (tutti/device_manager/nvme/libnvm/src/) ---
#  if __has_include(<dma.h>)
#    error "HYGIENE VIOLATION: <dma.h> (libnvm src private) is reachable from public include path"
#  endif
#  if __has_include(<regs.h>)
#    error "HYGIENE VIOLATION: <regs.h> (libnvm src private) is reachable from public include path"
#  endif
#  if __has_include(<lib_ctrl.h>)
#    error "HYGIENE VIOLATION: <lib_ctrl.h> (libnvm src private) is reachable from public include path"
#  endif
#  if __has_include(<rpc.h>)
#    error "HYGIENE VIOLATION: <rpc.h> (libnvm src private) is reachable from public include path"
#  endif

// --- nvmeservice private headers (backends/local/NVMeService/src/) ---
#  if __has_include(<nvmeservice_client.h>)
#    error "HYGIENE VIOLATION: <nvmeservice_client.h> (nvmeservice private) is reachable from public include path"
#  endif
#  if __has_include(<nvmeservice_server.h>)
#    error "HYGIENE VIOLATION: <nvmeservice_server.h> (nvmeservice private) is reachable from public include path"
#  endif
#  if __has_include(<nvmeservice_config.h>)
#    error "HYGIENE VIOLATION: <nvmeservice_config.h> (nvmeservice private) is reachable from public include path"
#  endif
#  if __has_include(<nvmeservice_state.h>)
#    error "HYGIENE VIOLATION: <nvmeservice_state.h> (nvmeservice private) is reachable from public include path"
#  endif

// --- snvme kernel module private headers ---
#  if __has_include(<peer_memory.h>)
#    error "HYGIENE VIOLATION: <peer_memory.h> (snvme kernel private) is reachable from public include path"
#  endif
#  if __has_include(<compat.h>)
#    error "HYGIENE VIOLATION: <compat.h> (snvme kernel private) is reachable from public include path"
#  endif
#  if __has_include(<nvfs-core.h>)
#    error "HYGIENE VIOLATION: <nvfs-core.h> (snvme kernel private) is reachable from public include path"
#  endif

// --- CUDA kernel private headers (data_paths/local_nvme/) ---
#  if __has_include(<local_nvme_data_path.h>)
#    error "HYGIENE VIOLATION: <local_nvme_data_path.h> (CUDA kernel private) is reachable from public include path"
#  endif

#endif // __has_include

// ---------------------------------------------------------------------------
// Minimal runtime check: instantiate a public type to prove the positive
// includes are not just syntactically valid but semantically usable.
// ---------------------------------------------------------------------------

int main() {
    tutti::Status s = tutti::Status::Ok();
    if (!s.ok()) {
        std::printf("header_hygiene_test: FAIL — Status::Ok() not ok\n");
        return 1;
    }

    tutti::IoRequest req;
    req.direction = tutti::IoDirection::READ;
    req.length = 4096;

    tutti::MemoryHandle mh{};
    tutti::TargetHandle th{};
    req.memory = mh;
    req.target = th;

    tutti::config::TuttiRuntimeSpec spec;
    tutti::ResourceInfo resource_info;
    if (!spec.storage.resources.empty() || !resource_info.id.empty()) {
        return 1;
    }

    std::printf("header_hygiene_test: all checks passed\n");
    return 0;
}
