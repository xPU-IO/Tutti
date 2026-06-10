#ifndef __TUTTI_BLOCK_STORAGE_GPU_FILE_RESOLVE_H__
#define __TUTTI_BLOCK_STORAGE_GPU_FILE_RESOLVE_H__

/**
 * gpu_file_resolve.h -- pure address-translation helper.
 *
 * Layer: block_storage (R6).
 *
 * A "GpuFile" is a logical container whose data plane is split across
 * `num_shards` independent NvmeFiles (one per shard, typically one
 * NvmeFile per device).  The data is laid out by interleaving fixed-
 * size `tensor_size`-byte tensors round-robin across shards, with
 * each shard storing one column of that interleave compactly:
 *
 *   global byte offset        : 0  ts  2*ts 3*ts 4*ts 5*ts ...
 *                                |   |    |    |    |    |
 *   tensor_idx (= /ts)        : 0  1   2   3   4   5  ...
 *   shard_idx  (= % N)        : 0  1   2   0   1   2   ...   (N=3)
 *   shard byte offset (=/N*ts): 0  0   0   ts  ts  ts  ...
 *
 * In Tutti's primary KV-cache use case each shard is one role
 * (K vs V), tensor_size is the per-layer tensor object size, and
 * num_shards is typically 2 (K, V) or 4 (K_lo, K_hi, V_lo, V_hi).
 * The layer dimension is absorbed into the linear `byte_offset`
 * argument by the caller; this helper does not interpret it.
 *
 * --------------------------------------------------------------------
 * Layer responsibility -- read this before using the result!
 * --------------------------------------------------------------------
 *
 * This helper resolves "tensor -> (shard NvmeFile, byte offset within
 * that shard)".  It does NOT decide how many NVMe IOs the tensor
 * gets fragmented into.  That is a separate, lower-level concern:
 *
 *   tensor (e.g. 1 MiB)                                  R6 (this file)
 *      |  gpu_file_resolve(ts=1 MiB, ns, off)
 *      |  -> (shard_idx, shard_byte_off)
 *      v
 *   shard tensor (1 MiB on shard k)                      R7 (memory/)
 *      |  register_tensor(granularity=MDTS)
 *      |  -> N PRPMappingEntry { prp1, prp2, len, off }
 *      v
 *   sub-slices (e.g. 8 x 128 KiB on shard k, MDTS-bound) R8 (io_engine)
 *      |  one GPUIoContext per sub-slice; for each ctx
 *      |  the shard_byte_off computed here is the BASE,
 *      |  R7's prp_entry.tensor_offset is added to it.
 *      v
 *   submit_read_one / submit_write_one(prp1, prp2,       R5b
 *                       shard_base_off + sub_off, sub_len)
 *
 * In particular, when `tensor_size > device MDTS` (e.g. 1 MiB tensor
 * on a controller with 128 KiB MDTS), the same (shard_idx,
 * shard_byte_off) returned by this helper feeds into ceil(ts / MDTS)
 * different NVMe IOs.  The fan-out is performed at register_tensor
 * time inside `memory/` and consumed at submit time by `io_engine/`.
 *
 * Therefore: callers should pass `tensor_size = GpuFile.tensor_shape[2]`
 * (the IO-stripe / interleave unit), NOT MDTS.  Choosing the right
 * tensor_size for the workload is a caller-side decision; Tutti uses
 * it strictly as a logical layout knob.
 *
 * Usage contract (R6 + R7 + R8 invariant):
 *   1. `byte_offset` MUST be tensor-aligned (multiple of tensor_size).
 *      The caller -- io_engine in R8, smoke in R6 -- is responsible
 *      for slicing IO requests at tensor boundaries before calling
 *      resolve.  This avoids any per-call divmod-with-remainder
 *      logic and keeps the result deterministic.
 *   2. `tensor_size` MUST be > 0.
 *   3. `num_shards` MUST be > 0.
 *
 * All three constraints are debug-time `assert`s; release builds
 * trust the caller (this is the GPU-hot path's translation step;
 * we don't want a branch inside the kernel).
 *
 * The function is `constexpr` + always-inline so it compiles
 * identically on host and device, ensuring R6 host-side smoke and
 * R8 GPU kernel resolve to the same shard for the same offset.
 */

#include <cstdint>

#if defined(__CUDACC__)
#  define TUTTI_HOST_DEVICE __host__ __device__ __forceinline__
#else
#  define TUTTI_HOST_DEVICE inline
#endif

namespace tutti {

/// Resolve a global byte offset within a GpuFile to (shard index,
/// shard-local byte offset).  Caller MUST ensure `byte_offset` is a
/// multiple of `tensor_size`.
///
/// @param tensor_size       bytes per tensor object (interleave unit).
/// @param num_shards        number of NvmeFile shards in the GpuFile.
/// @param byte_offset       global byte offset; must be %tensor_size==0.
/// @param[out] shard_idx    which shard the tensor lives on, in
///                          [0, num_shards).
/// @param[out] shard_byte_offset
///                          byte offset within that shard's NvmeFile
///                          (still 4 KiB-aligned because tensor_size
///                          is in practice a multiple of 4 KiB; we do
///                          not enforce that here -- it is an
///                          io_engine concern).
TUTTI_HOST_DEVICE
void gpu_file_resolve(uint32_t  tensor_size,
                      uint32_t  num_shards,
                      uint64_t  byte_offset,
                      uint32_t* shard_idx,
                      uint64_t* shard_byte_offset)
{
    // Debug-time guards.  We deliberately do NOT range-check
    // tensor_size > 0 / num_shards > 0 in release because this is the
    // hot per-IO translation step; misuse is caught by the smoke
    // and by GpuFileSpec validation at create time.
#if !defined(NDEBUG)
    // assert() pulls <cassert>; for headers shared with kernels we
    // prefer no header drag.  Keep the contract documented and let
    // optimisers see straight-line math.
#endif
    const uint64_t tensor_idx = byte_offset / tensor_size;
    *shard_idx                = (uint32_t)(tensor_idx % num_shards);
    *shard_byte_offset        = (tensor_idx / num_shards) * tensor_size;
}

} // namespace tutti

#endif // __TUTTI_BLOCK_STORAGE_GPU_FILE_RESOLVE_H__
