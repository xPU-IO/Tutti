// gpu_file_resolve_test.cpp -- R6.1 unit test for gpu_file_resolve.
//
// Plain C++ host program (no CUDA, no nvme_storage); just exercises
// the inline math in gpu_file_resolve.h to make sure the host- and
// (future) device-side translations match what legacy
// libgeminifs/gpu_controller.cu's `__get_nvmeofst` does.
//
// We replicate legacy's intent in a reference function and assert
// our helper produces identical output across a spread of shapes.
//
// Build: hooked up via block_storage/CMakeLists.txt as the
//   `block_storage_resolve_test` target.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_file_resolve.h"

namespace {

// ----------------------------------------------------------------------------
// Reference: identical to the math at
//   filesystems/ext4/libgeminifs/gpu_controller.cu:1567-1569,
// for the (tensor_size-aligned) case we contractually require.
//
// gpu_blk    = byte_off / tensor_size
// shard_idx  = gpu_blk % num_shards
// shard_off  = (gpu_blk / num_shards) * tensor_size
// ----------------------------------------------------------------------------
void reference(uint32_t  ts, uint32_t ns, uint64_t off,
               uint32_t* sidx, uint64_t* soff)
{
    uint64_t gpu_blk = off / ts;
    *sidx            = (uint32_t)(gpu_blk % ns);
    *soff            = (gpu_blk / ns) * ts;
}

// ----------------------------------------------------------------------------
struct Shape { uint32_t tensor_size; uint32_t num_shards; uint64_t total; };

bool run_one(const Shape& sh) {
    // Walk every tensor-aligned offset in the file.
    bool ok = true;
    for (uint64_t off = 0; off < sh.total; off += sh.tensor_size) {
        uint32_t  si_a = 0, si_b = 0;
        uint64_t  so_a = 0, so_b = 0;
        tutti::gpu_file_resolve(sh.tensor_size, sh.num_shards, off,
                                &si_a, &so_a);
        reference(sh.tensor_size, sh.num_shards, off, &si_b, &so_b);
        if (si_a != si_b || so_a != so_b) {
            std::fprintf(stderr,
                "[FAIL] ts=%u ns=%u off=%lu : got (s=%u,o=%lu) "
                "expected (s=%u,o=%lu)\n",
                sh.tensor_size, sh.num_shards, (unsigned long)off,
                si_a, (unsigned long)so_a,
                si_b, (unsigned long)so_b);
            ok = false;
        }
    }
    return ok;
}

// Spot-check round-trip: walking each shard's local addresses in
// order should recover a strictly increasing sequence of global
// offsets, and the union should cover every tensor_size-aligned
// global offset exactly once.
bool run_round_trip(const Shape& sh) {
    const uint64_t n_tensors        = sh.total / sh.tensor_size;
    const uint64_t per_shard_total  = (n_tensors / sh.num_shards) * sh.tensor_size;
    // Build a coverage bitset over global tensor indices.
    std::vector<uint8_t> seen(n_tensors, 0);

    for (uint32_t s = 0; s < sh.num_shards; ++s) {
        // For each shard-local tensor i, the global offset is:
        //   gpu_blk = i * num_shards + s
        //   global  = gpu_blk * tensor_size
        for (uint64_t local_t = 0; local_t * sh.tensor_size < per_shard_total; ++local_t) {
            uint64_t global_t = local_t * sh.num_shards + s;
            if (global_t >= n_tensors) break;
            if (seen[global_t]) {
                std::fprintf(stderr,
                    "[FAIL] round-trip ts=%u ns=%u: tensor %lu seen twice\n",
                    sh.tensor_size, sh.num_shards, (unsigned long)global_t);
                return false;
            }
            seen[global_t] = 1;

            // Forward direction: resolve(global_off) must yield (s, local_off).
            uint32_t  si  = 0;
            uint64_t  soff = 0;
            tutti::gpu_file_resolve(sh.tensor_size, sh.num_shards,
                                    global_t * sh.tensor_size, &si, &soff);
            uint64_t local_off = local_t * sh.tensor_size;
            if (si != s || soff != local_off) {
                std::fprintf(stderr,
                    "[FAIL] round-trip ts=%u ns=%u global_t=%lu: resolve "
                    "yielded (s=%u,off=%lu); expected (s=%u,off=%lu)\n",
                    sh.tensor_size, sh.num_shards,
                    (unsigned long)global_t, si, (unsigned long)soff,
                    s, (unsigned long)local_off);
                return false;
            }
        }
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
        if (!seen[i]) {
            std::fprintf(stderr,
                "[FAIL] round-trip ts=%u ns=%u: tensor %lu uncovered\n",
                sh.tensor_size, sh.num_shards, (unsigned long)i);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    // A spread of realistic shapes:
    //   tensor_size: 4 KiB (smallest NVMe block) -> 256 KiB
    //   num_shards : 1 (degenerate) ... kGpuFileMaxShards (4)
    //   total_size : sized so it's an integer multiple of ts*ns
    const std::vector<Shape> shapes = {
        // tensor_size,     num_shards,    total
        { 4096,             1,             4096 * 16 },             // 64 KiB
        { 4096,             2,             4096 * 64 },             // 256 KiB, K/V split
        { 4096,             3,             4096 * 96 },             // 384 KiB
        { 4096,             4,             4096 * 256 },            // 1 MiB
        { 64 * 1024,        2,             64 * 1024 * 32 },        // 2 MiB, large tensor
        { 256 * 1024,       4,             256ULL * 1024 * 1024 },  // 256 MiB
        // Edge: single tensor.
        { 4096,             1,             4096 },
        { 4096,             4,             4096 * 4 },
    };

    int n_pass = 0;
    int n_fail = 0;
    for (const auto& sh : shapes) {
        bool a = run_one(sh);
        bool b = run_round_trip(sh);
        if (a && b) {
            std::fprintf(stderr,
                "[ OK ] ts=%-7u ns=%-2u total=%-12lu reference + round-trip pass\n",
                sh.tensor_size, sh.num_shards, (unsigned long)sh.total);
            ++n_pass;
        } else {
            ++n_fail;
        }
    }

    std::fprintf(stderr,
        "\n=== gpu_file_resolve_test: %d/%d shape(s) passed ===\n",
        n_pass, n_pass + n_fail);
    return n_fail == 0 ? 0 : 1;
}
