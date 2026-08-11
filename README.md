# Tutti

**A GPU-centric, SSD-backed KV cache object store for long-context LLM serving.**

**Current release: v0.1.1** — stable `StorageRuntime` public API, striped multi-device data path, and a vendor-neutral GPU framework.

Tutti (Italian for "all instruments together") makes NVMe SSDs a practical KV-cache tier for LLM inference, building on the ideas of [GeminiFS](https://www.usenix.org/conference/fast25/presentation/qiu) (FAST'25), a companion file system for GPUs. **The CPU launches I/O kernels; the GPU executes them.** Each GPU kernel moves KV cache between SSDs and HBM on its own — giving SSD-backed KV cache DRAM-like performance with hundreds of times the capacity.

## 📰 News

- **2026.8** — **v0.1.1**: stable `StorageRuntime` public API (open/open_batch/register/submit/wait) over replaceable resolver/binding/DataPath boundaries; striped multi-device data path reaches 25.0 GB/s on 4× NVMe (98% of near-saturated per-drive bandwidth); vendor-neutral `cuda_like` GPU framework (CUDA proven; MUSA/MACA profiles); kernel P2P layer split into symmetric nvidia/metax backends; batch open (500 files in ~10 ms).

- **2026.7** — `snvme` kernel module update: dynamic CPU-side and GPU-side queue allocation after mount, a standard POSIX interface, and poll-based I/O submission from both user-space threads and GPU kernel threads.

- **2026.6** — Tutti adapted to the MetaX C600 GPU.

- **2026.5** — Tutti paper released on [arXiv](https://arxiv.org/abs/2605.03375) (SSD-backed KV cache: −78.3% TTFT, 2× request rate, −27% serving cost vs. GDS-enabled LMCache).

## ✨ Product Highlights

- **GPU io_uring** — the batch-entry array and IO handle are shared CPU↔GPU submission state: one `cudaMemcpyAsync` hands a whole batch across, one kernel launch is the only boundary crossing, and completion rides the CUDA stream so submit and harvest stay decoupled. Each GPU thread resolves LBAs, writes SQEs, rings doorbells, and polls completions on its own. No CPU on the data path.
- **Register once, then just look up** — PRP descriptors are pre-built at registration; one DMA mapping serves every NVMe controller. The I/O hot path is table lookup, not arithmetic.
- **Millions of files within GPU reach** — ~200-byte GPU-resident file handles plus two-tier (GPU L1 / pinned-host L2) caching; FIEMAP is walked exactly once per file.
- **Multi-device striping** — a striped target spans up to 4 NVMe SSDs in tensor-sized units (one K/V tensor lands whole on one drive, round-robin by tensor index); a single fused kernel submits to all drives. 25.0 GB/s on 4 drives.
- **Batch open** — `open_batch` resolves hundreds of per-layer files with a parallel FIEMAP pipeline and fail-closed per-item results: 500 files in ~10 ms vs ~50–115 ms serial (5–12×).
- **Beyond NVIDIA** — three-layer `cuda_like` framework (profile selector → vendor shim → kernel primitive macros): CUDA proven in production, MUSA/MACA build profiles in place; the kernel P2P layer is split into symmetric per-vendor backends (nvidia done, metax symmetric).

## What is Tutti

Modern LLM engines (vLLM, SGLang) page the KV cache into small, scattered GPU memory blocks. Once that cache is offloaded to SSDs, memory fragmentation becomes **I/O fragmentation**: restoring a long prefix means issuing hundreds of thousands of tiny random I/Os — and in a CPU-centric design, every single one is initiated by the CPU. The result is massive GPU stalls: reusing KV cache from SSD ends up slower than recomputing it, and the penalty only grows as GPU compute keeps getting faster.

The conductor doesn't play the instruments: the CPU only cues the orchestra (launches I/O kernels); the GPU musicians read their own descriptors and drive the NVMe drives directly:

<p align="center">
  <img src="doc/tutti-arch.png" alt="Tutti architecture: the CPU conducts — launching I/O kernels only — while the GPU drives NVMe SSDs directly for AI applications" width="900">
</p>

Tutti's answer is a GPU-centric data path where the CPU appears only **O(1) times per batch, not per I/O** — detailed in [doc/architecture/system-architecture.md](doc/architecture/system-architecture.md).

## System Architecture (v0.1.1)

```text
Application  (uri / offset / size)
    │  open / register / submit / wait      ← CPU appears O(1) per batch
    ▼
StorageRuntime  — stable public API
    │  resolver + binding → StorageTarget
    ▼
DataPaths  — local_nvme / striped_local_nvme
    │  PRP prebuilt at register; GPU-resident handle cache
    ▼
libnvm + tutti_daemon  →  snvme kernel module
    │  P2P DMA: GPU threads drive NVMe directly
    ▼
NVMe SSDs  ◄──────►  GPU HBM
```

Cross-cutting: the `cuda_like` three-layer GPU framework (profile selector →
vendor shim → kernel primitive macros) covers every layer above the kernel —
CUDA proven, MUSA/MACA profiles in place; the kernel P2P layer is split into
symmetric per-vendor backends (nvidia done, metax symmetric).

**Environment (tested)**

- OS: Linux, kernel 6.8.x (`snvme-6.8.0-public`) or 5.15.x (`snvme-5.15.0-public`); a 5.4.241 (tlinux4) module lineage is also maintained
- Accelerator: NVIDIA GPU + CUDA toolkit (`nvcc`); bare metal with IOMMU in passthrough mode. MUSA/MACA build profiles configure-checked only (no hardware validation yet)
- Runtime: daemon-only — `tutti_daemon` bring-up creates `/dev/snvme*`; queue depth is fixed at module install (`io_queue_depth=1024` for production)
- Host deps: CMake, protobuf / gRPC / uuid / yaml-cpp — one-shot setup via `scripts/prepare_env.sh`
- All file I/O is opened with `O_DIRECT` (project policy)

## Running the KV Cache Example

```bash
cmake --preset cuda --fresh -DTUTTI_BUILD_HARDWARE_TESTS=ON
cmake --build --preset cuda --target tutti_layerwise_kv_overlap --parallel 8
sudo ./build/cuda/bin/tutti_layerwise_kv_overlap     # 4-disk striped (default); --single for one drive
```

`layerwise_kv_overlap` is Tutti's standard KV-cache reference workload
(80 layers, 512 KiB K/V tensors, read∥compute∥write overlap). Prerequisites
in **strict order**: load the `snvme` kernel modules → start `tutti_daemon`
→ mount — the block devices only exist after daemon bring-up. Full setup,
parameters, expected output (~25 GB/s READ on 4 drives) and clean-up:
[examples/layerwise_kv_overlap/README.md](examples/layerwise_kv_overlap/README.md).
Also runnable as a gated test: `ctest -R tutti_layerwise_kv_overlap`.

## Configuration

Two files, two roles — one topology source of truth:

- **`config/local_nvme_config.yaml`** — the local-NVMe deployment fact file: NVMe controllers (`nvmes[]` with `allowed_gpus` ACL), GPU enumeration, mount points, queue/lease policy. Read by **both** the daemon (bring-up/mount) and the application loader (the CUDA↔ssd device map is derived from `nvmes[]` order + `allowed_gpus`). Daemon lookup: `--config` > `config/local_nvme_config.yaml` > legacy `sys_config.yaml` names (deprecation warnings).
- **`config/tutti_config.yaml`** — the application's entry point: DataPath cache knobs (`handle_cache_capacity`, `prp_cache_capacity`, L2), capacity knobs (`max_in_flight_operations`, `max_batch_entries`, `num_user_queues`), `io_granularity`, and a `local_nvme_config` link key pointing at the file above. Priority: programmatic injection > config file > built-in defaults (`TUTTI_*_CACHE_CAP` env vars are test-only backdoors). Queue depth is *not* here — it is kernel-authoritative (`io_queue_depth` at module install).

The repository root is the only supported CMake entry. All component
`CMakeLists.txt` files are reached from that target graph and are not
standalone project entries. Runtime is daemon-only.

## Deep Dive

- [System Architecture](doc/architecture/system-architecture.md) — the as-implemented layers, IO walkthrough, and deployment topology
- [Key Designs](doc/architecture/key-designs.md) — the five performance designs behind the GPU-centric data path, with measured numbers
- [Backend SPI](doc/design/backend-spi.md) — the DataPath / Resolver / Binding semantic contracts
- [GPU Porting Guide](doc/gpu-porting-guide.md) — the `cuda_like` three-layer framework, primitive semantic contracts, and Metax integration steps
- [Kernel Portability](doc/design/kernel-portability.md) — the snvme module across kernel versions and GPU vendors
- [Extending Tutti](doc/extending_tutti.md) — adding resolvers, bindings, and data paths behind the SPI
- [Build & SNVMe Testing](doc/build_and_test.md) — environment setup, build, module install, and the smoke-test ladder
- [Contributing](CONTRIBUTING.md) — install, test, and contribution rules

## Cite

If you use Tutti in your research, please cite our paper — and the GeminiFS work it builds on:

```bibtex
@article{tutti,
  title   = {Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving},
  author  = {Qiu, Shi and Hu, Yifan and Wang, Xintao and Zhu, Wenhao and Yan, Jianqin and Chen, Hao and Xu, Kaiqiang and Chen, Kai and Zhang, Yiming},
  journal = {arXiv preprint arXiv:2605.03375},
  year    = {2026}
}

@inproceedings{geminifs,
  author    = {Shi Qiu and Weinan Liu and Yifan Hu and Jianqin Yan and Zhirong Shen and Xin Yao and Renhai Chen and Gong Zhang and Yiming Zhang},
  title     = {{GeminiFS}: A Companion File System for {GPUs}},
  booktitle = {23rd USENIX Conference on File and Storage Technologies (FAST 25)},
  year      = {2025},
  isbn      = {978-1-939133-45-8},
  address   = {Santa Clara, CA},
  pages     = {221--236},
  publisher = {USENIX Association},
  month     = feb
}
```

## Acknowledgements

Tutti's pluggable GPU-vendor framework (`cuda_like`) draws on the design patterns
pioneered by [Mooncake](https://github.com/kvcache-ai/Mooncake) (Apache 2.0),
which similarly decouples the GPU runtime API surface from the kernel that
consumes it. See [`third_pkgs/Mooncake/LICENSE-APACHE`](third_pkgs/Mooncake/LICENSE-APACHE)
for the full license text.

## Contact & community

*Contact information to be added (maintainers / mailing list / Slack or WeChat group).* For bug reports, please use the issue template — see [CONTRIBUTING.md](CONTRIBUTING.md).

Questions, adaptation proposals, or interest in becoming a community maintainer / manager — reach out to Shi Qiu at ryeqiu@tencent.com.

## License

Apache-2.0 — see [LICENSE](LICENSE).
