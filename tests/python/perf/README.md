# perf/ — 性能基准

store/load 端到端带宽（对照分级：单盘 ~7 / NUMA 4 盘 ~28 / 全机 50+ GB/s）、
gather/scatter kernel 时延、submit 管线吞吐。后续功能在此补用例。

## Nsight Systems replay

The connector emits optional NVTX ranges.  Keep the same request batch alive
for several iterations so the trace shows cache reuse and layer overlap:

```bash
export TUTTI_NVTX=1
export TUTTI_NVTX_REPLAY_ROUNDS=4
nsys profile --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --output=run/tutti-replay \
  /data/home/ryeqiu/tutti-env/bin/python tests/e2e_walkthrough.py
```

Useful range names are:

- `tutti.request.load`, `tutti.request.wait_load`, and
  `tutti.request.prefetch_load` for the vLLM-facing layer schedule;
- `tutti.load.submit` / `tutti.store.submit` for staged transfer waves;
- `tutti.runtime.submit` and `tutti.runtime.wait` for the runtime boundary;
- `tutti.local_nvme.io_kernel` and `tutti.striped_nvme.io_kernel` for the
  device-side NVMe launch.

With `TUTTI_NVTX=0` (the default), these ranges are no-ops and do not add a
profiling dependency to normal tests or deployments.

## Real vLLM request driver

For the Hy3-FP8 TP4 deployment, start the server with the current connector
binding first, then drive two deterministic requests through the OpenAI API:

```bash
export PYTHONPATH=/data/home/ryeqiu/Tutti/integration/vllm-connector/bindings/python/src:$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=/data/home/ryeqiu/Phoenix/build:$LD_LIBRARY_PATH
export TUTTI_NVTX=1
/data/home/ryeqiu/tutti-env/bin/python scripts/vllm_profile_dummy.py \
  --port 8193 --model /data2/tencent/Hy3-FP8 \
  --tokens 65528 --reuse-pct 80 --requests 2
```

The token-id driver submits an exact 64K context (8 completion tokens fit
inside `max_model_len=65536`): request A is cold, request B shares 80% of A's
prefix.  Use `--tokens 131064` with a 128K server when the hybrid-model
worker path is ready for that context size.
