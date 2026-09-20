#!/usr/bin/env python
"""Profile two sequential requests through the real Tutti connector."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig


# Per-request NVTX colors chosen for max contrast in nsys timelines.
# A-COLD = warm orange (write/store association), B-80pct = cool cyan
# (read/load association), so the "reuse" boundary reads as a color flip.
_REQUEST_COLORS = {
    "A-cold": 0xFFFF8C00,  # ARGB warm orange
    "B-80pct": 0xFF00B8D9,  # ARGB cool cyan
}


def _make_request_annotate(request_id: str):
    """Return a context manager that opens a top-level NVTX range for one
    request, in domain "tutti.request" (this is what nsys --nvtx-capture
    matches against). Falls back to a no-op when NVTX is unavailable or
    the env flag is off. TUTTI_NVTX is also the flag Tutti's own call
    sites read, so toggling it here lights up both layers at once."""
    if os.environ.get("TUTTI_NVTX", "0").lower() not in {"1", "true", "yes", "on"}:
        return _NullAnnotate()
    try:
        import nvtx
    except Exception:
        return _NullAnnotate()
    color = _REQUEST_COLORS.get(request_id, 0xFF5B8FF9)
    # Use the connector's "tutti" domain (already populated by engine.nvtx.range
    # in 24+ call sites). Each request gets a top-level range with a clear
    # string prefix so it shows up as a separate band on the NVTX row.
    return nvtx.annotate(
        message=f"tutti.request|{request_id}",
        color=color,
        domain="tutti",
    )


class _NullAnnotate:
    """No-op stand-in so callers can always use ``with _make_request_annotate(...)``."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _tokens(model: str, length: int, reuse_pct: int) -> tuple[list[int], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    seed = tokenizer.encode(
        "Tutti layerwise KV overlap profile request. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer returned no seed tokens")

    request_a = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    shared = length * reuse_pct // 100
    suffix_length = length - shared
    reversed_seed = list(reversed(seed))
    suffix_b = (
        reversed_seed * ((suffix_length + len(seed) - 1) // len(seed))
    )[:suffix_length]
    return request_a, request_a[:shared] + suffix_b


def _distinct_pair(base_a: list[int], base_b: list[int], round_idx: int,
                   vocab_floor: int = 1000) -> tuple[list[int], list[int]]:
    """Derive a round-specific (A, B) pair with identical shapes.

    Rounds must not hit each other's KV entries, otherwise round N>0 would
    read round 0's cache and stop being a fresh cold/warm pair. Perturbing
    the first token is enough: the chunk key chain is prefix-derived, so a
    different first token invalidates every downstream chunk key while the
    tensor shapes (and therefore every GEMM/kernel shape) stay identical.
    """
    if round_idx == 0:
        return list(base_a), list(base_b)
    marker = vocab_floor + round_idx
    a = list(base_a)
    b = list(base_b)
    a[0] = marker
    b[0] = marker
    return a, b


def _start_bench_range() -> None:
    """推入一个固定名 NVTX 范围，标记整个基准循环的起点。

    存在的理由：nsys 要在只采集基准段的前提下给出可比报告。模型 299.9GB 的加载
    会产生约 468GB 的 H2D 流量，若被整段采集就会淹没报告——实测 488MB 报告、
    kernel 数是基准段报告的 60 倍，且与存储路径无关。

    为什么不复用车请求范围作触发条件：请求名带轮次与类型后缀（A-cold|r0 等），
    每个都不同；而本机 nsys 2025.3.2 **不支持 --nvtx-capture 通配符**，且必须
    指定域（省略 @domain 一律匹配失败）——已用最小程序逐一验证。

    只推入不弹出：配合 --capture-range-end=none，采集从本范围开始后持续到进程
    结束，正好覆盖基准段（其后只剩汇总输出，无 GPU 工作）。
    """
    if os.environ.get("TUTTI_NVTX", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        import nvtx
    except Exception:
        return
    nvtx.push_range(message="tutti.bench", domain="tutti")


def _generate(
    llm: LLM,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    request_id: str,
) -> float:
    # Per-request NVTX: each call gets a top-level "tutti.request" range so
    # the report shows a clear request boundary (start/end wall, sub-trees
    # for compute vs transfer). Domain "tutti.request" is what the nsys
    # --nvtx-capture filter matches on; other vLLM NVTX is filtered out.
    start = time.perf_counter()
    _annotate_request = _make_request_annotate(request_id)
    with _annotate_request:
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        token_ids = list(outputs[0].outputs[0].token_ids)
        finish_reason = outputs[0].outputs[0].finish_reason
        print(
            f"[{request_id}] prompt_tokens={len(outputs[0].prompt_token_ids)} "
            f"completion_tokens={completion_tokens} finish_reason={finish_reason} "
            f"token_ids={token_ids} wall={elapsed:.3f}s",
            flush=True,
        )
        if finish_reason == "error":
            raise RuntimeError(
                f"request {request_id} failed explicitly (finish_reason=error)"
            )
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data2/qwen")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument(
        "--block-size", type=int, choices=(64, 128, 256), default=64
    )
    parser.add_argument(
        "--direct-transfer-strict", action="store_true",
        help="已恒真（直连准入失败一律抛出）；保留以兼容既有脚本。",
    )
    parser.add_argument(
        "--without-tutti",
        action="store_true",
        help="run the model workload without a KV connector",
    )
    parser.add_argument(
        "--reset-local-prefix-between-requests",
        action="store_true",
        help="clear vLLM's local GPU prefix cache after request A",
    )
    parser.add_argument(
        "--layerwise-nvtx",
        action="store_true",
        help="enable verbose per-Module NVTX tracing",
    )
    parser.add_argument(
        "--disable-flashinfer-autotune",
        action="store_true",
        help="skip FlashInfer autotuning during vLLM startup",
    )
    parser.add_argument("--tokens", type=int, default=65528)
    parser.add_argument("--reuse-pct", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help=(
            "repeat the (A-cold, B-80pct) pair N times with identical tensor "
            "shapes but distinct KV keys; separates one-off first-shape host "
            "costs (cuBLAS algo selection, kernel image load) from steady state"
        ),
    )
    parser.add_argument(
        "--reset-local-prefix-between-rounds",
        action="store_true",
        help="clear vLLM's local prefix cache between rounds as well",
    )
    parser.add_argument(
        "--max-in-flight-operations",
        type=int,
        default=None,
        help=(
            "override the Tutti local-NVMe operation admission limit for "
            "this profile only"
        ),
    )
    parser.add_argument(
        "--kv-load-failure-policy",
        choices=("recompute", "fail"),
        default="recompute",
        help="vLLM policy after the connector reports invalid block IDs",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help=(
            "leave eager mode (enforce_eager=False) so vLLM's compilation "
            "path runs; this activates fuse_allreduce_rms (all-reduce + "
            "RMSNorm fused into one kernel, one cross-rank sync per layer "
            "instead of two) and CUDA graph capture. Default stays eager "
            "for clean, comparable NVTX timelines."
        ),
    )
    parser.add_argument(
        "--expect-b-failure", action="store_true",
        help="treat an explicit request-B failure as the expected test result",
    )
    parser.add_argument(
        "--kv-root",
        default="/mnt/nvme{LOCAL_RANK}/tutti-kv-profile-rank{LOCAL_RANK}",
    )
    parser.add_argument(
        "--kv-layout",
        choices=("file_per_chunk", "striped"),
        default="file_per_chunk",
        help=(
            "KV 落盘布局：file_per_chunk（默认，chunk 整块落单盘）或 "
            "striped（多盘条带，需配 --device-groups）"
        ),
    )
    parser.add_argument(
        "--stripe-unit",
        type=int,
        default=65536,
        help=(
            "striped 条带粒度（字节，默认 64KiB）。须不小于单个 IO entry "
            "（block_size × 每 token 每层每 rank 的 KV 字节），以免把提交切碎"
        ),
    )
    parser.add_argument(
        "--device-groups",
        default=None,
        help=(
            "striped 的 rank→盘组映射：分号分组、逗号列设备号，如 "
            "'0,1;2,3'（前一半 rank 用盘 0-1、后一半用盘 2-3）；"
            "组数必须整除 tensor_parallel_size"
        ),
    )
    parser.add_argument(
        "--num-queues",
        type=int,
        default=8,
        help=(
            "striped 每设备用户队列数（per-device）。每盘用户 QID 池约 "
            "119（QID 17..135，见内核 GET_DEV_INFO），须满足 "
            "num_queues × 每盘 rank 数 ≤ 119——8 卡每 4 rank 共用一组盘 "
            "时取 8（4×8=32），32（StripedNvmePreset 默认）会撑爆池"
        ),
    )
    parser.add_argument(
        "--wait-for-start-file",
        type=Path,
        help=(
            "initialize the LLM, then wait for this file before running the "
            "workload; create it after externally starting Nsight"
        ),
    )
    parser.add_argument(
        "--wait-for-exit-file",
        type=Path,
        help=(
            "keep the LLM and TP workers alive after the workload until this "
            "file exists; create it only after externally stopping Nsight"
        ),
    )
    parser.add_argument(
        "--torch-profiler-dir",
        default="",
        help=(
            "非空则改用 torch profiler 并把 trace 落到该目录（含 Python 调用栈）；"
            "留空保持 cuda profiler（给外部 Nsight 用）"
        ),
    )
    parser.add_argument(
        "--profile-window",
        choices=("all", "b-only"),
        default="all",
        help=(
            "torch profiler 采集窗口：all=整个工作负载；"
            "b-only=只采复用加载的 B 请求（A 的冷启动 prefill 不进 trace）"
        ),
    )
    parser.add_argument(
        "--torch-profiler-skip-table",
        action="store_true",
        help=(
            "跳过 stop_profile 后的 CUDA 耗时汇总表（key_averages 在长上下文"
            "trace 上要跑好几分钟，trace 本身早已落盘时可直接跳过）"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.reuse_pct < 100:
        parser.error("--reuse-pct must be in [1, 99]")
    for option, path in (
        ("--wait-for-start-file", args.wait_for_start_file),
        ("--wait-for-exit-file", args.wait_for_exit_file),
    ):
        if path and path.exists():
            parser.error(f"{option} already exists: {path}")

    request_a, request_b = _tokens(args.model, args.tokens, args.reuse_pct)
    shared = args.tokens * args.reuse_pct // 100
    print(
        f"profile workload: tokens={args.tokens} shared={shared} "
        f"reuse={args.reuse_pct}% rounds={args.rounds} "
        f"requests={2 * args.rounds}",
        flush=True,
    )

    kv_transfer_config = None
    if not args.without_tutti:
        # 对象池初始 slot 数必须覆盖单请求一整波 chunk（否则
        # PoolResourceExhausted：默认 initial_slots=32 < tokens/256）。
        chunk_tokens = 256
        per_request_chunks = -(-(args.tokens + args.max_tokens) // chunk_tokens)
        # 池容量（槽位上限）：测试规模 1w 槽位。槽位文件是稳定身份，
        # 复用不重跑 resolve（open+fstat+fsync+FIEMAP）。
        num_chunks = max(10000, per_request_chunks * 2 * args.rounds + 16)
        # 预建槽位数只覆盖单请求工作集：全量预建 1w × 20MiB ≈ 200GiB
        # 的实零写入会把 bind 变成分钟级；其余由启动期预热补齐。
        initial_slots = min(num_chunks, per_request_chunks + 8)
        # 水位必须覆盖整个 workload 的唯一 chunk 集合，否则后台分配器会在
        # 请求中途扩容：建槽位是磁盘实零写 + fsync（32k 实测 64 槽位
        # 2.85s），会直接阻塞前向线程（allocate 等就绪槽位）。
        # 唯一 chunk 上界 = 每轮(A 整请求 + B 新增部分) × 轮数
        #              ≈ per_request_chunks × (1 + (1 - reuse)) × rounds；
        # 再留一整轮 + 余量，保证分配完仍高于 low_watermark（否则触发补货）。
        high_watermark = min(
            num_chunks, per_request_chunks * (args.rounds + 2) + 16
        )
        low_watermark = max(1, per_request_chunks // 2)
        store_options = {
            "root": args.kv_root,
            "num_chunks": num_chunks,
            "initial_slots": initial_slots,
            "high_watermark": high_watermark,
            "low_watermark": low_watermark,
            "io_stream": "auto",
            "preset": {
                "daemon_config": (
                    "/data/home/ryeqiu/Tutti/"
                    "config/local/tutti_daemon.yaml"
                ),
                "gpu_id": "{LOCAL_RANK}",
            },
        }
        if args.kv_layout == "striped":
            if not args.device_groups:
                parser.error("--kv-layout striped 需要 --device-groups")
            groups = [
                [int(item) for item in group.split(",") if item.strip()]
                for group in args.device_groups.split(";")
                if group.strip()
            ]
            if not groups or any(not group for group in groups):
                parser.error("--device-groups 解析为空，示例：'0,1;2,3'")
            store_options["layout"] = "striped"
            store_options["stripe_unit"] = args.stripe_unit
            store_options["preset"].update({
                "type": "striped",
                "device_groups": groups,
                "stripe_unit": args.stripe_unit,
                "num_queues": args.num_queues,
            })
        else:
            store_options["preset"].update({
                "type": "local",
                "device_id": "{LOCAL_RANK}",
            })
        if args.max_in_flight_operations is not None:
            if args.max_in_flight_operations <= 0:
                parser.error("--max-in-flight-operations must be positive")
            store_options["preset"]["max_in_flight_operations"] = (
                args.max_in_flight_operations
            )
        connector_extra = {
            "chunk_tokens": 256,
            "max_chunks_per_wave": 512,
            "store": {
                "type": "tutti_nvme",
                "options": store_options,
            },
        }
        if args.direct_transfer_strict:
            # strict 语义已恒真（准入失败一律抛出）；保留 CLI 参数避免
            # 打断既有脚本，这里只再声明 direct 准入。
            connector_extra["direct_transfer"] = True
        kv_transfer_config = KVTransferConfig(
            kv_connector="TuttiConnectorV1",
            kv_connector_module_path="tutti.integration.vllm.connector",
            kv_role="kv_both",
            kv_load_failure_policy=args.kv_load_failure_policy,
            kv_connector_extra_config=connector_extra,
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        enforce_eager=not args.enable_compile,
        max_model_len=args.tokens + args.max_tokens,
        load_format=args.load_format,
        enable_prefix_caching=True,
        enable_layerwise_nvtx_tracing=args.layerwise_nvtx,
        enable_flashinfer_autotune=not args.disable_flashinfer_autotune,
        profiler_config=(
            {
                "profiler": "torch",
                "torch_profiler_dir": args.torch_profiler_dir,
                "torch_profiler_with_stack": True,
                "torch_profiler_dump_cuda_time_total": (
                    not args.torch_profiler_skip_table
                ),
            }
            if args.torch_profiler_dir
            else {"profiler": "cuda"}
        ),
        kv_transfer_config=kv_transfer_config,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    if args.wait_for_start_file:
        print(
            f"profile ready: waiting for {args.wait_for_start_file}",
            flush=True,
        )
        while not args.wait_for_start_file.exists():
            time.sleep(0.2)
        print("profile workload released", flush=True)

    # cuda profiler 全程开启，由外部 Nsight 停止；torch profiler 由本脚本 stop
    # 以落盘 trace，b-only 窗口时推迟到 B 请求前再开。
    torch_profiling = bool(args.torch_profiler_dir)
    b_only_window = torch_profiling and args.profile_window == "b-only"
    if not b_only_window:
        llm.start_profile()
    walls_a: list[float] = []
    walls_b: list[float] = []
    _start_bench_range()
    for round_idx in range(args.rounds):
        tokens_a, tokens_b = _distinct_pair(request_a, request_b, round_idx)
        suffix = "" if args.rounds == 1 else f"|r{round_idx}"
        walls_a.append(
            _generate(llm, tokens_a, sampling_params, f"A-cold{suffix}")
        )
        if args.reset_local_prefix_between_requests:
            if not llm.reset_prefix_cache(reset_connector=False):
                raise RuntimeError("failed to reset vLLM local prefix cache")
            print("local prefix cache reset; Tutti cache retained", flush=True)
        if b_only_window:
            llm.start_profile()
        try:
            walls_b.append(
                _generate(llm, tokens_b, sampling_params, f"B-80pct{suffix}")
            )
        except Exception as exc:
            if not args.expect_b_failure:
                raise
            print(
                f"[B-80pct{suffix}] expected_request_failure="
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        else:
            if args.expect_b_failure:
                raise RuntimeError(
                    "request B unexpectedly succeeded under fail policy"
                )
        finally:
            if b_only_window:
                # 即使请求失败也要落盘：失败现场正是要看的 trace。
                llm.stop_profile()
        if args.reset_local_prefix_between_rounds and round_idx + 1 < args.rounds:
            # Keep每轮 A 处于"vLLM 本地无前缀命中"状态，否则轮间本地
            # prefix cache 会让后续 A 不再是 cold（外部命中亦不触发）。
            if not llm.reset_prefix_cache(reset_connector=False):
                raise RuntimeError("failed to reset vLLM local prefix cache")

    if args.rounds > 1:
        def _stats(name: str, walls: list[float]) -> None:
            if not walls:
                return
            first = walls[0]
            rest = walls[1:]
            body = (
                f" steady_mean={sum(rest)/len(rest):.3f}s "
                f"steady_min={min(rest):.3f}s steady_max={max(rest):.3f}s"
                if rest else ""
            )
            print(
                f"[SUMMARY {name}] rounds={len(walls)} first={first:.3f}s"
                f"{body} all={[round(w, 3) for w in walls]}",
                flush=True,
            )
        _stats("A-cold", walls_a)
        _stats("B-80pct", walls_b)
    print("profile workload complete", flush=True)
    if torch_profiling and not b_only_window:
        llm.stop_profile()
    if args.wait_for_exit_file:
        print(
            f"profile hold: waiting for {args.wait_for_exit_file}",
            flush=True,
        )
        while not args.wait_for_exit_file.exists():
            time.sleep(0.2)
        print("profile hold released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
