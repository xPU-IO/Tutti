#!/usr/bin/env python
"""Profile two sequential requests through the real Tutti connector."""

from __future__ import annotations

import argparse
import importlib
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


def _tokens(model: str, length: int, reuse_pct: int,
            shared_tokens: int | None = None) -> tuple[list[int], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    seed = tokenizer.encode(
        "Tutti layerwise KV overlap profile request. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer returned no seed tokens")

    request_a = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    shared = length * reuse_pct // 100 if shared_tokens is None else shared_tokens
    if not 0 < shared < length:
        raise ValueError("shared tokens must be between zero and prompt length")
    suffix_length = length - shared
    reversed_seed = list(reversed(seed))
    suffix_b = (
        reversed_seed * ((suffix_length + len(seed) - 1) // len(seed))
    )[:suffix_length]
    if shared_tokens is not None and suffix_b[0] == request_a[shared]:
        suffix_b[0] = next(token for token in seed if token != request_a[shared])
    return request_a, request_a[:shared] + suffix_b


def _concurrent_batch(
    base: list[int],
    batch_idx: int,
    batch_size: int,
    vocab_floor: int = 1000,
) -> list[list[int]]:
    """Derive ``batch_size`` shape-identical but KV-distinct prompts.

    存在的理由：线上峰值读计划的 ``chunks`` 是「同一批所有请求的 chunk 数之和」
    （实测 3110 = 6 x 512，而单请求上限受 ``max_model_len`` 与
    ``max_chunks_per_wave`` 双重约束）。单请求串行永远复现不出该量级，必须在
    一个 ``llm.generate`` 批次里同时提交多个请求，让调度器把它们合并进同一个
    读计划。

    与 ``_distinct_pair`` 同一手法（首 token 扰动）：chunk key 由前缀链派生，
    改首 token 即令全部下游 chunk key 失效，从而保证各请求各自独立读盘；而
    token 序列长度不变，所以每个 GEMM/kernel 的 shape 完全一致，性能可比。
    """
    prompts: list[list[int]] = []
    for offset in range(batch_size):
        marker = vocab_floor + batch_idx * 16 + offset
        prompt = list(base)
        prompt[0] = marker
        prompts.append(prompt)
    return prompts


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
    prompt_token_ids: list[int] | list[list[int]],
    sampling_params: SamplingParams,
    request_id: str,
) -> float:
    # Per-request NVTX: each call gets a top-level "tutti.request" range so
    # the report shows a clear request boundary (start/end wall, sub-trees
    # for compute vs transfer). Domain "tutti.request" is what the nsys
    # --nvtx-capture filter matches on; other vLLM NVTX is filtered out.
    #
    # prompt_token_ids 可以是单条，也可以是一批（同一次 llm.generate 提交多个
    # 请求）。批量形态用于复现线上「一批请求合并进同一读计划」的 chunks 量级：
    # 调度器把同批请求合并后，读计划的 chunks 是各请求之和。
    if prompt_token_ids and isinstance(prompt_token_ids[0], int):
        prompts = [prompt_token_ids]
    else:
        prompts = list(prompt_token_ids)  # type: ignore[arg-type]
    start = time.perf_counter()
    _annotate_request = _make_request_annotate(request_id)
    with _annotate_request:
        outputs = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        total_prompt_tokens = sum(
            len(output.prompt_token_ids) for output in outputs
        )
        total_completion_tokens = sum(
            len(output.outputs[0].token_ids) for output in outputs
        )
        finish_reasons = {
            output.outputs[0].finish_reason for output in outputs
        }
        print(
            f"[{request_id}] requests={len(prompts)} "
            f"prompt_tokens={total_prompt_tokens} "
            f"completion_tokens={total_completion_tokens} "
            f"cached_tokens={[output.num_cached_tokens for output in outputs]} "
            f"finish_reason={sorted(finish_reasons)} wall={elapsed:.3f}s "
            f"token_ids={list(outputs[0].outputs[0].token_ids)}",
            flush=True,
        )
        if "error" in finish_reasons:
            raise RuntimeError(
                f"request {request_id} failed explicitly (finish_reason=error)"
            )
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--capacity", help="每rank KV payload预算，须带单位，例如 1 TiB")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--enable-ep-weight-filter", action="store_true")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument(
        "--indexer-kv-dtype", choices=("auto", "bf16", "fp8", "mxfp4", "nvfp4")
    )
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument(
        "--replica-io", action="store_true",
        help="enable cooperative replica reads in the Tutti connector",
    )
    parser.add_argument(
        "--block-size", type=int, choices=(64, 128, 256, 512), default=64
    )
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--shared-tokens", type=int,
                        help="exact common prefix for matched HBM/NVMe compute shapes")
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
    parser.add_argument("--kv-load-mode", choices=("sync", "async", "auto"), default="async",
                        help="Tutti request loading policy; sync forces external-hit validation")
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
        "--batch-size",
        type=int,
        default=1,
        help=(
            "number of shape-identical requests submitted in ONE llm.generate "
            "call. The scheduler merges a batch into a single read plan, so the "
            "plan's chunk count is the sum over requests: this is how the "
            "production peak (chunks=3110 = 6 x 512) is reproduced. Each "
            "request gets a perturbed first token so their KV stays distinct "
            "and every one of them actually reads from the Tutti path."
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
        "--enable-compile", "--enable-compiler",
        action="store_true",
        help=(
            "enable Inductor compilation; CUDA graph capture is selected with "
            "--cudagraph-mode (default NONE). Default stays eager."
        ),
    )
    parser.add_argument(
        "--cudagraph-mode", choices=("NONE", "PIECEWISE"), default="NONE",
        help="CUDA graph mode with --enable-compile; PIECEWISE keeps KV callbacks outside captured segments",
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
        "--daemon-config",
        default=",".join(str(Path(__file__).resolve().parents[2] / "config" / "local" / name)
                         for name in ("tutti_daemon_kv0.yaml", "tutti_daemon_md0.yaml")),
        help="运行中各 tutti_daemon 的配置，逗号分隔（合并为一份设备清单）",
    )
    parser.add_argument(
        "--device-groups",
        default=None,
        help=(
            "多盘 rank→盘组映射（不给则每 rank 单盘）：分号分组、逗号列设备号，如 "
            "'0,1;2,3'（前一半 rank 用盘 0-1、后一半用盘 2-3）；"
            "组数必须整除 tensor_parallel_size"
        ),
    )
    parser.add_argument(
        "--num-queues",
        type=int,
        default=8,
        help=(
            "每rank每设备用户队列数；rank数×队列数须在实际控制器可用QID预算内。"
            "默认8只是起点，不修改内核queue depth；参见配置与性能指南"
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
    if args.cudagraph_mode != "NONE" and not args.enable_compile:
        parser.error("--cudagraph-mode PIECEWISE requires --enable-compile")
    if not 1 <= args.reuse_pct < 100:
        parser.error("--reuse-pct must be in [1, 99]")
    if args.shared_tokens is not None and not 0 < args.shared_tokens < args.tokens:
        parser.error("--shared-tokens must be between zero and --tokens")
    if args.chunk_tokens <= 0 or args.chunk_tokens % args.block_size:
        parser.error("--chunk-tokens must be positive and divisible by --block-size")
    if args.max_model_len is not None and args.max_model_len < args.tokens + args.max_tokens:
        parser.error("--max-model-len must cover --tokens + --max-tokens")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.load_format == "phxsafetensors":
        try:
            importlib.import_module("phxloader")
        except (ImportError, OSError) as exc:
            parser.error(
                f"--load-format phxsafetensors requires importable phxloader; "
                f"configure PYTHONPATH for the selected Python environment: {exc}"
            )
    for option, path in (
        ("--wait-for-start-file", args.wait_for_start_file),
        ("--wait-for-exit-file", args.wait_for_exit_file),
    ):
        if path and path.exists():
            parser.error(f"{option} already exists: {path}")

    if not args.without_tutti:
        from tutti.common.utils import parse_capacity
        try:
            parse_capacity(args.capacity)
        except ValueError as exc:
            parser.error(str(exc))

    request_a, request_b = _tokens(args.model, args.tokens, args.reuse_pct, args.shared_tokens)
    shared = args.tokens * args.reuse_pct // 100 if args.shared_tokens is None else args.shared_tokens
    print(
        f"profile workload: tokens={args.tokens} shared={shared} "
        f"reuse={args.reuse_pct}% rounds={args.rounds} "
        f"requests={2 * args.rounds}",
        flush=True,
    )

    kv_transfer_config = None
    if not args.without_tutti:
        store_options = {
            "root": args.kv_root,
            "capacity": args.capacity,
            "preset": {
                "daemon_config": args.daemon_config,
                "gpu_id": "{LOCAL_RANK}",
            },
        }
        payload_tag = os.environ.get("TUTTI_E2E_PAYLOAD_TAG")
        if payload_tag:
            store_options["payload_tag"] = payload_tag
        if args.device_groups:
            # 多盘：每 rank 的槽位在其盘组间轮转（数据盘目录来自 daemon 配置）。
            groups = [
                [int(item) for item in group.split(",") if item.strip()]
                for group in args.device_groups.split(";")
                if group.strip()
            ]
            if not groups or any(not group for group in groups):
                parser.error("--device-groups 解析为空，示例：'0,1;2,3'")
            store_options["preset"].update({
                "type": "raid0",
                "device_groups": groups,
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
            "chunk_tokens": args.chunk_tokens,
            "store": {
                "type": "tutti_nvme",
                "options": store_options,
            },
        }
        connector_extra["direct_transfer"] = True
        connector_extra["load_mode"] = args.kv_load_mode
        if args.replica_io:
            connector_extra["replica_io"] = True
        kv_transfer_config = KVTransferConfig(
            kv_connector="TuttiConnectorV1",
            kv_connector_module_path="tutti.integration.vllm.connector",
            kv_role="kv_both",
            kv_load_failure_policy=args.kv_load_failure_policy,
            kv_connector_extra_config=connector_extra,
        )
    model_options = {}
    if args.enable_expert_parallel:
        model_options["enable_expert_parallel"] = True
    if args.enable_ep_weight_filter:
        model_options["enable_ep_weight_filter"] = True
    if args.kv_cache_dtype is not None:
        model_options["kv_cache_dtype"] = args.kv_cache_dtype
    if args.indexer_kv_dtype is not None:
        model_options["attention_config"] = {"indexer_kv_dtype": args.indexer_kv_dtype}
    if args.gpu_memory_utilization is not None:
        model_options["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_batched_tokens is not None:
        model_options["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.language_model_only:
        model_options["language_model_only"] = True
    if args.enable_compile:
        os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
        model_options["compilation_config"] = {"mode": 3, "cudagraph_mode": args.cudagraph_mode}
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        block_size=args.block_size,
        enforce_eager=not args.enable_compile,
        max_model_len=(
            args.max_model_len if args.max_model_len is not None
            else args.tokens + args.max_tokens
        ),
        # 必须 >= batch_size，否则批量请求被拆成多个调度步，读计划不会合并，
        # 复现不出线上「一批请求共享同一读计划」的 chunks 量级。
        max_num_seqs=(
            args.max_num_seqs if args.max_num_seqs is not None
            else max(256, args.batch_size)
        ),
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
        **model_options,
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
    request_b_label = f"B-{args.reuse_pct}pct"
    _start_bench_range()
    for round_idx in range(args.rounds):
        tokens_a, tokens_b = _distinct_pair(request_a, request_b, round_idx)
        suffix = "" if args.rounds == 1 else f"|r{round_idx}"
        # batch_size > 1：把 N 个 shape 相同、KV 互不相同的请求放进同一次
        # generate 提交，让调度器合并成单个读计划（复现线上 chunks 量级）。
        # A/B 必须用同一组 marker（同一 round_idx）：marker 决定首 token，
        # 从而决定整条 chunk-key 链；A 与 B 用不同 marker 会让 B 完全 miss、
        # 读计划根本不会产生。轮与轮之间换 marker，保证轮间不在 HBM 里互蹭。
        prompts_a = (
            _concurrent_batch(tokens_a, round_idx, args.batch_size)
            if args.batch_size > 1 else tokens_a
        )
        prompts_b = (
            _concurrent_batch(tokens_b, round_idx, args.batch_size)
            if args.batch_size > 1 else tokens_b
        )
        walls_a.append(
            _generate(llm, prompts_a, sampling_params, f"A-cold{suffix}")
        )
        if args.reset_local_prefix_between_requests:
            if not llm.reset_prefix_cache(reset_connector=False):
                raise RuntimeError("failed to reset vLLM local prefix cache")
            print("local prefix cache reset; Tutti cache retained", flush=True)
        if b_only_window:
            llm.start_profile()
        try:
            walls_b.append(
                _generate(llm, prompts_b, sampling_params, f"{request_b_label}{suffix}")
            )
        except Exception as exc:
            if not args.expect_b_failure:
                raise
            print(
                f"[{request_b_label}{suffix}] expected_request_failure="
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
        _stats(request_b_label, walls_b)
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
