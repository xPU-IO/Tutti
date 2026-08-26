#!/usr/bin/env python
"""Profile two sequential requests through the real Tutti connector."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig


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


def _generate(
    llm: LLM,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    request_id: str,
) -> None:
    start = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        sampling_params,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - start
    completion_tokens = len(outputs[0].outputs[0].token_ids)
    print(
        f"[{request_id}] prompt_tokens={len(outputs[0].prompt_token_ids)} "
        f"completion_tokens={completion_tokens} wall={elapsed:.3f}s",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data2/qwen")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
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
    parser.add_argument("--tokens", type=int, default=65528)
    parser.add_argument("--reuse-pct", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--kv-load-failure-policy",
        choices=("recompute", "fail"),
        default="recompute",
        help="vLLM policy after the connector reports invalid block IDs",
    )
    parser.add_argument(
        "--kv-root",
        default="/mnt/nvme{LOCAL_RANK}/tutti-kv-profile-rank{LOCAL_RANK}",
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
        f"reuse={args.reuse_pct}% requests=2",
        flush=True,
    )

    kv_transfer_config = None
    if not args.without_tutti:
        kv_transfer_config = KVTransferConfig(
            kv_connector="TuttiConnectorV1",
            kv_connector_module_path="adapter.connector",
            kv_role="kv_both",
            kv_load_failure_policy=args.kv_load_failure_policy,
            kv_connector_extra_config={
                "chunk_tokens": 256,
                "max_chunks_per_wave": 512,
                "store": {
                    "type": "tutti_nvme",
                    "options": {
                        "root": args.kv_root,
                        "num_chunks": 512,
                        "io_stream": "auto",
                        "preset": {
                            "type": "local",
                            "daemon_config": (
                                "/data/home/ryeqiu/Tutti/"
                                "config/local/tutti_daemon.yaml"
                            ),
                            "device_id": "{LOCAL_RANK}",
                            "gpu_id": "{LOCAL_RANK}",
                        },
                    },
                },
            },
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=64,
        enforce_eager=True,
        max_model_len=args.tokens + args.max_tokens,
        load_format=args.load_format,
        enable_prefix_caching=True,
        enable_layerwise_nvtx_tracing=args.layerwise_nvtx,
        profiler_config={"profiler": "cuda"},
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

    # Start profiling in all TP workers once. Nsight is stopped externally.
    llm.start_profile()
    _generate(llm, request_a, sampling_params, "A-cold")
    if args.reset_local_prefix_between_requests:
        if not llm.reset_prefix_cache(reset_connector=False):
            raise RuntimeError("failed to reset vLLM local prefix cache")
        print("local prefix cache reset; Tutti cache retained", flush=True)
    _generate(llm, request_b, sampling_params, "B-80pct")
    print("profile workload complete", flush=True)
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
