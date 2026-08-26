#!/usr/bin/env python
"""Drive a deterministic prefix-reuse workload through a vLLM server.

The request body uses token ids instead of a giant text prompt, so the
requested context length is exact.  Request A is cold, B shares 80% of A's
prefix, and C repeats B.  The server must already be running.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def _nvtx(name: str):
    try:
        import torch
        if os.environ.get("TUTTI_NVTX", "0").lower() in {"1", "true", "yes", "on"}:
            torch.cuda.nvtx.range_push(name)
            return True
    except Exception:
        pass
    return False


def _nvtx_pop(active: bool) -> None:
    if not active:
        return
    try:
        import torch
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _request(port: int, prompt: list[int], request_id: str, max_tokens: int) -> None:
    body = json.dumps({
        "model": "smoke",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    active = _nvtx(f"tutti.profile.request|id={request_id}|tokens={len(prompt)}")
    try:
        try:
            with urllib.request.urlopen(req, timeout=7200) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1000]
            raise RuntimeError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
    finally:
        _nvtx_pop(active)
    elapsed = time.perf_counter() - start
    usage = payload.get("usage", {})
    print(
        f"[{request_id}] prompt_tokens={usage.get('prompt_tokens', len(prompt))} "
        f"completion_tokens={usage.get('completion_tokens')} wall={elapsed:.3f}s",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data2/tencent/Hy3-FP8")
    parser.add_argument("--port", type=int, default=8192)
    # Leave room for the requested completion inside max_model_len=131072.
    parser.add_argument("--tokens", type=int, default=65528)
    parser.add_argument("--reuse-pct", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.reuse_pct < 100:
        parser.error("--reuse-pct must be in [1, 99]")
    if not 1 <= args.requests <= 3:
        parser.error("--requests must be in [1, 3]")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    seed = tokenizer.encode(
        "Tutti layerwise KV overlap profile request. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer returned no seed tokens")
    base = (seed * ((args.tokens + len(seed) - 1) // len(seed)))[:args.tokens]
    shared = args.tokens * args.reuse_pct // 100
    suffix_len = args.tokens - shared
    suffix_b = ((list(reversed(seed)) * ((suffix_len + len(seed) - 1) // len(seed)))
                [:suffix_len])
    suffix_c = (((seed[1:] + seed[:1]) * ((suffix_len + len(seed) - 1) // len(seed)))
                [:suffix_len])
    request_b = base[:shared] + suffix_b
    request_c = base[:shared] + suffix_c
    print(
        f"profile workload: tokens={args.tokens} shared={shared} "
        f"reuse={args.reuse_pct}% requests={args.requests}",
        flush=True,
    )
    requests = [(base, "A-cold"), (request_b, "B-80pct"), (request_c, "C-80pct")]
    for prompt, request_id in requests[:args.requests]:
        _request(args.port, prompt, request_id, args.max_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
