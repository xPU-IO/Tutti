#!/usr/bin/env python
"""vLLM 端到端冒烟客户端：真实 trace 请求驱动（同 trial 连续 turn）。

用法（服务须已在 8192 端口就绪）：
    python scripts/vllm_smoke_hy3.py [--port 8192] [--trial 0] [--turns 2]

从 trace 文件取指定 trial 的前 N 个 turn（同 trial 前缀递增，天然
覆盖跨请求前缀复用）；turn 0 请求两次（第二次应全量外部命中），
后续 turn 各请求一次（前缀部分命中）。记录每条 TTFT 与输出；
两次 turn 0 响应逐字比对。退出码 0 = 全部成功且 turn 0 两轮一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

TRACE = "/data2/traces/codex_swebenchpro/requests_hy3.jsonl"


def _load_turns(trial_id: int, turns: int) -> list[dict]:
    """取指定 trial 的前 N 个 turn（按 turn 序号升序）。"""
    rows = []
    with open(TRACE) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("trial_id") == trial_id:
                rows.append(row)
    rows.sort(key=lambda r: r.get("turn", 0))
    if not rows:
        raise SystemExit(f"trace 中无 trial {trial_id}")
    return rows[:turns]


def _request(port: int, prompt: str, max_tokens: int) -> tuple[float, str]:
    """发一次补全请求，返回 (首字时延（秒）, 输出文本)。"""
    payload = json.dumps({
        "model": "smoke",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    chunks: list[str] = []
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            body = line[len(b"data: "):]
            if body == b"[DONE]":
                break
            event = json.loads(body)
            for choice in event.get("choices", []):
                text = choice.get("text", "")
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    chunks.append(text)
    if ttft is None:
        ttft = time.perf_counter() - start
    return ttft, "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    turns = _load_turns(args.trial, args.turns)
    print(f"[smoke{args.tag}] trial={args.trial} turns="
          f"{[(t['turn'], t['prompt_tokens']) for t in turns]}")

    outputs = []
    first_text = None
    for idx, turn in enumerate(turns):
        repeat = 2 if idx == 0 else 1  # turn 0 请求两次：冷/热对比
        for rep in range(repeat):
            ttft, text = _request(
                args.port, turn["prompt"], int(turn.get("max_tokens", 64))
            )
            label = f"turn{turn['turn']}" + ("" if rep == 0 else "-repeat")
            print(f"[{label}] prompt_tokens={turn['prompt_tokens']} "
                  f"TTFT={ttft:.3f}s out_chars={len(text)}")
            print(f"[{label}] output[:100]={text[:100]!r}")
            outputs.append((label, text))
            if idx == 0 and rep == 0:
                first_text = text
            elif idx == 0:
                if text != first_text:
                    print(f"FAIL: turn0 两轮输出不一致\n"
                          f"  cold={first_text[:200]!r}\n  warm={text[:200]!r}",
                          file=sys.stderr)
                    return 1
    print(f"PASS: {len(outputs)} 次请求全部完成，turn0 两轮输出逐字一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
