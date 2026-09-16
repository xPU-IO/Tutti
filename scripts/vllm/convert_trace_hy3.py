#!/usr/bin/env python
"""Convert Inferact/codex_swebenchpro_traces into requests_hy3.jsonl.

用法:
    python scripts/convert_trace_hy3.py \
        --trace /mnt/nvme4/datasets/codex_swebenchpro/codex_swebenchpro.json \
        --model /mnt/nvme4/models/Hy3-FP8 \
        --out /mnt/nvme4/datasets/codex_swebenchpro/requests_hy3.jsonl

数据集是 610 个 trial 的 conversations（human/gpt 交替，gpt 回复为
长度保留的匿名文本）。每个人类轮次生成一条请求：
- prompt     = chat template 渲染的累积上下文（天然跨请求前缀复用）
- prompt_tokens = Hy3 tokenizer 对渲染后 prompt 的精确计数
- max_tokens = 实际（匿名）回复的 token 数，封顶 --max-tokens-cap
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

MODEL = "/mnt/nvme4/models/Hy3-FP8"
TRACE = "/mnt/nvme4/datasets/codex_swebenchpro/codex_swebenchpro.json"
OUT = "/mnt/nvme4/datasets/codex_swebenchpro/requests_hy3.jsonl"

_tok = None


def _init_worker(model: str) -> None:
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)


def _convert_trial(item: tuple[int, dict]) -> tuple[int, list[dict]]:
    trial_id, conv = item
    msgs: list[dict] = []
    rows: list[dict] = []
    turn = 0
    conversations = conv["conversations"]
    for i, m in enumerate(conversations):
        role = "user" if m["from"] == "human" else "assistant"
        msgs.append({"role": role, "content": m["value"]})
        if role != "user":
            continue
        reply = ""
        if (
            i + 1 < len(conversations)
            and conversations[i + 1]["from"] == "gpt"
        ):
            reply = conversations[i + 1]["value"]
        prompt = _tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        reply_tokens = (
            len(_tok.encode(reply, add_special_tokens=False))
            if reply
            else 0
        )
        rows.append(
            {
                "trial_id": trial_id,
                "turn": turn,
                "prompt": prompt,
                "prompt_tokens": len(
                    _tok.encode(prompt, add_special_tokens=False)
                ),
                "max_tokens": min(
                    reply_tokens if reply_tokens > 0 else 64, MAX_CAP
                ),
            }
        )
        turn += 1
    return trial_id, rows


MAX_CAP = 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=TRACE)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens-cap", type=int, default=1024)
    args = parser.parse_args()

    global MAX_CAP
    MAX_CAP = args.max_tokens_cap

    with open(args.trace) as fh:
        data = json.load(fh)
    print(f"trials={len(data)}", flush=True)

    total_tokens = 0
    n_requests = 0
    with open(args.out, "w") as fo, Pool(
        args.workers, initializer=_init_worker, initargs=(args.model,)
    ) as pool:
        for _, rows in pool.imap(
            _convert_trial, enumerate(data), chunksize=4
        ):
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                total_tokens += r["prompt_tokens"]
                n_requests += 1
    print(
        f"requests={n_requests} total_prompt_tokens={total_tokens} "
        f"avg={total_tokens // max(n_requests, 1)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
