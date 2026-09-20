#!/usr/bin/env python
"""Tutti 在线长跑压测驱动：服务模式 × HBM 隔离 × 输出一致性 × 周期汇总。

为什么这样设计
--------------
1. 固定一组 prompt 反复发送。从数据集取 N 条**不同 trial** 的 turn=0 请求
   （trial 内前缀递增，取不同 trial 可保证彼此的前缀不重叠），此后每轮都发
   同一组——第 1 轮起每个请求都应命中上一轮写入的 KV。

2. 每轮开始前 POST /reset_prefix_cache（默认 reset_external=false）：只清
   vLLM 自己的 HBM 前缀缓存，保留 Tutti 盘上数据。**这是实验成立的关键**，
   与 offline 驱动加 --reset-local-prefix-between-requests 同理：不 reset 时
   命中全部来自 HBM，Tutti 的读路径根本不会被触发，测的其实是 vLLM 自己。

3. 第 0 轮是冷启动（池空）：全量 prefill + 写盘，其输出作为基线。第 1 轮起
   HBM 已清空，命中只能来自 Tutti（经 NVMe 读回）。

4. temperature=0 + ignore_eos：同一 prompt 的输出应与基线**逐 token 一致**。
   不一致就是数据面正确性问题——这是比 offline 端 token 对比更直接的判据，
   因为它穿过完整的 prefill/KV 读回路径。

5. 长跑：--minutes 到点后跑完当前轮再退出；每轮一行 summary 追加落盘，
   单次中断不丢已完成轮次的数据。

用法
----
    python vllm_online_longrun.py \
        --base-url http://127.0.0.1:8192 \
        --dataset /mnt/nvme4/datasets/codex_swebenchpro/requests_hy3.jsonl \
        --samples 4 --max-prompt-tokens 20000 --max-tokens 8 \
        --minutes 15 --out-dir /mnt/nvme4/tutti-profile/online/<tag>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests


def _load_prompts(path: Path, samples: int, max_prompt_tokens: int) -> list[dict]:
    """取 samples 条不同 trial 的 turn=0 请求（前缀互不重叠）。

    只收 prompt_tokens 在 (0, max_prompt_tokens] 内的条目：超长 prompt 会被
    server 按 max_model_len 截断或直接拒绝，截断会静默改变请求形状，宁可跳过。
    """
    picked: dict[int, dict] = {}
    with path.open() as handle:
        for line in handle:
            if len(picked) >= samples:
                break
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if int(entry.get("turn", 0)) != 0:
                continue
            trial = int(entry["trial_id"])
            if trial in picked:
                continue
            tokens = int(entry.get("prompt_tokens") or 0)
            if tokens <= 0 or tokens > max_prompt_tokens:
                continue
            picked[trial] = {
                "trial_id": trial,
                "turn": 0,
                "prompt": entry["prompt"],
                "dataset_prompt_tokens": tokens,
            }
    if len(picked) < samples:
        raise SystemExit(
            f"数据集里只找到 {len(picked)} 条满足 prompt_tokens<="
            f"{max_prompt_tokens} 的 turn=0 请求（需要 {samples} 条）"
        )
    return [picked[key] for key in sorted(picked)]


def _post(base_url: str, path: str, payload: dict | None = None,
          timeout: float = 1800.0) -> dict:
    resp = requests.post(
        base_url.rstrip("/") + path, json=payload, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()


_reset_probe = {"available": True}


def _reset_hbm(base_url: str) -> None:
    """尽力清空 vLLM 的 HBM 前缀缓存（reset_external=false：不动 Tutti）。

    该端点属于 vLLM 的 endpoint plugin，需要安装元数据显式注册；本机 vLLM
    没有注册（`entry_points(group="vllm.endpoint_plugins")` 为空），HTTP 层不
    存在这个路由（404）。此时**不终止长跑**：改用启动参数
    --num-gpu-blocks-override 把 HBM 池压到装不下测试集，同样能迫使复用落到
    Tutti（见 serve-8gpu-striped.sh 里的推导）。只在首次遇到时提示一条。
    """
    if not _reset_probe["available"]:
        return
    try:
        result = _post(base_url, "/reset_prefix_cache")
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (404, 405):
            _reset_probe["available"] = False
            print(
                "[online] 提示：/reset_prefix_cache 不可用（vLLM 未注册 "
                "endpoint plugin）→ 依赖 --num-gpu-blocks-override 压小的 "
                "HBM 池做隔离，后续轮次不再尝试",
                flush=True,
            )
            return
        raise
    if not result.get("success", True):
        # 仍有 blocks 被占用（在途请求/异步 offload）时返回 false，可重试。
        print(f"[online] 警告：reset_prefix_cache 返回 success=false {result}",
              flush=True)


def _send_one(base_url: str, model: str, item: dict, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    started = time.perf_counter()
    body = _post(base_url, "/v1/completions", payload)
    wall = time.perf_counter() - started
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "trial_id": item["trial_id"],
        "wall_s": wall,
        "token_ids": list(choice.get("token_ids") or []),
        "text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "dataset_prompt_tokens": item["dataset_prompt_tokens"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8192")
    parser.add_argument("--model", default="tutti",
                        help="server 的 --served-model-name")
    parser.add_argument(
        "--dataset",
        default="/mnt/nvme4/datasets/codex_swebenchpro/requests_hy3.jsonl",
    )
    parser.add_argument("--samples", type=int, default=4,
                        help="每轮的请求数（取不同 trial 的 turn=0）")
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=20000,
        help="只选不超过该长度的请求；须 ≤ server 的 max_model_len - max_tokens",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--minutes", type=float, default=15.0,
                        help="总时长上限；跑完当前轮后退出")
    parser.add_argument("--rounds", type=int, default=0,
                        help="轮数上限（0 = 只受 --minutes 约束）")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="单请求 HTTP 超时（长 prompt 冷 prefill 较慢）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = out_dir / "requests.jsonl"
    sum_path = out_dir / "summary.jsonl"

    prompts = _load_prompts(
        Path(args.dataset), args.samples, args.max_prompt_tokens
    )
    total_tokens = sum(p["dataset_prompt_tokens"] for p in prompts)
    print(
        f"[online] samples={len(prompts)} "
        f"tokens={[p['dataset_prompt_tokens'] for p in prompts]} "
        f"total={total_tokens} max_tokens={args.max_tokens} "
        f"minutes={args.minutes} out={out_dir}",
        flush=True,
    )

    started = time.time()
    deadline = started + args.minutes * 60.0
    baseline: list[list[int]] | None = None
    round_idx = 0
    req_handle = req_path.open("a")
    sum_handle = sum_path.open("a")
    try:
        while True:
            phase = "cold" if round_idx == 0 else "reuse"
            _reset_hbm(args.base_url)
            records: list[dict] = []
            failures = 0
            for idx, item in enumerate(prompts):
                try:
                    record = _send_one(
                        args.base_url, args.model, item, args.max_tokens
                    )
                except Exception as exc:  # 长跑不允许单点失败终止整轮
                    failures += 1
                    print(
                        f"[online] r={round_idx} {phase}[{idx}] "
                        f"FAILED {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                record.update(round=round_idx, phase=phase, idx=idx)
                if baseline is not None and idx < len(baseline):
                    record["match_baseline"] = (
                        record["token_ids"] == baseline[idx]
                    )
                records.append(record)
                req_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                match = record.get("match_baseline")
                print(
                    f"[online] r={round_idx} {phase}[{idx}] "
                    f"trial={item['trial_id']} "
                    f"ptok={record['prompt_tokens']} "
                    f"cached={record['cached_tokens']} "
                    f"wall={record['wall_s']:.3f}s "
                    + ("" if match is None else
                       ("match=yes" if match else "match=NO")),
                    flush=True,
                )
            req_handle.flush()

            if not records:
                print(f"[online] r={round_idx} 全部请求失败，终止", flush=True)
                break
            if baseline is None:
                baseline = [r["token_ids"] for r in records]

            walls = [r["wall_s"] for r in records]
            mismatches = sum(
                1 for r in records if r.get("match_baseline") is False
            )
            cached = [
                r["cached_tokens"] for r in records
                if r.get("cached_tokens") is not None
            ]
            summary = {
                "round": round_idx,
                "phase": phase,
                "elapsed_s": round(time.time() - started, 1),
                "requests": len(records),
                "failures": failures,
                "walls_s": [round(w, 3) for w in walls],
                "wall_mean_s": round(statistics.mean(walls), 3),
                "wall_min_s": round(min(walls), 3),
                "wall_max_s": round(max(walls), 3),
                "mismatches": mismatches,
                "cached_tokens": cached,
            }
            sum_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            sum_handle.flush()
            print(
                f"[online] ROUND {round_idx} ({phase}) done: "
                f"mean={summary['wall_mean_s']}s "
                f"min={summary['wall_min_s']}s max={summary['wall_max_s']}s "
                f"mismatches={mismatches} failures={failures} "
                f"elapsed={summary['elapsed_s']}s",
                flush=True,
            )

            round_idx += 1
            if args.rounds and round_idx >= args.rounds:
                break
            if time.time() >= deadline:
                print("[online] 时间到，正常退出", flush=True)
                break
    finally:
        req_handle.close()
        sum_handle.close()
    print(f"[online] 完成：{round_idx} 轮 → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
