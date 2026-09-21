#!/usr/bin/env python3
"""持续采样 worker 内存 + Tutti 状态，用于泄漏相关性分析。

背景（2026-09-21）：长跑中单个 worker RSS 从 4.6 GB 缓增（约 6 GB/小时），
需要判断增长与谁同步——Tutti 的 resident（chunk 数）、vLLM 的请求数、
还是与两者都无关（例如 CUDA host 侧缓存）。

数据源：
  * /proc/<pid>/status 的 VmRSS / RssAnon / RssFile / RssShmem / VmHWM，
    以及 Threads 与 open fd 数——RssAnon 增长指向 Python/C 堆上的对象，
    RssFile 增长指向 mmap（模型权重、mmap 文件、CUDA 映射）；
  * /metrics —— 请求数 / token 数 / 在跑数（vLLM 侧活动量）；
  * 日志尾部的 [tutti] health —— resident / hit / evicted（Tutti 侧状态）。

被采样进程按 cmdline 角色分类（worker / engine / api）：泄漏未必在 worker，
上一代 OOM 也是单进程 255GB，先分清角色再谈归因。

每 INTERVAL 秒追加一行 JSON 到 OUT，便于事后算相关系数。
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
import urllib.request

LOG = os.environ.get(
    "MEMWATCH_LOG",
    "/mnt/nvme4/tutti-profile/online/long4t/server-fresh.log",
)
OUT = os.environ.get(
    "MEMWATCH_OUT",
    "/mnt/nvme4/tutti-profile/online/memwatch.jsonl",
)
INTERVAL = int(os.environ.get("MEMWATCH_INTERVAL", "60"))
# 精确 fd 计数的周期。代价与进程数×fd 数成正比，默认 5 分钟一次。
FD_INTERVAL = int(os.environ.get("MEMWATCH_FD_INTERVAL", "300"))
METRICS_URL = os.environ.get("MEMWATCH_METRICS", "http://127.0.0.1:8192/metrics")

_METRIC_PATTERNS = (
    ('reason="length"', "req_len"),
    ('reason="abort"', "req_abort"),
    ('reason="error"', "req_err"),
    ("generation_tokens_total", "gen_tokens"),
    ("prompt_tokens_total", "prompt_tokens"),
    ("num_requests_running", "running"),
    ("num_requests_waiting{", "waiting"),
)

_HEALTH_INT = (
    "resident", "pending", "pinned", "hit_tokens", "hit_reqs",
    "evicted", "drift", "reclaimed",
)

# cmdline 标记 → 角色。Worker TP 进程名形如 "VLLM::Worker_TP0"。
_ROLE_MARKERS = (
    ("Worker_TP", "worker"),
    ("EngineCore", "engine"),
    ("api_server", "api"),
)

# /proc/<pid>/status 里以 kB 计的字段 → 记录名（统一折算成 GB）。
_STATUS_GB = {
    "VmRSS": "rss_gb",
    "RssAnon": "anon_gb",
    "RssFile": "file_gb",
    "RssShmem": "shmem_gb",
    "VmHWM": "peak_gb",
    "VmData": "data_gb",
}
# 计数类字段（不做单位换算）。
_STATUS_COUNT = {"Threads": "threads"}


def _proc_detail(pid_dir: str, *, count_fds: bool) -> dict:
    detail: dict[str, float] = {}
    try:
        with open(pid_dir + "/status") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                value = rest.split()
                if not value:
                    continue
                if key in _STATUS_GB:
                    detail[_STATUS_GB[key]] = round(int(value[0]) / 1024 / 1024, 3)
                elif key in _STATUS_COUNT:
                    detail[_STATUS_COUNT[key]] = int(value[0])
    except Exception:
        return detail
    # fd 计数按低频调用：105k 个 fd 的 listdir 实测 ~66ms/进程，8 个 worker
    # 一轮 530ms（=15s 采样周期下 3.5% 单核），且要内核逐项走 fd 表。
    # 趋势用 /proc/sys/fs/file-nr（O(1)）看，精确值低频取。
    if count_fds:
        try:
            detail["fds"] = len(os.listdir(pid_dir + "/fd"))
        except Exception:
            pass
    return detail


def sample_processes(*, count_fds: bool = False) -> list[dict]:
    """按角色返回所有相关进程的内存明细。"""
    out: list[dict] = []
    for path in glob.glob("/proc/[0-9]*"):
        try:
            cmdline = open(path + "/cmdline", "rb").read().decode(
                "utf-8", "ignore"
            )
        except Exception:
            continue
        role = None
        for marker, name in _ROLE_MARKERS:
            if marker in cmdline:
                role = name
                break
        if role is None:
            continue
        entry = {"role": role, "pid": int(os.path.basename(path))}
        entry.update(_proc_detail(path, count_fds=count_fds))
        out.append(entry)
    return out


def system_file_handles() -> int:
    """全系统已分配的文件句柄数（O(1)，无逐进程遍历）。"""
    try:
        with open("/proc/sys/fs/file-nr") as handle:
            return int(handle.read().split()[0])
    except Exception:
        return 0


def read_metrics() -> dict:
    try:
        raw = urllib.request.urlopen(METRICS_URL, timeout=5).read().decode()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for marker, key in _METRIC_PATTERNS:
        for line in raw.splitlines():
            if line.startswith("#") or marker not in line:
                continue
            match = re.search(r"\} ([\d.eE+]+)$", line)
            if match:
                try:
                    out[key] = float(match.group(1))
                except ValueError:
                    pass
                break
    return out


def read_log_tail() -> str:
    try:
        with open(LOG, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 400_000))
            return handle.read().decode("utf-8", "ignore")
    except Exception:
        return ""


def _last_int(text: str, pattern: str, cast=int):
    hits = re.findall(pattern, text)
    return cast(hits[-1]) if hits else None


def read_storage_stats(text: str) -> dict:
    """从日志尾部抓 Tutti 侧的 per-rank 落盘计数。

    checkpoint 每隔 60s 被 8 个 rank 各调一次（实测 n=336 次 / 8 天），
    这里同时记录"日志里出现过的总量"与"采样周期内的增量"，用于判断
    index 落盘是否持续生效、以及是否与内存增长同步。
    """
    stats: dict[str, float] = {}
    written = _last_int(text, r"DIRECT_CHECKPOINT_WRITTEN", str)
    stats["ckpt_total"] = (
        float(len(re.findall(r"DIRECT_CHECKPOINT_WRITTEN", text)))
    )
    stats["ckpt_failed_total"] = float(
        len(re.findall(r"DIRECT_CHECKPOINT_FAILED", text))
    )
    elapsed = re.findall(r"DIRECT_CHECKPOINT_WRITTEN elapsed=([0-9.]+)s", text)
    if elapsed:
        stats["ckpt_last_elapsed_s"] = float(elapsed[-1])
    return stats


def read_health(text: str) -> dict:
    """读日志尾部最后一条 [tutti] health。"""
    line = None
    for candidate in text.splitlines():
        if "[tutti] health" in candidate:
            line = candidate
    if line is None:
        return {}
    line = None
    for candidate in text.splitlines():
        if "[tutti] health" in candidate:
            line = candidate
    if line is None:
        return {}
    out: dict[str, int] = {}
    for key in _HEALTH_INT:
        match = re.search(rf"{key}=(\d+)", line)
        if match:
            out[key] = int(match.group(1))
    for key in ("capacity",):
        match = re.search(rf"/\s*(\d+)\(", line)
        if match:
            out[key] = int(match.group(1))
    return out


def summarize(procs: list[dict]) -> dict:
    """从进程明细里抽出便于看趋势的聚合量。"""
    if not procs:
        return {"workers": 0, "rss_total_gb": 0.0, "rss_max_gb": 0.0,
                "rss_each": []}
    ranked = sorted(procs, key=lambda p: p.get("rss_gb", 0.0), reverse=True)
    biggest = ranked[0]
    workers = [p for p in procs if p["role"] == "worker"]
    record = {
        "workers": len(workers),
        "rss_total_gb": round(sum(p.get("rss_gb", 0.0) for p in procs), 3),
        "rss_max_gb": biggest.get("rss_gb", 0.0),
        "rss_each": [p.get("rss_gb", 0.0) for p in workers],
        # 全进程合计：区分匿名堆与文件映射谁在涨。
        "anon_total_gb": round(sum(p.get("anon_gb", 0.0) for p in procs), 3),
        "file_total_gb": round(sum(p.get("file_gb", 0.0) for p in procs), 3),
        "threads_total": sum(int(p.get("threads", 0)) for p in procs),
        "fds_total": sum(int(p.get("fds", 0)) for p in procs),
        # 最大进程的明细（上一代 OOM 就是单进程涨到 255GB）。
        "max_role": biggest["role"],
        "max_pid": biggest["pid"],
        "anon_max_gb": biggest.get("anon_gb", 0.0),
        "file_max_gb": biggest.get("file_gb", 0.0),
        "data_max_gb": biggest.get("data_gb", 0.0),
        "peak_max_gb": biggest.get("peak_gb", 0.0),
        "threads_max": int(biggest.get("threads", 0)),
        "fds_max": int(biggest.get("fds", 0)),
    }
    return record


def main() -> int:
    print(f"[memwatch] log={LOG}", flush=True)
    print(f"[memwatch] out={OUT} interval={INTERVAL}s "
          f"fd_interval={FD_INTERVAL}s", flush=True)
    tick = 0
    while True:
        # 精确 fd 计数很贵（105k fd 的进程 listdir ~66ms，8 个可达 530ms），
        # 只按 FD_INTERVAL 低频取；每轮用 O(1) 的全局句柄数看趋势。
        count_fds = tick % max(1, FD_INTERVAL // INTERVAL) == 0
        tick += 1
        procs = sample_processes(count_fds=count_fds)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": int(time.time()),
            "file_nr": system_file_handles(),
        }
        record.update(summarize(procs))
        record["procs"] = [
            {k: v for k, v in p.items() if v}
            for p in sorted(procs, key=lambda p: p["pid"])
        ]
        record.update(read_metrics())
        tail = read_log_tail()
        record.update({f"t_{k}": v for k, v in read_health(tail).items()})
        record.update(read_storage_stats(tail))
        ckpt_elapsed = re.findall(
            r"DIRECT_CHECKPOINT_WRITTEN elapsed=([0-9.]+)s", tail
        )
        if ckpt_elapsed:
            record["ckpt_last_elapsed_s"] = float(ckpt_elapsed[-1])
        with open(OUT, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        summary = {
            k: record[k] for k in
            ("ts", "rss_max_gb", "anon_max_gb", "file_max_gb", "threads_max",
             "req_len", "t_resident", "t_hit_reqs", "ckpt_total",
             "ckpt_last_elapsed_s")
            if k in record
        }
        print(json.dumps(summary), flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[memwatch] stopped", flush=True)
