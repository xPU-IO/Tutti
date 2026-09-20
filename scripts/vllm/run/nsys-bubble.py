#!/usr/bin/env python3
"""对比两份 nsys 报告的 GPU 气泡。

气泡 = 所有 stream 上核函数区间的并集之外的时间。若某段时间没有任何 stream 在跑
核函数，前向就只有主机侧工作在推进，那一段就是用户要看的"气泡"。

同时报告核函数之间的长间隔（>5ms），便于在 nsys UI 里定位具体位置。
"""
import sqlite3
import sys


def analyse(path: str, label: str) -> dict:
    con = sqlite3.connect(path)
    cur = con.cursor()

    rows = cur.execute(
        "SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE end IS NOT NULL ORDER BY start"
    ).fetchall()
    if not rows:
        print(f"{label}: 无核函数")
        return {}

    # 合并所有 stream 的核函数区间。
    merged = []
    for start, end, _ in rows:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])

    span = merged[-1][1] - merged[0][0]
    busy = sum(e - s for s, e in merged)
    idle = span - busy

    # 核函数之间的长间隔（跨 stream 的整体空档）。
    gaps = []
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        if gap >= 5_000_000:  # 5ms
            gaps.append((merged[i - 1][1], gap))

    # 每个 stream 的活动核函数数，用于识别"没用的 stream"。
    per_stream = cur.execute(
        "SELECT streamId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "GROUP BY streamId ORDER BY 2 DESC LIMIT 6"
    ).fetchall()

    con.close()

    print(f"\n=== {label} ===")
    print(f"  GPU 活动跨度      {span/1e9:8.3f} s")
    print(f"  其中繁忙          {busy/1e9:8.3f} s  ({busy/span*100:5.1f}%)")
    print(f"  其中空闲（气泡）  {idle/1e9:8.3f} s  ({idle/span*100:5.1f}%)")
    print(f"  核函数总数        {len(rows):8d}")
    if gaps:
        total_gap = sum(g for _, g in gaps)
        print(f"  >5ms 空档         {len(gaps):8d} 段，合计 {total_gap/1e6:7.1f} ms")
        print("  最大 5 段（起点相对跨度起点，秒 / 时长 ms）:")
        for t, g in sorted(gaps, key=lambda x: -x[1])[:5]:
            print(f"      +{(t-merged[0][0])/1e9:7.3f}s   {g/1e6:8.1f} ms")
    else:
        print("  >5ms 空档         无")
    print("  stream 分布（前 6）:", ", ".join(f"s{s}:{c}" for s, c in per_stream))

    return {
        "span": span,
        "busy": busy,
        "idle": idle,
        "gaps": gaps,
        "n": len(rows),
    }


if __name__ == "__main__":
    base = analyse(sys.argv[1], "HBM 基线（零存储工作）")
    tutti = analyse(sys.argv[2], "Tutti 直连（NVMe 复用）")

    if base and tutti:
        print("\n=== 差异（Tutti − 基线）===")
        print(f"  繁忙时间  {(tutti['busy']-base['busy'])/1e9:+8.3f} s")
        print(f"  气泡时间  {(tutti['idle']-base['idle'])/1e9:+8.3f} s")
        print(f"  核函数数  {tutti['n']-base['n']:+8d}")
        print(f"  >5ms 空档 {len(tutti['gaps'])-len(base['gaps']):+8d} 段")
