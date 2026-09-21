#!/usr/bin/env python3
"""按区域形状归类一个进程的匿名内存，回答"涨的是哪类内存"。

动机（2026-09-21 泄漏排查）：worker RSS 从 4.6GB 涨到 48GB 然后在 PCIe AER
噪声中崩掉，而 Tutti 侧的元数据容器理论上只有百 MB 量级。要判断增长是
"大量小对象"（pymalloc arena / glibc arena，按 thread 扩张）还是"少数巨大
连续分配"（增长的 list/dict/array），光看 RSS 总量不够——必须看区域的
数量、粒度与归属。

做法：解析 /proc/<pid>/smaps，按 (匿名/文件, 大小) 归类：
  * anon 区域按大小分桶：<64K / 64K-1M / 1M-64M / 64M-512M / >=512M；
    pymalloc arena 是 256KB 量级、glibc 主分配区与 mmap 阈值以上的大块是
    64MB 量级，可作为指纹；
  * file 区域按路径前缀归类（mmap 的模型文件 / CUDA 驱动 / .so / 其他），
    排除"其实是文件映射"的假阳性；
  * 报告 Rss/Anon/Shared 合计与 top-20 最大匿名区域（起始地址 + 大小），
    便于同一进程前后两次对比时逐区域对齐。

用法：smaps_shape.py <pid> [<pid> ...]
"""

from __future__ import annotations

import collections
import re
import sys

_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s*(.*)$")
_BUCKETS = (
    ("<64K", 1 << 16),
    ("64K-1M", 1 << 20),
    ("1M-64M", 64 << 20),
    ("64M-512M", 512 << 20),
    ("512M-2G", 2 << 30),
    (">=2G", 1 << 62),
)


def _bucket(size: int) -> str:
    for name, limit in _BUCKETS:
        if size < limit:
            return name
    return _BUCKETS[-1][0]


def _need(region: dict, key: str) -> int:
    return int(region.get(key, 0))


def scan(pid: int) -> None:
    regions: list[dict] = []
    current: dict | None = None
    path = f"/proc/{pid}/smaps"
    with open(path) as handle:
        for line in handle:
            match = _HEADER.match(line)
            if match:
                start, end, perms, name = match.groups()
                current = {
                    "start": int(start, 16),
                    "end": int(end, 16),
                    "size": int(end, 16) - int(start, 16),
                    "name": name.strip(),
                    "perms": perms,
                }
                regions.append(current)
                continue
            if current is None:
                continue
            key, _, rest = line.partition(":")
            value = rest.split()
            if not value:
                continue
            if key in ("Rss", "Anonymous", "Shared_Clean", "Shared_Dirty",
                       "Private_Clean", "Private_Dirty", "Swap"):
                current[key] = int(value[0]) * 1024

    anon_buckets: dict[str, list[int]] = collections.defaultdict(list)
    file_by_name: dict[str, list[int]] = collections.defaultdict(list)
    anon_regions: list[tuple[int, int]] = []
    total_rss = total_anon = total_swap = 0
    for region in regions:
        rss = _need(region, "Rss")
        total_rss += rss
        total_swap += _need(region, "Swap")
        name = region["name"]
        # 具名映射里只有 [heap]/[stack]/[anon:*] 是匿名的，其余是文件映射
        # （模型权重、CUDA 驱动、.so），必须分开统计，否则会把文件映射当成
        # "Python 对象在涨"。
        if name and not _is_anonymous_name(name):
            file_by_name[name.split()[0]].append(rss)
            continue
        total_anon += _need(region, "Anonymous")
        # 分桶按 **Rss** 而非虚拟大小：CUDA/驱动会预留几十 GB 的虚拟区间，
        # 按虚拟大小分桶会把"真正驻留的几 GB"埋掉。
        anon_buckets[_bucket(rss)].append(rss)
        if rss >= (4 << 20):
            anon_regions.append((region["start"], rss))

    print(f"=== pid {pid} ===")
    print(
        f"RSS={total_rss / 2**30:.2f} GB  Anon={total_anon / 2**30:.2f} GB  "
        f"Swap={total_swap / 2**30:.2f} GB  区域数={len(regions)}"
    )
    print("-- 匿名区域按 **驻留 RSS** 分桶（个数 / 合计 GB）--")
    for name, _ in _BUCKETS:
        sizes = anon_buckets.get(name)
        if not sizes:
            continue
        print(f"  {name:<10} n={len(sizes):>6}  {sum(sizes) / 2**30:>8.2f} GB")
    print("-- 非匿名映射 top（按合计 GB）--")
    ranked = sorted(
        file_by_name.items(), key=lambda kv: sum(kv[1]), reverse=True
    )[:6]
    for name, sizes in ranked:
        print(f"  {sum(sizes) / 2**30:>8.2f} GB  n={len(sizes):>5}  {name}")
    print("-- 最大匿名区域 top-10（地址 / 大小 GB）--")
    for start, size in sorted(anon_regions, key=lambda item: -item[1])[:10]:
        print(f"  0x{start:x}  {size / 2**30:>8.3f} GB  ({size / 2**20:.1f} MB)")


def _is_anonymous_name(name: str) -> bool:
    """具名但内容匿名：内核的 [heap]/[stack]/[anon:*]/[anon_shmem:*]。"""
    return name.startswith(("[heap]", "[stack]", "[anon", "[vdso]"))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for raw in argv[1:]:
        scan(int(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
