"""层标记目录的代际快照。

从 commit.py 拆出：提交凭证（commits/*.commit.json）整体删除后，只有
这一项与标记目录相关，且它读的是 meta/.scan-generation，与提交凭证
无关。
"""

from __future__ import annotations

from pathlib import Path


def marker_generation(meta_dir) -> str:
    try:
        return (Path(meta_dir) / ".scan-generation").read_bytes().hex()
    except OSError:
        return ""
