"""Optional NVTX ranges for connector profiling.

Set ``TUTTI_NVTX=1`` to expose request, layer, transfer, and completion
boundaries in Nsight Systems.  Profiling remains opt-in and degrades to a
no-op when CUDA/NVTX is unavailable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager


# ARGB colors chosen to stay distinct from Nsight's default blue compute
# kernels and from each other on both light and dark timelines.
_READ_COLOR = 0xFF00B8D9
_WRITE_COLOR = 0xFFFF8C00
_WAIT_COLOR = 0xFF8D99A6
_DEFAULT_COLOR = 0xFF5B8FF9


def enabled() -> bool:
    return os.environ.get("TUTTI_NVTX", "0").lower() in {
        "1", "true", "yes", "on"
    }


def _color_for(name: str) -> int:
    lowered = name.lower()
    if "op=read" in lowered or "load" in lowered or "scatter" in lowered:
        return _READ_COLOR
    if "op=write" in lowered or "store" in lowered or "save" in lowered \
            or "gather" in lowered:
        return _WRITE_COLOR
    if "wait" in lowered:
        return _WAIT_COLOR
    return _DEFAULT_COLOR


@contextmanager
def range(name: str, color: int | str | None = None):
    backend = None
    annotation = None
    if enabled():
        try:
            import nvtx
            annotation = nvtx.annotate(
                message=str(name),
                color=_color_for(str(name)) if color is None else color,
                domain="tutti",
            )
            annotation.__enter__()
            backend = "nvtx"
        except Exception:
            try:
                import torch
                torch.cuda.nvtx.range_push(str(name))
                backend = "torch"
            except Exception:
                pass
    try:
        yield
    finally:
        if backend == "nvtx":
            try:
                annotation.__exit__(None, None, None)
            except Exception:
                pass
        elif backend == "torch":
            try:
                import torch
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
