"""Optional NVTX ranges for connector profiling.

Set ``TUTTI_NVTX=1`` to expose request, layer, transfer, and completion
boundaries in Nsight Systems.  Profiling remains opt-in and degrades to a
no-op when CUDA/NVTX is unavailable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager


def enabled() -> bool:
    return os.environ.get("TUTTI_NVTX", "0").lower() in {
        "1", "true", "yes", "on"
    }


@contextmanager
def range(name: str):
    active = False
    if enabled():
        try:
            import torch
            torch.cuda.nvtx.range_push(str(name))
            active = True
        except Exception:
            pass
    try:
        yield
    finally:
        if active:
            try:
                import torch
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

