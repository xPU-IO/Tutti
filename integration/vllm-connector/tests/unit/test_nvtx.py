"""NVTX profiling annotations stay optional and direction-colored."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import nvtx as tutti_nvtx


def test_direction_palette():
    assert tutti_nvtx._color_for("tutti.runtime.submit|op=read") == 0xFF00B8D9
    assert tutti_nvtx._color_for("tutti.load.scatter") == 0xFF00B8D9
    assert tutti_nvtx._color_for("tutti.runtime.submit|op=write") == 0xFFFF8C00
    assert tutti_nvtx._color_for("tutti.store.gather") == 0xFFFF8C00
    assert tutti_nvtx._color_for("tutti.runtime.wait") == 0xFF8D99A6


def test_colored_nvtx_annotation(monkeypatch):
    calls = []

    class Annotation:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def __enter__(self):
            calls.append(("enter", None))

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", (exc_type, exc, tb)))

    monkeypatch.setenv("TUTTI_NVTX", "1")
    monkeypatch.setitem(sys.modules, "nvtx", SimpleNamespace(annotate=Annotation))

    with tutti_nvtx.range("tutti.runtime.submit|op=read"):
        calls.append(("body", None))

    assert calls[0] == (
        "init",
        {
            "message": "tutti.runtime.submit|op=read",
            "color": 0xFF00B8D9,
            "domain": "tutti",
        },
    )
    assert [name for name, _ in calls[1:]] == ["enter", "body", "exit"]


def test_disabled_nvtx_does_not_import_backend(monkeypatch):
    monkeypatch.delenv("TUTTI_NVTX", raising=False)
    monkeypatch.delitem(sys.modules, "nvtx", raising=False)

    with tutti_nvtx.range("tutti.runtime.submit|op=write"):
        pass
