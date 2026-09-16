"""Shared pytest fixtures and build prerequisite checks (task T-101).

Prerequisite: the extension must have been built against an existing tutti
C++ build tree (setup.py refuses to build tutti itself). If TUTTI_BUILD_DIR
is explicitly provided, verify libtutti_presets exists there so the failure
mode is obvious at collection time.
"""

import os

import pytest

import tutti_runtime  # noqa: F401


def _check_build_dir():
    build_dir = os.environ.get("TUTTI_BUILD_DIR")
    if not build_dir:
        return  # not explicitly provided; setup.py already resolved/probed
    expected = os.path.join(build_dir, "tutti", "presets", "libtutti_presets.a")
    if not os.path.exists(expected):
        pytest.exit(
            "TUTTI_BUILD_DIR=%s has no tutti/presets/libtutti_presets.a; "
            "per task T-101 this binding never builds tutti itself."
            % build_dir,
            returncode=1,
        )


_check_build_dir()
