"""Rank-local commit records for all-TP-rank cache visibility."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

COMMIT_VERSION = 1
COMMIT_DIR = "commits"
COMMIT_SUFFIX = ".commit.json"


@dataclass(frozen=True)
class RankCommitRecord:
    version: int
    namespace: str
    chunk_key: str
    generation: str
    num_layers: int
    slot_bytes: int
    rank_id: int
    marker_generation: str
    pool_generation: int

    @classmethod
    def parse(cls, payload: dict) -> "RankCommitRecord":
        record = cls(
            version=int(payload["version"]),
            namespace=str(payload["namespace"]),
            chunk_key=str(payload["chunk_key"]),
            generation=str(payload["generation"]),
            num_layers=int(payload["num_layers"]),
            slot_bytes=int(payload["slot_bytes"]),
            rank_id=int(payload["rank_id"]),
            marker_generation=str(payload["marker_generation"]),
            pool_generation=int(payload["pool_generation"]),
        )
        if record.version != COMMIT_VERSION:
            raise ValueError("rank commit version mismatch")
        if len(record.chunk_key) != 32:
            raise ValueError("rank commit chunk key must be 16-byte hex")
        bytes.fromhex(record.chunk_key)
        if not record.namespace or not record.generation:
            raise ValueError("rank commit namespace/generation is empty")
        if record.num_layers <= 0 or record.slot_bytes <= 0:
            raise ValueError("rank commit geometry must be positive")
        if record.rank_id < 0 or record.pool_generation <= 0:
            raise ValueError("rank commit rank/pool generation is invalid")
        if not record.marker_generation:
            raise ValueError("rank commit marker generation is empty")
        return record


def commit_path(root, chunk_key: bytes) -> Path:
    return Path(root) / COMMIT_DIR / (bytes(chunk_key).hex() + COMMIT_SUFFIX)


def write_rank_commit(root, record: RankCommitRecord) -> None:
    path = commit_path(root, bytes.fromhex(record.chunk_key))
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    temporary = path.with_name(f".{path.name}.{stamp}")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(asdict(record), handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_rank_commit(root, chunk_key: bytes) -> RankCommitRecord | None:
    try:
        payload = json.loads(commit_path(root, chunk_key).read_text("utf-8"))
        return RankCommitRecord.parse(payload)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError,
            json.JSONDecodeError):
        return None


def scan_rank_commits(root) -> dict[bytes, RankCommitRecord]:
    directory = Path(root) / COMMIT_DIR
    records: dict[bytes, RankCommitRecord] = {}
    if not directory.is_dir():
        return records
    for path in directory.glob("*" + COMMIT_SUFFIX):
        name = path.name[:-len(COMMIT_SUFFIX)]
        try:
            chunk_key = bytes.fromhex(name)
        except ValueError:
            continue
        if len(chunk_key) != 16:
            continue
        record = read_rank_commit(root, chunk_key)
        if record is not None:
            records[chunk_key] = record
    return records


def remove_rank_commit(root, chunk_key: bytes) -> None:
    path = commit_path(root, chunk_key)
    existed = path.exists()
    path.unlink(missing_ok=True)
    if existed and path.parent.is_dir():
        _fsync_dir(path.parent)


def marker_generation(meta_dir) -> str:
    try:
        return (Path(meta_dir) / ".scan-generation").read_bytes().hex()
    except OSError:
        return ""


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
