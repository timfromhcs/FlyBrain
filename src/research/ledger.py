"""Research ledger V4 (mission §35): every major event receives a full
provenance record with cryptographic hashes and a chained event stream.

Deterministic: event ids derive from content + sequence, never wall-clock
(timestamps are operational metadata only).
"""
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

LEDGER_VERSION = "research_ledger_v1"

EVENT_TYPES = ("birth", "death", "mutation", "crossover", "learning", "growth",
               "pruning", "synapse_creation", "synapse_deletion", "reproduction",
               "speciation", "checkpoint", "experiment", "milestone")


def code_sha() -> str:
    """Current commit as code identity; 'uncommitted' when dirty/unavailable."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return commit if (commit and not dirty) else f"{commit or 'unknown'}+dirty"
    except Exception:
        return "unknown"


def dataset_sha(soma_path: str, connections_path: str) -> str:
    from src.connectome.loader import _file_sha256
    return hashlib.sha256(
        _file_sha256(soma_path).encode()
        + _file_sha256(connections_path).encode()).hexdigest()


def shader_sha(shader_dir: str = "shaders") -> str:
    h = hashlib.sha256()
    for name in sorted(os.listdir(shader_dir)) if os.path.isdir(shader_dir) else []:
        p = os.path.join(shader_dir, name)
        if name.endswith(".spv") and os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(name.encode() + f.read())
    return h.hexdigest()


def _record_hash(rec: Dict[str, Any]) -> str:
    # Identity covers content + sequence + chain only. Operational metadata
    # (ts wall-clock, record_hash itself) is NEVER identity: two identical
    # appends must hash identically or deterministic replay breaks.
    payload = json.dumps({k: v for k, v in rec.items()
                          if k not in ("record_hash", "ts")},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class ResearchLedger:
    """Append-only, hash-chained research event ledger."""

    def __init__(self, experiment_id: str, seed: int,
                 path: Optional[str] = None):
        self.experiment_id = experiment_id
        self.seed = int(seed)
        self.path = path
        self.records: List[Dict[str, Any]] = []
        self._prev = "0" * 64
        self._static = {
            "experiment_id": experiment_id,
            "seed": self.seed,
            "code_sha": code_sha(),
            "dataset_sha": "unknown",
            "shader_sha": shader_sha(),
            "ledger_version": LEDGER_VERSION,
        }

    def set_dataset_sha(self, sha: str) -> None:
        self._static["dataset_sha"] = sha

    def append(self, event_type: str, tick: int, generation: int,
               payload: Dict[str, Any], genome_sha: str = "", brain_sha: str = "",
               world_sha: str = "", result_sha: str = "", parent_event: str = "") \
            -> Dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown ledger event type {event_type!r}")
        seq = len(self.records)
        rec = {
            **self._static,
            "event_id": hashlib.sha256(
                f"{self.experiment_id}|{event_type}|{seq}|{self._prev}"
                .encode()).hexdigest()[:16],
            "parent_event": parent_event or (self.records[-1]["event_id"]
                                             if self.records else ""),
            "event_type": event_type,
            "seq": seq,
            "tick": int(tick),
            "generation": int(generation),
            "genome_sha": genome_sha or "none",
            "brain_sha": brain_sha or "none",
            "world_sha": world_sha or "none",
            "result_sha": result_sha or "none",
            "payload": payload,
            "prev_hash": self._prev,
            "ts": time.time(),  # operational metadata only, never identity
        }
        rec["record_hash"] = _record_hash(rec)
        self.records.append(rec)
        self._prev = rec["record_hash"]
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        return rec

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for rec in self.records:
            if rec["prev_hash"] != prev:
                return False
            if rec["record_hash"] != _record_hash(rec):
                return False
            prev = rec["record_hash"]
        return True

    def event_stream_hash(self) -> str:
        h = hashlib.sha256()
        for rec in self.records:
            h.update(rec["record_hash"].encode())
        return h.hexdigest()
