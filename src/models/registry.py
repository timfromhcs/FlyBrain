"""Local model registry (V6 phase_02): every entry is measured, never claimed.

Tasks: TEXT_MODEL | VISION_MODEL | IMAGE_MODEL | STT_MODEL | TTS_MODEL |
EMBEDDING_MODEL. A model is VERIFIED only after load + inference/encode
evidence exists (see diagnostics/v6/*_verification.json).
"""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

REGISTRY_PATH = os.path.join("models", "registry.json")

# Default acquisition candidates (pinned revisions where known).
DEFAULTS: List[Dict[str, Any]] = [
    {"id": "Qwen/Qwen3-0.6B-GGUF", "task": "TEXT_MODEL",
     "file": "Qwen3-0.6B-Q8_0.gguf", "revision": "main",
     "license": "Apache-2.0", "backend": "llama_cpp",
     "precision": "Q8_0", "memory_estimate_mb": 700},
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "task": "EMBEDDING_MODEL",
     "revision": "main", "license": "Apache-2.0", "backend": "sentence_transformers",
     "precision": "fp32", "memory_estimate_mb": 120},
    {"id": "openai/whisper-small", "task": "STT_MODEL",
     "revision": "main", "license": "Apache-2.0", "backend": "transformers",
     "precision": "fp32", "memory_estimate_mb": 1000},
    {"id": "hexgrad/Kokoro-82M", "task": "TTS_MODEL",
     "revision": "main", "license": "Apache-2.0", "backend": "kokoro",
     "precision": "fp32", "memory_estimate_mb": 400},
    {"id": "Lykon/dreamshaper-8-lcm", "task": "IMAGE_MODEL",
     "revision": "main", "license": "CreativeML Open RAIL-M", "backend": "diffusers",
     "precision": "fp16-cpu-fp32", "memory_estimate_mb": 4500},
]


def _sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def task_dir(task: str) -> str:
    return os.path.join("models", task.lower())


def scan_local() -> List[Dict[str, Any]]:
    """Inspect models/ and report per-file measured facts (size, sha)."""
    entries = []
    for cand in DEFAULTS:
        d = task_dir(cand["task"])
        local = None
        if os.path.isdir(d):
            cands = []
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                if not os.path.isfile(p):
                    continue
                if cand.get("file") and fn != cand["file"]:
                    continue
                if fn.split(".")[-1] not in ("gguf", "safetensors", "bin", "onnx", "pt"):
                    continue
                cands.append((p, os.path.getsize(p)))
            if cands:
                # prefer safetensors, then largest weight file
                cands.sort(key=lambda t: (0 if t[0].endswith(".safetensors") else 1,
                                          -t[1]))
                p, size = cands[0]
                local = {"path": p, "size": size}
        entry = dict(cand)
        entry["local_path"] = local["path"] if local else None
        entry["file_size"] = local["size"] if local else None
        entry["sha256"] = _sha256(local["path"]) if local else None
        entry["present"] = local is not None
        entry["verified"] = False  # set True only by verification scripts
        entries.append(entry)
    return entries


def write_registry(verified: Optional[Dict[str, bool]] = None) -> List[Dict[str, Any]]:
    entries = scan_local()
    if verified:
        for e in entries:
            if e["id"] in verified:
                e["verified"] = bool(verified[e["id"]])
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    json.dump(entries, open(REGISTRY_PATH, "w", encoding="utf-8"), indent=2)
    return entries
