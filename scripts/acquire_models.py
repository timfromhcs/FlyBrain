#!/usr/bin/env python3
"""Explicit model acquisition (dev operation, NOT runtime).

Downloads pinned models from Hugging Face into models/<task>/ and records
SHA-256. Refuses when FLYBRAIN_OFFLINE=1. Runtime never downloads.
Usage: acquire_models.py [embedding|text|stt|tts|image|vision|all]
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    if os.environ.get("FLYBRAIN_OFFLINE", "0") == "1":
        sys.exit("OFFLINE: acquisition refused (FLYBRAIN_OFFLINE=1)")
    from huggingface_hub import snapshot_download, hf_hub_download
    from src.models.registry import task_dir, DEFAULTS
    which = sys.argv[1] if len(sys.argv) > 1 else "embedding"
    for cand in DEFAULTS:
        key = cand["task"].replace("_MODEL", "").lower()
        if which != "all" and which != key:
            continue
        dest = os.path.join(PROJECT_ROOT, task_dir(cand["task"]))
        os.makedirs(dest, exist_ok=True)
        if cand["task"] == "TEXT_MODEL" and cand.get("file"):
            p = hf_hub_download(cand["id"], cand["file"], revision=cand["revision"],
                                local_dir=dest)
            print(f"downloaded {p}")
        else:
            p = snapshot_download(cand["id"], revision=cand["revision"],
                                  local_dir=dest,
                                  allow_patterns=(["*.json", "*.txt", "*.model", "*.safetensors",
                                                   "*.bin", "*.onnx", "*.gguf"] if which != "image"
                                                  else ["*.json", "*.txt", "*.safetensors", "*.bin"]))
            print(f"downloaded {p}")
    from src.models.registry import write_registry
    for e in write_registry():
        print(f"{e['task']}: present={e['present']} size={e['file_size']}")


if __name__ == "__main__":
    main()
