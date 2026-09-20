"""FlyBrain Lab — Hugging Face Space entrypoint (Docker SDK).

Thin wrapper over the canonical FastAPI app (src/ui/server.py). The Space
runs CPU-only: FLYBRAIN_USE_GPU=0 unless the host truly provides Vulkan, and
/api/health always reports the real backend (never fake Vulkan).
"""
import os
import sys

SPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SPACE_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Space-safe defaults (overridable via Space Secrets / env).
os.environ.setdefault("FLYBRAIN_CIRCUIT_SIZE", "256")
os.environ.setdefault("FLYBRAIN_GRAPH_MODE", "REAL_SUBGRAPH")
os.environ.setdefault("FLYBRAIN_USE_GPU", "0")
os.environ.setdefault("FLYBRAIN_SEED", "42")

from src.ui.server import app  # noqa: E402  (re-exported for uvicorn app:app)

print(f"[flybrain-space] repo={REPO_ROOT} "
      f"circuit={os.environ['FLYBRAIN_CIRCUIT_SIZE']} "
      f"mode={os.environ['FLYBRAIN_GRAPH_MODE']} "
      f"use_gpu={os.environ['FLYBRAIN_USE_GPU']}", flush=True)
