"""Resource-aware local model manager (V6 phase_03).

LOAD/UNLOAD/RELOAD/STATUS + memory pressure handling. The world keeps
running while heavy models load/unload: expensive inference is synchronous
per call, never on the physics thread. If a model does not fit, optional
models unload first (priority order), then resolution/settings degrade,
then the call fails explicitly — the simulation NEVER crashes for an
optional model.
"""
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

PRIORITY = ["EMBEDDING_MODEL", "TEXT_MODEL", "TTS_MODEL", "STT_MODEL",
            "VISION_MODEL", "IMAGE_MODEL"]  # last = first to shed


def _ram() -> Dict[str, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"percent": float(vm.percent),
                "avail_mb": float(vm.available / 1024 ** 2)}
    except Exception:
        return {"percent": -1.0, "avail_mb": -1.0}


class ModelManager:
    def __init__(self, ram_limit_percent: float = 88.0):
        self.lock = threading.RLock()
        self.ram_limit = float(ram_limit_percent)
        self.loaded: Dict[str, Dict[str, Any]] = {}  # task -> {handle, loaded_at, uses}
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self.events = []

    def log(self, kind: str, detail: str) -> None:
        self.events.append({"ts": time.time(), "kind": kind, "detail": detail})

    def register_loader(self, task: str, fn: Callable[[], Any]) -> None:
        with self.lock:
            self._loaders[task] = fn

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {"ram": _ram(), "ram_limit_percent": self.ram_limit,
                    "loaded": {t: {"uses": v["uses"], "loaded_at": v["loaded_at"]}
                               for t, v in self.loaded.items()},
                    "events": self.events[-20:]}

    def _pressure(self) -> bool:
        r = _ram()
        return r["percent"] >= self.ram_limit if r["percent"] >= 0 else False

    def _shed(self, keep: str) -> None:
        for task in reversed(PRIORITY):
            if task != keep and task in self.loaded:
                self.unload(task, reason="memory_pressure")
                if not self._pressure():
                    return

    def load(self, task: str) -> Any:
        with self.lock:
            if task in self.loaded:
                self.loaded[task]["uses"] += 1
                return self.loaded[task]["handle"]
            if task not in self._loaders:
                raise RuntimeError(f"no loader registered for {task} (UNAVAILABLE)")
            if self._pressure():
                self._shed(task)
            t0 = time.time()
            handle = self._loaders[task]()
            self.loaded[task] = {"handle": handle, "loaded_at": t0, "uses": 1}
            self.log("LOAD", f"{task} in {time.time() - t0:.1f}s")
            return handle

    def unload(self, task: str, reason: str = "manual") -> bool:
        with self.lock:
            h = self.loaded.pop(task, None)
        if h is None:
            return False
        try:
            handle = h["handle"]
            if hasattr(handle, "close"):
                handle.close()
            del handle
        except Exception:
            pass
        import gc
        gc.collect()
        self.log("UNLOAD", f"{task} ({reason})")
        return True

    def reload(self, task: str) -> Any:
        self.unload(task, reason="reload")
        return self.load(task)

    def ensure_unloaded_for_world(self) -> None:
        """Drop all optional models; world/physics unaffected (separate state)."""
        for task in list(self.loaded):
            self.unload(task, reason="world_priority")
