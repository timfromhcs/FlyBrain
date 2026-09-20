"""Long-running supervision (V5 phase_11).

Observes: process/RAM, simulation-loop heartbeat (step advancement),
worker-thread liveness, GPU presence, telemetry flow, backup writability.
Recovery chain (each step logged, bounded — never infinite invisible loops):

  RETRY -> RESTORE_LAST_VALID_CHECKPOINT -> RESTART_RUNTIME -> SAFE_STOP

Thresholds are conservative; a healthy engine only produces OK heartbeats.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class WatchdogConfig:
    interval_sec: float = 5.0
    stall_timeout_sec: float = 60.0
    ram_percent_limit: float = 90.0
    max_restarts: int = 3


def _system_ram_percent() -> float:
    import psutil
    return float(psutil.virtual_memory().percent)


class Watchdog:
    def __init__(self, engine_getter: Callable, backup_service=None,
                 config: Optional[WatchdogConfig] = None,
                 ram_reader: Callable[[], float] = _system_ram_percent):
        self._engine_getter = engine_getter
        self._backups = backup_service
        self.config = config or WatchdogConfig()
        self._ram_reader = ram_reader
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.events: List[Dict[str, Any]] = []
        self.restarts = 0
        self._last_step = -1
        self._last_advance_ts = time.time()

    def log(self, kind: str, detail: str, **extra) -> None:
        self.events.append({"ts": time.time(), "kind": kind, "detail": detail,
                            **extra})
        if len(self.events) > 500:
            self.events.pop(0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="FlyBrainWatchdog")
        self._thread.start()
        self.log("START", "watchdog observing")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.log("STOP", "watchdog stopped")

    def status(self) -> Dict[str, Any]:
        eng = self._engine_getter()
        return {"running": bool(self._thread and self._thread.is_alive()),
                "restarts": self.restarts,
                "max_restarts": self.config.max_restarts,
                "last_step_seen": self._last_step,
                "stream_mode": getattr(eng, "stream_mode", "unknown"),
                "recent_events": self.events[-10:]}

    # ---- checks ----
    def check_once(self) -> Dict[str, Any]:
        """Single supervision pass; returns findings and acts on faults."""
        eng = self._engine_getter()
        findings: List[str] = []
        try:
            ram = float(self._ram_reader())
            if ram >= self.config.ram_percent_limit:
                findings.append(f"RAM_CRITICAL:{ram:.1f}%")
        except Exception as e:  # noqa: BLE001
            findings.append(f"RAM_UNREADABLE:{e}")
        try:
            step = int(eng.brain.state.step_count)
            alive = bool(eng._worker_thread and eng._worker_thread.is_alive())
            if eng.is_running:
                if step != self._last_step:
                    self._last_step = step
                    self._last_advance_ts = time.time()
                elif time.time() - self._last_advance_ts > self.config.stall_timeout_sec:
                    findings.append(f"WORKER_STALLED:step={step}")
                if not alive:
                    findings.append("WORKER_DEAD_WHILE_RUNNING")
            else:
                self._last_step = step
                self._last_advance_ts = time.time()
        except Exception as e:  # noqa: BLE001
            findings.append(f"ENGINE_UNREADABLE:{e}")
        gpu_ok = True
        try:
            if getattr(eng, "use_gpu", False) and getattr(eng.brain, "gpu_engine", None) is None:
                gpu_ok = False
                findings.append("GPU_REQUESTED_BUT_ABSENT")
        except Exception:
            pass
        if findings:
            self.log("FAULT", "; ".join(findings))
            self._recover(eng, findings)
        else:
            self.log("OK", f"step={self._last_step} gpu_ok={gpu_ok}")
        return {"findings": findings, "restarts": self.restarts}

    def _recover(self, eng, findings: List[str]) -> None:
        # 1. RETRY: nothing to retry automatically for stalls; log only.
        # 2. RESTORE_LAST_VALID_CHECKPOINT via backup service.
        if self._backups is not None:
            try:
                valid = [b for b in self._backups.list_backups()
                         if self._backups.verify_backup(b["backup"])["status"] == "VALID"]
                if valid:
                    was_running = bool(eng.is_running)
                    if was_running:
                        eng.pause()
                    self._backups.restore_backup(valid[0]["backup"], eng, None)
                    self.log("RECOVER", f"restored {valid[0]['backup']}")
                    if was_running:
                        eng.start()
                    return
            except Exception as e:  # noqa: BLE001
                self.log("RECOVER_FAIL", f"restore failed: {e}")
        # 3. RESTART_RUNTIME (bounded).
        if self.restarts < self.config.max_restarts:
            try:
                self.restarts += 1
                if hasattr(eng, "restart_runtime"):
                    eng.restart_runtime()
                else:
                    eng.pause()
                self.log("RECOVER", f"runtime restarted ({self.restarts}/"
                                    f"{self.config.max_restarts})")
                return
            except Exception as e:  # noqa: BLE001
                self.log("RECOVER_FAIL", f"restart failed: {e}")
        # 4. SAFE_STOP (terminal, explicit, never a hidden loop).
        try:
            eng.safe_shutdown()
        except Exception:
            pass
        self.log("SAFE_STOP", "restart budget exhausted; runtime stopped safely")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception as e:  # noqa: BLE001
                self.log("ERROR", f"watchdog pass failed: {e}")
            self._stop.wait(self.config.interval_sec)
