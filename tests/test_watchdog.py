"""V5 phase_11: watchdog observes, detects stalls, recovers boundedly."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import tempfile
import time
import unittest

from src.brain.simulation_engine import SimulationEngine
from src.connectome.types import GraphMode
from src.backup.service import BackupService
from src.runtime.watchdog import Watchdog, WatchdogConfig


def _engine(seed=91):
    return SimulationEngine(circuit_size=32, graph_mode=GraphMode.SYNTHETIC_TEST,
                            use_gpu=False, seed=seed,
                            db_path=os.path.join(tempfile.gettempdir(),
                                                 f"wd_test_{seed}.db"))


def _wd(engine_getter, backups=None, ram=42.0, **cfg):
    """Watchdog with deterministic RAM input (production reader is psutil)."""
    return Watchdog(engine_getter, backups, WatchdogConfig(**cfg),
                    ram_reader=lambda: ram)


class FakeStalledEngine:
    """Deterministic fault fixture: GPU requested but absent on every pass,
    so the recovery chain is exercised without timing flakes."""
    is_running = False
    use_gpu = True
    stream_mode = "PAUSED"
    _worker_thread = None
    rebuilds = 0

    class _Brain:
        class _State:
            step_count = 5

        state = _State()
        gpu_engine = None

    brain = _Brain()

    def pause(self):
        self.is_running = False

    def start(self):
        self.is_running = True

    def restart_runtime(self):
        self.rebuilds += 1

    def safe_shutdown(self):
        self.is_running = False
        self.stream_mode = "STOPPED"


class TestWatchdog(unittest.TestCase):
    def test_healthy_engine_ok(self):
        eng = _engine()
        eng.step_single(n_steps=2)
        wd = _wd(lambda: eng, None, stall_timeout_sec=60.0)
        rep = wd.check_once()
        self.assertEqual(rep["findings"], [])
        eng.cleanup()

    def test_ram_critical_fires_at_limit(self):
        eng = _engine(seed=94)
        wd = _wd(lambda: eng, None, ram=95.0)
        rep = wd.check_once()
        self.assertTrue(any("RAM_CRITICAL" in f for f in rep["findings"]))
        eng.cleanup()

    def test_stall_detected_and_bounded_recovery(self):
        eng = FakeStalledEngine()
        wd = _wd(lambda: eng, None, stall_timeout_sec=0.0, max_restarts=2)
        rep = wd.check_once()
        self.assertIn("GPU_REQUESTED_BUT_ABSENT", rep["findings"])
        # restart rung (no backups configured)
        self.assertEqual(wd.restarts, 1)
        self.assertEqual(eng.rebuilds, 1)
        wd.check_once()
        self.assertEqual(wd.restarts, 2)
        # budget exhausted -> safe stop, never a hidden loop
        wd.check_once()
        kinds = [e["kind"] for e in wd.events]
        self.assertIn("SAFE_STOP", kinds)
        self.assertEqual(wd.restarts, 2)
        self.assertEqual(eng.rebuilds, 2)
        self.assertEqual(eng.stream_mode, "STOPPED")

    def test_running_but_frozen_step_reports_stall(self):
        eng = FakeStalledEngine()
        eng.is_running = True
        eng.use_gpu = False  # isolate the stall fault (no GPU complaint)
        eng._worker_thread = type("T", (), {"is_alive": lambda self: True})()
        wd = _wd(lambda: eng, None, stall_timeout_sec=0.0, max_restarts=0)
        wd.check_once()  # baseline: first sighting of step 5
        time.sleep(0.02)
        rep = wd.check_once()  # step still 5 -> stalled
        self.assertTrue(any("STALLED" in f for f in rep["findings"]))
        self.assertIn("SAFE_STOP", [e["kind"] for e in wd.events])

    def test_restore_preferred_when_valid_backup_exists(self):
        with tempfile.TemporaryDirectory() as td:
            svc = BackupService(root=os.path.join(td, "backups"))
            eng = _engine(seed=92)
            eng.step_single(n_steps=4)
            man = svc.create_backup(eng, label="pre", trigger="manual")
            wd = _wd(lambda: eng, svc, stall_timeout_sec=1000.0)
            # force a fault path directly: restore rung
            wd._recover(eng, ["SYNTHETIC_FAULT"])
            kinds = [e["kind"] for e in wd.events]
            self.assertIn("RECOVER", kinds)
            self.assertEqual(wd.restarts, 0)  # restore preferred over restart
            eng.cleanup()

    def test_status_surface(self):
        eng = _engine(seed=93)
        wd = _wd(lambda: eng, None)
        st = wd.status()
        self.assertIn("recent_events", st)
        self.assertEqual(st["restarts"], 0)
        eng.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
