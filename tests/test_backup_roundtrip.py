"""V5 phase_12/25: real backup round trip on a clean runtime.

RUN -> CREATE -> HASH -> VERIFY -> TAMPER-DETECT -> RESTORE CLEAN RUNTIME ->
REPLAY -> COMPARE STATE HASH. Corrupt backups must be rejected, never restored.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import tempfile
import unittest
import numpy as np

from src.brain.simulation_engine import SimulationEngine
from src.connectome.types import GraphMode
from src.backup.service import BackupService, BACKUP_SCHEMA_VERSION


def _engine(size=64, seed=77):
    return SimulationEngine(circuit_size=size, graph_mode=GraphMode.SYNTHETIC_TEST,
                            use_gpu=False, seed=seed,
                            db_path=os.path.join(tempfile.gettempdir(),
                                                 f"bk_test_{seed}.db"))


class TestBackupRoundTrip(unittest.TestCase):
    def test_create_verify_restore_replay(self):
        with tempfile.TemporaryDirectory() as td:
            svc = BackupService(root=os.path.join(td, "backups"))
            eng = _engine()
            rng = np.random.RandomState(77)
            for _ in range(8):
                eng.step_single(sensory_inputs={"visual": rng.uniform(0.1, 0.5, 16).astype(np.float32)}, reward=0.3)
            man = svc.create_backup(eng, label="test", trigger="manual")
            self.assertEqual(man["schema_version"], BACKUP_SCHEMA_VERSION)
            verdict = svc.verify_backup(man["backup_name"])
            self.assertEqual(verdict["status"], "VALID")
            # tamper detection: corrupt the snapshot, verify must fail
            snap = os.path.join(td, "backups", man["backup_name"], "brain_snapshot.npz")
            with open(snap, "r+b") as f:
                f.seek(100)
                f.write(b"\x00\x01\x02\x03")
            bad = svc.verify_backup(man["backup_name"])
            self.assertEqual(bad["status"], "CORRUPT")
            with self.assertRaises(ValueError):
                svc.restore_backup(man["backup_name"], eng)
            eng.cleanup()

    def test_restore_clean_runtime_matches(self):
        with tempfile.TemporaryDirectory() as td:
            svc = BackupService(root=os.path.join(td, "backups"))
            eng = _engine(seed=78)
            rng = np.random.RandomState(78)
            for _ in range(6):
                eng.step_single(sensory_inputs={"visual": rng.uniform(0.1, 0.5, 16).astype(np.float32)}, reward=0.2)
            man = svc.create_backup(eng, label="rt", trigger="manual")
            # phase-B inputs: draws 6..11 for BOTH branches (fresh stream each)
            def phase_b_inputs():
                r = np.random.RandomState(78)
                for _ in range(6):
                    r.uniform(0.1, 0.5, 16)
                return [r.uniform(0.1, 0.5, 16).astype(np.float32) for _ in range(6)]
            inputs_b = phase_b_inputs()
            # clean runtime, restore, then BOTH branches replay phase B once
            eng2 = _engine(seed=78)
            res = svc.restore_backup(man["backup_name"], eng2)
            self.assertEqual(res["status"], "RESTORED")
            self.assertEqual(res["state_hash"], man["state_hash"])
            for stim in inputs_b:
                eng.step_single(sensory_inputs={"visual": stim}, reward=-0.1)
                eng2.step_single(sensory_inputs={"visual": stim}, reward=-0.1)
            h1 = eng.brain.state.membrane_potentials.tobytes()
            h2 = eng2.brain.state.membrane_potentials.tobytes()
            self.assertEqual(h1, h2)
            eng.cleanup()
            eng2.cleanup()

    def test_live_restore_refused_and_paths_validated(self):
        with tempfile.TemporaryDirectory() as td:
            svc = BackupService(root=os.path.join(td, "backups"))
            eng = _engine(seed=79)
            eng.step_single(n_steps=2)
            man = svc.create_backup(eng, label="x", trigger="manual")
            eng.start()
            try:
                with self.assertRaises(RuntimeError):
                    svc.restore_backup(man["backup_name"], eng)
            finally:
                eng.pause()
            with self.assertRaises(ValueError):
                svc.verify_backup("../escape")
            with self.assertRaises(ValueError):
                svc.verify_backup("bad;name!")
            eng.cleanup()

    def test_gdrive_blocked_without_auth(self):
        from src.backup import gdrive
        st = gdrive.status()
        self.assertIn(st["state"], ("NOT_CONFIGURED", "AUTH_REQUIRED", "AUTHENTICATED",
                                    "ERROR", "TOKEN_EXPIRED"))
        if st["state"] != "AUTHENTICATED":
            r = gdrive.upload_backup("/nonexistent")
            self.assertEqual(r["status"], "BLOCKED_AUTHENTICATION")
            self.assertTrue(r["consent_step"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
