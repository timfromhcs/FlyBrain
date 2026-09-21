"""V6 phase_28/29: offline hard gate. With FLYBRAIN_OFFLINE=1 and ALL
sockets blocked, the world must boot, simulate, perceive, remember,
backup and restore. Model acquisition must refuse. No hidden network."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
os.environ["FLYBRAIN_OFFLINE"] = "1"
import socket as _socket
import tempfile
import unittest

_REAL_CONNECT = _socket.socket.connect


def _blocked_connect(self, *a, **k):
    raise RuntimeError("OFFLINE_VIOLATION: outbound connect() blocked in offline test")


class TestOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _socket.socket.connect = _blocked_connect

    @classmethod
    def tearDownClass(cls):
        _socket.socket.connect = _REAL_CONNECT

    def test_acquisition_refuses_offline(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "scripts/acquire_models.py", "embedding"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "FLYBRAIN_OFFLINE": "1"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("OFFLINE", (r.stdout + r.stderr))

    def test_world_runs_fully_offline(self):
        from src.world3d.service import WorldService
        from src.backup.service import BackupService
        with tempfile.TemporaryDirectory() as td:
            svc = WorldService(seed=7, friends=1,
                               db_path=os.path.join(td, "w.db"))
            try:
                recs = svc.step(10)
                self.assertEqual(len(recs), 20)  # 10 ticks x 2 agents
                st = svc.state()
                self.assertIn("state_hash", st)
                hero = svc.agents["hero"]
                self.assertGreater(len(hero.org.episodes), 5)
                self.assertGreater(len(hero.eye.seen_entities), 0)
                bs = BackupService(root=os.path.join(td, "backups"))
                man = bs.create_world_backup(svc, label="offline", trigger="manual")
                self.assertEqual(bs.verify_backup(man["backup_name"])["status"], "VALID")
                h0 = svc.world.state_hash()
                svc.step(5)
                self.assertNotEqual(svc.world.state_hash(), h0)
                res = bs.restore_world_backup(man["backup_name"], svc)
                self.assertEqual(res["status"], "RESTORED")
                self.assertEqual(svc.world.state_hash(), h0)
            finally:
                svc.spatial.close()

    def test_local_models_load_offline(self):
        # present models must load from disk without network
        from src.models.loaders import load_text, load_embedding
        from src.models.manager import ModelManager
        mm = ModelManager()
        mm.register_loader("TEXT_MODEL", load_text)
        mm.register_loader("EMBEDDING_MODEL", load_embedding)
        if os.path.exists("models/text_model"):
            llm = mm.load("TEXT_MODEL")
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": "Say OK."}],
                max_tokens=8, temperature=0.0, seed=1)
            self.assertTrue(out["choices"][0]["message"]["content"])
            mm.unload("TEXT_MODEL")
        if os.path.exists("models/embedding"):
            emb = mm.load("EMBEDDING_MODEL")
            vec = emb.encode(["offline test"], convert_to_numpy=True)
            self.assertEqual(vec.shape[1], 384)
            mm.unload("EMBEDDING_MODEL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
