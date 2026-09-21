"""V6 model manager: lifecycle, pressure shedding, honest absence."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from src.models.manager import ModelManager
from src.models.loaders import default_manager
from src.models.registry import scan_local


class TestManager(unittest.TestCase):
    def test_load_unload_cycle(self):
        mm = ModelManager(ram_reader=lambda: {"percent": 10.0, "avail_mb": 8000.0})
        mm.register_loader("TINY", lambda: {"handle": True})
        h = mm.load("TINY")
        self.assertTrue(h["handle"])
        self.assertIn("TINY", mm.status()["loaded"])
        self.assertTrue(mm.unload("TINY"))
        self.assertNotIn("TINY", mm.status()["loaded"])

    def test_unknown_task_raises_not_fake(self):
        mm = ModelManager(ram_reader=lambda: {"percent": 10.0, "avail_mb": 8000.0})
        with self.assertRaises(RuntimeError):
            mm.load("NOPE_MODEL")

    def test_pressure_sheds_low_priority_first(self):
        calls = {"n": 0}

        def ram():
            # pressure only on the first check of an episode; shedding the
            # image model relieves it (freed memory is real in production)
            calls["n"] += 1
            return {"percent": 95.0 if calls["n"] == 1 else 40.0, "avail_mb": 500.0}

        mm = ModelManager(ram_limit_percent=88.0, ram_reader=ram)
        mm.register_loader("TEXT_MODEL", lambda: "text")
        mm.register_loader("IMAGE_MODEL", lambda: "image")
        mm.load("TEXT_MODEL")
        mm.load("IMAGE_MODEL")
        calls["n"] = 0  # fresh pressure episode for the TTS load below
        mm.register_loader("TTS_MODEL", lambda: "tts")
        mm.load("TTS_MODEL")
        loaded = mm.status()["loaded"]
        self.assertIn("TTS_MODEL", loaded)
        self.assertNotIn("IMAGE_MODEL", loaded)
        self.assertIn("TEXT_MODEL", loaded)  # higher priority survives

    def test_world_keeps_running_without_models(self):
        mm = default_manager(ram_reader=lambda: {"percent": 10.0, "avail_mb": 8000.0})
        self.assertEqual(mm.status()["loaded"], {})
        mm.ensure_unloaded_for_world()
        self.assertEqual(mm.status()["loaded"], {})


class TestRegistry(unittest.TestCase):
    def test_scan_reports_measured_facts(self):
        entries = {e["task"]: e for e in scan_local()}
        self.assertIn("TEXT_MODEL", entries)
        text = entries["TEXT_MODEL"]
        if text["present"]:
            self.assertTrue(text["file_size"] and text["file_size"] > 10 ** 6)
            self.assertEqual(len(text["sha256"]), 64)
        # absent models are explicit, never verified
        for e in entries.values():
            if not e["present"]:
                self.assertFalse(e["verified"])
                self.assertIsNone(e["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
