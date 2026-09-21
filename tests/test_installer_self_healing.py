import unittest
import os
import subprocess
import tempfile
import shutil
import json


class TestInstallerSelfHealing(unittest.TestCase):
    def setUp(self):
        self.temp_install_dir = tempfile.mkdtemp(prefix="flybrain_inst_test_")
        self.installer_path = os.path.abspath("installer/install.ps1")

    def tearDown(self):
        if os.path.exists(self.temp_install_dir):
            shutil.rmtree(self.temp_install_dir)

    def test_fresh_install_and_self_healing_repair(self):
        # 1. Execute Fresh Installation
        cmd = [
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", self.installer_path,
            "-TargetDir", self.temp_install_dir,
            "-SkipDoctor"
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, f"Installer failed: {p.stderr}")

        # Verify layout
        for sd in ["app", "runtime", "models", "data", "world", "assets", "cache", "backups", "logs", "diagnostics"]:
            self.assertTrue(os.path.isdir(os.path.join(self.temp_install_dir, sd)))

        # Verify config.json
        cfg_path = os.path.join(self.temp_install_dir, "config.json")
        self.assertTrue(os.path.exists(cfg_path))
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["version"], "10.0.0")

        # Verify launcher
        launch_path = os.path.join(self.temp_install_dir, "launch.bat")
        self.assertTrue(os.path.exists(launch_path))

        # 2. Simulate File Corruption
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("CORRUPTED_JSON_NOT_VALID")
        os.remove(launch_path)
        shutil.rmtree(os.path.join(self.temp_install_dir, "models"))

        self.assertFalse(os.path.exists(launch_path))
        self.assertFalse(os.path.exists(os.path.join(self.temp_install_dir, "models")))

        # 3. Execute Self-Healing Repair
        cmd_repair = [
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", self.installer_path,
            "-TargetDir", self.temp_install_dir,
            "-Repair",
            "-SkipDoctor"
        ]
        p_rep = subprocess.run(cmd_repair, capture_output=True, text=True)
        self.assertEqual(p_rep.returncode, 0, f"Repair failed: {p_rep.stderr}")

        # Verify restored directory
        self.assertTrue(os.path.isdir(os.path.join(self.temp_install_dir, "models")))

        # Verify restored config
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            cfg_rep = json.load(f)
        self.assertEqual(cfg_rep["version"], "10.0.0")

        # Verify restored launcher
        self.assertTrue(os.path.exists(launch_path))


if __name__ == "__main__":
    unittest.main()
