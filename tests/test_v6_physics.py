"""V6 phase_05: real MuJoCo physics - gravity, contact, blocking, pushing,
save/restore. Every claim executes the solver; nothing is hand-computed."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

import numpy as np

from src.world3d.spec import build_world_spec
from src.world3d.physics import PhysicsWorld, CHARACTER_R

try:
    import mujoco  # noqa
    HAS_MUJOCO = True
except Exception:
    HAS_MUJOCO = False


def _world(seed=7, chars=None):
    if not HAS_MUJOCO:
        raise unittest.SkipTest("mujoco unavailable")
    return PhysicsWorld(build_world_spec(seed), characters=chars)


class TestGravityAndFloor(unittest.TestCase):
    def test_drop_settles_on_floor(self):
        w = _world(chars=[{"name": "hero", "x": 0.0, "y": 4.0}])
        w.teleport("hero", 0.0, 4.0, 5.0)
        z0 = w.char_state("hero")["pos"][2]
        for _ in range(300):
            w.step()
        s = w.char_state("hero")
        self.assertLess(s["pos"][2], z0)          # gravity pulled down
        self.assertGreater(s["pos"][2], 0.5)      # but floor holds
        self.assertLess(s["pos"][2], 1.6)
        self.assertTrue(w.grounded("hero"))       # real contact with ground
        z1 = s["pos"][2]
        for _ in range(200):
            w.step()
        z2 = w.char_state("hero")["pos"][2]
        self.assertAlmostEqual(z1, z2, places=3)  # stable rest, no sinking

    def test_no_teleport_movement_bound(self):
        w = _world(chars=[{"name": "hero", "x": 0.0, "y": 0.0}])
        w.teleport("hero", 0.0, 0.0, 1.2)
        for _ in range(100):
            w.step()
        p0 = np.array(w.char_state("hero")["pos"])
        for _ in range(50):
            w.drive_character("hero", 1.0, 0.0)
            w.step()
            p1 = np.array(w.char_state("hero")["pos"])
            step_dist = float(np.linalg.norm((p1 - p0)[:2]))
            # velocity servo at 1 m/s * dt 0.01 + solver slack: hard bound
            self.assertLess(step_dist, 0.05)
            p0 = p1


class TestWallBlocking(unittest.TestCase):
    def test_wall_blocks_character(self):
        # home north wall at cy=13 (center y=10 + d/2=3); drive north into it
        w = _world(chars=[{"name": "hero", "x": 0.0, "y": 10.0}])
        w.teleport("hero", 0.0, 10.0, 1.2)
        for _ in range(100):
            w.step()
        for _ in range(400):
            w.drive_character("hero", 0.0, 2.0)
            w.step()
        s = w.char_state("hero")
        # wall inner face at y=13-0.15=12.85; capsule radius 0.3 -> rest ~12.5
        self.assertLess(s["pos"][1], 12.7)
        self.assertGreater(s["pos"][1], 11.0)
        names = {(c["geom1"], c["geom2"]) for c in w.contacts()}
        self.assertTrue(any("wall_north" in pair for pair in names),
                        f"expected wall contact, got {w.contacts()[:4]}")


class TestPushableObject(unittest.TestCase):
    def test_character_pushes_crate(self):
        # crate_1 sits INSIDE the house at (1.8, 9.0): start inside, shove north
        w = _world(chars=[{"name": "hero", "x": 1.8, "y": 7.8}])
        w.teleport("hero", 1.8, 7.8, 1.2)
        for _ in range(100):
            w.step()
        q0 = w.data.qpos.copy()
        # find crate body qpos
        bid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "bod_crate_1")
        jnt = int(w.model.body_jntadr[bid])
        qa = int(w.model.jnt_qposadr[jnt])
        crate0 = w.data.qpos[qa:qa + 2].copy()
        for _ in range(400):
            w.drive_character("hero", 0.0, 1.5)  # drive north into crate at y=9
            w.step()
        crate1 = w.data.qpos[qa:qa + 2].copy()
        moved = float(np.linalg.norm(crate1 - crate0))
        self.assertGreater(moved, 0.2)  # real impulse transfer, not scripted


class TestPosture(unittest.TestCase):
    def test_push_then_release_recovers_upright(self):
        w = _world(chars=[{"name": "hero", "x": 1.8, "y": 7.8}])
        w.teleport("hero", 1.8, 7.8, 1.2)
        for _ in range(100):
            w.step()
        for _ in range(150):  # shove the crate (may tip the hero: real fall)
            w.drive_character("hero", 0.0, 1.5)
            w.step()
        for _ in range(300):  # release: get-up maneuver must right the body
            w.drive_character("hero", 0.0, 0.0)
            w.step()
        self.assertGreater(w.upright("hero"), 0.7)


class TestSaveRestore(unittest.TestCase):
    def test_snapshot_resume_identical(self):
        w = _world(chars=[{"name": "hero", "x": 0.0, "y": 0.0}])
        w.teleport("hero", 0.0, 0.0, 1.2)
        for _ in range(50):
            w.drive_character("hero", 0.8, 0.3)
            w.step()
        snap = w.snapshot()
        traj_a = []
        for _ in range(50):
            w.drive_character("hero", 0.8, 0.3)
            w.step()
            traj_a.append(w.char_state("hero")["pos"])
        w.restore(snap)
        traj_b = []
        for _ in range(50):
            w.drive_character("hero", 0.8, 0.3)
            w.step()
            traj_b.append(w.char_state("hero")["pos"])
        np.testing.assert_allclose(np.array(traj_a), np.array(traj_b), atol=1e-9)

    def test_wrong_spec_rejected(self):
        w = _world(seed=7, chars=[{"name": "hero", "x": 0.0, "y": 0.0}])
        snap = w.snapshot()
        snap["spec_hash"] = "tampered"
        with self.assertRaises(ValueError):
            w.restore(snap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
