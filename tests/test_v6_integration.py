"""V6 phase_38: the complete closed loop executes for real.

WORLD->BODY->SENSORS->PERCEPTION->BRAIN->GOALS->ACTION->PHYSICS->WORLD
->REWARD->MEMORY, plus save/restore determinism. No mocks: every record
comes from the running simulation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import tempfile
import unittest

from src.genome.schema import Genome
from src.organism.organism import Organism
from src.connectome.types import GraphMode
from src.world3d.spec import build_world_spec
from src.world3d.world import World3D
from src.world3d.agent import EmbodiedAgent, validate_action
from src.world3d.spatial_memory import SpatialMemory


def _make_agent(world, spatial, oid="org_001", seed=11):
    genome = Genome.founder(seed, legacy=False)
    org = Organism(genome, oid, generation=0,
                   seeds={"organism_seed": seed, "development_seed": seed + 1},
                   graph_mode=GraphMode.SYNTHETIC_TEST, circuit_size=32,
                   autonomy_mode=True)
    return EmbodiedAgent(org, "hero", world, spatial, yaw=0.0)


class TestClosedLoop(unittest.TestCase):
    def test_full_loop_lives_and_remembers(self):
        with tempfile.TemporaryDirectory() as td:
            world = World3D(seed=7, characters=[{"name": "hero", "x": 0.0, "y": 4.0}])
            spatial = SpatialMemory(os.path.join(td, "world.db"))
            agent = _make_agent(world, spatial)
            for _ in range(60):
                rec = agent.tick()
            # loop is alive: moved, needs evolved, clock ran
            moved = (abs(agent.body3d.pos[0]) + abs(agent.body3d.pos[1] - 4.0)) > 0.3
            self.assertTrue(moved, f"agent never moved: {agent.body3d.pos}")
            self.assertGreater(world.tick, 50)
            self.assertGreater(world.clock_sec, 0.0)
            self.assertGreater(agent.body3d.hunger, 0.0)
            # §27 causal record complete
            for key in ("tick", "organism_id", "visual_observation", "neural",
                        "goal", "action", "collision", "position_before",
                        "position_after", "reward", "memory_created"):
                self.assertIn(key, rec)
            self.assertEqual(rec["visual_observation"]["kind"], "GEOMETRIC")
            # memory created for real
            self.assertGreater(len(agent.org.episodes), 50)
            n_frames = spatial.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            self.assertGreater(n_frames, 0)
            self.assertGreater(len(agent.eye.seen_entities), 0)
            spatial.close()

    def test_action_gate_rejects_garbage(self):
        self.assertTrue(validate_action({"type": "MOVE", "heading": 0.5, "speed": 1.0})["ok"])
        self.assertFalse(validate_action({"type": "FLY", "speed": 99})["ok"])
        self.assertFalse(validate_action({"type": "MOVE", "heading": 99, "speed": 1.0})["ok"])
        self.assertFalse(validate_action("rm -rf /")["ok"])

    def test_save_restore_deterministic_continuation(self):
        with tempfile.TemporaryDirectory() as td:
            world = World3D(seed=7, characters=[{"name": "hero", "x": 0.0, "y": 4.0}])
            spatial = SpatialMemory(os.path.join(td, "world.db"))
            agent = _make_agent(world, spatial)
            for _ in range(20):
                agent.tick()
            wsnap, asnap = world.snapshot(), agent.snapshot()
            branch_a = [agent.tick() for _ in range(20)]
            ha = (world.state_hash(), list(agent.body3d.pos),
                  agent.org.brain.state.spikes.copy())
            # restore into the SAME world+agent objects (clean continuation)
            world.restore(wsnap)
            agent.restore(asnap)
            branch_b = [agent.tick() for _ in range(20)]
            hb = (world.state_hash(), list(agent.body3d.pos),
                  agent.org.brain.state.spikes.copy())
            self.assertEqual(ha[0], hb[0])
            self.assertEqual(ha[1], hb[1])
            import numpy as _np
            _np.testing.assert_array_equal(ha[2], hb[2])
            # causal streams identical too
            self.assertEqual([r["action"] for r in branch_a],
                             [r["action"] for r in branch_b])
            spatial.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
