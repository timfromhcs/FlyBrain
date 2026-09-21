"""V6 mind: real LLM dreams, imagination w/ generated images, social
emergence, speech loop. Heavy models load once per class (setUpClass)."""
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
from src.world3d.agent import EmbodiedAgent
from src.world3d.spatial_memory import SpatialMemory
from src.world3d.dreaming import dream_cycle, replay_dream, imagine
from src.world3d.social import social_encounter
from src.models.loaders import load_text, load_image


def _agent(world, spatial, oid, seed, x=0.0, y=4.0):
    org = Organism(Genome.founder(seed, legacy=False), oid, generation=0,
                   seeds={"organism_seed": seed, "development_seed": seed + 1},
                   graph_mode=GraphMode.SYNTHETIC_TEST, circuit_size=32,
                   autonomy_mode=True)
    return EmbodiedAgent(org, oid, world, spatial)


@unittest.skipUnless(os.path.exists("models/text_model"), "no local LLM")
class TestDreamLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = load_text()

    def test_dream_narrative_only(self):
        with tempfile.TemporaryDirectory() as td:
            spatial = SpatialMemory(os.path.join(td, "w.db"))
            try:
                world = World3D(seed=7, characters=[{"name": "d1", "x": 0.0, "y": 4.0}])
                ag = _agent(world, spatial, "d1", 21)
                for _ in range(10):
                    ag.tick()
                rec = dream_cycle(ag, self.text, image_model=None)
                self.assertTrue(rec["narrative"])
                self.assertIn("NARRATIVE_ONLY", rec["provenance"])
                self.assertIn("DREAM/COUNTERFACTUAL", rec["provenance"])
                back = replay_dream(ag, rec["id"])
                self.assertEqual(back["replays"], 1)
            finally:
                spatial.close()

    def test_dream_with_generated_image(self):
        with tempfile.TemporaryDirectory() as td:
            spatial = SpatialMemory(os.path.join(td, "w.db"))
            try:
                world = World3D(seed=7, characters=[{"name": "d2", "x": 0.0, "y": 4.0}])
                ag = _agent(world, spatial, "d2", 22)
                for _ in range(5):
                    ag.tick()
                from src.models.image_adapter import LocalImageModel
                img = LocalImageModel()
                try:
                    rec = dream_cycle(ag, self.text, image_model=img)
                finally:
                    img.unload()
                self.assertIn("GENERATED_DREAM", rec["provenance"])
                self.assertTrue(os.path.exists(rec["image_path"]))
                from PIL import Image
                im = Image.open(rec["image_path"])
                im.load()
                self.assertEqual(list(im.size), [512, 512])
            finally:
                spatial.close()


@unittest.skipUnless(os.path.exists("models/text_model"), "no local LLM")
class TestImagination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = load_text()

    def test_imagine_forest_night(self):
        with tempfile.TemporaryDirectory() as td:
            spatial = SpatialMemory(os.path.join(td, "w.db"))
            try:
                world = World3D(seed=7, characters=[{"name": "i1", "x": 0.0, "y": 4.0}])
                ag = _agent(world, spatial, "i1", 23)
                for _ in range(5):
                    ag.tick()
                from src.models.image_adapter import LocalImageModel
                rec = imagine(ag, "the forest at night", self.text, LocalImageModel())
                self.assertEqual(rec["provenance"], "GENERATED_IMAGE")
                self.assertTrue(os.path.exists(rec["image_path"]))
            finally:
                spatial.close()


class TestSocialEmergence(unittest.TestCase):
    def test_trust_emerges_from_teaching(self):
        with tempfile.TemporaryDirectory() as td:
            world = World3D(seed=7, characters=[{"name": "s1", "x": 0.0, "y": 4.0},
                                                {"name": "s2", "x": 0.5, "y": 4.0}])
            spatial = SpatialMemory(os.path.join(td, "w.db"))
            a = _agent(world, spatial, "s1", 31)
            b = _agent(world, spatial, "s2", 32)
            a.body3d.pos = [0.0, 4.0, 0.75]
            b.body3d.pos = [0.5, 4.0, 0.75]
            # skilled teacher vs novice student: measurable positive transfer
            a.org.skills["forage"] = 0.9
            b.org.skills["forage"] = 0.1
            t_before = b.org.social_mem.trust_of(a.org.id)
            res = social_encounter(a, b)
            self.assertEqual(res["status"], "INTERACTED")
            t_after = b.org.social_mem.trust_of(a.org.id)
            self.assertGreater(t_after, t_before)
            self.assertGreater(res["learning_gain"], 0.0)
            self.assertTrue(any(e.get("action") == "SOCIAL_INTERACT" for e in b.org.episodes))
            spatial.close()

    def test_distant_agents_do_not_interact(self):
        with tempfile.TemporaryDirectory() as td:
            world = World3D(seed=7, characters=[{"name": "f1", "x": -10.0, "y": -10.0},
                                                {"name": "f2", "x": 10.0, "y": 10.0}])
            spatial = SpatialMemory(os.path.join(td, "w.db"))
            a = _agent(world, spatial, "f1", 33)
            b = _agent(world, spatial, "f2", 34)
            a.body3d.pos = [-10.0, -10.0, 0.75]
            b.body3d.pos = [10.0, 10.0, 0.75]
            res = social_encounter(a, b)
            self.assertEqual(res["status"], "TOO_FAR")
            spatial.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
