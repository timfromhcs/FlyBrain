"""v4.1 Phase 2: versioned checkpoint envelope preserves state + research
history; save -> terminate -> restore -> continue == uninterrupted run."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from src.common.determinism import SeedBundle
from src.connectome.types import GraphMode
from src.population.population import Population
from src.population.checkpoint import (save_checkpoint, restore_checkpoint,
                                        research_hash_of,
                                        CHECKPOINT_SCHEMA_VERSION)


def _seeds(s):
    return SeedBundle(experiment_seed=s, generation_seed=s + 1, organism_seed=s + 2,
                      development_seed=s + 3, mutation_seed=s + 4, world_seed=s + 5,
                      teacher_seed=s + 6)


class TestCheckpointEnvelope(unittest.TestCase):
    def test_save_restore_continue_matches_uninterrupted(self):
        pop = Population(4, _seeds(21), GraphMode.SYNTHETIC_TEST, 32, experiment_seed=21)
        pop.step(20)
        env = save_checkpoint(pop)
        self.assertEqual(env["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertIn("state_hash", env)
        self.assertIn("research_hash", env)
        self.assertIn("combined_hash", env)
        self.assertIn("rng_bundle", env)
        self.assertIn("experiment_identity", env)
        self.assertTrue(env["event_history"])
        # uninterrupted continuation
        pop.step(20)
        ref_state, ref_research = pop.population_hash(), research_hash_of(pop)
        # terminate; restore from envelope; continue identically
        pop2 = restore_checkpoint(json_roundtrip(env))
        pop2.step(20)
        self.assertEqual(pop2.population_hash(), ref_state)
        self.assertEqual(research_hash_of(pop2), ref_research)
        self.assertEqual(len(pop2.events.events), len(pop.events.events))

    def test_hash_mismatch_raises(self):
        pop = Population(4, _seeds(22), GraphMode.SYNTHETIC_TEST, 32, experiment_seed=22)
        pop.step(5)
        env = save_checkpoint(pop)
        env["state_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            restore_checkpoint(env)

    def test_research_hash_covers_lineage_and_teaching(self):
        pop = Population(4, _seeds(23), GraphMode.SYNTHETIC_TEST, 32, experiment_seed=23)
        h0 = research_hash_of(pop)
        pop.step(30)
        self.assertNotEqual(research_hash_of(pop), h0)


def json_roundtrip(env):
    import json
    return json.loads(json.dumps(env, default=str))


if __name__ == "__main__":
    unittest.main(verbosity=2)
