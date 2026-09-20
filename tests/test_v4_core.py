"""V4 core tests: ablation enforcement, deterministic research IDs, research
ledger, deep-time EXACT/ACCELERATED modes, doctor, UI endpoints."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.research.ablation import (AblationConfig, ABLATION_KEYS, ABLATION_PRESETS,
                                   run_ablation, verify_ablation_enforcement)
from src.research.ledger import ResearchLedger, code_sha, shader_sha
from src.timeline.deeptime import DeepTimeConfig, DeepTimeRunner
from src.common.determinism import SeedBundle
from src.connectome.types import GraphMode
from src.population.population import Population


def make_pop(seed, size=3, autonomy=True):
    seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                       organism_seed=seed + 2, development_seed=seed + 3,
                       mutation_seed=seed + 4, world_seed=seed + 5,
                       teacher_seed=seed + 6)
    return Population(size, seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=seed,
                      autonomy_mode=autonomy, genome_version="2.0" if autonomy else "1.0")


class TestAblations(unittest.TestCase):
    def test_preset_validation(self):
        for name in ABLATION_PRESETS:
            AblationConfig.from_preset(name).to_dict()
        with self.assertRaises(ValueError):
            AblationConfig.from_preset("nope")
        with self.assertRaises(ValueError):
            AblationConfig({"growth": "yes"})

    def test_no_growth_never_grows(self):
        r = run_ablation(AblationConfig.from_preset("no_growth"), seed=301, ticks=25)
        self.assertTrue(r["enforced"]["no_growth_upheld"])
        self.assertFalse(r["measured"]["grew"])
        self.assertTrue(verify_ablation_enforcement(r))

    def test_no_plasticity_never_changes_weights(self):
        r = run_ablation(AblationConfig.from_preset("no_plasticity"), seed=302, ticks=20)
        self.assertTrue(r["enforced"]["no_plasticity_upheld"])
        self.assertFalse(r["measured"]["weights_changed"])

    def test_full_ablation_allows_growth_and_learning(self):
        r = run_ablation(AblationConfig.from_preset("full"), seed=303, ticks=25)
        # with everything enabled, no enforcement claims exist
        self.assertTrue(all(v is None for v in r["enforced"].values()))

    def test_no_teaching_yields_zero_sessions(self):
        r = run_ablation(AblationConfig.from_preset("no_teaching"), seed=304, ticks=20)
        self.assertTrue(r["enforced"]["no_teaching_upheld"])
        self.assertEqual(r["measured"]["teaching_sessions"], 0)


class TestDeterministicResearchIds(unittest.TestCase):
    def _runtime(self):
        from src.llm.control import ControlPlane, ResearchRuntime, CommandEnvelope
        return ControlPlane(ResearchRuntime(experiment_seed=77))

    def test_proposal_ids_are_deterministic_content_ids(self):
        env_factory = lambda: __import__("src.llm.control", fromlist=["CommandEnvelope"]) \
            .CommandEnvelope
        CE = env_factory()
        cp1, cp2 = self._runtime(), self._runtime()
        r1 = cp1.execute(CE("PROPOSE_HYPOTHESIS",
                            {"text": "growth follows energy", "based_on_experiments": []}))
        r2 = cp2.execute(CE("PROPOSE_HYPOTHESIS",
                            {"text": "growth follows energy", "based_on_experiments": []}))
        self.assertEqual(r1["hypothesis"]["hypothesis_id"],
                         r2["hypothesis"]["hypothesis_id"],
                         "same content in a fresh runtime -> same id (no wall-clock)")
        r3 = cp1.execute(CE("PROPOSE_HYPOTHESIS",
                            {"text": "different text", "based_on_experiments": []}))
        self.assertNotEqual(r1["hypothesis"]["hypothesis_id"],
                            r3["hypothesis"]["hypothesis_id"])
        # sequence sensitivity: same content twice in ONE runtime gets new id
        r4 = cp1.execute(CE("PROPOSE_HYPOTHESIS",
                            {"text": "growth follows energy", "based_on_experiments": []}))
        self.assertNotEqual(r1["hypothesis"]["hypothesis_id"],
                            r4["hypothesis"]["hypothesis_id"])

    def test_task_and_curriculum_ids_deterministic(self):
        CE = __import__("src.llm.control", fromlist=["CommandEnvelope"]).CommandEnvelope
        cp1, cp2 = self._runtime(), self._runtime()
        a = cp1.execute(CE("PROPOSE_TASK", {"description": "find food",
                                            "success_criterion": "energy>1"}))
        b = cp2.execute(CE("PROPOSE_TASK", {"description": "find food",
                                            "success_criterion": "energy>1"}))
        self.assertEqual(a["task"]["task_id"], b["task"]["task_id"])
        c1 = cp1.execute(CE("PROPOSE_CURRICULUM", {"stages": ["a", "b"]}))
        c2 = cp2.execute(CE("PROPOSE_CURRICULUM", {"stages": ["a", "b"]}))
        self.assertEqual(c1["curriculum"]["curriculum_id"], c2["curriculum"]["curriculum_id"])


class TestResearchLedger(unittest.TestCase):
    def test_chained_records_with_full_provenance(self):
        led = ResearchLedger("exp-ledger-test", seed=9)
        led.set_dataset_sha("abc123")
        r1 = led.append("experiment", tick=0, generation=0,
                        payload={"type": "baseline"}, world_sha="w1")
        r2 = led.append("milestone", tick=10, generation=1,
                        payload={"milestone": "TEST"}, genome_sha="g1",
                        parent_event=r1["event_id"])
        self.assertTrue(led.verify_chain())
        self.assertEqual(r2["parent_event"], r1["event_id"])
        self.assertEqual(r1["prev_hash"], "0" * 64)
        self.assertNotEqual(led.event_stream_hash(), "")
        self.assertEqual(led._static["shader_sha"], shader_sha())

    def test_unknown_event_type_rejected(self):
        led = ResearchLedger("exp-x", seed=1)
        with self.assertRaises(ValueError):
            led.append("magic_event", 0, 0, {})

    def test_deterministic_record_ids(self):
        l1 = ResearchLedger("exp-d", seed=5)
        l2 = ResearchLedger("exp-d", seed=5)
        a = l1.append("birth", 3, 0, {"oid": "x"})
        b = l2.append("birth", 3, 0, {"oid": "x"})
        self.assertEqual(a["event_id"], b["event_id"])
        self.assertEqual(a["record_hash"], b["record_hash"])


class TestDeepTimeExact(unittest.TestCase):
    def test_exact_mode_is_tick_by_tick(self):
        pop = make_pop(311)
        runner = DeepTimeRunner(pop, DeepTimeConfig(mode="EXACT",
                                                    coarse_ticks_per_step=20))
        s = runner.fast_forward(10)
        self.assertEqual(s["mode"], "EXACT")
        self.assertEqual(s["resolution"], "full")
        self.assertIsNone(s["approximation_model"])
        self.assertEqual(s["sim_ticks"], 10)

    def test_accelerated_never_claims_exact(self):
        pop = make_pop(312)
        runner = DeepTimeRunner(pop, DeepTimeConfig(mode="ACCELERATED"))
        s = runner.fast_forward(2)
        self.assertEqual(s["mode"], "ACCELERATED")
        self.assertEqual(s["resolution"], "coarse")
        self.assertIsNotNone(s["approximation_model"])
        self.assertGreater(s["sim_ticks"], 2)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            DeepTimeConfig(mode="WARP").validate()

    def test_exact_replay_hash_verified(self):
        pop = make_pop(313)
        runner = DeepTimeRunner(pop, DeepTimeConfig(mode="EXACT"))
        runner.fast_forward(5)
        runner.escalate("exact-ck")
        rep = runner.replay_high_resolution("exact-ck", ticks=3)
        self.assertTrue(rep["checkpoint_hash_verified"])


class TestDoctor(unittest.TestCase):
    def test_doctor_reports_real_checks(self):
        from src.diagnostics.doctor import run_doctor
        from src.version import VERSION
        rep = run_doctor()
        self.assertIn(rep["overall"], ("READY", "READY_DEGRADED", "DEGRADED"))
        self.assertEqual(rep["flybrain_version"], VERSION)
        names = [c["name"] for c in rep["checks"]]
        for required in ("os", "python", "dataset", "gpu", "storage", "core_runtime"):
            self.assertIn(required, names)
        self.assertEqual(rep["summary"]["errors"], 0)
        for c in rep["checks"]:
            self.assertTrue(c["detail"], f"check {c['name']} must explain itself")

    def test_doctor_core_runtime_actually_executes_lif(self):
        from src.diagnostics.doctor import run_doctor
        rep = run_doctor()
        core = next(c for c in rep["checks"] if c["name"] == "core_runtime")
        self.assertEqual(core["status"], "OK")
        self.assertIn("LIF step executed", core["detail"])


class TestUIV4Endpoints(unittest.TestCase):
    def test_version_and_doctor_endpoints(self):
        from fastapi.testclient import TestClient
        from src.ui.server import app
        from src.version import VERSION
        client = TestClient(app)
        r = client.get("/api/version")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["version"], VERSION)
        r2 = client.get("/api/doctor")
        self.assertEqual(r2.status_code, 200)
        self.assertIn(r2.json()["overall"], ("READY", "READY_DEGRADED", "DEGRADED"))
        r3 = client.get("/api/health")
        self.assertEqual(r3.json()["version"], VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
