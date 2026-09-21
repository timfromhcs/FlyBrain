import unittest
from unittest.mock import MagicMock, patch
from src.world.planner import WorldPlanner


class TestV10ModelHonesty(unittest.TestCase):
    def test_rule_based_fallback_provenance(self):
        planner = WorldPlanner()
        # Ensure LLM extraction returns None
        with patch.object(planner, "_extract_intent_llm", return_value=(None, {"error_detail": "Simulated offline weights missing"})):
            plan = planner.plan_world("Create a small wooden bridge over the nearby stream.")
            self.assertEqual(plan.inference_provenance["source"], "RULE_BASED_FALLBACK")
            self.assertEqual(plan.inference_provenance["runtime_backend"], "procedural")
            self.assertTrue(plan.inference_provenance["is_deterministic"])
            self.assertEqual(plan.inference_provenance["upstream_error"], "Simulated offline weights missing")
            self.assertGreaterEqual(len(plan.structures), 1)

    def test_local_llm_provenance(self):
        planner = WorldPlanner()
        llm_intent = {
            "biome": "pine_forest",
            "structures": [{"type": "wooden_bridge", "material": "wood", "x": 1.0, "y": 2.0, "z": 0.0}]
        }
        llm_prov = {
            "source": "LOCAL_LLM",
            "model_id": "Qwen/Qwen3-0.6B-GGUF",
            "quantization": "Q8_0",
            "backend": "llama_cpp"
        }
        with patch.object(planner, "_extract_intent_llm", return_value=(llm_intent, llm_prov)):
            plan = planner.plan_world("Build a wooden bridge in the pine woods.")
            self.assertEqual(plan.inference_provenance["source"], "LOCAL_LLM")
            self.assertEqual(plan.inference_provenance["model_id"], "Qwen/Qwen3-0.6B-GGUF")
            self.assertFalse(plan.inference_provenance["is_deterministic"])


if __name__ == "__main__":
    unittest.main()
