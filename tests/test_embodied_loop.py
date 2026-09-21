import unittest
import os
from src.world.integration import GenerativeLoopExecutor


class TestEmbodiedLoop(unittest.TestCase):
    def test_biological_closed_loop_execution(self):
        executor = GenerativeLoopExecutor()
        res = executor.run_acceptance_loop(
            prompt="Create a small wooden bridge over the nearby stream.",
            seed=42,
            use_biological_loop=True
        )
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["policy_source"], "BIOLOGICAL_CLOSED_LOOP")
        self.assertTrue(res["physics"]["deck_contact_verified"])
        self.assertGreater(res["total_trace_steps"], 50)
        self.assertTrue(len(res["causal_trace_samples"]) > 0)
        sample = res["causal_trace_samples"][0]
        self.assertEqual(sample["policy_source"], "BIOLOGICAL_CLOSED_LOOP")
        self.assertIn("sensory_input", sample)
        self.assertIn("neural_output", sample)
        self.assertIn("motor_output", sample)
        self.assertIn("physical_action", sample)
        self.assertIn("world_result", sample)

    def test_test_driver_only_labelling(self):
        executor = GenerativeLoopExecutor()
        res = executor.run_acceptance_loop(
            prompt="Create a small wooden bridge over the nearby stream.",
            seed=42,
            use_biological_loop=False
        )
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["policy_source"], "TEST_DRIVER_ONLY")
        sample = res["causal_trace_samples"][0]
        self.assertEqual(sample["policy_source"], "TEST_DRIVER_ONLY")


if __name__ == "__main__":
    unittest.main()
