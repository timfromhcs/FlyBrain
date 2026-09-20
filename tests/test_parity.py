"""CPU/Vulkan parity with explicit metrics (P9): tolerance-based, never 'bit-exact'.

Reports max_abs_error, max_relative_error, spike_mismatch_count over a grid of
seeds x sizes x input patterns x refractory states, plus a long trajectory.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.compute.cpu_reference import cpu_lif_step
from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode


def parity_metrics(cp, gp, cs, gs):
    d = np.abs(cp - gp)
    denom = np.abs(cp) + 1e-7
    return {
        "max_abs_error": float(np.max(d)),
        "max_relative_error": float(np.max(d / denom)),
        "spike_mismatch_count": int(np.sum(cs != gs)),
    }


class TestParityMetrics(unittest.TestCase):
    def test_parity_grid(self):
        try:
            from src.compute.vulkan_backend import VulkanComputeEngine
            eng = VulkanComputeEngine()
        except Exception as e:
            self.skipTest(f"Vulkan unavailable: {e}")
            return
        try:
            worst = {"max_abs_error": 0.0, "max_relative_error": 0.0, "spike_mismatch_count": 0}
            cases = 0
            for size, seed in ((32, 11), (32, 12), (64, 11), (64, 13)):
                g = get_or_create_circuit(size, mode=GraphMode.SYNTHETIC_TEST, seed=seed,
                                          cache_name=f"parity_grid_{size}_{seed}.npz")
                n = g.num_neurons
                for pat, refpat in (("quiet", "zero"), ("driven", "mixed"), ("negative", "ones")):
                    rng = np.random.RandomState(seed * 100 + size)
                    prev = (rng.rand(n) < (0.4 if pat == "driven" else 0.05)).astype(np.float32)
                    ext = {"quiet": rng.uniform(0, 0.2, n), "driven": rng.uniform(0, 0.8, n),
                           "negative": rng.uniform(-0.5, 0.1, n)}[pat].astype(np.float32)
                    pot = rng.uniform(-0.2, 0.5, n).astype(np.float32)
                    ref = {"zero": np.zeros(n, dtype=np.int32),
                           "mixed": (rng.rand(n) < 0.2).astype(np.int32) * 2,
                           "ones": np.ones(n, dtype=np.int32)}[refpat]
                    cp, cs, cr = cpu_lif_step(g.row_offsets, g.col_indices, g.weights,
                                              prev, ext, pot, ref)
                    eng.load_circuit(g.row_offsets, g.col_indices, g.weights, pot, prev, ref)
                    gp, gs, gr = eng.run_step_persistent(external_inputs=ext, readback=True)
                    m = parity_metrics(cp, gp, cs, gs)
                    np.testing.assert_array_equal(cr, gr)
                    cases += 1
                    for k in ("max_abs_error", "max_relative_error"):
                        worst[k] = max(worst[k], m[k])
                    worst["spike_mismatch_count"] += m["spike_mismatch_count"]
            print(f"[parity] {cases} cases: worst={worst}")
            self.assertLess(worst["max_abs_error"], 1e-4)
            self.assertEqual(worst["spike_mismatch_count"], 0)
        finally:
            eng.cleanup()

    def test_long_trajectory_parity(self):
        from copy import deepcopy
        from src.brain.runtime import BrainRuntime
        try:
            from src.compute.vulkan_backend import VulkanComputeEngine
            eng = VulkanComputeEngine()
            eng.cleanup()
        except Exception as e:
            self.skipTest(f"Vulkan unavailable: {e}")
            return
        g = get_or_create_circuit(48, mode=GraphMode.SYNTHETIC_TEST, seed=21,
                                  cache_name="parity_long_48.npz")
        a = BrainRuntime(deepcopy(g), use_gpu=False, seed=21, enable_plasticity=True)
        b = BrainRuntime(deepcopy(g), use_gpu=True, seed=21, enable_plasticity=True)
        rng = np.random.RandomState(9)
        mism = 0
        maxw = 0.0
        for _ in range(25):
            stim = rng.uniform(0.05, 0.6, 16).astype(np.float32)
            rew = float(rng.uniform(-0.3, 0.6))
            a.step(sensory_inputs={"visual": stim}, reward=rew)
            b.step(sensory_inputs={"visual": stim}, reward=rew)
            mism += int(np.sum(a.state.spikes != b.state.spikes))
            # v4.1 lazy GPU weight sync: the CPU mirror refreshes at explicit
            # sync points; sync before comparing (thresholds unchanged).
            b.sync_gpu_weights()
            maxw = max(maxw, float(np.max(np.abs(a.graph.weights - b.graph.weights))))
        print(f"[parity] 25-step trajectory: spike_mismatches={mism}, max_weight_diff={maxw:.2e}")
        a.cleanup(); b.cleanup()
        self.assertEqual(mism, 0)
        self.assertLess(maxw, 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
