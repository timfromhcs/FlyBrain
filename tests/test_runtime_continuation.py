"""Runtime snapshot continuation tests (P6): restore must yield identical futures."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import hashlib
import unittest
import numpy as np

from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.brain.runtime import BrainRuntime


def full_state_hash(rt: BrainRuntime) -> str:
    # v4.1: sync the lazy GPU weight mirror so the hash covers true state.
    if hasattr(rt, "sync_gpu_weights"):
        rt.sync_gpu_weights()
    h = hashlib.sha256()
    s = rt.state
    for arr in (s.membrane_potentials, s.spikes, s.refractory_steps,
                s.activations, s.attention, s.goal_embedding):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(rt.graph.weights.tobytes())
    h.update(rt.graph.row_offsets.tobytes())
    h.update(rt.graph.col_indices.tobytes())
    h.update(str((s.step_count, s.total_spikes, s.prediction_error,
                  s.predicted_reward, s.drives.energy, s.drives.curiosity,
                  s.drives.social, s.drives.integrity,
                  s.active_goal, s.tool_associations)).encode())
    return h.hexdigest()


def drive_runtime(use_gpu: bool, tag: str, steps_a: int, steps_b: int):
    circ = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=99,
                                 cache_name="runtime_cont_synth_64.npz")
    rt = BrainRuntime(circ, use_gpu=use_gpu, seed=99)
    rng = np.random.RandomState(5)
    n = circ.num_neurons
    for _ in range(steps_a):
        rt.step(sensory_inputs={"visual": rng.uniform(0.1, 0.5, 16).astype(np.float32)},
                reward=0.3)
    path = f"diagnostics/runtime_cont_{tag}.npz"
    rt.save_snapshot(path)
    for _ in range(steps_b):
        rt.step(sensory_inputs={"visual": rng.uniform(0.1, 0.5, 16).astype(np.float32)},
                reward=-0.1)
    h_branch = full_state_hash(rt)
    rt.cleanup()

    # restore branch: fresh runtime on identical circuit, load, same N steps
    circ2 = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=99,
                                  cache_name="runtime_cont_synth_64.npz")
    rt2 = BrainRuntime(circ2, use_gpu=use_gpu, seed=99)
    rt2.load_snapshot(path)
    rng2 = np.random.RandomState(5)
    for _ in range(steps_a):  # advance rng identically past phase A
        rng2.uniform(0.1, 0.5, 16)
    for _ in range(steps_b):
        rt2.step(sensory_inputs={"visual": rng2.uniform(0.1, 0.5, 16).astype(np.float32)},
                 reward=-0.1)
    h_restored = full_state_hash(rt2)
    rt2.cleanup()
    if os.path.exists(path):
        os.remove(path)
    return h_branch, h_restored


class TestRuntimeContinuation(unittest.TestCase):
    def test_cpu_snapshot_continuation(self):
        hb, hr = drive_runtime(False, "cpu", 10, 15)
        self.assertEqual(hb, hr)

    def test_gpu_snapshot_continuation_where_available(self):
        try:
            from src.compute.vulkan_backend import VulkanComputeEngine
            eng = VulkanComputeEngine()
            eng.cleanup()
        except Exception as e:
            self.skipTest(f"Vulkan unavailable: {e}")
            return
        hb, hr = drive_runtime(True, "gpu", 10, 15)
        self.assertEqual(hb, hr)

    def test_cpu_gpu_final_state_parity(self):
        # WITH plasticity: CPU and GPU implement the identical three-factor
        # rule on identical pre/post spikes (runtime restores S(t-1) around the
        # GPU ping-pong copy), so trajectories must agree to float precision.
        try:
            from src.compute.vulkan_backend import VulkanComputeEngine
            eng = VulkanComputeEngine()
            eng.cleanup()
        except Exception as e:
            self.skipTest(f"Vulkan unavailable: {e}")
            return
        circ = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=99,
                                     cache_name="runtime_cont_synth_64.npz")
        from copy import deepcopy
        rt_cpu = BrainRuntime(deepcopy(circ), use_gpu=False, seed=99,
                              enable_plasticity=True)
        rt_gpu = BrainRuntime(deepcopy(circ), use_gpu=True, seed=99,
                              enable_plasticity=True)
        rng = np.random.RandomState(5)
        for _ in range(12):
            stim = rng.uniform(0.1, 0.5, 16).astype(np.float32)
            rt_cpu.step(sensory_inputs={"visual": stim}, reward=0.2)
            rt_gpu.step(sensory_inputs={"visual": stim}, reward=0.2)
        np.testing.assert_array_equal(rt_cpu.state.spikes, rt_gpu.state.spikes)
        np.testing.assert_array_equal(rt_cpu.state.refractory_steps,
                                      rt_gpu.state.refractory_steps)
        self.assertEqual(rt_cpu.state.total_spikes, rt_gpu.state.total_spikes)
        self.assertLess(float(np.max(np.abs(rt_cpu.state.membrane_potentials -
                                            rt_gpu.state.membrane_potentials))), 1e-4)
        rt_gpu.sync_gpu_weights()
        self.assertLess(float(np.max(np.abs(rt_cpu.graph.weights -
                                            rt_gpu.graph.weights))), 1e-4)
        rt_cpu.cleanup()
        rt_gpu.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
