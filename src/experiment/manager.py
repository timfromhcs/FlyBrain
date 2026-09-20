import os
import time
import json
import hashlib
import subprocess
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field

from src.connectome.types import GraphMode, ConnectomeGraph
from src.connectome.loader import get_or_create_circuit, DEFAULT_SOMA_PATH, DEFAULT_CONNECTIONS_PATH
from src.brain.runtime import BrainRuntime

EXPERIMENTS_DIR = os.path.join("diagnostics", "experiments")
SNAPSHOTS_DIR = os.path.join("diagnostics", "snapshots")

def get_git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown_commit"

def get_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class ExperimentManifest:
    experiment_id: str
    timestamp: float
    seed: int
    graph_mode: str
    neuron_scale: int
    duration_steps: int
    dataset_version: str
    soma_file_hash: str
    connections_file_hash: str
    brain_shader_hash: str
    plasticity_shader_hash: str
    git_commit: str
    hardware_device: str
    backend: str
    graph_hash: str
    initial_state_hash: str
    final_state_hash: str
    metrics: Dict[str, Any]
    snapshot_path: str
    graph_provenance: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

class ExperimentManager:
    def __init__(self, exp_dir: str = EXPERIMENTS_DIR, snap_dir: str = SNAPSHOTS_DIR):
        self.exp_dir = exp_dir
        self.snap_dir = snap_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.snap_dir, exist_ok=True)

    def run_experiment(
        self,
        experiment_id: Optional[str] = None,
        seed: int = 42,
        graph_mode: GraphMode = GraphMode.REAL,
        neuron_scale: int = 256,
        duration_steps: int = 50,
        stimulus_intensity: float = 0.5,
        reward_schedule: str = "periodic",
        use_gpu: bool = True
    ) -> ExperimentManifest:
        """Executes a fully deterministic, provenance-tracked research experiment."""
        t_start = time.time()
        if experiment_id is None:
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            experiment_id = f"exp_{graph_mode.value.lower()}_{neuron_scale}_{ts_str}"

        # 1. Provenance metadata
        commit = get_git_commit()
        soma_hash = get_file_sha256(DEFAULT_SOMA_PATH)
        conn_hash = get_file_sha256(DEFAULT_CONNECTIONS_PATH)
        b_spv_hash = get_file_sha256(os.path.join("shaders", "brain_step.spv"))
        p_spv_hash = get_file_sha256(os.path.join("shaders", "plasticity.spv"))

        # 2. Build circuit & runtime
        circuit = get_or_create_circuit(neuron_scale, mode=graph_mode, seed=seed)
        brain = BrainRuntime(circuit, use_gpu=use_gpu, seed=seed)
        
        backend = "vulkan_gpu" if (brain.gpu_engine and brain.use_gpu) else "cpu_reference"
        device_name = brain.gpu_engine.device_name if brain.gpu_engine else "CPU Reference"

        # Record initial state hash
        init_h = hashlib.sha256()
        init_h.update(brain.state.membrane_potentials.tobytes())
        init_h.update(brain.state.spikes.tobytes())
        init_h.update(circuit.weights.tobytes())
        initial_state_hash = init_h.hexdigest()

        # 3. Deterministic execution
        rng = np.random.RandomState(seed)
        step_latencies = []
        spike_trajectory = []
        reward_history = []
        
        for step_idx in range(duration_steps):
            # Deterministic sensory input
            vis_stim = rng.uniform(0.1, stimulus_intensity, 32).astype(np.float32)
            sensory = {"visual": vis_stim}
            
            # Deterministic reward schedule
            if reward_schedule == "periodic":
                rew = 0.5 if (step_idx % 5 == 0) else -0.1
            elif reward_schedule == "constant":
                rew = 0.2
            else:
                rew = 0.0
                
            t0 = time.perf_counter()
            step_res = brain.step(sensory_inputs=sensory, reward=rew)
            t1 = time.perf_counter()
            
            step_latencies.append((t1 - t0) * 1000.0)
            spike_trajectory.append(step_res["spikes"])
            reward_history.append(rew)

        # 4. Record final state hash (sync GPU weight mirror first so the
        # hash covers learned weights, not a stale CPU copy)
        brain.sync_gpu_weights()
        fin_h = hashlib.sha256()
        fin_h.update(brain.state.membrane_potentials.tobytes())
        fin_h.update(brain.state.spikes.tobytes())
        fin_h.update(circuit.weights.tobytes())
        final_state_hash = fin_h.hexdigest()

        # Save snapshot
        snapshot_file = os.path.join(self.snap_dir, f"{experiment_id}_final.npz")
        brain.save_snapshot(snapshot_file)

        metrics = {
            "total_steps": duration_steps,
            "total_spikes": brain.state.total_spikes,
            "mean_spikes_per_step": round(float(np.mean(spike_trajectory)), 2),
            "final_energy": round(brain.state.drives.energy, 4),
            "final_curiosity": round(brain.state.drives.curiosity, 4),
            "final_prediction_error": round(brain.state.prediction_error, 4),
            "mean_step_latency_ms": round(float(np.mean(step_latencies)), 3),
            "median_step_latency_ms": round(float(np.median(step_latencies)), 3),
            "min_step_latency_ms": round(float(np.min(step_latencies)), 3),
            "max_step_latency_ms": round(float(np.max(step_latencies)), 3),
            "throughput_steps_per_sec": round(1000.0 / max(0.001, float(np.mean(step_latencies))), 1)
        }

        brain.cleanup()

        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            timestamp=t_start,
            seed=seed,
            graph_mode=graph_mode.value,
            neuron_scale=neuron_scale,
            duration_steps=duration_steps,
            dataset_version="male-cns:v1.0",
            soma_file_hash=soma_hash,
            connections_file_hash=conn_hash,
            brain_shader_hash=b_spv_hash,
            plasticity_shader_hash=p_spv_hash,
            git_commit=commit,
            hardware_device=device_name,
            backend=backend,
            graph_hash=circuit.graph_hash,
            initial_state_hash=initial_state_hash,
            final_state_hash=final_state_hash,
            metrics=metrics,
            snapshot_path=snapshot_file,
            graph_provenance={
                "mode": circuit.mode.value,
                "provenance_status": circuit.provenance_status.value,
                "csr_convention": circuit.provenance_metadata.get("csr_convention", "unknown"),
                "selection_strategy": circuit.provenance_metadata.get("selection_strategy", "unknown"),
                "selection_seed": seed,
                "sampled_neurons": circuit.num_neurons,
                "sampled_edges": circuit.num_synapses,
                "weight_source": circuit.provenance_metadata.get("weight_source", "simulation"),
                "weight_transform": circuit.provenance_metadata.get("weight_transform", "none"),
            },
        )

        manifest_path = os.path.join(self.exp_dir, f"{experiment_id}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json())

        print(f"[ExperimentManager] Run complete: {experiment_id} | Final State Hash: {final_state_hash[:16]}")
        return manifest

    def verify_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Re-runs experiment using exact recorded seed and parameters,
        and verifies bitwise / floating-point identity of the final state hash.
        """
        manifest_path = os.path.join(self.exp_dir, f"{experiment_id}.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        orig_final_hash = data["final_state_hash"]
        
        # Re-run identical experiment
        re_manifest = self.run_experiment(
            experiment_id=f"{experiment_id}_reverify",
            seed=data["seed"],
            graph_mode=GraphMode(data["graph_mode"]),
            neuron_scale=data["neuron_scale"],
            duration_steps=data["duration_steps"],
            use_gpu=(data["backend"] == "vulkan_gpu")
        )

        matches = (re_manifest.final_state_hash == orig_final_hash)
        
        # Cleanup reverify manifest
        rev_man_path = os.path.join(self.exp_dir, f"{experiment_id}_reverify.json")
        if os.path.exists(rev_man_path):
            os.remove(rev_man_path)

        return {
            "experiment_id": experiment_id,
            "original_final_state_hash": orig_final_hash,
            "reproduced_final_state_hash": re_manifest.final_state_hash,
            "deterministic_match": matches,
            "status": "PASS" if matches else "FAIL"
        }

    def compare_experiments(self, exp_id_a: str, exp_id_b: str) -> Dict[str, Any]:
        """Compares two experiment runs across metrics, latency, and determinism."""
        man_a = json.load(open(os.path.join(self.exp_dir, f"{exp_id_a}.json"), "r", encoding="utf-8"))
        man_b = json.load(open(os.path.join(self.exp_dir, f"{exp_id_b}.json"), "r", encoding="utf-8"))

        return {
            "comparison": {
                "exp_a": exp_id_a,
                "exp_b": exp_id_b,
                "same_seed": man_a["seed"] == man_b["seed"],
                "same_graph_mode": man_a["graph_mode"] == man_b["graph_mode"],
                "same_graph_hash": man_a["graph_hash"] == man_b["graph_hash"],
                "same_final_state": man_a["final_state_hash"] == man_b["final_state_hash"],
                "latency_speedup": round(man_a["metrics"]["mean_step_latency_ms"] / max(0.001, man_b["metrics"]["mean_step_latency_ms"]), 2),
                "metrics_diff": {
                    "total_spikes_diff": man_b["metrics"]["total_spikes"] - man_a["metrics"]["total_spikes"],
                    "latency_diff_ms": round(man_b["metrics"]["mean_step_latency_ms"] - man_a["metrics"]["mean_step_latency_ms"], 3)
                }
            }
        }
