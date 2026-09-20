import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from src.connectome.types import ConnectomeGraph, GraphMode, ProvenanceStatus
from src.brain.state import BrainState, HomeostaticDrives
from src.brain.plasticity import PlasticityEngine
from src.brain.eligibility import (
    EligibilityState, EligibilityEngine, NeuromodulationConfig, PLASTICITY_MODES,
)
from src.compute.vulkan_backend import VulkanComputeEngine
from src.compute.cpu_reference import cpu_lif_step

class BrainRuntime:
    """
    Central Brain Runtime integrating biological connectome graph,
    persistent Vulkan GPU acceleration, LIF spiking neural dynamics,
    synaptic plasticity, homeostatic drives, and biological population mapping.

    plasticity_mode:
      "v1_hebbian"    legacy three-factor reward rule (compat baseline).
      "v2_eligibility" persistent eligibility traces + versioned neuromodulatory
                       signal (reward/novelty/prediction-error/social/goal).
    """
    def __init__(
        self,
        graph: ConnectomeGraph,
        use_gpu: bool = True,
        enable_plasticity: bool = True,
        seed: int = 42,
        plasticity_mode: str = "v1_hebbian",
        neuromod: Optional[NeuromodulationConfig] = None,
        trace_decay: float = 0.9,
        prediction_influence: bool = False,
        prediction_gain: float = 0.5,
    ):
        if plasticity_mode not in PLASTICITY_MODES:
            raise ValueError(f"unknown plasticity_mode {plasticity_mode!r}; "
                             f"expected one of {PLASTICITY_MODES}")
        self.graph = graph
        self.use_gpu = use_gpu
        self.enable_plasticity = enable_plasticity
        self.seed = seed
        self.plasticity_mode = plasticity_mode
        self.prediction_influence = bool(prediction_influence)
        self.prediction_gain = float(np.clip(prediction_gain, 0.0, 1.0))
        self.state = BrainState.create_initial(graph.num_neurons, seed=seed)
        self.plasticity = PlasticityEngine()
        self.plasticity_updates = 0  # cumulative synapses updated (observability)
        self.eligibility: Optional[EligibilityState] = None
        self.eligibility_engine: Optional[EligibilityEngine] = None
        if plasticity_mode == "v2_eligibility":
            self.eligibility = EligibilityState(len(graph.weights), seed=seed)
            self.eligibility_engine = EligibilityEngine(
                trace_decay=trace_decay,
                neuromod=neuromod if neuromod is not None else NeuromodulationConfig())
        
        self.gpu_engine: Optional[VulkanComputeEngine] = None
        # Lazy GPU weight sync (v4.1, measured: weight readback dominates
        # rewarded-step cost at scale, e.g. ~0.85ms of ~1.0ms at N=1024).
        # GPU weights are authoritative; the CPU mirror (self.graph.weights)
        # is refreshed only at explicit sync points. Telemetry reports
        # weights_synced so staleness is never silent.
        self._gpu_weights_dirty = False
        if self.use_gpu:
            try:
                self.gpu_engine = VulkanComputeEngine()
                self.gpu_engine.load_circuit(
                    graph.row_offsets,
                    graph.col_indices,
                    graph.weights,
                    self.state.membrane_potentials,
                    self.state.spikes
                )
            except Exception as e:
                print(f"[BrainRuntime] Notice: Vulkan physical initialization not available ({e}). Using verified CPU LIF reference engine.")
                self.gpu_engine = None
                self.use_gpu = False

        # Population mapping derived from biological connectome anatomy
        N = graph.num_neurons
        self.sensory_visual_indices = graph.get_population_indices("visual")
        if len(self.sensory_visual_indices) == 0:
            self.sensory_visual_indices = np.arange(0, min(64, N), dtype=np.int32)
            
        self.sensory_audio_indices = graph.get_population_indices("auditory")
        if len(self.sensory_audio_indices) == 0:
            self.sensory_audio_indices = np.arange(min(64, N), min(128, N), dtype=np.int32)
            
        self.sensory_olfactory_indices = graph.get_population_indices("olfactory")
        if len(self.sensory_olfactory_indices) == 0:
            self.sensory_olfactory_indices = self.sensory_audio_indices

        self.sensory_memory_indices = graph.get_population_indices("memory_association")
        if len(self.sensory_memory_indices) == 0:
            self.sensory_memory_indices = np.arange(min(128, N), min(192, N), dtype=np.int32)

        motor_all = graph.get_population_indices("motor")
        if len(motor_all) >= 32:
            self.motor_speak_indices = motor_all[:16]
            self.motor_act_indices = motor_all[16:32]
        else:
            self.motor_speak_indices = np.arange(max(0, N - 64), max(0, N - 32), dtype=np.int32)
            self.motor_act_indices = np.arange(max(0, N - 32), N, dtype=np.int32)

        desc_all = graph.get_population_indices("descending")
        if len(desc_all) >= 16:
            self.motor_image_indices = desc_all[:16]
            self.motor_remember_indices = desc_all[16:32] if len(desc_all) >= 32 else desc_all[:16]
        else:
            self.motor_image_indices = np.arange(max(0, N - 96), max(0, N - 64), dtype=np.int32)
            self.motor_remember_indices = self.motor_image_indices

    def sync_gpu_weights(self) -> bool:
        """Refresh the CPU weight mirror from authoritative GPU weights.

        Returns True if a sync was performed, False if the mirror was current
        (CPU path, GPU unavailable, or nothing changed). Called automatically
        by save_snapshot(); call explicitly before reading graph.weights /
        graph_hash after rewarded GPU steps (experiment manifests, validation,
        curriculum measurements).
        """
        if not self._gpu_weights_dirty:
            return False
        if self.gpu_engine is None or not self.use_gpu:
            self._gpu_weights_dirty = False
            return False
        downloaded = self.gpu_engine.download_weights()
        if downloaded is not None:
            self.graph.weights = downloaded
            self.graph.graph_hash = self.graph.compute_graph_hash()
        self._gpu_weights_dirty = False
        return downloaded is not None

    @property
    def gpu_weights_dirty(self) -> bool:
        return self._gpu_weights_dirty

    def step(
        self,
        sensory_inputs: Optional[Dict[str, np.ndarray]] = None,
        reward: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes one full biological LIF cognitive step:
        1. Injects sensory currents into biological populations
        2. Dispatches leaky integration, threshold check, spike generation, and refractory clamping
        3. Updates homeostatic drives and prediction errors
        4. Applies synaptic plasticity if enabled
        5. Computes motor readout
        """
        N = self.graph.num_neurons
        ext_inputs = np.zeros(N, dtype=np.float32)

        # Inject sensory currents
        if sensory_inputs:
            if "visual" in sensory_inputs:
                v_in = np.asarray(sensory_inputs["visual"], dtype=np.float32).flatten()
                length = min(len(v_in), len(self.sensory_visual_indices))
                ext_inputs[self.sensory_visual_indices[:length]] += v_in[:length]
            if "audio" in sensory_inputs:
                a_in = np.asarray(sensory_inputs["audio"], dtype=np.float32).flatten()
                length = min(len(a_in), len(self.sensory_audio_indices))
                ext_inputs[self.sensory_audio_indices[:length]] += a_in[:length]
            if "olfactory" in sensory_inputs:
                o_in = np.asarray(sensory_inputs["olfactory"], dtype=np.float32).flatten()
                length = min(len(o_in), len(self.sensory_olfactory_indices))
                ext_inputs[self.sensory_olfactory_indices[:length]] += o_in[:length]
            if "memory" in sensory_inputs:
                m_in = np.asarray(sensory_inputs["memory"], dtype=np.float32).flatten()
                length = min(len(m_in), len(self.sensory_memory_indices))
                ext_inputs[self.sensory_memory_indices[:length]] += m_in[:length]

        prev_spikes = self.state.spikes.copy()
        pot_in = self.state.membrane_potentials.copy()
        ref_in = self.state.refractory_steps.copy()

        # Execute LIF step (Persistent GPU or CPU)
        if self.gpu_engine is not None and self.use_gpu:
            new_pot, new_spk, new_ref = self.gpu_engine.run_step_persistent(
                external_inputs=ext_inputs,
                readback=True
            )
            # Authoritative GPU refractory counters are used directly (P4/P5:
            # never recompute GPU state locally).
        else:
            new_pot, new_spk, new_ref = cpu_lif_step(
                self.graph.row_offsets,
                self.graph.col_indices,
                self.graph.weights,
                prev_spikes,
                ext_inputs,
                pot_in,
                refractory_in=ref_in,
                decay=0.85,
                threshold=1.0,
                v_reset=0.0,
                v_rest=0.0,
                t_ref=2
            )

        # Update running activations (moving average spike rate)
        self.state.activations = self.state.activations * 0.9 + new_spk * 0.1
        self.state.membrane_potentials = new_pot
        self.state.spikes = new_spk
        self.state.refractory_steps = new_ref

        active_spikes = int(np.sum(new_spk > 0.5))
        self.state.total_spikes += active_spikes
        self.state.step_count += 1
        self.state.current_reward = reward

        # Predictive coding & homeostasis
        activity_mean = float(np.mean(self.state.activations))
        prediction = self.state.predicted_reward
        error = abs(reward - prediction)
        self.state.prediction_error = float(error)
        self.state.predicted_reward += 0.1 * (reward - prediction)

        novelty = float(np.std(self.state.activations))
        self.state.drives.step(activity_level=activity_mean, novelty=novelty)

        # Prediction-influenced attention/curiosity (opt-in; off in compat mode)
        if self.prediction_influence and len(self.state.attention) == N:
            err = float(self.state.prediction_error)
            gate = np.float32(np.clip(1.0 + self.prediction_gain * err, 0.5, 2.0))
            for idx in (self.sensory_visual_indices, self.sensory_olfactory_indices,
                        self.sensory_memory_indices):
                if len(idx):
                    self.state.attention[idx] = np.clip(
                        self.state.attention[idx] * gate, 0.0, 1.0)
                    self.state.attention[idx] /= max(
                        1e-6, float(np.mean(self.state.attention[idx])))
            self.state.drives.curiosity = float(np.clip(
                self.state.drives.curiosity + 0.1 * self.prediction_gain * err, 0.0, 1.5))

        # Synaptic Plasticity
        synapses_updated = 0
        neuromod_signal = 0.0
        if self.plasticity_mode == "v2_eligibility" and self.eligibility is not None:
            # v2: traces update every step; weights move when the versioned
            # neuromodulatory signal is nonzero (structural changes resize traces).
            self.eligibility.sync_size(len(self.graph.weights))
            if self.enable_plasticity:
                res = self.eligibility_engine.step(
                    self.graph, self.eligibility,
                    pre_spikes=prev_spikes, post_spikes=new_spk,
                    reward=reward, novelty=novelty,
                    prediction_error=float(self.state.prediction_error))
                neuromod_signal = res["signal"]
                synapses_updated = res["synapses_updated"] if abs(res["signal"]) > 1e-6 else 0
                if synapses_updated:
                    # Provenance: weights changed -> refresh graph hash.
                    self.graph.graph_hash = self.graph.compute_graph_hash()
        elif self.enable_plasticity and abs(reward) > 1e-4:
            if self.gpu_engine is not None and self.use_gpu:
                # P4: the persistent step's ping-pong copy already overwrote the
                # prev_spikes buffer with S(t); restore true S(t-1) so the
                # three-factor rule sees identical pre/post on CPU and GPU,
                # then re-upload S(t) to leave next-step state intact.
                # v4.1: weights stay GPU-resident (no per-step readback);
                # the CPU mirror syncs lazily via sync_gpu_weights().
                self.gpu_engine.upload_buffer_data("prev_spikes", prev_spikes)
                self.gpu_engine.run_plasticity_persistent(
                    learning_rate=0.05,
                    reward=reward,
                    readback=False
                )
                self.gpu_engine.upload_buffer_data("prev_spikes", new_spk)
                self._gpu_weights_dirty = True
                synapses_updated = len(self.graph.weights)
            else:
                synapses_updated = self.plasticity.apply_hebbian_update(
                    self.graph,
                    prev_activations=prev_spikes,
                    post_activations=new_spk,
                    reward=reward
                )
        self.plasticity_updates += synapses_updated

        # Motor decoding
        motor_speak = float(np.mean(self.state.activations[self.motor_speak_indices])) if len(self.motor_speak_indices) else 0.0
        motor_image = float(np.mean(self.state.activations[self.motor_image_indices])) if len(self.motor_image_indices) else 0.0
        motor_act = float(np.mean(self.state.activations[self.motor_act_indices])) if len(self.motor_act_indices) else 0.0
        motor_remember = float(np.mean(self.state.activations[self.motor_remember_indices])) if len(self.motor_remember_indices) else 0.0

        self.state.tool_associations = {
            "speak": round(motor_speak, 3),
            "generate_image": round(motor_image, 3),
            "act_in_environment": round(motor_act, 3),
            "remember": round(motor_remember, 3)
        }

        # Determine dominant motor action
        candidate_actions = sorted(self.state.tool_associations.items(), key=lambda x: x[1], reverse=True)
        selected_action = candidate_actions[0][0] if candidate_actions else "explore"

        return {
            "step": self.state.step_count,
            "spikes": active_spikes,
            "mean_potential": float(np.mean(new_pot)),
            "mean_activation": activity_mean,
            "prediction_error": round(self.state.prediction_error, 4),
            "drives": {
                "energy": round(self.state.drives.energy, 3),
                "curiosity": round(self.state.drives.curiosity, 3),
                "social": round(self.state.drives.social, 3),
                "integrity": round(self.state.drives.integrity, 3)
            },
            "selected_action": selected_action,
            "synapses_updated": synapses_updated,
            "weights_synced": not self._gpu_weights_dirty,
            "plasticity_mode": self.plasticity_mode,
            "neuromod_signal": round(neuromod_signal, 6),
            "backend": "vulkan_gpu" if (self.use_gpu and self.gpu_engine) else "cpu_reference"
        }

    def save_snapshot(self, filepath: str):
        """Saves full deterministic state snapshot for perfect resumption."""
        self.sync_gpu_weights()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            step_count=self.state.step_count,
            total_spikes=self.state.total_spikes,
            membrane_potentials=self.state.membrane_potentials,
            spikes=self.state.spikes,
            refractory_steps=self.state.refractory_steps,
            activations=self.state.activations,
            attention=self.state.attention,
            goal_embedding=self.state.goal_embedding,
            active_memory_refs=np.array(self.state.active_memory_refs, dtype=str),
            prediction_error=self.state.prediction_error,
            predicted_reward=self.state.predicted_reward,
            current_reward=self.state.current_reward,
            weights=self.graph.weights,
            row_offsets=self.graph.row_offsets,
            col_indices=self.graph.col_indices,
            neuron_ids=self.graph.neuron_ids,
            energy=self.state.drives.energy,
            curiosity=self.state.drives.curiosity,
            social=self.state.drives.social,
            integrity=self.state.drives.integrity,
            graph_hash=self.graph.graph_hash,
            graph_mode=self.graph.mode.value,
            tool_associations=json.dumps(self.state.tool_associations),
            active_goal=self.state.active_goal,
            plasticity_mode=self.plasticity_mode,
            eligibility_traces=(self.eligibility.traces if self.eligibility is not None
                                else np.zeros(0, dtype=np.float32)),
            eligibility_updates=(self.eligibility.updates if self.eligibility is not None
                                 else 0),
        )

    def load_snapshot(self, filepath: str):
        """Restores complete brain state and connectome weights from snapshot."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")
        data = np.load(filepath, allow_pickle=True)
        
        self.state.step_count = int(data["step_count"])
        self.state.total_spikes = int(data["total_spikes"])
        self.state.membrane_potentials = np.copy(data["membrane_potentials"])
        self.state.spikes = np.copy(data["spikes"])
        self.state.refractory_steps = np.copy(data["refractory_steps"])
        self.state.activations = np.copy(data["activations"])
        if "attention" in data and len(data["attention"]) == len(self.state.activations):
            self.state.attention = np.copy(data["attention"])
        if "goal_embedding" in data:
            self.state.goal_embedding = np.copy(data["goal_embedding"])
        if "active_memory_refs" in data:
            self.state.active_memory_refs = [str(x) for x in list(data["active_memory_refs"])]
        self.state.prediction_error = float(data["prediction_error"])
        self.state.predicted_reward = float(data["predicted_reward"])
        self.state.current_reward = float(data["current_reward"])
        
        self.state.drives.energy = float(data["energy"])
        self.state.drives.curiosity = float(data["curiosity"])
        self.state.drives.social = float(data["social"])
        self.state.drives.integrity = float(data["integrity"])
        
        if "tool_associations" in data:
            self.state.tool_associations = json.loads(str(data["tool_associations"]))
        if "active_goal" in data:
            self.state.active_goal = str(data["active_goal"])
        if "plasticity_mode" in data:
            self.plasticity_mode = str(data["plasticity_mode"])
        if self.plasticity_mode == "v2_eligibility":
            if self.eligibility is None:
                self.eligibility = EligibilityState(len(self.graph.weights), seed=self.seed)
                self.eligibility_engine = EligibilityEngine()
            if "eligibility_traces" in data:
                self.eligibility.traces = np.copy(data["eligibility_traces"]).astype(np.float32)
                self.eligibility.updates = int(data["eligibility_updates"])
        else:
            self.eligibility = None
            self.eligibility_engine = None
            
        # Restore graph weights and topology
        self.graph.weights = np.copy(data["weights"])
        self.graph.row_offsets = np.copy(data["row_offsets"])
        self.graph.col_indices = np.copy(data["col_indices"])
        self.graph.graph_hash = self.graph.compute_graph_hash()
        
        # If GPU backend active, reload persistent circuit
        if self.gpu_engine is not None and self.use_gpu:
            self.gpu_engine.load_circuit(
                self.graph.row_offsets,
                self.graph.col_indices,
                self.graph.weights,
                self.state.membrane_potentials,
                self.state.spikes,
                self.state.refractory_steps
            )

    def restore_snapshot(self, filepath: str):
        return self.load_snapshot(filepath)

    def cleanup(self):
        if self.gpu_engine:
            self.gpu_engine.cleanup()
            self.gpu_engine = None
