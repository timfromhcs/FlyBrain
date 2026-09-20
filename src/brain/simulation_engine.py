import os
import time
import json
import asyncio
import threading
import numpy as np
from typing import Dict, Any, Optional, List, Callable

from src.connectome.types import ConnectomeGraph, GraphMode, ProvenanceStatus
from src.connectome.loader import get_or_create_circuit
from src.brain.runtime import BrainRuntime
from src.memory.persistence import PersistentMemoryManager
from src.trainer.curriculum import CurriculumTrainer
from src.evolution.scheduler import EvolutionScheduler
from src.dream.engine import DreamEngine

class SimulationEngine:
    """
    Single Authoritative Simulation Engine.
    Owns the central simulation loop, manages thread-safe state access,
    dispatches telemetry to connected WebSocket clients, and coordinates
    subsystems (brain, memory, learning, evolution, dreaming).
    """
    def __init__(
        self,
        circuit_size: int = 512,
        graph_mode: GraphMode = GraphMode.REAL,
        use_gpu: bool = True,
        seed: int = 42,
        db_path: str = "diagnostics/flybrain_live_memory.db"
    ):
        self.lock = threading.RLock()
        self.seed = seed
        self.graph_mode = graph_mode
        self.circuit_size = circuit_size
        self.use_gpu = use_gpu
        self.db_path = db_path
        
        # Initialize subsystems
        self.circuit = get_or_create_circuit(circuit_size, mode=graph_mode, seed=seed)
        self.memory = PersistentMemoryManager(db_path=db_path)
        self.brain = BrainRuntime(self.circuit, use_gpu=use_gpu, seed=seed)
        self.trainer = CurriculumTrainer(self.brain, self.memory)
        self.evolution = EvolutionScheduler(self.circuit, history_file="diagnostics/evolution_history.json")
        self.dream_engine = DreamEngine(self.brain, self.memory)
        
        # Concurrency & Execution Control
        self.is_running = False
        self.target_hz = 10.0
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        # Sensory Stimulus Queue
        self.pending_sensory: Dict[str, np.ndarray] = {}
        self.pending_reward: float = 0.0
        
        # Telemetry broadcast queues (for async WebSockets)
        self.telemetry_listeners: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Performance Telemetry
        self.last_step_time_ms = 0.0
        self.step_history: List[float] = []
        # Stream mode (V5 phase_10): 24/7 supervision surface.
        self.boot_time = time.time()
        self.stream_mode = "STOPPED"  # STOPPED | RUNNING | PAUSED
        self.heartbeat = {"ts": time.time(), "step": 0, "note": "boot"}

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def start(self):
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self.stream_mode = "RUNNING"
                self._stop_event.clear()
                self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="FlyBrainSimWorker")
                self._worker_thread.start()

    def pause(self):
        with self.lock:
            self.is_running = False
            if self.stream_mode == "RUNNING":
                self.stream_mode = "PAUSED"
            self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def resume(self) -> Dict[str, Any]:
        """Resume a paused stream (no-op when already running)."""
        with self.lock:
            was = self.stream_mode
        if was == "PAUSED":
            self.start()
        return {"status": "RESUMED" if was == "PAUSED" else "NOOP",
                "previous": was, "stream_mode": self.stream_mode}

    def stop(self) -> Dict[str, Any]:
        """Stop the stream loop; simulation state is preserved."""
        self.pause()
        with self.lock:
            self.stream_mode = "STOPPED"
        return {"status": "STOPPED", "step": self.brain.state.step_count}

    def safe_shutdown(self) -> Dict[str, Any]:
        """Stop loop + close GPU; state stays in memory for backup/snapshot."""
        res = self.stop()
        try:
            self.brain.cleanup()
        except Exception:
            pass
        return {**res, "status": "SAFE_SHUTDOWN"}

    def restart_runtime(self) -> Dict[str, Any]:
        """Destructive fallback: rebuild BrainRuntime on the same circuit
        (fresh neural state). Prefer restore-from-backup; this is the
        watchdog's third recovery rung, always logged by the caller."""
        self.pause()
        try:
            self.brain.cleanup()
        except Exception:
            pass
        self.brain = BrainRuntime(self.circuit, use_gpu=self.use_gpu, seed=self.seed)
        self.last_step_time_ms = 0.0
        self.step_history = []
        return {"status": "RUNTIME_RESTARTED", "step": 0}

    def stream_status(self) -> Dict[str, Any]:
        """24/7 stream status: uptime, sim time, population vitals, backend."""
        with self.lock:
            now = time.time()
            hb = dict(self.heartbeat)
            hb.update({"ts": now, "step": self.brain.state.step_count,
                       "stream_mode": self.stream_mode, "is_running": self.is_running})
            self.heartbeat = dict(hb)
            lat = list(self.step_history)
            return {
                "stream_mode": self.stream_mode,
                "is_running": self.is_running,
                "uptime_sec": round(now - self.boot_time, 1),
                "simulation_step": self.brain.state.step_count,
                "simulation_time_note": "1 step per loop iteration at target_hz",
                "target_hz": self.target_hz,
                "active_spikes": int(np.sum(self.brain.state.spikes > 0.5)),
                "total_spikes": self.brain.state.total_spikes,
                "mean_latency_ms": round(float(np.mean(lat)), 3) if lat else 0.0,
                "p95_latency_ms": round(float(np.percentile(lat, 95)), 3) if lat else 0.0,
                "backend": "vulkan_gpu" if (self.brain.gpu_engine
                                            and self.brain.use_gpu) else "cpu_reference",
                "heartbeat": hb,
            }

    def _run_loop(self):
        """Authoritative single simulation thread."""
        while not self._stop_event.is_set() and self.is_running:
            delay = 1.0 / max(1.0, self.target_hz)
            t0 = time.perf_counter()
            
            # Execute step under lock
            with self.lock:
                sensory = dict(self.pending_sensory) if self.pending_sensory else None
                self.pending_sensory.clear()
                reward = self.pending_reward
                self.pending_reward = 0.0
                
                res = self.brain.step(sensory_inputs=sensory, reward=reward)
                t1 = time.perf_counter()
                self.last_step_time_ms = round((t1 - t0) * 1000.0, 3)
                self.step_history.append(self.last_step_time_ms)
                if len(self.step_history) > 100:
                    self.step_history.pop(0)

            # Broadcast telemetry to async listeners
            self._broadcast_telemetry(res)
            
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.001, delay - elapsed)
            time.sleep(sleep_time)

    def step_single(
        self,
        n_steps: int = 1,
        sensory_inputs: Optional[Dict[str, np.ndarray]] = None,
        reward: float = 0.0
    ) -> Dict[str, Any]:
        """Manually steps the simulation by N steps under lock."""
        with self.lock:
            res = {}
            for _ in range(n_steps):
                t0 = time.perf_counter()
                res = self.brain.step(sensory_inputs=sensory_inputs, reward=reward)
                t1 = time.perf_counter()
                self.last_step_time_ms = round((t1 - t0) * 1000.0, 3)
                self.step_history.append(self.last_step_time_ms)
                if len(self.step_history) > 100:
                    self.step_history.pop(0)
            self._broadcast_telemetry(res)
            return res

    def set_sensory_stimulus(self, population_name: str, values: np.ndarray):
        with self.lock:
            self.pending_sensory[population_name] = values

    def add_reward(self, reward: float):
        with self.lock:
            self.pending_reward += reward

    def register_telemetry_queue(self, queue: asyncio.Queue):
        with self.lock:
            self.telemetry_listeners.append(queue)

    def unregister_telemetry_queue(self, queue: asyncio.Queue):
        with self.lock:
            if queue in self.telemetry_listeners:
                self.telemetry_listeners.remove(queue)

    def _broadcast_telemetry(self, step_result: Dict[str, Any]):
        """Dispatches lightweight telemetry to registered WebSocket queues."""
        payload = self.get_telemetry_payload()
        if self._loop and self.telemetry_listeners:
            for q in list(self.telemetry_listeners):
                try:
                    self._loop.call_soon_threadsafe(q.put_nowait, payload)
                except Exception:
                    pass

    def get_telemetry_payload(self) -> Dict[str, Any]:
        """Returns compact real-time telemetry dictionary without full connectome flooding."""
        with self.lock:
            brain = self.brain
            active_spikes = int(np.sum(brain.state.spikes > 0.5))
            top_active = [
                {"idx": int(i), "act": round(float(brain.state.activations[i]), 2)}
                for i in np.argsort(brain.state.activations)[-10:]
            ]
            gpu_diag = brain.gpu_engine.get_diagnostics() if brain.gpu_engine else {}
            
            return {
                "step": brain.state.step_count,
                "spikes": active_spikes,
                "total_spikes": brain.state.total_spikes,
                "mean_activation": round(float(np.mean(brain.state.activations)), 4),
                "mean_potential": round(float(np.mean(brain.state.membrane_potentials)), 4),
                "prediction_error": round(brain.state.prediction_error, 4),
                "predicted_reward": round(brain.state.predicted_reward, 4),
                "current_reward": round(brain.state.current_reward, 4),
                "drives": {
                    "energy": round(brain.state.drives.energy, 3),
                    "curiosity": round(brain.state.drives.curiosity, 3),
                    "social": round(brain.state.drives.social, 3),
                    "integrity": round(brain.state.drives.integrity, 3)
                },
                "tool_associations": dict(brain.state.tool_associations),
                "active_neurons": top_active,
                "step_latency_ms": self.last_step_time_ms,
                "backend": "vulkan_gpu" if (brain.gpu_engine and brain.use_gpu) else "cpu_reference",
                "device_name": brain.gpu_engine.device_name if brain.gpu_engine else "CPU Reference",
                "is_running": self.is_running,
                "graph_mode": self.circuit.mode.value,
                "graph_hash": self.circuit.graph_hash[:16]
            }

    def get_full_state(self) -> Dict[str, Any]:
        """Returns complete state for diagnostics and dashboard inspection."""
        with self.lock:
            return {
                "simulation": self.get_telemetry_payload(),
                "connectome": {
                    "num_neurons": self.circuit.num_neurons,
                    "num_synapses": self.circuit.num_synapses,
                    "mode": self.circuit.mode.value,
                    "provenance_status": self.circuit.provenance_status.value,
                    "graph_hash": self.circuit.graph_hash,
                    "populations": self.circuit.populations.to_dict() if self.circuit.populations else {}
                },
                "gpu_diagnostics": self.brain.gpu_engine.get_diagnostics() if self.brain.gpu_engine else {
                    "status": "UNAVAILABLE",
                    "device_name": "CPU Reference Mode"
                }
            }

    def get_connectome_3d_view(self, max_nodes: int = 512, max_edges: int = 384) -> Dict[str, Any]:
        """
        Returns normalized 3D coordinates, biological soma metadata,
        and active synaptic edges for the interactive 3D WebGL viewer.
        """
        with self.lock:
            graph = self.circuit
            N = min(max_nodes, graph.num_neurons)
            
            coords = graph.coordinates[:N]
            c_min = coords.min(axis=0)
            c_max = coords.max(axis=0)
            norm_coords = (coords - c_min) / (c_max - c_min + 1e-5) * 2.0 - 1.0  # Normalized to [-1, 1]

            nodes = []
            for i in range(N):
                nodes.append({
                    "id": int(graph.neuron_ids[i]),
                    "idx": i,
                    "pos": [round(float(norm_coords[i, 0]), 3),
                            round(float(norm_coords[i, 1]), 3),
                            round(float(norm_coords[i, 2]), 3)],
                    "side": graph.sides[i],
                    "tbars": int(graph.tbars[i]),
                    "act": round(float(self.brain.state.activations[i]), 3),
                    "pot": round(float(self.brain.state.membrane_potentials[i]), 3),
                    "spk": int(self.brain.state.spikes[i] > 0.5)
                })

            edges = []
            edge_count = 0
            # CSR v3: row i stores INCOMING sources; displayed edge = source -> i.
            for i in range(min(128, N)):
                start = graph.row_offsets[i]
                end = min(start + 4, graph.row_offsets[i + 1])
                for k in range(start, end):
                    src = int(graph.col_indices[k])
                    if src < N:
                        edges.append({
                            "src": src,
                            "tgt": i,
                            "w": round(float(graph.weights[k]), 2)
                        })
                        edge_count += 1
                        if edge_count >= max_edges:
                            break
                if edge_count >= max_edges:
                    break

            return {
                "num_neurons": graph.num_neurons,
                "num_synapses": graph.num_synapses,
                "mode": graph.mode.value,
                "provenance_status": graph.provenance_status.value,
                "graph_hash": graph.graph_hash,
                "nodes": nodes,
                "edges": edges
            }

    def close(self):
        self.pause()
        self.brain.cleanup()

    def cleanup(self):
        self.close()

    def get_latest_telemetry(self) -> Dict[str, Any]:
        return self.get_telemetry_payload()

    def get_current_state(self) -> Dict[str, Any]:
        return self.get_full_state()
