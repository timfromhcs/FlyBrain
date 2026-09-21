import os
import sys
import json
import time
import asyncio
import platform
import psutil
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from dataclasses import asdict
from typing import Dict, Any, Optional, List

from src.connectome.types import GraphMode, ProvenanceStatus
from src.models.offline import apply_offline_env as _apply_offline_env
_apply_offline_env()
from src.brain.simulation_engine import SimulationEngine
from src.experiment.manager import ExperimentManager, get_file_sha256, get_git_commit
from src.common.determinism import SeedBundle
from src.population.population import Population
from src.version import VERSION as __version__

app = FastAPI(title="FlyBrain Lab — Biological Connectome Research Platform")

# Central Simulation Engine instance
SIMULATION_ENGINE: Optional[SimulationEngine] = None
EXPERIMENT_MGR = ExperimentManager()
COLONY: Optional[Population] = None


def get_colony() -> Population:
    """Live ALife colony (REAL state; small CPU circuit for interactivity)."""
    global COLONY
    if COLONY is None:
        seeds = SeedBundle(experiment_seed=7, generation_seed=8, organism_seed=9,
                           development_seed=10, mutation_seed=11, world_seed=12,
                           teacher_seed=13)
        COLONY = Population(6, seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=7)
    return COLONY

def get_engine() -> SimulationEngine:
    global SIMULATION_ENGINE
    if SIMULATION_ENGINE is None:
        # Environment-safe defaults (Hugging Face Spaces are CPU-only):
        # FLYBRAIN_CIRCUIT_SIZE, FLYBRAIN_GRAPH_MODE, FLYBRAIN_USE_GPU.
        _size = int(os.environ.get("FLYBRAIN_CIRCUIT_SIZE", "512"))
        _mode_raw = os.environ.get("FLYBRAIN_GRAPH_MODE", "REAL")
        try:
            from src.connectome.types import coerce_graph_mode as _coerce
            _mode = _coerce(_mode_raw)
        except ValueError:
            _mode = GraphMode.REAL
        _use_gpu = os.environ.get("FLYBRAIN_USE_GPU", "1") not in ("0", "false", "no")
        SIMULATION_ENGINE = SimulationEngine(
            circuit_size=_size,
            graph_mode=_mode,
            use_gpu=_use_gpu,
            seed=int(os.environ.get("FLYBRAIN_SEED", "42"))
        )
    return SIMULATION_ENGINE

@app.on_event("startup")
async def startup_event():
    engine = get_engine()
    engine.set_event_loop(asyncio.get_event_loop())
    if os.environ.get("FLYBRAIN_AUTOSTART", "0") == "1":
        engine.start()
    _get_watchdog()


def _server_event_loop():
    """Best-effort event loop handle (uvicorn provides one; TestClient threads
    may not — fall back to a fresh loop instead of raising)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.new_event_loop()

@app.on_event("shutdown")
def shutdown_event():
    if SIMULATION_ENGINE:
        SIMULATION_ENGINE.close()

# Static directories
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs("visual_evidence", exist_ok=True)
app.mount("/visual_evidence", StaticFiles(directory="visual_evidence"), name="visual_evidence")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>FlyBrain Lab is initializing...</h1>")

@app.get("/api/health")
def get_health():
    engine = get_engine()
    return {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "version": __version__,
        "graph_mode": engine.circuit.mode.value,
        "provenance_status": engine.circuit.provenance_status.value,
        "backend": "vulkan_gpu" if (engine.brain.gpu_engine and engine.brain.use_gpu) else "cpu_reference",
        "device_name": engine.brain.gpu_engine.device_name if engine.brain.gpu_engine else "CPU Reference Mode"
    }


@app.get("/api/version")
def get_version():
    return {"version": __version__, "release": f"FlyBrain V{__version__}"}


@app.get("/api/doctor")
def get_doctor():
    """Real environment verification for the SYSTEM view (never fabricated)."""
    from src.diagnostics.doctor import run_doctor
    return run_doctor()

@app.get("/api/state")
def get_state():
    engine = get_engine()
    return engine.get_full_state()

@app.get("/api/telemetry")
def get_telemetry():
    engine = get_engine()
    return engine.get_telemetry_payload()

@app.get("/api/connectome")
def get_connectome(max_nodes: int = 512, max_edges: int = 384):
    engine = get_engine()
    return engine.get_connectome_3d_view(max_nodes=max_nodes, max_edges=max_edges)

@app.post("/api/simulation/start")
def post_start():
    engine = get_engine()
    engine.start()
    return {"status": "STARTED", "is_running": True}

@app.post("/api/simulation/pause")
def post_pause():
    engine = get_engine()
    engine.pause()
    return {"status": "PAUSED", "is_running": False}

class StepRequest(BaseModel):
    steps: int = 1
    sensory_inputs: Optional[Dict[str, List[float]]] = None
    reward: float = 0.0

@app.post("/api/simulation/step")
def post_step(req: StepRequest):
    engine = get_engine()
    s_in = None
    if req.sensory_inputs:
        s_in = {k: np.array(v, dtype=np.float32) for k, v in req.sensory_inputs.items()}
    res = engine.step_single(n_steps=req.steps, sensory_inputs=s_in, reward=req.reward)
    return res

class ResetRequest(BaseModel):
    circuit_size: int = 512
    graph_mode: str = "REAL"
    seed: int = 42

@app.post("/api/simulation/reset")
def post_reset(req: ResetRequest):
    global SIMULATION_ENGINE
    if SIMULATION_ENGINE:
        SIMULATION_ENGINE.close()
    from src.connectome.types import coerce_graph_mode
    mode = coerce_graph_mode(req.graph_mode)
    SIMULATION_ENGINE = SimulationEngine(
        circuit_size=req.circuit_size,
        graph_mode=mode,
        use_gpu=True,
        seed=req.seed
    )
    SIMULATION_ENGINE.set_event_loop(_server_event_loop())
    return {"status": "RESET_COMPLETE", "graph_mode": mode.value,
            "graph_identity": GraphMode.canonical(mode),
            "neurons": req.circuit_size}

@app.get("/api/provenance")
def get_provenance():
    """Scientific status contract: graph identity, sampling, annotation levels,
    weight semantics, dataset hashes. All values live from the loaded circuit."""
    from src.connectome.types import GRAPH_IDENTITIES
    engine = get_engine()
    g = engine.circuit
    pm = dict(getattr(g, "provenance_metadata", None) or {})
    pops = g.populations.to_dict() if g.populations else {}
    return {
        "graph_identity": pm.get("graph_identity", GraphMode.canonical(g.mode)),
        "graph_mode": g.mode.value,
        "provenance_status": g.provenance_status.value,
        "graph_identities": GRAPH_IDENTITIES,
        "sampling": {
            "strategy": pm.get("selection_strategy", "unknown"),
            "detail": pm.get("selection_detail", ""),
            "seed": pm.get("selection_seed"),
            "bias": pm.get("sampling_bias", ""),
            "sampled_neurons": pm.get("sampled_neuron_count", g.num_neurons),
            "source_neurons": pm.get("source_neuron_total", pm.get("source_neuron_count")),
            "source_edges": pm.get("source_edge_total"),
            "sampled_edges": pm.get("circuit_synapses", g.num_synapses),
            "full_graph_available_locally": pm.get("full_graph_available_locally", False),
        },
        "weight_semantics": {
            "source": pm.get("weight_source", ""),
            "transform": pm.get("weight_transform", ""),
            "simulation_semantics": pm.get("simulation_semantics", ""),
            "note": "Derived simulation transform. NOT a measured conductance.",
        },
        "populations": {
            name: {"heuristic": p.get("heuristic", True),
                   "classification_method": p.get("classification_method", ""),
                   "annotation_status": p.get("annotation_status", ""),
                   "annotation_level": p.get("annotation_level", "HEURISTIC"),
                   "count": p.get("count", 0)}
            for name, p in pops.items()
        },
        "dataset": {"name": pm.get("dataset_name", "Janelia MaleCNS"),
                    "version": pm.get("version", "male-cns:v1.0"),
                    "soma_sha256": pm.get("soma_sha256", ""),
                    "connections_sha256": pm.get("connections_sha256", "")},
        "graph_hash": g.graph_hash,
    }

@app.get("/api/memory")
def get_memory(query: Optional[str] = None, limit: int = 10):
    engine = get_engine()
    mem = engine.memory
    episodes = mem.get_recent_episodes(limit=limit)
    skills = mem.get_skills()
    dreams = mem.get_recent_dreams(limit=limit)
    
    if query:
        # Filter episodes containing query
        episodes = [e for e in episodes if query.lower() in json.dumps(e).lower()]
        
    return {
        "working": mem.working.get_all_active(),
        "episodes": episodes,
        "skills": skills,
        "dreams": dreams
    }

@app.get("/api/evolution/lineage")
def get_evolution_lineage():
    engine = get_engine()
    return engine.evolution.get_lineage()

@app.post("/api/evolution/generation")
def post_evolution_generation(num_candidates: int = 4):
    engine = get_engine()
    with engine.lock:
        res = engine.evolution.run_generation(num_candidates=num_candidates)
    return res

@app.get("/api/dreams")
def get_dreams(limit: int = 10):
    engine = get_engine()
    return engine.memory.get_recent_dreams(limit=limit)

@app.post("/api/dreams/replay")
def post_dream_replay(mode: str = "deterministic", count: int = 2, seed: Optional[int] = None):
    engine = get_engine()
    # R8: default seed derives deterministically from engine state, never wall-clock.
    dream_seed = int(seed) if seed is not None else int(engine.brain.state.step_count)
    with engine.lock:
        res = engine.dream_engine.run_dream_cycle(mode=mode, seed=dream_seed, num_episodes_to_replay=count)
    return res

@app.get("/api/tools")
def get_tools():
    engine = get_engine()
    return engine.trainer.registry.list_tools()

class ToolExecRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any] = {}

@app.post("/api/tools/execute")
def post_execute_tool(req: ToolExecRequest):
    engine = get_engine()
    return engine.trainer.registry.execute(req.tool_name, req.params)

@app.get("/api/experiments")
def list_experiments():
    exp_dir = EXPERIMENT_MGR.exp_dir
    files = [f for f in os.listdir(exp_dir) if f.endswith(".json")]
    manifests = []
    for f in sorted(files, reverse=True)[:20]:
        try:
            with open(os.path.join(exp_dir, f), "r", encoding="utf-8") as fp:
                manifests.append(json.load(fp))
        except Exception:
            pass
    return manifests

class RunExpRequest(BaseModel):
    experiment_id: Optional[str] = None
    seed: int = 42
    graph_mode: str = "REAL"
    neuron_scale: int = 256
    duration_steps: int = 50

@app.post("/api/experiments/run")
def post_run_experiment(req: RunExpRequest):
    mode = GraphMode(req.graph_mode)
    manifest = EXPERIMENT_MGR.run_experiment(
        experiment_id=req.experiment_id,
        seed=req.seed,
        graph_mode=mode,
        neuron_scale=req.neuron_scale,
        duration_steps=req.duration_steps
    )
    return asdict(manifest)

@app.post("/api/experiments/verify")
def post_verify_experiment(experiment_id: str):
    res = EXPERIMENT_MGR.verify_experiment(experiment_id)
    return res

@app.get("/api/diagnostics")
def get_diagnostics():
    engine = get_engine()
    brain = engine.brain
    gpu_diag = brain.gpu_engine.get_diagnostics() if brain.gpu_engine else {
        "status": "UNAVAILABLE",
        "device_name": "CPU Reference Mode (Headless / No Vulkan Device)"
    }

    vm = psutil.virtual_memory()
    return {
        "system": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_percent": vm.percent,
            "git_commit": get_git_commit()
        },
        "vulkan": gpu_diag,
        "dataset_provenance": {
            "dataset_name": "Janelia MaleCNS",
            "version": "male-cns:v1.0",
            "soma_sha256": get_file_sha256(os.path.join("malecns", "data-raw", "2023-27-2 soma_sides.csv")),
            "connections_sha256": get_file_sha256(os.path.join("malecns", "data-raw", "malecns_v1_0_connections.csv")),
            "brain_shader_sha256": get_file_sha256(os.path.join("shaders", "brain_step.spv")),
            "plasticity_shader_sha256": get_file_sha256(os.path.join("shaders", "plasticity.spv"))
        },
        "runtime": {
            "is_running": engine.is_running,
            "circuit_neurons": engine.circuit.num_neurons,
            "circuit_synapses": engine.circuit.num_synapses,
            "graph_mode": engine.circuit.mode.value,
            "graph_hash": engine.circuit.graph_hash,
            "step_count": brain.state.step_count,
            "total_spikes": brain.state.total_spikes,
            "last_step_latency_ms": engine.last_step_time_ms
        }
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    """
    Real-time streaming telemetry WebSocket.
    Clients receive updates from the SimulationEngine broadcast queue.
    Clients do NOT independently advance or step the brain.
    """
    await ws.accept()
    engine = get_engine()
    queue = asyncio.Queue(maxsize=10)
    engine.register_telemetry_queue(queue)
    try:
        # Send initial state immediately
        await ws.send_json(engine.get_telemetry_payload())
        while True:
            payload = await queue.get()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        engine.unregister_telemetry_queue(queue)


# ---------------- ALife colony endpoints (REAL population state) ----------------

class ColonyResetRequest(BaseModel):
    size: int = 6
    seed: int = 7
    circuit_size: int = 32


@app.post("/api/colony/reset")
def post_colony_reset(req: ColonyResetRequest):
    global COLONY
    seeds = SeedBundle(experiment_seed=req.seed, generation_seed=req.seed + 1,
                       organism_seed=req.seed + 2, development_seed=req.seed + 3,
                       mutation_seed=req.seed + 4, world_seed=req.seed + 5,
                       teacher_seed=req.seed + 6)
    COLONY = Population(max(1, min(req.size, 50)), seeds,
                        GraphMode.SYNTHETIC_TEST, req.circuit_size,
                        experiment_seed=req.seed)
    return {"status": "COLONY_RESET", "size": len(COLONY.organisms)}


@app.post("/api/colony/step")
def post_colony_step(ticks: int = 5):
    pop = get_colony()
    pop.step(max(1, min(ticks, 50)))
    return {"status": "STEPPED", "tick": pop.tick, "living": len(pop.living())}


@app.post("/api/colony/reproduce")
def post_colony_reproduce(n_offspring: int = 2, mode: str = "sexual"):
    pop = get_colony()
    kids = pop.reproduce(max(1, min(n_offspring, 8)), mode=mode)
    return {"newborns": [k.id for k in kids], "total": len(pop.organisms)}


@app.get("/api/colony")
def get_colony_status():
    """Colony view: real per-organism state for sorting/filtering."""
    pop = get_colony()
    return {
        "tick": pop.tick,
        "living": len(pop.living()),
        "total": len(pop.organisms),
        "population_hash": pop.population_hash(),
        "world_hash": pop.world.world_hash(),
        "organisms": [
            {"id": o.id, "generation": o.generation, "stage": o.stage.value,
             "alive": o.alive, "age": o.age, "energy": round(o.energy, 3),
             "health": round(o.health, 3), "neurons": o.graph.num_neurons,
             "synapses": o.graph.num_synapses,
             "genome_hash": o.genome.genome_hash()[:16],
             "fitness": o.fitness_vector(),
             "skills": {k: round(v, 3) for k, v in o.skills.items()},
             "position": list(o.position), "parents": o.parents,
             "children": o.children}
            for o in sorted(pop.organisms, key=lambda x: x.id)
        ],
    }


@app.get("/api/colony/organism/{organism_id}")
def get_organism_detail(organism_id: str):
    """Organism inspector: real identity, lineage, brain, memory, culture."""
    pop = get_colony()
    for o in pop.organisms:
        if o.id == organism_id:
            return {
                "id": o.id, "generation": o.generation, "parents": o.parents,
                "children": o.children, "stage": o.stage.value, "alive": o.alive,
                "age": o.age, "energy": o.energy, "health": o.health,
                "genome": o.genome.to_dict(), "genome_hash": o.genome.genome_hash(),
                "brain": {"neurons": o.graph.num_neurons, "synapses": o.graph.num_synapses,
                          "graph_hash": o.graph.graph_hash,
                          "complexity": o.dev_engine.complexity_metrics(o.graph, o.dev)},
                "memory": {"episodes": len(o.episodes), "concepts": list(o.semantic.keys()),
                           "recent": o.episodes[-3:]},
                "culture": o.cultural_knowledge, "skills": o.skills,
                "organism_hash": o.organism_hash(),
            }
    raise HTTPException(status_code=404, detail="organism not found")


@app.get("/api/colony/lineage")
def get_colony_lineage():
    """Genetic vs cultural lineage trees (REAL tracked graphs)."""
    pop = get_colony()
    return {"genetic": pop.genetic_lineage, "cultural": pop.cultural_lineage,
            "teaching_sessions": [
                {"teacher": s.teacher, "student": s.student, "domain": s.domain,
                 "gain": s.learning_gain} for s in pop.teaching_sessions[-50:]]}


@app.get("/api/colony/experiments")
def list_alife_experiments():
    d = "diagnostics/alife_experiments"
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d), reverse=True)[:20]:
        if f.endswith(".json"):
            try:
                with open(os.path.join(d, f), encoding="utf-8") as fp:
                    out.append(json.load(fp))
            except Exception:
                pass
    return out


# ---------------- API v1 (V5 phase_07, typed contracts; legacy routes above stay) ----------------
# v1 delegates to the same live handlers: one backend, two contract versions.

def _v1_backend() -> str:
    engine = get_engine()
    return "vulkan_gpu" if (engine.brain.gpu_engine and engine.brain.use_gpu) else "cpu_reference"


@app.get("/api/v1/health")
def v1_health():
    h = get_health()
    return {"api": "v1", **h}


@app.get("/api/v1/readiness")
def v1_readiness():
    """Readiness for orchestrators: engine loaded + dataset files present."""
    engine = get_engine()
    soma = os.path.join("malecns", "data-raw", "2023-27-2 soma_sides.csv")
    conn = os.path.join("malecns", "data-raw", "malecns_v1_0_connections.csv")
    ready = (engine.circuit.num_neurons > 0
             and os.path.exists(soma) and os.path.exists(conn))
    return {"api": "v1", "ready": ready,
            "checks": {"circuit_loaded": engine.circuit.num_neurons > 0,
                       "soma_csv": os.path.exists(soma),
                       "connections_csv": os.path.exists(conn)},
            "backend": _v1_backend(), "version": get_version()["version"]}


@app.get("/api/v1/version")
def v1_version():
    return {"api": "v1", **get_version(),
            "commit": get_git_commit()}


@app.get("/api/v1/doctor")
def v1_doctor():
    return {"api": "v1", **get_doctor()}


@app.get("/api/v1/state")
def v1_state():
    return {"api": "v1", **get_state()}


@app.get("/api/v1/runtime")
def v1_runtime():
    """Runtime vitals: stream status + resources + backup/drive status."""
    from src.backup import gdrive
    engine = get_engine()
    vm = psutil.virtual_memory()
    return {"api": "v1", "stream": engine.stream_status(),
            "resources": {"ram_percent": vm.percent,
                          "ram_available_gb": round(vm.available / (1024 ** 3), 2),
                          "cpu_percent": psutil.cpu_percent(interval=None)},
            "backup": {"service": "local_disk", "root": os.path.abspath(
                os.environ.get("FLYBRAIN_BACKUP_DIR", "backups"))},
            "google_drive": gdrive.status()}


@app.get("/api/v1/metrics")
def v1_metrics():
    engine = get_engine()
    lat = list(engine.step_history)
    vm = psutil.virtual_memory()
    tele = engine.get_telemetry_payload()
    return {"api": "v1", "step": tele["step"], "spikes": tele["spikes"],
            "total_spikes": tele["total_spikes"],
            "latency_ms": {"last": engine.last_step_time_ms,
                           "mean": round(float(np.mean(lat)), 3) if lat else 0.0,
                           "p50": round(float(np.median(lat)), 3) if lat else 0.0,
                           "p95": round(float(np.percentile(lat, 95)), 3) if lat else 0.0,
                           "p99": round(float(np.percentile(lat, 99)), 3) if lat else 0.0},
            "memory": {"ram_percent": vm.percent},
            "backend": tele["backend"], "uptime_sec": engine.stream_status()["uptime_sec"]}


@app.get("/api/v1/events")
def v1_events(limit: int = 20):
    """Honest event surface: recent experiment manifests + dreams + heartbeat."""
    exps = list_experiments()[:max(1, min(limit, 20))]
    dreams = get_dreams(limit=5)
    engine = get_engine()
    return {"api": "v1", "heartbeat": engine.stream_status()["heartbeat"],
            "experiments": [{"experiment_id": e.get("experiment_id"),
                             "seed": e.get("seed"),
                             "final_state_hash": e.get("final_state_hash")}
                            for e in exps],
            "dreams": dreams}


@app.get("/api/v1/provenance")
def v1_provenance():
    return {"api": "v1", **get_provenance()}


@app.get("/api/v1/connectome")
def v1_connectome(max_nodes: int = 512, max_edges: int = 384):
    return {"api": "v1", **get_connectome(max_nodes=max_nodes, max_edges=max_edges)}


@app.get("/api/v1/neuron/{body_id}")
def v1_neuron(body_id: int):
    """Single-neuron sync point: metadata + live activity + capped connectivity."""
    engine = get_engine()
    g = engine.circuit
    idx = [i for i, b in enumerate(g.neuron_ids) if int(b) == int(body_id)]
    if not idx:
        raise HTTPException(status_code=404, detail="neuron body_id not in sampled circuit")
    i = idx[0]
    st = engine.brain.state
    row_s, row_e = int(g.row_offsets[i]), int(g.row_offsets[i + 1])
    incoming = [{"from_idx": int(g.col_indices[k]),
                 "from_body": int(g.neuron_ids[int(g.col_indices[k])]),
                 "w": round(float(g.weights[k]), 4)}
                for k in range(row_s, min(row_e, row_s + 64))]
    # outgoing: scan rows owning i as source (capped)
    outgoing = []
    for r in range(g.num_neurons):
        if len(outgoing) >= 64:
            break
        s, e = int(g.row_offsets[r]), int(g.row_offsets[r + 1])
        for k in range(s, e):
            if int(g.col_indices[k]) == i:
                outgoing.append({"to_idx": r, "to_body": int(g.neuron_ids[r]),
                                 "w": round(float(g.weights[k]), 4)})
                if len(outgoing) >= 64:
                    break
    pops = g.populations.to_dict() if g.populations else {}
    member_of = [n for n, p in pops.items() if i in p.get("neuron_indices", [])]
    return {"api": "v1", "body_id": int(body_id), "idx": i,
            "side": g.sides[i], "tbars": int(g.tbars[i]),
            "coordinates_nm": [float(x) for x in g.coordinates[i]],
            "annotation": {"position": "EMPIRICAL", "side": "EMPIRICAL",
                           "cell_type": "UNKNOWN", "hemilineage": "UNKNOWN",
                           "neurotransmitter": "UNKNOWN"},
            "populations": [{"name": n, "classification": "HEURISTIC"} for n in member_of],
            "activity": {"potential": round(float(st.membrane_potentials[i]), 4),
                         "spike": int(st.spikes[i] > 0.5),
                         "activation": round(float(st.activations[i]), 4)},
            "incoming": incoming, "outgoing": outgoing,
            "incoming_total": row_e - row_s}


@app.post("/api/v1/simulation/start")
def v1_sim_start():
    return {"api": "v1", **post_start(), **get_engine().stream_status()}


@app.post("/api/v1/simulation/pause")
def v1_sim_pause():
    return {"api": "v1", **post_pause()}


@app.post("/api/v1/simulation/resume")
def v1_sim_resume():
    return {"api": "v1", **get_engine().resume()}


@app.post("/api/v1/simulation/stop")
def v1_sim_stop():
    return {"api": "v1", **get_engine().stop()}


@app.post("/api/v1/simulation/reset")
def v1_sim_reset(req: ResetRequest):
    return {"api": "v1", **post_reset(req)}


@app.post("/api/v1/simulation/step")
def v1_sim_step(req: StepRequest):
    return {"api": "v1", **post_step(req)}


@app.post("/api/v1/stream/start")
def v1_stream_start(hz: float = 10.0):
    engine = get_engine()
    engine.target_hz = max(0.5, min(hz, 100.0))
    engine.start()
    return {"api": "v1", **engine.stream_status()}


@app.post("/api/v1/stream/stop")
def v1_stream_stop():
    return {"api": "v1", **get_engine().stop()}


@app.get("/api/v1/stream/status")
def v1_stream_status():
    return {"api": "v1", **get_engine().stream_status()}


def _backup_service():
    from src.backup.service import BackupService
    return BackupService()


@app.post("/api/v1/backup/create")
def v1_backup_create(label: str = "manual", trigger: str = "manual"):
    return {"api": "v1", **_backup_service().create_backup(get_engine(), None,
                                                           label=label, trigger=trigger)}


@app.get("/api/v1/backup/list")
def v1_backup_list():
    return {"api": "v1", "backups": _backup_service().list_backups()}


@app.get("/api/v1/backup/verify")
def v1_backup_verify(name: str):
    return {"api": "v1", **_backup_service().verify_backup(name)}


@app.post("/api/v1/backup/restore")
def v1_backup_restore(name: str):
    return {"api": "v1", **_backup_service().restore_backup(name, get_engine(), None)}


@app.get("/api/v1/backup/google")
def v1_backup_google():
    from src.backup import gdrive
    return {"api": "v1", **gdrive.status()}


@app.get("/api/v1/backup/download")
def v1_backup_download(name: str):
    """Verified backup as a streamed zip (hash-checked before serving)."""
    import shutil as _shutil
    import tempfile as _tf
    from fastapi.responses import FileResponse as _FR
    svc = _backup_service()
    verdict = svc.verify_backup(name)
    if verdict["status"] != "VALID":
        raise HTTPException(status_code=409,
                            detail=f"backup {name!r} is {verdict['status']}; not served")
    src = os.path.join(svc.root, name)
    tmp = _tf.mkdtemp(prefix="flybrain_dl_")
    archive = _shutil.make_archive(os.path.join(tmp, name), "zip", src)
    return _FR(archive, media_type="application/zip", filename=f"{name}.zip")


WATCHDOG = None


def _get_watchdog():
    global WATCHDOG
    if WATCHDOG is None and os.environ.get("FLYBRAIN_WATCHDOG", "1") == "1":
        from src.runtime.watchdog import Watchdog
        WATCHDOG = Watchdog(get_engine, _backup_service())
        WATCHDOG.start()
    return WATCHDOG


@app.get("/api/v1/watchdog/status")
def v1_watchdog_status():
    wd = _get_watchdog()
    if wd is None:
        return {"api": "v1", "running": False,
                "detail": "watchdog disabled (FLYBRAIN_WATCHDOG=0)"}
    return {"api": "v1", **wd.status()}


@app.get("/api/v1/experiments")
def v1_experiments():
    return {"api": "v1", "experiments": list_experiments()}


@app.post("/api/v1/experiments/run")
def v1_experiments_run(req: RunExpRequest):
    from src.connectome.types import coerce_graph_mode
    req.graph_mode = coerce_graph_mode(req.graph_mode).value
    return {"api": "v1", **post_run_experiment(req)}


@app.get("/api/v1/organisms")
def v1_organisms():
    return {"api": "v1", **get_colony_status()}


@app.get("/api/v1/organisms/{organism_id}")
def v1_organism_detail(organism_id: str):
    return {"api": "v1", **get_organism_detail(organism_id)}


@app.get("/api/v1/evolution/lineage")
def v1_evo_lineage():
    return {"api": "v1", **get_evolution_lineage()}


@app.get("/api/v1/memory")
def v1_memory(query: Optional[str] = None, limit: int = 10):
    return {"api": "v1", **get_memory(query=query, limit=limit)}


@app.get("/api/v1/dreams")
def v1_dreams(limit: int = 10):
    return {"api": "v1", "dreams": get_dreams(limit=limit)}


@app.get("/api/v1/llm/status")
def v1_llm_status():
    return {"api": "v1", **get_llm_status()}


# ---------------- V6 embodied world (server-authoritative MuJoCo) ----------------

def get_world():
    from src.world3d.service import get_service
    return get_service()


_MODEL_MANAGER = None


def get_models():
    global _MODEL_MANAGER
    if _MODEL_MANAGER is None:
        from src.models.loaders import default_manager
        _MODEL_MANAGER = default_manager()
    return _MODEL_MANAGER


@app.get("/api/v1/world/state")
def v1_world_state():
    return {"api": "v1", **get_world().state()}


@app.get("/api/v1/world/geometry")
def v1_world_geometry():
    return {"api": "v1", **get_world().geometry()}


@app.post("/api/v1/world/step")
def v1_world_step(ticks: int = 1):
    recs = get_world().step(max(1, min(ticks, 50)))
    last = recs[-1] if recs else {}
    return {"api": "v1", "advanced": len(recs), "last": last,
            "state_hash": get_world().world.state_hash()}


@app.get("/api/v1/world/agent")
def v1_world_agent(name: str = "hero"):
    svc = get_world()
    ag = svc.agents.get(name)
    if ag is None:
        raise HTTPException(status_code=404, detail="unknown agent")
    return {"api": "v1", "body": ag.body3d.to_dict(),
            "causal_tail": ag.causal_log[-5:]}


@app.get("/api/v1/world/map")
def v1_world_map():
    """Observer map (TRUE layout, labelled) + organism known map."""
    svc = get_world()
    hero = svc.agents["hero"]
    return {"api": "v1", "scope": "OBSERVER_TRUE_MAP",
            "known_scope": "ORGANISM_KNOWN_MAP",
            "known_free": len(hero.nav.known_free),
            "known_blocked": len(hero.nav.known_blocked),
            "remembered_paths": list(hero.nav.remembered_paths),
            "door_state": dict(svc.world.door_state)}


@app.post("/api/v1/world/interact")
def v1_world_interact(target: str = "food", agent: str = "hero"):
    svc = get_world()
    ag = svc.agents.get(agent)
    if ag is None:
        raise HTTPException(status_code=404, detail="unknown agent")
    return {"api": "v1", **ag._interact(target)}


@app.get("/api/v1/world/friends")
def v1_world_friends():
    svc = get_world()
    out = []
    for name, ag in svc.agents.items():
        if name == "hero":
            continue
        trust = None
        if ag.org.social_mem is not None:
            trust = ag.org.social_mem.trust_of("hero")
        out.append({"name": name, "goal": ag.body3d.goal,
                    "energy": round(ag.body3d.base.energy, 3),
                    "trust_hero": trust, "alive": ag.body3d.alive,
                    "pos": ag.body3d.pos})
    return {"api": "v1", "friends": out}


@app.post("/api/v1/world/dream")
def v1_world_dream(agent: str = "hero", with_image: bool = False):
    from src.world3d.dreaming import dream_cycle
    svc = get_world()
    ag = svc.agents.get(agent)
    if ag is None:
        raise HTTPException(status_code=404, detail="unknown agent")
    mm = get_models()
    try:
        text = mm.load("TEXT_MODEL")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"TEXT_MODEL UNAVAILABLE: {e}")
    img = None
    if with_image:
        try:
            img = mm.load("IMAGE_MODEL")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"IMAGE_MODEL UNAVAILABLE: {e}")
    return {"api": "v1", **dream_cycle(ag, text, image_model=img)}


@app.post("/api/v1/world/imagine")
def v1_world_imagine(theme: str, agent: str = "hero"):
    from src.world3d.dreaming import imagine
    svc = get_world()
    ag = svc.agents.get(agent)
    if ag is None:
        raise HTTPException(status_code=404, detail="unknown agent")
    mm = get_models()
    try:
        text = mm.load("TEXT_MODEL")
        img = mm.load("IMAGE_MODEL")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model UNAVAILABLE: {e}")
    return {"api": "v1", **imagine(ag, theme, text, img)}


@app.post("/api/v1/world/speak")
def v1_world_speak(text: str, voice: str = "af_heart"):
    import tempfile as _tf
    from src.world3d.speech_loop import synthesize_reply
    mm = get_models()
    try:
        tts = mm.load("TTS_MODEL")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"TTS_MODEL UNAVAILABLE: {e}")
    out = os.path.join(_tf.gettempdir(), f"flybrain_say_{abs(hash(text)) % 999999}.wav")
    return {"api": "v1", **synthesize_reply(tts, voice, text, out)}


@app.get("/api/v1/models")
def v1_models():
    from src.models.registry import scan_local
    return {"api": "v1", "registry": scan_local(), "manager": get_models().status(),
            "offline": os.environ.get("FLYBRAIN_OFFLINE", "0") == "1"}


@app.post("/api/v1/models/load")
def v1_models_load(task: str):
    try:
        get_models().load(task)
        return {"api": "v1", "status": "LOADED", "task": task}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"{task} UNAVAILABLE: {e}")


@app.post("/api/v1/models/unload")
def v1_models_unload(task: str):
    ok = get_models().unload(task)
    return {"api": "v1", "status": "UNLOADED" if ok else "NOT_LOADED", "task": task}


@app.post("/api/v1/world/backup")
def v1_world_backup(label: str = "world", trigger: str = "manual"):
    return {"api": "v1", **_backup_service().create_world_backup(
        get_world(), label=label, trigger=trigger)}


@app.post("/api/v1/world/restore")
def v1_world_restore(name: str):
    return {"api": "v1", **_backup_service().restore_world_backup(name, get_world())}


# ---------------- Local GGUF LLM endpoints (REAL runtime state) ----------------

LLM_RUNTIME = None


def get_llm():
    """Lazy local GGUF runtime. Never fabricates availability."""
    global LLM_RUNTIME
    if LLM_RUNTIME is None:
        from src.llm.runtime import LocalLLM
        LLM_RUNTIME = LocalLLM.auto(n_ctx=2048)
        LLM_RUNTIME.load()
    return LLM_RUNTIME


@app.get("/api/llm/status")
def get_llm_status():
    """Reports discovered models and honest runtime status (no fake 'ready')."""
    from src.llm.discovery import discover_models
    models = discover_models()
    llm = get_llm()
    return {
        "runtime_status": llm.status,
        "last_error": llm.last_error,
        "runtime": "llama-cpp-python",
        "active_model": llm.model.to_dict() if llm.model else None,
        "discovered": [m.to_dict() for m in models],
    }


class LLMGenerateRequest(BaseModel):
    prompt: str
    mode: str = "RESEARCH_DETERMINISTIC"
    max_tokens: int = 128
    seed: int = 42
    temperature: float = 0.0


@app.post("/api/llm/generate")
def post_llm_generate(req: LLMGenerateRequest):
    from src.llm.runtime import GenerationConfig
    llm = get_llm()
    cfg = GenerationConfig(mode=req.mode, max_tokens=max(1, min(req.max_tokens, 512)),
                           seed=req.seed, temperature=req.temperature)
    return llm.generate(req.prompt, cfg)


@app.get("/api/llm/tools")
def get_llm_tools():
    from src.llm.tools import build_toolset
    tools = build_toolset(get_engine=get_engine, get_colony=get_colony,
                          memory=get_engine().memory)
    return {name: {"description": t.description, "params_schema": t.params_schema}
            for name, t in tools.items()}


class LLMToolRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = {}


@app.post("/api/llm/tool")
def post_llm_tool(req: LLMToolRequest):
    """Validated, logged tool execution. Rejects anything outside the whitelist."""
    from src.llm.scientist import ScientistLoop
    from src.llm.tools import build_toolset
    llm = get_llm()
    loop = ScientistLoop(llm)
    for name, spec in build_toolset(get_engine=get_engine, get_colony=get_colony,
                                    memory=get_engine().memory).items():
        loop.register(spec)
    result = loop.execute_tool_request({"tool": req.tool, "params": req.params})
    # Auditable, non-authoritative: tool output is data, LLM text is not.
    try:
        import time as _t
        log_dir = "diagnostics/llm_agent"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "tool_calls.jsonl"), "a", encoding="utf-8") as fp:
            fp.write(json.dumps({"ts": _t.time(), "tool": req.tool, "params": req.params,
                                 "status": result["status"]}, sort_keys=True) + "\n")
    except Exception:
        pass
    return result
