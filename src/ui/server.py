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
