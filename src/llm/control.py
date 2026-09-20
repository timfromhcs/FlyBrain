"""LLM control plane (STAGE K): typed, schema-validated commands.

The LLM scientist may SPAWN, START, PAUSE, STOP, SAVE, LOAD, CONFIGURE,
REQUEST_EXPERIMENTS and PROPOSE hypotheses/curricula — through this strictly
validated command surface only. No shell, no code mutation, no direct weight
or memory edits, no fabricated results. Every execution is logged with
provenance; every rejection states the exact reason.
"""
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

COMMAND_SCHEMA_VERSION = "control_v1"

# command -> {required params: {name: type}, optional params, constraints}
COMMAND_SPECS: Dict[str, Dict[str, Any]] = {
    "SPAWN_POPULATION": {
        "required": {"size": int},
        "optional": {"world_seed": int, "config_name": str},
        "constraints": {"size": (1, 64)},
    },
    "START_RUN": {"required": {}, "optional": {"ticks": int},
                  "constraints": {"ticks": (1, 10000)}},
    "PAUSE_RUN": {"required": {}, "optional": {}, "constraints": {}},
    "STOP_RUN": {"required": {}, "optional": {}, "constraints": {}},
    "SAVE_CHECKPOINT": {"required": {"name": str}, "optional": {},
                        "constraints": {"name": (1, 200)}},
    "LOAD_CHECKPOINT": {"required": {"name": str}, "optional": {},
                        "constraints": {"name": (1, 200)}},
    "SET_WORLD_CONFIG": {
        "required": {"n_resources": int},
        "optional": {"n_hazards": int, "regrow_interval": int},
        "constraints": {"n_resources": (4, 512), "n_hazards": (0, 64),
                        "regrow_interval": (1, 100)},
    },
    "SET_EVOLUTION_CONFIG": {
        "required": {"offspring_per_generation": int},
        "optional": {"reproduction_mode": str},
        "constraints": {"offspring_per_generation": (0, 32)},
        "enum": {"reproduction_mode": ["sexual", "asexual"]},
    },
    "REQUEST_EXPERIMENT": {
        "required": {"experiment_type": str, "seed": int},
        "optional": {"ticks": int, "population_size": int},
        "constraints": {"seed": (0, 2 ** 31), "ticks": (1, 2000),
                        "population_size": (1, 32)},
        "enum": {"experiment_type": ["baseline", "ablation_no_teaching",
                                     "ablation_no_growth", "comparison"]},
    },
    "REQUEST_COMPARISON": {
        "required": {"experiment_a": str, "experiment_b": str},
        "optional": {}, "constraints": {},
    },
    "REQUEST_REPLAY": {"required": {"checkpoint_name": str}, "optional": {"ticks": int},
                       "constraints": {"ticks": (1, 2000)}},
    "PROPOSE_HYPOTHESIS": {
        "required": {"text": str, "based_on_experiments": list},
        "optional": {}, "constraints": {"text": (1, 2000)},
    },
    "PROPOSE_TASK": {
        "required": {"description": str, "success_criterion": str},
        "optional": {}, "constraints": {"description": (1, 500),
                                        "success_criterion": (1, 500)},
    },
    "PROPOSE_CURRICULUM": {
        "required": {"stages": list},
        "optional": {}, "constraints": {"stages": (1, 8)},
    },
}

# Capability allowlist: role -> commands the role may invoke. Typed dispatch is
# the security boundary (text params are data, never executed). Identifier
# params (checkpoint names, experiment ids, ...) must additionally match
# _IDENTIFIER_RE; free-text fields (hypothesis text, descriptions, curriculum
# stages) are never scanned and never executed.
CAPABILITY_ROLES: Dict[str, frozenset] = {
    "llm-scientist": frozenset(COMMAND_SPECS.keys()),
    "viewer": frozenset({"REQUEST_COMPARISON", "REQUEST_REPLAY", "PROPOSE_HYPOTHESIS"}),
}

# Identifier-shaped params: strict allowlist, no shell metachars possible.
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
IDENTIFIER_PARAMS = {"name", "checkpoint_name", "experiment_a", "experiment_b",
                     "config_name", "reproduction_mode", "experiment_type"}

# Deprecated: whole-blob substring blacklists were fragile (false positives on
# legitimate scientific text, false negatives via obfuscation). Kept as an
# empty tuple for backward-compatible imports; enforcement is capability-based.
FORBIDDEN_SUBSTRINGS: tuple = ()


@dataclass
class CommandEnvelope:
    command: str
    params: Dict[str, Any] = field(default_factory=dict)
    requested_by: str = "llm-scientist"
    schema_version: str = COMMAND_SCHEMA_VERSION
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"command": self.command, "params": self.params,
                "requested_by": self.requested_by,
                "schema_version": self.schema_version,
                "timestamp": self.timestamp or time.time()}


def validate_envelope(envelope: Any, role: str = "llm-scientist") -> tuple:
    """Returns (ok, error). Structural + schema + capability validation, no execution."""
    if not isinstance(envelope, CommandEnvelope):
        return False, "payload is not a CommandEnvelope"
    if envelope.schema_version != COMMAND_SCHEMA_VERSION:
        return False, f"unsupported schema version {envelope.schema_version!r}"
    spec = COMMAND_SPECS.get(envelope.command)
    if spec is None:
        return False, f"unknown command {envelope.command!r}"
    allowed_cmds = CAPABILITY_ROLES.get(role, frozenset())
    if envelope.command not in allowed_cmds:
        return False, f"command {envelope.command!r} not permitted for role {role!r}"
    params = envelope.params
    if not isinstance(params, dict):
        return False, "params must be a dict"
    for name, typ in spec["required"].items():
        if name not in params:
            return False, f"missing required param {name!r}"
        if typ is int and isinstance(params[name], bool):
            return False, f"param {name!r} must be int"
        if typ is int and not isinstance(params[name], int):
            return False, f"param {name!r} must be int"
        if typ is str and not isinstance(params[name], str):
            return False, f"param {name!r} must be str"
        if typ is list and not isinstance(params[name], list):
            return False, f"param {name!r} must be list"
    allowed = set(spec["required"]) | set(spec["optional"])
    extra = set(params) - allowed
    if extra:
        return False, f"unknown params {sorted(extra)}"
    for name, (lo, hi) in spec["constraints"].items():
        if name in params:
            v = params[name]
            if isinstance(v, str):
                if not (lo <= len(v) <= hi):
                    return False, f"param {name!r} length outside [{lo},{hi}]"
            elif isinstance(v, list):
                if not (lo <= len(v) <= hi):
                    return False, f"param {name!r} list length outside [{lo},{hi}]"
            elif not (lo <= v <= hi):
                return False, f"param {name!r}={v} outside [{lo},{hi}]"
    for name, allowed_vals in spec.get("enum", {}).items():
        if name in params and params[name] not in allowed_vals:
            return False, f"param {name!r} must be one of {allowed_vals}"
    # Capability-based identifier guard: identifier params must match the
    # strict allowlist (no shell metachars can pass). Free-text params
    # (text/description/success_criterion/stages) are data, never executed,
    # and are intentionally NOT scanned.
    for name in IDENTIFIER_PARAMS:
        if name in params and isinstance(params[name], str):
            if _IDENTIFIER_RE.fullmatch(params[name]) is None:
                return False, f"param {name!r} is not a valid identifier"
    return True, ""


class ResearchRuntime:
    """Headless research facade the control plane operates on. Owns the
    population, checkpoints and the experiment ledger. NO source mutation."""

    def __init__(self, experiment_seed: int = 42):
        self.experiment_seed = int(experiment_seed)
        self.population = None
        self.running = False
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.experiment_ledger: Dict[str, Dict[str, Any]] = {}
        self.hypotheses: List[Dict[str, Any]] = []
        self.curricula: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.execution_log: List[Dict[str, Any]] = []
        self._ticks_target = 0
        self._ticks_done = 0

    # ---- operations invoked by the control plane ----
    def op_spawn_population(self, size: int, world_seed: int = 47, **_):
        from src.common.determinism import SeedBundle
        from src.population.population import Population
        from src.connectome.types import GraphMode
        if self.population is not None:
            return {"status": "FAILED", "reason": "population already exists; STOP+RESET first"}
        seeds = SeedBundle(experiment_seed=self.experiment_seed,
                           generation_seed=self.experiment_seed + 1,
                           organism_seed=self.experiment_seed + 2,
                           development_seed=self.experiment_seed + 3,
                           mutation_seed=self.experiment_seed + 4,
                           world_seed=world_seed,
                           teacher_seed=self.experiment_seed + 6)
        self.population = Population(size, seeds, GraphMode.SYNTHETIC_TEST, 32,
                                     experiment_seed=self.experiment_seed,
                                     autonomy_mode=True, genome_version="2.0")
        return {"status": "EXECUTED", "population_size": size,
                "population_hash": self.population.population_hash()}

    def op_start_run(self, ticks: int = 10, **_):
        if self.population is None:
            return {"status": "FAILED", "reason": "no population"}
        self.running = True
        self._ticks_target = int(ticks)
        self._ticks_done = 0
        self.population.step(int(ticks))
        self._ticks_done = int(ticks)
        self.running = False
        return {"status": "EXECUTED", "ticks_run": self._ticks_done,
                "population_hash": self.population.population_hash()}

    def op_pause_run(self, **_):
        self.running = False
        return {"status": "EXECUTED", "paused": True}

    def op_stop_run(self, **_):
        self.running = False
        return {"status": "EXECUTED", "stopped": True,
                "tick": self.population.tick if self.population else 0}

    def op_save_checkpoint(self, name: str, **_):
        if self.population is None:
            return {"status": "FAILED", "reason": "no population"}
        self.checkpoints[name] = self.population.snapshot()
        return {"status": "EXECUTED", "checkpoint": name,
                "population_hash": self.population.population_hash()}

    def op_load_checkpoint(self, name: str, **_):
        if name not in self.checkpoints:
            return {"status": "FAILED", "reason": f"unknown checkpoint {name!r}"}
        from src.common.determinism import SeedBundle
        from src.population.population import Population
        seeds = SeedBundle(experiment_seed=self.experiment_seed,
                           generation_seed=self.experiment_seed + 1,
                           organism_seed=self.experiment_seed + 2,
                           development_seed=self.experiment_seed + 3,
                           mutation_seed=self.experiment_seed + 4,
                           world_seed=self.experiment_seed + 5,
                           teacher_seed=self.experiment_seed + 6)
        self.population = Population.restore(self.checkpoints[name], seeds)
        return {"status": "EXECUTED", "checkpoint": name,
                "population_hash": self.population.population_hash()}

    def op_request_experiment(self, experiment_type: str, seed: int,
                              ticks: int = 20, population_size: int = 4, **_):
        from src.common.determinism import SeedBundle
        from src.population.population import Population
        from src.connectome.types import GraphMode
        exp_id = f"exp-{experiment_type}-{seed}"
        seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                           organism_seed=seed + 2, development_seed=seed + 3,
                           mutation_seed=seed + 4, world_seed=seed + 5,
                           teacher_seed=seed + 6)
        pop = Population(population_size, seeds, GraphMode.SYNTHETIC_TEST, 32,
                         experiment_seed=seed, autonomy_mode=(experiment_type != "baseline"),
                         genome_version="2.0")
        pop.step(int(ticks))
        pop.reproduce(2)
        result = {
            "experiment_id": exp_id, "type": experiment_type, "seed": seed,
            "ticks": ticks, "population_size": population_size,
            "final_population_hash": pop.population_hash(),
            "teaching_sessions": len(pop.teaching_sessions),
            "living": len(pop.living()), "total_organisms": len(pop.organisms),
            "generations": sorted({o.generation for o in pop.organisms}),
        }
        self.experiment_ledger[exp_id] = result
        return {"status": "EXECUTED", "result": result}

    def op_request_comparison(self, experiment_a: str, experiment_b: str, **_):
        ra, rb = self.experiment_ledger.get(experiment_a), self.experiment_ledger.get(experiment_b)
        if ra is None or rb is None:
            return {"status": "FAILED", "reason": "unknown experiment id(s)"}
        comparison = {
            "a": experiment_a, "b": experiment_b,
            "hash_equal": ra["final_population_hash"] == rb["final_population_hash"],
            "teaching_sessions": {"a": ra["teaching_sessions"], "b": rb["teaching_sessions"]},
            "living": {"a": ra["living"], "b": rb["living"]},
            "generations": {"a": ra["generations"], "b": rb["generations"]},
        }
        return {"status": "EXECUTED", "comparison": comparison}

    def op_propose_hypothesis(self, text: str, based_on_experiments: list, **_):
        known = [e for e in based_on_experiments if e in self.experiment_ledger]
        unknown = [e for e in based_on_experiments if e not in self.experiment_ledger]
        # Deterministic research identity (V4 §34): content + sequence, never wall-clock.
        hid = hashlib.sha256(
            f"{text}|{len(self.hypotheses)}|{self.experiment_seed}".encode()).hexdigest()[:12]
        rec = {"hypothesis_id": hid, "text": text, "based_on_experiments": known,
               "unknown_references": unknown, "status": "HYPOTHESIS"}
        self.hypotheses.append(rec)
        return {"status": "EXECUTED", "hypothesis": rec}

    def op_propose_task(self, description: str, success_criterion: str, **_):
        tid = hashlib.sha256(
            f"{description}|{success_criterion}|{len(self.tasks)}|{self.experiment_seed}"
            .encode()).hexdigest()[:12]
        rec = {"task_id": tid, "description": description,
               "success_criterion": success_criterion, "status": "PROPOSED"}
        self.tasks.append(rec)
        return {"status": "EXECUTED", "task": rec}

    def op_propose_curriculum(self, stages: list, **_):
        if not all(isinstance(s, str) for s in stages):
            return {"status": "REJECTED", "reason": "curriculum stages must be strings"}
        cid = hashlib.sha256(
            f"{'|'.join(stages)}|{len(self.curricula)}|{self.experiment_seed}"
            .encode()).hexdigest()[:12]
        rec = {"curriculum_id": cid, "stages": stages, "status": "PROPOSED"}
        self.curricula.append(rec)
        return {"status": "EXECUTED", "curriculum": rec}


class ControlPlane:
    """Validates and executes LLM command envelopes against a ResearchRuntime."""

    def __init__(self, runtime: Optional[ResearchRuntime] = None,
                 role: str = "llm-scientist"):
        self.runtime = runtime if runtime is not None else ResearchRuntime()
        self.role = role

    def execute(self, envelope: Any) -> Dict[str, Any]:
        ok, err = validate_envelope(envelope, role=self.role)
        if not ok:
            result = {"status": "REJECTED", "reason": err,
                      "command": getattr(envelope, "command", str(envelope)[:80])}
            self.runtime.execution_log.append({**result, "ts": time.time()})
            return result
        op = getattr(self.runtime, f"op_{envelope.command.lower()}", None)
        if op is None:
            result = {"status": "REJECTED", "reason": "command has no executor",
                      "command": envelope.command}
        else:
            try:
                result = op(**envelope.params)
            except Exception as e:  # noqa: BLE001
                result = {"status": "FAILED", "command": envelope.command,
                          "reason": f"{type(e).__name__}: {e}"}
        self.runtime.execution_log.append({**result, "command": envelope.command,
                                           "ts": time.time()})
        return result
