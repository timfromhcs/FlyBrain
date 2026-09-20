"""Versioned population checkpoint envelope (v4.1, Phase 2).

Separates simulation state from research/provenance history while storing
both, so save -> terminate -> restore -> continue is exactly equivalent to
uninterrupted execution:

  state_hash     - Population.population_hash() (organisms + world)
  research_hash  - event history + lineages + teaching sessions
  combined_hash  - sha256(state_hash + research_hash)
  version        - FlyBrain version (src/version.py)
  schema_version - this envelope format (population_checkpoint_v1)
  graph_identity - canonical graph identity (REAL_SUBGRAPH / ...)
  experiment_identity - experiment seed + seed bundle + graph/circuit fingerprint
  rng_bundle     - full SeedBundle (deterministic continuation)
  world/population state - complete Population.snapshot()
  event history  - population EventLog + teaching sessions (dropped by the
                   legacy Population.restore; preserved here)

Legacy Population.snapshot()/restore() are untouched (backward compatible).
"""
import dataclasses
import hashlib
import json
from typing import Any, Dict

from src.common.events import EventLog

CHECKPOINT_SCHEMA_VERSION = "population_checkpoint_v1"


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode()


def research_hash_of(pop) -> str:
    h = hashlib.sha256()
    h.update(pop.events.compute_hash().encode())
    h.update(_canon({"genetic": pop.genetic_lineage, "cultural": pop.cultural_lineage}))
    h.update(_canon([dataclasses.asdict(s) for s in pop.teaching_sessions]))
    return h.hexdigest()


def experiment_identity_of(pop) -> str:
    seeds = pop.seeds
    return hashlib.sha256(_canon({
        "experiment_seed": pop.experiment_seed,
        "seeds": {k: getattr(seeds, k) for k in
                  ("experiment_seed", "generation_seed", "organism_seed",
                   "development_seed", "mutation_seed", "world_seed", "teacher_seed")},
        "graph_mode": pop.graph_mode.value,
        "circuit_size": pop.circuit_size,
        "genome_version": pop.genome_version,
        "autonomy_mode": pop.autonomy_mode,
    })).hexdigest()


def save_checkpoint(pop) -> Dict[str, Any]:
    from src.version import VERSION
    from src.connectome.types import GraphMode as _GM
    state_hash = pop.population_hash()
    research_hash = research_hash_of(pop)
    combined = hashlib.sha256(f"{state_hash}|{research_hash}".encode()).hexdigest()
    seeds = pop.seeds
    return {
        "version": VERSION,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_hash": state_hash,
        "research_hash": research_hash,
        "combined_hash": combined,
        "graph_identity": _GM.canonical(pop.graph_mode),
        "graph_mode": pop.graph_mode.value,
        "experiment_identity": experiment_identity_of(pop),
        "rng_bundle": {k: getattr(seeds, k) for k in
                       ("experiment_seed", "generation_seed", "organism_seed",
                        "development_seed", "mutation_seed", "world_seed", "teacher_seed")},
        "population": pop.snapshot(),
        "event_history": list(pop.events.events),
        "teaching_sessions": [dataclasses.asdict(s) for s in pop.teaching_sessions],
    }


def restore_checkpoint(payload: Dict[str, Any]):
    """Restore a population from an envelope. Verifies hashes; raises on mismatch."""
    from src.common.determinism import SeedBundle
    from src.connectome.types import GraphMode as _GM
    from src.culture.transmission import TeachingSession
    from src.population.population import Population
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {payload.get('schema_version')!r}")
    rb = payload["rng_bundle"]
    seeds = SeedBundle(experiment_seed=rb["experiment_seed"],
                       generation_seed=rb["generation_seed"],
                       organism_seed=rb["organism_seed"],
                       development_seed=rb["development_seed"],
                       mutation_seed=rb["mutation_seed"],
                       world_seed=rb["world_seed"],
                       teacher_seed=rb["teacher_seed"])
    pop = Population.restore(payload["population"], seeds)
    # Preserve research history the legacy path drops.
    pop.events = EventLog()
    pop.events.events = [dict(e) for e in payload.get("event_history", [])]
    pop.teaching_sessions = [TeachingSession(**s)
                             for s in payload.get("teaching_sessions", [])]
    # Verify: restored hashes must equal the recorded ones.
    if pop.population_hash() != payload["state_hash"]:
        raise ValueError("checkpoint state_hash mismatch after restore")
    if research_hash_of(pop) != payload["research_hash"]:
        raise ValueError("checkpoint research_hash mismatch after restore")
    return pop
