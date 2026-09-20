"""Deterministic identity layers (Phase 1, v4.1).

Separates the distinct identity concerns that must never collapse into one
ambiguous hash:

  source_identity        - dataset name/version + raw file hashes
  neuron_topology_identity - which neurons (body IDs) + geometry + metadata
  synapse_topology_identity - CSR row_offsets + col_indices (+ biological record ids)
  parameter_identity     - LIF scalars (lambda, V_thresh, V_reset, V_rest, t_ref)
  dynamic_state_identity - membrane potentials + spikes + refractory counters
  plasticity_identity    - weights + eligibility traces + neuromodulatory state
  structural_identity    - living-brain registry: added/pruned neurons/synapses + events
  genome_identity        - genome schema version + genes + mutation provenance
  organism_identity      - organism id + lifecycle stage + body state
  population_identity    - population composition + generation + lineage links
  experiment_identity    - experiment id + seed bundle + graph/param fingerprint
  research_identity      - ledger hash chain + hypotheses/milestones provenance

Implemented as a thin deterministic layer over the existing BrainIdentity
12-layer model (src/provenance/v4.py) plus experiment/research ledger inputs,
so stored identities stay comparable across v4.0/v4.1.
"""
import hashlib
import json
from typing import Any, Dict, Optional

from src.provenance.v4 import BrainIdentity

LAYER_NAMES = (
    "source_identity",
    "neuron_topology_identity",
    "synapse_topology_identity",
    "parameter_identity",
    "dynamic_state_identity",
    "plasticity_identity",
    "structural_identity",
    "genome_identity",
    "organism_identity",
    "population_identity",
    "experiment_identity",
    "research_identity",
)


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode()


def _h(payload: Any) -> str:
    return hashlib.sha256(_canon(payload)).hexdigest()


def compute_identity_layers(
    graph=None,
    brain_state=None,
    genome=None,
    memory_hash: str = "none",
    population_hash: str = "none",
    event_stream_hash: str = "none",
    synapse_provenance_hash: str = "none",
    development_hash: str = "none",
    lif_params: Optional[Dict[str, Any]] = None,
    plasticity_state: Optional[Dict[str, Any]] = None,
    structural_state: Optional[Dict[str, Any]] = None,
    organism_state: Optional[Dict[str, Any]] = None,
    experiment_spec: Optional[Dict[str, Any]] = None,
    research_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Compute all 12 deterministic identity layers. Missing inputs hash as 'none'
    only for layers with no data; topology/state layers require the graph."""
    layers: Dict[str, str] = {}
    if graph is not None:
        pm = getattr(graph, "provenance_metadata", {}) or {}
        layers["source_identity"] = _h({
            "dataset": pm.get("dataset_name", "Janelia MaleCNS"),
            "version": pm.get("version", "male-cns:v1.0"),
            "soma_sha256": pm.get("soma_sha256", ""),
            "connections_sha256": pm.get("connections_sha256", ""),
            "graph_identity": pm.get("graph_identity", "REAL_SUBGRAPH"),
        })
        layers["neuron_topology_identity"] = _h({
            "neuron_ids": getattr(graph, "neuron_ids", []).tolist()
            if hasattr(getattr(graph, "neuron_ids", []), "tolist") else [],
            "geometry": BrainIdentity._geometry(graph),
            "neuron_metadata": BrainIdentity._neuron_metadata(graph),
        })
        layers["synapse_topology_identity"] = BrainIdentity._topology(graph)
        layers["plasticity_identity"] = _h({
            "weights": hashlib.sha256(bytes(getattr(graph, "weights", b"") if isinstance(
                getattr(graph, "weights", None), (bytes, bytearray)) else
                getattr(graph, "weights").tobytes())).hexdigest()
            if graph is not None and hasattr(graph, "weights") else "none",
            "extra": plasticity_state or "none",
        })
    else:
        layers["source_identity"] = "none"
        layers["neuron_topology_identity"] = "none"
        layers["synapse_topology_identity"] = "none"
        layers["plasticity_identity"] = _h({"extra": plasticity_state or "none"})
    layers["parameter_identity"] = _h(lif_params or "none")
    if brain_state is not None:
        layers["dynamic_state_identity"] = hashlib.sha256(
            brain_state.membrane_potentials.tobytes()
            + brain_state.spikes.tobytes()
            + brain_state.refractory_steps.tobytes()).hexdigest()
    else:
        layers["dynamic_state_identity"] = "none"
    layers["structural_identity"] = _h(structural_state or development_hash or "none")
    layers["genome_identity"] = (genome.genome_hash()
                                 if genome is not None and hasattr(genome, "genome_hash")
                                 else _h(genome or "none"))
    layers["organism_identity"] = _h(organism_state or "none")
    layers["population_identity"] = _h(population_hash or "none")
    layers["experiment_identity"] = _h(experiment_spec or "none")
    layers["research_identity"] = _h(research_state or event_stream_hash or "none")
    # Combined (never a substitute for the individual layers)
    layers["combined_identity"] = _h({k: layers[k] for k in LAYER_NAMES})
    return layers
