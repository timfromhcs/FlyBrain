"""Canonical Dataset Registry & Scientific Truth Separation for FlyBrain V10.

Maintains strict boundary between:
- Biological empirical connectome data (Janelia MaleCNS v1.0)
- Derived subgraphs and hub extractions
- Synthetic surrogates and control networks
- Dataset licensing (CC-BY-4.0) vs software licensing (Apache-2.0 / GPL-3.0)
"""
import os
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


class DatasetCategory(str, Enum):
    BIOLOGICAL = "BIOLOGICAL"
    DERIVED = "DERIVED"
    SURROGATE = "SURROGATE"
    SYNTHETIC = "SYNTHETIC"
    EMERGENT = "EMERGENT"
    EXPERIMENTAL = "EXPERIMENTAL"
    HEURISTIC = "HEURISTIC"
    UNKNOWN = "UNKNOWN"


@dataclass
class DatasetMetadata:
    dataset_id: str
    category: DatasetCategory
    name: str
    version: str
    organism: str
    source_url: str
    upstream_revision: str
    local_revision: str
    neuron_count: int
    edge_count: int
    files: Dict[str, Dict[str, Any]]
    derivation_relation: Optional[str]
    dataset_license: str
    software_license: str
    annotation_provenance: str
    is_full_connectome: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "category": self.category.value,
            "name": self.name,
            "version": self.version,
            "organism": self.organism,
            "source_url": self.source_url,
            "upstream_revision": self.upstream_revision,
            "local_revision": self.local_revision,
            "neuron_count": self.neuron_count,
            "edge_count": self.edge_count,
            "files": self.files,
            "derivation_relation": self.derivation_relation,
            "dataset_license": self.dataset_license,
            "software_license": self.software_license,
            "annotation_provenance": self.annotation_provenance,
            "is_full_connectome": self.is_full_connectome,
        }


def _file_hash_or_none(rel_path: str) -> Optional[str]:
    if not os.path.exists(rel_path):
        return None
    h = hashlib.sha256()
    with open(rel_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DatasetRegistry:
    """Single canonical registry for all datasets referenced in FlyBrain."""

    _REGISTRY: Dict[str, DatasetMetadata] = {
        "MALECNS_V1_FULL": DatasetMetadata(
            dataset_id="MALECNS_V1_FULL",
            category=DatasetCategory.BIOLOGICAL,
            name="Janelia MaleCNS Complete Adult Drosophila Brain & VNC",
            version="1.0",
            organism="Drosophila melanogaster (male)",
            source_url="https://github.com/natverse/malecns",
            upstream_revision="v1.0-release",
            local_revision="upstream-reference",
            neuron_count=125506,
            edge_count=99301,  # Full hub/filtered connection table in release
            files={
                "upstream_source": {
                    "url": "https://github.com/natverse/malecns",
                    "note": "Full Janelia dataset; not instantiated in a single monolithic circuit locally."
                }
            },
            derivation_relation=None,
            dataset_license="CC-BY-4.0",
            software_license="GPL-3.0",
            annotation_provenance="Janelia Research Campus (HHMI) & University of Cambridge",
            is_full_connectome=True,
        ),
        "MALECNS_V1_DERIVED_SUBGRAPH": DatasetMetadata(
            dataset_id="MALECNS_V1_DERIVED_SUBGRAPH",
            category=DatasetCategory.DERIVED,
            name="Janelia MaleCNS v1.0 Hub-Biased Derived Subgraph",
            version="1.0-subgraph",
            organism="Drosophila melanogaster (male)",
            source_url="malecns/data-raw/malecns_v1_0_connections.csv",
            upstream_revision="v1.0",
            local_revision="v10.0-verified",
            neuron_count=125506,  # 125,506 verified somas available in soma_sides.csv
            edge_count=99301,     # 99,301 empirical connection pairs
            files={
                "soma_sides": {
                    "path": "malecns/data-raw/2023-27-2 soma_sides.csv",
                    "expected_count": 125506,
                },
                "connections": {
                    "path": "malecns/data-raw/malecns_v1_0_connections.csv",
                    "expected_count": 99301,
                }
            },
            derivation_relation="DERIVED_FROM:MALECNS_V1_FULL(HUB_SAMPLING_TRANSFORM)",
            dataset_license="CC-BY-4.0",
            software_license="Apache-2.0",
            annotation_provenance="Derived hub extraction from Janelia MaleCNS v1.0",
            is_full_connectome=False,
        ),
        "MALECNS_SPATIAL_SURROGATE": DatasetMetadata(
            dataset_id="MALECNS_SPATIAL_SURROGATE",
            category=DatasetCategory.SURROGATE,
            name="MaleCNS Spatial Surrogate Control Network",
            version="1.0-surrogate",
            organism="Synthetic Control (Biological Somas)",
            source_url="procedural:scipy.spatial.cKDTree",
            upstream_revision="N/A",
            local_revision="v10.0",
            neuron_count=125506,
            edge_count=0,  # Procedurally synthesized on demand
            files={},
            derivation_relation="SURROGATE_CONTROL(SOMA_PRESERVED_DISTANCE_KNN)",
            dataset_license="Apache-2.0",
            software_license="Apache-2.0",
            annotation_provenance="Synthetic control preserving soma coordinates and T-bar capacities",
            is_full_connectome=False,
        ),
        "SYNTHETIC_TEST_V1": DatasetMetadata(
            dataset_id="SYNTHETIC_TEST_V1",
            category=DatasetCategory.SYNTHETIC,
            name="Deterministic Synthetic Test Connectome",
            version="1.0",
            organism="Artificial Synthetic Benchmark",
            source_url="procedural:deterministic_ring_lattice",
            upstream_revision="N/A",
            local_revision="v10.0",
            neuron_count=512,
            edge_count=4096,
            files={},
            derivation_relation=None,
            dataset_license="Apache-2.0",
            software_license="Apache-2.0",
            annotation_provenance="Deterministic non-biological synthetic graph for CI and regression",
            is_full_connectome=False,
        ),
    }

    @classmethod
    def get(cls, dataset_id: str) -> DatasetMetadata:
        if dataset_id not in cls._REGISTRY:
            raise KeyError(f"Unknown dataset_id: {dataset_id!r}. Registered: {list(cls._REGISTRY.keys())}")
        return cls._REGISTRY[dataset_id]

    @classmethod
    def list_all(cls) -> List[DatasetMetadata]:
        return list(cls._REGISTRY.values())

    @classmethod
    def compute_experiment_provenance(cls, dataset_id: str, circuit_size: int,
                                      seed: int, extra_params: Optional[Dict[str, Any]] = None) -> str:
        """Computes deterministic experiment provenance strictly binding dataset identity."""
        ds = cls.get(dataset_id)
        payload = {
            "dataset_id": ds.dataset_id,
            "dataset_category": ds.category.value,
            "dataset_version": ds.version,
            "dataset_license": ds.dataset_license,
            "is_full_connectome": ds.is_full_connectome,
            "circuit_size": circuit_size,
            "seed": seed,
            "extra": extra_params or {}
        }
        import json
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
