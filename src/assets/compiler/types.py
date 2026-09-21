"""FlyAsset Schema and Classification Types for FlyBrain V8/V9.

Implements Sections 32–36:
- AssetClassification: STATIC, RIGID_DYNAMIC, ARTICULATED, CHARACTER, CREATURE, VEHICLE, STRUCTURE, VEGETATION
- FlyAsset v1 Manifest
- Collision Proxy Types: SPHERE, CAPSULE, BOX, CYLINDER, CONVEX_HULL
"""
import time
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


class AssetClass(str, Enum):
    STATIC = "STATIC"
    RIGID_DYNAMIC = "RIGID_DYNAMIC"
    ARTICULATED = "ARTICULATED"
    CHARACTER = "CHARACTER"
    CREATURE = "CREATURE"
    VEHICLE = "VEHICLE"
    STRUCTURE = "STRUCTURE"
    VEGETATION = "VEGETATION"


class ColliderType(str, Enum):
    BOX = "BOX"
    CAPSULE = "CAPSULE"
    SPHERE = "SPHERE"
    CYLINDER = "CYLINDER"
    CONVEX_HULL = "CONVEX_HULL"


# Physical reference scales in meters (Section 34)
REFERENCE_SCALES_M = {
    "chair": [0.6, 0.6, 0.9],
    "table": [1.4, 0.8, 0.75],
    "wooden_bridge": [4.0, 1.8, 0.8],
    "tree": [2.5, 2.5, 6.0],
    "rock": [1.2, 1.0, 0.8],
    "crate": [0.6, 0.6, 0.6],
    "house": [8.0, 10.0, 6.5],
    "fly_organism": [0.003, 0.0015, 0.001]
}


@dataclass
class CollisionProxy:
    type: ColliderType
    dimensions_m: List[float]
    offset_m: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mass_kg: float = 1.0
    friction: List[float] = field(default_factory=lambda: [0.7, 0.1, 0.01])
    restitution: float = 0.1


@dataclass
class FlyAsset:
    schema_version: int = 1
    asset_id: str = ""
    asset_class: AssetClass = AssetClass.STATIC
    source_hash: str = ""
    generator: str = "flybrain_asset_compiler"
    model_revision: str = "v10.0"
    visual_mesh: str = ""  # path to .obj
    visual_glb: Optional[str] = None  # path to .glb
    dimensions_m: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    origin: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    materials: Dict[str, Any] = field(default_factory=dict)
    lod: Dict[str, str] = field(default_factory=dict)
    collision: Dict[str, Any] = field(default_factory=dict)
    parts: List[Dict[str, Any]] = field(default_factory=list)
    rig: Dict[str, Any] = field(default_factory=dict)
    animations: Dict[str, Any] = field(default_factory=dict)
    physics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["asset_class"] = self.asset_class.value
        return d
