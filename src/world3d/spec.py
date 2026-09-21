"""Deterministic world specification (V6).

Meters, +Z up, origin at meadow center. Layout derives from seed; the spec
dict is hashed (spec_hash) and persisted with every backup.
"""
import hashlib
import json
from typing import Any, Dict, List

WORLD_SPEC_VERSION = "world3d_v1"
WORLD_SIZE = 40.0  # 40 x 40 m ground


def build_world_spec(seed: int = 7) -> Dict[str, Any]:
    import numpy as _np
    rng = _np.random.RandomState(seed)
    trees = [{"id": f"tree_{i}", "x": round(float(x), 2), "y": round(float(y), 2)}
             for i, (x, y) in enumerate([(-12.0, -9.0), (-9.5, -12.5), (11.0, -10.0),
                                         (13.5, 6.0), (-13.0, 8.0), (9.0, 12.0)])]
    rocks = [{"id": f"rock_{i}", "x": x, "y": y, "r": r} for i, (x, y, r) in
             enumerate([(6.0, -4.0, 0.7), (-5.0, 5.5, 0.5), (2.0, 10.0, 0.9)])]
    foods = [{"id": f"food_{i}", "x": round(float(rng.uniform(-14, 14)), 2),
              "y": round(float(rng.uniform(-14, 14)), 2), "energy": 0.25}
             for i in range(8)]
    spec = {
        "version": WORLD_SPEC_VERSION, "seed": int(seed), "size": WORLD_SIZE,
        "ground": {"size": WORLD_SIZE},
        "home": {"cx": 0.0, "cy": 10.0, "w": 7.0, "d": 6.0, "wall_h": 2.4,
                 "wall_t": 0.3, "door_w": 1.4, "door_side": "south"},
        "furniture": [
            {"id": "table", "kind": "static_box", "x": -1.5, "y": 10.5,
             "sx": 1.6, "sy": 0.9, "sz": 0.75},
            {"id": "crate_1", "kind": "dynamic_box", "x": 1.8, "y": 9.0,
             "sx": 0.7, "sy": 0.7, "sz": 1.2, "mass": 4.0},
        ],
        "trees": trees, "rocks": rocks, "foods": foods,
        "water": {"id": "lake", "cx": -9.0, "cy": -2.0, "sx": 6.0, "sy": 4.0,
                  "note": "zone effect (thirst relief + speed factor), no fluid sim"},
        "doors": [{"id": "home_door", "x": 0.0, "y": 7.0, "w": 1.4, "h": 2.0,
                   "hinge": "west", "state": "closed"}],
        "spawn": {"x": 0.0, "y": 4.0, "heading": 0.0},
    }
    spec["spec_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in spec.items() if k != "spec_hash"},
                   sort_keys=True).encode()).hexdigest()
    return spec
