"""First-person geometric perception (V6 phase_07).

The organism NEVER receives hidden world state. Each observation is built
from MuJoCo raycasts through the live physics (occlusion included):

  eye position + yaw -> ray fan -> hits -> entities + depth panorama

Perception kind is honestly labelled GEOMETRIC (VLM slot: UNAVAILABLE until
a verified local vision model exists). Events fire on: new entity, close
obstacle ahead, novel location cell.
"""
import math
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.world3d.physics import EYE_HEIGHT

FAN_YAW = [-60, -40, -20, 0, 20, 40, 60]          # degrees, 7 horizontal rays
FAN_PITCH = [-10, 0, 8]                            # degrees, 3 rows
MAX_RANGE = 15.0


def _dir_vec(yaw_rad: float, pitch_rad: float) -> List[float]:
    return [math.sin(yaw_rad) * math.cos(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad)]


class EyeSensor:
    def __init__(self):
        self.seen_entities: Set[str] = set()
        self.seen_cells: Set[str] = set()

    def observe(self, world, char_name: str, yaw: float) -> Dict[str, Any]:
        st = world.physics.char_state(char_name)
        eye = [st["pos"][0], st["pos"][1], st["pos"][2] + 0.55]
        depths: List[float] = []
        entities: Dict[str, Dict[str, Any]] = {}
        for pitch in FAN_PITCH:
            for dyaw in FAN_YAW:
                d = _dir_vec(yaw + math.radians(dyaw), math.radians(pitch))
                hit = world.physics.raycast(eye, d, MAX_RANGE, exclude_body=char_name)
                depths.append(hit["dist"] if hit["hit"] else MAX_RANGE)
                if hit["hit"]:
                    ent = hit["entity"]
                    prev = entities.get(ent, {"dist": 1e9, "count": 0})
                    prev["count"] += 1
                    prev["dist"] = min(prev["dist"], hit["dist"])
                    prev["geom"] = hit["geom"]
                    entities[ent] = prev
        events: List[Dict[str, Any]] = []
        for ent, info in entities.items():
            if ent not in self.seen_entities:
                self.seen_entities.add(ent)
                events.append({"type": "new_entity", "entity": ent,
                               "dist": info["dist"]})
        # close obstacle straight ahead (center column rays)
        center = [depths[r * len(FAN_YAW) + 3] for r in range(len(FAN_PITCH))]
        if min(center) < 1.5:
            events.append({"type": "obstacle_ahead", "dist": round(min(center), 2)})
        cell = f"{int(eye[0])}:{int(eye[1])}"
        if cell not in self.seen_cells:
            self.seen_cells.add(cell)
            events.append({"type": "novel_location", "cell": cell})
        pano = np.array(depths, dtype=np.float32)
        import hashlib
        frame_hash = hashlib.sha256(pano.tobytes()).hexdigest()
        return {"kind": "GEOMETRIC", "tick": world.tick,
                "eye": [round(v, 3) for v in eye], "yaw": round(float(yaw), 3),
                "depth_panorama": [round(float(v), 2) for v in depths],
                "entities": {k: {"dist": v["dist"], "count": v["count"],
                                 "geom": v.get("geom", "")}
                             for k, v in sorted(entities.items())},
                "frame_hash": frame_hash, "events": events}
