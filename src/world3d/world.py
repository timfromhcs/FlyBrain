"""Persistent 3D world runtime (V6 phase_04/19/22).

Owns: spec, physics, world clock, object states (doors/food/water), character
roster. Snapshot/restore covers physics + clock + consumed + roster config.
TRUE_WORLD lives here; organism knowledge lives in spatial memory (never the
reverse).
"""
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional

from src.world3d.spec import build_world_spec
from src.world3d.physics import PhysicsWorld

DAY_LENGTH_SEC = 600.0  # 10-minute full day cycle


class World3D:
    def __init__(self, seed: int = 7, characters: Optional[List[Dict[str, Any]]] = None):
        self.spec = build_world_spec(seed)
        self.seed = seed
        self.physics = PhysicsWorld(self.spec, characters)
        self.clock_sec = 0.0
        self.tick = 0
        self.consumed: List[str] = []       # food ids eaten (bodies parked)
        self.door_state: Dict[str, str] = {d["id"]: "closed" for d in self.spec["doors"]}
        self.events: List[Dict[str, Any]] = []

    # ---- clock ----
    @property
    def day_fraction(self) -> float:
        return (self.clock_sec % DAY_LENGTH_SEC) / DAY_LENGTH_SEC

    @property
    def is_night(self) -> bool:
        return self.day_fraction < 0.25 or self.day_fraction > 0.8

    # ---- stepping ----
    def step(self, dt: float = 0.05, substeps: int = 5) -> None:
        for _ in range(substeps):
            self.physics.step(dt / substeps)
        # door torques are single impulses: clear after integration
        for d in self.spec["doors"]:
            try:
                da = int(self.physics.model.jnt_dofadr[self._hinge(d["id"])])
                self.physics.data.qfrc_applied[da] = 0.0
            except Exception:
                pass
        self.clock_sec += dt
        self.tick += 1
        self._update_doors()

    def _update_doors(self) -> None:
        for d in self.spec["doors"]:
            try:
                jid = self._hinge(d["id"])
                ang = float(self.physics.data.qpos[int(self.physics.model.jnt_qposadr[jid])])
                state = "open" if abs(ang) > 0.35 else "closed"
                if state != self.door_state[d["id"]]:
                    self.door_state[d["id"]] = state
                    self.log("door", {"door": d["id"], "state": state, "angle": round(ang, 3)})
            except Exception:
                pass

    def _hinge(self, door_id: str) -> int:
        import mujoco
        jid = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"hinge_{door_id}")
        if jid < 0:
            raise KeyError(door_id)
        return jid

    # ---- queries ----
    def log(self, kind: str, payload: Dict[str, Any]) -> None:
        self.events.append({"tick": self.tick, "t": round(self.clock_sec, 2),
                            "kind": kind, **payload})

    def in_water(self, x: float, y: float) -> bool:
        w = self.spec["water"]
        return abs(x - w["cx"]) <= w["sx"] / 2 and abs(y - w["cy"]) <= w["sy"] / 2

    def nearest_food(self, x: float, y: float, max_d: float = 1.2) -> Optional[Dict[str, Any]]:
        best, bd = None, max_d
        for f in self.spec["foods"]:
            if f["id"] in self.consumed:
                continue
            d = math.hypot(f["x"] - x, f["y"] - y)
            if d < bd:
                best, bd = f, d
        return best

    # ---- interactions (real state transitions) ----
    def push_door(self, door_id: str, torque: float = 30.0) -> Dict[str, Any]:
        try:
            jid = self._hinge(door_id)
            da = int(self.physics.model.jnt_dofadr[jid])
            self.physics.data.qfrc_applied[da] += float(torque)
            return {"status": "EXECUTED", "door": door_id, "torque": torque,
                    "state": self.door_state.get(door_id, "unknown")}
        except KeyError:
            return {"status": "ERROR", "reason": f"unknown door {door_id!r}"}

    def eat_food(self, char_name: str) -> Dict[str, Any]:
        st = self.physics.char_state(char_name)
        f = self.nearest_food(st["pos"][0], st["pos"][1])
        if f is None:
            return {"status": "NOTHING_NEARBY"}
        self.consumed.append(f["id"])
        # park the body far below the world (deterministic, no rebuild)
        self.physics.teleport(f"bod_{f['id']}", 0.0, 0.0, -50.0)
        self.log("consumed", {"food": f["id"], "by": char_name, "energy": f["energy"]})
        return {"status": "CONSUMED", "food": f["id"], "energy": f["energy"]}

    def object_states(self) -> Dict[str, Any]:
        return {"doors": dict(self.door_state),
                "foods_remaining": [f["id"] for f in self.spec["foods"]
                                    if f["id"] not in self.consumed],
                "foods_consumed": list(self.consumed)}

    # ---- persistence ----
    def state_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.physics.data.qpos.tobytes())
        h.update(self.physics.data.qvel.tobytes())
        h.update(json.dumps({"tick": self.tick, "clock": round(self.clock_sec, 4),
                             "consumed": sorted(self.consumed),
                             "doors": self.door_state}, sort_keys=True).encode())
        return h.hexdigest()

    def snapshot(self) -> Dict[str, Any]:
        from src.version import VERSION
        return {"version": "world3d_v1", "app_version": VERSION, "seed": self.seed,
                "spec_hash": self.spec.get("spec_hash", ""),
                "physics": self.physics.snapshot(), "clock_sec": self.clock_sec,
                "tick": self.tick, "consumed": list(self.consumed),
                "events": self.events[-200:]}

    def restore(self, snap: Dict[str, Any]) -> None:
        if snap.get("spec_hash", "") != self.spec.get("spec_hash", ""):
            raise ValueError("world snapshot belongs to a different spec")
        # re-park consumed bodies BEFORE physics restore (teleport is part of state)
        for fid in snap.get("consumed", []):
            try:
                self.physics.teleport(f"bod_{fid}", 0.0, 0.0, -50.0)
            except KeyError:
                pass
        self.consumed = list(snap.get("consumed", []))
        self.physics.restore(snap["physics"])
        self.clock_sec = float(snap.get("clock_sec", 0.0))
        self.tick = int(snap.get("tick", 0))
        self.events = list(snap.get("events", []))
        self._update_doors()
