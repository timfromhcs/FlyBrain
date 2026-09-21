"""Embodied agent: the V6 closed loop (phase_38).

WORLD -> BODY -> SENSORS -> PERCEPTION -> BIOLOGICAL BRAIN -> GOALS ->
ACTION -> BODY -> PHYSICS -> WORLD -> REWARD -> MEMORY -> FUTURE DECISION

Reuses V5 systems (Organism brain/autonomy/social/language, culture,
dreams) inside a 3D body. Cognition receives ONLY sensor observations +
memories (TRUE_WORLD never leaks: inputs derive from the EyeSensor fan).
Every tick persists the §27 causal record.
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.genome.schema import Genome
from src.organism.organism import Organism
from src.world3d.body3d import Body3D
from src.world3d.perception import EyeSensor
from src.world3d.navigation import Navigator
from src.world3d.spatial_memory import SpatialMemory

ACTIONS = ("MOVE", "LOOK", "SPEAK", "LISTEN", "INTERACT", "NAVIGATE", "REST",
           "SLEEP", "DREAM", "GENERATE_IMAGE", "REMEMBER", "SOCIAL_INTERACT")


def validate_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Typed action gate: unknown types / bad args -> structured error."""
    if not isinstance(action, dict) or action.get("type") not in ACTIONS:
        return {"ok": False, "error": f"unknown action {action!r}"[:120]}
    t = action["type"]
    try:
        if t == "MOVE":
            h, s = float(action["heading"]), float(action["speed"])
            assert -3.2 <= h <= 3.2 and 0.0 <= s <= 2.0
        elif t == "NAVIGATE":
            float(action["x"])
            float(action["y"])
        elif t in ("SPEAK", "REMEMBER"):
            assert isinstance(action.get("text", ""), str)
        elif t == "INTERACT":
            assert action.get("target", "") in ("food", "door", "object", "friend")
        elif t == "SOCIAL_INTERACT":
            assert isinstance(action.get("peer", ""), str)
    except (KeyError, TypeError, ValueError, AssertionError):
        return {"ok": False, "error": f"invalid args for {t}"}
    return {"ok": True}


class EmbodiedAgent:
    def __init__(self, organism: Organism, char_name: str, world,
                 spatial: SpatialMemory, yaw: float = 0.0):
        self.org = organism
        self.char = char_name
        self.world = world
        self.body3d = Body3D()
        self.eye = EyeSensor()
        self.nav = Navigator(world.spec)
        self.spatial = spatial
        self.yaw = float(yaw)
        self.sleeping = False
        self._last_reward = 0.0
        self.causal_log: List[Dict[str, Any]] = []

    # ---- goal arbitration (needs + autonomy, explicit, logged) ----
    def arbitrate_goal(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        b = self.body3d
        cands = []
        if b.hunger > 0.55:
            cands.append({"goal": "find_food", "source": "need:hunger",
                          "urgency": round(b.hunger, 3)})
        if b.thirst > 0.6:
            cands.append({"goal": "find_water", "source": "need:thirst",
                          "urgency": round(b.thirst, 3)})
        if b.needs_sleep:
            cands.append({"goal": "sleep", "source": "need:fatigue",
                          "urgency": round(b.fatigue, 3)})
        n_novel = sum(1 for e in obs["events"] if e["type"] == "novel_location")
        if n_novel or len(obs["entities"]) == 0:
            cands.append({"goal": "explore", "source": "curiosity",
                          "urgency": 0.4})
        for ent in obs["entities"]:
            if ent.startswith("food_") and b.hunger > 0.3:
                cands.append({"goal": "eat_seen_food", "source": "perception",
                              "urgency": 0.6, "target": ent})
        if self.org.autonomy is not None:
            try:
                self.org.autonomy.generate_goals(
                    self.world.tick, b.base.energy, b.base.health,
                    drives={}, prediction_error=0.0, world_sense={}, position=(0, 0),
                    skills=self.org.skills)
                g = self.org.autonomy.active_goal()
                if g is not None:
                    cands.append({"goal": g.kind, "source": "autonomy",
                                  "urgency": 0.35})
            except Exception:
                pass
        if not cands:
            return {"goal": "wander", "source": "default", "urgency": 0.2}
        cands.sort(key=lambda c: c["urgency"], reverse=True)
        return cands[0]

    # ---- one closed-loop tick ----
    def tick(self, dt: float = 0.2, brain_reward: Optional[float] = None) -> Dict[str, Any]:
        w = self.world
        pos_before = list(self.body3d.pos)
        # 1-2. SENSORS -> PERCEPTION (observation ONLY)
        obs = self.eye.observe(w, self.char, self.yaw)
        # 3. observation -> brain currents (perception-gated, no hidden state)
        pano = np.array(obs["depth_panorama"], dtype=np.float32)
        vis = np.zeros(16, dtype=np.float32)
        vis[:16] = (1.0 - np.clip(pano.reshape(3, 7).mean(axis=0).repeat(3)[:16]
                                  / 15.0, 0, 1)).astype(np.float32)
        seen_food = min(1.0, sum(1 for e in obs["entities"] if e.startswith("food_")) / 3.0)
        seen_friend = min(1.0, sum(1 for e in obs["entities"] if e.startswith("friend")) / 2.0)
        olf = np.full(8, seen_food * 0.8, dtype=np.float32)
        mem = np.full(8, seen_friend * 0.7, dtype=np.float32)
        # 4. BIOLOGICAL BRAIN
        out = self.org.brain.step(
            sensory_inputs={"visual": vis, "olfactory": olf, "memory": mem},
            reward=float(self._last_reward if brain_reward is None else brain_reward))
        spikes = int(np.sum(self.org.brain.state.spikes > 0.5))
        # 5. GOALS
        goal = self.arbitrate_goal(obs)
        self.body3d.goal = goal["goal"]
        # 6. ACTION (typed + validated)
        action = self._decide(obs, goal, out)
        chk = validate_action(action)
        if not chk["ok"]:
            action = {"type": "MOVE", "heading": self.yaw, "speed": 0.0}
        collided, outcome_note = False, ""
        if self.sleeping:
            action = {"type": "SLEEP"}
        if action["type"] == "SLEEP" or goal["goal"] == "sleep":
            self.sleeping = True
            self.body3d.rest(dt * 4)
            self.body3d.action = "sleep"
            if self.body3d.fatigue < 0.25:
                self.sleeping = False
            outcome_note = "sleeping"
        elif action["type"] == "NAVIGATE":
            wp = self.nav.next_waypoint(self.body3d.pos[0], self.body3d.pos[1])
            if wp is None:
                route = self.nav.plan(self.body3d.pos[0], self.body3d.pos[1],
                                      action["x"], action["y"])
                if route is None:
                    outcome_note = "no_route_known_map"
                    action = {"type": "MOVE", "heading": self.yaw, "speed": 0.2}
                else:
                    wp = self.nav.next_waypoint(self.body3d.pos[0], self.body3d.pos[1])
            if wp is not None:
                h = math.atan2(wp[0] - self.body3d.pos[0], wp[1] - self.body3d.pos[1])
                action = {"type": "MOVE", "heading": h, "speed": 0.8}
        if action["type"] == "MOVE" and not self.sleeping:
            self.body3d.action = "move"
            speed = min(1.6, action["speed"] * self.org.body.speed_capacity()
                        if self.org.body else action["speed"])
            if w.in_water(self.body3d.pos[0], self.body3d.pos[1]):
                speed *= 0.7
            vx, vy = math.sin(action["heading"]) * speed, math.cos(action["heading"]) * speed
            w.physics.drive_character(self.char, vx, vy)
            self.yaw = float(action["heading"])
        elif action["type"] == "INTERACT":
            res = self._interact(action.get("target", "object"))
            outcome_note = res.get("status", "")
        elif action["type"] == "REST":
            self.body3d.rest(dt)
            self.body3d.action = "rest"
        # 7-8. PHYSICS -> WORLD
        for _ in range(max(1, int(dt / 0.01))):
            w.step(0.01, substeps=1)
        st = w.physics.char_state(self.char)
        self.body3d.update_from_physics(st, w.physics.grounded(self.char),
                                        w.physics.upright(self.char), self.yaw)
        self.body3d.metabolize(dt, moving=(self.body3d.action == "move"),
                               in_water=w.in_water(*self.body3d.pos[:2]))
        self.nav.observe_walkable(*self.body3d.pos[:2])
        nav_note = self.nav.note_progress(self.body3d.pos[0], self.body3d.pos[1],
                                          driving=(self.body3d.action == "move"))
        if nav_note == "replan" and self.nav.destination:
            gx, gy = self.nav.destination
            self.nav.plan(self.body3d.pos[0], self.body3d.pos[1], gx, gy)
        for c in w.physics.contacts():
            if f"char_{self.char}" in (c["geom1"], c["geom2"]):
                other = c["geom2"] if c["geom1"] == f"char_{self.char}" else c["geom1"]
                if "ground" not in other:
                    collided = True
                    break
        # 9. REWARD / OUTCOME
        reward = 0.0
        if self.body3d.action == "move" and not collided:
            reward += 0.02
        if collided:
            reward -= 0.05
        if self.body3d.hunger < 0.2:
            reward += 0.05
        self._last_reward = float(reward)
        # 10. MEMORY (episodic + spatial + visual frames on events)
        internal = {"hunger": round(self.body3d.hunger, 3),
                    "energy": round(self.body3d.base.energy, 3),
                    "reward": round(reward, 3)}
        self.org.episodes.append({"tick": w.tick, "organism_id": self.org.id,
                                  "observation": sorted(obs["entities"]),
                                  "action": action.get("type"), "reward": round(reward, 4),
                                  "goal": goal["goal"]})
        for ev in obs["events"]:
            self.spatial.record_frame(obs, internal, event=ev["type"])
            if ev["type"] == "novel_location":
                self.spatial.record_place(f"cell_{ev['cell']}", *self.body3d.pos[:2],
                                          w.tick, note="explored")
        for ent, info in obs["entities"].items():
            self.spatial.record_sighting(ent, w.tick, *self.body3d.pos[:2], info["dist"])
        self.org.age += 1
        # 11. causal record (§27)
        rec = {"tick": w.tick, "organism_id": self.org.id,
               "visual_observation": {"kind": obs["kind"],
                                      "entities": sorted(obs["entities"]),
                                      "frame": obs["frame_hash"][:12],
                                      "events": [e["type"] for e in obs["events"]]},
               "neural": {"spikes": spikes,
                          "action": out.get("selected_action", "")},
               "goal": goal,
               "action": {k: v for k, v in action.items() if k != "text"},
               "collision": collided,
               "position_before": [round(v, 3) for v in pos_before],
               "position_after": [round(v, 3) for v in self.body3d.pos],
               "reward": round(reward, 4), "memory_created": True,
               "outcome": outcome_note or nav_note}
        self.causal_log.append(rec)
        return rec

    def snapshot(self) -> Dict[str, Any]:
        return {"organism": self.org.snapshot(),
                "body3d": self.body3d.to_exact_dict(),
                "spatial": self.spatial.snapshot(),
                "yaw": self.yaw, "sleeping": self.sleeping,
                "last_reward": self._last_reward,
                "nav": {"free": sorted(self.nav.known_free),
                        "blocked": sorted(self.nav.known_blocked),
                        "route": self.nav.route, "destination": self.nav.destination,
                        "remembered": self.nav.remembered_paths},
                "eye": {"entities": sorted(self.eye.seen_entities),
                        "cells": sorted(self.eye.seen_cells)}}

    def restore(self, snap: Dict[str, Any]) -> None:
        self.org = Organism.restore(snap["organism"])
        self.spatial.restore(snap["spatial"])
        self.body3d = Body3D.from_dict(snap["body3d"])
        self.yaw = float(snap["yaw"])
        self.sleeping = bool(snap["sleeping"])
        self._last_reward = float(snap["last_reward"])
        self.nav.known_free = set(map(tuple, snap["nav"]["free"]))
        self.nav.known_blocked = set(map(tuple, snap["nav"]["blocked"]))
        self.nav.route = [tuple(w) for w in snap["nav"]["route"]]
        self.nav.destination = (tuple(snap["nav"]["destination"])
                                if snap["nav"]["destination"] else None)
        self.nav.remembered_paths = {k: [tuple(w) for w in v]
                                     for k, v in snap["nav"]["remembered"].items()}
        self.eye.seen_entities = set(snap["eye"]["entities"])
        self.eye.seen_cells = set(snap["eye"]["cells"])

    def _decide(self, obs: Dict[str, Any], goal: Dict[str, Any],
                out: Dict[str, Any]) -> Dict[str, Any]:
        g = goal["goal"]
        if g == "eat_seen_food":
            return {"type": "INTERACT", "target": "food"}
        if g == "find_food":
            best, bd = None, 1e9
            for s in self.spatial.where_seen("food_0") + self.spatial.where_seen("food_1"):
                d = math.hypot(s["x"] - self.body3d.pos[0], s["y"] - self.body3d.pos[1])
                if d < bd:
                    best, bd = s, d
            if best:
                return {"type": "NAVIGATE", "x": best["x"], "y": best["y"]}
            return {"type": "MOVE", "heading": self.yaw + 0.6, "speed": 0.7}
        if g == "find_water":
            return {"type": "NAVIGATE", "x": -9.0, "y": -2.0}
        if g == "explore":
            return {"type": "MOVE", "heading": self.yaw + 0.9, "speed": 0.8}
        if g == "sleep":
            return {"type": "SLEEP"}
        if self.org.autonomy is not None:
            try:
                cand = self.org.autonomy.synthesize_action(
                    self.world.tick, None, out, {}, self.body3d.base.energy,
                    self.body3d.base.health, exploration=0.4)
                return {"type": "MOVE", "heading": float(cand.heading),
                        "speed": float(cand.speed)}
            except Exception:
                pass
        return {"type": "MOVE", "heading": self.yaw, "speed": 0.4}

    def _interact(self, target: str) -> Dict[str, Any]:
        if target == "food":
            res = self.world.eat_food(self.char)
            if res.get("status") == "CONSUMED":
                self.body3d.eat(res["energy"])
                self.org.skills["forage"] = min(1.0, self.org.skills["forage"] + 0.02)
            return res
        if target == "door":
            near = [d for d in self.world.spec["doors"]
                    if math.hypot(d["x"] - self.body3d.pos[0],
                                  d["y"] - self.body3d.pos[1]) < 2.0]
            if not near:
                return {"status": "NOTHING_NEARBY"}
            return self.world.push_door(near[0]["id"])
        return {"status": "NO_HANDLER", "target": target}
