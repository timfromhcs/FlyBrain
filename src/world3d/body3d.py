"""Embodied 3D body state (V6). Wraps the V5 BodyState (energy/health/damage)
with the physical and physiological state a world organism needs:

hunger/thirst/fatigue/temperature, 3D position/velocity/orientation,
grounded flag, current action/goal, fall-impact damage. All values are
measured from physics or integrated from explicit metabolic rules —
never decorative.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.embodiment.body import BodyState


@dataclass
class Body3D:
    base: BodyState = field(default_factory=BodyState)
    hunger: float = 0.0        # 0 sated .. 1 starving
    thirst: float = 0.0        # 0 quenched .. 1 dehydrated
    fatigue: float = 0.0       # 0 rested .. 1 exhausted
    temperature: float = 37.0  # Celsius model state
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw: float = 0.0
    grounded: bool = False
    upright: float = 1.0
    action: str = "idle"
    goal: str = ""
    fall_events: int = 0

    def update_from_physics(self, char_state: Dict[str, Any], grounded: bool,
                            upright: float, yaw: float) -> None:
        prev_vz = self.vel[2] if len(self.vel) == 3 else 0.0
        self.pos = list(char_state["pos"])
        self.vel = list(char_state["vel"])
        self.grounded = bool(grounded)
        self.upright = float(upright)
        self.yaw = float(yaw)
        # fall impact: hard vertical stop from fast descent damages health
        if grounded and prev_vz < -4.0 and self.vel[2] > -0.5:
            impact = min(0.5, (-prev_vz - 4.0) * 0.1)
            self.base.apply_damage(impact, cause="fall_impact")
            self.fall_events += 1

    def metabolize(self, dt_world: float, moving: bool, in_water: bool) -> None:
        """Explicit physiological integration over world dt (seconds)."""
        k = dt_world
        self.hunger = min(1.0, self.hunger + 0.002 * k * (1.5 if moving else 1.0))
        self.thirst = min(1.0, self.thirst + 0.003 * k)
        self.fatigue = min(1.0, self.fatigue + (0.004 if moving else 0.001) * k)
        if in_water:
            self.thirst = max(0.0, self.thirst - 0.05 * k)
            self.temperature += (20.0 - self.temperature) * 0.01 * k
        else:
            self.temperature += (37.0 - self.temperature) * 0.005 * k
        self.base.consume_energy(self.base.metabolic_cost(0.2 if moving else 0.0,
                                                          1.0 if moving else 0.0) * k)
        if self.hunger > 0.8 or self.thirst > 0.8:
            self.base.health = max(0.0, self.base.health - 0.005 * k)

    def eat(self, energy: float) -> None:
        self.hunger = max(0.0, self.hunger - energy * 2.0)
        self.base.gain_energy(energy)

    def rest(self, dt_world: float) -> None:
        self.fatigue = max(0.0, self.fatigue - 0.02 * dt_world)
        self.base.recover(0.01 * dt_world)

    @property
    def needs_sleep(self) -> bool:
        return self.fatigue > 0.85

    @property
    def alive(self) -> bool:
        return self.base.health > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"base": self.base.to_dict(), "hunger": round(self.hunger, 4),
                "thirst": round(self.thirst, 4), "fatigue": round(self.fatigue, 4),
                "temperature": round(self.temperature, 2), "pos": self.pos,
                "vel": self.vel, "yaw": round(self.yaw, 3), "grounded": self.grounded,
                "upright": round(self.upright, 3), "action": self.action,
                "goal": self.goal, "fall_events": self.fall_events,
                "alive": self.alive}

    def to_exact_dict(self) -> Dict[str, Any]:
        """Full-precision snapshot form (exact replay; to_dict rounds)."""
        d = self.to_dict()
        d.update({"base": {"energy": float(self.base.energy),
                           "health": float(self.base.health),
                           "damage": float(self.base.damage),
                           "position": list(self.base.position),
                           "heading": float(self.base.heading),
                           "age": self.base.age,
                           "rest_ticks": self.base.rest_ticks,
                           "damage_events": self.base.damage_events,
                           "last_damage_cause": getattr(self.base, "last_damage_cause", "")},
                  "hunger": float(self.hunger), "thirst": float(self.thirst),
                  "fatigue": float(self.fatigue),
                  "temperature": float(self.temperature),
                  "yaw": float(self.yaw), "upright": float(self.upright)})
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Body3D":
        b = cls()
        b.base = BodyState.from_dict(d.get("base", {}))
        for k in ("hunger", "thirst", "fatigue", "temperature", "yaw", "upright",
                  "action", "goal", "fall_events"):
            if k in d:
                setattr(b, k, d[k])
        b.pos = list(d.get("pos", [0.0, 0.0, 0.0]))
        b.vel = list(d.get("vel", [0.0, 0.0, 0.0]))
        b.grounded = bool(d.get("grounded", False))
        return b
