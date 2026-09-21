"""Persistent world runtime service (V6): one World3D + hero + friends.

Singleton per process (server owns it). Friends are full EmbodiedAgents
(genome, needs, memory, goals, social state). Backup/restore covers world +
all agents + spatial DB. Loading AI models is event-driven and lazy via
ModelManager; the world loop never waits for them.
"""
import os
from typing import Any, Dict, List, Optional

from src.genome.schema import Genome
from src.organism.organism import Organism
from src.connectome.types import GraphMode
from src.world3d.world import World3D
from src.world3d.agent import EmbodiedAgent
from src.world3d.spatial_memory import SpatialMemory
from src.world3d.social import social_encounter

from src.world.chunks.chunk_manager import WorldManager

_SERVICE = None


def _organism(oid: str, seed: int) -> Organism:
    return Organism(Genome.founder(seed, legacy=False), oid, generation=0,
                    seeds={"organism_seed": seed, "development_seed": seed + 1},
                    graph_mode=GraphMode.SYNTHETIC_TEST, circuit_size=32,
                    autonomy_mode=True)


class WorldService:
    def __init__(self, seed: int = 7, friends: int = 2,
                 db_path: str = "diagnostics/world_memory.db"):
        self.seed = seed
        spawn = [{"name": "hero", "x": 0.0, "y": 4.0}]
        for i in range(friends):
            spawn.append({"name": f"friend_{i}", "x": -2.0 - i, "y": 2.0})
        self.world = World3D(seed=seed, characters=spawn)
        self.spatial = SpatialMemory(db_path)
        self.world_manager = WorldManager(seed=seed)
        self.agents: Dict[str, EmbodiedAgent] = {}
        self.agents["hero"] = EmbodiedAgent(
            _organism("hero", 101), "hero", self.world, self.spatial, yaw=0.0)
        for i in range(friends):
            oid = f"friend_{i}"
            self.agents[oid] = EmbodiedAgent(
                _organism(oid, 200 + i), oid, self.world, self.spatial, yaw=0.0)

    def chunks(self) -> Dict[str, Any]:
        hero_pos = self.world.physics.char_state("hero")["pos"]
        active = self.world_manager.chunks.update_center(hero_pos[0], hero_pos[1], radius_chunks=2)
        return {
            "active_chunk_count": len(active),
            "chunks": [self.world_manager.chunks.loaded_chunks[k].to_dict() for k in active]
        }

    def step(self, agent_ticks: int = 1) -> List[Dict[str, Any]]:

        out = []
        for _ in range(max(1, agent_ticks)):
            for ag in self.agents.values():
                if ag.body3d.alive:
                    out.append(ag.tick())
            # pairwise social encounters (close friends interact for real)
            names = list(self.agents)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    social_encounter(self.agents[names[i]], self.agents[names[j]])
        return out

    def state(self) -> Dict[str, Any]:
        chars = []
        for name, ag in self.agents.items():
            st = self.world.physics.char_state(name)
            chars.append({"name": name, "pos": st["pos"], "yaw": round(ag.yaw, 3),
                          "upright": self.world.physics.upright(name),
                          "grounded": self.world.physics.grounded(name),
                          "goal": ag.body3d.goal, "action": ag.body3d.action,
                          "energy": round(ag.body3d.base.energy, 3),
                          "hunger": round(ag.body3d.hunger, 3),
                          "alive": ag.body3d.alive,
                          "sleeping": ag.sleeping})
        return {"tick": self.world.tick, "clock_sec": round(self.world.clock_sec, 2),
                "day_fraction": round(self.world.day_fraction, 3),
                "is_night": self.world.is_night,
                "state_hash": self.world.state_hash(),
                "objects": self.world.object_states(),
                "characters": chars}

    def geometry(self) -> Dict[str, Any]:
        """Static render geometry derived from the spec (observer view)."""
        spec = self.world.spec
        h = spec["home"]
        cx, cy, w, d, t, dw, wh = (h["cx"], h["cy"], h["w"], h["d"], h["wall_t"],
                                   h["door_w"], h["wall_h"])
        q = (w - dw) / 2
        walls = [
            {"x": cx, "y": cy + d / 2, "sx": w, "sy": t, "h": wh},
            {"x": cx - w / 2, "y": cy, "sx": t, "sy": d, "h": wh},
            {"x": cx + w / 2, "y": cy, "sx": t, "sy": d, "h": wh},
            {"x": cx - (dw / 2 + q / 2), "y": cy - d / 2, "sx": q, "sy": t, "h": wh},
            {"x": cx + (dw / 2 + q / 2), "y": cy - d / 2, "sx": q, "sy": t, "h": wh},
        ]
        return {"size": spec["size"], "home": spec["home"], "walls": walls,
                "furniture": spec["furniture"], "trees": spec["trees"],
                "rocks": spec["rocks"],
                "foods": [f for f in spec["foods"] if f["id"] not in self.world.consumed],
                "water": spec["water"], "doors": spec["doors"],
                "door_state": dict(self.world.door_state)}

    def snapshot(self) -> Dict[str, Any]:
        return {"world": self.world.snapshot(),
                "agents": {n: a.snapshot() for n, a in self.agents.items()},
                "seed": self.seed}

    def restore(self, snap: Dict[str, Any]) -> None:
        if snap.get("seed") != self.seed:
            raise ValueError("world backup belongs to a different seed")
        self.world.restore(snap["world"])
        for n, a in snap["agents"].items():
            if n in self.agents:
                self.agents[n].restore(a)


def get_service() -> WorldService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = WorldService(seed=int(os.environ.get("FLYBRAIN_WORLD_SEED", "7")))
    return _SERVICE


def reset_service(seed: int = 7, friends: int = 2) -> WorldService:
    global _SERVICE
    if _SERVICE is not None:
        try:
            _SERVICE.spatial.close()
        except Exception:
            pass
    _SERVICE = WorldService(seed=seed, friends=friends)
    return _SERVICE
