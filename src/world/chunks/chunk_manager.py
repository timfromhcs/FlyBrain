"""Deterministic Chunk-Streamed Endless World Engine for FlyBrain V10.

Implements Phase 4 & Phase 5 Requirements:
- WorldManager
- ChunkManager
- WorldGenerator (terrain, biomes, procedural structures)
- Bounded RAM streaming with persistent modification overlays on disk
- Deterministic formula: hash(world_seed, cx, cy, generator_revision)
- Complete survival of chunk modifications across process restart and reload
"""
import os
import math
import time
import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

WORLD_GENERATOR_REVISION = "v10.0"
ASSET_SCHEMA_VERSION = 1


class BiomeType:
    TEMPERATE_VALLEY = "temperate_valley"
    PINE_FOREST = "pine_forest"
    RIVER_BASIN = "river_basin"
    ROCKY_HIGHLANDS = "rocky_highlands"
    SAVANNA_PLAINS = "savanna_plains"


class Chunk:
    def __init__(self, cx: int, cy: int, size_m: float = 32.0, resolution: int = 32):
        self.cx = cx
        self.cy = cy
        self.size_m = size_m
        self.resolution = resolution
        self.biome: str = BiomeType.TEMPERATE_VALLEY
        self.heightmap: np.ndarray = np.zeros((resolution, resolution), dtype=np.float32)
        self.objects: List[Dict[str, Any]] = []
        self.modifications: List[Dict[str, Any]] = []
        self.is_loaded: bool = False
        self.last_accessed: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.cx}_{self.cy}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "size_m": self.size_m,
            "biome": self.biome,
            "height_min": float(self.heightmap.min()),
            "height_max": float(self.heightmap.max()),
            "object_count": len(self.objects),
            "modifications": self.modifications
        }


class WorldGenerator:
    """Deterministic terrain and structure synthesizer."""
    def __init__(self, seed: int = 42, revision: str = WORLD_GENERATOR_REVISION):
        self.seed = seed
        self.revision = revision

    def chunk_hash(self, cx: int, cy: int) -> str:
        return hashlib.sha256(f"{self.seed}_{cx}_{cy}_{self.revision}".encode("utf-8")).hexdigest()

    def _hash_coords(self, cx: int, cy: int) -> int:
        h = self.chunk_hash(cx, cy)
        return int(h[:8], 16)

    def generate_chunk(self, cx: int, cy: int, chunk_size: float = 32.0) -> Chunk:
        chunk = Chunk(cx, cy, size_m=chunk_size)
        rng = np.random.RandomState(self._hash_coords(cx, cy))
        
        # Biome determination from macro-coordinates
        macro = math.sin(cx * 0.15) + math.cos(cy * 0.15)
        if macro > 0.8:
            chunk.biome = BiomeType.ROCKY_HIGHLANDS
            base_h = 4.0
        elif macro < -0.8:
            chunk.biome = BiomeType.RIVER_BASIN
            base_h = -1.5
        elif macro > 0.2:
            chunk.biome = BiomeType.PINE_FOREST
            base_h = 1.0
        else:
            chunk.biome = BiomeType.TEMPERATE_VALLEY
            base_h = 0.0

        # Deterministic perlin-like smooth procedural terrain
        res = chunk.resolution
        x = np.linspace(cx * chunk_size, (cx + 1) * chunk_size, res)
        y = np.linspace(cy * chunk_size, (cy + 1) * chunk_size, res)
        xv, yv = np.meshgrid(x, y)
        
        h = base_h + 1.2 * np.sin(xv * 0.08) * np.cos(yv * 0.08)
        h += 0.4 * np.sin(xv * 0.2 + yv * 0.1)
        chunk.heightmap = h.astype(np.float32)

        # Procedural natural props (rocks, trees)
        num_trees = rng.randint(2, 8) if chunk.biome == BiomeType.PINE_FOREST else rng.randint(0, 3)
        for i in range(num_trees):
            tx = float(rng.uniform(cx * chunk_size + 2.0, (cx + 1) * chunk_size - 2.0))
            ty = float(rng.uniform(cy * chunk_size + 2.0, (cy + 1) * chunk_size - 2.0))
            chunk.objects.append({
                "id": f"tree_{cx}_{cy}_{i}",
                "type": "vegetation",
                "subtype": "pine_tree",
                "pos": [round(tx, 2), round(ty, 2), round(float(np.interp(tx, x, h[:, 0])), 2)],
                "scale": [1.0, 1.0, round(float(rng.uniform(2.5, 4.5)), 2)]
            })

        chunk.is_loaded = True
        return chunk


class ChunkManager:
    """Manages streaming radius, bounded RAM cache, and persistent delta overlays on disk."""
    def __init__(self, world_gen: WorldGenerator, max_loaded_chunks: int = 25, storage_dir: str = "data/world_chunks"):
        self.generator = world_gen
        self.max_loaded = max_loaded_chunks
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.loaded_chunks: Dict[str, Chunk] = {}
        self.modifications_db: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all_persisted_deltas()

    def _delta_file_path(self, key: str) -> str:
        return os.path.join(self.storage_dir, f"chunk_{key}_deltas.json")

    def _load_all_persisted_deltas(self) -> None:
        """Loads all serialized chunk deltas from storage_dir into memory cache."""
        if not os.path.exists(self.storage_dir):
            return
        for fname in os.listdir(self.storage_dir):
            if fname.startswith("chunk_") and fname.endswith("_deltas.json"):
                k = fname[len("chunk_"):-len("_deltas.json")]
                p = os.path.join(self.storage_dir, fname)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        deltas = json.load(f)
                        if isinstance(deltas, list):
                            self.modifications_db[k] = deltas
                except Exception:
                    pass

    def _save_chunk_deltas(self, key: str) -> None:
        """Serializes chunk deltas atomically to disk."""
        deltas = self.modifications_db.get(key, [])
        p = self._delta_file_path(key)
        tmp_p = f"{p}.tmp"
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(deltas, f, indent=2)
        if os.path.exists(p):
            os.remove(p)
        os.rename(tmp_p, p)

    def update_center(self, agent_x: float, agent_y: float, radius_chunks: int = 2) -> List[str]:
        """Loads chunks within radius and streams out far chunks to maintain bounded RAM."""
        center_cx = int(math.floor(agent_x / 32.0))
        center_cy = int(math.floor(agent_y / 32.0))
        
        needed_keys = set()
        now = time.time()
        for dx in range(-radius_chunks, radius_chunks + 1):
            for dy in range(-radius_chunks, radius_chunks + 1):
                cx = center_cx + dx
                cy = center_cy + dy
                k = f"{cx}_{cy}"
                needed_keys.add(k)
                if k not in self.loaded_chunks:
                    chunk = self.generator.generate_chunk(cx, cy)
                    # Apply persistent modifications if existing
                    if k in self.modifications_db:
                        chunk.modifications = list(self.modifications_db[k])
                    chunk.last_accessed = now
                    self.loaded_chunks[k] = chunk
                else:
                    self.loaded_chunks[k].last_accessed = now

        # Evict unneeded chunks beyond max limit
        while len(self.loaded_chunks) > self.max_loaded:
            candidates = [k for k in self.loaded_chunks if k not in needed_keys]
            if not candidates:
                break
            lru_key = min(candidates, key=lambda k: self.loaded_chunks[k].last_accessed)
            self._save_and_unload(lru_key)

        return list(self.loaded_chunks.keys())

    def apply_modification(self, cx: int, cy: int, mod: Dict[str, Any]) -> str:
        """Applies persistent world modification and writes to disk immediately."""
        k = f"{cx}_{cy}"
        mod_copy = dict(mod)
        if "timestamp" not in mod_copy:
            mod_copy["timestamp"] = time.time()
        event_hash = hashlib.sha256(json.dumps(mod_copy, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        mod_copy["event_hash"] = event_hash

        if k not in self.modifications_db:
            self.modifications_db[k] = []
        self.modifications_db[k].append(mod_copy)
        
        if k in self.loaded_chunks:
            self.loaded_chunks[k].modifications.append(mod_copy)

        self._save_chunk_deltas(k)
        return event_hash

    def get_deltas(self, cx: int, cy: int) -> List[Dict[str, Any]]:
        k = f"{cx}_{cy}"
        return list(self.modifications_db.get(k, []))

    def _save_and_unload(self, key: str) -> None:
        chunk = self.loaded_chunks.pop(key, None)
        if chunk and chunk.modifications:
            self.modifications_db[key] = chunk.modifications
            self._save_chunk_deltas(key)


class WorldManager:
    """Master High-Level World Orchestrator."""
    def __init__(self, seed: int = 42, max_loaded: int = 25, storage_dir: str = "data/world_chunks"):
        self.seed = seed
        self.generator = WorldGenerator(seed=seed, revision=WORLD_GENERATOR_REVISION)
        self.chunks = ChunkManager(self.generator, max_loaded_chunks=max_loaded, storage_dir=storage_dir)
        self.world_time_sec: float = 0.0

    def tick(self, dt: float, agent_pos: Tuple[float, float, float]) -> Dict[str, Any]:
        self.world_time_sec += dt
        active_chunks = self.chunks.update_center(agent_pos[0], agent_pos[1])
        return {
            "world_time": round(self.world_time_sec, 2),
            "active_chunk_count": len(active_chunks),
            "active_chunks": active_chunks,
            "generator_revision": self.generator.revision
        }

    def place_structure(self, cx: int, cy: int, structure_data: Dict[str, Any]) -> str:
        return self.chunks.apply_modification(cx, cy, {
            "action": "place_structure",
            "data": structure_data
        })
