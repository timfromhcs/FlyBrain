import unittest
import shutil
import tempfile
import os

from src.world.chunks.chunk_manager import (
    WorldManager,
    WorldGenerator,
    ChunkManager,
    WORLD_GENERATOR_REVISION
)


class TestV10MemoryPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="flybrain_chunks_test_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_chunk_modification_persistence_across_restart(self):
        # 1. Initialize WorldManager and apply modification
        wm1 = WorldManager(seed=42, storage_dir=self.test_dir)
        struct_data = {
            "id": "wooden_bridge_persist_test",
            "x": 64.0,
            "y": 160.0,
            "type": "bridge"
        }
        cx, cy = 2, 5
        event_hash = wm1.place_structure(cx, cy, struct_data)
        self.assertTrue(len(event_hash) >= 8)

        # Verify delta file written to disk
        expected_file = os.path.join(self.test_dir, f"chunk_{cx}_{cy}_deltas.json")
        self.assertTrue(os.path.exists(expected_file))

        # 2. Simulate complete process termination by deleting manager
        del wm1

        # 3. Spawn fresh WorldManager pointing to same directory
        wm2 = WorldManager(seed=42, storage_dir=self.test_dir)
        # Update center around (64, 160) which corresponds to chunk (2, 5)
        tick_info = wm2.tick(dt=0.1, agent_pos=(64.0, 160.0, 0.0))
        self.assertIn(f"{cx}_{cy}", tick_info["active_chunks"])

        loaded_chunk = wm2.chunks.loaded_chunks[f"{cx}_{cy}"]
        self.assertEqual(len(loaded_chunk.modifications), 1)
        retrieved_mod = loaded_chunk.modifications[0]
        self.assertEqual(retrieved_mod["action"], "place_structure")
        self.assertEqual(retrieved_mod["data"]["id"], "wooden_bridge_persist_test")
        self.assertEqual(retrieved_mod["event_hash"], event_hash)

    def test_chunk_streaming_bounded_cache(self):
        wm = WorldManager(seed=42, max_loaded=25, storage_dir=self.test_dir)
        # Move agent across 10 chunks distance
        for i in range(10):
            agent_x = float(i * 32.0)
            agent_y = float(i * 32.0)
            tick_info = wm.tick(dt=0.1, agent_pos=(agent_x, agent_y, 0.0))
            self.assertLessEqual(tick_info["active_chunk_count"], 25)

    def test_deterministic_terrain_hash(self):
        gen1 = WorldGenerator(seed=42, revision=WORLD_GENERATOR_REVISION)
        gen2 = WorldGenerator(seed=42, revision=WORLD_GENERATOR_REVISION)
        gen3 = WorldGenerator(seed=999, revision=WORLD_GENERATOR_REVISION)

        h1 = gen1.chunk_hash(3, 7)
        h2 = gen2.chunk_hash(3, 7)
        h3 = gen3.chunk_hash(3, 7)

        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)


if __name__ == "__main__":
    unittest.main()
