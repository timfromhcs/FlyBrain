import unittest
import os
import struct
import tempfile
import shutil
import numpy as np

from src.assets.mesh_generator import MeshGenerator, MeshData
from src.assets.compiler.compiler import AssetCompiler, to_portable_relpath
from src.assets.compiler.types import AssetClass
from src.assets.resolver import AssetResolver


class TestV10AssetPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="flybrain_asset_test_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_mesh_generation_and_validation(self):
        mesh = MeshGenerator.generate_bridge(length_m=4.0, width_m=1.5)
        val = mesh.validate()
        self.assertTrue(val.is_valid, f"Mesh failed validation: {val.errors}")
        self.assertEqual(val.degenerate_count, 0)
        self.assertTrue(val.is_manifold)
        self.assertGreater(val.vertex_count, 20)
        self.assertGreater(val.face_count, 20)

    def test_glb_binary_export(self):
        mesh = MeshGenerator.generate_bridge(length_m=4.0, width_m=1.5)
        glb_path = os.path.join(self.test_dir, "bridge_test.glb")
        mesh.export_glb(glb_path)
        
        self.assertTrue(os.path.exists(glb_path))
        self.assertGreater(os.path.getsize(glb_path), 100)
        
        # Verify GLB Header
        with open(glb_path, "rb") as f:
            header = f.read(12)
            magic, version, length = struct.unpack("<4sII", header)
            self.assertEqual(magic, b"glTF")
            self.assertEqual(version, 2)
            self.assertEqual(length, os.path.getsize(glb_path))

    def test_degenerate_triangle_detection(self):
        # Degenerate 1: duplicate vertex index
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 0, 1]], dtype=np.int32)
        m_dup = MeshData(verts, faces)
        val_dup = m_dup.validate()
        self.assertFalse(val_dup.is_valid)
        self.assertGreaterEqual(val_dup.degenerate_count, 1)

        # Degenerate 2: collinear vertices (zero area)
        verts_col = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        faces_col = np.array([[0, 1, 2]], dtype=np.int32)
        m_col = MeshData(verts_col, faces_col)
        val_col = m_col.validate()
        self.assertFalse(val_col.is_valid)
        self.assertGreaterEqual(val_col.degenerate_count, 1)

    def test_asset_compiler_portable_paths_and_authoritative_physics(self):
        mesh = MeshGenerator.generate_bridge(length_m=4.0, width_m=1.5)
        obj_path = os.path.join(self.test_dir, "bridge.obj")
        glb_path = os.path.join(self.test_dir, "bridge.glb")
        mesh.export_obj(obj_path)
        mesh.export_glb(glb_path)

        compiler = AssetCompiler(output_dir=self.test_dir)
        asset = compiler.compile_asset(
            semantic_name="wooden_bridge",
            category=AssetClass.STRUCTURE,
            visual_mesh_path=obj_path,
            visual_glb_path=glb_path,
            custom_dimensions=[4.0, 1.5, 0.18]
        )

        self.assertIn("wooden_bridge_", asset.asset_id)
        # Check portable paths
        self.assertFalse(asset.visual_mesh.startswith("E:\\"))
        self.assertFalse(asset.visual_mesh.startswith("C:\\"))
        self.assertIn("/", asset.visual_mesh)
        # Check authoritative physics
        self.assertGreater(asset.physics["mass"], 10.0)
        self.assertEqual(len(asset.physics["friction"]), 3)
        self.assertFalse(asset.physics["dynamic"])


if __name__ == "__main__":
    unittest.main()
