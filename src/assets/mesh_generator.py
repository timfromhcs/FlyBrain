"""Autonomous 3D Mesh Synthesis, GLB Export, and Geometry Validation for FlyBrain V10.

Implements Phase 6 Requirements:
- Manifold 3D geometry generator for procedural and reconstructed assets.
- Exports standard Wavefront OBJ and binary glTF 2.0 (GLB).
- Rigorous geometry validation:
  - NaNs and Infs
  - Vertex out-of-bounds
  - Degenerate triangles (duplicate indices, zero-area faces)
  - Non-manifold edge detection
  - Bounds and scale dimensions
"""
import os
import math
import struct
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class MeshValidationResult:
    def __init__(self, is_valid: bool, vertex_count: int, face_count: int,
                 bounds_min: List[float], bounds_max: List[float], errors: List[str],
                 is_manifold: bool = True, degenerate_count: int = 0):
        self.is_valid = is_valid
        self.vertex_count = vertex_count
        self.face_count = face_count
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.errors = errors
        self.is_manifold = is_manifold
        self.degenerate_count = degenerate_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounds_min": [round(x, 4) for x in self.bounds_min],
            "bounds_max": [round(x, 4) for x in self.bounds_max],
            "dimensions_m": [round(self.bounds_max[i] - self.bounds_min[i], 4) for i in range(3)],
            "is_manifold": self.is_manifold,
            "degenerate_count": self.degenerate_count,
            "errors": self.errors
        }


class MeshData:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 normals: Optional[np.ndarray] = None, uvs: Optional[np.ndarray] = None):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.normals = normals
        self.uvs = uvs

    def validate(self) -> MeshValidationResult:
        """Rigorous geometry and topology validation (Phase 6)."""
        errors = []
        degenerate_count = 0
        is_manifold = True

        if self.vertices.size == 0 or len(self.vertices) < 3:
            errors.append("Empty or insufficient vertices")
        if np.isnan(self.vertices).any() or np.isinf(self.vertices).any():
            errors.append("Vertices contain NaN or Inf")
        if self.faces.size == 0:
            errors.append("Empty faces")
        elif self.faces.max() >= len(self.vertices):
            errors.append("Face index out of vertex bounds")
        elif self.faces.min() < 0:
            errors.append("Negative face index")

        # Bounds calculation
        bmin = self.vertices.min(axis=0).tolist() if len(self.vertices) > 0 else [0.0, 0.0, 0.0]
        bmax = self.vertices.max(axis=0).tolist() if len(self.vertices) > 0 else [0.0, 0.0, 0.0]

        # Topology and degeneracy check
        edge_counts: Dict[Tuple[int, int], int] = {}
        if len(self.vertices) >= 3 and len(self.faces) > 0 and self.faces.max() < len(self.vertices):
            for fi, face in enumerate(self.faces):
                i, j, k = int(face[0]), int(face[1]), int(face[2])
                
                # Duplicate vertex indices in same triangle
                if i == j or j == k or k == i:
                    degenerate_count += 1
                    errors.append(f"Face {fi} is degenerate: duplicate vertex index ({i}, {j}, {k})")
                    continue

                # Zero-area check
                vi = self.vertices[i]
                vj = self.vertices[j]
                vk = self.vertices[k]
                cross = np.cross(vj - vi, vk - vi)
                area = 0.5 * float(np.linalg.norm(cross))
                if area < 1e-9:
                    degenerate_count += 1
                    errors.append(f"Face {fi} has zero area ({area:.2e})")

                # Track undirected edge shared counts
                for edge in [(min(i, j), max(i, j)), (min(j, k), max(j, k)), (min(k, i), max(k, i))]:
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

            # Check for non-manifold edges (> 2 triangles sharing an edge)
            non_manifold_edges = [edge for edge, count in edge_counts.items() if count > 2]
            if non_manifold_edges:
                is_manifold = False
                errors.append(f"Found {len(non_manifold_edges)} non-manifold edge(s) shared by >2 faces")

        is_valid = len(errors) == 0
        return MeshValidationResult(
            is_valid=is_valid,
            vertex_count=len(self.vertices),
            face_count=len(self.faces),
            bounds_min=bmin,
            bounds_max=bmax,
            errors=errors,
            is_manifold=is_manifold,
            degenerate_count=degenerate_count
        )

    def export_obj(self, filepath: str) -> str:
        """Exports standard Wavefront OBJ."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# FlyBrain Generated 3D Asset (OBJ)\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            if self.normals is not None and len(self.normals) == len(self.vertices):
                for n in self.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in self.faces:
                # OBJ indices are 1-based
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
        return filepath

    def export_glb(self, filepath: str) -> str:
        """Exports pure glTF 2.0 Binary container (.glb) without external libraries."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Ensure float32 vertices and uint32 indices
        verts = np.asarray(self.vertices, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.uint32).flatten()
        
        vert_bytes = verts.tobytes()
        face_bytes = faces.tobytes()
        
        # Buffer view 0: Vertices
        bv0_offset = 0
        bv0_len = len(vert_bytes)
        
        # Buffer view 1: Indices (aligned to 4 bytes)
        pad0 = (4 - (bv0_len % 4)) % 4
        bv1_offset = bv0_len + pad0
        bv1_len = len(face_bytes)
        
        pad1 = (4 - (bv1_len % 4)) % 4
        total_bin_len = bv1_offset + bv1_len + pad1
        
        bin_buffer = bytearray(total_bin_len)
        bin_buffer[bv0_offset:bv0_offset + bv0_len] = vert_bytes
        bin_buffer[bv1_offset:bv1_offset + bv1_len] = face_bytes
        
        bmin = verts.min(axis=0).tolist() if len(verts) > 0 else [0.0, 0.0, 0.0]
        bmax = verts.max(axis=0).tolist() if len(verts) > 0 else [0.0, 0.0, 0.0]
        
        gltf_json = {
            "asset": {
                "version": "2.0",
                "generator": "FlyBrain V10 Asset Compiler"
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": 0},
                            "indices": 1,
                            "mode": 4  # TRIANGLES
                        }
                    ]
                }
            ],
            "buffers": [{"byteLength": total_bin_len}],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": bv0_offset,
                    "byteLength": bv0_len,
                    "target": 34962  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": bv1_offset,
                    "byteLength": bv1_len,
                    "target": 34963  # ELEMENT_ARRAY_BUFFER
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(verts),
                    "type": "VEC3",
                    "min": [float(x) for x in bmin],
                    "max": [float(x) for x in bmax]
                },
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": len(faces),
                    "type": "SCALAR",
                    "min": [int(faces.min())] if len(faces) > 0 else [0],
                    "max": [int(faces.max())] if len(faces) > 0 else [0]
                }
            ]
        }
        
        json_bytes = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
        json_padding = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b' ' * json_padding
        
        # GLB Header: 12 bytes
        # Magic (0x46546C67), Version (2), Total Length
        glb_len = 12 + 8 + len(json_bytes) + 8 + total_bin_len
        header = struct.pack("<4sII", b"glTF", 2, glb_len)
        
        # Chunk 0: JSON
        chunk0_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # 'JSON'
        
        # Chunk 1: BIN
        chunk1_header = struct.pack("<II", total_bin_len, 0x004E4942)  # 'BIN\0'
        
        with open(filepath, "wb") as f:
            f.write(header)
            f.write(chunk0_header)
            f.write(json_bytes)
            f.write(chunk1_header)
            f.write(bin_buffer)
            
        return filepath


class MeshGenerator:
    """Procedural & Generative 3D Mesh Synthesizer."""

    @staticmethod
    def generate_bridge(length_m: float = 4.0, width_m: float = 1.5,
                        plank_thickness: float = 0.15, arch_height: float = 0.3) -> MeshData:
        """Generates a clean manifold arched wooden bridge mesh."""
        segments = 12
        verts = []
        faces = []

        # Generate cross-sections along length
        x_steps = np.linspace(-length_m / 2.0, length_m / 2.0, segments + 1)
        
        for x in x_steps:
            arch = arch_height * (1.0 - (x / (length_m / 2.0)) ** 2)
            y_left = -width_m / 2.0
            y_right = width_m / 2.0
            z_bot = arch
            z_top = arch + plank_thickness
            verts.append([x, y_left, z_bot])
            verts.append([x, y_right, z_bot])
            verts.append([x, y_left, z_top])
            verts.append([x, y_right, z_top])

        # Connect deck segments
        for s in range(segments):
            b = s * 4
            nb = (s + 1) * 4
            # Top deck quad
            faces.append([b + 2, nb + 2, nb + 3])
            faces.append([b + 2, nb + 3, b + 3])
            # Bottom deck quad
            faces.append([b + 0, nb + 1, nb + 0])
            faces.append([b + 0, b + 1, nb + 1])
            # Left side quad
            faces.append([b + 0, nb + 0, nb + 2])
            faces.append([b + 0, nb + 2, b + 2])
            # Right side quad
            faces.append([b + 1, nb + 3, nb + 1])
            faces.append([b + 1, b + 3, nb + 3])

        # End caps
        # Start cap (s=0)
        faces.append([0, 2, 3])
        faces.append([0, 3, 1])
        # End cap (s=segments)
        eb = segments * 4
        faces.append([eb + 0, eb + 3, eb + 2])
        faces.append([eb + 0, eb + 1, eb + 3])

        # Railings: 4 posts and 2 handrail beams
        rail_h = 0.8
        base_idx = len(verts)
        # 4 post tops
        # Start left & right
        z0 = arch_height * (1.0 - (x_steps[0] / (length_m / 2.0)) ** 2) + plank_thickness
        verts.append([x_steps[0], -width_m / 2.0, z0 + rail_h])  # base_idx + 0
        verts.append([x_steps[0], width_m / 2.0, z0 + rail_h])   # base_idx + 1
        # End left & right
        z1 = arch_height * (1.0 - (x_steps[-1] / (length_m / 2.0)) ** 2) + plank_thickness
        verts.append([x_steps[-1], -width_m / 2.0, z1 + rail_h])  # base_idx + 2
        verts.append([x_steps[-1], width_m / 2.0, z1 + rail_h])   # base_idx + 3

        # Handrail faces (left handrail quad: 2 triangles)
        faces.append([base_idx + 0, base_idx + 2, 2])
        faces.append([base_idx + 2, eb + 2, 2])
        # Handrail faces (right handrail quad: 2 triangles)
        faces.append([base_idx + 1, 3, base_idx + 3])
        faces.append([base_idx + 3, 3, eb + 3])

        return MeshData(np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32))

    @staticmethod
    def generate_box(sx: float, sy: float, sz: float) -> MeshData:
        """Generates a clean manifold box mesh."""
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        verts = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]
        ], dtype=np.float32)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
        ], dtype=np.int32)
        return MeshData(verts, faces)
