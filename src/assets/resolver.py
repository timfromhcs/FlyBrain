"""Asset Resolver & Cache Manager (V8/V9).

Implements Sections 47, 48, 90:
- Resolves requested assets by checking cache/database before generation.
- Generates missing assets via image synthesis -> 3D mesh -> FlyAsset compilation.
- Content-addressed hashing avoiding redundant generation.
"""
import os
import json
import hashlib
from typing import Dict, Any, Optional

from src.assets.compiler.types import AssetClass, FlyAsset
from src.assets.compiler.compiler import AssetCompiler
from src.assets.mesh_generator import MeshGenerator


class AssetResolver:
    def __init__(self, cache_dir: str = "assets/compiled", image_cache_dir: str = "visual_evidence/imagined"):
        self.cache_dir = os.path.abspath(cache_dir)
        self.image_cache_dir = os.path.abspath(image_cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.image_cache_dir, exist_ok=True)
        self.compiler = AssetCompiler(output_dir=self.cache_dir)

    def compute_content_id(self, semantic_request: str, params: Dict[str, Any], seed: int = 42) -> str:
        """Section 48: SHA256(semantic_request + model_revision + seed + params)."""
        content = f"{semantic_request}_v9.0_{seed}_{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def resolve_asset(self,
                      semantic_name: str,
                      category: AssetClass = AssetClass.STRUCTURE,
                      params: Optional[Dict[str, Any]] = None,
                      seed: int = 42,
                      force_regenerate: bool = False) -> Dict[str, Any]:
        params = params or {}
        cid = self.compute_content_id(semantic_name, params, seed=seed)
        target_manifest = os.path.join(self.cache_dir, f"{semantic_name}_{cid}.json")

        # 1. Check Cache (Section 47)
        if not force_regenerate and os.path.exists(target_manifest):
            try:
                with open(target_manifest, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                return {
                    "status": "CACHE_HIT",
                    "asset_id": manifest.get("asset_id"),
                    "manifest_path": target_manifest,
                    "visual_mesh": manifest.get("visual_mesh"),
                    "dimensions_m": manifest.get("dimensions_m"),
                    "collision": manifest.get("collision"),
                    "category": category.value
                }
            except Exception:
                pass

        # 2. Cache Miss: Generate 3D geometry & FlyAsset package
        mesh_path = os.path.join(self.cache_dir, f"{semantic_name}_{cid}.obj")
        
        if "bridge" in semantic_name.lower():
            span = float(params.get("span_m", 4.0))
            width = float(params.get("width_m", 1.5))
            mesh_data = MeshGenerator.generate_bridge(length_m=span, width_m=width)
            custom_dims = [span, width, 0.18]
        else:
            sx = float(params.get("sx", 2.0))
            sy = float(params.get("sy", 2.0))
            sz = float(params.get("sz", 1.5))
            mesh_data = MeshGenerator.generate_box(sx, sy, sz)
            custom_dims = [sx, sy, sz]

        # Validate mesh
        val = mesh_data.validate()
        if not val.is_valid:
            raise RuntimeError(f"Generated 3D mesh failed validation: {val.errors}")

        # Export OBJ
        mesh_data.export_obj(mesh_path)

        # Optional: generate reference image if image model is available
        img_path = os.path.join(self.image_cache_dir, f"{semantic_name}_{cid}.png")
        try:
            from src.models.image_adapter import LocalImageModel
            img_model = LocalImageModel()
            img_model.generate(f"A detailed realistic {semantic_name}, physical 3D asset",
                               mode="FAST", seed=seed, out_path=img_path)
        except Exception:
            # When image model checkpoint is not present or in lightweight mode, write reference metadata
            with open(img_path + ".meta", "w", encoding="utf-8") as mf:
                json.dump({"semantic": semantic_name, "cid": cid, "mode": "PROCEDURAL_FALLBACK"}, mf)

        # 3. Compile Asset with Scale Normalization & Collision Proxies
        asset = self.compiler.compile_asset(
            semantic_name=semantic_name,
            category=category,
            visual_mesh_path=mesh_path,
            generator_info="FlyBrain_AssetCompiler_v9.0",
            input_hash=cid,
            custom_dimensions=custom_dims,
            explicit_asset_id=f"{semantic_name}_{cid}"
        )

        return {
            "status": "CACHE_MISS_GENERATED",
            "asset_id": asset.asset_id,
            "manifest_path": target_manifest,
            "visual_mesh": mesh_path,
            "dimensions_m": asset.dimensions_m,
            "collision": asset.collision,
            "category": category.value,
            "mesh_validation": val.to_dict()
        }
