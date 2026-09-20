import os
try:
    import torch
    from diffusers import AutoencoderTiny
except ImportError:  # CPU-only / minimal hosts (e.g. HF Space): VAE path disabled
    torch = None
    AutoencoderTiny = None
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from src.tools.base import ToolConnector

# R9: resolution order env > project-local models/ dir > remote cache > procedural.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_VAE_PATH = os.environ.get(
    "FLYBRAIN_VAE_PATH",
    os.path.join(_PROJECT_ROOT, "models", "dreamlite-vae"))
REMOTE_VAE_ID = "madebyollin/taesd"

class GenerateImageConnector(ToolConnector):
    def __init__(self, output_dir: str = "visual_evidence/images"):
        super().__init__(
            name="generate_image",
            description="Generates 256x256 visual imagery (local/remote AutoencoderTiny VAE when available, otherwise deterministic procedural rendering; renderer reported per call).",
            timeout_sec=20.0
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.vae = None
        self.renderer = "procedural"
        self._load_vae()

    def _load_vae(self):
        if torch is None or AutoencoderTiny is None:
            print("[GenerateImage] torch/diffusers unavailable: procedural rendering only.")
            self.vae = None
            self.renderer = "procedural"
            return
        for candidate, tag in ((LOCAL_VAE_PATH, "vae_local"), (REMOTE_VAE_ID, "vae_remote")):
            try:
                self.vae = AutoencoderTiny.from_pretrained(candidate)
                self.vae.eval()
                self.renderer = tag
                return
            except Exception as e:
                print(f"[GenerateImage] VAE source unavailable ({candidate}): {e}")
        self.vae = None
        self.renderer = "procedural"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "seed": {"type": "integer", "default": 42},
                "latent_mod": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["prompt"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "mean_luminance": {"type": "number"}
            },
            "required": ["image_path", "width", "height", "mean_luminance"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        prompt = str(params["prompt"])
        seed = int(params.get("seed", 42))
        
        # Deterministic PRNG from seed + prompt hash
        prompt_hash = sum(ord(c) for c in prompt)
        latent = None
        if torch is not None:
            torch.manual_seed(seed + prompt_hash)

            # 4-channel latent for AutoencoderTiny: [1, 4, 32, 32]
            latent = torch.randn(1, 4, 32, 32, dtype=torch.float32)

            # Modulate if brain latent provided
            if "latent_mod" in params and params["latent_mod"]:
                mod_arr = np.asarray(params["latent_mod"], dtype=np.float32).flatten()
                mod_len = min(len(mod_arr), 4 * 32 * 32)
                latent_flat = latent.view(-1)
                latent_flat[:mod_len] += torch.from_numpy(mod_arr[:mod_len])
                latent = latent_flat.view(1, 4, 32, 32)

        if self.vae is not None:
            with torch.no_grad():
                decoded = self.vae.decode(latent).sample

            # Convert [-1, 1] tensor to [0, 255] uint8 image
            img_np = decoded.squeeze(0).permute(1, 2, 0).clamp(-1.0, 1.0).numpy()
            img_np = ((img_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
        else:
            # High-resolution neural procedural rendering
            grid_y, grid_x = np.mgrid[0:256, 0:256]
            freq = 0.05 + 0.02 * (seed % 5)
            r = np.sin(grid_x * freq + prompt_hash * 0.1) * 0.5 + 0.5
            g = np.cos(grid_y * freq + seed * 0.2) * 0.5 + 0.5
            b = np.sin((grid_x + grid_y) * freq * 0.5) * 0.5 + 0.5
            img_np = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

        img = Image.fromarray(img_np)
        img_filename = f"gen_{execution_id[:8]}.png"
        img_path = os.path.abspath(os.path.join(self.output_dir, img_filename))
        img.save(img_path)

        mean_lum = float(np.mean(img_np) / 255.0)

        return {
            "image_path": img_path,
            "width": img.width,
            "height": img.height,
            "mean_luminance": round(mean_lum, 4),
            "renderer": self.renderer
        }
