"""Local SD1.5 + LCM image adapter (V6 phase_14).

Single-file DreamShaper-8-LCM checkpoint, swappable via config
(any SD1.5-compatible .safetensors works: pass checkpoint=...).
Modes: FAST (512px/LCM/4 steps), BALANCED (512/8), QUALITY (768/12).
CPU-first (no CUDA requirement); every generation records full provenance.
"""
import hashlib
import os
import time
from typing import Any, Dict, Optional

MODES = {
    "FAST": {"size": 512, "steps": 4},
    "BALANCED": {"size": 512, "steps": 8},
    "QUALITY": {"size": 768, "steps": 12},
}


def find_checkpoint(explicit: Optional[str] = None) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    d = os.path.join("models", "image_model")
    for fn in ("DreamShaper8_LCM.safetensors",):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    # any single-file sd1.5 checkpoint
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors"):
                return os.path.join(d, fn)
    raise FileNotFoundError("no local SD checkpoint (run scripts/acquire_models.py image)")


class LocalImageModel:
    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = find_checkpoint(checkpoint)
        h = hashlib.sha256()
        with open(self.checkpoint, "rb") as f:
            for c in iter(lambda: f.read(65536), b""):
                h.update(c)
        self.sha256 = h.hexdigest()
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import StableDiffusionPipeline, LCMScheduler
        pipe = StableDiffusionPipeline.from_single_file(
            self.checkpoint, torch_dtype=torch.float32)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        self._pipe = pipe
        return pipe

    def unload(self):
        self._pipe = None
        import gc
        gc.collect()

    def generate(self, prompt: str, mode: str = "FAST", seed: int = 42,
                 out_path: Optional[str] = None) -> Dict[str, Any]:
        import torch
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        cfg = MODES[mode]
        pipe = self.load()
        gen = torch.Generator("cpu").manual_seed(int(seed))
        t0 = time.time()
        image = pipe(prompt, height=cfg["size"], width=cfg["size"],
                     num_inference_steps=cfg["steps"],
                     guidance_scale=1.0, generator=gen).images[0]
        dt = time.time() - t0
        if out_path is None:
            os.makedirs(os.path.join("visual_evidence", "imagined"), exist_ok=True)
            out_path = os.path.join("visual_evidence", "imagined",
                                    f"gen_{int(time.time())}_{seed}.png")
        image.save(out_path)
        assert os.path.exists(out_path) and os.path.getsize(out_path) > 10000
        opened = image.size == (cfg["size"], cfg["size"])
        assert opened, f"bad dimensions {image.size}"
        return {"status": "GENERATED", "path": out_path,
                "bytes": os.path.getsize(out_path),
                "mode": mode, "size": cfg["size"], "steps": cfg["steps"],
                "seed": seed, "seconds": round(dt, 1),
                "checkpoint": os.path.basename(self.checkpoint),
                "checkpoint_sha256": self.sha256,
                "provenance": "GENERATED_IMAGE"}
