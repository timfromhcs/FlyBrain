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

CONTROLNETS = {
    "canny": {"repo": "lllyasviel/sd-controlnet-canny", "revision": "7f2f69197050",
              "dir": os.path.join("models", "controlnet", "canny")},
    "depth": {"repo": "lllyasviel/sd-controlnet-depth", "revision": "35e42a3ea498",
              "dir": os.path.join("models", "controlnet", "depth")},
    "openpose": {"repo": "lllyasviel/sd-controlnet-openpose", "revision": "df796456519d",
                 "dir": os.path.join("models", "controlnet", "openpose")},
}


def controlnet_status() -> Dict[str, Any]:
    out = {}
    for name, spec in CONTROLNETS.items():
        p = os.path.join(spec["dir"], "diffusion_pytorch_model.safetensors")
        out[name] = {"present": os.path.exists(p),
                     "repo": spec["repo"], "revision": spec["revision"],
                     "size": os.path.getsize(p) if os.path.exists(p) else None}
    return out


def edge_map(image_path: str, low_pct: float = 80.0, high_pct: float = 92.0):
    """Local Sobel-hysteresis edge map (no cv2 dependency).

    Returns a 3-channel uint8 edge image + the applied thresholds.
    Honestly labelled EDGE_MAP (Sobel), not Canny-OpenCV.
    """
    import numpy as _np
    from PIL import Image as _Image
    from scipy.ndimage import sobel as _sobel, binary_dilation as _dil
    img = _Image.open(image_path).convert("L")
    gray = _np.asarray(img, dtype=_np.float32) / 255.0
    gx, gy = _sobel(gray, axis=1), _sobel(gray, axis=0)
    mag = _np.hypot(gx, gy)
    lo, hi = _np.percentile(mag, [low_pct, high_pct])
    strong = mag >= hi
    weak = (mag >= lo) & _dil(strong)
    edges = ((strong | weak) * 255).astype(_np.uint8)
    rgb = _np.stack([edges] * 3, axis=-1)
    return _Image.fromarray(rgb), {"low": round(float(lo), 4),
                                   "high": round(float(hi), 4),
                                   "method": "sobel_hysteresis"}


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
        self._cn_pipe = None

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
        self._cn_pipe = None
        import gc
        gc.collect()

    def generate_controlled(self, prompt: str, control_image, controlnet: str = "canny",
                            mode: str = "FAST", seed: int = 42,
                            out_path: Optional[str] = None,
                            control_scale: float = 1.0) -> Dict[str, Any]:
        """ControlNet render: structure from control_image, style from prompt.

        SD1.5 ControlNet + DreamShaper weights + LCM scheduler, CPU.
        Raises FileNotFoundError when the ControlNet is absent (UNAVAILABLE).
        """
        import torch
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        spec = CONTROLNETS.get(controlnet)
        if spec is None:
            raise ValueError(f"unknown controlnet {controlnet!r}")
        ckpt = os.path.join(spec["dir"], "diffusion_pytorch_model.safetensors")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"controlnet {controlnet} absent "
                                    f"(pinned {spec['repo']}@{spec['revision']})")
        from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
                               LCMScheduler)
        cfg = MODES[mode]
        t0 = time.time()
        cnet = ControlNetModel.from_single_file(ckpt, torch_dtype=torch.float32)
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            self.checkpoint, controlnet=cnet, torch_dtype=torch.float32)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        load_s = round(time.time() - t0, 1)
        size = cfg["size"]
        ctrl = control_image.resize((size, size)).convert("RGB")
        gen = torch.Generator("cpu").manual_seed(int(seed))
        t0 = time.time()
        image = pipe(prompt, image=ctrl, height=size, width=size,
                     num_inference_steps=cfg["steps"], guidance_scale=1.0,
                     controlnet_conditioning_scale=float(control_scale),
                     generator=gen).images[0]
        dt = time.time() - t0
        if out_path is None:
            os.makedirs(os.path.join("visual_evidence", "imagined"), exist_ok=True)
            out_path = os.path.join("visual_evidence", "imagined",
                                    f"cn_{controlnet}_{int(time.time())}_{seed}.png")
        image.save(out_path)
        assert os.path.exists(out_path) and os.path.getsize(out_path) > 10000
        assert image.size == (size, size)
        del pipe, cnet
        import gc
        gc.collect()
        return {"status": "GENERATED", "path": out_path,
                "bytes": os.path.getsize(out_path), "mode": mode,
                "controlnet": controlnet, "control_repo": spec["repo"],
                "control_revision": spec["revision"],
                "controlnet_load_s": load_s,
                "size": size, "steps": cfg["steps"], "seed": seed,
                "seconds": round(dt, 1),
                "checkpoint": os.path.basename(self.checkpoint),
                "checkpoint_sha256": self.sha256,
                "provenance": "GENERATED_IMAGE_CONTROLLED"}

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
