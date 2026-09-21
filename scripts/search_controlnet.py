#!/usr/bin/env python3
"""Search HF Hub for SD1.5-compatible ControlNet models (V6 plan update input)."""
import sys

CANDIDATES = [
    "lllyasviel/sd-controlnet-canny",
    "lllyasviel/sd-controlnet-depth",
    "lllyasviel/sd-controlnet-openpose",
    "lllyasviel/sd-controlnet-seg",
    "lllyasviel/sd-controlnet-normal",
]


def main():
    from huggingface_hub import model_info
    print(f"{'model':45} {'files':6} {'has_diffusion_pytorch_model':28} {'sha-pinned'}")
    for mid in CANDIDATES:
        try:
            info = model_info(mid)
            sibs = [s.rfilename for s in info.siblings]
            has_dpm = "diffusion_pytorch_model.safetensors" in sibs
            print(f"{mid:45} {len(sibs):<6} {str(has_dpm):28} {info.sha[:12]}")
        except Exception as e:
            print(f"{mid:45} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
