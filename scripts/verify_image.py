#!/usr/bin/env python3
"""V6 image_generation_verification: real local DreamShaper-LCM FAST render.

Asserts: checkpoint loads, image generates, file validates (decodes,
512x512), provenance recorded. Benchmarks load + render. OOM/exceptions are
recorded honestly (never fake success). Writes
diagnostics/v6/image_generation_verification.json.
"""
import json
import os
import subprocess
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "image_generation_verification.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main():
    from src.version import VERSION
    rep = {"timestamp": time.time(), "version": VERSION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip()}
    try:
        import psutil
        rep["ram_total_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    try:
        from src.models.image_adapter import LocalImageModel
        t0 = time.time()
        model = LocalImageModel()
        rep["checkpoint"] = os.path.basename(model.checkpoint)
        rep["checkpoint_sha256"] = model.sha256
        rep["model_id"] = "Lykon/dreamshaper-8-lcm"
        model.load()
        rep["load_s"] = round(time.time() - t0, 1)
        out = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "genesis_test.png")
        res = model.generate("a small cabin in a pine forest at dusk, painting",
                             mode="FAST", seed=7, out_path=out)
        rep.update({k: res[k] for k in ("status", "path", "bytes", "mode", "size",
                                        "steps", "seed", "seconds", "provenance")})
        rep["path"] = os.path.relpath(res["path"], PROJECT_ROOT)
        from PIL import Image
        img = Image.open(out)
        img.load()
        rep["decoded"] = {"format": img.format, "size": list(img.size), "mode": img.mode}
        assert list(img.size) == [512, 512]
        model.unload()
    except Exception as e:  # noqa: BLE001
        rep.update({"status": "FAIL" if "Memory" not in type(e).__name__ else "UNAVAILABLE",
                    "error": f"{type(e).__name__}: {str(e)[:400]}",
                    "trace": traceback.format_exc()[-1500:]})
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"IMAGE {rep.get('status')}: load={rep.get('load_s')}s "
          f"render={rep.get('seconds')}s bytes={rep.get('bytes')}")


if __name__ == "__main__":
    main()
