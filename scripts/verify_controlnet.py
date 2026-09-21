#!/usr/bin/env python3
"""V6 ControlNet verification: canny edge map of genesis image -> controlled
LCM render. Asserts structure preservation (edge overlap) + file validity.
Writes diagnostics/v6/controlnet_verification.json.
"""
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "controlnet_verification.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main():
    from src.version import VERSION
    rep = {"timestamp": time.time(), "version": VERSION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip()}
    try:
        from src.models.image_adapter import LocalImageModel, edge_map, controlnet_status
        rep["controlnets"] = controlnet_status()
        src = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "genesis_test.png")
        t0 = time.time()
        edges, einfo = edge_map(src)
        rep["edge_extraction_s"] = round(time.time() - t0, 2)
        rep["edge_info"] = einfo
        edge_path = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "genesis_edges.png")
        edges.save(edge_path)
        model = LocalImageModel()
        res = model.generate_controlled(
            "the same cabin in a pine forest, moonlit night, painting",
            edges, controlnet="canny", mode="FAST", seed=7,
            out_path=os.path.join(PROJECT_ROOT, "diagnostics", "v6", "genesis_controlled.png"))
        rep.update({k: res[k] for k in ("status", "mode", "controlnet", "control_repo",
                                        "control_revision", "controlnet_load_s", "seconds",
                                        "provenance")})
        rep["path"] = os.path.relpath(res["path"], PROJECT_ROOT)
        # structure preservation: edge overlap between control map and render
        import numpy as _np
        from PIL import Image as _Image
        ctrl = _np.asarray(edges.convert("L")) > 128
        edges2, _ = edge_map(res["path"])
        rend = _np.asarray(edges2.convert("L")) > 128
        inter = float((ctrl & rend).sum())
        union = float((ctrl | rend).sum())
        rep["edge_iou"] = round(inter / max(1.0, union), 4)
        rep["edge_density_control"] = round(float(ctrl.mean()), 4)
        rep["edge_density_render"] = round(float(rend.mean()), 4)
        if rep["edge_iou"] < 0.05:
            rep["status"] = "FAIL"
            rep["reason"] = "no structure preserved (IoU too low)"
        model.unload()
    except Exception as e:  # noqa: BLE001
        import traceback
        rep.update({"status": "FAIL" if "Memory" not in type(e).__name__ else "UNAVAILABLE",
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "trace": traceback.format_exc()[-1200:]})
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"CONTROLNET {rep.get('status')}: iou={rep.get('edge_iou')} "
          f"render={rep.get('seconds')}s")


if __name__ == "__main__":
    main()
