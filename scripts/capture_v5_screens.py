#!/usr/bin/env python3
"""V5 phase_20: real screenshots from the running application (never fabricated).

Boots a local server, visits each tab, screenshots to visual_evidence/screens/.
Writes diagnostics/visual_evidence.json manifest. Requires playwright+chromium.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("V5_SHOT_PORT", "7862"))
BASE = f"http://127.0.0.1:{PORT}"
TABS = ["overview", "stream", "connectome", "colony", "provenance", "experiments",
        "memory", "evolution", "dreams", "backups", "diagnostics"]
OUTDIR = os.path.join(PROJECT_ROOT, "visual_evidence", "screens")


def main():
    env = dict(os.environ)
    env.update({"FLYBRAIN_CIRCUIT_SIZE": "256", "FLYBRAIN_GRAPH_MODE": "REAL_SUBGRAPH",
                "FLYBRAIN_USE_GPU": "0", "FLYBRAIN_SEED": "42",
                "FLYBRAIN_WATCHDOG": "0"})
    py = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    proc = subprocess.Popen([py, "-m", "uvicorn", "src.ui.server:app",
                             "--host", "127.0.0.1", "--port", str(PORT)],
                            cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env)
    shots = []
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                urllib.request.urlopen(BASE + "/api/health", timeout=5).read()
                break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError("server did not boot")
        from playwright.sync_api import sync_playwright
        os.makedirs(OUTDIR, exist_ok=True)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--use-gl=swiftshader"])
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.goto(BASE + "/", wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(4000)
            for tab in TABS:
                page.evaluate(f"switchTab('{tab}')")
                page.wait_for_timeout(2500)
                # connectome: let WebGL settle; colony/backups: let tables fill
                if tab == "connectome":
                    page.wait_for_timeout(4000)
                path = os.path.join(OUTDIR, f"v5_{tab}.png")
                page.screenshot(path=path)
                shots.append({"tab": tab, "file": f"visual_evidence/screens/v5_{tab}.png",
                              "bytes": os.path.getsize(path)})
                print(f"shot {tab}: {os.path.getsize(path)} bytes", flush=True)
            browser.close()
        result = {"overall": "PASS", "shots": shots}
    except Exception as e:  # noqa: BLE001
        result = {"overall": "ERROR", "error": f"{type(e).__name__}: {e}", "shots": shots}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    json.dump(result, open(os.path.join(PROJECT_ROOT, "diagnostics",
                                        "visual_evidence.json"), "w", encoding="utf-8"), indent=2)
    print("visual evidence:", result["overall"], f"({len(shots)}/{len(TABS)})")
    sys.exit(0 if result["overall"] == "PASS" and len(shots) == len(TABS) else 1)


if __name__ == "__main__":
    main()
