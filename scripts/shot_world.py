#!/usr/bin/env python3
"""Boot server, screenshot WORLD tab, report console errors."""
import os
import subprocess
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 7863
BASE = f"http://127.0.0.1:{PORT}"


def main():
    env = dict(os.environ)
    env.update({"FLYBRAIN_CIRCUIT_SIZE": "128", "FLYBRAIN_GRAPH_MODE": "SYNTHETIC_TEST",
                "FLYBRAIN_USE_GPU": "0", "FLYBRAIN_WATCHDOG": "0"})
    py = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    proc = subprocess.Popen([py, "-m", "uvicorn", "src.ui.server:app",
                             "--host", "127.0.0.1", "--port", str(PORT)],
                            cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL,
                            stderr=open(os.path.join(PROJECT_ROOT, "diagnostics", "v6",
                                                     "world_shot_stderr.log"), "w"),
                            env=env)
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                urllib.request.urlopen(BASE + "/api/health", timeout=5).read()
                break
            except Exception:
                time.sleep(3)
        from playwright.sync_api import sync_playwright
        errors = []
        with sync_playwright() as pw:
            b = pw.chromium.launch(args=["--use-gl=swiftshader"])
            pg = b.new_page(viewport={"width": 1600, "height": 900})
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(BASE + "/", wait_until="networkidle", timeout=60000)
            pg.wait_for_timeout(3000)
            pg.evaluate("switchTab('world')")
            pg.wait_for_timeout(5000)
            pg.screenshot(path=os.path.join(PROJECT_ROOT, "visual_evidence", "screens", "v6_world.png"))
            b.close()
        print("console errors:", len(errors))
        for e in errors[:8]:
            print("  -", e[:200])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
