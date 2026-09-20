#!/usr/bin/env python3
"""Hugging Face Space smoke verification (Phase 10, real boot).

Boots the Space entrypoint (huggingface/app.py) under a real uvicorn server
with Space-like CPU-only env, then exercises health/version/doctor/state/
telemetry/provenance/connectome/UI/static assets and one harmless simulation
step. Writes diagnostics/huggingface_verification.json. No fake telemetry:
every check is a live HTTP response.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PORT = int(os.environ.get("HF_SMOKE_PORT", "7861"))
BASE = f"http://127.0.0.1:{PORT}"


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read()


def post(path, payload, timeout=60):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def main():
    env = dict(os.environ)
    env["FLYBRAIN_CIRCUIT_SIZE"] = "128"
    env["FLYBRAIN_GRAPH_MODE"] = "REAL_SUBGRAPH"
    env["FLYBRAIN_USE_GPU"] = "0"
    env["FLYBRAIN_SEED"] = "42"
    py = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    proc = subprocess.Popen(
        [py, "-m", "uvicorn", "app:app", "--app-dir",
         os.path.join(PROJECT_ROOT, "huggingface"),
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    checks = {}
    try:
        # wait for boot
        t0 = time.time()
        boot_log = ""
        while time.time() - t0 < 180:
            try:
                s, body = get("/api/health", timeout=5)
                checks["health"] = {"status": s, "body": json.loads(body)}
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("space did not boot within 180s")
        checks["boot_seconds"] = round(time.time() - t0, 1)

        s, body = get("/api/version")
        checks["version"] = {"status": s, "body": json.loads(body)}
        s, body = get("/api/doctor")
        checks["doctor"] = {"status": s, "overall": json.loads(body).get("overall")}
        s, body = get("/api/provenance")
        prov = json.loads(body)
        checks["provenance"] = {"status": s, "graph_identity": prov["graph_identity"],
                                "backend_note": "live"}
        s, body = get("/api/telemetry")
        checks["telemetry"] = {"status": s, "step": json.loads(body).get("step")}
        s, body = get("/api/connectome?max_nodes=64&max_edges=64")
        cc = json.loads(body)
        checks["connectome"] = {"status": s, "nodes": len(cc["nodes"]),
                                "edges": len(cc["edges"])}
        s, body = get("/")
        html = body.decode("utf-8", "replace")
        checks["ui"] = {"status": s, "has_title": "FlyBrain Lab" in html,
                        "local_js": "/static/js/lab.js" in html,
                        "no_cdn": "cdnjs.cloudflare.com" not in html
                        and "cdn.jsdelivr.net" not in html}
        s, body = get("/static/vendor/three.min.js")
        checks["vendor_js"] = {"status": s, "bytes": len(body)}
        s, body = post("/api/simulation/step",
                       {"steps": 2, "reward": 0.1})
        checks["step"] = {"status": s, "spikes": json.loads(body).get("spikes")}

        health = checks["health"]["body"]
        honest_backend = health.get("backend") == "cpu_reference"
        checks["backend_honest_cpu"] = honest_backend
        ok = all(checks[k].get("status") == 200
                 for k in ("health", "version", "doctor", "provenance",
                           "telemetry", "connectome", "ui", "vendor_js", "step")) \
            and checks["ui"]["has_title"] and checks["ui"]["local_js"] \
            and checks["ui"]["no_cdn"] and honest_backend
        checks["overall"] = "PASS" if ok else "FAIL"
    except Exception as e:  # noqa: BLE001
        checks["overall"] = "ERROR"
        checks["error"] = f"{type(e).__name__}: {e}"
    finally:
        proc.terminate()
        try:
            boot_log = proc.communicate(timeout=20)[0][-4000:]
        except Exception:
            proc.kill()
            boot_log = ""
    try:
        import subprocess as _sp
        commit = _sp.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip()
    except Exception:
        commit = "unknown"
    from src.version import VERSION
    report = {"timestamp": time.time(), "commit": commit, "version": VERSION,
              "entrypoint": "huggingface/app.py (uvicorn app:app, port 7860 in image; "
                            f"{PORT} for this smoke)",
              "env": {"FLYBRAIN_CIRCUIT_SIZE": "128", "FLYBRAIN_GRAPH_MODE": "REAL_SUBGRAPH",
                      "FLYBRAIN_USE_GPU": "0", "FLYBRAIN_SEED": "42"},
              "checks": checks, "startup_log_tail": boot_log}
    out = os.path.join(PROJECT_ROOT, "diagnostics", "huggingface_verification.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: (v if k != "startup_log_tail" else "...") for k, v in report.items()
                      if k in ("checks",)}, indent=2)[:3000])
    print("overall:", checks["overall"], "->", out)
    sys.exit(0 if checks["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
