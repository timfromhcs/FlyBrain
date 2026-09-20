#!/usr/bin/env python3
"""Remote Space verification (V5 phase_15/26/27, bounded ~6 min).

Live HTTP against the deployed Space: health/version/provenance/UI/step +
backup create -> download -> hash-verify -> restore -> state-compare.
Writes the remote section of diagnostics/huggingface_verification.json.
"""
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("FLYBRAIN_SPACE_ID", "timfromhcs/FlyBrain-Lab")
RUNTIME = f"https://{SPACE_ID.replace('/', '-')}.hf.space"


def get(path, timeout=30):
    with urllib.request.urlopen(RUNTIME + path, timeout=timeout) as r:
        return r.status, r.read()


def post(path, payload=None, timeout=90):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(RUNTIME + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def main():
    checks = {}
    try:
        s, body = get("/api/health", timeout=30)
        health = json.loads(body)
        checks["health"] = {"status": s, "backend": health.get("backend"),
                            "version": health.get("version"),
                            "graph_mode": health.get("graph_mode")}
        s, body = get("/api/version", timeout=30)
        checks["version"] = {"status": s, "body": json.loads(body)}
        s, body = get("/api/provenance", timeout=30)
        prov = json.loads(body)
        checks["provenance"] = {"status": s, "graph_identity": prov.get("graph_identity"),
                                "status_label": prov.get("provenance_status")}
        s, body = get("/", timeout=30)
        html = body.decode("utf-8", "replace")
        checks["ui"] = {"status": s, "has_title": "FlyBrain Lab" in html,
                        "local_js": "/static/js/lab.js" in html,
                        "no_cdn": "cdnjs.cloudflare.com" not in html}
        s, body = post("/api/v1/simulation/step", {"steps": 2, "reward": 0.0}, timeout=60)
        checks["step"] = {"status": s, "spikes": json.loads(body).get("spikes")}
        # backup round trip on the Space (phase_26)
        s, body = post("/api/v1/backup/create?label=remote-verify&trigger=manual",
                       {}, timeout=120)
        man = json.loads(body)
        bname = man.get("backup_name")
        checks["backup_create"] = {"status": s, "backup": bname,
                                   "state_hash": (man.get("state_hash") or "")[:16]}
        with urllib.request.urlopen(
                RUNTIME + f"/api/v1/backup/download?name={bname}", timeout=120) as r:
            blob = r.read()
        z = zipfile.ZipFile(io.BytesIO(blob))
        names = z.namelist()
        manifest = json.load(z.open("manifest.json"))
        mhash = manifest.get("state_hash")
        # verify downloaded manifest files against their recorded hashes
        dl_ok = True
        for fname, expected in manifest.get("files", {}).items():
            h = hashlib.sha256()
            with z.open(fname) as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                dl_ok = False
        checks["backup_download"] = {"bytes": len(blob), "files": names,
                                     "hashes_ok": dl_ok,
                                     "state_hash": (mhash or "")[:16]}
        s, body = get(f"/api/v1/backup/verify?name={bname}", timeout=60)
        checks["backup_verify"] = {"status": s,
                                   "verdict": json.loads(body).get("status")}
        s, body = get("/api/v1/stream/status", timeout=30)
        before = json.loads(body)["simulation_step"]
        s, body = post(f"/api/v1/backup/restore?name={bname}", {}, timeout=120)
        restored = json.loads(body)
        s, body = get("/api/v1/stream/status", timeout=30)
        after = json.loads(body)["simulation_step"]
        checks["backup_restore"] = {"status": s,
                                    "restored": restored.get("status"),
                                    "state_hash": (restored.get("state_hash") or "")[:16],
                                    "hash_match": restored.get("state_hash") == mhash,
                                    "step_before": before, "step_after": after}
        ok = (checks["health"]["status"] == 200
              and checks["health"]["backend"] == "cpu_reference"
              and checks["provenance"]["graph_identity"] == "REAL_SUBGRAPH"
              and checks["ui"]["has_title"] and checks["ui"]["no_cdn"]
              and checks["backup_download"]["hashes_ok"]
              and checks["backup_verify"]["verdict"] == "VALID"
              and checks["backup_restore"]["hash_match"])
        overall = "PASS" if ok else "FAIL"
    except Exception as e:  # noqa: BLE001
        overall = "ERROR"
        checks["error"] = f"{type(e).__name__}: {e}"
    out = os.path.join(PROJECT_ROOT, "diagnostics", "huggingface_verification.json")
    doc = json.load(open(out, encoding="utf-8"))
    doc["remote_space_id"] = SPACE_ID
    doc["remote_verified_at"] = time.time()
    doc["remote"] = {"space_url": f"https://huggingface.co/spaces/{SPACE_ID}",
                     "runtime_url": RUNTIME, "overall": overall, "checks": checks}
    json.dump(doc, open(out, "w", encoding="utf-8"), indent=2)
    print("remote overall:", overall)
    for k, v in checks.items():
        print(f"  {k}: {str(v)[:150]}")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
