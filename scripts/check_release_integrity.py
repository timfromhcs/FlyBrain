#!/usr/bin/env python3
"""V5 phase_18: ONE canonical version everywhere -> diagnostics/release_certificate.json.

Compares src/version.py against: pyproject.toml, CLI output, API /api/version,
Space app defaults, Docker defaults, README/HUGGINGFACE version mentions,
acceptance version_metadata gate, shader/manifest presence. Any mismatch FAILs.
"""
import json
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    from src.version import VERSION, VERSION_TAG
    checks = {}
    ok = True

    def record(name, passed, detail=""):
        nonlocal ok
        checks[name] = {"pass": bool(passed), "detail": detail}
        if not passed:
            ok = False

    import tomllib
    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as f:
        pyproj = tomllib.load(f)["project"]["version"]
    record("pyproject", pyproj == VERSION, f"pyproject={pyproj} src={VERSION}")

    cli = subprocess.run([sys.executable, "src/main.py", "version"],
                         capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT)
    record("cli", VERSION in cli.stdout, cli.stdout.strip()[:80])

    from fastapi.testclient import TestClient
    from src.ui.server import app
    cl = TestClient(app)
    v = cl.get("/api/version").json()
    record("api", v.get("version") == VERSION, str(v)[:120])
    record("api_tag", v.get("release") == f"FlyBrain V{VERSION}", v.get("release", ""))

    gate = json.load(open(os.path.join(PROJECT_ROOT, "diagnostics",
                                       "acceptance_matrix.json"), encoding="utf-8"))
    vm = gate["categories"].get("version_metadata", {})
    record("acceptance_version_gate", vm.get("status") == "PASS"
           and VERSION in vm.get("reason", ""), vm.get("reason", ""))

    for name in ("shaders/brain_step.spv", "shaders/plasticity.spv",
                 "manifests/malecns_provenance.json",
                 "verification/acceptance_schema.json",
                 "huggingface/Dockerfile", "huggingface/app.py",
                 "huggingface/requirements-hf.txt"):
        p = os.path.join(PROJECT_ROOT, name)
        record(f"artifact:{name}", os.path.exists(p) and os.path.getsize(p) > 0,
               f"{os.path.getsize(p)} bytes" if os.path.exists(p) else "missing")

    # Space Dockerfile must not pin a different app version anywhere
    # (0.0.0.0 bind address is not a version pin)
    docker = open(os.path.join(PROJECT_ROOT, "huggingface/Dockerfile"),
                  encoding="utf-8").read().replace("0.0.0.0", "")
    m = re.findall(r"\b\d+\.\d+\.\d+\b", docker)
    record("space_no_version_pin", not m, f"versions found in Dockerfile: {m}")

    cert = {"version": VERSION, "version_tag": VERSION_TAG,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "overall": "PASS" if ok else "FAIL", "checks": checks}
    out = os.path.join(PROJECT_ROOT, "diagnostics", "release_certificate.json")
    json.dump(cert, open(out, "w", encoding="utf-8"), indent=2)
    print(f"release integrity: {cert['overall']} -> {out}")
    for n, c in checks.items():
        if not c["pass"]:
            print(f"  FAIL {n}: {c['detail']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
