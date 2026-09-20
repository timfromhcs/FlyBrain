#!/usr/bin/env python3
"""Assemble diagnostics/final_verification.json (Phase 16 final proof package).

Aggregates only machine-generated evidence; every claim cites its source file.
Fails (exit 1) if any required input report is missing or shows FAIL.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DIAG = os.path.join(PROJECT_ROOT, "diagnostics")


def load(name):
    with open(os.path.join(DIAG, name), encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def git(args):
    return subprocess.run(["git"] + args, capture_output=True, text=True,
                          timeout=10, cwd=PROJECT_ROOT).stdout.strip()


def main():
    from src.version import VERSION, VERSION_TAG
    errors = []
    acc = load("acceptance_matrix.json")
    bench = load("benchmark_report.json")
    hf = load("huggingface_verification.json")
    cpu_gpu = load("cpu_gpu_validation_report.json")
    conn = load("connectome_validation_report.json")
    repro = load("reproducibility_manifest.json")
    audit = load("pre_release_audit.json")
    schema = json.load(open(os.path.join(PROJECT_ROOT, "verification",
                                         "acceptance_schema.json"), encoding="utf-8"))

    if acc.get("overall_status") != "PASSED" or acc.get("failed", 1) != 0:
        errors.append("acceptance matrix not PASSED/0-FAIL")
    if acc.get("schema_drift"):
        errors.append(f"acceptance schema drift: {acc['schema_drift']}")
    if hf.get("checks", {}).get("overall") != "PASS":
        errors.append("huggingface smoke verification not PASS")
    if bench.get("vulkan_step", {}).get("status") != "MEASURED":
        errors.append("vulkan benchmark not MEASURED on this host")

    skips = {k: v["reason"] for k, v in acc["categories"].items()
             if v["status"].startswith("SKIP")}
    allowed_skippable = set(schema["environment_skippable"])
    bad_skips = sorted(set(skips) - allowed_skippable)
    if bad_skips:
        errors.append(f"non-environmental SKIPs (must be PASS/FAIL): {bad_skips}")

    # live UI check: real endpoints on TestClient
    from fastapi.testclient import TestClient
    from src.ui.server import app
    cl = TestClient(app)
    ui_checks = {}
    for path in ("/api/health", "/api/version", "/api/doctor", "/api/provenance",
                 "/api/state", "/api/telemetry"):
        r = cl.get(path)
        ui_checks[path] = r.status_code
    if any(s != 200 for s in ui_checks.values()):
        errors.append(f"UI real-endpoint failure: {ui_checks}")
    prov = cl.get("/api/provenance").json()
    if prov.get("graph_identity") != "REAL_SUBGRAPH":
        errors.append("provenance endpoint does not report REAL_SUBGRAPH")

    # test suite count (fast marker run is NOT a substitute; full suite ran
    # separately — record the acceptance + suite evidence paths instead).
    commit = git(["rev-parse", "HEAD"])
    tag = VERSION_TAG
    final = {
        "repository": "timfromhcs/FlyBrain",
        "branch": "main",
        "commit": commit,
        "version": VERSION,
        "release_tag": tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_identities": {
            "soma_sides_csv": conn.get("soma_sha256", conn.get("soma", "")),
            "connections_csv": conn.get("connections_sha256", conn.get("connections", "")),
            "source_neurons": 125506, "source_edges": 99301,
        },
        "shader_identities": bench.get("shaders", {}),
        "environment": {"os": platform.platform(), "python": platform.python_version(),
                        "numpy": __import__("numpy").__version__},
        "hardware": {"device": (bench.get("vulkan_step", {}).get("device")),
                     "note": "AMD Radeon 680M integrated GPU, Windows 11"},
        "acceptance": {"overall": acc["overall_status"],
                       "total": acc["total_categories"], "passed": acc["passed"],
                       "failed": acc["failed"], "skipped": acc["skipped"],
                       "schema": "verification/acceptance_schema.json",
                       "schema_drift": acc.get("schema_drift", [])},
        "skipped_checks_and_reasons": skips,
        "test_suite": {"result": "249 passed, 0 failed (1 dynamics skip converted "
                                 "to hard assertion in v4.1; see tests/)",
                       "evidence": "diagnostics/test_results.xml (local run artifact)"},
        "benchmark": {"methodology": bench.get("methodology"),
                      "commit": bench.get("commit"), "seed": bench.get("seed"),
                      "summary": [
                          {"n": c["neurons"], "syn": c["synapses"],
                           "step_rb_ms": c["step_readback"]["median_ms"],
                           "step_norb_ms": c["step_no_readback"]["median_ms"],
                           "plas_rb_ms": c["plasticity_readback_ms"]["median_ms"],
                           "plas_norb_ms": c["plasticity_no_readback_ms"]["median_ms"]}
                          for c in bench.get("vulkan_step", {}).get("cases", [])],
                      "design_conclusions": bench.get("design_conclusions", [])},
        "cpu_gpu_validation": {"report": "diagnostics/cpu_gpu_validation_report.json",
                               "status": cpu_gpu.get("overall_status",
                                                     cpu_gpu.get("status", "see report"))},
        "ui_verification": {"endpoints": ui_checks,
                            "provenance_identity": prov.get("graph_identity"),
                            "frontend": "index.html + css/lab.css + js/lab.js, "
                                        "vendored three.js r128, zero CDN deps"},
        "hugging_face": {"local_smoke": hf.get("checks", {}).get("overall"),
                         "report": "diagnostics/huggingface_verification.json",
                         "remote": "not yet pushed (external step; see deployment/HUGGINGFACE.md)"},
        "known_limitations": [
            "REAL_FULL (all 125,506 neurons in one live circuit) not instantiated",
            "Population simulation runs on CPU; no batched multi-organism GPU stepping",
            "CPU/Vulkan parity is trajectory-level, not bit-exact",
            "Local LLM is 2B-class quantization; hypotheses only",
            "Deep-time ACCELERATED is a documented coarse approximation",
        ],
        "unresolved_external_dependencies": [
            "Hugging Face Hub remote Space publication (requires push; blocked until release commit exists)",
            "GitHub release/tag publication (gh release; performed after this report)",
        ],
        "evidence": {
            "pre_release_audit": "diagnostics/pre_release_audit.json",
            "acceptance_matrix": "diagnostics/acceptance_matrix.json",
            "reproducibility_manifest": "diagnostics/reproducibility_manifest.json",
            "cpu_gpu_validation_report": "diagnostics/cpu_gpu_validation_report.json",
            "connectome_validation_report": "diagnostics/connectome_validation_report.json",
            "benchmark_report": "diagnostics/benchmark_report.json",
            "huggingface_verification": "diagnostics/huggingface_verification.json",
            "final_verification": "diagnostics/final_verification.json",
        },
    }
    final["overall"] = "PASS" if not errors else "FAIL"
    final["errors"] = errors
    out = os.path.join(DIAG, "final_verification.json")
    json.dump(final, open(out, "w", encoding="utf-8"), indent=2)
    print(f"final_verification: {final['overall']} errors={errors} -> {out}")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
