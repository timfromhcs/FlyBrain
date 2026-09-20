#!/usr/bin/env python3
"""V5 Phase 00: repository reality check -> diagnostics/master_inventory.json."""
import ast
import glob
import json
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def git(args):
    try:
        return subprocess.run(["git"] + args, capture_output=True, text=True,
                              timeout=10, cwd=PROJECT_ROOT).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def api_routes():
    src = open(os.path.join(PROJECT_ROOT, "src/ui/server.py"), encoding="utf-8").read()
    out = []
    for m in re.finditer(r'@app\.(get|post|put|delete|websocket)\("([^"]+)"', src):
        out.append({"method": m.group(1).upper(), "path": m.group(2)})
    return sorted(out, key=lambda r: r["path"])


def cli_commands():
    src = open(os.path.join(PROJECT_ROOT, "src/main.py"), encoding="utf-8").read()
    parsers = sorted(set(re.findall(r'add_parser\("([^"]+)"', src)))
    flags = sorted(set(re.findall(r'add_argument\("(--[^"]+)"', src)))
    return {"subcommands": parsers, "global_flags": flags}


def ui_tabs():
    html = open(os.path.join(PROJECT_ROOT, "src/ui/static/index.html"), encoding="utf-8").read()
    tabs = re.findall(r"switchTab\('([^']+)'", html)
    return sorted(set(tabs))


def tests():
    files = sorted(glob.glob("tests/test_*.py", root_dir=PROJECT_ROOT))
    counts = {}
    for f in files:
        src = open(os.path.join(PROJECT_ROOT, f), encoding="utf-8").read()
        counts[f] = len(re.findall(r"def (test_\w+)", src))
    return {"files": files, "test_methods": counts,
            "total_methods": sum(counts.values())}


def main():
    from src.version import VERSION
    from src.connectome.types import GraphMode
    inv = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "version": VERSION,
        "source_modules": sorted(glob.glob("src/**/*.py", root_dir=PROJECT_ROOT,
                                           recursive=True)),
        "api_routes": api_routes(),
        "api_versioned_v1": [r for r in api_routes() if r["path"].startswith("/api/v1/")],
        "legacy_api": [r for r in api_routes() if not r["path"].startswith("/api/v1/")],
        "ui_tabs": ui_tabs(),
        "ui_assets": sorted(glob.glob("src/ui/static/**/*", root_dir=PROJECT_ROOT,
                                      recursive=True)),
        "cli": cli_commands(),
        "graph_modes": [m.value for m in GraphMode],
        "graph_identities": ["REAL_FULL", "REAL_SUBGRAPH", "SPATIAL_SURROGATE",
                             "SYNTHETIC_TEST"],
        "simulation_backends": ["vulkan_gpu", "cpu_reference"],
        "model_providers": ["local_gguf_llama_cpp"],
        "storage_backends": ["sqlite", "npz_snapshot", "json_manifest"],
        "deployment_targets": ["local_uvicorn", "huggingface_docker_space"],
        "backup_targets": ["local_disk"],
        "tests": tests(),
        "shaders": {s: os.path.getsize(os.path.join(PROJECT_ROOT, s))
                    for s in ["shaders/brain_step.comp", "shaders/brain_step.spv",
                              "shaders/plasticity.comp", "shaders/plasticity.spv"]},
    }
    out = os.path.join(PROJECT_ROOT, "diagnostics", "master_inventory.json")
    json.dump(inv, open(out, "w", encoding="utf-8"), indent=2)
    print(f"modules={len(inv['source_modules'])} routes={len(inv['api_routes'])} "
          f"v1={len(inv['api_versioned_v1'])} tests={inv['tests']['total_methods']} -> {out}")


if __name__ == "__main__":
    main()
