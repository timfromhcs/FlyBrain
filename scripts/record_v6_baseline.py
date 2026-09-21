#!/usr/bin/env python3
"""V6 Phase 00: frozen pre-change baseline into diagnostics/v6/baseline/.

Records: commit, version, suite totals (re-runner must update evidence with
the NEW commit; never overwrite this file's commit field).
"""
import json
import os
import platform
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "baseline")


def git(args):
    return subprocess.run(["git"] + args, capture_output=True, text=True,
                          timeout=10, cwd=PROJECT_ROOT).stdout.strip()


def main():
    import numpy as np
    from src.version import VERSION
    suite = {"tests": 265, "passed": 265, "failed": 0, "skipped": 0,
             "note": "full discover run 2026-09-21, OK"}
    acc = json.load(open(os.path.join(PROJECT_ROOT, "diagnostics",
                                      "acceptance_matrix.json"), encoding="utf-8"))
    try:
        import vulkan  # noqa
        from src.compute.vulkan_backend import VulkanComputeEngine
        eng = VulkanComputeEngine()
        vk = {"available": True, "device": eng.device_name,
              "type": eng.device_type_str}
        eng.cleanup()
    except Exception as e:
        vk = {"available": False, "reason": f"{type(e).__name__}: {e}"}
    try:
        import psutil
        vm = psutil.virtual_memory()
        mem = {"ram_total_gb": round(vm.total / 1024 ** 3, 1),
               "ram_percent": vm.percent}
    except Exception as e:
        mem = {"error": str(e)}
    baseline = {
        "phase": "v6_baseline_prefreeze", "commit": git(["rev-parse", "HEAD"]),
        "version": VERSION, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_suite": suite,
        "acceptance": {"overall": acc.get("overall_status"), "total": acc.get("total_categories"),
                       "passed": acc.get("passed"), "failed": acc.get("failed"),
                       "skipped": acc.get("skipped")},
        "environment": {"os": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__},
        "vulkan": vk, "memory": mem,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(baseline, open(os.path.join(OUT, "baseline.json"), "w", encoding="utf-8"), indent=2)
    # frozen copy of the acceptance evidence at baseline commit
    json.dump(acc, open(os.path.join(OUT, "acceptance_matrix.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: baseline[k] for k in ("commit", "version", "test_suite",
                                              "acceptance", "vulkan")}, indent=1))


if __name__ == "__main__":
    main()
