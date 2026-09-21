#!/usr/bin/env python3
"""Run and record V7 baseline test suite into reports/v7-baseline/tests.json."""
import subprocess
import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

py_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")

tests = [
    ("validate_malecns_data", [py_exe, "scripts/validate_malecns_data.py"]),
    ("test_suite", [py_exe, "tests/test_suite.py"]),
    ("test_state_persistence", [py_exe, "scripts/test_state_persistence.py"]),
    ("test_dreaming", [py_exe, "scripts/test_dreaming.py"]),
    ("test_evolution", [py_exe, "scripts/test_evolution.py"]),
    ("verify_docs_consistency", [py_exe, "scripts/verify_docs_consistency.py"]),
    ("check_release_integrity", [py_exe, "scripts/check_release_integrity.py"]),
]

results = {}
all_pass = True
for name, cmd in tests:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    dt = time.time() - t0
    ok = (p.returncode == 0)
    if not ok:
        all_pass = False
    status_str = "PASS" if ok else "FAIL"
    results[name] = {
        "pass": ok,
        "returncode": p.returncode,
        "duration_s": round(dt, 2),
        "stdout_tail": p.stdout[-400:] if p.stdout else "",
        "stderr_tail": p.stderr[-400:] if p.stderr else ""
    }
    print(f"[{status_str}] {name} ({round(dt, 2)}s)")
    if not ok:
        print("  Stderr:", p.stderr.strip()[:200])

out_file = os.path.join(PROJECT_ROOT, "reports", "v7-baseline", "tests.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"all_pass": all_pass, "tests": results, "timestamp": time.time()}, f, indent=2)

print("Overall V7 baseline tests:", "ALL PASS" if all_pass else "FAILURES DETECTED")
sys.exit(0 if all_pass else 1)
