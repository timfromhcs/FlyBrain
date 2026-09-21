#!/usr/bin/env python3
"""
Automated Documentation and Code Consistency Verification Script for FlyBrain V10.
Evaluates:
  1. MaleCNS biological data counts
  2. Vulkan shader bindings
  3. Compiled SPIR-V shaders
  4. UI Endpoints
  5. Connectome types & graph modes
  6. Provenance manifest
  7. Verification contract terminology
  8. Version consistency (10.0.0 across all manifests and docs)
  9. Dataset registry consistency
  10. Installer version consistency
  11. Path portability in generated assets
"""

import os
import sys
import json
import re
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.version import VERSION


def verify_docs_consistency() -> bool:
    print("=" * 60)
    print(f"FLYBRAIN V10: DOCUMENTATION & CODE CONSISTENCY VERIFIER (v{VERSION})")
    print("=" * 60)
    inconsistencies = []

    # 1. Check MaleCNS Data Counts
    soma_csv = os.path.join(PROJECT_ROOT, "malecns", "data-raw", "2023-27-2 soma_sides.csv")
    conn_csv = os.path.join(PROJECT_ROOT, "malecns", "data-raw", "malecns_v1_0_connections.csv")

    if not os.path.exists(soma_csv) or not os.path.exists(conn_csv):
        inconsistencies.append("Missing MaleCNS biological CSV files in malecns/data-raw/")
    else:
        with open(soma_csv, "r", encoding="utf-8") as f:
            soma_count = sum(1 for _ in f) - 1
        with open(conn_csv, "r", encoding="utf-8") as f:
            conn_count = sum(1 for _ in f) - 1

        print(f"Verified MaleCNS biological somas: {soma_count:,}")
        print(f"Verified MaleCNS biological connections: {conn_count:,}")

        if soma_count != 125506:
            inconsistencies.append(f"MaleCNS soma count mismatch: expected 125,506, got {soma_count}")
        if conn_count != 99301:
            inconsistencies.append(f"MaleCNS connection count mismatch: expected 99,301, got {conn_count}")

    # 2. Check Vulkan Shader Bindings
    brain_comp = os.path.join(PROJECT_ROOT, "shaders", "brain_step.comp")
    if os.path.exists(brain_comp):
        with open(brain_comp, "r", encoding="utf-8") as f:
            content = f.read()
            bindings = re.findall(r"binding\s*=\s*(\d+)", content)
            binding_count = len(bindings)
            print(f"Verified brain_step.comp descriptor bindings: {binding_count}")
            if binding_count != 11:
                inconsistencies.append(f"brain_step.comp binding count mismatch: expected 11, got {binding_count}")
    else:
        inconsistencies.append("shaders/brain_step.comp not found")

    plasticity_comp = os.path.join(PROJECT_ROOT, "shaders", "plasticity.comp")
    if os.path.exists(plasticity_comp):
        with open(plasticity_comp, "r", encoding="utf-8") as f:
            content = f.read()
            bindings = re.findall(r"binding\s*=\s*(\d+)", content)
            binding_count = len(bindings)
            print(f"Verified plasticity.comp descriptor bindings: {binding_count}")
            if binding_count != 6:
                inconsistencies.append(f"plasticity.comp binding count mismatch: expected 6, got {binding_count}")
    else:
        inconsistencies.append("shaders/plasticity.comp not found")

    # 3. Check Compiled SPIR-V Shaders
    for spv in ["brain_step.spv", "plasticity.spv"]:
        spv_path = os.path.join(PROJECT_ROOT, "shaders", spv)
        if not os.path.exists(spv_path) or os.path.getsize(spv_path) == 0:
            inconsistencies.append(f"Missing or empty SPIR-V shader binary: shaders/{spv}")
        else:
            print(f"Verified SPIR-V binary: shaders/{spv} ({os.path.getsize(spv_path)} bytes)")

    # 4. Check UI Endpoints
    server_py = os.path.join(PROJECT_ROOT, "src", "ui", "server.py")
    expected_endpoints = [
        "/api/health",
        "/api/state",
        "/api/telemetry",
        "/api/connectome",
        "/api/simulation/start",
        "/api/simulation/pause",
        "/api/simulation/step",
        "/api/simulation/reset",
        "/api/memory",
        "/api/evolution/lineage",
        "/api/evolution/generation",
        "/api/dreams",
        "/api/tools",
        "/api/experiments",
        "/api/diagnostics",
        "/ws/telemetry"
    ]
    if os.path.exists(server_py):
        with open(server_py, "r", encoding="utf-8") as f:
            server_src = f.read()
            for ep in expected_endpoints:
                if ep not in server_src:
                    inconsistencies.append(f"UI Server missing required endpoint: {ep}")
    else:
        inconsistencies.append("src/ui/server.py not found")

    # 5. Check Connectome Types & Graph Modes
    from src.connectome.types import GraphMode
    modes = [m.value for m in GraphMode]
    expected_modes = ["REAL", "SPATIAL_SURROGATE", "SYNTHETIC_TEST"]
    for em in expected_modes:
        if em not in modes:
            inconsistencies.append(f"GraphMode missing mode: {em}")

    # 6. Check Provenance Manifest
    prov_manifest = os.path.join(PROJECT_ROOT, "manifests", "malecns_provenance.json")
    if os.path.exists(prov_manifest):
        with open(prov_manifest, "r", encoding="utf-8") as f:
            prov_data = json.load(f)
            if "provenance_metadata" not in prov_data:
                inconsistencies.append("malecns_provenance.json missing 'provenance_metadata'")
    else:
        inconsistencies.append("manifests/malecns_provenance.json not found")

    # 7. Verification Contract Conformance
    doc_files = ["README.md", "docs/alife_architecture.md", "RESEARCH_STATUS.md"]
    for df in doc_files:
        p = os.path.join(PROJECT_ROOT, df)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().lower()
        for line in text.splitlines():
            if "bit-exact" in line and ("parity" in line or "vulkan" in line):
                negated = ("not bit-exact" in line or "not-bit-exact" in line or "kein bit-exact" in line)
                if not negated:
                    inconsistencies.append(f"Verification contract violation in {df}: 'bit-exact' affirms Vulkan parity: {line.strip()[:80]}")
                    break

    # 8. Version Consistency
    expected_version = "10.0.0"
    version_checks = [
        ("src/version.py", rf'VERSION\s*=\s*"{expected_version}"'),
        ("pyproject.toml", rf'version\s*=\s*"{expected_version}"'),
        ("package.json", rf'"version":\s*"{expected_version}"'),
        ("installer/install.ps1", rf'version\s*=\s*"{expected_version}"'),
        ("scripts/install.ps1", rf'version\s*=\s*"{expected_version}"'),
        ("V10_MASTER_PLAN.md", rf'{expected_version}'),
        ("verification/v10_master_plan.json", rf'"{expected_version}"'),
        ("SECURITY.md", r'10\.0\.x'),
        ("RESEARCH_STATUS.md", rf'System Version:\*\*\s*{expected_version}'),
        ("CHANGELOG.md", rf'\[{expected_version}\]'),
        ("RELEASE_NOTES.md", rf'v{expected_version}')
    ]
    for rel_path, pattern in version_checks:
        fp = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(fp):
            inconsistencies.append(f"Version check file not found: {rel_path}")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            if not re.search(pattern, f.read()):
                inconsistencies.append(f"Version mismatch in {rel_path}: pattern {pattern} not matched")

    # 9. Canonical Dataset Registry
    try:
        from src.connectome.dataset_registry import DatasetRegistry
        datasets = DatasetRegistry.list_all()
        ds_ids = {d.dataset_id for d in datasets}
        required_ds = {"MALECNS_V1_FULL", "MALECNS_V1_DERIVED_SUBGRAPH", "MALECNS_SPATIAL_SURROGATE", "SYNTHETIC_TEST_V1"}
        if not required_ds.issubset(ds_ids):
            inconsistencies.append(f"DatasetRegistry missing required datasets: {required_ds - ds_ids}")
    except Exception as e:
        inconsistencies.append(f"Failed to import/query DatasetRegistry: {e}")

    # 10. Installer & Doctor Headers
    doctor_checks = [
        ("installer/doctor.ps1", "V10"),
        ("scripts/doctor.ps1", "V10")
    ]
    for rel_p, pattern in doctor_checks:
        fp = os.path.join(PROJECT_ROOT, rel_p)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                if pattern not in f.read():
                    inconsistencies.append(f"Installer doctor check mismatch in {rel_p}: '{pattern}' missing")

    # 11. Path Portability in Assets
    assets_dir = os.path.join(PROJECT_ROOT, "assets", "compiled")
    if os.path.exists(assets_dir):
        for fn in os.listdir(assets_dir):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(assets_dir, fn), "r", encoding="utf-8") as af:
                        raw = af.read()
                        if "E:\\" in raw or "C:\\Users" in raw:
                            inconsistencies.append(f"Non-portable Windows absolute path found in asset package {fn}")
                except Exception:
                    pass

    print("=" * 60)
    report = {
        "status": "PASS" if not inconsistencies else "FAIL",
        "timestamp": time.time(),
        "version": VERSION,
        "inconsistencies_count": len(inconsistencies),
        "inconsistencies": inconsistencies
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "diagnostics"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "diagnostics", "v10_documentation_report.json"), "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    if inconsistencies:
        print(f"FAILED: {len(inconsistencies)} documentation/code inconsistencies found:")
        for inc in inconsistencies:
            print(f"  - [FAIL] {inc}")
        return False
    else:
        print(f"ALL DOCUMENTATION AND CODE CONSISTENCY CHECKS PASSED [100% OK, v{VERSION}]")
        return True


def main():
    success = verify_docs_consistency()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
