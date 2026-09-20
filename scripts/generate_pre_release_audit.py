#!/usr/bin/env python3
"""Generate diagnostics/pre_release_audit.json (Phase 0 required audit)."""
import os, sys, json, hashlib, subprocess, platform, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def git(cmd):
    try:
        return subprocess.check_output(["git"] + cmd, cwd=PROJECT_ROOT, text=True).strip()
    except Exception as e:
        return f"unavailable: {e}"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()

import glob
src_modules = sorted(glob.glob("src/**/*.py", root_dir=PROJECT_ROOT))
tests = sorted(glob.glob("tests/test_*.py", root_dir=PROJECT_ROOT))
shaders = {}
for s in ["shaders/brain_step.comp", "shaders/brain_step.spv",
          "shaders/plasticity.comp", "shaders/plasticity.spv"]:
    p = os.path.join(PROJECT_ROOT, s)
    if os.path.exists(p):
        shaders[s] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
try:
    import src.version as v
    detected_version = v.VERSION
except Exception as e:
    detected_version = f"unavailable: {e}"
try:
    import tomllib
    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as f:
        pyproj_version = tomllib.load(f)["project"]["version"]
except Exception as e:
    pyproj_version = f"unavailable: {e}"
dep_versions = {}
for pkg in ["numpy", "scipy", "fastapi", "uvicorn", "vulkan", "torch",
            "transformers", "pydantic", "psutil"]:
    try:
        m = __import__(pkg)
        dep_versions[pkg] = getattr(m, "__version__", "installed-unknown")
    except Exception as e:
        dep_versions[pkg] = f"missing: {e}"
tracked = git(["ls-files"]).splitlines()
runtime_artifact_exts = (".db", ".npz", ".wav", ".log", ".xml")
tracked_runtime = [t for t in tracked if t.lower().endswith(runtime_artifact_exts)]
doc_claims = {}
for doc in ["README.md", "ARCHITECTURE.md", "RESEARCH_STATUS.md",
            "REPRODUCIBILITY.md", "final_verification_report.md"]:
    p = os.path.join(PROJECT_ROOT, doc)
    if os.path.exists(p):
        txt = open(p, encoding="utf-8").read()
        import re
        doc_claims[doc] = sorted(set(re.findall(r"\d+/\d+", txt)))
# acceptance categories from runner source
import re
runner = open(os.path.join(PROJECT_ROOT, "scripts/run_acceptance_matrix.py"), encoding="utf-8").read()
cats = sorted(set(re.findall(r'matrix\["([a-z0-9_]+)"\]', runner)))
acc = {}
acc_path = os.path.join(PROJECT_ROOT, "diagnostics/acceptance_matrix.json")
if os.path.exists(acc_path):
    acc = json.load(open(acc_path))
# HF-related files
hf_files = [t for t in tracked if "huggingface" in t.lower() or t == "app.py"
            or t.startswith("huggingface/") or "Dockerfile" in t or "Space" in t]
audit = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "repository_commit": git(["rev-parse", "HEAD"]),
    "branch": git(["branch", "--show-current"]),
    "tags": [t for t in git(["tag", "--sort=-v:refname"]).splitlines() if t],
    "detected_version_src": detected_version,
    "detected_version_pyproject": pyproj_version,
    "dependency_versions": dep_versions,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "test_inventory": tests,
    "test_count": len(tests),
    "source_modules": src_modules,
    "source_module_count": len(src_modules),
    "shaders": shaders,
    "datasets": {
        "soma": "malecns/data-raw/2023-27-2 soma_sides.csv",
        "connections": "malecns/data-raw/malecns_v1_0_connections.csv",
    },
    "tracked_runtime_artifacts": tracked_runtime,
    "documentation_claims_acceptance_counts": doc_claims,
    "acceptance_categories_in_runner": cats,
    "acceptance_category_count_in_runner": len(cats),
    "last_acceptance_report": {
        "total": acc.get("total_categories"), "passed": acc.get("passed"),
        "failed": acc.get("failed"), "skipped": acc.get("skipped"),
        "overall": acc.get("overall_status"),
    },
    "deployment_configuration": {
        "hf_tracked_files": hf_files,
        "has_app_py": os.path.exists(os.path.join(PROJECT_ROOT, "app.py")),
        "has_dockerfile": any(os.path.exists(os.path.join(PROJECT_ROOT, f)) for f in ["Dockerfile", "huggingface/Dockerfile"]),
        "has_hf_dir": os.path.isdir(os.path.join(PROJECT_ROOT, "huggingface")),
        "has_verification_dir": os.path.isdir(os.path.join(PROJECT_ROOT, "verification")),
    },
    "discovered_contradictions": [
        "pyproject version 1.0.0 != src/version 4.0.0 and tag v4.0.0",
        "README claims 44/44 acceptance; runner implements 57 categories; last report 56 PASS + 1 SKIP",
        "ARCHITECTURE.md describes 25-category matrix (stale)",
        "RESEARCH_STATUS.md claims 19/19 (stale)",
        "final_verification_report.md claims 44/44 (stale vs 57-category runner)",
        "No Hugging Face Space deployment files tracked (deployment is documentation-only)",
        "Tracked runtime artifacts (.db/.npz/.wav) committed despite .gitignore ignoring *.db/*.npz",
        "UI is a single 48KB static monolith with CDN Three.js dependency (reproducibility risk)",
        "GraphMode.REAL is a bounded sampled subgraph (hub-biased) but named REAL without SUBGRAPH qualifier in enum/CLI/UI",
    ],
    "known_environment_dependencies": [
        "Vulkan SDK with glslc for shader rebuild",
        "AMD Radeon 680M iGPU (Vulkan 1.3) on Windows 11 workstation",
        "Local GGUF weights (llm/gguf/*.gguf, git-ignored) for LLM inference gates",
        "Janelia MaleCNS data under malecns/data-raw/",
        "Python 3.12 via .venv (system python is 3.14)",
    ],
    "planned_repairs": [
        "Bump to v4.1.0 in src/version.py and pyproject.toml (semver minor, additive hardening)",
        "Introduce GraphMode.REAL_SUBGRAPH as explicit canonical name; keep REAL as legacy alias",
        "Create verification/acceptance_schema.json as single canonical schema; fix docs to derive counts from report",
        "Add src/provenance/identity_layers.py mapping 11 required identity layers onto BrainIdentity",
        "Add capability-enforcement hardening to LLM control plane (keep substring list as defense-in-depth only)",
        "Measure Vulkan memory/plasticity/command-recording behavior; document benchmark-backed decisions",
        "Split/polish UI with explicit REAL_SUBGRAPH + sampling + heuristic + weight-transform disclosures",
        "Create real huggingface/ Space (app.py, requirements, Dockerfile, README front matter) and verify locally",
        "Hygiene: stop tracking new runtime artifacts; document canonical vs disposable evidence policy",
        "Regenerate docs (README/ARCHITECTURE/REPRODUCIBILITY/RESEARCH_STATUS/RELEASE_NOTES/deployment/HUGGINGFACE.md)",
        "Produce final proof package of 8 diagnostics reports",
    ],
}
out = os.path.join(PROJECT_ROOT, "diagnostics/pre_release_audit.json")
json.dump(audit, open(out, "w", encoding="utf-8"), indent=2)
print(f"wrote {out}: commit={audit['repository_commit']} version={detected_version} cats={len(cats)}")
