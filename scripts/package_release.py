"""Local and Cloud Release Packager for FlyBrain V8/V9.

Produces Section 80 & 152 Build Artifacts:
- FlyBrain-Windows-x64-Portable.zip
- SHA256SUMS.txt
- release-manifest.json
- SBOM.json
"""
import os
import sys
import json
import zipfile
import hashlib
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
os.makedirs(DIST_DIR, exist_ok=True)

from src.version import VERSION, VERSION_TAG, RELEASE_NAME


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Building Release Artifacts for {RELEASE_NAME} ({VERSION})...")
    zip_name = f"FlyBrain-{VERSION_TAG}-Windows-x64-Portable.zip"
    zip_path = os.path.join(DIST_DIR, zip_name)

    # Core release payload (clean, lightweight runtime bundle - models download to %LOCALAPPDATA%)
    include_dirs = ["src", "shaders", "manifests", "scripts", "installer", "docs"]
    include_files = [
        "README.md", "RELEASE_NOTES.md", "RESEARCH_STATUS.md", "GEMINI.md",
        "CHANGELOG.md", "SECURITY.md", "LICENSE", "pyproject.toml", "requirements.txt",
        "package.json", "models/registry.yaml"
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in include_files:
            fp = os.path.join(PROJECT_ROOT, f)
            if os.path.exists(fp):
                z.write(fp, f)
        for d in include_dirs:
            dp = os.path.join(PROJECT_ROOT, d)
            if os.path.exists(dp):
                for root, _, files in os.walk(dp):
                    if "__pycache__" in root or ".git" in root:
                        continue
                    for file in files:
                        if file.endswith(".pyc") or file.endswith(".safetensors") or file.endswith(".gguf"):
                            # Skip heavy model binaries from lightweight portable zip
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                        z.write(full_path, rel_path)

    zip_size_mb = round(os.path.getsize(zip_path) / (1024 ** 2), 2)
    zip_sha = hash_file(zip_path)
    print(f"  [+] Created {zip_name} ({zip_size_mb} MB) -> SHA256: {zip_sha}")

    # SHA256SUMS.txt
    sums_path = os.path.join(DIST_DIR, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write(f"{zip_sha}  {zip_name}\n")

    # SBOM.json (Software Bill of Materials)
    sbom_path = os.path.join(DIST_DIR, "SBOM.json")
    try:
        import tomllib
        with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as pf:
            pyproj = tomllib.load(pf)
            deps = pyproj.get("project", {}).get("dependencies", [])
    except Exception:
        deps = []

    sbom = {
        "bomFormat": "CycloneDX-FlyBrain",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": {
                "name": "FlyBrain",
                "version": VERSION,
                "type": "application",
                "licenses": [{"license": {"id": "Apache-2.0"}}]
            }
        },
        "components": [{"name": d.split(">=")[0].split("==")[0], "version_spec": d} for d in deps]
    }
    with open(sbom_path, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2)

    # release-manifest.json
    manifest_path = os.path.join(DIST_DIR, "release-manifest.json")
    manifest = {
        "version": VERSION,
        "release_tag": VERSION_TAG,
        "release_name": RELEASE_NAME,
        "timestamp": time.time(),
        "artifacts": {
            zip_name: {"sha256": zip_sha, "bytes": os.path.getsize(zip_path), "size_mb": zip_size_mb}
        },
        "verification": {
            "license_audit": "reports/license-audit.json",
            "v7_baseline": "reports/v7-baseline/tests.json"
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"All release artifacts compiled to {DIST_DIR}")


if __name__ == "__main__":
    main()
