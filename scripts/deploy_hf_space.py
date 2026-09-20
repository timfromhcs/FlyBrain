#!/usr/bin/env python3
"""Create/upload the FlyBrain Lab Hugging Face Space (Docker SDK) and verify
it remotely. Real deployment: every result is a live Hub API response.

Writes remote results into diagnostics/huggingface_verification.json
(remote_* fields). Never prints the token.
"""
import json
import os
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SPACE_ID = os.environ.get("FLYBRAIN_SPACE_ID", "timfromhcs/FlyBrain-Lab")
TOKEN = os.environ.get("HF_TOKEN", "")


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    me = api.whoami(token=TOKEN)["name"]
    print(f"authenticated as {me}")
    repo = api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker",
                           exist_ok=True, token=TOKEN)
    print(f"space repo: {repo.repo_id} ({repo.url})")
    # Upload exactly the Docker build context (+ Space card).
    api.upload_folder(
        repo_id=SPACE_ID, repo_type="space", token=TOKEN,
        folder_path=PROJECT_ROOT,
        path_in_repo=".",
        allow_patterns=["huggingface/**", "src/**", "shaders/*", "manifests/*",
                        "malecns/data-raw/*"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="FlyBrain v4.1.0 Space build (REAL_SUBGRAPH, CPU-only, honest backend)",
    )
    print("upload complete; waiting for Space build...")
    # Poll runtime: the Space needs build+boot time; poll /api/health.
    host = f"https://huggingface.co/spaces/{SPACE_ID}"
    # Direct container port is served via the Space subdomain after build.
    runtime_url = f"https://{SPACE_ID.replace('/', '-')}.hf.space"
    deadline = time.time() + 1500
    last = ""
    remote = {"space_url": host, "runtime_url": runtime_url, "checks": {}}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(runtime_url + "/api/health", timeout=15) as r:
                body = json.loads(r.read())
            remote["checks"]["health"] = {"status": r.status, "body": body}
            for path in ("/api/version", "/api/provenance"):
                try:
                    with urllib.request.urlopen(runtime_url + path, timeout=20) as r2:
                        remote["checks"][path] = {"status": r2.status,
                                                  "keys": sorted(json.loads(r2.read()).keys())}
                except Exception as e:
                    remote["checks"][path] = {"error": f"{type(e).__name__}: {e}"}
            remote["overall"] = ("PASS" if remote["checks"]["health"].get("status") == 200
                                 and remote["checks"]["health"]["body"].get("backend")
                                 == "cpu_reference" else "FAIL")
            break
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(30)
    else:
        remote["overall"] = "BLOCKED"
        remote["blocker"] = f"Space did not serve HTTP within 25 min: {last}"
    out = os.path.join(PROJECT_ROOT, "diagnostics", "huggingface_verification.json")
    doc = json.load(open(out, encoding="utf-8"))
    doc["remote_space_id"] = SPACE_ID
    doc["remote_verified_at"] = time.time()
    doc["remote"] = remote
    json.dump(doc, open(out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(remote, indent=2)[:2000])
    print("remote overall:", remote["overall"])


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("HF_TOKEN is required")
    main()
