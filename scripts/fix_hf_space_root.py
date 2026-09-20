#!/usr/bin/env python3
"""Upload root-level Space files (Docker SDK requires Dockerfile/app.py/README
at repo root) and poll the Space until it serves HTTP or a blocker is proven."""
import json
import os
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("FLYBRAIN_SPACE_ID", "timfromhcs/FlyBrain-Lab")
TOKEN = os.environ.get("HF_TOKEN", "")
RUNTIME = f"https://{SPACE_ID.replace('/', '-')}.hf.space"


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    root_files = {
        "huggingface/Dockerfile": "Dockerfile",
        "huggingface/app.py": "app.py",
        "huggingface/requirements-hf.txt": "requirements-hf.txt",
        "huggingface/README.md": "README.md",
    }
    for local, remote in root_files.items():
        api.upload_file(repo_id=SPACE_ID, repo_type="space", token=TOKEN,
                        path_or_fileobj=os.path.join(PROJECT_ROOT, local),
                        path_in_repo=remote,
                        commit_message=f"FlyBrain v4.1.0 Space root layout: {remote}")
        print(f"uploaded {remote}")
    try:
        rt = api.get_space_runtime(SPACE_ID, token=TOKEN)
        print("stage:", rt.stage, "| hardware:", rt.hardware)
    except Exception as e:
        print("runtime:", type(e).__name__, str(e)[:200])
    deadline = time.time() + 2400
    remote = {"space_url": f"https://huggingface.co/spaces/{SPACE_ID}",
              "runtime_url": RUNTIME, "checks": {}}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(RUNTIME + "/api/health", timeout=15) as r:
                body = json.loads(r.read())
            remote["checks"]["health"] = {"status": r.status, "body": body}
            for path in ("/api/version", "/api/provenance", "/api/doctor"):
                try:
                    with urllib.request.urlopen(RUNTIME + path, timeout=20) as r2:
                        payload = json.loads(r2.read())
                        remote["checks"][path] = {
                            "status": r2.status,
                            "summary": (payload.get("version")
                                        or payload.get("graph_identity")
                                        or payload.get("overall"))}
                except Exception as e:
                    remote["checks"][path] = {"error": f"{type(e).__name__}: {e}"}
            try:
                with urllib.request.urlopen(RUNTIME + "/", timeout=20) as r3:
                    html = r3.read().decode("utf-8", "replace")
                remote["checks"]["ui"] = {"status": r3.status,
                                          "has_title": "FlyBrain Lab" in html,
                                          "no_cdn": "cdnjs.cloudflare.com" not in html}
            except Exception as e:
                remote["checks"]["ui"] = {"error": f"{type(e).__name__}: {e}"}
            ok = (remote["checks"]["health"].get("status") == 200
                  and remote["checks"]["health"]["body"].get("backend") == "cpu_reference")
            remote["overall"] = "PASS" if ok else "FAIL"
            break
        except Exception as e:  # noqa: BLE001
            print(f"waiting... {type(e).__name__}", flush=True)
            time.sleep(45)
    else:
        remote["overall"] = "BLOCKED"
        remote["blocker"] = "Space did not serve HTTP within 40 min (build quota/scheduler external)"
    out = os.path.join(PROJECT_ROOT, "diagnostics", "huggingface_verification.json")
    doc = json.load(open(out, encoding="utf-8"))
    doc["remote_space_id"] = SPACE_ID
    doc["remote_verified_at"] = time.time()
    doc["remote"] = remote
    json.dump(doc, open(out, "w", encoding="utf-8"), indent=2)
    print("remote overall:", remote["overall"])


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("HF_TOKEN is required")
    main()
