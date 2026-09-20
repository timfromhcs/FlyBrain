#!/usr/bin/env python3
"""Upload fixed Space files (bounded): tool import guards + requirements."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("FLYBRAIN_SPACE_ID", "timfromhcs/FlyBrain-Lab")
TOKEN = os.environ.get("HF_TOKEN", "")
FILES = ["src/tools/speech.py", "src/tools/audio.py", "src/tools/image_gen.py",
         "src/compute/vulkan_backend.py",
         "src/ui/server.py", "src/brain/simulation_engine.py",
         "src/backup/service.py", "src/backup/gdrive.py", "src/backup/__init__.py",
         "src/runtime/__init__.py", "src/runtime/watchdog.py",
         "src/main.py", "src/version.py",
         "huggingface/Dockerfile", "huggingface/requirements-hf.txt"]


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    for f in FILES:
        local = os.path.join(PROJECT_ROOT, f)
        if not os.path.exists(local):
            print(f"SKIP missing {f}", flush=True)
            continue
        remote = f if not f.startswith("huggingface/") else f.split("/", 1)[1]
        # root layout: huggingface/* -> root; src/* stays
        if f.startswith("huggingface/"):
            remote = {"Dockerfile": "Dockerfile", "app.py": "app.py",
                      "requirements-hf.txt": "requirements-hf.txt",
                      "README.md": "README.md"}[f.split("/", 1)[1]]
        api.upload_file(repo_id=SPACE_ID, repo_type="space", token=TOKEN,
                        path_or_fileobj=local, path_in_repo=remote,
                        commit_message=f"FlyBrain V5: {f} (optional-dep guards, v1, backup)")
        print(f"uploaded {remote}", flush=True)
    print("stage:", api.get_space_runtime(SPACE_ID, token=TOKEN).stage, flush=True)


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("HF_TOKEN is required")
    main()
