#!/usr/bin/env python3
"""Bulk-sync Space repo with working tree (deltas only), then report stage."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("FLYBRAIN_SPACE_ID", "timfromhcs/FlyBrain-Lab")
TOKEN = os.environ.get("HF_TOKEN", "")


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        repo_id=SPACE_ID, repo_type="space", token=TOKEN,
        folder_path=PROJECT_ROOT, path_in_repo=".",
        allow_patterns=["huggingface/**", "src/**", "shaders/*", "manifests/*",
                        "malecns/data-raw/*.csv"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="FlyBrain V5.0.0 Space sync (v1 API, backup, watchdog, fixed tabs)",
    )
    print("sync uploaded", flush=True)
    # root layout files
    for local, remote in [("huggingface/Dockerfile", "Dockerfile"),
                          ("huggingface/app.py", "app.py"),
                          ("huggingface/requirements-hf.txt", "requirements-hf.txt"),
                          ("huggingface/README.md", "README.md")]:
        api.upload_file(repo_id=SPACE_ID, repo_type="space", token=TOKEN,
                        path_or_fileobj=os.path.join(PROJECT_ROOT, local),
                        path_in_repo=remote, commit_message=f"V5 Space root: {remote}")
        print(f"root {remote}", flush=True)
    print("stage:", api.get_space_runtime(SPACE_ID, token=TOKEN).stage, flush=True)


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("HF_TOKEN is required")
    main()
