#!/usr/bin/env python3
"""Upload root-level Space files only (fast, bounded). Triggers Space rebuild."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("FLYBRAIN_SPACE_ID", "timfromhcs/FlyBrain-Lab")
TOKEN = os.environ.get("HF_TOKEN", "")


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    for local, remote in [("huggingface/Dockerfile", "Dockerfile"),
                          ("huggingface/app.py", "app.py"),
                          ("huggingface/requirements-hf.txt", "requirements-hf.txt"),
                          ("huggingface/README.md", "README.md")]:
        api.upload_file(repo_id=SPACE_ID, repo_type="space", token=TOKEN,
                        path_or_fileobj=os.path.join(PROJECT_ROOT, local),
                        path_in_repo=remote,
                        commit_message="FlyBrain v4.1.0 Space: optional-dep guards + vulkan bindings")
        print(f"uploaded {remote}", flush=True)
    rt = api.get_space_runtime(SPACE_ID, token=TOKEN)
    print("stage:", rt.stage, flush=True)


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("HF_TOKEN is required")
    main()
