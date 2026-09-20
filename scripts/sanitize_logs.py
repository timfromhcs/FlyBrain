#!/usr/bin/env python3
"""Strip machine-specific paths/usernames from committed diagnostic logs."""
import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [os.path.join(PROJECT_ROOT, "diagnostics", "huggingface_verification.json"),
           os.path.join(PROJECT_ROOT, "diagnostics", "environment_report.json"),
           os.path.join(PROJECT_ROOT, "diagnostics", "milestones", "milestone_evidence.json")]


def clean(text: str) -> str:
    text = text.replace("E:\\FlyBain LLM Evolve", "<REPO>")
    text = text.replace("E:/FlyBain LLM Evolve", "<REPO>")
    text = re.sub(r"[A-Za-z]:\\+Users\\+[^\\/:\"']+", "<HOME>", text)
    return text


def main():
    for path in TARGETS:
        doc = json.load(open(path, encoding="utf-8"))
        blob = json.dumps(doc)
        blob = clean(blob)
        json.dump(json.loads(blob), open(path, "w", encoding="utf-8"), indent=2)
        leftovers = [m for m in re.findall(r"[A-Za-z]:\\(?:Users|FlyBain)[^\"']{0,40}", blob)]
        print(path, "leftover machine paths:", leftovers if leftovers else "none")


if __name__ == "__main__":
    main()
