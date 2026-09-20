#!/usr/bin/env python3
"""Render acceptance tables into docs from the canonical machine-readable report.

Single source of truth: verification/acceptance_schema.json (categories) +
diagnostics/acceptance_matrix.json (results). README.md and RESEARCH_STATUS.md
contain only markers; totals are never hard-coded:

    <!-- ACCEPTANCE-TABLE-START -->
    ... generated ...
    <!-- ACCEPTANCE-TABLE-END -->

Usage: render_status_tables.py [--check]  (--check fails if docs are stale)
"""
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = "<!-- ACCEPTANCE-TABLE-START -->"
END = "<!-- ACCEPTANCE-TABLE-END -->"


def build_table(report):
    cats = report["categories"]
    lines = [
        f"> Canonical source: `verification/acceptance_schema.json` → "
        f"`diagnostics/acceptance_matrix.json` "
        f"(overall **{report['overall_status']}**: "
        f"**{report['passed']}/{report['total_categories']} PASS**, "
        f"{report['failed']} FAIL, {report['skipped']} SKIP). "
        f"Do not copy totals elsewhere — regenerate with "
        f"`scripts/render_status_tables.py`.",
        "",
        "| Index | Category | Status | Details |",
        "| :---: | :--- | :---: | :--- |",
    ]
    for i, (name, res) in enumerate(cats.items(), 1):
        status = res["status"]
        badge = "**PASS**" if status == "PASS" else (
            f"**{status}**" if status == "FAIL" else f"*{status}*")
        reason = res.get("reason", "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {i} | `{name}` | {badge} | {reason} |")
    return "\n".join(lines)


def render(path, table, check):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if START not in text or END not in text:
        print(f"SKIP {path}: markers not found")
        return False
    new = (text[:text.index(START) + len(START)] + "\n" + table + "\n"
           + text[text.index(END):])
    if new == text:
        print(f"OK {path}: up to date")
        return False
    if check:
        print(f"STALE {path}: regenerate with scripts/render_status_tables.py")
        return True
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"WROTE {path}")
    return False


def main():
    check = "--check" in sys.argv
    report = json.load(open(os.path.join(PROJECT_ROOT, "diagnostics",
                                         "acceptance_matrix.json"), encoding="utf-8"))
    table = build_table(report)
    stale = False
    for doc in ("README.md", "RESEARCH_STATUS.md"):
        stale |= render(os.path.join(PROJECT_ROOT, doc), table, check)
    sys.exit(1 if (check and stale) else 0)


if __name__ == "__main__":
    main()
