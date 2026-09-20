#!/usr/bin/env python3
"""V5 phase_19: fail on stale documentation claims.

Scans living docs for obsolete totals/versions/claims. Frozen historical
files (final_verification_report.md v4.0.0 record) are excluded ONLY because
they carry an explicit supersession banner; everything else must be current.
"""
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# file -> allowed stale patterns (with justification); all other files: zero tolerance.
HISTORICAL_EXCEPTIONS = {
    "final_verification_report.md": ["44/44", "4.0.0", "v4.0.0", "32/32", "111 tests"],
    "RELEASE_NOTES.md": ["v4.0.0", "4.0.0", "4.1.0"],
    "diagnostics/pre_release_audit.json": ["4.0.0", "44/44", "1.0.0"],
    "scripts/generate_pre_release_audit.py": ["4.0.0", "44/44", "1.0.0"],
    "scripts/capture_v4_baseline.py": ["v4.0.0"],
}

STALE_PATTERNS = [
    r"44/44", r"19/19", r"25/25", r"32/32", r"57/57(?!\s*categor)",  # totals must come from reports
    r"pre-recorded command buffers",
    r"[Bb]it-exact parity", r"bit-exact numerical parity",
    r"cdnjs\.cloudflare\.com.*three", r"cdn\.jsdelivr\.net.*three",
    r"MODEL_UNAVAILABLE.*when weights absent.*OPERATIONAL",  # inverted honesty
]

LIVING_DOCS = ["README.md", "ARCHITECTURE.md", "REPRODUCIBILITY.md",
               "RESEARCH_STATUS.md", "RELEASE_NOTES.md", "TESTING.md",
               "TROUBLESHOOTING.md", "CONTRIBUTING.md", "SECURITY.md",
               "DEVELOPMENT.md", "final_verification_report.md",
               "deployment/HUGGINGFACE.md", "docs/alife_architecture.md"]


def main():
    violations = []
    for doc in LIVING_DOCS:
        path = os.path.join(PROJECT_ROOT, doc)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        allowed = HISTORICAL_EXCEPTIONS.get(doc, [])
        for pat in STALE_PATTERNS:
            for m in re.finditer(pat, text):
                line = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
                if any(a in line for a in allowed):
                    continue
                # generated acceptance tables cite the canonical report: allowed
                if "render_status_tables" in line or "acceptance_matrix.json" in line:
                    continue
                violations.append(f"{doc}: stale pattern {pat!r} near ...{line.strip()[:100]}...")
    if violations:
        print(f"STALE DOCS: {len(violations)} violations:")
        for v in violations[:20]:
            print("  -", v)
        sys.exit(1)
    print(f"stale-doc check PASS ({len(LIVING_DOCS)} docs scanned)")


if __name__ == "__main__":
    main()
