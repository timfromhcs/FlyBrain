#!/usr/bin/env python3
"""Generate software and model license audit for FlyBrain V8/V9."""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

audit = {
    "audit_date": "2026-09-21",
    "compliance_status": "PASS",
    "policy": "Local-First / EU Distribution Compliance / Strict Licensing",
    "components": [
        {
            "component": "FlyBrain Core",
            "version": "7.0.0",
            "license": "Apache-2.0",
            "redistribution": "PERMITTED",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "INCLUDED"
        },
        {
            "component": "Janelia MaleCNS v1.0",
            "version": "1.0",
            "license": "CC-BY-4.0",
            "redistribution": "PERMITTED_WITH_ATTRIBUTION",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "INCLUDED_RAW_CSV"
        },
        {
            "component": "MuJoCo",
            "version": "3.13.0",
            "license": "Apache-2.0",
            "redistribution": "PERMITTED",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "DEPENDENCY"
        },
        {
            "component": "Vulkan Python Bindings",
            "version": "1.3.275.1",
            "license": "Apache-2.0",
            "redistribution": "PERMITTED",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "DEPENDENCY"
        },
        {
            "component": "Qwen3-0.6B-GGUF",
            "version": "main",
            "license": "Apache-2.0",
            "model_license": "Apache-2.0",
            "redistribution": "PERMITTED",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "STANDALONE_MODEL_STORE"
        },
        {
            "component": "DreamShaper-8-LCM",
            "version": "main",
            "license": "CreativeML-OpenRAIL-M",
            "model_license": "OpenRAIL-M",
            "redistribution": "PERMITTED_NON_MILITARY",
            "commercial_use": "PERMITTED_WITH_RESTRICTIONS",
            "eu_distribution": "PERMITTED",
            "bundling": "STANDALONE_MODEL_STORE"
        },
        {
            "component": "Whisper-small",
            "version": "main",
            "license": "MIT",
            "model_license": "MIT",
            "redistribution": "PERMITTED",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "STANDALONE_MODEL_STORE"
        },
        {
            "component": "Kokoro-82M",
            "version": "main",
            "license": "Apache-2.0",
            "model_license": "Apache-2.0",
            "redistribution": "PERMITTED",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "STANDALONE_MODEL_STORE"
        },
        {
            "component": "TRELLIS (Microsoft)",
            "version": "main",
            "license": "MIT",
            "model_license": "MIT",
            "redistribution": "PERMITTED",
            "commercial_use": "PERMITTED",
            "eu_distribution": "PERMITTED",
            "bundling": "NATIVE_BACKEND_OPTIONAL"
        },
        {
            "component": "Hunyuan3D-2.0/2.1 (Tencent)",
            "version": "2.0/2.1",
            "license": "Tencent Hunyuan Community License",
            "model_license": "Tencent Hunyuan Community License (Excludes EU, UK, KR)",
            "redistribution": "RESTRICTED",
            "commercial_use": "RESTRICTED",
            "eu_distribution": "PROHIBITED_FOR_PRODUCTION",
            "bundling": "REFERENCE_ONLY",
            "notes": "Explicitly flagged as REFERENCE_ONLY due to EU/Germany territorial exclusion clause in Tencent Community License."
        }
    ]
}

os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
audit_path = os.path.join(PROJECT_ROOT, "reports", "license-audit.json")
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

doc_path = os.path.join(PROJECT_ROOT, "docs", "THIRD_PARTY_LICENSES.md")
with open(doc_path, "w", encoding="utf-8") as f:
    f.write("# FlyBrain Third-Party Software & Model License Audit\n\n")
    f.write("**Audit Date:** 2026-09-21 | **Compliance Status:** PASS\n\n")
    f.write("| Component | Version | License | EU Distribution | Bundling Policy |\n")
    f.write("| :--- | :--- | :--- | :---: | :--- |\n")
    for c in audit["components"]:
        f.write(f"| **{c['component']}** | {c['version']} | {c['license']} | {c['eu_distribution']} | {c['bundling']} |\n")
    f.write("\n## Special Legal & Territorial Compliance Notes\n\n")
    f.write("### Tencent Hunyuan3D-2.0/2.1 Community License\n")
    f.write("In accordance with Section 103 of the FlyBrain Autonomous Build Specification, Tencent Hunyuan3D-2.0/2.1 contains explicit geographical exclusions (EU, UK, South Korea). Therefore, it is strictly classified as `REFERENCE_ONLY` and is **never** bundled or required for European/German production builds.\n\n")
    f.write("### Production 3D Generation Policy\n")
    f.write("For distributable production 3D generation, FlyBrain uses MIT-licensed TRELLIS/trellis.cpp architectures or procedural collision/mesh synthesis pipelines.\n")

print(f"Generated {audit_path} and {doc_path}")
