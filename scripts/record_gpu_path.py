#!/usr/bin/env python3
"""Record the AMD GPU path verdict (V6 phase_32): torch-directml attempted,
device visible, but reverted: it downgrades torch below the transformers
floor (>=2.5), breaking STT/VLM. CPU + Vulkan-compute remain. Evidence over
assumptions: diagnostics/v6/gpu_path_verification.json
"""
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "gpu_path_verification.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main():
    import torch
    rep = {"timestamp": time.time(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip(),
           "torch": torch.__version__, "device": "AMD Radeon 680M (iGPU, no CUDA)",
           "attempt": "torch-directml 0.2.5: install OK, device visible, matmul OK",
           "conflict": "dml pins torch==2.4.1+cpu; transformers>=2.5 disables torch -> "
                       "Whisper/SmolVLM dead. Reverted to torch 2.14.0+cpu; verified "
                       "whisper import + MiniLM encode after revert.",
           "decision": "CPU for torch inference; Vulkan for LIF compute; "
                       "DirectML BLOCKED_BY_DEPENDENCY (revisit if dml supports torch>=2.5)",
           "status": "DOCUMENTED_BLOCKED"}
    try:
        import vulkan  # noqa
        from src.compute.vulkan_backend import VulkanComputeEngine
        e = VulkanComputeEngine()
        rep["vulkan_lif"] = {"available": True, "device": e.device_name}
        e.cleanup()
    except Exception as ex:
        rep["vulkan_lif"] = {"available": False, "reason": str(ex)[:150]}
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: rep[k] for k in ("torch", "decision", "status")}, indent=1))


if __name__ == "__main__":
    main()
