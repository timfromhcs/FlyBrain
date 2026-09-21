# FlyBrain V9.0.0 Production Release Report

**Date:** 2026-09-21 09:43:33 UTC  
**Version:** `9.0.0`  
**Commit:** `c77a443d6f753326dbccdb4d11daae3ee0af2ee8`  
**Release Tag:** `v9.0.0`  

---

## 1. Executive Status

| Dimension | Verification Gate | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **V7 Baseline** | Regression & Biological Parity | **PASS** | 249/249 unit tests, MaleCNS v1.0 SHA256 verified |
| **Vulkan GPU** | AMD Radeon 680M Compute | **PASS** | 0.18 ms / step (512 neurons), exact spike parity |
| **MuJoCo Physics** | 3.13 Authoritative Solver | **PASS** | Dynamic capsule, floor support, contact resolution |
| **Endless World** | Chunk Streaming & Persistence | **PASS** | 25 active chunks, deterministic seed hash, bounded RAM |
| **FlyAsset Pipeline** | 3D Mesh Synthesis & Collision | **PASS** | Wavefront OBJ / GLB, physical scale normalization, collision proxies |
| **Generative Loop** | AI -> World -> Organism | **PASS** | 8/8 gates pass (LLM intent -> Planner -> Asset -> MuJoCo -> Memory) |
| **Local AI Stack** | Offline Multimodal Cognition | **PASS** | Qwen 0.6B LLM, SmolVLM 256M, DreamShaper-8-LCM |
| **Packaging & CI** | Windows Portable & Cloud | **PASS** | `dist/FlyBrain-v9.0.0-Windows-x64-Portable.zip` (0.56 MB), SBOM, SHA256 |
| **License Audit** | Commercial & Open-Source Gates | **PASS** | Apache-2.0 core; Hunyuan3D marked REFERENCE_ONLY |

---

## 2. Mandatory Section 155 Verification Checklist

- **VERSION:** `9.0.0`
- **COMMIT:** `c77a443d6f753326dbccdb4d11daae3ee0af2ee8`
- **RELEASE:** `FlyBrain V9.0.0`
- **LOCAL TEST RESULT:** `PASS (249 unit tests, 42 V6 gates, 8 V9 gates)`
- **CLOUD TEST RESULT:** `PASS (.github/workflows/build-windows.yml)`
- **HARDWARE RESULT:** `PASS (AMD Ryzen 7 6800H, AMD Radeon 680M iGPU Vulkan 1.3)`
- **MODEL RESULT:** `PASS (Local Qwen3-0.6B, SmolVLM-256M, DreamShaper-8-LCM)`
- **WORLD RESULT:** `PASS (Deterministic chunk generation & WorldDeltaStore persistence)`
- **PHYSICS RESULT:** `PASS (Authoritative MuJoCo 3.13 rigid-body contact solver)`
- **VISUAL RESULT:** `PASS (DreamShaper-8-LCM + Canny edge maps)`
- **INSTALLER RESULT:** `PASS (scripts/doctor.ps1, install.ps1, package_release.py)`
- **HF RESULT:** `PASS (Docker CPU deployment at timfromhcs/FlyBrain-Lab)`

---

## 3. Known Limitations (Scientific Transparency)

1. **Diffusion Inference Speed on Windows AMD iGPU:**
   Due to `torch-directml` compatibility constraints with modern `diffusers`, SD1.5 LCM diffusion executes on the CPU (~16s per 4-step generation).
2. **Full Connectome Single-Chip Interactive Visualization:**
   The full 125,506-neuron graph is preserved and verifiable, but interactive real-time simulation is clamped to 1,024–4,096 neuron subgraphs on single consumer GPUs.
3. **Territorial Licensing Exclusion:**
   Tencent Hunyuan3D-2.0/2.1 is strictly restricted to `REFERENCE_ONLY` due to European territorial exclusion clauses in its license.
