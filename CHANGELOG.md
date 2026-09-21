# FlyBrain Changelog

All notable changes to the FlyBrain framework will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [9.0.0] - 2026-09-21

### Added
- **Generative Endless World Engine (`src/world/chunks/`)**:
  - Deterministic chunk streaming (`ChunkManager`, `WorldManager`, `WorldGenerator`) with seeded hash-coordinate generator.
  - Multi-tier chunk persistence with delta overlays (`WorldDeltaStore`) and spatial bounds clamping.
  - Integration with `/api/v1/world/chunks` endpoint for streaming chunk tiles, elevation, and biome features to the workstation UI.
- **FlyAsset Compiler (`src/assets/compiler/`)**:
  - Validates and packages 3D assets to `FlyAsset v1` specification.
  - Automated collision proxy compilation (Box, Capsule, Sphere, Convex Hull) with scale normalization to metric units.
  - Physics material binding (wood, stone, water, foliage) with MuJoCo contact friction parameters.
- **Closed-Loop AI -> World -> Organism Pipeline (`src/world/planner.py`, `scripts/verify_v9_loop.py`)**:
  - Full acceptance loop: Natural language intent -> LLM structured WorldIntent -> Procedural World Planner -> Asset Compiler -> MuJoCo Placement -> Connectome Navigation -> Memory encoding.
- **Central Resource Manager (`src/runtime/resource_manager/`)**:
  - Hardware probing for CPU cores, RAM, Disk, and Vulkan GPU.
  - 5-Tier dynamic profiles (`MINIMAL`, `LOW`, `BALANCED`, `PERFORMANCE`, `MAXIMUM`).
  - Memory pressure state machine (`GREEN`, `YELLOW`, `ORANGE`, `RED`, `CRITICAL`) with hysteresis and bounded OOM recovery.
- **Central Capability Manager (`src/runtime/capability_manager.py`)**:
  - Scientific honest capability reporting (`AVAILABLE`, `DEGRADED`, `UNAVAILABLE`, `NOT_APPLICABLE`) with zero mock.
- **Model Registry V8/V9 (`models/registry.yaml`, `src/models/registry_v8.py`)**:
  - Unified SHA-256 verified registry for LLM, VLM, STT, TTS, Diffusion (DreamShaper-8-LCM), and ControlNet Canny.
- **Provider-Independent Backup System (`src/backup/provider.py`)**:
  - Cryptographic archive validation for local filesystem, portable zip archives, and remote Hugging Face Storage.
- **Packaging, Installer & Diagnostics Tooling**:
  - PowerShell one-line installer (`scripts/install.ps1`), system doctor (`scripts/doctor.ps1`), uninstaller (`scripts/uninstall.ps1`).
  - Automated release packager (`scripts/package_release.py`) emitting portable zip, SBOM, SHA-256 sums, and release manifest.
  - GitHub Actions cloud build workflow (`.github/workflows/build-windows.yml`).

---

## [8.0.0] - 2026-09-21

### Added
- Architectural transition release:
  - Hardware probing and multi-tier resource budgeting.
  - Unified typed event bus (`src/common/events.py`) alongside legacy biological EventLog.
  - Initial chunk manager and FlyAsset compiler specification.
  - Third-party license audit (`docs/THIRD_PARTY_LICENSES.md`).

---

## [7.0.0] - 2026-09-21

### Added
- Stable production release:
  - Local multimodal perception loop (Qwen 0.6B, SmolVLM 256M, Whisper small, Kokoro 82M, DreamShaper-8-LCM).
  - Sub-second microsecond timestamp collision resolution for tamper-evident backup and restore.
  - Complete MuJoCo 3.13 physical embodiment and Vulkan compute shader parity.
  - Cloud deployment to Hugging Face Spaces (`timfromhcs/FlyBrain-Lab`).
