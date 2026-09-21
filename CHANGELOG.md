# FlyBrain Changelog

All notable changes to the FlyBrain framework will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [10.0.0] - 2026-09-21

### Added
- **Canonical Dataset Registry (`src/connectome/dataset_registry.py`)**:
  - Formal scientific decoupling between full Janelia MaleCNS v1.0 biological data (125,506 somas, 99,301 connections, CC-BY-4.0) and lightweight derived subgraphs or synthetic surrogates.
  - Strict provenance hashing binding dataset identity into experimental records.
- **Genuine Biological Closed-Loop Autonomy (`src/world/embodied_loop.py`)**:
  - Closed-loop causal loop: Perception -> LIF SNN -> Neural Firing Rates -> Motor Propulsion -> MuJoCo Contact -> Spatial Memory -> Reward.
  - Mandatory labelling: `policy_source="BIOLOGICAL_CLOSED_LOOP"` for organism control and `policy_source="TEST_DRIVER_ONLY"` for diagnostic harnesses.
  - Full causal trace output (`diagnostics/v10_closed_loop_report.json`).
- **Persistent World Chunk Modifications (`src/world/chunks/chunk_manager.py`)**:
  - Deterministic serialization of chunk delta overlays (`data/world_chunks/chunk_{cx}_{cy}_deltas.json`) with event hashes and timestamps.
  - Full modification persistence surviving process restart and reload.
- **Binary glTF (GLB) Export & 3D Geometry Validation (`src/assets/mesh_generator.py`)**:
  - Native binary glTF 2.0 (.glb) exporter with aligned BIN/JSON chunks.
  - Rigorous mesh validation checking for degenerate zero-area triangles, duplicate vertices, and non-manifold topology.
  - Portable relative paths normalized across Windows and Linux platforms.
- **Model Inference Honesty (`src/world/planner.py`)**:
  - Elimination of bare `except Exception: pass` swallows; structured error capture.
  - Explicit inference provenance reporting (`LOCAL_LLM` vs `RULE_BASED_FALLBACK`).
- **Self-Healing Installer (`installer/install.ps1`)**:
  - Dynamic canonical version injection (10.0.0).
  - Checksum-verified `-Repair` mode restoring missing or corrupted configuration files and directory structures.
  - Verified clean-room installation tests (`tests/test_installer_self_healing.py`).
- **Unified V10 56-Gate Release Matrix (`scripts/run_v10_acceptance_matrix.py`)**:
  - Machine-readable release gate verification ensuring fail-closed zero-mock integrity.

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
