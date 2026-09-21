# FlyBrain V10 Master Implementation & Verification Plan

**Canonical Target Version:** `10.0.0`  
**Execution Agent:** `SHWTY-FlyBrain-V10`  
**Target Architecture:** Vulkan-first, Local-first, Physically Authoritative, Biologically Authentic  
**Baseline Git Commit:** `98e1c5d41f486a43b4690e54256241a79857aa75`  
**Date:** 2026-09-21  

---

## 1. Executive Summary & Audit Baseline (Phase 0)

A comprehensive repository inspection was conducted across the source tree, documentation, configuration files, workflows, and diagnostic artifacts.

### 1.1 Key Audit Findings
1. **Version Fragmentation:** Version identifiers are inconsistently distributed:
   - `src/version.py`: `9.0.0`
   - `pyproject.toml`: `9.0.0`
   - `package.json`: `9.0.0`
   - `installer/install.ps1`: hardcoded `version = "8.0.0"`
   - `installer/doctor.ps1`: references `V8/V9`
   - `src/world/chunks/chunk_manager.py`: hardcoded `revision = "v8.0"`
   - `RESEARCH_STATUS.md`: references `v6.0.0` in headers
2. **Embodied Closed-Loop Autonomy Gap:** In `src/world/integration.py`, the agent was propelled across the bridge via direct programmatic drive commands (`world.physics.drive_character("hero", 0.0, 0.8)`), bypassing the LIF brain and sensory-motor neural control loop.
3. **World Modification Persistence Gap:** `ChunkManager` declared an in-memory dictionary `self.modifications_db` and created `self.storage_dir`, but never read or wrote chunk modification deltas to disk. All world modifications were lost upon process exit.
4. **Asset Pipeline & 3D Geometry Gap:**
   - `src/assets/mesh_generator.py` claimed GLB/GLTF binary export in docstrings, but only implemented Wavefront OBJ.
   - Validation omitted degenerate triangle detection (zero-area faces, duplicate vertex indices) and manifold topology verification.
   - Railing generation in `generate_bridge` contained an array index out-of-bounds defect on railing faces (`base_idx + 2`, `base_idx + 3`).
   - Manifests stored absolute Windows paths (`E:\FlyBain LLM Evolve\...`), breaking cross-host reproducibility.
5. **Model Execution Honesty Gap:** `src/world/planner.py` swallowed all model errors via `except Exception: pass` and fell back to heuristic parsing without recording inference provenance (`LOCAL_LLM` vs `RULE_BASED_FALLBACK`), model ID, revision, or error details.
6. **Dataset & Scientific Registry Gap:** The Janelia MaleCNS v1.0 biological dataset (125,506 somas, 99,301 connections) lacked a formal `DatasetRegistry` tracking license separation (CC-BY-4.0 vs Apache-2.0), upstream vs derived provenance, and categorical tagging (`BIOLOGICAL`, `DERIVED`, `SURROGATE`, `SYNTHETIC`).
7. **Dependency Drift:** `pyproject.toml` was missing `mujoco`, `kokoro`, `sentence-transformers`, and `llama-cpp-python`, which were present in `requirements.txt`.
8. **Documentation Verification Gap:** `scripts/verify_docs_consistency.py` evaluated only 7 static checks, omitting version consistency, dataset consistency, release commit verification, installer checks, and path portability.

---

## 2. Prioritized Implementation Tasks (P0–P3)

### TASK-V10-VER-01 (P0: Release Blocker) — Canonical Version 10.0.0 Unification
- **Problem:** Stale and mismatched version strings throughout the codebase.
- **Root Cause:** Decentralized version updates without build-time single-source enforcement.
- **Target Files:**
  - `src/version.py`
  - `pyproject.toml`
  - `package.json`
  - `installer/install.ps1`
  - `installer/doctor.ps1`
  - `src/world/chunks/chunk_manager.py`
  - `docs/*`, `README.md`, `RESEARCH_STATUS.md`, `GEMINI.md`
- **Implementation Method:** Set `VERSION = "10.0.0"` in `src/version.py`. Update `pyproject.toml` and `package.json`. Make installer read canonical metadata or dynamically inject `10.0.0`.
- **Tests Required:** `tests/test_version_consistency.py`
- **Evidence Required:** `diagnostics/v10_documentation_report.json`
- **Completion Condition:** Zero occurrences of stale active version strings in core engine, installer, and docs.

### TASK-V10-DATA-02 (P0: Release Blocker) — Canonical Dataset Registry & Scientific Truth
- **Problem:** Absence of formal dataset provenance separation and explicit biological vs surrogate classification.
- **Root Cause:** Connectome loading code read raw CSVs directly without metadata contracts.
- **Target Files:**
  - `src/connectome/dataset_registry.py` (New)
  - `src/connectome/loader.py`
  - `src/connectome/types.py`
  - `manifests/malecns_provenance.json`
  - `tests/test_dataset_registry.py` (New)
- **Implementation Method:** Implement `DatasetRegistry` defining:
  - Upstream Janelia MaleCNS v1.0 (125,506 somas, 99,301 synapses, CC-BY-4.0)
  - Local Light/Real Subgraph exports with exact cryptographic sha256 checksums
  - Category tags: `BIOLOGICAL`, `DERIVED`, `SURROGATE`, `SYNTHETIC`, `EMERGENT`, `EXPERIMENTAL`, `HEURISTIC`, `UNKNOWN`
  - License decoupling: dataset license separate from Apache-2.0 software license.
- **Tests Required:** Unit tests verifying registry lookup, file hashing, and provenance change when dataset identity alters.
- **Evidence Required:** `diagnostics/v10_dataset_provenance.json`
- **Completion Condition:** All connectome experiments explicitly identify dataset ID; dataset tests pass.

### TASK-V10-LOOP-03 (P0: Release Blocker) — True Closed-Loop Autonomy & Test Driver Tagging
- **Problem:** Acceptance loop used `drive_character` to bypass the biological controller.
- **Root Cause:** Shortcut in acceptance testing to force movement without sensory-neural propagation.
- **Target Files:**
  - `src/world/embodied_loop.py` (New)
  - `src/world/integration.py`
  - `src/world3d/agent.py`
  - `src/world3d/physics.py`
  - `tests/test_embodied_loop.py` (New)
- **Implementation Method:**
  - Build `EmbodiedClosedLoop` implementing: `Sensory Perception` -> `LIF Brain Step` -> `Neural Firing Rates` -> `Motor Action` -> `MuJoCo Physics Step` -> `Contact / Spatial Memory Update`.
  - Tag any direct test-only bypass as `TEST_DRIVER_ONLY`.
  - Record step-by-step causal trace (`policy_source`, `sensory_input`, `neural_output`, `motor_output`, `physical_action`, `world_result`).
- **Tests Required:** Test proving organism navigates and contacts structure through neural motor outputs.
- **Evidence Required:** `diagnostics/v10_closed_loop_report.json`
- **Completion Condition:** Full sensory-to-physics causal loop verified; no direct drive used without `TEST_DRIVER_ONLY` tag.

### TASK-V10-MEM-04 (P0: Release Blocker) — Persistent World Modifications & Restart Replay
- **Problem:** Chunk modifications were stored in memory only. Process restart wiped state.
- **Root Cause:** Unimplemented file persistence in `ChunkManager`.
- **Target Files:**
  - `src/world/chunks/chunk_manager.py`
  - `src/world3d/spatial_memory.py`
  - `tests/test_v10_memory_persistence.py` (New)
- **Implementation Method:**
  - Write persistent chunk delta overlays to `data/world_chunks/{cx}_{cy}_deltas.json` with sha256 event hashes and timestamps.
  - Load delta overlays on chunk generation/demand.
  - Implement restart-replay test: write modification -> terminate/unload -> reload -> verify presence and agent behavior.
- **Tests Required:** Test writing modifications, tearing down manager, reinitializing, and verifying exact delta restoration.
- **Evidence Required:** `diagnostics/v10_world_persistence_test.json`
- **Completion Condition:** Process restart preserves world state across restarts without loss.

### TASK-V10-STREAM-05 (P1: Major Correctness) — Deterministic Endless World Streaming
- **Problem:** Stale generator revision `v8.0`, lack of strict bounded active-chunk assertions.
- **Root Cause:** Hardcoded parameter in `WorldGenerator` and missing streaming invariant tests.
- **Target Files:**
  - `src/world/chunks/chunk_manager.py`
  - `tests/test_chunk_streaming.py`
- **Implementation Method:**
  - Introduce `WorldGeneratorRevision.V10_TERRAIN`.
  - Enforce bounded memory cache (`<= max_loaded_chunks`).
  - Prove determinism: `hash(seed, cx, cy, generator_revision)` produces identical heightmaps and props.
- **Tests Required:** Coordinate determinism test and cache eviction bound test.
- **Evidence Required:** Part of `diagnostics/v10_acceptance_matrix.json`.
- **Completion Condition:** Streaming maintains `< 25` chunks active; deterministic hashes pass.

### TASK-V10-ASSET-06 (P1: Major Correctness) — Binary GLB Export, Geometry Validation & Path Portability
- **Problem:** Missing binary GLB export; missing degenerate triangle / non-manifold validation; railing indexing defect; absolute paths in manifests.
- **Root Cause:** Incomplete initial 3D synthesizer.
- **Target Files:**
  - `src/assets/mesh_generator.py`
  - `src/assets/compiler/compiler.py`
  - `src/assets/resolver.py`
  - `tests/test_asset_pipeline.py`
- **Implementation Method:**
  - Implement `export_glb(filepath: str)` generating valid glTF 2.0 binary chunks.
  - Implement geometry validation checking: zero-area triangles, duplicate vertices, non-manifold edges.
  - Fix railing index math in `generate_bridge`.
  - Store relative portable paths in asset package manifests.
  - Apply compiled mass/friction/restitution authoritatively to MuJoCo physics geoms.
- **Tests Required:** Asset roundtrip test (synthesize -> export OBJ + GLB -> validate geometry -> compile -> load in MuJoCo).
- **Evidence Required:** `diagnostics/v10_asset_validation.json`
- **Completion Condition:** Valid GLB output; 0 degenerate triangles; portable manifests; authoritative physics.

### TASK-V10-HONEST-07 (P1: Major Correctness) — Model Inference Honesty & Structured Provenance
- **Problem:** Silent fallbacks and bare exception swallows in planner.
- **Root Cause:** Catch-all exception block returning `None` instead of structured diagnostics.
- **Target Files:**
  - `src/world/planner.py`
  - `src/models/registry.py`
  - `src/models/manager.py`
  - `tests/test_model_honesty.py`
- **Implementation Method:**
  - Return `InferenceProvenance`: `source` (`LOCAL_LLM`, `RULE_BASED_FALLBACK`, `LOCAL_VLM`, `PROCEDURAL`, `UNAVAILABLE`), `model_id`, `quantization`, `backend`, `error_detail`.
  - Remove bare `except Exception: pass`; log structured failure reason.
- **Tests Required:** Honesty test verifying fallback is explicitly labelled as `RULE_BASED_FALLBACK` when weights are absent.
- **Evidence Required:** Recorded in `diagnostics/v10_closed_loop_report.json`.
- **Completion Condition:** 100% honesty in model telemetry; no fake LLM execution.

### TASK-V10-VULKAN-08 (P1: Major Correctness) — Vulkan & CPU Numerical vs Spike-Exact Parity
- **Problem:** Loose documentation claims regarding "bit-exact" floating-point parity between CPU and Vulkan GPU.
- **Root Cause:** Ambiguous terminology conflating spike-time exactness with IEEE 754 float precision.
- **Target Files:**
  - `src/compute/cpu_reference.py`
  - `src/compute/vulkan_engine.py`
  - `tests/test_vulkan_parity.py`
  - Documentation files
- **Implementation Method:**
  - Define three distinct precision tiers: `NUMERICAL_EQUIVALENCE` (L1/L2 within epsilon), `SPIKE_EXACT` (100% identical spike trains), and `BIT_EXACT` (CPU deterministic replay).
  - Assert spike-exact parity between Vulkan and CPU baseline.
  - Disallow "bit-exact" claims for GPU floating-point outputs.
- **Tests Required:** Parity test asserting exact spike rasters and bounded membrane potential deltas.
- **Evidence Required:** Part of `diagnostics/v10_acceptance_matrix.json`.
- **Completion Condition:** Mathematically sound, honestly labelled parity assertions.

### TASK-V10-INSTALL-09 (P1: Major Correctness) — Self-Healing Installer & Isolated Artifact Test
- **Problem:** Installer used hardcoded version `8.0.0`; lacked automatic repair and scratch installation verification.
- **Root Cause:** Minimal installer script.
- **Target Files:**
  - `installer/install.ps1`
  - `installer/doctor.ps1`
  - `scripts/verify_installer_self_healing.py` (New)
  - `tests/test_installer_self_healing.py` (New)
- **Implementation Method:**
  - Update installer to version `10.0.0`.
  - Implement `-Repair` switch that checks file checksums against release manifest and restores missing/corrupt components.
  - Add test that provisions a fresh temporary directory, simulates file corruption, invokes repair, and tests launch.
- **Tests Required:** Automated test running install, intentionally deleting files, running repair, and testing health.
- **Evidence Required:** `diagnostics/v10_install_test.json`
- **Completion Condition:** Installer passes fresh install and self-healing repair tests in clean directory.

### TASK-V10-DEP-10 (P1: Major Correctness) — Dependency Lock Synchronization
- **Problem:** Drift between `pyproject.toml`, `requirements.txt`, and `huggingface/requirements-hf.txt`.
- **Root Cause:** Ad-hoc package additions.
- **Target Files:**
  - `pyproject.toml`
  - `requirements.txt`
  - `huggingface/requirements-hf.txt`
- **Implementation Method:**
  - Align `pyproject.toml` with complete workstation dependencies.
  - Keep `huggingface/requirements-hf.txt` strictly pinned for CPU-only Docker environment.
- **Tests Required:** Test checking dependency file synchronization.
- **Evidence Required:** `diagnostics/v10_dependency_report.json`
- **Completion Condition:** Full alignment between project dependency declarations.

### TASK-V10-DOCS-11 (P0: Release Blocker) — Documentation Consistency & Unified Verification
- **Problem:** Outdated consistency checker missing version, dataset, and release verification.
- **Root Cause:** Stale verification script.
- **Target Files:**
  - `scripts/verify_docs_consistency.py`
  - `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `RESEARCH_STATUS.md`, `REPRODUCIBILITY.md`
- **Implementation Method:**
  - Expand `scripts/verify_docs_consistency.py` with 9 rigorous checks: version consistency, dataset consistency, release commit consistency, installer version consistency, model registry consistency, benchmark claim consistency, capability claim consistency, path portability, provenance consistency.
- **Tests Required:** `python scripts/verify_docs_consistency.py`
- **Evidence Required:** `diagnostics/v10_documentation_report.json`
- **Completion Condition:** 100% consistency across all documentation and source files.

### TASK-V10-GATE-12 (P0: Release Blocker) — Definitive V10 Release Gate Matrix & Acceptance Runner
- **Problem:** Fragmented acceptance tests across versions.
- **Root Cause:** Evolution from V5 to V9 without a unified master V10 gate runner.
- **Target Files:**
  - `verification/acceptance_schema.json`
  - `scripts/run_v10_acceptance_matrix.py` (New)
  - `diagnostics/v10_acceptance_matrix.json` (New)
- **Implementation Method:**
  - Define all 56 required V10 gates in machine-readable format.
  - Fail closed: any ERROR or unverified claim becomes FAIL.
  - Allow skips only for genuinely missing environment hardware/models with clear justification.
- **Tests Required:** Execution of all 56 minimum gates.
- **Evidence Required:** `diagnostics/v10_acceptance_matrix.json`, `diagnostics/v10_release_certificate.json`.
- **Completion Condition:** Zero FAILs; all allowed skips justified; clean evidence bundle generated.

---

## 3. Minimum V10 Acceptance Matrix (56 Gates)

| # | Gate Name | Required Category | Target | Allowed Skip Condition |
|---|---|---|:---:|---|
| 1 | `repository_integrity` | System | PASS | None |
| 2 | `source_tree_integrity` | System | PASS | None |
| 3 | `dependency_lock_consistency` | Packaging | PASS | None |
| 4 | `dataset_provenance_integrity` | Science | PASS | None |
| 5 | `dataset_identity_consistency` | Science | PASS | None |
| 6 | `graph_mode_separation` | Science | PASS | None |
| 7 | `biological_vs_synthetic_separation` | Science | PASS | None |
| 8 | `lif_correctness` | SNN | PASS | None |
| 9 | `refractory_correctness` | SNN | PASS | None |
| 10 | `reset_correctness` | SNN | PASS | None |
| 11 | `cpu_gpu_numeric_parity` | Compute | PASS | If no Vulkan GPU present |
| 12 | `persistent_gpu_resource_lifecycle` | Compute | PASS | If no Vulkan GPU present |
| 13 | `deterministic_experiment_replay` | Science | PASS | None |
| 14 | `snapshot_restore_replay` | Science | PASS | None |
| 15 | `plasticity_causal_effect` | SNN | PASS | None |
| 16 | `living_brain_identity` | SNN | PASS | None |
| 17 | `structural_growth_integrity` | ALife | PASS | None |
| 18 | `genome_provenance` | ALife | PASS | None |
| 19 | `heredity_separation` | ALife | PASS | None |
| 20 | `autonomy_goal_generation` | ALife | PASS | None |
| 21 | `grounded_language` | Cognition | PASS | None |
| 22 | `social_state` | ALife | PASS | None |
| 23 | `llm_control_safety` | Cognition | PASS | None |
| 24 | `research_memory_chain` | Memory | PASS | None |
| 25 | `deep_time_exact_mode` | Simulation | PASS | None |
| 26 | `deep_time_approximation_honesty` | Simulation | PASS | None |
| 27 | `world_generation_determinism` | World | PASS | None |
| 28 | `chunk_streaming` | World | PASS | None |
| 29 | `chunk_persistence` | World | PASS | None |
| 30 | `world_restart_replay` | World | PASS | None |
| 31 | `asset_geometry_validation` | Assets | PASS | None |
| 32 | `asset_physics_consistency` | Assets | PASS | None |
| 33 | `asset_portability` | Assets | PASS | None |
| 34 | `embodied_closed_loop_control` | Embodiment | PASS | None |
| 35 | `memory_write` | Memory | PASS | None |
| 36 | `memory_reload` | Memory | PASS | None |
| 37 | `memory_retrieval` | Memory | PASS | None |
| 38 | `world_action_from_memory` | Embodiment | PASS | None |
| 39 | `planner_source_honesty` | Cognition | PASS | None |
| 40 | `model_registry_integrity` | Models | PASS | None |
| 41 | `model_inference_honesty` | Models | PASS | None |
| 42 | `ui_live_backend_state` | UI | PASS | None |
| 43 | `ui_no_fake_telemetry` | UI | PASS | None |
| 44 | `api_contracts` | API | PASS | None |
| 45 | `backup_create` | Backup | PASS | None |
| 46 | `backup_verify` | Backup | PASS | None |
| 47 | `backup_restore` | Backup | PASS | None |
| 48 | `backup_hash_integrity` | Backup | PASS | None |
| 49 | `watchdog_behavior` | Reliability | PASS | None |
| 50 | `installer_fresh_install` | Packaging | PASS | None |
| 51 | `installer_repair` | Packaging | PASS | None |
| 52 | `portable_artifact_self_test` | Packaging | PASS | If no portable package built |
| 53 | `docs_consistency` | Quality | PASS | None |
| 54 | `license_consistency` | Quality | PASS | None |
| 55 | `reproducibility_manifest` | Quality | PASS | None |
| 56 | `v10_version_metadata` | Quality | PASS | None |

---

## 4. Execution Sequence & Safeguards

1. **Safety First:** Create recovery git branch `release/v10-prep`.
2. **Phase 1 Plan Files:** Complete `V10_MASTER_PLAN.md` and `verification/v10_master_plan.json`.
3. **Phased Iteration:**
   - Execute Tasks 1–12 in strict priority order.
   - Run tests after every component edit.
   - Self-heal any regressions immediately without mocking or test weakening.
4. **Clean Room Verification:** Build isolated test in fresh directory (`dist/test_clean_room`).
5. **Release Gating:** Zero failures on all 56 gates; build release artifacts; commit; tag `v10.0.0`; push GitHub; sync HF Space; verify remote live health.
