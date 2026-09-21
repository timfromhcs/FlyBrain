# FlyBrain Reproducibility Guide

## 1. Deterministic Execution Policy
FlyBrain guarantees reproducible execution through:
1. **Explicit PRNG Seeds**: All simulations, mutations, and replays record their seed (default: `seed = 42`).
2. **Biological Source Integrity**: The biological dataset is pinned to SHA-256 hash `d20bb1b48b99cfbe1a0efd0614f85108ce8a30644e5917fa97bc8a873138b309` (`malecns/data-raw/2023-27-2 soma_sides.csv`).
3. **Deterministic Connectome Extraction**: KD-tree spatial queries and CSR matrix layout are deterministic given the seed and neuron count.
4. **Vulkan Kernel Consistency**: Shaders use standard IEEE-754 float32 arithmetic matching CPU reference by trajectory parity (max abs diff < 1e-4, exact spike trains) — explicitly NOT bit-exact.
5. **Snapshot Hash Verification**: All snapshots record and verify SHA-256 checksums before and after restoration.
6. **Separated identity layers**: source / neuron-topology / synapse-topology /
   parameters / dynamic-state / plasticity / structural / genome / organism /
   population / experiment / research (`src/provenance/identity_layers.py`) —
   never collapsed into one ambiguous hash.
7. **Versioned checkpoints**: `population_checkpoint_v1` envelopes carry
   state/research/combined hashes, the RNG bundle, graph + experiment
   identity, and full event history (`src/population/checkpoint.py`).

## 2. Pinned Environment Hashes & Versions
- Windows 11 Build: 10.0.26200
- Python: 3.12.13 (x86_64)
- Vulkan SDK: 1.4.357.0
- GPU Driver: 32.0.21043.12001
- Model: `Qwen3-4B-Q4_K_M.gguf` (SHA-256 recorded in `diagnostics/dependency_manifest.json`)
- VAE: `AutoencoderTiny` (safetensors format)
- Hugging Face Space (CPU-only): pinned in `huggingface/requirements-hf.txt`

## 3. Replay Test Command
```powershell
.venv\Scripts\python.exe scripts/test_state_persistence.py
```
Outputs `diagnostics/brain_snapshot_manifest.json` showing identical bitwise continuity.

## 4. Local Models (V6)
Binaries live under `models/<task>/` (git-ignored, ~15 GB total) and are
acquired once via `scripts/acquire_models.py <task>` (refuses with
`FLYBRAIN_OFFLINE=1`). `src/models/registry.py` pins id/revision/license/
SHA-256/size; `diagnostics/v6/model_registry.json` is the release record.
Runtime loads local-only (`HF_HUB_OFFLINE=1` enforced when offline).

## 5. Offline Operation (V6)
`FLYBRAIN_OFFLINE=1`: no downloads, no cloud inference, no CDN (vendored
frontend), `connect()`-blockade test in `tests/test_v6_offline.py` proves
world boot/simulate/backup/restore/model-load without network.

## 4. Clean-Room Reproduction (v4.1)
From a clean checkout, with no caches or environment leaks:
```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
glslc shaders/brain_step.comp -o shaders/brain_step.spv
glslc shaders/plasticity.comp -o shaders/plasticity.spv
.venv\Scripts\python.exe scripts/validate_malecns_data.py
.venv\Scripts\python.exe scripts/verify_docs_consistency.py
.venv\Scripts\python.exe -m unittest tests.test_checkpoint_v41 tests.test_parity tests.test_control_plane
.venv\Scripts\python.exe scripts/run_acceptance_matrix.py
.venv\Scripts\python.exe scripts/render_status_tables.py --check
.venv\Scripts\python.exe scripts/smoke_hf_space.py
```
Shader binaries must match a rebuild from source (`glslc`); generated SPIR-V
hashes are recorded in `diagnostics/benchmark_report.json` and experiment
manifests.

## 5. Artifact Policy: Canonical Evidence vs Generated Data
Tracked (canonical, small, release-relevant):
- `diagnostics/acceptance_matrix.json`, `final_verification.json`,
  `reproducibility_manifest.json`, `cpu_gpu_validation_report.json`,
  `connectome_validation_report.json`, `benchmark_report.json`,
  `huggingface_verification.json`, `pre_release_audit.json`
- `manifests/malecns_provenance.json`, `diagnostics/verification_contract.json`
- `visual_evidence/screens/*.png` (current-UI screenshots referenced by README)
- `verification/acceptance_schema.json`

Local-only (git-ignored, reproducible at will):
- `diagnostics/connectome_cache/`, `diagnostics/snapshots/`, `*.db`,
  `*.npz`, `*.log`, `diagnostics/test_*`, runtime checkpoints, WAV dumps
  (`llm/gguf/*.gguf` weights are provided separately and never committed).

Every committed report identifies commit, version, timestamp, seed,
environment, and source/data identity.
