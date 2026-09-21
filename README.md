# FlyBrain: Vulkan-First Biological Connectome Research Framework
### Grounded in the Authentic Janelia *Drosophila* Male Central Nervous System (`male-cns:v1.0`)

<p align="center">
  <img src="visual_evidence/screens/screen_01_main_application_live_brain.png" alt="FlyBrain Lab Scientific Workstation" width="900">
</p>

<p align="center">
  <a href="https://github.com/timfromhcs/FlyBrain/actions"><img src="https://github.com/timfromhcs/FlyBrain/actions/workflows/ci.yml/badge.svg" alt="CI/CD Status"></a>
  <a href="https://github.com/timfromhcs/FlyBrain/releases"><img src="https://img.shields.io/github/v/release/timfromhcs/FlyBrain" alt="Latest Release"></a>
  <a href="https://huggingface.co/spaces/timfromhcs/FlyBrain-Lab"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow" alt="Hugging Face Space"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Vulkan-1.2%2B%20Compute-red?logo=vulkan&logoColor=white" alt="Vulkan Compute">
  <img src="https://img.shields.io/badge/Biological%20Data-Janelia%20MaleCNS-059669" alt="Janelia MaleCNS">
  <img src="https://img.shields.io/badge/Acceptance%20Matrix-canonical%20schema-brightgreen" alt="Acceptance Matrix">
  <img src="https://img.shields.io/badge/Platform-Windows%2011%20%7C%20Linux-0284c7" alt="Platform">
</p>

---

## Table of Contents
1. [Overview](#overview)
2. [Biological Grounding & Provenance Contract](#biological-grounding--provenance-contract)
3. [Classical Leaky Integrate-and-Fire (LIF) Dynamics](#classical-leaky-integrate-and-fire-lif-dynamics)
4. [Persistent Vulkan Compute Engine](#persistent-vulkan-compute-engine)
5. [FlyBrain Lab: Scientific Workstation UI](#flybrain-lab-scientific-workstation-ui)
6. [Deterministic Experimentation CLI](#deterministic-experimentation-cli)
7. [Automated Release Acceptance Matrix (canonical, generated)](#automated-release-acceptance-matrix-canonical-generated)
8. [Multi-Store Persistent Memory](#multi-store-persistent-memory)
9. [Installation & Quick Start](#installation--quick-start)
10. [Hardware Benchmark Results](#hardware-benchmark-results)
11. [What's New in v6.0](#whats-new-in-v60)
12. [What's New in v5.0](#whats-new-in-v50)
13. [What's New in v4.1](#whats-new-in-v41)
14. [Hugging Face Space](#hugging-face-space)
15. [Artificial-Life Layer — Honest Status](#artificial-life-layer--honest-status)
14. [Citation & Third-Party Notices](#citation--third-party-notices)

---

## Overview

**FlyBrain** is an autonomous artificial-organism research framework that translates empirical connectomics into a real, local, Windows 11 compatible, Vulkan-accelerated computational organism.

Unlike prompt-based agent wrappers, toy gridworlds, or ungrounded simulations:
- Neural circuitry is grounded in **125,506 biological neurons** and **99,301 authentic synaptic connections** from the Janelia FlyEM Male Central Nervous System connectome (`male-cns:v1.0`).
- Simulation executes with **genuine classical Leaky Integrate-and-Fire (LIF)** dynamics featuring membrane decay, action potential thresholding, hard reset clamping, and absolute refractory periods.
- High-throughput neural integration is computed via a **persistent Vulkan 1.2+ compute backend** with zero per-step GPU reallocations, running on physical discrete or integrated GPUs (AMD Radeon 680M verified).
- Subsystems are bound to a strict **Non-Hallucination Contract**, strictly distinguishing empirical biological connections (`GraphMode.REAL`), spatial surrogates (`GraphMode.SPATIAL_SURROGATE`), and synthetic regression networks (`GraphMode.SYNTHETIC_TEST`).
- Every experiment is cryptographically tracked with 256-bit SHA-256 state hashes, ensuring bit-exact deterministic reproduction.

---

## Biological Grounding & Provenance Contract

Every connectome circuit in FlyBrain is explicitly categorized by its provenance mode in `src/connectome/types.py`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Janelia MaleCNS Biological Source                     │
│  - 125,506 neuron somas, coordinates, T-bars: 2023-27-2 soma_sides.csv      │
│  - 99,301 verified biological synaptic pairs: malecns_v1_0_connections.csv  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│  GraphMode.REAL  │          │ SPATIAL_SURROGATE│          │  SYNTHETIC_TEST  │
│ [VERIFIED STATUS]│          │[SURROGATE STATUS]│          │[EXPERIMENTAL ST.]│
│ Authentic EM     │          │ Morphological    │          │ Deterministic CI │
│ synaptic tables  │          │ k-d tree graph   │          │ regression graph │
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

1. **`GraphMode.REAL` (`VERIFIED`, canonical identity `REAL_SUBGRAPH`)**: A bounded
   sampled subgraph of the Janelia MaleCNS v1.0 synaptic table (hub-biased
   `REAL_HUB_SUBGRAPH` strategy; sampled N of 125,506 somas, sampled M of 99,301
   pairs — exact counts in `provenance_metadata` and `/api/provenance`). Every
   included edge corresponds to an empirically reconstructed biological synapse;
   `REAL_FULL` (all neurons in one live circuit) is explicitly unavailable and
   raises instead of silently substituting.
2. **`GraphMode.SPATIAL_SURROGATE` (`SURROGATE`)**: Connects empirical somas via 3D Euclidean k-d tree proximity weighted by presynaptic T-bar capacities. Honestly marked as a surrogate in all telemetry and manifests.
3. **`GraphMode.SYNTHETIC_TEST` (`EXPERIMENTAL`)**: Deterministic synthetic circuit for fast regression testing and CI verification.

Provenance manifest hashes are cryptographically verified in `manifests/malecns_provenance.json`:
- `soma_sides.csv` SHA-256: `6c1c415ab748ab1cc65c9a274885cfadc880a88eabb17915afe643c105bac84d`
- `connections.csv` SHA-256: `039e929b4ccc776f13a8d17eb9833d85de9454b8f43a0a0de34fdabb7ca2a668`

---

## Classical Leaky Integrate-and-Fire (LIF) Dynamics

FlyBrain simulates genuine biophysical LIF dynamics rather than continuous sigmoids.

### Mathematical Formulation
For each neuron $i \in \{0, \dots, N-1\}$:

1. **Synaptic Current Summation:**
   $$I_{\text{syn}, i}(t) = \sum_{j \in \text{Pre}(i)} W_{ij} \cdot S_j(t-1)$$

2. **Absolute Refractory Period Check:**
   If $R_i(t-1) > 0$:
   $$R_i(t) = R_i(t-1) - 1, \quad V_i(t) = V_{\text{reset}}, \quad S_i(t) = 0.0$$

3. **Subthreshold Leaky Integration:**
   If $R_i(t-1) == 0$:
   $$V_{\text{cand}, i} = V_{\text{rest}} + \left(V_i(t-1) - V_{\text{rest}}\right) \cdot \lambda + I_{\text{syn}, i}(t) + I_{\text{ext}, i}(t)$$

4. **Action Potential Threshold & Reset:**
   If $V_{\text{cand}, i} \ge V_{\text{thresh}}$:
   $$S_i(t) = 1.0, \quad V_i(t) = V_{\text{reset}}, \quad R_i(t) = t_{\text{ref}}$$
   Otherwise:
   $$S_i(t) = 0.0, \quad V_i(t) = \max\left(V_{\text{cand}, i}, V_{\text{reset}} - 1.0\right), \quad R_i(t) = 0$$

Both the compiled GLSL compute shader (`shaders/brain_step.comp`) and CPU reference engine (`src/compute/cpu_reference.py`) implement this exact formulation with trajectory parity (max abs diff < 1e-4; spike outputs identical), explicitly not bit-exact.

---

## Persistent Vulkan Compute Engine

The `VulkanComputeEngine` (`src/compute/vulkan_backend.py`) is engineered for persistent, low-overhead GPU execution:

- **Capability-Based Selection:** Automatically scores and binds the most capable compute queue (Discrete GPU > Integrated GPU > CPU).
- **Persistent GPU Buffers:** All CSR arrays, membrane potentials, spikes, and refractory counters stay resident in GPU VRAM across steps. Zero per-step memory allocations.
- **11 Descriptor Bindings:** Binds row offsets, column indices, weights, previous spikes, external currents, input/output potentials, input/output spikes, input/output refractory counters, and simulation parameters.
- **Plasticity Compute Pipeline:** Dedicated compute shader (`shaders/plasticity.comp`) executes three-factor reward-modulated Hebbian learning directly on GPU weights.
- **Lazy weight sync (v4.1, measured):** GPU weights are authoritative; the CPU
  mirror refreshes only at explicit sync points (snapshots, experiment
  manifests, validation). Measured weight-readback cost removed from rewarded
  steps (~0.85 ms saved per rewarded step at N=1024 on AMD Radeon 680M; see
  `diagnostics/benchmark_report.json`). Telemetry exposes `weights_synced` so
  staleness is never silent.
- **Retained host-visible/coherent memory (measured):** on the unified-memory
  AMD iGPU, device-local+staging would add copies without benefit (measured
  upload bandwidth up to ~4.2 GB/s host-visible); the design is kept for
  correctness and AMD compatibility, with the measurement in
  `diagnostics/benchmark_report.json`.

---

## FlyBrain Lab: Scientific Workstation UI

The user interface has been completely transformed into **FlyBrain Lab**, a dark scientific research workstation:

- **Interactive 3D Connectome Viewer:** Pinned vendored Three.js r128 (zero CDN
  runtime dependencies), featuring 3D orbit controls, anatomical axes, and
  raycasting neuron inspection with live membrane/spike state.
- **Biophysical Telemetry:** Real-time sparklines for spike rates, mean membrane potential, prediction error, and homeostatic drives (*energy*, *curiosity*, *social*, *integrity*).
- **Experiment Control Hub:** Run deterministic experiments directly from the dashboard and inspect reproduction hashes.
- **Evolutionary Lineage Tree:** Visualize generational mutations, benchmark scores, and candidate rollbacks.
- **Zero Blocking Alerts:** Built strictly with asynchronous toast notifications; never blocks the event loop with `alert()` or `prompt()`.

Access FlyBrain Lab by running:
```bash
flybrain lab
```
and navigating to `http://localhost:8080`.

---

## Deterministic Experimentation CLI

FlyBrain provides a dedicated CLI for provenance-tracked, deterministic research:

```bash
# Launch interactive scientific workstation
flybrain lab

# Run a deterministic experiment with real connectome
flybrain experiment run --mode REAL --scale 256 --steps 100 --seed 42

# Verify bit-exact cryptographic replication of an experiment
flybrain experiment verify --result <experiment_id>

# Compare two experiment runs across metrics and state hashes
flybrain experiment compare --a <exp_id_1> --b <exp_id_2>

# Run full 25-category release acceptance matrix
flybrain acceptance-matrix

# Verify consistency between code, shaders, and documentation
flybrain docs-verify
```

---

## Automated Release Acceptance Matrix (canonical, generated)

Release readiness is verified by `scripts/run_acceptance_matrix.py` against the
single canonical schema `verification/acceptance_schema.json`, producing
`diagnostics/acceptance_matrix.json`. Every PASS corresponds to an executable
behavioral assertion (no source-text-only checks). The table below is
**generated** by `scripts/render_status_tables.py` — never hand-edited:

<!-- ACCEPTANCE-TABLE-START -->
> Canonical source: `verification/acceptance_schema.json` → `diagnostics/acceptance_matrix.json` (overall **PASSED**: **56/57 PASS**, 0 FAIL, 1 SKIP). Do not copy totals elsewhere — regenerate with `scripts/render_status_tables.py`.

| Index | Category | Status | Details |
| :---: | :--- | :---: | :--- |
| 1 | `repository_cleanliness` | **PASS** | All core repository directories intact and organized. |
| 2 | `provenance_manifest_integrity` | **PASS** | Janelia MaleCNS soma and connection SHA-256 hashes match provenance manifest exactly. |
| 3 | `connectome_contract_separation` | **PASS** | Explicit separation of GraphMode contracts: ['REAL', 'SPATIAL_SURROGATE', 'SYNTHETIC_TEST']; REAL canonicalizes to REAL_SUBGRAPH; REAL_FULL honestly raises. |
| 4 | `biological_vs_synthetic_separation` | **PASS** | REAL graph verified from MaleCNS; SYNTHETIC_TEST marked EXPERIMENTAL with distinct topology. |
| 5 | `spatial_surrogate_behavior` | **PASS** | Spatial surrogate generated 546 synapses via 3D k-d tree proximity. |
| 6 | `lif_dynamics_correctness` | **PASS** | LIF integration correctly decays membrane potential, fires spike, and clamps to reset. |
| 7 | `refractory_period_invariance` | **PASS** | Refractory period strictly prevents firing and decrements counter during active refraction. |
| 8 | `reset_potential_invariance` | **PASS** | Membrane potential instantly clamped to V_reset (-70.0 mV) upon spike generation. |
| 9 | `vulkan_discovery_and_selection` | **PASS** | Vulkan 1.3 physical device discovered and selected: 'AMD Radeon(TM) Graphics' |
| 10 | `persistent_resource_lifecycle` | **PASS** | Vulkan buffers, descriptor sets, and command buffers remain resident across simulation steps. |
| 11 | `cpu_vulkan_numerical_parity` | **PASS** | Tolerance-based parity verified (9/9 cases): max abs diff < 1e-4 with exact spike trains (NOT bit-exact). |
| 12 | `single_loop_telemetry_isolation` | **PASS** | SimulationEngine executed 3 steps in background thread without blocking telemetry. |
| 13 | `thread_lock_concurrency` | **PASS** | Thread-safe RLock prevented data races during concurrent telemetry and state reads. |
| 14 | `deterministic_experiment_replication` | **PASS** | Exact 256-bit SHA-256 match (9ff3e4faca75180f...) across independent runs. |
| 15 | `local_model_degradation_honesty` | **PASS** | System honestly declared cognitive status: OPERATIONAL |
| 16 | `continuous_learning_weight_change` | **PASS** | Synaptic plasticity modified weights. Max delta_w: 1.490564e-01. |
| 17 | `ui_no_blocking_alerts` | **PASS** | FlyBrain Lab workstation contains zero blocking alert()/prompt() calls (non-intrusive toasts) and zero CDN runtime dependencies (vendored three.js). |
| 18 | `full_pipeline_e2e_runnable` | **PASS** | End-to-end pipeline (real connectome load -> runtime step -> snapshot save) executed flawlessly. |
| 19 | `documentation_claim_consistency` | **PASS** | All documentation claims match datasets, shader descriptors, and API routes exactly. |
| 20 | `alife_branch_replay_determinism` | **PASS** | Population snapshot branch replay produced identical hash (52f731d079a83912). |
| 21 | `developmental_structural_integrity` | **PASS** | Full developmental pipeline: 1 born, 6 synapses grown, 4 died; invariants hold, dead neurons edgeless. |
| 22 | `genome_mutation_crossover_provenance` | **PASS** | Mutation/crossover deterministic with complete parent provenance. |
| 23 | `overlapping_reproduction` | **PASS** | 4 reproductions across generations [0, 1]; parents alive at every birth: True. |
| 24 | `cultural_transmission_gain` | **PASS** | Measured learning gain 0.3766 with teacher chain ['mx-teacher']. |
| 25 | `real_mode_zero_surrogate_edges` | **PASS** | REAL-64 graph: 100 edges, 0 non-empirical; metadata confirms zero surrogate. |
| 26 | `csr_directionality` | **PASS** | A->B drives B, never A; reverse direction has no effect. |
| 27 | `biological_edge_semantics` | **PASS** | REAL weight transform declared: w = min(0.8, 0.05 + 0.02 * synapse_count) [simulation transform, NOT a measured conductance] |
| 28 | `real_annotation_integrity` | **PASS** | Unavailable annotations flagged UNKNOWN; available ones EMPIRICAL/DERIVED. |
| 29 | `plasticity_causal_effect` | **PASS** | reward=0 no change; reward>0 potentiation; reward<0 depression. |
| 30 | `checkpoint_continuation` | **PASS** | Checkpoint/resume final population hash equals uninterrupted run. |
| 31 | `llm_model_discovery_and_inference` | **PASS** | Discovered 2 GGUF; inference=SUCCESS; model=MiniCPM5-2B-Q8_0.gguf. |
| 32 | `llm_failure_mode_and_tool_safety` | **PASS** | Unavailable model returns structured error (no fake text); shell/unknown tools rejected. |
| 33 | `living_brain_identity` | **PASS** | Persistent identities + EMERGENT provenance; growth 32->33. |
| 34 | `structural_growth_resource_constrained` | **PASS** | Zero growth budget blocks neurogenesis (energy is the constraint). |
| 35 | `eligibility_neuromodulation` | **PASS** | v2 traces + novelty-driven neuromodulation change weights (reward=0). |
| 36 | `autonomy_self_generated_goals` | **PASS** | All living organisms self-generate goals; no human task commands. |
| 37 | `grounded_language_and_social` | **PASS** | Symbols bind to grounded concepts; trust emerges from outcomes. |
| 38 | `genome_v2_architecture_genes` | **PASS** | v2 encodes learning architecture; v1 legacy remains valid. |
| 39 | `speciation_evidence` | **PASS** | Divergence recorded only with measured genome distance. |
| 40 | `llm_control_plane_safety` | **PASS** | Invalid/shell commands rejected; valid typed commands execute. |
| 41 | `research_memory_chain` | **PASS** | Hash-chained append-only research memory; tamper-evident. |
| 42 | `deeptime_escalation_replay` | **PASS** | Coarse deep-time escalates to full-res checkpoint; replay hash-verified. |
| 43 | `milestone_evidence` | **PASS** | Detected ['STRUCTURAL_EXPANSION', 'OVERLAPPING_GENERATIONS'] with evidence + certificate. |
| 44 | `benchmark_fairness` | **PASS** | Per-arm budgets documented; LLM arms SKIP with reason when unmet. |
| 45 | `version_metadata` | **PASS** | FlyBrain version metadata = 8.0.0 (v8.0.0). |
| 46 | `immutable_bio_baseline` | **PASS** | Biological baseline fingerprint unchanged after lifetime development. |
| 47 | `synapse_identity_provenance` | **PASS** | Stable synapse IDs; seed records cite source dataset; new synapses are EMERGENT, never BIOLOGICAL. |
| 48 | `brain_identity_layers` | **PASS** | 12-layer identity: same state -> same identity; weight change -> different identity. |
| 49 | `heredity_separation` | **PASS** | Parent learned; child starts from deterministic development, not inherited learned weights. |
| 50 | `evolution_parent_identity` | **PASS** | Accepted child never becomes its own parent in history. |
| 51 | `ablation_enforcement` | **PASS** | no_growth grew=False; no_plasticity weights unchanged (enforced, verified by measurement). |
| 52 | `deterministic_research_ids` | **PASS** | Research IDs derive from content+sequence; no wall-clock identity. |
| 53 | `research_ledger_chain` | **PASS** | Hash-chained ledger with code/dataset/shader provenance. |
| 54 | `deeptime_exact_mode` | **PASS** | EXACT mode is tick-by-tick with no approximation claim; ACCELERATED labels its approximation model. |
| 55 | `doctor_functional` | **PASS** | Doctor reports READY with 12 OK checks (real values). |
| 56 | `ui_v4_endpoints` | **PASS** | /api/version + /api/doctor serve real system state. |
| 57 | `portable_package` | *SKIP_ENVIRONMENT* | No built portable bundle on this host; set FLYBRAIN_PORTABLE_DIR to a built dist for artifact self-test. |
<!-- ACCEPTANCE-TABLE-END -->

> **Environment-dependent gates are `SKIP`, never fake-PASS:** on GPU-less
> runners the Vulkan gates skip; on hosts without local GGUF weights the LLM
> inference gate skips; without a built portable bundle `portable_package`
> skips. Profiles are defined in `verification/acceptance_schema.json`
> (`workstation`, `cpu-only`, `vulkan`, `ci`, `huggingface`, `release`).
> The release gate requires **0 FAIL**.

---

## Multi-Store Persistent Memory

FlyBrain maintains a multi-tiered SQLite memory architecture in `src/memory/persistence.py`:
- **Working Memory:** 7-slot recency-bounded active working buffer.
- **Episodic Store:** Structured sensory observations, motor actions, rewards, and prediction errors.
- **Semantic Associative Memory:** Concept vectors queryable via cosine similarity.
- **Dream Studio Store:** Counterfactual replay traces exploring hypothetical policies during offline sleep cycles.

---

## Installation & Quick Start

### Prerequisites
- Windows 11 64-bit (or Linux x86_64)
- Python 3.11+ (Python 3.12 recommended)
- Vulkan SDK 1.3+ (installed with `glslc` on system `PATH`)

### Setup
```bash
# Clone repository
git clone https://github.com/timfromhcs/FlyBrain.git
cd FlyBrain

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run documentation and code consistency checker
python src/main.py docs-verify

# Run full release acceptance matrix
python src/main.py acceptance-matrix

# Launch FlyBrain Lab Workstation
python src/main.py lab
```

---

## Hardware Benchmark Results

Re-measured 2026-09-14 on physical hardware (AMD Ryzen 7 7735HS, AMD Radeon 680M GPU,
Windows 11) against the current empirical REAL topology (20 persistent steps each):

| Circuit Size (Neurons) | Synapses | Vulkan Latency (ms) | Throughput (Steps/Sec) | Throughput (Synapses/Sec) |
| :---: | :---: | :---: | :---: | :---: |
| 256 | 1,449 | 0.374 ms | 2,676.3 | 3.88 M/s |
| 512 | 6,557 | 0.237 ms | 4,221.5 | 27.68 M/s |
| 1,024 | 25,749 | 0.260 ms | 3,839.2 | 98.86 M/s |

v4.1 re-measurement (`diagnostics/benchmark_report.json`, 2026-09-20, same
hardware, 20 measured steps after 5 warmup, seed 42) adds readback/plasticity
splits and percentile statistics: per-step host-readback overhead 0.04–0.21 ms
(state readback retained — CPU owns telemetry/motor state); plasticity weight
readback dominates rewarded steps at scale (N=1024: 1.03 ms with readback vs
0.17 ms GPU-resident — hence lazy weight sync); host-visible upload bandwidth
up to ~4.2 GB/s (device-local/staging retained-out by measurement on the
unified-memory iGPU). Methodology and design conclusions live in the report;
no optimization is kept without a measurement behind it.

---

## What's New in v6.0

Embodied artificial life (local-first, offline-capable):

- **Persistent 3D world** (MuJoCo rigid-body physics, server-authoritative):
  terrain, home with pushable door, furniture, trees, rocks, food, lake zone.
- **Closed loop**: WORLD → BODY → SENSORS → PERCEPTION → BRAIN → GOALS →
  ACTION → PHYSICS → WORLD → REWARD → MEMORY (per-tick causal records).
- **First-person geometric perception** (raycast fan, occlusion-aware),
  honest `GEOMETRIC` label; VLM slot filled by local SmolVLM when present.
- **Local models, all verified**: Qwen3-0.6B-GGUF (structured goals),
  SmolVLM-256M (grounded captions), Whisper-small (file STT),
  Kokoro-82M (TTS), DreamShaper-8-LCM + ControlNet-canny (IoU 0.63
  continuity), MiniLM (semantic RAG). Model manager with pressure shedding;
  `models/` binaries git-ignored, registry pins SHA/license.
- **Dreams + imagination** with `DREAM/COUNTERFACTUAL/GENERATED_*`
  provenance; Dream Studio gallery; narrative-only path when no GPU model.
- **Speech loop**: file/mic → STT → memory → LLM → TTS (all local).
- **Friends**: full agents with genomes, needs, trust emerging from real
  teaching; `/api/v1/world/friends`.
- **World UI tab**: server-rendered 3D, 4 cameras, minimap, dream/speak/
  imagine controls, zero console errors.
- **Offline hard gate**: `FLYBRAIN_OFFLINE=1` + connect() blockade tests
  green; acquisition refuses; vendored frontend.
- **V6 matrix 42/42** (`scripts/run_v6_matrix.py`), evidence in
  `diagnostics/v6/`, benchmarks in `performance_report.json`.

## What's New in v5.0

- **Versioned API:** `/api/v1/*` contracts (readiness, runtime, metrics,
  events, `neuron/{body_id}`, stream, backup, watchdog) with legacy compat.
- **Backups:** tamper-evident create/verify/restore/download, retention,
  proven round trip locally and on the live Space.
- **24/7 stream + watchdog:** bounded supervision and recovery, presets
  (`flybrain lab --preset 24_7_STREAM`), Live/Colony/Backups UI tabs.
- **Null topology controls** for honest treatment-vs-topology experiments.
- **Release machinery:** stale-doc detector, integrity gate, release
  certificate, 11 real UI screenshots.

## What's New in v4.1

- **Explicit graph identities:** `REAL_SUBGRAPH` (canonical sampled subgraph,
  legacy name `REAL`), `REAL_FULL` (honestly unavailable — raises),
  `SPATIAL_SURROGATE`, `SYNTHETIC_TEST` (`src/connectome/types.py`,
  `/api/provenance`).
- **Twelve separated identity layers** (`src/provenance/identity_layers.py`):
  source, neuron/synapse topology, parameters, dynamic state, plasticity,
  structural, genome, organism, population, experiment, research — never one
  ambiguous hash.
- **Versioned checkpoint envelope** (`src/population/checkpoint.py`,
  `population_checkpoint_v1`): state/research/combined hashes, RNG bundle,
  graph + experiment identity, preserved event history; save → restore →
  continue provably equals uninterrupted execution (`tests/test_checkpoint_v41.py`).
- **Measured GPU optimization:** lazy weight sync + retained host-visible
  memory, both benchmark-backed (`scripts/benchmark_v41.py`).
- **Capability-based LLM control:** typed allowlists + role capabilities +
  strict identifier grammar; fragile whole-blob substring blacklists removed
  (`src/llm/control.py`).
- **Embodied causal telemetry:** every organism step returns its
  sensory → neural → motor → body → world → reward causal path with an
  explicit `policy_source` label.
- **Canonical acceptance schema** (`verification/acceptance_schema.json`) with
  workstation/cpu-only/vulkan/ci/huggingface/release profiles; docs render
  totals from the machine-readable report.
- **FlyBrain Lab frontend:** split `index.html` + `css/lab.css` + `js/lab.js`,
  vendored pinned three.js r128 (zero CDN runtime dependencies), live
  Provenance tab backed by `/api/provenance`, reconnect-resilient telemetry.
- **Real Hugging Face Space** (`huggingface/`, Docker SDK, CPU-only, honest
  backend reporting) with boot-verified smoke test
  (`diagnostics/huggingface_verification.json`). Deploy: `deployment/HUGGINGFACE.md`.

---

## Hugging Face Space

A reproducible CPU-only Space serves the full FlyBrain Lab UI with live
backend state: see `deployment/HUGGINGFACE.md` and `huggingface/README.md`.
The Space reports `backend: cpu_reference` honestly (no fake Vulkan), labels
the sampled `REAL_SUBGRAPH`, heuristic populations, and derived weights in
the Provenance tab.

---

## Artificial-Life Layer — Honest Status

FlyBrain contains a real, small-scale artificial-life layer on top of the connectome core, now extended with local LLM-driven scientific tooling and reproducible campaign infrastructure.
Nothing below is mocked: every claimed behavior is implemented, tested in
`tests/test_alife.py` (12/12 pass) and the broader suite (111 tests passing), and reproducible via `scripts/run_alife_experiment.py`.
Details: `docs/alife_architecture.md`.

```bash
# Canonical overlapping-generation experiment (CPU, deterministic)
.venv\Scripts\python.exe scripts/run_alife_experiment.py --population 6 --ticks 60 --seed 7

# Bit-exact replay verification (must print match=True)
.venv\Scripts\python.exe scripts/run_alife_experiment.py --verify diagnostics/alife_experiments/alife_p6_t60_s7.json

# Multi-generation campaign with checkpoint/resume
.venv\Scripts\python.exe scripts/run_long_campaign.py --generations 4 --population 6 --seed 11
.venv\Scripts\python.exe scripts/run_long_campaign.py --resume diagnostics/campaigns/camp_11_6/checkpoint.json --generations 4

# Honest performance benchmarks (never overwrites; timestamped provenance report)
.venv\Scripts\python.exe scripts/run_benchmarks.py

# Adversarial boundary and failure-mode tests
.venv\Scripts\python.exe -m unittest tests.test_adversarial

# Live colony API (after `flybrain lab`)
# GET /api/colony, /api/colony/organism/{id}, /api/colony/lineage
```

Verified canonical result (`seed 7, pop 6, 60 ticks`): 8 living, 2 births, 0 deaths,
generations `[0, 1]` coexisting, 0 teaching sessions in this short canonical run
(teaching gain is separately verified by gate `cultural_transmission_gain`),
replay hash match `True` (`c2da7f2ae763d56c`).

### IMPLEMENTED + VERIFIED

- Deterministic seeds/IDs, 26-type event sourcing, state hashing, and layered provenance fingerprints (`src/common/`)
- **LivingBrain v1**: persistent neuron/synapse identities (never reused), per-element
  provenance classes (BIOLOGICAL/DERIVED/EMERGENT/EVOLVED/SYNTHETIC), structural
  event log, resource-constrained growth orchestration, checkpointable registries (`src/brain/living.py`)
- **v2 eligibility plasticity**: persistent eligibility traces + versioned neuromodulatory
  signal (reward/novelty/prediction-error/social/goal) alongside the v1 Hebbian
  compat baseline (`src/brain/eligibility.py`); prediction influences attention/curiosity (opt-in)
- **Autonomy engine**: self-generated goals from needs/curiosity/prediction-error/
  opportunity/social signals, compositional continuous actions (heading/speed/
  duration/intensity) modulated by neural state — no tiny action menu (`src/autonomy/`)
- **Embodiment**: body state with damage/speed-capacity/recovery constraining action (`src/embodiment/`)
- **Grounded language**: symbols bound to grounded concepts; production from internal
  state; measured information transfer via receiver behavior (`src/language/`)
- **Emergent social model**: identity recognition, interaction history, trust from
  outcomes (never hard-coded friendship), persistent relationships (`src/social/`)
- **Genome v2.0**: 9 learning-architecture genes (eligibility decay, neuromodulation
  weights, growth budget, prediction gain, social-learning bias, sleep, communication)
  evolving the learning architecture itself; v1.0 remains exactly valid (`src/genome/`)
- **Speciation**: genome-distance clustering + evidence-backed divergence records (`src/evolution/speciation.py`)
- **LLM control plane**: 14 typed schema-validated commands (SPAWN/START/PAUSE/STOP/
  SAVE/LOAD/CONFIG/EXPERIMENT/COMPARISON/REPLAY/PROPOSE_*); shell/code injection
  rejected; execution log (`src/llm/control.py`) + hash-chained research memory (`src/llm/research_memory.py`)
- **Deep time**: event-driven coarse acceleration with documented approximation model,
  milestone escalation to full-resolution checkpoints, hash-verified replay (`src/timeline/deeptime.py`)
- **Milestone detection**: evidence-backed milestones (structural expansion, cultural
  transmission, overlapping generations, emergent communication, social persistence,
  speciation) with machine-readable certificates (`src/science/milestones.py`)
- **Benchmark suite**: FlyBrain vs LLM-only vs LLM+tools with per-arm budgets and
  per-category results — no aggregate superiority claims (`src/research/benchmark.py`)
- Versioned genome v1.0/v2.0 with deterministic mutation/crossover + provenance (`src/genome/`)
- Real development: neurogenesis, differentiation, migration, axon/dendrite growth,
  synaptogenesis, pruning, apoptosis — invariants enforced (`src/development/`)
- Closed sensorimotor loop in a deterministic grid world; metabolism accounting
  (total energy never exceeds initial + tracked regrowth influx) (`src/world/`, `src/organism/`)
- Population simulation on CPU with overlapping generations, sexual reproduction,
  Pareto selection, separate genetic/cultural lineages (`src/population/`, `src/culture/`)
- Measured teacher→student learning gain with provenance chains; cumulative
  cultural transmission demonstrated across 3 generations in tests (`tests/test_campaign.py`)
- Local GGUF model discovery and scientist loop (`src/llm/`); unavailable models
  never produce text (`tests/test_llm_local.py`)
- Checkpoint/resume campaigns; replay determinism across interrupted execution
- Adversarial robustness: invalid graph dimensions, out-of-bounds indices, stale
  caches, corrupted caches, genome bound violations, population death races
- Hardware benchmarks with provenance (`scripts/run_benchmarks.py`, `diagnostics/benchmarks/`)

### EXPERIMENTAL

- Population simulation runs on **CPU** (single-brain Vulkan path is verified, but
  batched multi-organism GPU stepping is not implemented).
- Dream consolidation is replay-based; no synaptic downscaling model yet.
- CPU/Vulkan integration is trajectory-parity, not bit-exact (documented).
- Local LLM outputs are generated by a 2B-class quantization locally and are
  scientifically weak; the scientist loop always treats them as hypotheses, never ground truth.
- Deep-time coarse mode aggregates lifecycle events (documented approximation);
  it is not neural-resolution-equivalent and never claims to be.

### NOT IMPLEMENTED (no placeholders — explicitly unavailable)

- 3D colony / development-timeline visualizations (data APIs exist).
- Multi-GPU batched organism stepping.
- Synaptic downscaling / homeostatic sleep consolidation models.
- Cross-ecosystem world evolution (world events beyond resources/hazards).

---

## Citation & Third-Party Notices

If you use FlyBrain or the Janelia MaleCNS connectome in your research, please cite:

```bibtex
@article{takemura2023malecns,
  title={A connectome of the male Drosophila central nervous system},
  author={Takemura, Shin-ya and Aso, Yoshinori and Hige, Tatsuya and Wong, Aaron M and Lu, Zhiyuan and Xu, C Shan and Hess, Harald F and Rubin, Gerald M and others},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full licensing information.
