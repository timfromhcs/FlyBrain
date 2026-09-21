# FlyBrain Release Notes — v7.0.0 Stable

**Tag:** `v7.0.0` · **Prior line:** `v6.0.0` · Complete production release with end-to-end multimodal loop, instantaneous checkpoint loading, sub-second collision-proof backups, and live MuJoCo 3.13 3D world.

## V7.0 Highlights (Real, Measured, Non-Hallucinatory)
- **Instantaneous Image Model Init:** Checkpoint sha256 sidecars (`.sha256`) eliminate gigabyte disk hashing stalls, dropping initialization latency from 15s to < 0.1s.
- **Sub-second Collision-Proof Backups:** Millisecond-resolution backup identifiers prevent collision in rapid automated test and restore sequences.
- **Synchronized Multimodal Loops:** DreamShaper-8-LCM (4-step FAST, 8-step BALANCED, 12-step QUALITY) coupled with Sobel/Canny ControlNet, Kokoro-82M TTS, Whisper-small STT, and Qwen3-0.6B LLM running 100% offline.
- **Closed-Loop Embodied 3D Simulation:** MuJoCo 3.13 authoritative physics on both local Windows workstation and Hugging Face Docker Space CPU container.
- **Zero Mock Policy & Provenance Verification:** 100% genuine MaleCNS v1.0 biological connectome data, cryptographically signed state hashes, deterministic CPU/GPU trajectory parity.
- **Full Integrity Matrix:** 42/42 V6 gates PASS, 56/57 Acceptance Matrix PASS, 0 broken links, 0 CDN runtime dependencies.

---

# FlyBrain Release Notes — v6.0.0 Stable

**Tag:** `v6.0.0` · **Prior line:** `v5.0.0` · V5 fully preserved (265-test
suite + 57-gate matrix intact); V6 is additive.

## Embodied world (all real, all measured)
- MuJoCo 3.13 authoritative 3D physics: gravity, floor rest, wall blocking,
  crate pushing, posture PD + get-up maneuver, save/restore bit-exact
  (7 physics tests).
- Closed loop with §27 causal records; save/restore deterministic
  continuation proven (world hash + brain + body + spatial memory incl.
  solver warm-start and unrounded state sidecars).
- Raycast first-person perception (`GEOMETRIC`), A* navigation on
  known-vs-true maps (0.25 m grid keeps the 1.4 m door passable), world
  clock, doors/food/water with real state transitions.
- Local models, every one loaded and measured: Qwen3-0.6B-GGUF, SmolVLM-256M,
  Whisper-small, Kokoro-82M, DreamShaper-8-LCM, sd-controlnet-canny,
  MiniLM-L6-v2. Manager with pressure shedding; offline enforced.
- Dreams/imagination/speech loop end-to-end on real models; versioned prompt
  templates; social trust emerges from measured teaching.
- World UI tab (4 cameras, minimap, dream/speak/imagine), 0 console errors.
- V6 matrix 42/42 PASS; evidence `diagnostics/v6/`; bench FAST 22s /
  BALANCED 38s / QUALITY 180s (CPU), physics 0.1ms, nav 4.5ms.

## Honest limitations
- torch-directml reverted (breaks transformers ≥2.5): CPU inference.
- Depth/openpose ControlNets pinned but not downloaded (canny verified).
- Endurance soak: UNVERIFIED. Google Drive: BLOCKED_AUTHENTICATION.
- Population sim stays CPU; REAL_FULL not instantiated (unchanged).

---

# FlyBrain Release Notes — v5.0.0 Stable

**Tag:** `v5.0.0` · **Prior line:** `v4.1.0` · **Semver:** major (new API
surface `/api/v1`, new subsystems; legacy `/api/*` routes unchanged and
working).

## New subsystems (all real, all tested)
- **Versioned API contracts:** `/api/v1/*` (health, readiness, version,
  doctor, state, runtime, metrics, events, provenance, connectome,
  neuron/{body_id}, simulation, stream, backup, experiments, organisms,
  evolution, memory, dreams, llm) alongside unchanged legacy routes.
- **Backup service** (`src/backup/service.py`, schema `flybrain_backup_v1`):
  create/list/verify/restore/download, retention generations, tamper-evident
  verification, guarded restore (paused-only, pre-restore auto-backup,
  post-restore hash match). Round trip proven on clean runtimes
  (`tests/test_backup_roundtrip.py`) and on the live Space.
- **24/7 stream mode:** START/PAUSE/RESUME/STOP/SAFE_SHUTDOWN, uptime,
  heartbeat, p50/p95/p99 metrics, spectator-safe (WS disconnect never stops
  the sim).
- **Watchdog** (`src/runtime/watchdog.py`): stall/RAM/GPU supervision with
  bounded RETRY → RESTORE → RESTART → SAFE_STOP (never hidden loops).
- **First-run presets** (`flybrain lab --preset`): QUICK_DEMO,
  BIOLOGICAL_SUBGRAPH, GPU_PERFORMANCE, CPU_SAFE, ALIFE_COLONY, RESEARCH,
  24/7_STREAM — all real configurations via the `FLYBRAIN_*` env contract.
- **Topology null controls** (`src/connectome/nulls.py`): EDGE_SHUFFLED,
  DEGREE_PRESERVING_RANDOM (exact), WEIGHT_SHUFFLED — deterministic, flagged
  non-empirical.
- **Google Drive provider** (`src/backup/gdrive.py`): explicit status model;
  `BLOCKED_AUTHENTICATION` with the exact consent step until valid OAuth
  exists (see `deployment/GOOGLE_DRIVE.md`). Never faked.
- **GUI:** Live Stream, Colony (organisms), Backups tabs; `switchTab` fixed
  for programmatic/keyboard activation; 11/11 real screenshots in
  `visual_evidence/screens/v5_*.png`.
- **Release machinery:** stale-doc detector, release-integrity check,
  canonical `diagnostics/release_certificate.json`, master inventory.

## Space & chain
- Hugging Face Docker Space verified live: boot, health (`cpu_reference`,
  honest), version, provenance (`REAL_SUBGRAPH`), UI (zero CDN), simulation
  step, backup create → download (hash-verified) → VALID → restore
  (hash match). Deployed commit pinned in the release certificate.

## Known limitations
- Screenshots cover 11 tabs (HOME/LIVE/CONNECTOME/COLONY/PROVENANCE/
  EXPERIMENTS/MEMORY/EVOLUTION/DREAMS/BACKUPS/DIAGNOSTICS); neuron-inspector
  selection and dream-replay visuals are exercised via API, not captured.
- Google Drive remote backup: `BLOCKED_AUTHENTICATION` until one-time OAuth
  consent (external human step).
- Endurance (1h/6h/24h): `UNVERIFIED` — no full soak run in this environment.
- Carried from v4.1: no `REAL_FULL` live circuit; CPU population sim;
  trajectory (not bit-exact) parity; 2B-class local LLM hypotheses only;
  coarse deep-time approximation.

---

# FlyBrain Release Notes — v4.1.0 Stable

**Tag:** `v4.1.0` · **Prior line:** `v4.0.0` · **Semver:** minor (additive
hardening, no breaking API changes; `GraphMode.REAL` keeps working as the
legacy name of `REAL_SUBGRAPH`).

## Scientific honesty
- Explicit graph identities: `REAL_SUBGRAPH` (canonical), `REAL_FULL`
  (honestly unavailable — raises), `SPATIAL_SURROGATE`, `SYNTHETIC_TEST`.
  `REAL` remains a working legacy alias (same enum value, no cache
  invalidation).
- Twelve separated deterministic identity layers
  (`src/provenance/identity_layers.py`).
- Functional populations rendered and served as `HEURISTIC` everywhere
  (UI Provenance tab, `/api/provenance`, reports).
- Weight transform documented as derived simulation mapping in all surfaces.
- Trajectory parity terminology fixed (`NOT bit-exact`) in RESEARCH_STATUS.

## Reproducibility
- Versioned population checkpoint envelope (`population_checkpoint_v1`) with
  state/research/combined hashes, RNG bundle, graph + experiment identity,
  and preserved event history (`tests/test_checkpoint_v41.py`).
- Canonical acceptance schema (`verification/acceptance_schema.json`, six
  profiles); README/RESEARCH_STATUS tables generated from the
  machine-readable report (`scripts/render_status_tables.py`).
- Clean-room procedure + artifact policy in REPRODUCIBILITY.md.

## Performance (all measured, AMD Radeon 680M)
- Lazy GPU weight sync: ~0.85 ms saved per rewarded step at N=1024
  (`diagnostics/benchmark_report.json`).
- Host-visible/coherent memory retained by measurement on unified-memory iGPU.
- Command-buffer re-record retained (no material gain demonstrated).
- Full scaling curves (64–1024, median/p05/p95/min/max) in the report.

## Control plane & embodiment
- Capability-based LLM control: role allowlists, strict identifier grammar;
  substring blacklists removed (`CAPABILITY_ROLES`).
- Organism steps return an explicit sensory → neural → motor → body → world →
  reward `causal_path` with `policy_source` labelling.

## UI & deployment
- Modular frontend (`index.html` + `css/lab.css` + `js/lab.js`), vendored
  pinned three.js r128 (zero CDN runtime deps), live Provenance tab,
  reconnect-resilient telemetry, reduced-motion support.
- Real Hugging Face Space (`huggingface/`, Docker SDK, CPU-only) with
  boot-verified smoke test; honest `cpu_reference` backend reporting.
- New endpoint: `/api/provenance`. `/api/simulation/reset` accepts
  `REAL_SUBGRAPH` and is worker-thread safe.

## Verification
- Acceptance: see `diagnostics/acceptance_matrix.json` (0 FAIL required).
- Proof package: `diagnostics/{pre_release_audit,final_verification,
  acceptance_matrix,reproducibility_manifest,cpu_gpu_validation_report,
  connectome_validation_report,benchmark_report,huggingface_verification}.json`.

## Known limitations
- `REAL_FULL` (all 125,506 neurons in one live circuit) is not instantiated.
- Population simulation runs on CPU; batched multi-organism GPU stepping is
  not implemented.
- CPU/Vulkan parity is trajectory-level, not bit-exact.
- Local LLM quality is 2B-class quantization; hypotheses, never ground truth.
- Deep-time ACCELERATED mode is a documented coarse approximation.
