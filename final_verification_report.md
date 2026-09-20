# FlyBrain: Final Technical & Behavioral Verification Report

> **v4.1.0 note (2026-09-20):** this narrative report is frozen at v4.0.0 and
> kept for history. The canonical machine-readable evidence for v4.1.0 is
> `diagnostics/final_verification.json` (with the 8-report proof package);
> acceptance totals in README/RESEARCH_STATUS are generated from
> `diagnostics/acceptance_matrix.json` via `verification/acceptance_schema.json`.
> Where this document says 44/44, read the canonical report (57-category schema).

**Framework**: FlyBrain — biologically grounded artificial-life research framework
**Target Biological Dataset**: Janelia MaleCNS v1.0 (`male-cns:v1.0`)
**Host Environment**: Windows 11, AMD Ryzen 7 7735HS, AMD Radeon(TM) Graphics, Vulkan SDK 1.4.357
**Report date**: 2026-09-15 · **Overall Status**: **PASSED (44/44 acceptance gates, 200+ tests)**

> Supersedes all earlier verification reports. Historical claims (e.g. 32/32 gates,
> 111 tests, 7-living/4-births canonical numbers, hash `70a5e4b29e06c320`) are obsolete.
> The canonical hash changed honestly with the provenance fix (stale graph_hash after
> weight mutations — see bugs fixed, #22).

---

## 1. Verification commands (all runnable, all green)

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"   # 200+ pass
.venv\Scripts\python.exe scripts/run_acceptance_matrix.py               # 44/44 PASS
.venv\Scripts\python.exe scripts/verify_docs_consistency.py             # incl. verification contract
.venv\Scripts\python.exe src/main.py validate-vulkan                    # 9/9 CPU/GPU parity (spike-exact)
.venv\Scripts\python.exe scripts/run_alife_experiment.py --population 6 --ticks 60 --seed 7
.venv\Scripts\python.exe scripts/run_alife_experiment.py --verify diagnostics/alife_experiments/alife_p6_t60_s7.json  # match=True
.venv\Scripts\python.exe scripts/capture_milestone_evidence.py          # milestones + escalation + replay
.venv\Scripts\python.exe scripts/run_long_campaign.py --generations 4 --population 6 --seed 11
.venv\Scripts\python.exe scripts/run_benchmarks.py
```

## 2. Architecture (as built)

1. **Biological source**: 125,506 somas; 99,301 connection rows; provenance manifests
   hash-verified; `GraphMode.REAL/SPATIAL_SURROGATE/SYNTHETIC_TEST` strictly separated.
2. **LivingBrain v1** (`src/brain/living.py`): persistent neuron/synapse identities
   (never reused), per-element provenance classes (BIOLOGICAL/DERIVED/EMERGENT/EVOLVED/
   SYNTHETIC), structural event log, resource-constrained growth orchestration
   (`growth_budget=0` blocks neurogenesis), versioned checkpoint/restore of registries.
3. **LIF + plasticity**: identical equations CPU/GPU; v1 three-factor Hebbian compat
   baseline; **v2 eligibility mode** with persistent traces + versioned neuromodulatory
   signal (reward/novelty/prediction-error/social/goal); prediction influences
   attention/curiosity (opt-in, `prediction_gain` gene). Stale graph_hash after weight
   mutations is FIXED (both paths refresh provenance hash).
4. **Autonomy engine** (`src/autonomy/`): self-generated goals from needs/curiosity/
   prediction-error/opportunity/social signals; compositional continuous actions
   (heading/speed/duration/intensity) modulated by neural state; self-evaluation with
   progress/abandonment; no human task commands required.
5. **Embodiment** (`src/embodiment/`): body state with damage/speed-capacity/recovery.
6. **Grounded language** (`src/language/`): symbols bind only to grounded concepts;
   production from internal state; comprehension; grounding strength measured.
7. **Social model** (`src/social/`): recognition, interaction history, trust via delta
   rule from outcomes, persistent relationships; v2 teaching is trust-biased.
8. **Genome v2.0** (`src/genome/`): 15 v1 params + 9 learning-architecture genes
   (eligibility decay, neuromodulation weights, growth budget fraction, prediction gain,
   social-learning bias, sleep duration, communication tendency, curiosity drive);
   v1.0 exactly valid (bit-exact legacy replays); version-preserving mutation/crossover.
9. **Speciation** (`src/evolution/speciation.py`): genome-distance clustering;
   divergence recorded only with measured distance evidence.
10. **LLM control plane** (`src/llm/control.py`): 14 typed schema-validated commands;
    shell/code injection rejected; full execution log. **Research memory**
    (`src/llm/research_memory.py`): hash-chained append-only, tamper-evident.
11. **Deep time** (`src/timeline/deeptime.py`): coarse event-driven acceleration with
    documented approximation model; milestone escalation to full-resolution checkpoints;
    hash-verified high-res replay.
12. **Milestones** (`src/science/milestones.py`): evidence-backed detection
    (structural expansion, cultural transmission, overlapping generations, emergent
    communication, social persistence, speciation) + machine-readable certificates.
13. **Benchmark suite** (`src/research/benchmark.py`): FlyBrain vs LLM-only vs LLM+tools;
    per-arm budgets; embodied categories NOT_APPLICABLE for LLM-only; no aggregate
    superiority claims; LLM arms SKIP with reason without a model.
14. **Vulkan engine**: scored device selection, persistent buffers, lifecycle
    stress-tested; CPU reference remains the correctness baseline.
15. **Concurrency/memory/CLI/UI**: unchanged from v2.0.0 baseline (RLock engine,
    SQLite stores, non-blocking UI).

## 3. Canonical results

- **ALife legacy canonical** (seed 7, pop 6, 60 ticks): 8 living, 2 births, 0 deaths,
  generations `[0, 1]`, 0 teaching sessions (short run), replay hash match `True`
  (`c2da7f2ae763d56c`). Teaching gain separately verified (gate 24).
- **Autonomous v2 evidence run** (`diagnostics/milestones/milestone_evidence.json`,
  seed 301): coarse deep-time campaign; milestones STRUCTURAL_EXPANSION +
  EMERGENT_COMMUNICATION (+OVERLAPPING_GENERATIONS in variants) with certificates;
  checkpoint escalation + replay_verified=true.
- **Acceptance matrix**: workstation 44/44 PASS; cloud CI 40 PASS / 4 SKIP / 0 FAIL.
- **Shaders**: committed SPV byte-identical to fresh `glslc` compiles.
- **CPU/Vulkan parity**: trajectory parity, spike-exact, max abs diff < 1e-4
  (NUMERICALLY_EQUIVALENT + SPIKE_EXACT — see `diagnostics/verification_contract.json`).

## 4. Capability classification

### IMPLEMENTED + VERIFIED
LivingBrain identity/provenance/growth, v2 eligibility+neuromodulation, autonomy engine,
embodiment, grounded language, emergent social trust, genome v2 architecture genes,
speciation evidence, LLM control plane + research memory, deep-time escalation/replay,
milestone detection/certificates, benchmark suite, canonical ALife replay, memory,
dreams, evolution, colony API, 44-gate matrix.

### IMPLEMENTED + EXPERIMENTAL
CPU-only population stepping; replay-based dream consolidation; deep-time coarse
approximation; 2B-class local LLM hypotheses (never ground truth).

### NOT IMPLEMENTED
3D colony/development-timeline visualizations; multi-GPU batched stepping; synaptic
downscaling; world events beyond resources/hazards (day/night, sound, terrain).

### UNAVAILABLE ON CURRENT ENVIRONMENT
None — Vulkan + local GGUF present on workstation; CI skips documented.

## 5. Bugs fixed in this release pass

1. **Stale graph_hash after weight mutations** (v1 Hebbian CPU/GPU + v2 eligibility):
   snapshot/restore equivalence broke because restored graphs recomputed the hash while
   originals carried stale hashes. Both plasticity paths now refresh `graph_hash`.
2. `LivingBrain.attach` merged seed-reconciled registries with payload instead of
   replacing them (restored synapse count inflated). Fixed: clear-then-restore.
3. DevelopmentEngine activity arrays not resized after same-cycle neurogenesis
   (IndexError in prune/apoptosis). Fixed via `_fit_activity`.
4. v1 Genome accepted architecture params (version not strict). Fixed: strict per-version
   required sets; v1 stays exactly 15 params.
5. Legacy teacher pairing preserved but v2 trust-bias added without touching v1 path.
6. Founder RNG consumption order preserved bit-exactly when introducing v2 genes.
7. Documentation drift: canonical numbers in README did not match the stored artifact
   (7 living/4 births/12 teaching vs actual 8/2/0). Corrected to measured values.

## 6. Proof locations
- `diagnostics/acceptance_matrix.json` (44/44), `diagnostics/pre_evolution_baseline.json`
  (never overwritten), `diagnostics/engineering_audit.json`,
  `diagnostics/verification_contract.json` (normative terminology),
  `diagnostics/milestones/milestone_evidence.json` (milestones + escalation + replay),
  `diagnostics/alife_experiments/alife_p6_t60_s7.json` (canonical replay),
  `diagnostics/benchmarks/`, `diagnostics/campaigns/`.
- New test files: `tests/test_living_brain.py` (12), `tests/test_eligibility.py` (14),
  `tests/test_autonomy.py` (18), `tests/test_language_social.py` (11),
  `tests/test_genome_v2_speciation.py` (18), `tests/test_control_plane.py` (15),
  `tests/test_deeptime_milestones.py` (11), `tests/test_benchmark.py` (6).
