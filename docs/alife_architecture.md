# FlyBrain Artificial-Life Layer — Implementation Status (2026-09-15)

> Scientific positioning: FlyBrain is a **biologically grounded artificial-life research
> framework**, not a literal brain model. Research question: *What structures and behaviors
> emerge when a biologically grounded neural substrate develops, learns, socializes,
> transmits knowledge, reproduces, mutates, and evolves across overlapping generations?*

## IMPLEMENTED + VERIFIED (matrix gates 20–44; full suite 200+ tests)

| Layer | Module | Evidence |
|---|---|---|
| Deterministic identity/seeds | `src/common/determinism.py` | stable SHA-256 IDs, derived subseeds |
| Event sourcing (26 types) | `src/common/events.py` | hash-chained, JSONL serializable |
| Layered provenance hashing | `src/common/provenance.py` | dataset/model/graph/brain/world/organism/population/event layers folded into experiment fingerprint |
| **LivingBrain v1** | `src/brain/living.py` | persistent identities, provenance classes (BIOLOGICAL/DERIVED/EMERGENT/EVOLVED/SYNTHETIC), structural events, resource-constrained growth; `tests/test_living_brain.py` |
| **v2 eligibility + neuromodulation** | `src/brain/eligibility.py` | persistent traces, versioned signal (reward/novelty/prediction-error/social/goal); `tests/test_eligibility.py` |
| **Autonomy engine** | `src/autonomy/engine.py` | self-generated goals, compositional continuous actions, self-evaluation; `tests/test_autonomy.py` |
| **Embodiment** | `src/embodiment/body.py` | damage/speed-capacity/recovery constraints |
| **Grounded language** | `src/language/grounded.py` | symbol↔concept↔sensory grounding, production from internal state; `tests/test_language_social.py` |
| **Emergent social model** | `src/social/model.py` | trust from interaction outcomes, persistent relationships |
| **Genome v2.0** | `src/genome/schema.py` | 9 learning-architecture genes; v1 exact-compat; `tests/test_genome_v2_speciation.py` |
| **Speciation** | `src/evolution/speciation.py` | genome-distance clustering, evidence-backed divergence |
| **LLM control plane** | `src/llm/control.py` | 14 typed schema-validated commands, injection-rejecting; `tests/test_control_plane.py` |
| **Research memory** | `src/llm/research_memory.py` | hash-chained append-only, tamper-evident |
| **Deep time** | `src/timeline/deeptime.py` | coarse acceleration + milestone escalation + hash-verified replay; `tests/test_deeptime_milestones.py` |
| **Milestone detection** | `src/science/milestones.py` | evidence-backed milestones + certificates |
| **Benchmark suite** | `src/research/benchmark.py` | FlyBrain vs LLM-only vs LLM+tools, budgets documented; `tests/test_benchmark.py` |
| Genome v1.0 (15 params) | `src/genome/schema.py` | validate/hash/serialize, bounds-checked |
| Mutation + crossover | `src/genome/operators.py` | deterministic, full provenance records, version-preserving |
| Development lifecycle | `src/development/engine.py` | graph invariants enforced after every op |
| Grid world + sensors/actuators | `src/world/environment.py` | grazing, hazards, regrowth w/ tracked influx |
| Organism + lifecycle + metabolism | `src/organism/organism.py` | legacy + autonomy modes, living-brain wiring, sleep gene |
| Population + overlapping generations | `src/population/population.py` | coexistence, teaching with emergent-trust bias (v2 only) |
| Teaching + cultural transmission | `src/culture/transmission.py` | gain measured, provenance chains |
| Canonical experiment + verify | `scripts/run_alife_experiment.py` | bit-exact replay (hash match) |
| Milestone evidence capture | `scripts/capture_milestone_evidence.py` | real run → milestones + escalation + replay verification |
| Colony UI API | `src/ui/server.py` | serves real live state, tested |

Canonical result (`seed 7, pop 6, 60 ticks`, legacy v1 path): 8 living, 2 births,
0 deaths, generations `[0, 1]` coexisting, 0 teaching sessions in this short run
(teaching gain separately verified), replay hash match `True` (`c2da7f2ae763d56c`).

Autonomous v2 evidence run (`scripts/capture_milestone_evidence.py`, seed 301):
coarse deep-time campaign with self-generated goals, emergent communication,
structural expansion, milestone certificates, checkpoint escalation with
hash-verified replay (`diagnostics/milestones/milestone_evidence.json`).

## EXPERIMENTAL

- Population simulation runs on **CPU** (single-brain Vulkan path is verified with
  trajectory parity, but batched multi-organism GPU stepping is not implemented).
- Dream consolidation is replay-based; no synaptic downscaling model yet.
- CPU/Vulkan integration is trajectory-parity, not bit-exact (see
  `diagnostics/verification_contract.json` — terms are normative).
- Deep-time coarse mode aggregates lifecycle events; NOT neural-resolution-equivalent.
- Local LLM outputs are 2B-class and scientifically weak; always hypotheses.

## NOT IMPLEMENTED (explicitly unavailable, never faked)

- 3D colony / development-timeline UI panels (data APIs exist; visualization pending).
- Multi-GPU batched organism stepping.
- Synaptic downscaling / homeostatic sleep consolidation models.
- World events beyond resources/hazards (day/night, sound, terrain).

## Key model parameters (documented, not hidden)

- Metabolism: `cost = rate*(0.5 + activity + 0.25*move)`; growth `0.02` energy/neuron;
  grazing radius 1, assimilation `0.8`; reproduction cost `0.25` energy/parent.
- Autonomy-mode growth: `budget = clip((energy-0.4)*5*(0.5+growth_budget_fraction), 0, 4)`
  per development cycle — poor organisms cannot grow (resource-constrained).
- Genome v2 architecture genes (evolvable): `eligibility_decay`,
  `neuromod_novelty_weight`, `neuromod_prediction_weight`, `growth_budget_fraction`,
  `prediction_gain`, `social_learning_bias`, `sleep_duration`,
  `communication_tendency`, `curiosity_drive`.
- v1 organisms keep the exact legacy path (bit-exact with pre-v2 replays).

## Acceptance matrix (canonical; totals generated, never copied)

All gates are executable behavioral assertions, including:
CSR directionality (`csr_directionality`), biological edge semantics and provenance
(`biological_edge_semantics`), REAL annotation honesty (`real_annotation_integrity`),
plasticity reward causal effect (`plasticity_causal_effect`), checkpoint/resume
equivalence (`checkpoint_continuation`), local model discovery/inference
(`llm_model_discovery_and_inference`), and LLM failure mode/tool safety
(`llm_failure_mode_and_tool_safety`). Totals live in
`diagnostics/acceptance_matrix.json` (schema
`verification/acceptance_schema.json`).
