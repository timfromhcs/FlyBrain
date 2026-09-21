---
title: FlyBrain Lab
emoji: 🪰
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
short_description: Drosophila connectome research workstation
---

# FlyBrain Lab — Hugging Face Space (CPU build)

Production build of [timfromhcs/FlyBrain](https://github.com/timfromhcs/FlyBrain) v7.0.x:
connectome research backend + embodied 3D world (MuJoCo physics runs for
real on Space CPUs).

- **Graph:** `REAL_SUBGRAPH` — bounded sampled subgraph of Janelia MaleCNS v1.0
  (256 neurons by default; the full 125,506-neuron graph is not instantiated —
  see the in-app Provenance tab).
- **Backend:** CPU LIF reference (`cpu_reference`). This Space has no Vulkan
  GPU; `/api/health` reports the real backend and never fakes GPU telemetry.
- **Populations** (visual, auditory, …) are coordinate heuristics, labelled
  `HEURISTIC` in the UI and `/api/provenance` — never EM annotations.
- **Weights** are a `synapse_count`-derived simulation transform, not measured
  conductances.
- **World:** `/api/v1/world/*` runs the real MuJoCo simulation; heavy AI
  models (LLM/VLM/STT/TTS/image) report `UNAVAILABLE` here (no weights/GPU
  on the Space) and stay fully local in the desktop build.

Endpoints: `/api/health` · `/api/version` · `/api/doctor` · `/api/state` ·
`/api/telemetry` · `/api/provenance` · `/api/connectome` · `/ws/telemetry` ·
`/api/v1/world/state` · `/api/v1/world/geometry` · `/api/v1/models`.
