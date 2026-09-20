# FlyBrain Hugging Face Space Deployment

## Model
Docker SDK Space (CPU-only) built from this repository. The image contains
only what the server needs: `src/`, `shaders/`, `manifests/`,
`malecns/data-raw/`, pinned CPU requirements
(`huggingface/requirements-hf.txt`), and the entrypoint
(`huggingface/app.py` → `uvicorn app:app --port 7860`).

Space defaults (`FLYBRAIN_CIRCUIT_SIZE=256`, `FLYBRAIN_GRAPH_MODE=REAL_SUBGRAPH`,
`FLYBRAIN_USE_GPU=0`, `FLYBRAIN_SEED=42`) are overridable via Space variables.

## Honest capability surface
- No Vulkan GPU exists on Spaces: `/api/health` reports
  `backend: cpu_reference` with the CPU device name. Vulkan is never faked.
- No local GGUF weights: the LLM gates report `MODEL_UNAVAILABLE` with
  structured errors; the scientist loop treats model output as hypotheses.
- Graph is the sampled `REAL_SUBGRAPH` (256 neurons default); populations are
  `HEURISTIC`; weights are the derived `synapse_count` transform. All visible
  in the in-app Provenance tab (`/api/provenance`).

## Deploy a new revision
1. Verify locally:
   ```powershell
   .venv\Scripts\python.exe scripts/smoke_hf_space.py
   ```
   Requires `overall: PASS` and `diagnostics/huggingface_verification.json`.
2. Push `huggingface/` (+ `src/`, `shaders/`, `manifests/`, `malecns/data-raw/`)
   to the Space repository (see Space `README.md` front matter in
   `huggingface/README.md`). The Docker build fails fast if imports or data
   are broken (`RUN python -c "import ..."` health gate in the Dockerfile).
3. After the Space builds, verify remotely:
   `GET /api/health`, `/api/version`, `/api/doctor`, `/api/provenance`,
   load `/` (FlyBrain Lab, no CDN required), run one
   `POST /api/simulation/step`. Record results in
   `diagnostics/huggingface_verification.json` (remote section).

## Verification evidence
Local boot verification: `diagnostics/huggingface_verification.json`
(real uvicorn boot, live HTTP checks, CPU backend honesty asserted).
Remote deployment status is recorded in `diagnostics/final_verification.json`
(`hugging_face` block) — `DEPLOYED_VERIFIED` only after real remote checks,
otherwise the exact external blocker with local evidence linked.
