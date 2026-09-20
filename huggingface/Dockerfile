# FlyBrain Lab — Hugging Face Space (Docker SDK, CPU-only).
#
# Build context is the Space repository ROOT, which mirrors this repo's
# Space-relevant subset:
#   Dockerfile  (this file, uploaded from huggingface/Dockerfile)
#   app.py      (uploaded from huggingface/app.py)
#   requirements-hf.txt (uploaded from huggingface/requirements-hf.txt)
#   README.md   (uploaded from huggingface/README.md, carries Space front matter)
#   src/ shaders/ manifests/ malecns/data-raw/
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLYBRAIN_CIRCUIT_SIZE=256 \
    FLYBRAIN_GRAPH_MODE=REAL_SUBGRAPH \
    FLYBRAIN_USE_GPU=0 \
    FLYBRAIN_SEED=42

WORKDIR /app

# Pinned CPU-only dependencies first (better layer caching).
COPY requirements-hf.txt ./requirements-hf.txt
RUN pip install --no-cache-dir -r requirements-hf.txt

# Application source + connectome data + provenance manifests.
COPY src/ ./src/
COPY shaders/ ./shaders/
COPY manifests/ ./manifests/
COPY malecns/data-raw/ ./malecns/data-raw/
COPY app.py ./app.py

EXPOSE 7860

# Health-gated boot: fail fast if imports or data are broken.
RUN python -c "import src.version, src.connectome.types, src.ui.server; print('boot-import OK', src.version.VERSION)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
