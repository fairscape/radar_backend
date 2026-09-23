# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the RADAR FastAPI service — lean profile.
#
# What this image carries: the API, SPECTER2 (pre-downloaded so a fresh
# pod does not fetch ~450 MB before it can embed) and the OpenAlex
# gatherer. UMLS extraction, the MedCPT reranker and the Ollama chat
# path are still in the code but off by configuration
# (RADAR_UMLS_ENABLED / RADAR_RERANKER_ENABLED / RADAR_LLM_PROVIDER);
# scispacy and its sci-lg model are NOT installed here. To bring UMLS
# back, install the `umls` extra (needs Python <3.13) and bake
# scripts/build_topic_index.py output into RADAR_UMLS_CACHE_DIR.
#
# At container start we run `python -m cli.db init` (idempotent) so the
# SQLite file under /app/data is migrated to head before uvicorn binds.

ARG PYTHON_VERSION=3.14

# ---------- builder ----------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps for native wheels (pdfplumber pulls cffi/cryptography on some
# platforms, chromadb's hnswlib needs a C++ compiler, git for the
# researcher-profiles pins).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy only what `pip install` needs to resolve the package.
COPY pyproject.toml README.md ./

# Install CPU-only torch first so the resolver doesn't pull the default
# CUDA wheel (which drags in ~1.8 GB of nvidia_* libs we never use —
# the SPECTER2 path only runs torch.no_grad() + .cpu().numpy()).
RUN pip install --prefix=/install --index-url https://download.pytorch.org/whl/cpu torch

# Install into an isolated prefix so the runtime stage can copy a clean
# tree without dragging build tooling along. NOT editable: the source
# tree is COPY'd separately in the runtime stage because pyproject.toml's
# `packages.find` excludes cli/ and ships no package-data for the SQL
# migrations.
RUN pip install --prefix=/install ".[dev,phase1b]"

# ---------- runtime base ----------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Runtime libs: curl for healthchecks, libgomp1 for torch / scikit-learn.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pull the installed Python tree out of the builder.
COPY --from=builder /install /usr/local

# ---------- models: pre-download SPECTER2 -------------------------------
# Mirrors rag_lib/embedders.py (allenai/specter2_base + the allenai/specter2
# proximity adapter) with the libraries directly so this stage is cached
# independently of the source tree. If the model id changes there, update
# it here too — otherwise the pod just downloads at first use.
FROM runtime-base AS models

ENV HF_HOME=/app/hf_cache

RUN python -c "\
from transformers import AutoTokenizer; \
from adapters import AutoAdapterModel; \
AutoTokenizer.from_pretrained('allenai/specter2_base'); \
m = AutoAdapterModel.from_pretrained('allenai/specter2_base'); \
m.load_adapter('allenai/specter2', source='hf', load_as='proximity', set_active=True); \
print('specter2 baked')"

# ---------- runtime ----------------------------------------------------
FROM runtime-base AS runtime

WORKDIR /app

# rag_lib's deps are installed in site-packages from the builder, but
# we COPY the source tree directly because:
#   - pyproject.toml ships no `package-data` config, so the .sql files
#     under rag_lib/db/migrations/ aren't bundled with the wheel; the
#     migration runner globs them off disk via Path(__file__).parent.
#   - cli/ is excluded from `packages.find`, so it has to land on
#     PYTHONPATH explicitly anyway.
# PYTHONPATH=/app puts /app/rag_lib ahead of the site-packages copy,
# so the runtime imports the COPY'd tree (with migrations).
COPY rag_lib ./rag_lib
COPY cli ./cli
COPY scripts ./scripts
COPY pyproject.toml ./

# State directories. The deployment mounts scratch over data/vault/chroma;
# creating them here lets the image run standalone and keeps
# `cli.db init` happy on first start.
RUN mkdir -p /app/data /app/vault /app/chroma

# Pre-downloaded SPECTER2. A deployment that overrides HF_HOME to a
# scratch mount throws the bake away and re-downloads on every restart.
COPY --from=models /app/hf_cache /app/hf_cache
ENV HF_HOME=/app/hf_cache

EXPOSE 8000

# `cli.db init` is idempotent — applies any pending migrations, otherwise
# logs "no pending migrations" and exits 0. Runs every container start
# so deployments that bumped the migration set self-heal without a manual
# `docker exec`.
CMD ["sh", "-c", "python -m cli.db init && exec python -m cli.serve --host 0.0.0.0 --port 8000"]
