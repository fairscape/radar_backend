# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the RADAR FastAPI service.
#
# - builder stage installs build deps and the package (with the `phase1b`
#   extra so SPECTER2 / Chroma / Ollama clients are available at runtime).
# - runtime stage is a slim image that only carries the installed
#   site-packages + the source tree.
#
# At container start we run `python -m cli.db init` (idempotent) so the
# SQLite file under /app/data is migrated to head before uvicorn binds.

# ---------- builder ----------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps for native wheels (pdfplumber pulls cffi/cryptography on some
# platforms, sentence-transformers wants build-essential to fall back to
# source builds, chromadb's hnswlib needs a C++ compiler).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy only what `pip install` needs to resolve the package.
COPY pyproject.toml ./


# Install CPU-only torch first so the resolver doesn't pull the default
# CUDA wheel (which drags in ~1.8 GB of nvidia_* libs we never use —
# the SPECTER2 path only runs torch.no_grad() + .cpu().numpy()).
RUN pip install --prefix=/install --index-url https://download.pytorch.org/whl/cpu torch

# Install into an isolated prefix so the runtime stage can copy a clean
# tree without dragging build tooling along. NOT editable: the runtime
# stage's site-packages must own a real copy of rag_lib (we still
# COPY cli/ separately because pyproject.toml's `packages.find` excludes
# it — cli is a top-level script package).
RUN pip install --prefix=/install ".[dev,phase1b]"
RUN pip install --prefix=/install ".[dev]"

COPY README.md ./
COPY rag_lib ./rag_lib
COPY cli ./cli

# ---------- runtime ----------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Runtime libs: curl for the compose healthcheck, libgomp1 for
# scikit-learn / sentence-transformers, libstdc++ for hnswlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pull the installed Python tree out of the builder.
COPY --from=builder /install /usr/local

WORKDIR /app

# rag_lib's deps are installed in site-packages from the builder, but
# we COPY the source tree directly because:
#   - pyproject.toml ships no `package-data` config, so the .sql files
#     under rag_lib/db/migrations/ aren't bundled with the wheel; the
#     migration runner globs them off disk via Path(__file__).parent.
#   - cli/ is excluded from `packages.find`, so it has to land on
#     PYTHONPATH explicitly anyway.
# PYTHONPATH=/app puts /app/rag_lib ahead of the site-packages copy,
# so the runtime imports the COPY'd tree (with migrations) — the
# site-packages install only mattered for transitive deps.
COPY rag_lib ./rag_lib
COPY cli ./cli
COPY pyproject.toml ./

# State directories. compose mounts host paths over these, but creating
# them here lets the image run standalone (e.g. `docker run` without
# compose) and keeps `cli.db init` happy on first start.
RUN mkdir -p /app/data /app/vault /app/chroma

EXPOSE 8000

# `cli.db init` is idempotent — applies any pending migrations, otherwise
# logs "no pending migrations" and exits 0. Runs every container start
# so deployments that bumped the migration set self-heal without a manual
# `docker exec`.
CMD ["sh", "-c", "python -m cli.db init && exec python -m cli.serve --host 0.0.0.0 --port 8000"]
