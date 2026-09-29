#!/bin/bash
# Start RADAR backend using the conda env (CUDA-enabled PyTorch).
# The Apptainer SIF only has torch+cpu; this env has torch+cu124
# so the MedCPT reranker can run on GPU.
# Usage: ./start_backend.sh

set -e

DEPLOY_DIR="/bigtemp/nkw3mr/radar_deployment"
# Python 3.12: rp-sdk (the Prosopia reader) needs >=3.12 and scispacy 0.6.2
# caps Python at <3.13, so 3.12 is the only version that runs both. Built
# and verified 2026-09-28 -- nmslib, the UMLS index, SPECTER2 + the
# proximity adapter on GPU, MedCPT, and an anonymous Prosopia read.
CONDA_ENV="$DEPLOY_DIR/conda_env312"
SRC_DIR="/p/realai/lei/radar_deployment/radar_backend"
ENV_FILE="$(dirname "$0")/.env"

# Ensure data directories exist
mkdir -p "$DEPLOY_DIR"/{data,vault,chroma,logs}

echo "Starting RADAR backend on port 8000 (conda env, CUDA enabled)..."
echo "  Env:   $CONDA_ENV"
echo "  Src:   $SRC_DIR"
echo "  Data:  $DEPLOY_DIR/data"
echo "  Vault: $DEPLOY_DIR/vault"
echo "  Logs:  $DEPLOY_DIR/logs"

# Copy .env into the working directory so the app picks it up. pydantic
# reads env_file relative to the cwd, and the cwd has to stay $SRC_DIR
# because RADAR_VAULT_DIR and friends are relative paths that resolve to
# symlinks living there -- so this copy cannot be pointed elsewhere.
#
# $SRC_DIR is on a filesystem that has hit 100% full before (2026-09-28).
# With `set -e` a failed copy meant the backend would not start at all,
# and the cron watchdog would then retry every 5 minutes forever. An
# out-of-date .env that is already correct is far better than no service,
# so only a missing destination is fatal.
if ! cp "$ENV_FILE" "$SRC_DIR/.env" 2>/dev/null; then
    if [ -s "$SRC_DIR/.env" ]; then
        echo "WARN: could not refresh $SRC_DIR/.env (disk full?) -- reusing the existing one" >&2
        diff -q "$ENV_FILE" "$SRC_DIR/.env" >/dev/null 2>&1 \
            || echo "WARN: and it differs from $ENV_FILE" >&2
    else
        echo "FATAL: cannot write $SRC_DIR/.env and no usable copy is there" >&2
        exit 1
    fi
fi

cd "$SRC_DIR"

# Refuse to start on a missing database when backups exist. The backend
# would otherwise create an empty one and serve it as if nothing happened
# (2026-09-21: the /var/tmp directory had vanished under a long-running
# process; a restart silently started fresh). Restore first:
#   gunzip -c $DEPLOY_DIR/backups/hourly/<newest>.db.gz > <RADAR_DB_PATH>
# (evaluated after cd so a relative RADAR_DB_PATH resolves like the app does)
DB_PATH=$(grep -E '^RADAR_DB_PATH=' "$ENV_FILE" | cut -d= -f2- | tr -d '"\'' ')
if [ -n "$DB_PATH" ] && [ ! -f "$DB_PATH" ] && ls "$DEPLOY_DIR"/backups/hourly/*.db.gz >/dev/null 2>&1; then
    echo "ERROR: database $DB_PATH is missing but backups exist in $DEPLOY_DIR/backups/hourly." >&2
    echo "       Restore the newest backup to that path before starting (see comment above)." >&2
    echo "       To start with a fresh empty database on purpose: RADAR_ALLOW_EMPTY_DB=1 $0" >&2
    [ "${RADAR_ALLOW_EMPTY_DB:-0}" = "1" ] || exit 1
fi


# Export NVIDIA lib paths so CuPy/spaCy can find CUDA 12 runtime libs
# bundled inside the PyTorch wheel (libcudart, libcublas, etc.).
# Derived, not hardcoded: this said python3.11 and the env moved to 3.12,
# so the directory test below failed and the whole block was skipped --
# LD_LIBRARY_PATH went unset entirely and CuPy/scispacy found libcudart
# only by luck. Ask the interpreter instead of spelling its version.
NVIDIA_PKG="$("$CONDA_ENV/bin/python" -c 'import sysconfig,os; print(os.path.join(sysconfig.get_paths()["purelib"], "nvidia"))')"
if [ -d "$NVIDIA_PKG" ]; then
    NVIDIA_LIBS=$(find "$NVIDIA_PKG" -maxdepth 2 -name "lib" -type d ! -path "*/~*" | tr '\n' ':')
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Loopback only. The public entry point is Caddy on :8080 (behind the
# Cloudflare tunnel), and that is also where the verified identity header
# gets set. Binding 0.0.0.0 would let anyone on the campus network reach
# the API directly and bypass that.
# ~/.local/lib/python3.12 holds 187 user-level packages that would
# otherwise shadow this env's. pip already mistook one for "already
# satisfied" once and left pydantic out, which only surfaced at runtime.
export PYTHONNOUSERSITE=1

exec "$CONDA_ENV/bin/python" -m uvicorn rag_lib.api.app:app --host 127.0.0.1 --port 8000
