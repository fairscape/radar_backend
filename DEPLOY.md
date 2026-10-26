# Deploying RADAR on your own machine

RADAR is two repositories, deployed together from the same branch name:

- **radar_backend** (this one): FastAPI + SQLite, the scans, scoring and
  topic mapping. Listens on `127.0.0.1:8000`.
- **radar_frontend**: a static single-page app, built once into `dist/`.

A reverse proxy serves `dist/` and forwards `/api/*` to the backend, so
the browser talks to one origin. This guide covers those three pieces on
one Linux machine. It does not cover what the live site adds on top of
them -- the Cloudflare tunnel and Access sign-in, the R2 backups and the
cron entries in `deploy/` -- none of which RADAR needs to run.

## What you need

| | |
|---|---|
| OS | Linux (tested on Ubuntu 22.04) |
| Python | 3.12 (`researcher-profiles` needs >= 3.12; scispacy 0.6.2 has no 3.13 wheels) |
| Node | 18 or newer (tested on 22) |
| Ollama | running locally, with `ollama pull mxbai-embed-large` |
| Disk | about 15 GB: Python environment ~8 GB, models and UMLS data ~4 GB |
| GPU | optional. NVIDIA with CUDA 12.4 makes the reranker and PDF parsing faster; everything runs on CPU |
| Reverse proxy | Caddy 2 (example in `deploy/Caddyfile.local`) or nginx |

Ollama provides the `mxbai-embed-large` embeddings that map UMLS
concepts, PDF content and Claude-generated research topics to OpenAlex
topics. Vault chat can also use an Ollama chat model; it is optional.

**OpenAlex now asks for an API key.** Since 2026-10-14, paging through
results without one returns 401 and the daily allowance is small. RADAR
does not send a key yet, so scans on a fresh machine can come back with
few or no papers. This is a known open issue.

## 1. Backend

```sh
git clone -b release-2026-10 https://github.com/fairscape/radar_backend.git
cd radar_backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-py312.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e . --no-deps
```

`requirements-py312.txt` is the exact environment the live site runs.
Without an NVIDIA GPU, change its `torch==2.6.0+cu124` line to
`torch==2.6.0` and use `--extra-index-url https://download.pytorch.org/whl/cpu`.

Run every Python command below with `PYTHONNOUSERSITE=1` if your account
has packages under `~/.local/lib/python3.12`: an old torch there is
imported ahead of the environment's and breaks model loading.

### Configuration

Settings come from environment variables and `.env` in the directory
you start the backend from. **The repository tracks a `.env` holding the
live site's choices; replace it:**

```sh
cp .env.example .env
```

then set at least:

```sh
RADAR_DEFAULT_MAILTO=you@your.org      # OpenAlex contact address
RADAR_REQUIRE_AUTH=false               # single user, see "Users" below
RADAR_UMLS_ENABLED=true                # topics from UMLS concepts and PDF content
RADAR_UMLS_EMBEDDING_MODEL=mxbai-embed-large
RADAR_RERANKER_ENABLED=true            # MedCPT reranking of scan results
RADAR_RERANKER_DEVICE=cuda             # or cpu
```

Paths default to `data/` under the backend directory (`RADAR_DB_PATH`,
`RADAR_VAULT_DIR`, `RADAR_CHROMA_DIR`, `RADAR_UMLS_CACHE_DIR`). Keep the
database on a local disk: on NFS RADAR falls back to SQLite's slower
rollback journal, because WAL is not safe there. `.env.example` documents
every other key.

Models download on first use. To keep them off a small home directory:

```sh
export HF_HOME=/big/disk/radar-cache/huggingface    # SPECTER2, MedCPT
```

With `RADAR_UMLS_ENABLED=true` the backend loads the UMLS linker before
it opens its port, so **the first start downloads the UMLS knowledge
base (~2 GB) into `RADAR_UMLS_CACHE_DIR/datasets` and does not answer
until that is done** -- about seven minutes on a fresh machine. Later
starts still load the models and linker before answering: one to four
minutes depending on the machine's load. Watch the backend's output for
the progress bars.

### The OpenAlex topic index

Mapping UMLS concepts, PDF content and research topics onto OpenAlex
topics needs a prebuilt index of all ~4,500 topics in
`RADAR_UMLS_CACHE_DIR`:

```
data/umls_cache/openalex_topics.json
data/umls_cache/openalex_topic_embeddings.npz
```

Ask the RADAR maintainers for `radar-openalex-topic-index.tar.gz`
(17 MB) and unpack it in the backend directory:
`tar xzf radar-openalex-topic-index.tar.gz -C data`.
`scripts/build_topic_index.py` can rebuild them, but it pages through
OpenAlex (see the API key note above), and it must be run with
`--embedding-model mxbai-embed-large`: its default embedder produces
768-dimension vectors where queries use 1024, which silently breaks the
mapping. Without the index RADAR still runs; those topic suggestions are
just missing.

### Database and start

```sh
python -m cli.db init --db data/radar.db    # same path as RADAR_DB_PATH
python -m cli.serve                         # 127.0.0.1:8000
curl http://127.0.0.1:8000/api/health
```

The backend applies any pending migrations at every start, so upgrading
is `git pull`, then restart. Back the database up before an upgrade:

```sh
sqlite3 data/radar.db ".backup data/radar.db.bak-$(date +%Y%m%d)"
```

Keep it running with whatever you use for services (systemd, tmux,
supervisord). `deploy/start_backend.sh` and `deploy/radar.sh` are the
live site's versions, with its paths.

## 2. Frontend

```sh
git clone -b release-2026-10 https://github.com/fairscape/radar_frontend.git
cd radar_frontend
npm ci
npm run build          # -> dist/
```

The production build calls `/api/...` on its own origin
(`.env.production` leaves `VITE_API_BASE_URL` empty), so it needs the
proxy below; nothing about the backend's address is baked in.

## 3. Reverse proxy

With Caddy, edit the `root` line of `deploy/Caddyfile.local` to point at
`radar_frontend/dist`, then:

```sh
caddy run --config deploy/Caddyfile.local --adapter caddyfile
```

and open <http://127.0.0.1:8080>. With nginx the same thing is:

```nginx
server {
    listen 127.0.0.1:8080;
    root /path/to/radar_frontend/dist;
    client_max_body_size 200m;                  # PDF uploads
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header X-User-Email "";      # see "Users"
        proxy_read_timeout 600s;                # trial scans run for minutes
        proxy_send_timeout 600s;
    }
    location / { try_files $uri /index.html; }
}
```

Both bind to loopback. To reach RADAR from other machines, change the
bind address deliberately and put TLS in front.

For a quick look without a proxy, `npm run dev` in `radar_frontend`
serves on <http://localhost:5173> and forwards `/api` to
`localhost:8000`.

## Users

The backend takes the user's identity from the `X-User-Email` request
header and checks nothing behind it.

- **One user** (the default here): `RADAR_REQUIRE_AUTH=false` and a proxy
  that removes any `X-User-Email` a client sends. Every request is the
  built-in default user.
- **Several users**: put a proxy that signs people in (OAuth2 Proxy,
  your SSO, Cloudflare Access as the live site does) in front, have it
  set `X-User-Email` to the signed-in address, set
  `RADAR_REQUIRE_AUTH=true`, and make sure nothing but that proxy can
  reach port 8000 or 8080. Anyone who can reach them directly can claim
  any address -- including other accounts on a shared machine, since
  loopback is open to every local user.

## Check that it works

1. `curl http://127.0.0.1:8080/api/health` answers with the version.
2. In single-user mode you are the built-in `demo@example.com`, whose
   OpenAlex contact address is that too (`RADAR_DEFAULT_MAILTO` only
   applies to users created later). Set your own under **Settings ->
   Contact email for OpenAlex** before the first scan.
3. In the browser, **New interest**: add a few seed papers (upload PDFs,
   pick works from an ORCID, or import a Prosopia profile by its URL),
   let step 3 suggest topics, run the trial scan in step 4, save.
   Concept extraction in step 3 took about a minute per PDF here.
4. The feed fills once the first scan finishes (**Scans** shows it).

## Known issues in this release

- OpenAlex API key not supported yet (above).
- Vault chat scoped to one interest misses papers that belong to several
  interests (`tests/test_rag_retriever.py`).
- The test suite has 8 known failures; run it with
  `PYTHONNOUSERSITE=1 python -m pytest tests`.
