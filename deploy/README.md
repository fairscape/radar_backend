# deploy/ — the operational layer, under version control at last

These are the files that actually run the service. They lived outside both
repositories until 2026-09-28, which meant the only copy of the deployment
was on one machine's filesystem: a `chmod`, an edit or a purge would have
left nothing to compare against and nothing to restore from. A review of
this layer then found a live authentication bypass, a backup path that
could delete every good snapshot, and a restore path that corrupted what it
restored — none of which had any history to explain when or why they got
that way.

## Layout

The canonical copy is here. The paths the service and cron use are
symlinks into this directory, so the absolute paths baked into the crontab
keep working while git tracks the content:

    /p/realai/lei/radar_deployment/radar.sh                         -> deploy/radar.sh
    /p/realai/lei/radar_deployment/backup.sh                        -> deploy/backup.sh
    /p/realai/lei/radar_deployment/scripts/background_setup/start_backend.sh
                                                                    -> deploy/start_backend.sh
    /bigtemp/nkw3mr/radar_deployment/Caddyfile                      -> deploy/Caddyfile
    /bigtemp/nkw3mr/radar_deployment/start_ollama.sh                -> deploy/start_ollama.sh
    /bigtemp/nkw3mr/radar_deployment/cloudflared-config.yml         -> deploy/cloudflared-config.yml

Editing either path edits the same file. `radar.cron` is a copy rather than
a symlink, because `crontab` reads it once at install time
(`crontab deploy/radar.cron`) and keeps its own copy.

## What is here, and what runs it

| file | run by |
|---|---|
| `radar.sh` | cron, every 5 minutes (`watchdog`) and at `@reboot` (`start`) |
| `backup.sh` | cron, hourly at :07 |
| `start_backend.sh` | `radar.sh start_backend`, inside a tmux session |
| `start_ollama.sh` | `radar.sh start_ollama`, via nohup |
| `Caddyfile` | the long-running `caddy run` |
| `cloudflared-config.yml` | the long-running `cloudflared tunnel run` |

## Not here, deliberately

**`scripts/background_setup/.env`** — the backend's runtime config. It
holds this machine's paths (`/var/tmp/radar-nkw3mr/radar.db`, complete with
a username), the OpenAlex polite-pool address, and the GPU device. The
repository's `.gitignore` excludes `.env` by policy and `.env.example`
documents every key; `env.example` beside this file records what the
*deployment* sets differently from that default, which is the part a
reader cannot reconstruct.

**`~/.cloudflared/<uuid>.json`** — the tunnel's credentials (mode 400).
`cloudflared-config.yml` names the path but never the secret. The tunnel
UUID itself is not one: it is public in DNS as `<uuid>.cfargotunnel.com`.

**`~/.config/rclone/rclone.conf`** — the R2 access key and secret
(mode 600).

## The one thing a reader should know before changing Caddyfile

The backend trusts the `X-User-Email` request header as identity with
nothing signed behind it. Caddy overwriting that header with
`Cf-Access-Authenticated-User-Email` is the *entire* authentication
mechanism, and `bind 127.0.0.1` is what makes it trustworthy: a client that
can reach the port can send the Cf-Access header itself. It was bound to
every interface until 2026-09-28, and anyone on the campus network could be
anybody. `radar.sh status` now asserts the bind address for that reason.
