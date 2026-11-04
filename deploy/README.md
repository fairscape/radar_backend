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

**`$DEPLOY/bin/`** (`/bigtemp/nkw3mr/radar_deployment/bin`) — the `caddy`,
`rclone` and `cloudflared` binaries `radar.sh` and `backup.sh` run. They
are plain upstream releases, kept out of git for their size; to rebuild,
download the same versions (`<binary> --version`) into that directory.

## The one thing a reader should know before changing Caddyfile

Identity is the session cookie the backend's `/auth` sign-in (Google,
Prosopia) sets; `X-User-Email` is ignored. Until 2026-11 it was the other
way round: Caddy copied Cloudflare Access's
`Cf-Access-Authenticated-User-Email` into `X-User-Email` and the backend
trusted it. With Access removed nothing vouches for that header, so the
copy line is gone and Caddy now *drops* both headers. Never put it back:
a client can send the Cf-Access header itself.

Two checks hold this up:
- the backend refuses to start with `RADAR_REQUIRE_AUTH=true` and no
  sign-in provider unless `RADAR_TRUST_EMAIL_HEADER=true` (the live `.env`
  must not set it), so a missing client secret stops the site instead of
  reopening the header;
- `radar.sh status` sends a forged header through :8080 and expects 401.

`bind 127.0.0.1` still matters: the backend's own port and Caddy's admin
socket stay off the network.

## When the data shrinks: pinned copies

`backup.sh` does not try to tell an accident from a deliberate cleanup.
When the data shrinks sharply it keeps backing up as usual and keeps the
last copy from before the shrink for good:

- **Database** (L14): a snapshot with fewer than a tenth of the previous
  one's papers+profiles rows, or none at all after a non-empty one. The
  previous snapshot is copied to `backups/pinned/` (and `pinned/` on R2),
  which the rolling windows (48 hourly, 60 daily) never prune.
- **Vault** (CR.9): fewer than half the files R2 holds. The sync runs; what
  it removes goes to `vault-pinned/<day>/` on R2, never pruned. (A normal
  sync moves removed or replaced files to `vault-deleted/<day>/`, kept 60
  days.)

Each pin is a line in `backups/pinned.log` and a `WARN` in `backup.log`;
`radar.sh watchdog` reports each new one once, and `radar.sh status` shows
the latest. Nothing needs to be done to keep backups running.

When you see one: if the shrink was an accident, restore from the pinned
copy (`backup.sh restore backups/pinned/<file>`). If it was deliberate,
delete the pinned copy when you no longer want it, here and on R2.

The first vault sync after the 2026-11-02 switch to an empty database will
pin: R2's `vault/` still holds the files of the vault from before it. They
end up in `vault-pinned/`.
