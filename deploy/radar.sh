#!/bin/bash
# RADAR service manager — the four processes behind https://radar.bioxplorer.org
#
#   ./radar.sh start      bring everything up (idempotent)
#   ./radar.sh stop       take everything down
#   ./radar.sh restart    stop then start
#   ./radar.sh status     what's up, what's down
#   ./radar.sh watchdog   restart only what's down; quiet when all is well
#   ./radar.sh logs       tail every service log at once
#
# The stack, in dependency order:
#
#   ollama      127.0.0.1:11434  embeddings for UMLS concepts (GPU 0)
#   backend     127.0.0.1:8000   FastAPI, conda env, runs inside tmux
#   caddy       :8080            serves the SPA, proxies /api, sets identity
#   cloudflared                  publishes :8080 as radar.bioxplorer.org
#
# Nothing here needs root. `watchdog` is what cron calls every 5 minutes;
# `start` is what cron calls at @reboot. Both are safe to run by hand at
# any time — every start_* is a no-op when its service already answers.

set -uo pipefail

# cron gives us almost no PATH, and we need tmux, curl, and pgrep.
export PATH=/usr/local/bin:/usr/bin:/bin:${PATH:-}

DEPLOY=/bigtemp/nkw3mr/radar_deployment
SRC=/p/realai/lei/radar_deployment
CLOUDFLARED=/p/realai/BioXplorer/LLama-BioXplorer/cloudflared
LOGS="$DEPLOY/logs"
SESSION=radar-backend
DB_PATH=/var/tmp/radar-nkw3mr/radar.db

mkdir -p "$LOGS"

# --- health probes ---------------------------------------------------------
# Probe the service, not the process: a backend that is alive but wedged
# should count as down so the watchdog replaces it.

ollama_up() { curl -sf -m 5 -o /dev/null http://127.0.0.1:11434/ 2>/dev/null; }
backend_up() { curl -sf -m 5 -o /dev/null http://127.0.0.1:8000/api/health 2>/dev/null; }
# Probed through /api/health, not /: the SPA build lives on /p/realai, which
# has hit 100% full, and a missing or truncated dist made this 404 -- so a
# perfectly healthy Caddy read as DOWN, the watchdog restarted it every five
# minutes, and the operator was pointed at the wrong service. This also
# covers the /api proxy, which nothing else checked.
caddy_up() { curl -sf -m 5 -o /dev/null http://127.0.0.1:8080/api/health 2>/dev/null; }

# The tunnel has no local endpoint to probe, so this is a liveness check.
# The pattern lives in this file rather than on a command line, so it
# can't match the shell that is running it.
tunnel_up() { pgrep -f "cloudflared tunnel --config" >/dev/null 2>&1; }

# The database moved to this machine's local disk (fast, WAL-safe) and is
# therefore only as durable as the backups. Nothing here can repair a stale
# backup by restarting a process, but silence about one would mean the data
# is quietly unprotected, so both status and watchdog report it.
backup_age_min() {
	local newest
	newest=$(ls -1t "$DEPLOY/backups/hourly"/radar-*.db.gz 2>/dev/null | head -1)
	[ -n "$newest" ] || return 1
	echo $((($(date +%s) - $(stat -c %Y "$newest")) / 60))
}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# --- start -----------------------------------------------------------------

start_ollama() {
	ollama_up && return 0
	# Replace, do not merely start. Without this a wedged ollama still
	# holding :11434 made every tick spawn a second `ollama serve` that
	# could not bind and exited -- noise every five minutes while UMLS
	# embeddings stayed dead. Same reasoning as the backend's kill-session.
	pkill -f "[o]llama serve" 2>/dev/null && sleep 2
	log "starting ollama"
	nohup "$DEPLOY/start_ollama.sh" >>"$LOGS/ollama.log" 2>&1 &
	for _ in $(seq 1 20); do
		ollama_up && return 0
		sleep 1
	done
	log "ollama did not come up — see $LOGS/ollama.log"
	return 1
}

start_backend() {
	backend_up && return 0
	log "starting backend (SciSpacy UMLS index load takes ~4 min)"
	tmux kill-session -t "$SESSION" 2>/dev/null
	tmux new-session -d -s "$SESSION" "$SRC/scripts/background_setup/start_backend.sh" || {
		log "tmux refused to start the backend session"
		return 1
	}
	for _ in $(seq 1 90); do
		backend_up && {
			log "backend ready"
			return 0
		}
		sleep 5
	done
	log "backend did not come up — see: tmux attach -t $SESSION"
	return 1
}

start_caddy() {
	caddy_up && return 0
	# See start_ollama: `caddy start` against an already-listening instance
	# just fails to bind. Matched on the binary rather than the command line
	# because a pattern that appears in this script's own argv matches the
	# shell running it -- that once killed the shell mid-deploy.
	for proc in /proc/[0-9]*; do
		case "$(readlink "$proc/exe" 2>/dev/null)" in
		*caddy) kill "$(basename "$proc")" 2>/dev/null ;;
		esac
	done
	sleep 2
	log "starting caddy"
	(cd "$DEPLOY" && "$DEPLOY/bin/caddy" start --config Caddyfile >>"$LOGS/caddy-stdout.log" 2>&1)
	for _ in $(seq 1 15); do
		caddy_up && return 0
		sleep 1
	done
	log "caddy did not come up — see $LOGS/caddy-stdout.log"
	return 1
}

start_tunnel() {
	# Not `tunnel_up && return 0` on its own: do_stop's pkill is
	# asynchronous and cloudflared drains its connections, so on restart the
	# dying process was often still visible here and this returned success
	# without starting anything -- the site stayed down until the next tick.
	if tunnel_up; then
		local waited=0
		while tunnel_up && [ "$waited" -lt 15 ]; do sleep 1; waited=$((waited + 1)); done
		tunnel_up && return 0
	fi
	log "starting cloudflared tunnel"
	nohup "$CLOUDFLARED" tunnel --config "$DEPLOY/cloudflared-config.yml" run \
		>>"$LOGS/tunnel.log" 2>&1 &
	sleep 5
	tunnel_up || {
		log "tunnel did not stay up — see $LOGS/tunnel.log"
		return 1
	}
}

# The tunnel goes last: publishing the hostname before the backend answers
# would only serve 502s to whoever happens to load the page.
do_start() {
	start_ollama
	# The comment above promised the tunnel goes last so a not-yet-ready
	# backend never serves 502s publicly, but the failure was discarded
	# (`set -uo pipefail`, no -e) and the hostname got published anyway.
	if ! start_backend; then
		log "backend did not start -- not publishing the tunnel"
		do_status
		return 1
	fi
	start_caddy
	start_tunnel
	echo
	do_status
}

# --- stop ------------------------------------------------------------------

do_stop() {
	log "stopping cloudflared tunnel"
	pkill -f "cloudflared tunnel --config" 2>/dev/null

	log "stopping caddy"
	(cd "$DEPLOY" && "$DEPLOY/bin/caddy" stop 2>/dev/null)

	log "stopping backend"
	tmux kill-session -t "$SESSION" 2>/dev/null

	log "stopping ollama"
	pkill -f "ollama serve" 2>/dev/null

	echo "stopped"
}

# --- status ----------------------------------------------------------------

do_status() {
	printf '%-12s %s\n' "ollama" "$(ollama_up && echo 'up   127.0.0.1:11434' || echo 'DOWN')"
	printf '%-12s %s\n' "backend" "$(backend_up && echo 'up   127.0.0.1:8000' || echo 'DOWN')"
	printf '%-12s %s\n' "caddy" "$(caddy_up && echo 'up   :8080' || echo 'DOWN')"
	printf '%-12s %s\n' "tunnel" "$(tunnel_up && echo 'up' || echo 'DOWN')"

	local code verdict
	code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' https://radar.bioxplorer.org/ 2>/dev/null)
	# 302 is the healthy answer from outside: Cloudflare Access redirecting
	# an unauthenticated request to the login page. A 200 is NOT better --
	# it means Access is no longer in front of the origin, and since the
	# backend trusts X-User-Email with nothing behind it, that is the whole
	# trust model gone. Printing the bare number made the worst state look
	# like the best one.
	case "$code" in
	302) verdict="(Access is enforcing)" ;;
	200) verdict="** Access is NOT enforcing -- anyone can reach the origin **" ;;
	"")  verdict="(no answer)" ;;
	*)   verdict="(unexpected)" ;;
	esac
	printf '%-12s %s\n' "public" "https://radar.bioxplorer.org -> ${code:-no answer} $verdict"

	# The identity rewrite is the only thing standing between the network and
	# every account, and it has failed silently before: the listener was
	# bound to every interface, so a client could send the Cf-Access header
	# itself and be whoever it liked. Assert it rather than assume it.
	local ident
	ident=$(curl -s -m 10 -o /dev/null -w '%{http_code}' \
		-H "Cf-Access-Authenticated-User-Email: probe@invalid.test" \
		"http://127.0.0.1:8080/api/users/me" 2>/dev/null)
	local bound
	bound=$(ss -lntH 2>/dev/null | awk '$4 ~ /:8080$/ {print $4}' | head -1)
	case "$bound" in
	127.0.0.1:*) printf '%-12s %s\n' "auth" "proxy bound to $bound, identity header injected (probe -> $ident)" ;;
	"")          printf '%-12s %s\n' "auth" "** :8080 has no listener **" ;;
	*)           printf '%-12s %s\n' "auth" "** proxy listening on $bound -- reachable off-host, identity can be forged **" ;;
	esac

	local age
	if age=$(backup_age_min); then
		printf '%-12s %s\n' "backup" "$age 分钟前  ($DB_PATH)"
	else
		printf '%-12s %s\n' "backup" "NONE  ($DB_PATH 没有任何备份)"
	fi
}

# --- watchdog --------------------------------------------------------------
# Cron runs this every 5 minutes. It prints nothing when everything is
# healthy, so the log only ever contains real events.

do_watchdog() {
	local acted=0
	ollama_up || { start_ollama; acted=1; }
	backend_up || { start_backend; acted=1; }
	caddy_up || { start_caddy; acted=1; }
	tunnel_up || { start_tunnel; acted=1; }
	local age
	if age=$(backup_age_min); then
		[ "$age" -gt 180 ] && log "WARN: 最近一次数据库备份是 $age 分钟前 -- 见 backup.log"
	else
		log "WARN: 一份数据库备份都没有 -- 见 backup.log"
	fi

	[ "$acted" = 1 ] && log "watchdog finished repairs"
	return 0
}

# One at a time. start_backend polls for up to 90x5s = 450s while cron
# invokes watchdog every 300s, so a slow start used to be killed by the next
# tick -- which then started another, indefinitely, turning a slow start
# into a permanent outage. @reboot start racing the first tick is the same
# collision. status and logs are read-only and do not need the lock.
case "${1:-status}" in
start | stop | restart | watchdog)
	exec 9>"$DEPLOY/.radar.lock"
	if ! flock -n 9; then
		# Not an error: the other holder is doing the work.
		[ "${1:-}" = "watchdog" ] || log "another radar.sh is running; skipping"
		exit 0
	fi
	;;
esac

case "${1:-status}" in
start) do_start ;;
stop) do_stop ;;
restart)
	do_stop
	sleep 3
	do_start
	;;
status) do_status ;;
watchdog) do_watchdog ;;
logs)
	tail -n 20 -F "$LOGS/ollama.log" "$LOGS/caddy-stdout.log" \
		"$LOGS/tunnel.log" "$LOGS/watchdog.log" 2>/dev/null
	;;
*)
	echo "usage: $0 {start|stop|restart|status|watchdog|logs}" >&2
	exit 2
	;;
esac
