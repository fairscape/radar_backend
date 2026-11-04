#!/bin/bash
# RADAR backup.
#
#   ./backup.sh            take a snapshot; push to R2 if it is configured
#   ./backup.sh status     what exists locally and remotely, and how old
#   ./backup.sh restore <file>   put a snapshot back (stops the backend)
#
# The working database lives on this machine's local disk, because the GPU
# does and the query path cannot afford a network round trip. That disk is
# not durable: the machine can be reimaged, and /var/tmp is nobody's idea
# of permanent storage -- so the copy that has to survive lives in Cloudflare R2.
# Local snapshots are for fast rollback; R2 is the authoritative one.
#
# Never `cp` a live SQLite file: you get a torn image that only announces
# itself as "database disk image is malformed" the next time something
# reads it. `.backup` uses SQLite's online backup API, which is safe while
# the backend is writing. Every snapshot is then integrity-checked before
# it is allowed to count as a backup.

set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin:${PATH:-}

# Every path below may be overridden from the environment (BACKUP_DB,
# BACKUP_VAULT, ...): tests/test_cr_backup.py runs this script against a
# scratch database and a local directory standing in for R2. cron sets
# none of them.
DB=${BACKUP_DB:-/var/tmp/radar-nkw3mr/radar.db}
VAULT=${BACKUP_VAULT:-/bigtemp/nkw3mr/radar_deployment/vault}
DEPLOY=/bigtemp/nkw3mr/radar_deployment
LOCAL=${BACKUP_LOCAL:-"$DEPLOY/backups"}
# The env the backend actually runs from. Pointed at conda_env while the
# service moved to conda_env312, so deleting the orphaned env would have
# silently broken every snapshot and the restore path's integrity check,
# announced only by a WARN three hours later. Version skew matters too: the
# backup CLI must not be older than the library that writes the file.
SQ=${BACKUP_SQLITE3:-"$DEPLOY/conda_env312/bin/sqlite3"}
RCLONE=${BACKUP_RCLONE:-"$DEPLOY/bin/rclone"}
# The bucket is named for the org, not this app, so everything RADAR
# writes lives under one prefix and other things can share the bucket.
REMOTE=${BACKUP_REMOTE:-r2:bioxplorer/radar}

KEEP_HOURLY=48   # ~2 days of hourly rollback points
KEEP_DAILY=60    # ~2 months
KEEP_VAULT_DELETED=60   # days a PDF the sync removed or replaced stays in vault-deleted/ (CR.9)

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# `du` on the /bigtemp NFS mount reports allocation units, not bytes, and
# called a 3 MB snapshot "512". Everything here sizes from stat instead.
human() { awk -v b="${1:-0}" 'BEGIN{ if(b>=1048576) printf "%.1f MB", b/1048576; else if(b>=1024) printf "%.0f KB", b/1024; else printf "%d B", b }'; }
dirbytes() { find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }

# R2 is optional: the script is installed before the bucket exists, and it
# must keep making local snapshots until then rather than failing.
r2_ready() {
	# A test's local directory "remote" needs no config.
	[ -n "${BACKUP_REMOTE:-}" ] && return 0
	local cfg="$HOME/.config/rclone/rclone.conf"
	[ -x "$RCLONE" ] && [ -f "$cfg" ] || return 1
	# The config ships with <placeholders> before the token exists. Treating
	# that as configured would fail an upload every hour and train whoever
	# reads backup.log to ignore it. Match only the three value lines: the
	# comments in that file talk about the placeholders, and a looser
	# pattern reads its own documentation as an unconfigured remote.
	grep -qE '^[[:space:]]*(access_key_id|secret_access_key|endpoint)[[:space:]]*=.*<' "$cfg" && return 1
	"$RCLONE" listremotes 2>/dev/null | grep -q '^r2:'
}

# Keep $2 for good: copy it to pinned/ (never pruned; backed up to R2 by the
# next push) and say so in backup.log and pinned.log, which radar.sh
# watchdog reports once. $1 names what shrank, $3 says by how much.
pin() {
	local what=$1 file=$2 why=$3
	mkdir -p "$LOCAL/pinned"
	cp -n "$file" "$LOCAL/pinned/" || { log "FAILED: could not pin $(basename "$file")"; return 1; }
	local line="$(date '+%Y-%m-%d %H:%M:%S') $what shrank: $why -- kept $(basename "$file") in pinned/"
	echo "$line" >>"$LOCAL/pinned.log"
	log "WARN: $line"
}

do_backup() {
	mkdir -p "$LOCAL/hourly" "$LOCAL/daily"
	local stamp day tmp out
	stamp=$(date +%Y%m%d-%H%M)
	day=$(date +%Y%m%d)
	tmp=$(mktemp "$LOCAL/.snap-XXXXXX.db")

	# The database must already exist. `sqlite3 <missing> .backup` CREATES an
	# empty database and exits 0, and an empty database passes
	# integrity_check -- so a vanished /var/tmp/radar-nkw3mr (which has
	# happened) would produce 48 valid-looking 4 KB snapshots, the retention
	# rules below would delete every real one locally and on R2, and
	# `radar.sh status` would report a fresh backup the whole way down.
	if [ ! -f "$DB" ]; then
		log "FAILED: $DB does not exist -- refusing to write a snapshot of nothing"
		rm -f "$tmp"
		return 1
	fi

	"$SQ" "$DB" ".backup '$tmp'" || {
		log "FAILED: .backup did not complete"
		rm -f "$tmp"
		return 1
	}

	# An unverified backup is not a backup. Refuse to keep a corrupt one --
	# silently storing it would overwrite good snapshots with garbage as
	# the retention window rolls forward.
	local check
	check=$("$SQ" "$tmp" "PRAGMA integrity_check;" 2>&1 | head -1)
	if [ "$check" != "ok" ]; then
		log "FAILED: snapshot is corrupt ($check) -- keeping previous backups, discarding this one"
		rm -f "$tmp"
		return 1
	fi

	# integrity_check says the file is well-formed, not that it is this
	# database: an empty or half-written one is well-formed too. Such a
	# snapshot, kept hourly, would push every good one out of the rolling
	# windows below (48 hourly, 60 daily) within days. A collapse in row
	# count is not proof of that -- deleting interests or switching to an
	# empty database shrink it too -- so it does not stop the backup
	# (the user's choice, 2026-11-02). Instead the last snapshot from before
	# the shrink is pinned: copied to pinned/, which nothing prunes, here and
	# on R2, and reported. An accident leaves the full database there to
	# restore; a deliberate shrink costs one file, deleted by hand.
	local rows prev_rows prev
	rows=$("$SQ" "$tmp" "SELECT (SELECT COUNT(*) FROM papers) + (SELECT COUNT(*) FROM profiles);" 2>/dev/null || echo 0)
	prev=$(ls -1t "$LOCAL/hourly"/radar-*.db.gz 2>/dev/null | head -1)
	if [ -n "$prev" ]; then
		# Decompressed to a real file, not piped: SQLite needs to seek, so
		# `gunzip -c | sqlite3 /dev/stdin` fails to open the database and
		# quietly yielded 0 -- which the `-gt 0` below then treated as "no
		# previous snapshot to compare against", disabling this guard
		# entirely. It read as working because nothing ever failed.
		local prev_tmp
		prev_tmp=$(mktemp "$LOCAL/.prev-XXXXXX.db")
		if gunzip -c "$prev" >"$prev_tmp" 2>/dev/null; then
			prev_rows=$("$SQ" "$prev_tmp" \
				"SELECT (SELECT COUNT(*) FROM papers) + (SELECT COUNT(*) FROM profiles);" 2>/dev/null || echo 0)
		else
			prev_rows=0
		fi
		rm -f "$prev_tmp"
		# Below a tenth, or empty after non-empty (L14). Multiplied, not
		# divided: with a small baseline `prev_rows / 10` rounds to 0, and
		# 9 rows -> 0 rows went unnoticed.
		rows=${rows:-0}
		if [ "${prev_rows:-0}" -gt 0 ] && { [ "$rows" -eq 0 ] || [ $((rows * 10)) -lt "$prev_rows" ]; }; then
			pin "database" "$prev" "snapshot has $rows papers+profiles, the previous one had $prev_rows"
		fi
	fi

	out="$LOCAL/hourly/radar-$stamp.db.gz"
	# gzip's exit status was ignored, so a full disk left a truncated .gz
	# that then got promoted to daily, logged as a success, pushed to R2, and
	# kept while the good snapshots were pruned to make room for it.
	if ! gzip -c "$tmp" >"$out"; then
		log "FAILED: could not write $out (disk full?) -- discarding, keeping previous backups"
		rm -f "$tmp" "$out"
		return 1
	fi
	# And the integrity check ran on $tmp, before compression -- nothing had
	# ever decompressed what actually gets stored.
	if ! gzip -t "$out" 2>/dev/null; then
		log "FAILED: $out does not decompress -- discarding"
		rm -f "$tmp" "$out"
		return 1
	fi
	rm -f "$tmp"

	# One snapshot per day is promoted to the long-retention set.
	[ -f "$LOCAL/daily/radar-$day.db.gz" ] || cp "$out" "$LOCAL/daily/radar-$day.db.gz"

	log "snapshot $(basename "$out") ($(human "$(stat -c %s "$out")"))"

	# Prune locally: keep the newest N by name, which sorts chronologically
	# because the stamps are zero-padded.
	ls -1t "$LOCAL/hourly"/radar-*.db.gz 2>/dev/null | tail -n +$((KEEP_HOURLY + 1)) | xargs -r rm -f
	ls -1t "$LOCAL/daily"/radar-*.db.gz 2>/dev/null | tail -n +$((KEEP_DAILY + 1)) | xargs -r rm -f

	if r2_ready; then
		# Say where it went, not just that something happened. "No news is
		# good news" is the wrong default for a backup: reading this log
		# should answer "is the data off this machine?" without having to
		# go list the bucket.
		if "$RCLONE" copy "$out" "$REMOTE/hourly/" --no-traverse -q; then
			log "pushed $(basename "$out") -> $REMOTE/hourly/"
		else
			log "WARN: R2 push failed for $(basename "$out")"
		fi
		"$RCLONE" copy "$LOCAL/daily/radar-$day.db.gz" "$REMOTE/daily/" --no-traverse -q ||
			log "WARN: R2 push failed for daily"
		# Never pruned, here or there (see pin).
		if [ -d "$LOCAL/pinned" ]; then
			"$RCLONE" copy "$LOCAL/pinned/" "$REMOTE/pinned/" -q || log "WARN: R2 push failed for pinned/"
		fi
		# Mirror the local retention window into the bucket.
		"$RCLONE" delete "$REMOTE/hourly/" --min-age "$((KEEP_HOURLY))h" -q 2>/dev/null
		"$RCLONE" delete "$REMOTE/daily/" --min-age "$((KEEP_DAILY))d" -q 2>/dev/null

		# The PDFs are user uploads and equally unreproducible, but they
		# change rarely -- a daily incremental is enough, and `sync` only
		# moves what differs.
		if [ ! -f "$LOCAL/.vault-synced-$day" ]; then
			if vault_sync "$day"; then
				rm -f "$LOCAL"/.vault-synced-*
				touch "$LOCAL/.vault-synced-$day"
			fi
		fi
	else
		log "R2 not configured yet -- local snapshot only"
	fi
}

# The daily vault mirror (CR.9). `sync` deletes on the remote whatever is
# gone locally, so a vanished or emptied vault directory would have erased
# the only other copy of every upload by the next morning. So a file the
# sync removes or overwrites is moved aside instead (--backup-dir):
#  - normally to vault-deleted/<day>/, kept KEEP_VAULT_DELETED days;
#  - when the local vault has fewer than half the remote's files, to
#    vault-pinned/<day>/, which nothing prunes, and reported -- as the
#    database's pinned snapshot: the sync still runs (the user's choice,
#    2026-11-02), and nothing it removed is lost.
vault_sync() {
	local day=$1 here there keep_in
	here=$(find "$VAULT" -type f 2>/dev/null | wc -l)
	if ! there=$("$RCLONE" lsf -R --files-only "$REMOTE/vault/" 2>/dev/null | wc -l); then
		log "FAILED: vault sync: could not list $REMOTE/vault/ -- not syncing"
		return 1
	fi
	keep_in="vault-deleted/$day"
	if [ "$there" -gt 0 ] && [ $((here * 2)) -lt "$there" ]; then
		keep_in="vault-pinned/$day"
		local line="$(date '+%Y-%m-%d %H:%M:%S') vault shrank: $here files here, $there on R2 -- what the sync removes is kept in $keep_in/ on R2"
		mkdir -p "$LOCAL"
		echo "$line" >>"$LOCAL/pinned.log"
		log "WARN: $line"
	fi
	if ! "$RCLONE" sync "$VAULT" "$REMOTE/vault/" --backup-dir "$REMOTE/$keep_in" -q; then
		log "WARN: vault sync failed"
		return 1
	fi
	log "vault synced to R2 ($here files; removed or replaced ones under $keep_in)"
	prune_vault_deleted
	return 0
}

# By the day in the directory's name, not --min-age: a moved file keeps
# its own modification time, so a PDF uploaded months ago and deleted
# today would be pruned the same night.
prune_vault_deleted() {
	local cutoff d
	cutoff=$(date -d "-${KEEP_VAULT_DELETED} days" +%Y%m%d)
	"$RCLONE" lsf --dirs-only "$REMOTE/vault-deleted/" 2>/dev/null | while read -r d; do
		d=${d%/}
		[[ "$d" =~ ^[0-9]{8}$ ]] || continue
		if [ "$d" -lt "$cutoff" ]; then
			"$RCLONE" purge "$REMOTE/vault-deleted/$d" -q && log "pruned vault-deleted/$d"
		fi
	done
	# Nothing to list (no file was ever removed) is not a failure.
	return 0
}

do_status() {
	local newest age
	newest=$(ls -1t "$LOCAL/hourly"/radar-*.db.gz 2>/dev/null | head -1)
	if [ -n "$newest" ]; then
		age=$((($(date +%s) - $(stat -c %Y "$newest")) / 60))
		printf '%-14s %s (%d 分钟前)\n' "最近备份" "$(basename "$newest")" "$age"
	else
		printf '%-14s %s\n' "最近备份" "无"
	fi
	printf '%-14s %s 份 hourly, %s 份 daily, 共 %s\n' "本地" \
		"$(ls -1 "$LOCAL/hourly" 2>/dev/null | wc -l)" \
		"$(ls -1 "$LOCAL/daily" 2>/dev/null | wc -l)" \
		"$(human "$(dirbytes "$LOCAL")")"
	if r2_ready; then
		printf '%-14s %s 份 hourly, %s 份 daily\n' "R2" \
			"$("$RCLONE" lsf "$REMOTE/hourly/" 2>/dev/null | wc -l)" \
			"$("$RCLONE" lsf "$REMOTE/daily/" 2>/dev/null | wc -l)"
	else
		printf '%-14s %s\n' "R2" "未配置"
	fi
}

do_restore() {
	local src=${1:-} yes=${2:-}
	# The paths above can be pointed elsewhere (tests), but this function
	# also stops, pauses and restarts the live service, which they cannot
	# redirect: with any override set it would restore a scratch file and
	# still take the live backend down (review L5).
	if [ -n "${BACKUP_DB:-}${BACKUP_VAULT:-}${BACKUP_LOCAL:-}${BACKUP_REMOTE:-}${BACKUP_RCLONE:-}${BACKUP_SQLITE3:-}" ]; then
		echo "restore refuses to run with BACKUP_* overrides set: it acts on the live service" >&2
		return 2
	fi
	[ -f "$src" ] || { echo "用法: $0 restore <快照文件.db.gz> [--yes]" >&2; exit 2; }

	# Overwriting the live database is the most destructive thing here, and
	# an operator script has already been run against it by accident once.
	if [ "$yes" != "--yes" ]; then
		echo "This replaces $DB with $(basename "$src")." >&2
		echo "The current file is kept as $DB.replaced-<timestamp>." >&2
		printf 'Type the word restore to continue: ' >&2
		local answer; read -r answer
		[ "$answer" = "restore" ] || { echo "aborted" >&2; exit 1; }
	fi

	log "校验 $src"
	local tmp
	tmp=$(mktemp "$(dirname "$DB")/.restore-XXXXXX.db") || {
		log "无法在 $(dirname "$DB") 建临时文件 —— 目录还在吗？"; exit 1; }
	gunzip -c "$src" >"$tmp" || { rm -f "$tmp"; exit 1; }
	[ "$("$SQ" "$tmp" 'PRAGMA integrity_check;' | head -1)" = "ok" ] || {
		log "拒绝恢复：这份快照本身是坏的"; rm -f "$tmp"; exit 1; }

	# The cron watchdog restarts the backend every 5 minutes. A tick landing
	# between the kill below and the move would start uvicorn against a
	# database that is momentarily absent; with luck start_backend.sh's guard
	# catches it, and without luck it races the move.
	local had_watchdog=""
	if crontab -l 2>/dev/null | grep -q watchdog; then
		had_watchdog=1
		crontab -l > "$LOCAL/.crontab-restore-$$" 2>/dev/null
		crontab -l | grep -v watchdog | crontab -
		log "watchdog 已暂停"
	fi

	log "停后端"
	tmux kill-session -t =radar-backend 2>/dev/null
	sleep 2

	# The -wal and -shm sidecars must go with the file they belong to. The
	# database is in WAL mode, and an uncleanly killed backend leaves a valid
	# WAL behind; left next to the restored snapshot, SQLite replays those
	# frames into it on the next open -- pages from the OLD database written
	# over the one just restored. WAL frames are checksummed against the WAL
	# header's salts, not against a particular database file, so nothing
	# catches it. The recovery path would be manufacturing the exact
	# corruption class it is recovering from.
	local ts; ts=$(date +%Y%m%d-%H%M%S)
	for suffix in "" "-wal" "-shm"; do
		[ -e "$DB$suffix" ] && mv -f "$DB$suffix" "$DB$suffix.replaced-$ts"
	done
	mv "$tmp" "$DB"; chmod 600 "$DB"

	log "已恢复，重启后端"
	/p/realai/lei/radar_deployment/radar.sh start
	if [ -n "$had_watchdog" ]; then
		crontab "$LOCAL/.crontab-restore-$$" && rm -f "$LOCAL/.crontab-restore-$$"
		log "watchdog 已恢复"
	fi
}

# Sourced (tests), only the functions are defined.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0

case "${1:-backup}" in
backup) do_backup ;;
status) do_status ;;
restore) do_restore "${2:-}" "${3:-}" ;;
*) echo "用法: $0 {backup|status|restore <file> [--yes]}" >&2; exit 2 ;;
esac
