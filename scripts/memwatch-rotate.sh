#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# memwatch-rotate.sh <container> — rotate logs/memwatch-<container>.log when it
# exceeds 10 MB. memwatch holds the fd open, so in-place truncation does not
# shrink it and copy-truncate breaks nothing but frees space (review §4.5).
# Also prunes probe-latency.log and logs/archive/ (newest 20 sets).
set -uo pipefail

CONTAINER="${1:?container}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG="$REPO_DIR/logs/memwatch-${CONTAINER}.log"
MAX_BYTES="${MEMWATCH_LOG_MAX:-10485760}"

if [[ -f "$LOG" ]] && [[ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt "$MAX_BYTES" ]]; then
    TS=$(date '+%Y%m%dT%H%M%S')
    mkdir -p "$REPO_DIR/logs/archive"
    cp -f "$LOG" "$REPO_DIR/logs/archive/${CONTAINER}-${TS}-memwatch.log" 2>/dev/null || true
    : > "$LOG" 2>/dev/null || true
    echo "$(date '+%F %T') rotated memwatch log (> $MAX_BYTES bytes) to logs/archive/${CONTAINER}-${TS}-memwatch.log"
fi

# Prune probe-latency.log past the newest 20k lines (free trend metric). Only
# rewrite when it is actually over the cap — this runs every 10 s supervisor
# tick and the file grows one line/minute.
_MAXLINES="${PROBE_LATENCY_MAX_LINES:-20000}"
if [[ -f "$REPO_DIR/logs/probe-latency.log" ]] \
        && (( $(wc -l < "$REPO_DIR/logs/probe-latency.log" 2>/dev/null || echo 0) > _MAXLINES )); then
    tail -n "$_MAXLINES" "$REPO_DIR/logs/probe-latency.log" > "$REPO_DIR/logs/.probe-tmp" 2>/dev/null \
        && mv "$REPO_DIR/logs/.probe-tmp" "$REPO_DIR/logs/probe-latency.log"
fi

# Keep newest 20 archive sets (same rule as start.sh/stop.sh). A set whose
# container log is newer than MIN_SET_AGE_S is never pruned: start.sh/stop.sh
# may have just written it between their own archive and prune, and the 10 s
# tick must not race them. Sets are identified by their timestamp prefix,
# whether they carry a -container.log member (start/stop archives) or only a
# -memwatch.log member (this script's copy-truncate rotation).
_MIN_SET_AGE="${MIN_SET_AGE_S:-300}"
_now=$(date +%s)
# Unique set prefixes by stripping the known member suffixes, newest first
# (strict mtime ordering, not per-glob `ls` grouping).
_anchor="$REPO_DIR/logs/archive"
ls -1 "$_anchor"/*-container.log "$_anchor"/*-memwatch.log 2>/dev/null \
    | sed -E 's/-memwatch\.log$//; s/-container\.log$//' \
    | sort -u \
    | while read -r _prefix; do
        [[ -n "$_prefix" ]] || continue
        _ct=$(stat -c %Y "${_prefix}-container.log" 2>/dev/null || echo 0)
        _mt=$(stat -c %Y "${_prefix}-memwatch.log" 2>/dev/null || echo 0)
        if [[ "$_ct" == "0" && "$_mt" == "0" ]]; then
            continue
        fi
        _set_mtime=$_ct
        (( _mt > _set_mtime )) && _set_mtime=$_mt
        printf '%s %s\n' "$_set_mtime" "$_prefix"
    done \
    | sort -rn \
    | cut -d' ' -f2- \
    | tail -n +21 \
    | while read -r _prefix; do
        [[ -n "$_prefix" ]] || continue
        _mtime=$(stat -c %Y "${_prefix}-container.log" 2>/dev/null || stat -c %Y "${_prefix}-memwatch.log" 2>/dev/null || echo 0)
        (( _now - _mtime < _MIN_SET_AGE )) && continue
        rm -f "${_prefix}-container.log" "${_prefix}-memwatch.log" \
              "${_prefix}-probe-latency.log" "${_prefix}-timeout.log" 2>/dev/null || true
    done

exit 0
