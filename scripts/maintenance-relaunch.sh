#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# maintenance-relaunch.sh — scheduled graceful relaunch (review §4.3, plan 1.4).
# Weekly via qwen38-flash-maintenance.timer (Sun 04:00):
#   1. Touch logs/stopping — the supervisor's signal: do NOT fight maintenance.
#   2. Drain in-flight requests: poll vllm:num_requests_running on /metrics
#      until 0, up to MAINT_DRAIN_S (default 600) — then proceed regardless;
#      requests get stop.sh's SIGTERM behavior.
#   3. stop.sh -> start.sh -> smoke-test.sh, alerting on any failure.
#   4. Remove logs/stopping ONLY after the new server is healthy. A crash
#      between stop and healthy leaves the flag in place — correct: the
#      supervisor must not fight the maintenance window while the alert goes
#      out. If the supervisor is down, this wrapper does the full relaunch
#      itself; the next supervisor tick adopts the running server.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

if [[ -f .env ]]; then
    # shellcheck source=.env
    source .env
fi
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
PORT="${PORT:-8888}"
MAINT_DRAIN_S="${MAINT_DRAIN_S:-600}"
STOPPING_FLAG="$REPO_DIR/logs/stopping"
BASE="http://localhost:$PORT"

alert() { "$REPO_DIR/scripts/alert.sh" "$*" || true; }
log()   { echo "$(date '+%F %T') [maintenance] $*"; }

mkdir -p "$REPO_DIR/logs"
touch "$STOPPING_FLAG"
log "maintenance window opened (touched $STOPPING_FLAG)"

# Drain. If the server is already down, skip the drain loop quickly.
_drained=0
_drain_start=$(date +%s)
while true; do
    _r=$(curl -s -m 5 "$BASE/metrics" 2>/dev/null | grep -oE '^vllm:num_requests_running [0-9]+' | tail -1 | awk '{print $2}' || echo 0)
    _r="${_r:-0}"
    if [[ -z "$_r" || "$_r" == "0" ]]; then
        _drained=1
        break
    fi
    if (( $(date +%s) - _drain_start >= MAINT_DRAIN_S )); then
        log "drain timeout after ${MAINT_DRAIN_S}s (${_r} still running); proceeding (SIGTERM path)"
        break
    fi
    log "draining: $_r requests running"
    sleep 10
done

"$REPO_DIR/stop.sh" >"$REPO_DIR/logs/maintenance-stop.log" 2>&1 || true
log "stop complete"

if "$REPO_DIR/start.sh" >"$REPO_DIR/logs/maintenance-start.log" 2>&1; then
    log "relaunch complete; running smoke test"
    if "$REPO_DIR/scripts/smoke-test.sh" >"$REPO_DIR/logs/maintenance-smoke.log" 2>&1; then
        log "smoke test PASS — closing maintenance window"
        rm -f "$STOPPING_FLAG"
    else
        alert "MAINTENANCE: relaunch healthy but smoke test FAILED (see $REPO_DIR/logs/maintenance-smoke.log). Stopping flag left in place."
        exit 1
    fi
else
    alert "MAINTENANCE: relaunch FAILED (see $REPO_DIR/logs/maintenance-start.log). Stopping flag left in place; supervisor holds off."
    exit 1
fi
