#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# heartbeat.sh — daily unconditional heartbeat (review §4.6): uptime, restart
# count this week (reads supervisor state), MemAvailable, disk free on the
# checkpoint volume. Unconditional by design: a heartbeat that only fires on
# failure is worse than none, because silence reads as "running".
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$REPO_DIR/.env" ]]; then
    # shellcheck source=.env
    source "$REPO_DIR/.env"
fi
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"

_state=""
if [[ -f "$REPO_DIR/logs/supervisor.state" ]]; then
    _state=$(tr '\n' ' ' < "$REPO_DIR/logs/supervisor.state" 2>/dev/null || echo "")
fi
_uptime=$(uptime -p 2>/dev/null || uptime)
_mem=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{printf "%.1f GiB", $2/1048576}' || echo "?")
_disk=$(df -h "$HF_CACHE_DIR" 2>/dev/null | tail -1 | awk '{print $4 " free of " $2 " (" $5 " used)"}' || echo "?")

MSG="HEARTBEAT: up $_uptime; MemAvailable $_mem; checkpoint volume: $_disk; supervisor state: ${_state:-none}; container: $CONTAINER_NAME"
"$REPO_DIR/scripts/alert.sh" "$MSG" || true
