#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# health-probe.sh — stateless single-shot health probe for the supervisor.
# Exit 0 = healthy, 1 = unhealthy. The supervisor owns the consecutive-failure
# counter in its state file; this script keeps no state.
#
# Contents (review §4.2 / §5.9 / §6.3):
#   1. GET /health -> 200 required.
#   2. A REAL generation request (max_tokens 16), asserting HTTP 200, parseable
#      JSON, a finish_reason, and usage.completion_tokens > 0. max_tokens is a
#      ceiling, not a target — the empty-cell trap passes a naive check while
#      the model is still inside its thinking block, so the assertion is on
#      completion_tokens, never on the budget.
#
# Env: PORT, SERVED_MODEL_NAME (read from .env first).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$REPO_DIR/.env" ]]; then
    # shellcheck source=.env
    source "$REPO_DIR/.env"
fi
PORT="${PORT:-8888}"
MODEL="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
BASE="http://localhost:$PORT"
PROBE_LATENCY_LOG="$REPO_DIR/logs/probe-latency.log"

_code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null || echo "000")
if [[ "$_code" != "200" ]]; then
    echo "health: HTTP $_code (000 = connection failure, not warming up)" >&2
    exit 1
fi

T0=$(date +%s.%N)
OUT=$(curl -s -m 60 -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "{
  \"model\": \"$MODEL\", \"max_tokens\": 16, \"temperature\": 0,
  \"messages\": [{\"role\":\"user\",\"content\":\"Reply with the single word: ok.\"}]}" 2>/dev/null)
T1=$(date +%s.%N)
if [[ -z "$OUT" ]]; then
    echo "probe: empty response body" >&2
    exit 1
fi

OK=1
echo "$OUT" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    raise SystemExit(f"probe: unparseable JSON: {e}")
fr = d.get("choices", [{}])[0].get("finish_reason")
comp = d.get("usage", {}).get("completion_tokens")
if not fr:
    raise SystemExit("probe: no finish_reason")
if not comp or comp <= 0:
    raise SystemExit("probe: completion_tokens=0 (empty-cell trap)")
' || OK=0

elapsed=$(python3 -c "print(f'{$T1-$T0:.3f}')" 2>/dev/null || echo "0")
# Latency trend metric, pruned by the supervisor log rotation. Free.
printf '%s %s\n' "$(date '+%s')" "$elapsed" >> "$PROBE_LATENCY_LOG" 2>/dev/null || true

[[ "$OK" == "1" ]] && exit 0 || exit 1
