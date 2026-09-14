#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# start-memwatch.sh <container> <min_avail_gib> — start files/memwatch.sh with
# exactly the args/env the launcher uses, so start.sh and supervise.sh cannot
# drift. Runs memwatch in the background (nohup) and prints the log path.
#
# Env honoured (passed through to memwatch): MEMWATCH_MIN_FREE_GIB,
# MEMWATCH_FREE_GATE_GIB, MEMWATCH_GRACE. Defaults match memwatch.sh's own.
set -euo pipefail

CONTAINER="${1:?container}"
MIN_GIB="${2:-6}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
GRACE="${MEMWATCH_GRACE:-30}"
MEMWATCH_LOG="${MEMWATCH_LOG:-$REPO_DIR/logs/memwatch-${CONTAINER}.log}"

mkdir -p "$REPO_DIR/logs/archive"
# Anchor to the memwatch binary path so we kill a running memwatch but never
# our own argv (which contains "start-memwatch.sh <container>"); the [f] class
# stops the pattern from matching the very pkill/-f command line we run.
pkill -f "[f]iles/memwatch.sh $CONTAINER" 2>/dev/null || true

MEMWATCH_MIN_FREE_GIB="$MIN_FREE_GIB" MEMWATCH_FREE_GATE_GIB="$FREE_GATE_GIB" \
    MEMWATCH_GRACE="$GRACE" MEMWATCH_LOG="$MEMWATCH_LOG" \
    nohup bash "$REPO_DIR/files/memwatch.sh" "$CONTAINER" "$MIN_GIB" \
    > "$MEMWATCH_LOG" 2>&1 &
echo "$MEMWATCH_LOG"
