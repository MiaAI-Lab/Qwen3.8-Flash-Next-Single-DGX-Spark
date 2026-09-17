#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# stop.sh — stop the single-Spark vLLM container and its memory watchdog.
#
# Stops gracefully by default: vLLM gets SIGTERM and a chance to unlink the
# POSIX shared-memory segments the PLE offload handshake allocates. The
# container runs with --ipc host, so anything it leaves behind leaks onto the
# host's /dev/shm and survives until reboot. Use --force to skip the wait.
#
# Touches logs/stopping while the stop is in progress so the supervisor waits
# instead of relaunching a container the human deliberately stopped. The flag
# carries a "manual" first line: the supervisor NEVER reclaims it (unlike a
# maintenance-window flag, which is reclaimed after STOPPING_MAX_AGE_S), so a
# manual stop stays down until you relaunch (start.sh, maintenance-
# relaunch.sh), remove logs/stopping, or reboot (which clears it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CONTAINER_NAME="${TP1_CONTAINER_NAME:-}"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    # shellcheck source=.env
    source "$SCRIPT_DIR/.env"
fi
if [[ -n "$_CONTAINER_NAME" ]]; then
    TP1_CONTAINER_NAME="$_CONTAINER_NAME"
fi
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"      # seconds before docker escalates to SIGKILL
# Validate before use: docker stop -t rejects a bad value with exit 125, which
# the `|| true` below would swallow — the unconditional forced removal would
# then SIGKILL the container while the output still reads "stopped", silently
# downgrading the graceful path (the exact failure the #13 shm fix exists for).
if ! [[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]]; then
    echo "STOP_TIMEOUT must be a non-negative integer (got: '$STOP_TIMEOUT')" >&2
    exit 1
fi

FORCE=false
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=true ;;
        -h|--help)  sed -n '4,9p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)          echo "unknown option: $arg (try --help)" >&2; exit 1 ;;
    esac
done

# Stop the watchdog first so it cannot race a slow, graceful shutdown and turn
# it into a kill. pkill -f never matches its own process; this script's command
# line does not contain the pattern either.
if pkill -f "memwatch.sh $CONTAINER_NAME" 2>/dev/null; then
    echo "watchdog stopped"
fi

# Signal the supervisor not to fight us: while this flag exists the
# supervisor holds off relaunching. The "manual" first line marks authorship —
# the supervisor never reclaims a manual flag (a deliberate stop stays down
# until the operator resumes it); start.sh / maintenance-relaunch.sh manage
# the flag's lifecycle around their own relaunches. A flag that ALREADY
# exists belongs to a maintenance window (or a concurrent stop): keep it —
# overwriting it with a manual marker would make an abandoned maintenance
# window un-reclaimable forever.
mkdir -p "$SCRIPT_DIR/logs"
if [[ ! -f "$SCRIPT_DIR/logs/stopping" ]]; then
    printf 'manual\n%s\n' "$(date -Is)" > "$SCRIPT_DIR/logs/stopping"
fi

if [[ -z "$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" ]]; then
    echo "$CONTAINER_NAME was not running"
else
    # docker rm below discards the container's log; keep it for the post-mortem,
    # next to the watchdog's, the way start.sh and memwatch.sh do.
    ARCHIVE_DIR="$SCRIPT_DIR/logs/archive"; TS=$(date '+%Y%m%dT%H%M%S')
    mkdir -p "$ARCHIVE_DIR"
    docker logs --tail 3000 "$CONTAINER_NAME" > "$ARCHIVE_DIR/${CONTAINER_NAME}-${TS}-container.log" 2>&1 || true
    [[ -s "$SCRIPT_DIR/logs/memwatch-${CONTAINER_NAME}.log" ]] \
        && cp -f "$SCRIPT_DIR/logs/memwatch-${CONTAINER_NAME}.log" "$ARCHIVE_DIR/${CONTAINER_NAME}-${TS}-memwatch.log"
    echo "archived logs to logs/archive/${CONTAINER_NAME}-${TS}-{container,memwatch}.log"
    # Keep newest 20 archive sets (same rule as start.sh and the supervisor).
    ls -1t "$ARCHIVE_DIR"/*-container.log 2>/dev/null | tail -n +21 | while read -r f; do
        _set="${f%-container.log}"
        rm -f "${_set}-container.log" "${_set}-memwatch.log" "${_set}-probe-latency.log" "${_set}-timeout.log" 2>/dev/null || true
    done
    if [[ "$FORCE" == false ]]; then
        echo "stopping $CONTAINER_NAME (SIGTERM, up to ${STOP_TIMEOUT}s)..."
        docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "stopped $CONTAINER_NAME"
fi

# Report, but never delete: other containers on this host also run --ipc host,
# so their segments live here too and are not ours to remove.
leaked=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) 2>/dev/null | wc -l)
if (( leaked > 0 )); then
    bytes=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -printf '%s\n' 2>/dev/null \
            | awk '{s+=$1} END {print s+0}')
    echo "note: $leaked multiprocessing segment(s) in /dev/shm ($((bytes/1048576)) MiB allocated)."
    echo "      Inspect with: ls -la /dev/shm"
    echo "      Only remove them once no vLLM/sglang container is running."
fi
