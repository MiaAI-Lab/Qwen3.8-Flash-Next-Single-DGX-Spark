#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# stop.sh — stop the single-Spark vLLM container and its memory watchdog.
#
# Stops gracefully by default: vLLM gets SIGTERM and a chance to unlink the
# POSIX shared-memory segments the PLE offload handshake allocates. The
# container runs with --ipc host, so anything it leaves behind leaks onto the
# host's /dev/shm and survives until reboot. Use --force to skip the wait.
set -euo pipefail

CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"      # seconds before docker escalates to SIGKILL

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

if [[ -z "$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" ]]; then
    echo "$CONTAINER_NAME was not running"
else
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
    echo "note: $leaked multiprocessing segment(s) in /dev/shm ($((bytes/1048576)) MiB)."
    echo "      Inspect with: ls -la /dev/shm"
    echo "      Only remove them once no vLLM/sglang container is running."
fi
