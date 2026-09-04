#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# memwatch.sh <container> [min_avail_gib] [consecutive_samples]
#
# Host-memory watchdog for the single-Spark deployment. On unified memory an
# exhausted pool hangs the kernel instead of raising an OOM, so this stops the
# container when the host runs out of margin. It is a second line of defence
# behind start.sh's HOST_RESERVE_GIB budget; a userspace poller cannot catch a
# GiB/s collapse alone.
#
# Two independent triggers, each debounced over CONSEC consecutive samples
# (a lone excursion is logged and resets the counter):
#   * MemAvailable < min_avail_gib      (arg 2, default 6)  -- page cache gone,
#     PLE lookups about to hit NVMe.
#   * MemFree < MEMWATCH_MIN_FREE_GIB   (env, default 2)    -- the NVIDIA driver
#     starts refusing allocations (NV_ERR_NO_MEMORY in `journalctl -k`) with
#     MemFree around 3 GiB while MemAvailable still reads 6+ GiB, so the
#     MemAvailable floor alone reacts late. Counted ONLY while MemAvailable is
#     also under MEMWATCH_FREE_GATE_GIB (default 10): with the stock kernel
#     watermarks MemFree legitimately falls to the ~170 MB low watermark
#     whenever the page cache is full of reclaimable data. Measured here
#     2026-09-05 00:49 during weight loading: MemFree 0.9 GiB, MemAvailable
#     32 GiB, zero NVRM errors. Free pages backed by reclaimable cache are not
#     what the driver runs out of; free pages with no cache left to reclaim are.
#
# Every 10 s it also counts NV_ERR_NO_MEMORY lines in the kernel log; that is
# the earliest signal this box gives and is logged whenever it is non-zero.
#
# Timeline every 5 s (every sample once within 1 GiB of either floor) with the
# /proc/meminfo fields needed to tell page cache from anon from driver memory:
#   driver = MemTotal - MemFree - Buffers - Cached - AnonPages - Slab
#            - PageTables - KernelStack
# i.e. memory that is neither free, page cache, anon, kernel slab nor page
# tables: taken through the NVIDIA driver (GPU allocations, pinned host
# buffers). A permanent step up in `driver` with `free` flat is a request
# growing the CUDA caching allocator; that memory does not come back.
#
# Before stopping the container it archives `docker logs --tail 3000` and a
# copy of its own log to logs/archive/, then `docker stop -t $MEMWATCH_GRACE`
# (SIGTERM; a SIGKILL leaks the container's POSIX shm onto the host's
# /dev/shm until reboot because of --ipc host), falling back to docker kill.
#
# Env: MEMWATCH_MIN_FREE_GIB (2), MEMWATCH_FREE_GATE_GIB (10), MEMWATCH_GRACE (30), MEMWATCH_LOG (this
# script's own log, for archiving; default logs/memwatch-<container>.log),
# MEMWATCH_ARCHIVE_DIR (logs/archive).
CONTAINER="${1:?container}"; MIN_GIB="${2:-6}"; CONSEC="${3:-5}"
MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
GRACE="${MEMWATCH_GRACE:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OWN_LOG="${MEMWATCH_LOG:-$REPO_DIR/logs/memwatch-${CONTAINER}.log}"
ARCHIVE_DIR="${MEMWATCH_ARCHIVE_DIR:-$REPO_DIR/logs/archive}"

MIN_KB=$(( MIN_GIB * 1048576 ))
MIN_FREE_KB=$(( MIN_FREE_GIB * 1048576 ))
FREE_GATE_KB=$(( FREE_GATE_GIB * 1048576 ))
NEAR_KB=$(( MIN_KB + 1048576 ))            # verbose band: avail floor + 1 GiB
NEAR_FREE_KB=$(( MIN_FREE_KB + 1048576 ))  # verbose band: free floor + 1 GiB

echo "$(date '+%F %T') watchdog start: container=$CONTAINER" \
     "floors: MemAvailable<${MIN_GIB}GiB, MemFree<${MIN_FREE_GIB}GiB (while MemAvailable<${FREE_GATE_GIB}GiB);" \
     "trigger=${CONSEC} consecutive samples; grace=${GRACE}s; archive=$ARCHIVE_DIR"

archive_logs() {  # <timestamp>
    mkdir -p "$ARCHIVE_DIR"
    docker logs --tail 3000 "$CONTAINER" > "$ARCHIVE_DIR/${CONTAINER}-$1-container.log" 2>&1 || true
    [[ -f "$OWN_LOG" ]] && cp -f "$OWN_LOG" "$ARCHIVE_DIR/${CONTAINER}-$1-memwatch.log"
    echo "$(date '+%F %T') archived container log + watchdog log to $ARCHIVE_DIR/${CONTAINER}-$1-*.log"
}

stop_container() {  # <reason>
    local ts; ts=$(date '+%Y%m%dT%H%M%S')
    echo "$(date '+%F %T') $1 -> stopping $CONTAINER"
    archive_logs "$ts"
    docker stop -t "$GRACE" "$CONTAINER" >/dev/null 2>&1 \
        || docker kill "$CONTAINER" >/dev/null 2>&1
    echo "$(date '+%F %T') stopped (NV_ERR_NO_MEMORY seen since watchdog start: $nvrm_total)"
    [[ -f "$OWN_LOG" ]] && cp -f "$OWN_LOG" "$ARCHIVE_DIR/${CONTAINER}-$ts-memwatch.log"
    exit 2
}

tick=0
below_avail=0
below_free=0
nvrm_total=0
cg_path=""
while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; do
    eval "$(awk '/^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapFree|AnonPages|Shmem|Mapped|Slab|SUnreclaim|PageTables|KernelStack):/ {
                     sub(":", "", $1); print "m_" $1 "=" $2 }' /proc/meminfo)"
    avail=$m_MemAvailable; free=$m_MemFree
    driver=$(( m_MemTotal - m_MemFree - m_Buffers - m_Cached - m_AnonPages - m_Slab - m_PageTables - m_KernelStack ))
    if [[ -z "$cg_path" || ! -f "$cg_path" ]]; then
        cg_path="/sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null).scope/memory.current"
    fi
    cg=$(cat "$cg_path" 2>/dev/null || echo 0)

    if (( avail < MIN_KB )); then
        below_avail=$(( below_avail + 1 ))
        echo "$(date '+%F %T') below MemAvailable floor ${below_avail}/${CONSEC}: MemAvailable=$((avail/1024)) MiB MemFree=$((free/1024)) MiB"
        (( below_avail >= CONSEC )) && stop_container "MemAvailable under ${MIN_GIB} GiB for ${CONSEC} samples"
    else
        (( below_avail > 0 )) && echo "$(date '+%T') recovered after ${below_avail} sub-floor MemAvailable sample(s): MemAvailable=$((avail/1024)) MiB"
        below_avail=0
    fi
    if (( free < MIN_FREE_KB && avail < FREE_GATE_KB )); then
        below_free=$(( below_free + 1 ))
        echo "$(date '+%F %T') below MemFree floor ${below_free}/${CONSEC}: MemFree=$((free/1024)) MiB MemAvailable=$((avail/1024)) MiB"
        (( below_free >= CONSEC )) && stop_container "MemFree under ${MIN_FREE_GIB} GiB for ${CONSEC} samples"
    else
        (( below_free > 0 )) && echo "$(date '+%T') recovered after ${below_free} sub-floor MemFree sample(s): MemFree=$((free/1024)) MiB"
        below_free=0
    fi

    if (( tick % 10 == 0 )); then
        nvrm=$(journalctl -k --since "-12s" -q 2>/dev/null | grep -c NV_ERR_NO_MEMORY || true)
        if (( nvrm > 0 )); then
            nvrm_total=$(( nvrm_total + nvrm ))
            echo "$(date '+%F %T') NVRM: ${nvrm} NV_ERR_NO_MEMORY in the last 12 s (total ${nvrm_total}); MemFree=$((free/1024)) MiB MemAvailable=$((avail/1024)) MiB"
        fi
    fi

    if (( tick % 5 == 0 || avail < NEAR_KB || (free < NEAR_FREE_KB && avail < FREE_GATE_KB) )); then
        echo "$(date '+%T') avail=$((avail/1024))MiB free=$((free/1024))MiB swapfree=$((m_SwapFree/1024))MiB container=$((cg/1048576))MiB" \
             "cached=$((m_Cached/1024))MiB anon=$((m_AnonPages/1024))MiB shmem=$((m_Shmem/1024))MiB mapped=$((m_Mapped/1024))MiB" \
             "sunreclaim=$((m_SUnreclaim/1024))MiB driver=$((driver/1024))MiB"
    fi
    tick=$((tick+1))
    sleep 1
done
echo "$(date '+%F %T') container gone; watchdog exit (NV_ERR_NO_MEMORY seen since watchdog start: $nvrm_total)"
