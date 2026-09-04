#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# memwatch.sh <container> <min_avail_gib>
# Host-memory watchdog for the single-Spark deployment. On unified memory an
# exhausted pool hangs the kernel instead of raising an OOM, so this kills the
# container the moment host MemAvailable drops below the floor. It is a second
# line of defence behind the container's cgroup cap (which is what actually
# bounds UVM on GB10); a userspace poller cannot catch a GiB/s collapse alone.
# Also logs a memory timeline every 5 s for post-mortems.
CONTAINER="${1:?container}"; MIN_GIB="${2:-6}"
MIN_KB=$(( MIN_GIB * 1048576 ))
echo "$(date '+%F %T') watchdog start: container=$CONTAINER floor=${MIN_GIB}GiB"
tick=0
while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; do
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    free=$(awk '/MemFree/{print $2}' /proc/meminfo)
    swapfree=$(awk '/SwapFree/{print $2}' /proc/meminfo)
    cg=$(cat /sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null).scope/memory.current 2>/dev/null || echo 0)
    if (( avail < MIN_KB )); then
        echo "$(date '+%F %T') MemAvailable=$((avail/1024)) MiB < floor -> docker kill $CONTAINER"
        docker kill "$CONTAINER" >/dev/null 2>&1
        echo "$(date '+%F %T') killed"
        exit 2
    fi
    if (( tick % 5 == 0 )); then
        echo "$(date '+%T') avail=$((avail/1024))MiB free=$((free/1024))MiB swapfree=$((swapfree/1024))MiB container=$((cg/1048576))MiB"
    fi
    tick=$((tick+1))
    sleep 1
done
echo "$(date '+%F %T') container gone; watchdog exit"
