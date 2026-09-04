#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# memwatch.sh <container> <min_avail_gib> [consecutive_samples]
# Host-memory watchdog for the single-Spark deployment. On unified memory an
# exhausted pool hangs the kernel instead of raising an OOM, so this kills the
# container when host MemAvailable stays below the floor. It is a second line
# of defence behind the container's cgroup cap; a userspace poller cannot catch
# a GiB/s collapse alone.
#
# MemAvailable is noisy on this workload -- measured here, ~107 MiB mean change
# between 5 s samples with excursions past 1 GiB. Two servers were killed by a
# single sample dipping under the floor while the samples either side sat
# 400-700 MiB above it. So the trigger requires CONSEC consecutive sub-floor
# readings; a lone excursion resets the counter and is logged instead.
#
# Also logs a memory timeline every 5 s for post-mortems, and every sample once
# within 1 GiB of the floor so the reading that triggers a kill is in the log.
CONTAINER="${1:?container}"; MIN_GIB="${2:-6}"; CONSEC="${3:-5}"
MIN_KB=$(( MIN_GIB * 1048576 ))
NEAR_KB=$(( MIN_KB + 1048576 ))     # verbose logging band: floor + 1 GiB
GRACE="${MEMWATCH_GRACE:-10}"       # seconds to let vLLM unlink its POSIX shm
echo "$(date '+%F %T') watchdog start: container=$CONTAINER floor=${MIN_GIB}GiB" \
     "trigger=${CONSEC} consecutive samples"

# Resolve the container's memory.current path once. The systemd-driver path
# below does not exist with the cgroupfs driver, cgroup v1, or rootless
# docker; probing at startup turns a silent "container=0MiB forever" into a
# one-time warning.
CID=$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null || echo "")
CG_PATH=""
for candidate in \
    "/sys/fs/cgroup/system.slice/docker-${CID}.scope/memory.current" \
    "/sys/fs/cgroup/system.slice/docker-${CID}.scope/memory.usage_in_bytes" \
    "/sys/fs/cgroup/memory/docker/${CID}/memory.usage_in_bytes" \
    "/sys/fs/cgroup/memory/system.slice/docker-${CID}.scope/memory.usage_in_bytes"; do
    if [[ -n "$CID" && -r "$candidate" ]]; then
        CG_PATH="$candidate"
        break
    fi
done
if [[ -z "$CG_PATH" ]]; then
    echo "$(date '+%F %T') WARN: container cgroup memory file not readable; falling back to docker stats for the container= timeline"
fi

tick=0
below=0
cg=0
while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; do
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    free=$(awk '/MemFree/{print $2}' /proc/meminfo)
    swapfree=$(awk '/SwapFree/{print $2}' /proc/meminfo)
    if [[ -n "$CG_PATH" ]]; then
        cg=$(cat "$CG_PATH" 2>/dev/null || echo 0)
    fi

    if (( avail < MIN_KB )); then
        below=$(( below + 1 ))
        echo "$(date '+%F %T') below floor ${below}/${CONSEC}: MemAvailable=$((avail/1024)) MiB"
        if (( below >= CONSEC )); then
            echo "$(date '+%F %T') MemAvailable under ${MIN_GIB} GiB for ${CONSEC} samples -> stopping $CONTAINER"
            # SIGTERM first: a SIGKILL leaks this container's POSIX shm onto the
            # host's /dev/shm until reboot (--ipc host). Fall back if it hangs.
            docker stop -t "$GRACE" "$CONTAINER" >/dev/null 2>&1 \
                || docker kill "$CONTAINER" >/dev/null 2>&1
            echo "$(date '+%F %T') stopped"
            exit 2
        fi
    else
        if (( below > 0 )); then
            echo "$(date '+%T') recovered after ${below} sub-floor sample(s): MemAvailable=$((avail/1024)) MiB"
        fi
        below=0
    fi

    if (( tick % 5 == 0 || avail < NEAR_KB )); then
        if [[ -z "$CG_PATH" ]]; then
            # ~100 ms per call, so only on printed ticks. docker stats prints
            # e.g. "8.87GiB / 105GiB" — honour the unit suffix, and note it
            # reports the working set while memory.current includes page
            # cache, so the fallback reads lower than the cgroup path.
            raw=$(docker stats --no-stream --format '{{.MemUsage}}' "$CONTAINER" 2>/dev/null)
            cg=$(awk '{tok=$1; v=tok; gsub(/[A-Za-z]/,"",v); u=tok; gsub(/[0-9.]/,"",u);
                       m=(u=="GiB"?2**30:(u=="MiB"?2**20:(u=="KiB"?2**10:(u=="B"?1:2**30))));
                       printf "%d", v*m}' <<< "${raw:-0MiB}")
        fi
        echo "$(date '+%T') avail=$((avail/1024))MiB free=$((free/1024))MiB swapfree=$((swapfree/1024))MiB container=$((cg/1048576))MiB"
    fi
    tick=$((tick+1))
    sleep 1
done
echo "$(date '+%F %T') container gone; watchdog exit"
