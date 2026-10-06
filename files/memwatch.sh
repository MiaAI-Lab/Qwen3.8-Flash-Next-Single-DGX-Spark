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
# Relief step for the MemFree trigger (MEMWATCH_RELIEF=drop_caches; default
# off). At the default kernel watermarks kswapd reclaims page cache only near
# 150 MiB free, so the MemFree trigger can fire while GiBs of clean page cache
# are still resident (2026-09-23 on one Spark: stop at MemFree 1.2 GiB,
# MemAvailable 8.3 GiB, 5.7 GiB cached). When the MemFree trigger has counted
# MEMWATCH_RELIEF_AT samples (default 2, less than CONSEC) and the reclaimable
# file cache is at least MEMWATCH_RELIEF_MIN_GIB (default 1), the watchdog
# writes 1 to /proc/sys/vm/drop_caches through `sudo -n tee` under
# `timeout -k 2 10`, logs the result, resets the MemFree counter and keeps
# sampling. If the condition holds for CONSEC more samples, it stops the
# container as before. No `sync`: drop_caches drops only clean pages, and
# sync can block on dirty data while the watchdog must keep sampling.
#   * The relief runs in the background. The watchdog waits at most 1 s for
#     it in the sample that starts it, then keeps sampling: both triggers and
#     the NV_ERR_NO_MEMORY check continue while a slow relief runs, and the
#     MemFree counter resets only when the relief ends. A signal cannot stop
#     the kernel's drop_caches scan, so `timeout` alone does not bound it.
#     A relief that has not ended 14 s (timeout + 4 s) after its start is
#     logged as stuck and relief is disabled for the rest of the run.
#   * At most one relief per MEMWATCH_RELIEF_INTERVAL seconds (default 60).
#     While that limit holds, the MemFree trigger works as with relief off.
#   * If the relief command fails (for example no NOPASSWD sudo rule), the
#     watchdog logs it once and disables relief for the rest of the run.
#   * reclaimable = Active(file) + Inactive(file) - Mapped - Dirty - Writeback:
#     drop_caches skips mapped, dirty and writeback pages. Mapped also counts
#     mapped shmem, so this figure is low, never high.
#   * The MemAvailable trigger has no relief: page cache already counts as
#     available there.
#   * The PLE table rows that the gather reads stay mapped through its
#     np.memmap, so drop_caches skips them. It drops only unmapped cache,
#     for example the checkpoint read at load time.
# The sudoers rule the relief needs (visudo -f /etc/sudoers.d/memwatch):
#   <user> ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
#
# Every 10 s it also counts NV_ERR_NO_MEMORY lines the kernel log gained since
# the previous check; that is the earliest signal this box gives and is logged
# whenever it is non-zero. (A fixed 12 s window every 10 s double-counted lines
# in the 2 s overlap: 6 counted for 5 logged on 2026-09-05 09:00.)
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
# MEMWATCH_ARCHIVE_DIR (logs/archive), MEMWATCH_RELIEF (off | drop_caches),
# MEMWATCH_RELIEF_AT (2), MEMWATCH_RELIEF_MIN_GIB (1),
# MEMWATCH_RELIEF_INTERVAL (60), MEMWATCH_RELIEF_CMD (the drop_caches command
# above; tests replace it with a stub).
# Test knobs: MEMWATCH_MEMINFO (/proc/meminfo), MEMWATCH_SAMPLE_S (1),
# MEMWATCH_RELIEF_TIMEOUT_S (10).
CONTAINER="${1:?container}"; MIN_GIB="${2:-6}"; CONSEC="${3:-5}"
MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
GRACE="${MEMWATCH_GRACE:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OWN_LOG="${MEMWATCH_LOG:-$REPO_DIR/logs/memwatch-${CONTAINER}.log}"
ARCHIVE_DIR="${MEMWATCH_ARCHIVE_DIR:-$REPO_DIR/logs/archive}"
MEMINFO="${MEMWATCH_MEMINFO:-/proc/meminfo}"
SAMPLE_S="${MEMWATCH_SAMPLE_S:-1}"
RELIEF="${MEMWATCH_RELIEF:-off}"
RELIEF_AT="${MEMWATCH_RELIEF_AT:-2}"
RELIEF_MIN_GIB="${MEMWATCH_RELIEF_MIN_GIB:-1}"
RELIEF_INTERVAL="${MEMWATCH_RELIEF_INTERVAL:-60}"
RELIEF_CMD="${MEMWATCH_RELIEF_CMD:-echo 1 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null}"
RELIEF_TIMEOUT_S="${MEMWATCH_RELIEF_TIMEOUT_S:-10}"
RELIEF_KILL_S=2                   # timeout -k: SIGKILL this long after SIGTERM
RELIEF_WAIT_STEPS=20              # wait up to 20 x 0.05 s for a relief inline

MIN_KB=$(( MIN_GIB * 1048576 ))
MIN_FREE_KB=$(( MIN_FREE_GIB * 1048576 ))
FREE_GATE_KB=$(( FREE_GATE_GIB * 1048576 ))
NEAR_KB=$(( MIN_KB + 1048576 ))            # verbose band: avail floor + 1 GiB
NEAR_FREE_KB=$(( MIN_FREE_KB + 1048576 ))  # verbose band: free floor + 1 GiB

echo "$(date '+%F %T') watchdog start: container=$CONTAINER" \
     "floors: MemAvailable<${MIN_GIB}GiB, MemFree<${MIN_FREE_GIB}GiB (while MemAvailable<${FREE_GATE_GIB}GiB);" \
     "trigger=${CONSEC} consecutive samples; grace=${GRACE}s; archive=$ARCHIVE_DIR"

# Relief knobs are checked once here. A bad value disables relief with one log
# line; it never stops the watchdog.
RELIEF_MIN_KB=0
case "$RELIEF" in
    off) ;;
    drop_caches)
        if ! [[ "$RELIEF_AT" =~ ^[0-9]+$ && "$RELIEF_MIN_GIB" =~ ^[0-9]+$ && "$RELIEF_INTERVAL" =~ ^[0-9]+$ && "$RELIEF_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]]; then
            echo "$(date '+%F %T') relief disabled: MEMWATCH_RELIEF_AT ($RELIEF_AT), MEMWATCH_RELIEF_MIN_GIB ($RELIEF_MIN_GIB), MEMWATCH_RELIEF_INTERVAL ($RELIEF_INTERVAL) and MEMWATCH_RELIEF_TIMEOUT_S ($RELIEF_TIMEOUT_S) must be whole numbers"
            RELIEF=off
        elif (( RELIEF_AT < 1 || RELIEF_AT >= CONSEC )); then
            echo "$(date '+%F %T') relief disabled: MEMWATCH_RELIEF_AT ($RELIEF_AT) must be at least 1 and less than the trigger ($CONSEC samples)"
            RELIEF=off
        elif ! RELIEF_DIR=$(mktemp -d "${TMPDIR:-/tmp}/memwatch-relief.XXXXXX" 2>/dev/null); then
            echo "$(date '+%F %T') relief disabled: mktemp -d in ${TMPDIR:-/tmp} failed"
            RELIEF=off
        else
            trap 'rm -rf "$RELIEF_DIR"' EXIT
            RELIEF_MIN_KB=$(( RELIEF_MIN_GIB * 1048576 ))
            RELIEF_STUCK_S=$(( RELIEF_TIMEOUT_S + RELIEF_KILL_S + 2 ))
            echo "$(date '+%F %T') relief: drop_caches after ${RELIEF_AT}/${CONSEC} sub-floor MemFree samples" \
                 "when reclaimable cache >= ${RELIEF_MIN_GIB} GiB; at most one per ${RELIEF_INTERVAL}s; timeout ${RELIEF_TIMEOUT_S}s;" \
                 "runs in the background"
        fi
        ;;
    *)
        echo "$(date '+%F %T') relief disabled: MEMWATCH_RELIEF must be off or drop_caches (got '$RELIEF')"
        RELIEF=off
        ;;
esac

# read_meminfo [prefix]: sets <prefix><field> (default prefix m_) from
# $MEMINFO. Active(file) and Inactive(file) become <prefix>Activefile and
# <prefix>Inactivefile.
read_meminfo() {
    eval "$(awk -v p="${1:-m_}" '/^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapFree|AnonPages|Shmem|Mapped|Slab|SUnreclaim|PageTables|KernelStack|Active\(file\)|Inactive\(file\)|Dirty|Writeback):/ {
                     sub(":", "", $1); gsub(/[()]/, "", $1); print p $1 "=" $2 }' "$MEMINFO")"
}

# reclaimable_kb [prefix]: file cache that drop_caches can drop, in kB, from
# the <prefix> fields (default m_). See the header for the formula.
reclaimable_kb() {
    local p="${1:-m_}"
    local af="${p}Activefile" inf="${p}Inactivefile" mp="${p}Mapped" dt="${p}Dirty" wb="${p}Writeback"
    local r=$(( ${!af:-0} + ${!inf:-0} - ${!mp:-0} - ${!dt:-0} - ${!wb:-0} ))
    (( r < 0 )) && r=0
    echo "$r"
}

now_ms() { local t; t=$(date '+%s%N'); echo $(( t / 1000000 )); }

relief_last_s=""
relief_pid=""                     # set while a relief runs in the background
# Runs from the MemFree trigger at each sample once it has counted RELIEF_AT
# samples, so a relief can still run if the rate limit ends mid-count. Skip
# reasons are logged once per count, at RELIEF_AT. Returns 0 always: a relief
# problem must never end the watchdog, even under set -e.
try_relief() {
    [[ "$RELIEF" == drop_caches && -z "$relief_pid" ]] || return 0
    (( below_free >= RELIEF_AT )) || return 0
    local rec_before; rec_before=$(reclaimable_kb)
    if (( rec_before < RELIEF_MIN_KB )); then
        (( below_free == RELIEF_AT )) && echo "$(date '+%F %T') relief skipped: reclaimable cache $((rec_before/1024)) MiB is under ${RELIEF_MIN_GIB} GiB"
        return 0
    fi
    if [[ -n "$relief_last_s" ]] && (( SECONDS - relief_last_s < RELIEF_INTERVAL )); then
        (( below_free == RELIEF_AT )) && echo "$(date '+%F %T') relief skipped: last relief $(( SECONDS - relief_last_s ))s ago, limit one per ${RELIEF_INTERVAL}s"
        return 0
    fi
    relief_free_before=$free relief_avail_before=$avail relief_rec_before=$rec_before
    relief_count=$below_free relief_start_s=$SECONDS
    rm -f "$RELIEF_DIR/result" "$RELIEF_DIR/out"
    # The subshell writes "<rc> <ms> <MemFree> <MemAvailable> <reclaimable>"
    # (kB, read right after the command) to result when the command ends.
    # Its own stdout and stderr go to /dev/null, so a relief that ends after
    # the watchdog exits writes nothing into the log.
    (
        s0=$(now_ms); rc=0
        timeout -k "$RELIEF_KILL_S" "$RELIEF_TIMEOUT_S" sh -c "$RELIEF_CMD" > "$RELIEF_DIR/out" 2>&1 || rc=$?
        s1=$(now_ms)
        read_meminfo r_
        echo "$rc $(( s1 - s0 )) ${r_MemFree:-0} ${r_MemAvailable:-0} $(reclaimable_kb r_)" > "$RELIEF_DIR/result.tmp"
        mv -f "$RELIEF_DIR/result.tmp" "$RELIEF_DIR/result"
    ) < /dev/null > /dev/null 2>&1 &
    relief_pid=$!
    local i
    for (( i = 0; i < RELIEF_WAIT_STEPS; i++ )); do
        [[ -f "$RELIEF_DIR/result" ]] && break
        sleep 0.05
    done
    if [[ -f "$RELIEF_DIR/result" ]]; then
        relief_poll
    else
        echo "$(date '+%F %T') relief still running after 1 s; sampling continues, the MemFree counter resets when it ends"
    fi
    return 0
}

# relief_poll: handles the end of a background relief. Called once per
# sample before the triggers and from try_relief. Returns 0 always.
relief_poll() {
    [[ -n "$relief_pid" ]] || return 0
    if [[ ! -f "$RELIEF_DIR/result" ]]; then
        if (( SECONDS - relief_start_s > RELIEF_STUCK_S )); then
            echo "$(date '+%F %T') RELIEF STUCK: drop_caches still running after $(( SECONDS - relief_start_s ))s" \
                 "(timeout ${RELIEF_TIMEOUT_S}s + ${RELIEF_KILL_S}s kill did not end it)"
            echo "$(date '+%F %T') relief disabled for this run; the MemFree trigger stops the container after ${CONSEC} samples as with relief off"
            RELIEF=off relief_pid=""
        fi
        return 0
    fi
    wait "$relief_pid" 2>/dev/null || true
    relief_pid=""
    relief_last_s=$SECONDS
    local rc="" ms="" a_free="" a_avail="" a_rec="" out=""
    read -r rc ms a_free a_avail a_rec < "$RELIEF_DIR/result" || true
    [[ "$rc" =~ ^[0-9]+$ ]] || rc=1
    [[ "$ms" =~ ^[0-9]+$ ]] || ms=0
    [[ "$a_free" =~ ^[0-9]+$ ]] || a_free=0
    [[ "$a_avail" =~ ^[0-9]+$ ]] || a_avail=0
    [[ "$a_rec" =~ ^[0-9]+$ ]] || a_rec=0
    if (( rc != 0 )); then
        [[ -f "$RELIEF_DIR/out" ]] && out=$(head -n 1 "$RELIEF_DIR/out" 2>/dev/null || true)
        local why="exit $rc"
        (( rc == 124 || rc == 137 )) && why="timed out after ${RELIEF_TIMEOUT_S}s"
        echo "$(date '+%F %T') RELIEF FAILED ($why, $ms ms): $out"
        echo "$(date '+%F %T') relief disabled for this run; the MemFree trigger stops the container after ${CONSEC} samples as with relief off"
        RELIEF=off
        return 0
    fi
    echo "$(date '+%F %T') RELIEF drop_caches at ${relief_count}/${CONSEC} sub-floor MemFree samples ($ms ms):" \
         "before MemFree=$((relief_free_before/1024)) MiB MemAvailable=$((relief_avail_before/1024)) MiB reclaimable=$((relief_rec_before/1024)) MiB;" \
         "after MemFree=$(( a_free / 1024 )) MiB MemAvailable=$(( a_avail / 1024 )) MiB reclaimable=$(( a_rec / 1024 )) MiB;" \
         "MemFree counter reset"
    below_free=0
    return 0
}

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
    # The marker line is how the supervisor tells an emergency stop from a
    # clean stop.sh: only the emergency path emits it (review §4.1). It goes
    # into the live log BEFORE the final archive copy so both carry it.
    echo "WATCHDOG EMERGENCY STOP $1" >> "$OWN_LOG"
    # Alert after the stop so the reason carries the final state; a failure
    # here must not change control flow (review §4.6).
    if [[ -x "$REPO_DIR/scripts/alert.sh" ]]; then
        "$REPO_DIR/scripts/alert.sh" "memwatch emergency stop: $1" || true
    fi
    [[ -f "$OWN_LOG" ]] && cp -f "$OWN_LOG" "$ARCHIVE_DIR/${CONTAINER}-$ts-memwatch.log"
    exit 2
}

tick=0
below_avail=0
below_free=0
nvrm_total=0
nvrm_since=$(date '+%Y-%m-%d %H:%M:%S')
cg_path=""
# LEAK TREND (review §4.3): baseline driver figure over a post-load window,
# then flag growth >= TREND_GIB (4) once per day. The baseline window opens
# only after TREND_WARMUP_S — the weight-load ramp (multi-tens-of-GiB) grows
# `driver` immediately, and a min-including-that-ramp would make every cold
# start look like a 4 GiB leak.
baseline_driver=""
trend_logged_day=""
TREND_GIB="${MEMWATCH_TREND_GIB:-4}"
TREND_WARMUP_S="${MEMWATCH_TREND_WARMUP_S:-900}"
TREND_BASELINE_S="${MEMWATCH_TREND_BASELINE_S:-600}"
while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; do
    read_meminfo
    avail=$m_MemAvailable; free=$m_MemFree
    relief_poll
    driver=$(( m_MemTotal - m_MemFree - m_Buffers - m_Cached - m_AnonPages - m_Slab - m_PageTables - m_KernelStack ))
    if (( driver > 0 )); then
        # Baseline window: [TREND_WARMUP_S, TREND_WARMUP_S + TREND_BASELINE_S)
        # after memwatch start. Take the minimum driver in that post-load
        # window, then freeze it.
        if (( tick >= TREND_WARMUP_S && tick < TREND_WARMUP_S + TREND_BASELINE_S )); then
            if [[ -z "$baseline_driver" || "$driver" -lt "$baseline_driver" ]]; then
                baseline_driver="$driver"
            fi
        elif (( tick == TREND_WARMUP_S + TREND_BASELINE_S )) && [[ -z "$baseline_driver" ]]; then
            baseline_driver="$driver"
        fi
    fi
    if [[ -n "$baseline_driver" ]] && (( driver - baseline_driver >= TREND_GIB * 1048576 )); then
        _today=$(date '+%F')
        if [[ "$trend_logged_day" != "$_today" ]]; then
            trend_logged_day="$_today"
            echo "$(date '+%F %T') LEAK TREND: driver ${TREND_GIB} GiB above the baseline $((baseline_driver/1048576)) MiB (now $((driver/1048576)) MiB). The 2-3 GiB per-request growth is accumulating; scheduled relaunch is the response."
        fi
    fi
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
        try_relief
    else
        (( below_free > 0 )) && echo "$(date '+%T') recovered after ${below_free} sub-floor MemFree sample(s): MemFree=$((free/1024)) MiB"
        below_free=0
    fi

    if (( tick % 10 == 0 )); then
        # journalctl --since is inclusive at second granularity, so exclude the
        # boundary second on the next pass by advancing it past this check.
        nvrm_now=$(date '+%Y-%m-%d %H:%M:%S')
        nvrm=$(journalctl -k --since "$nvrm_since" --until "$nvrm_now" -q 2>/dev/null | grep -c NV_ERR_NO_MEMORY || true)
        nvrm_since="$nvrm_now"
        if (( nvrm > 0 )); then
            nvrm_total=$(( nvrm_total + nvrm ))
            echo "$(date '+%F %T') NVRM: ${nvrm} NV_ERR_NO_MEMORY since last check (total ${nvrm_total}); MemFree=$((free/1024)) MiB MemAvailable=$((avail/1024)) MiB"
        fi
    fi

    if (( tick % 5 == 0 || avail < NEAR_KB || (free < NEAR_FREE_KB && avail < FREE_GATE_KB) )); then
        echo "$(date '+%T') avail=$((avail/1024))MiB free=$((free/1024))MiB swapfree=$((m_SwapFree/1024))MiB container=$((cg/1048576))MiB" \
             "cached=$((m_Cached/1024))MiB anon=$((m_AnonPages/1024))MiB shmem=$((m_Shmem/1024))MiB mapped=$((m_Mapped/1024))MiB" \
             "sunreclaim=$((m_SUnreclaim/1024))MiB driver=$((driver/1024))MiB"
    fi
    tick=$((tick+1))
    sleep "$SAMPLE_S"
done
echo "$(date '+%F %T') container gone; watchdog exit (NV_ERR_NO_MEMORY seen since watchdog start: $nvrm_total)"
