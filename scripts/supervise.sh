#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# supervise.sh — the single state machine that owns the 24/7 lifecycle of the
# single-Spark vLLM container. Started as a systemd USER unit
# (qwen38-flash-supervisor.service, Restart=on-failure). The container itself
# runs with NO docker --restart (deliberately: docker restarting it behind our
# back resurrects it unwatched — memwatch dead, stale shm). The supervisor is
# the only actor that restarts.
#
# State lives in logs/supervisor.state (gitignored under logs/):
#   emergency_count, window_start, last_probe_fail, launch_failures.
# Restarting the supervisor must not forget the circuit breaker: heartbeats
# report unconditionally; guards do not reset on the guard's own restart.
#
# Ticks every 10 s:
#   1. Boot gate: docker info must succeed (user unit cannot order after
#      docker.service; poll instead). No alert storm while docker is down.
#   2. comfy-h3.service (user + system scope) active -> do nothing, alert at
#      most once/hour (port thief, review §4.7).
#   3. Container missing -> if stop.sh-initiated (flag file logs/stopping
#      touched by maintenance/stop) wait; else clean_shm -> start.sh -> on
#      non-zero exit backoff 30s*2^n cap 15min, count launch_failures; reset
#      only after a launch reaches /health 200.
#   4. Container up but memwatch not running -> start it via the shared
#      scripts/start-memwatch.sh (same invocation start.sh uses).
#   5. Probe cadence 1/min: 5 consecutive failures -> emergency: alert ->
#      stop.sh -> emergency_count++ -> step 3.
#   6. Circuit breaker: 3 emergencies in a 2h rolling window -> BREAKER_OPEN in
#      state; while open, only alert every 30 min, never relaunch. Human
#      re-arms by removing the state file; BREAKER_RESET_ON_BOOT=1 clears it
#      on reboot (a reboot is a human's hand on the box).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

STOPPING_FLAG="$REPO_DIR/logs/stopping"
STATE_FILE="$REPO_DIR/logs/supervisor.state"
TICK_S=10

# Read the same env snapshot logic start.sh uses for the knobs the supervisor
# needs (container name, memwatch floors, alerting). Environment wins; .env
# is the fallback (same precedence as start.sh).
if [[ -f "$REPO_DIR/.env" ]]; then
    # shellcheck source=.env
    source "$REPO_DIR/.env"
fi
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
MEMWATCH_GRACE="${MEMWATCH_GRACE:-30}"
# Windows during which an adopted container (one the supervisor did not launch)
# is treated as possibly still starting: do NOT probe. start.sh is the only
# authority on readiness; the supervisor must not emergency-stop a container
# that is mid-weight-load. Default ~15 min covers cold start + first-boot PLE
# build.
ADOPT_GRACE_S="${ADOPT_GRACE_S:-900}"
# A logs/stopping flag older than this is abandoned (maintenance crashed, or a
# manual stop nobody resumed); the supervisor reclaims it instead of waiting
# forever. 2h is far longer than any normal maintenance window.
STOPPING_MAX_AGE_S="${STOPPING_MAX_AGE_S:-7200}"
# Mirror start.sh's readiness window: a launch the supervisor starts should get
# the same deadline before its stale launching flag expires.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"

# Env overrides (defaults match plan 1.1).
BOOT_GATE_S="${BOOT_GATE_S:-10}"
ALERT_COMFY_HOUR_S="${ALERT_COMFY_HOUR_S:-3600}"
MAX_LAUNCH_FAILURES="${MAX_LAUNCH_FAILURES:-3}"
BACKOFF_INIT_S="${BACKOFF_INIT_S:-30}"
BACKOFF_MAX_S="${BACKOFF_MAX_S:-900}"
PROBE_FAILS_BEFORE_EMERGENCY="${PROBE_FAILS_BEFORE_EMERGENCY:-5}"
BREAKER_EMERGENCY_MAX="${BREAKER_EMERGENCY_MAX:-3}"
BREAKER_WINDOW_S="${BREAKER_WINDOW_S:-7200}"
BREAKER_RESET_ON_BOOT="${BREAKER_RESET_ON_BOOT:-1}"
BREAKER_OPEN_ALERT_S="${BREAKER_OPEN_ALERT_S:-1800}"
PROBE_RETRY_S="${PROBE_RETRY_S:-60}"

alert() { "$REPO_DIR/scripts/alert.sh" "$*" || true; }
log()   { echo "$(date '+%F %T') [supervise] $*"; }

state_get() {  # <key> <default>
    local k="$1" d="${2:-}"
    local v
    v=$(grep -E "^${k}=" "$STATE_FILE" 2>/dev/null | tail -1 | cut -d= -f2-)
    [[ -n "$v" ]] || v="$d"
    printf '%s' "$v"
}
state_set() {  # <key> <value>
    local k="$1" v="$2"
    [[ -f "$STATE_FILE" ]] || { mkdir -p "$REPO_DIR/logs"; : > "$STATE_FILE"; }
    # awk-based rewrite: no sed delimiter to collide with values (the memwatch
    # marker and mtime compose a value containing '|' and '&'); the key regex
    # is anchored and the value is literal, so state cannot be corrupted.
    awk -v k="$k" -v v="$v" '
        BEGIN { found = 0 }
        $0 ~ "^" k "=" { print k "=" v; found = 1; next }
        { print }
        END { if (!found) print k "=" v }
    ' "$STATE_FILE" > "$STATE_FILE.new" && mv "$STATE_FILE.new" "$STATE_FILE"
}

ensure_state() {
    local boot_ok="$(state_get boot "0")"
    if [[ "$boot_ok" != "1" ]]; then
        # Fresh process start (or host reboot). The breaker resets on boot; a
        # mere supervisor restart must NOT (jschmied's rule: guards must not
        # reset on the guard's own restart).
        local uptime; uptime=$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo 0)
        if [[ "$BREAKER_RESET_ON_BOOT" == "1" && "$uptime" -lt 300 ]]; then
            state_set emergency_count 0
            state_set window_start ""
            state_set last_probe_fail 0
            state_set launch_failures 0
            state_set breaker_open 0
            state_set breaker_open_alert ""
            state_set launching 0
            state_set launch_until 0
            state_set adopt_since 0
            state_set last_memwatch_emergency none
            state_set stop_source manual
            # A stale logs/stopping from a pre-reboot manual stop must not keep
            # the supervisor waiting forever after boot (it survives on disk;
            # nothing else removes it).
            rm -f "$STOPPING_FLAG" 2>/dev/null || true
        fi
        state_set boot 1
    fi
}

container_up() { docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; }
# Anchored to the memwatch binary path: the wrapper's own argv
# (start-memwatch.sh <container>) must not match, same class of bug as the
# start-memwatch.sh pkill self-kill found in the drills.
memwatch_up()  { pgrep -f "[f]iles/memwatch.sh $CONTAINER_NAME" >/dev/null 2>&1; }

clean_shm() {
    # Only when our container is not running and no vLLM/sglang container is
    # running: their segments are not ours to remove (stop.sh's rule).
    if container_up; then
        return 0
    fi
    if docker ps --format '{{.Image}}' 2>/dev/null | grep -qiE 'vllm|sglang'; then
        _shm_warn_ts="${_shm_warn_ts:-0}"
        _now=$(date +%s)
        if (( _now - _shm_warn_ts >= 3600 )); then
            log "WARN foreign engine container present; skipping shm cleanup (risk, not blocker)"
            _shm_warn_ts=$_now
        fi
        return 0
    fi
    leaked_count=0
    leaked=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -print0 2>/dev/null | tr -cd '\0' | wc -c)
    if [[ "$leaked" -gt 0 ]]; then
        # Authoritative guard: a live process holding a segment means an
        # engine we failed to identify by image name still uses it. Refuse to
        # delete anything that is open elsewhere (fuser/lsof may be absent).
        # NUL-safe throughout: /dev/shm is world-writable and a whitespace or
        # newline in a planted file name must not become a second target.
        _held=""
        _fuser=""
        if command -v fuser >/dev/null 2>&1; then
            _fuser=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -print0 2>/dev/null \
                     | xargs -0 -r fuser 2>/dev/null || true)
        elif command -v lsof >/dev/null 2>&1; then
            _fuser=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -print0 2>/dev/null \
                     | xargs -0 -r -n1 lsof 2>/dev/null || true)
        fi
        if [[ -n "$_fuser" ]]; then
            _shm_warn_ts="${_shm_warn_ts:-0}"
            _now=$(date +%s)
            if (( _now - _shm_warn_ts >= 3600 )); then
                log "WARN /dev/shm segments still held by another process; skipping cleanup (risk, not blocker)"
                _shm_warn_ts=$_now
            fi
            return 0
        fi
        bytes=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -print0 2>/dev/null \
                | xargs -0 -r stat -c '%s' 2>/dev/null | awk '{s+=$1} END {print s+0}')
        find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -print0 2>/dev/null \
            | xargs -0 -r rm -f 2>/dev/null
        log "shm cleanup: removed $leaked segment(s) ($((bytes/1048576)) MiB)"
    fi
}

probe_once() {
    # Runs the stateless probe; returns 0 healthy, 1 unhealthy.
    "$REPO_DIR/scripts/health-probe.sh" >/dev/null 2>&1
}

relaunch() {
    # One launch attempt, then return. Persistent exponential backoff lives in
    # the outer loop (state launch_failures), so a supervisor restart mid-crash
    # loop resumes the backoff instead of resetting it. While launching the
    # probe is suppressed (state launching=1): start.sh is authoritative until
    # it exits, and probing a container mid-weight-load would emergency-stop a
    # healthy launch (~11 min cold start). A stale launching=1 (supervisor died
    # mid-launch) is self-clearing: launch_until carries the deadline.
    _launch_until=$(( $(date +%s) + (READY_TIMEOUT_S > 0 ? READY_TIMEOUT_S : 1800) ))
    state_set launching 1
    state_set launch_until "$_launch_until"
    if "$REPO_DIR/start.sh" >"$REPO_DIR/logs/supervise-start.log" 2>&1; then
        state_set launch_failures 0
        state_set launching 0
        state_set launch_until 0
        # adopt_since=1: sentinel meaning "no grace needed" — this launch just
        # reached /health; probe immediately.
        state_set adopt_since 1
        state_set last_probe_fail 0
        last_probe_ts=0
        log "relaunch succeeded (start.sh exit 0 = /health reached)"
        return 0
    fi
    state_set launching 0
    state_set launch_until 0
    state_set launch_failures "$(( $(state_get launch_failures 0) + 1 ))"
    state_set last_probe_fail 0
    log "launch attempt failed (launch_failures=$(state_get launch_failures 0))"
    return 1
}

emergency_stop() {
    # 5 consecutive probe failures -> alert -> stop.sh -> emergency_count++.
    local reason="$1"
    alert "SUPERVISOR emergency: $reason (container $CONTAINER_NAME)"
    # stop.sh touches the stopping flag (supervisor hold-off). An emergency
    # must NOT hold us off — recovery is the whole point — so clear the flag
    # stop.sh just raised. But a flag that was ALREADY there when this
    # emergency started belongs to a maintenance window (drill-6 finding):
    # the supervisor must not tear down a handshake it did not raise.
    state_set stop_source emergency
    local _pre_stop_flag=0
    [[ -f "$STOPPING_FLAG" ]] && _pre_stop_flag=1
    "$REPO_DIR/stop.sh" >/dev/null 2>&1 || true
    if [[ "$_pre_stop_flag" == "0" ]]; then
        rm -f "$STOPPING_FLAG" 2>/dev/null || true
    else
        log "logs/stopping predates this emergency (maintenance window); leaving it in place"
    fi
    # Roll the window BEFORE counting so a stale-window boundary cannot reset
    # the increment that just happened on the same tick.
    local ws; ws=$(state_get window_start "")
    if [[ -n "$ws" ]] && (( $(date +%s) - ws > BREAKER_WINDOW_S )); then
        state_set window_start ""
    fi
    if [[ -z "$(state_get window_start)" ]]; then
        state_set window_start "$(date +%s)"
    fi
    local ec=$(( $(state_get emergency_count 0) + 1 ))
    state_set emergency_count "$ec"
    log "emergency stop recorded ($reason); emergency_count=$ec"
}

# Current loop timekeeping.
last_probe_ts=0
ensure_state

while true; do
    if ! docker info >/dev/null 2>&1; then
        sleep "$BOOT_GATE_S"
        continue
    fi

    # comfy-h3 port thief (review §4.7): both scopes.
    _comfy=0
    systemctl is-active comfy-h3.service >/dev/null 2>&1 && _comfy=1
    systemctl --user is-active comfy-h3.service >/dev/null 2>&1 && _comfy=1
    if [[ "$_comfy" == "1" ]]; then
        _ca=$(state_get comfy_last_alert "0")
        if (( $(date +%s) - _ca >= ALERT_COMFY_HOUR_S )); then
            alert "SUPERVISOR: comfy-h3.service is active — it will steal the API port."
            state_set comfy_last_alert "$(date +%s)"
        fi
        sleep "$TICK_S"
        continue
    fi

    # Circuit breaker check.
    if [[ "$(state_get breaker_open 0)" == "1" ]]; then
        _bts=$(state_get breaker_open_alert "0")
        if (( $(date +%s) - _bts >= BREAKER_OPEN_ALERT_S )); then
            alert "SUPERVISOR: circuit breaker OPEN (${BREAKER_EMERGENCY_MAX} emergencies in ${BREAKER_WINDOW_S}s). Not relaunching. Human must remove logs/supervisor.state."
            state_set breaker_open_alert "$(date +%s)"
        fi
        sleep "$TICK_S"
        continue
    fi

    # Maintenance / stop.sh-initiated downtime: a fresh flag means "do not
    # RELAUNCH a missing container". It must not blind supervision of a
    # container that is actually up (a maintenance wrapper whose smoke test
    # failed leaves the flag while the server may be healthy — stop.sh already
    # killed the watchdog, so memwatch must resume). But while the flag is
    # fresh (the maintenance window is mid-flight: its own start.sh is the
    # readiness authority), the PROBE must also hold: probing a
    # cold-starting maintenance container to 5 fails would emergency-stop
    # inside the window and fight the handshake (drill-6 finding). A flag
    # older than STOPPING_MAX_AGE_S is abandoned — reclaim it loudly.
    _stopping_fresh=0
    if [[ -f "$STOPPING_FLAG" ]]; then
        _flag_age=$(( $(date +%s) - $(stat -c %Y "$STOPPING_FLAG" 2>/dev/null || echo 0) ))
        if (( _flag_age > STOPPING_MAX_AGE_S )); then
            alert "SUPERVISOR: logs/stopping is ${_flag_age}s old (>${STOPPING_MAX_AGE_S}s) — treating as abandoned and clearing it."
            rm -f "$STOPPING_FLAG"
        else
            _stopping_fresh=1
        fi
    fi
    if [[ "$_stopping_fresh" == "1" ]]; then
        if container_up; then
            # memwatch must run for an up container even mid-window; probe holds.
            if ! memwatch_up; then
                log "memwatch not running; starting it"
                MEMWATCH_MIN_FREE_GIB="$MEMWATCH_MIN_FREE_GIB" \
                    MEMWATCH_FREE_GATE_GIB="$MEMWATCH_FREE_GATE_GIB" \
                    MEMWATCH_GRACE="$MEMWATCH_GRACE" \
                    bash "$REPO_DIR/scripts/start-memwatch.sh" "$CONTAINER_NAME" "$MEMWATCH_MIN_GIB" || true
            fi
            "$REPO_DIR/scripts/memwatch-rotate.sh" "$CONTAINER_NAME" || true
            state_set last_probe_fail 0
            sleep "$TICK_S"
            continue
        else
            sleep "$TICK_S"
            continue
        fi
    fi

    if ! container_up; then
        clean_shm
        # A memwatch emergency stop (memory floors) is an emergency stop too:
        # it emits WATCHDOG EMERGENCY STOP as its last log line. Count it into
        # the breaker the same way a probe emergency is counted — but exactly
        # once per marker (dedupe on the tail line + mtime). A stop the
        # supervisor initiated itself (emergency_stop's stop.sh call re-archives
        # and the marker is still in the live log for this tick) must not be
        # counted a second time: stop_source=emergency marks that.
        _stop_src="$(state_get stop_source manual)"
        if [[ "$_stop_src" != "manual" ]]; then
            state_set stop_source manual
        fi

        # Roll the window BEFORE any count so a stale-window boundary cannot
        # reset an increment that just happened on the same tick.
        ws=$(state_get window_start "0")
        if [[ "$ws" != "0" ]] && (( $(date +%s) - ws > BREAKER_WINDOW_S )); then
            state_set emergency_count 0
            state_set window_start ""
        fi

        if [[ "$_stop_src" != "emergency" ]]; then
            mw_log="$REPO_DIR/logs/memwatch-${CONTAINER_NAME}.log"
            mw_marker=""
            if [[ -f "$mw_log" ]]; then
                mw_marker=$(grep "WATCHDOG EMERGENCY STOP" "$mw_log" | tail -1 || true)
            fi
            if [[ -n "$mw_marker" ]]; then
                _mw_id="${mw_marker}|$(stat -c %Y "$mw_log" 2>/dev/null || echo 0)"
                _last_mw="$(state_get last_memwatch_emergency "none")"
                if [[ "$_last_mw" == "none" ]]; then
                    # Fresh state (boot reset or human re-arm) seeing an OLD
                    # marker: the memwatch log persists across state resets,
                    # so a marker from a previous generation must not seed a
                    # fresh breaker count. Adopt it as seen, count nothing.
                    state_set last_memwatch_emergency "$_mw_id"
                    log "stale memwatch marker adopted without counting (state was reset)"
                elif [[ "$_last_mw" != "$_mw_id" ]]; then
                    alert "SUPERVISOR: memwatch emergency stop detected: $mw_marker"
                    state_set last_memwatch_emergency "$_mw_id"
                    ec=$(( $(state_get emergency_count 0) + 1 ))
                    state_set emergency_count "$ec"
                    if [[ -z "$(state_get window_start)" ]]; then
                        state_set window_start "$(date +%s)"
                    fi
                    log "memwatch emergency counted -> emergency_count=$ec"
                fi
            fi
        fi
        if [[ "$(state_get breaker_open 0)" != "1" ]]; then
            # Circuit breaker: 3 emergencies in the rolling window -> open.
            ec=$(state_get emergency_count 0)
            if (( ec >= BREAKER_EMERGENCY_MAX )); then
                state_set breaker_open 1
                state_set breaker_open_alert "0"
                alert "SUPERVISOR: circuit breaker OPEN (${ec} emergencies). Not relaunching. Human must remove logs/supervisor.state."
                sleep "$TICK_S"
                continue
            fi
            # Persistent exponential backoff: 30 s * 2^n, cap 15 min, keyed on
            # launch_failures in state so a supervisor crash restart resumes it.
            lf=$(state_get launch_failures 0)
            if (( lf >= MAX_LAUNCH_FAILURES )); then
                alert "SUPERVISOR: ${lf} launch failures (cap ${MAX_LAUNCH_FAILURES}) — holding off. Remove logs/supervisor.state to re-arm, or fix the launch cause."
                sleep "$TICK_S"
                continue
            fi
            _bs=$(( BACKOFF_INIT_S * 2 ** lf ))
            if (( _bs > BACKOFF_MAX_S )); then
                _bs=$BACKOFF_MAX_S
            fi
            if (( _bs > 0 )); then
                log "backing off ${_bs}s (launch_failures=$lf)"
                sleep "$_bs"
            fi
            if ! relaunch; then
                alert "SUPERVISOR: launch failed (see logs/supervise-start.log). Backoff in effect, will re-try."
            fi
        fi
        sleep "$TICK_S"
        continue
    fi

    # Container up:
    if ! memwatch_up; then
        log "memwatch not running; starting it"
        MEMWATCH_MIN_FREE_GIB="$MEMWATCH_MIN_FREE_GIB" \
            MEMWATCH_FREE_GATE_GIB="$MEMWATCH_FREE_GATE_GIB" \
            MEMWATCH_GRACE="$MEMWATCH_GRACE" \
            bash "$REPO_DIR/scripts/start-memwatch.sh" "$CONTAINER_NAME" "$MEMWATCH_MIN_GIB" || true
    fi
    # memwatch log rotation (10 MB copy-truncate) every tick.
    "$REPO_DIR/scripts/memwatch-rotate.sh" "$CONTAINER_NAME" || true

    # Probe gate: launching/supervisor-launched containers and freshly adopted
    # ones are not probed until they have had their grace window. start.sh
    # clears launching on exit 0; a restart of the supervisor mid-launch sees
    # launching still set (state file survives — the guard must not reset on
    # the guard's own restart). A stale launching=1 (supervisor died before
    # start.sh exited, or start.sh hung past READY_TIMEOUT_S) self-clears once
    # launch_until passes, so the probe resumes against whatever is actually
    # up.
    if [[ "$(state_get launching 0)" == "1" ]]; then
        _lu=$(state_get launch_until 0)
        if (( _lu > 0 && $(date +%s) > _lu )); then
            state_set launching 0
            state_set launch_until 0
            state_set last_probe_fail 0
            log "stale launching state expired; resuming probe"
        else
            sleep "$TICK_S"
            continue
        fi
    fi
    _adopt_ok=$(state_get adopt_since "0")
    if [[ "$_adopt_ok" == "0" ]]; then
        # First tick seeing an up container that this supervisor did not
        # launch: start the adoption grace.
        state_set adopt_since "$(date +%s)"
    elif [[ "$_adopt_ok" == "1" ]]; then
        :  # sentinel: our own launch just reached /health; probe now.
    elif (( $(date +%s) - _adopt_ok < ADOPT_GRACE_S )); then
        sleep "$TICK_S"
        continue
    else
        # Grace expired: this is a fresh container we adopted; its first probe
        # must not inherit failures counted against a previous generation.
        # One-shot: move to the probe-now sentinel so the reset runs exactly
        # once instead of on every tick.
        state_set last_probe_fail 0
        state_set adopt_since 1
    fi

    # Probe cadence: 1/min.
    _now=$(date +%s)
    if (( _now - last_probe_ts >= PROBE_RETRY_S )); then
        last_probe_ts=$_now
        if probe_once; then
            state_set last_probe_fail 0
        else
            pf=$(( $(state_get last_probe_fail 0) + 1 ))
            state_set last_probe_fail "$pf"
            log "probe failed ${pf}/${PROBE_FAILS_BEFORE_EMERGENCY}"
            if (( pf >= PROBE_FAILS_BEFORE_EMERGENCY )); then
                # Corroborate before escalating: a probe that fails because it
                # queued behind sustained traffic is congestion, not a dead
                # engine. If /health still answers 200, do not emergency-stop a
                # healthy saturated server (this box: stop/relaunch is the
                # riskiest operation, unified memory).
                _health=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://localhost:${PORT:-8888}/health" 2>/dev/null || echo "000")
                if [[ "$_health" == "200" ]]; then
                    alert "SUPERVISOR: ${pf} probe failures but /health is 200 — treating as queue congestion, not escalating."
                    state_set last_probe_fail 0
                    log "probe failures are congestion (health 200); resetting"
                else
                    emergency_stop "probe failed ${pf} consecutive times (health $_health)"
                    state_set last_probe_fail 0
                    sleep "$TICK_S"
                    continue
                fi
            fi
        fi
    fi
    sleep "$TICK_S"
done
