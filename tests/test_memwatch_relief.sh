#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Hermetic tests for the files/memwatch.sh relief step (MEMWATCH_RELIEF).
#
#   bash tests/test_memwatch_relief.sh
#
# The test copies memwatch.sh into a temporary repo layout and puts fake
# docker, journalctl and sudo executables first on PATH. Nothing touches a real
# container, the kernel log, sudo or /proc/sys.
#   * The fake `docker ps` is the sample clock: each call copies the next
#     meminfo frame into the file that MEMWATCH_MEMINFO names. When the
#     frames end, it lists no container and the watchdog exits.
#   * The fake journalctl prints nothing and exits 1, so the NV_ERR_NO_MEMORY
#     grep finds no match on every check (the pipefail case).
#   * The relief command is a stub that records the call and writes the
#     "after drop_caches" meminfo, or fails, or runs slowly, or ignores
#     SIGTERM.
#   * TMPDIR points into the per-run state directory, so the watchdog's
#     relief directory stays inside the test tree.
# Each case runs twice: with plain bash (how start-memwatch.sh runs it) and
# with bash -euo pipefail, to show that no relief path ends the watchdog.
# The two cases that wait for real timeouts (about 15 s) run only with
# bash -euo pipefail.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BASELINE_REV=6b50864   # memwatch.sh before the relief step

FAKEBIN="$TMP/bin"
mkdir -p "$FAKEBIN" "$TMP/repo/files"
cp "$REPO/files/memwatch.sh" "$TMP/repo/files/memwatch.sh"

cat > "$FAKEBIN/docker" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    ps)
        n=$(( $(cat "$T_STATE/tick" 2>/dev/null || echo 0) + 1 ))
        echo "$n" > "$T_STATE/tick"
        [[ -f "$T_STATE/stopped" ]] && exit 0
        if [[ -f "$T_FRAMES/$n" ]]; then
            cp "$T_FRAMES/$n" "$MEMWATCH_MEMINFO"
            echo "$T_CONTAINER"
        fi
        ;;
    inspect) echo fakeid ;;
    logs) echo "fake container log" ;;
    stop|kill) echo "$*" >> "$T_STATE/docker_calls"; touch "$T_STATE/stopped" ;;
    *) echo "fake docker: unexpected $*" >&2; exit 1 ;;
esac
EOF
cat > "$FAKEBIN/journalctl" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
# Fake sudo: records its argv and never runs the command.
cat > "$FAKEBIN/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$T_STATE/sudo_calls"
echo "sudo: a password is required" >&2
exit 1
EOF
cat > "$FAKEBIN/relief-ok" <<'EOF'
#!/usr/bin/env bash
echo call >> "$T_STATE/relief_calls"
[[ -f "$T_STATE/after" ]] && cp "$T_STATE/after" "$MEMWATCH_MEMINFO"
exit 0
EOF
# Slow relief: works, but only after T_RELIEF_SLEEP seconds (default 4).
cat > "$FAKEBIN/relief-slow" <<'EOF'
#!/usr/bin/env bash
echo call >> "$T_STATE/relief_calls"
sleep "${T_RELIEF_SLEEP:-4}"
[[ -f "$T_STATE/after" ]] && cp "$T_STATE/after" "$MEMWATCH_MEMINFO"
exit 0
EOF
# Hung relief: ignores SIGTERM (sleep inherits the ignored signal), so only
# the SIGKILL from `timeout -k` ends it.
cat > "$FAKEBIN/relief-hang" <<'EOF'
#!/usr/bin/env bash
echo call >> "$T_STATE/relief_calls"
trap '' TERM
sleep 30
EOF
# Fake timeout for the stuck case: it models a drop_caches write that no
# signal ends. It sits in its own directory so that only that case uses it.
mkdir -p "$TMP/stuckbin"
cat > "$TMP/stuckbin/timeout" <<'EOF'
#!/usr/bin/env bash
echo call >> "$T_STATE/relief_calls"
exec sleep 9
EOF
chmod +x "$TMP/stuckbin/timeout"
cat > "$FAKEBIN/relief-fail" <<'EOF'
#!/usr/bin/env bash
echo call >> "$T_STATE/relief_calls"
echo "relief-fail: operation not permitted" >&2
echo "second line that the log must not carry"
exit 1
EOF
chmod +x "$FAKEBIN"/*

# meminfo <free_mib> <avail_mib> <reclaimable_mib>: one fake /proc/meminfo.
# Active(file) equals Mapped and Dirty/Writeback are 0, so the watchdog's
# reclaimable figure equals Inactive(file) = <reclaimable_mib>.
meminfo() {
    local free=$(( $1 * 1024 )) avail=$(( $2 * 1024 )) rec=$(( $3 * 1024 )) mapped=$(( 2048 * 1024 ))
    cat <<EOF
MemTotal:       127598608 kB
MemFree:        $free kB
MemAvailable:   $avail kB
Buffers:           63156 kB
Cached:         $(( mapped + rec + 187392 )) kB
SwapCached:       267876 kB
Active(file):   $mapped kB
Inactive(file): $rec kB
SwapFree:        3440000 kB
Dirty:                 0 kB
Writeback:             0 kB
AnonPages:      14395764 kB
Mapped:         $mapped kB
Shmem:            187392 kB
Slab:            3535520 kB
SUnreclaim:      1376776 kB
KernelStack:       76032 kB
PageTables:       290164 kB
EOF
}

# Frame shorthands (MiB free:avail:reclaimable).
#   OK  healthy
#   B   MemFree trigger condition, 5.7 GiB reclaimable (the 08:36 state)
#   BS  MemFree trigger condition, only 500 MiB reclaimable
#   A   MemAvailable trigger only
#   AB  both triggers
frame_spec() {
    case "$1" in
        OK) echo "30000 40000 5700" ;;
        B)  echo "1270 8342 5700" ;;
        BS) echo "1270 8342 500" ;;
        A)  echo "3000 5000 1500" ;;
        AB) echo "1000 5000 1500" ;;
        *)  echo "bad frame $1" >&2; exit 1 ;;
    esac
}

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ok   $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL $1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# run_mw <shell flags|""> <frames...>: runs the watchdog on the frames with
# the current MW_ENV. Sets RC, OUT (the watchdog log file) and ELAPSED (wall
# seconds). A frame N:<name> stands for N copies of <name>.
run_mw() {
    local flags="$1"; shift
    local st="$TMP/state"
    rm -rf "$st"; mkdir -p "$st/frames" "$TMP/repo/logs"
    local i=0 f
    local n
    for f in "$@"; do
        n=1
        if [[ "$f" == *:* ]]; then n=${f%%:*}; f=${f#*:}; fi
        while (( n-- > 0 )); do
            i=$((i+1))
            # shellcheck disable=SC2046
            meminfo $(frame_spec "$f") > "$st/frames/$i"
        done
    done
    meminfo 6000 13000 1200 > "$st/after"
    OUT="$TMP/repo/logs/memwatch-tc.log"
    rm -f "$OUT"; rm -rf "$TMP/repo/logs/archive"
    local t0=$SECONDS
    # The real timeout runs first, so a fake timeout in MW_ENV's PATH reaches
    # only the watchdog.
    # shellcheck disable=SC2086
    timeout 60 env PATH="$FAKEBIN:$PATH" T_STATE="$st" T_FRAMES="$st/frames" T_CONTAINER=tc \
        TMPDIR="$st" MEMWATCH_MEMINFO="$st/meminfo" MEMWATCH_SAMPLE_S=0 \
        MEMWATCH_LOG="$OUT" MEMWATCH_ARCHIVE_DIR="$TMP/repo/logs/archive" \
        "${MW_ENV[@]}" \
        bash $flags "$TMP/repo/files/memwatch.sh" tc 6 5 > "$OUT" 2>&1
    RC=$?
    # shellcheck disable=SC2034  # read by check strings through eval
    ELAPSED=$(( SECONDS - t0 ))
    STATE="$st"
}

count() { grep -c -- "$1" "$OUT" || true; }
first_line() { grep -n -m 1 -- "$1" "$OUT" | cut -d: -f1; }
left_dirs() { find "$STATE" -maxdepth 1 -name 'memwatch-relief.*' | wc -l; }
calls() { if [[ -f "$STATE/$1" ]]; then wc -l < "$STATE/$1"; else echo 0; fi; }

MODES=("" "-euo pipefail")
mode_name() { if [[ -z "$1" ]]; then echo "plain bash"; else echo "bash $1"; fi; }

on_env=(MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-ok")

for mode in "${MODES[@]}"; do
    echo "== $(mode_name "$mode")"

    echo "-- relief off: MemFree trigger stops the container as before"
    MW_ENV=(MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-ok")
    run_mw "$mode" B B B B B OK
    check "exit 2 (emergency stop)" '[[ $RC == 2 ]]'
    check "stop reason is the MemFree floor" '[[ $(count "MemFree under 2 GiB for 5 samples -> stopping tc") == 1 ]]'
    check "docker stop called" '[[ $(calls docker_calls) == 1 ]]'
    check "relief command never called" '[[ $(calls relief_calls) == 0 ]]'
    check "no relief lines in the log" '[[ $(count "RELIEF\|relief") == 0 ]]'

    echo "-- relief on, cache available, MemFree recovers"
    MW_ENV=("${on_env[@]}")
    run_mw "$mode" B B OK OK OK
    check "exit 0 (container gone, no stop)" '[[ $RC == 0 ]]'
    check "relief called once" '[[ $(calls relief_calls) == 1 ]]'
    check "relief logged with before/after state" \
        '[[ $(count "RELIEF drop_caches at 2/5 sub-floor MemFree samples ([0-9]* ms): before MemFree=1270 MiB MemAvailable=8342 MiB reclaimable=5700 MiB; after MemFree=6000 MiB MemAvailable=13000 MiB reclaimable=1200 MiB; MemFree counter reset") == 1 ]]'
    check "start line names the relief" '[[ $(count "relief: drop_caches after 2/5 sub-floor MemFree samples when reclaimable cache >= 1 GiB; at most one per 60s; timeout 10s") == 1 ]]'
    check "docker stop not called" '[[ $(calls docker_calls) == 0 ]]'

    echo "-- relief on: the counter restarts from zero after the relief"
    run_mw "$mode" B B B B B OK
    check "exit 0: 2 + 3 sub-floor samples do not reach 5" '[[ $RC == 0 ]]'
    check "count restarts at 1/5 after the relief" '[[ $(count "below MemFree floor 1/5") == 2 ]]'
    check "recovered after 3 samples" '[[ $(count "recovered after 3 sub-floor MemFree sample(s)") == 1 ]]'

    echo "-- relief on, condition persists after the relief"
    run_mw "$mode" B B B B B B B OK
    check "exit 2" '[[ $RC == 2 ]]'
    check "relief called once" '[[ $(calls relief_calls) == 1 ]]'
    check "stop after 5 more sub-floor samples" '[[ $(count "below MemFree floor") == 7 && $(count "below MemFree floor 5/5") == 1 ]]'
    check "stop reason is the MemFree floor" '[[ $(count "WATCHDOG EMERGENCY STOP MemFree under 2 GiB for 5 samples") == 1 ]]'

    echo "-- relief command fails: logged once, then normal behaviour"
    MW_ENV=(MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-fail")
    run_mw "$mode" B B B OK B B B B B OK
    check "exit 2" '[[ $RC == 2 ]]'
    check "relief tried once" '[[ $(calls relief_calls) == 1 ]]'
    check "failure logged once with the first output line" '[[ $(count "RELIEF FAILED (exit 1, [0-9]* ms): relief-fail: operation not permitted$") == 1 ]]'
    check "only the first output line is logged" '[[ $(count "second line") == 0 ]]'
    check "fallback logged once" '[[ $(count "relief disabled for this run") == 1 ]]'
    check "counter not reset by a failed relief" '[[ $(count "recovered after 3 sub-floor MemFree sample(s)") == 1 ]]'
    check "stop at 5/5 in the second episode" '[[ $(count "below MemFree floor 5/5") == 1 && $(count "MemFree under 2 GiB for 5 samples -> stopping") == 1 ]]'
    check "relief directory removed on exit" '[[ $(left_dirs) == 0 ]]'

    echo "-- relief command fails: relief stays off even with no rate limit"
    # MEMWATCH_RELIEF_INTERVAL=0 takes the rate limit out, so only the
    # disable after the failure can stop a second try in the second episode.
    MW_ENV=(MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-fail" MEMWATCH_RELIEF_INTERVAL=0)
    run_mw "$mode" B B B OK B B B B B OK
    check "exit 2" '[[ $RC == 2 ]]'
    check "relief tried once in two episodes" '[[ $(calls relief_calls) == 1 ]]'
    check "failure logged once" '[[ $(count "RELIEF FAILED") == 1 ]]'
    check "no rate-limit skip line" '[[ $(count "relief skipped") == 0 ]]'

    echo "-- slow relief: the MemAvailable trigger keeps working while it runs"
    MW_ENV=(MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-slow" T_RELIEF_SLEEP=4)
    run_mw "$mode" B B 5:A OK
    check "exit 2 on the MemAvailable floor" '[[ $RC == 2 && $(count "MemAvailable under 6 GiB for 5 samples -> stopping") == 1 ]]'
    check "relief started once" '[[ $(calls relief_calls) == 1 ]]'
    check "still-running line logged once" '[[ $(count "relief still running after 1 s; sampling continues") == 1 ]]'
    check "stop came before the relief ended" '[[ $(count "RELIEF drop_caches") == 0 && $ELAPSED -lt 4 ]]'

    echo "-- slow relief: the MemFree counter keeps counting while it runs"
    run_mw "$mode" 5:B OK
    check "exit 2 on the MemFree floor" '[[ $RC == 2 && $(count "MemFree under 2 GiB for 5 samples -> stopping") == 1 ]]'
    check "counted 3/5 to 5/5 during the relief" '[[ $(count "below MemFree floor [345]/5") == 3 && $(count "RELIEF drop_caches") == 0 ]]'

    echo "-- slow relief ends while sampling continues"
    MW_ENV=(MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-slow" T_RELIEF_SLEEP=1.5 MEMWATCH_SAMPLE_S=0.2)
    run_mw "$mode" B B 12:OK
    check "exit 0" '[[ $RC == 0 ]]'
    check "relief logged once when it ends" '[[ $(count "RELIEF drop_caches at 2/5 sub-floor MemFree samples (1[5-9][0-9][0-9] ms): before MemFree=1270 MiB") == 1 ]]'
    check "samples continued before the relief ended" \
        '[[ -n $(first_line "recovered after 2") && $(first_line "recovered after 2") -lt $(first_line "RELIEF drop_caches") ]]'

    echo "-- default relief command goes through sudo -n (fake sudo)"
    if [[ "$(PATH="$FAKEBIN:$PATH" command -v sudo)" != "$FAKEBIN/sudo" ]]; then
        echo "fake sudo is not first on PATH; refusing to continue" >&2; exit 1
    fi
    MW_ENV=(MEMWATCH_RELIEF=drop_caches)
    run_mw "$mode" B B B B B OK
    check "exit 2" '[[ $RC == 2 ]]'
    check "sudo called once with -n and the drop_caches command" \
        '[[ $(calls sudo_calls) == 1 && "$(cat "$STATE/sudo_calls")" == "-n tee /proc/sys/vm/drop_caches" ]]'
    check "sudo refusal logged" '[[ $(count "RELIEF FAILED (exit 1, [0-9]* ms): sudo: a password is required") == 1 ]]'

    echo "-- rate limit: one relief per MEMWATCH_RELIEF_INTERVAL"
    MW_ENV=("${on_env[@]}")
    run_mw "$mode" B B OK B B B B B OK
    check "interval 60: exit 2" '[[ $RC == 2 ]]'
    check "interval 60: relief called once" '[[ $(calls relief_calls) == 1 ]]'
    check "interval 60: skip logged once" '[[ $(count "relief skipped: last relief [0-9]*s ago, limit one per 60s") == 1 ]]'
    MW_ENV=("${on_env[@]}" MEMWATCH_RELIEF_INTERVAL=0)
    run_mw "$mode" B B OK B B B B B
    check "interval 0: exit 0" '[[ $RC == 0 ]]'
    check "interval 0: relief called three times" '[[ $(calls relief_calls) == 3 ]]'

    echo "-- small reclaimable cache: no relief, normal stop"
    MW_ENV=("${on_env[@]}")
    run_mw "$mode" BS BS BS BS BS OK
    check "exit 2" '[[ $RC == 2 ]]'
    check "relief not called" '[[ $(calls relief_calls) == 0 ]]'
    check "skip logged once" '[[ $(count "relief skipped: reclaimable cache 500 MiB is under 1 GiB") == 1 ]]'

    echo "-- MemAvailable trigger is unchanged"
    run_mw "$mode" A A A A A OK
    check "A only: exit 2 on the MemAvailable floor" '[[ $RC == 2 && $(count "MemAvailable under 6 GiB for 5 samples -> stopping") == 1 ]]'
    check "A only: relief not called" '[[ $(calls relief_calls) == 0 ]]'
    run_mw "$mode" AB AB AB AB AB OK
    check "A+B: relief runs for the MemFree count" '[[ $(calls relief_calls) == 1 ]]'
    check "A+B: stop still at the 5th MemAvailable sample" \
        '[[ $RC == 2 && $(count "below MemAvailable floor 5/5") == 1 && $(count "MemAvailable under 6 GiB for 5 samples -> stopping") == 1 ]]'

    echo "-- bad knobs disable the relief with one log line"
    MW_ENV=("${on_env[@]}" MEMWATCH_RELIEF_AT=5)
    run_mw "$mode" B B B B B OK
    check "RELIEF_AT >= trigger: disabled, normal stop" \
        '[[ $RC == 2 && $(count "relief disabled: MEMWATCH_RELIEF_AT (5)") == 1 && $(calls relief_calls) == 0 ]]'
    MW_ENV=(MEMWATCH_RELIEF=yes MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-ok")
    run_mw "$mode" B B B B B OK
    check "unknown MEMWATCH_RELIEF: disabled, normal stop" \
        '[[ $RC == 2 && $(count "relief disabled: MEMWATCH_RELIEF must be off or drop_caches") == 1 && $(calls relief_calls) == 0 ]]'
    MW_ENV=("${on_env[@]}" MEMWATCH_RELIEF_TIMEOUT_S=0)
    run_mw "$mode" B B B B B OK
    check "MEMWATCH_RELIEF_TIMEOUT_S=0: disabled, normal stop" \
        '[[ $RC == 2 && $(count "relief disabled: .*MEMWATCH_RELIEF_TIMEOUT_S (0) must be whole numbers") == 1 && $(calls relief_calls) == 0 ]]'
done

echo "== bash -euo pipefail, real timeouts"
echo "-- relief ignores SIGTERM: timeout -k kills it, failure logged, normal stop"
# exec makes the stub the direct child of timeout, as sudo is when sh execs
# it. SIGTERM then does not end it, and only the SIGKILL does.
MW_ENV=(MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_CMD="exec $FAKEBIN/relief-hang" MEMWATCH_RELIEF_TIMEOUT_S=1 MEMWATCH_SAMPLE_S=0.25)
run_mw "-euo pipefail" B B 16:OK 5:B OK
check "exit 2 on the MemFree floor" '[[ $RC == 2 && $(count "MemFree under 2 GiB for 5 samples -> stopping") == 1 ]]'
check "relief tried once" '[[ $(calls relief_calls) == 1 ]]'
check "timed out and logged once" '[[ $(count "RELIEF FAILED (timed out after 1s, [0-9]* ms)") == 1 && $(count "relief disabled for this run") == 1 ]]'
check "no stuck line" '[[ $(count "RELIEF STUCK") == 0 ]]'

echo "-- relief that no signal ends: logged as stuck, relief off, normal stop"
MW_ENV=(PATH="$TMP/stuckbin:$FAKEBIN:$PATH" MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-ok" MEMWATCH_RELIEF_TIMEOUT_S=1 MEMWATCH_SAMPLE_S=0.25)
run_mw "-euo pipefail" B B 28:OK 5:B OK
check "exit 2 on the MemFree floor" '[[ $RC == 2 && $(count "MemFree under 2 GiB for 5 samples -> stopping") == 1 ]]'
check "fake timeout reached once" '[[ $(calls relief_calls) == 1 ]]'
check "stuck logged once, relief disabled" '[[ $(count "RELIEF STUCK: drop_caches still running after [0-9]*s") == 1 && $(count "relief disabled for this run") == 1 ]]'
check "no second relief in the second episode" '[[ $(count "relief still running") == 1 ]]'

echo "== relief off matches the pre-relief script line for line"
if base_src="$(git -C "$REPO" show "$BASELINE_REV:files/memwatch.sh" 2>/dev/null)"; then
    mkdir -p "$TMP/base/files"
    # Point the old script at the fake meminfo and a 0 s sample; nothing else
    # changes.
    printf '%s\n' "$base_src" \
        | sed -e "s|' /proc/meminfo)\"|' \"\$MEMWATCH_MEMINFO\")\"|" -e 's|^    sleep 1$|    sleep 0|' \
        > "$TMP/base/files/memwatch.sh"
    if ! grep -q 'MEMWATCH_MEMINFO' "$TMP/base/files/memwatch.sh"; then
        bad "baseline rewrite did not apply"
    else
        norm() { sed -E 's/[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/<ts>/g; s/[0-9]{2}:[0-9]{2}:[0-9]{2}/<t>/g; s/[0-9]{8}T[0-9]{6}/<arch>/g' "$1"; }
        seq=(OK OK A OK B B OK OK OK OK OK OK A A B B B OK B B B B B OK)
        MW_ENV=(MEMWATCH_RELIEF_CMD="$FAKEBIN/relief-ok")
        run_mw "" "${seq[@]}"; new_rc=$RC; norm "$OUT" > "$TMP/new.log"
        cp "$TMP/base/files/memwatch.sh" "$TMP/repo/files/memwatch.sh"
        run_mw "" "${seq[@]}"; base_rc=$RC; norm "$OUT" > "$TMP/base.log"
        cp "$REPO/files/memwatch.sh" "$TMP/repo/files/memwatch.sh"
        check "same exit code ($new_rc)" '[[ $new_rc == "$base_rc" ]]'
        if diff -u "$TMP/base.log" "$TMP/new.log" > "$TMP/diff.txt"; then
            ok "same log, $(wc -l < "$TMP/new.log") lines"
        else
            bad "log differs:"; cat "$TMP/diff.txt"
        fi
    fi
else
    echo "  skip baseline $BASELINE_REV not in this checkout"
fi

echo "== launch paths pass the relief knobs through"
KNOBS=(MEMWATCH_RELIEF MEMWATCH_RELIEF_AT MEMWATCH_RELIEF_MIN_GIB MEMWATCH_RELIEF_INTERVAL)
for k in "${KNOBS[@]}"; do
    check "start.sh snapshots $k" "awk '/^_ENV_SNAPSHOT_VARS=\\(/,/\\)/' '$REPO/start.sh' | tr -s ' ()' '\\n' | grep -qx '$k'"
    check "start.sh forwards $k" "[[ \$(grep -c '$k=\"\\\$$k\"' '$REPO/start.sh') == 1 ]]"
    check "supervise.sh forwards $k at both call sites" "[[ \$(grep -c '$k=\"\\\$$k\"' '$REPO/scripts/supervise.sh') == 2 ]]"
done
# start-memwatch.sh against a stub memwatch.sh that prints its environment.
mkdir -p "$TMP/sm/scripts" "$TMP/sm/files"
cp "$REPO/scripts/start-memwatch.sh" "$TMP/sm/scripts/"
cat > "$TMP/sm/files/memwatch.sh" <<'EOF'
env | grep '^MEMWATCH_RELIEF' | sort
echo done
EOF
sm_name="relief-test-$$"
sm_log=$(MEMWATCH_RELIEF=drop_caches MEMWATCH_RELIEF_AT=3 MEMWATCH_RELIEF_INTERVAL=90 \
         bash "$TMP/sm/scripts/start-memwatch.sh" "$sm_name" 6)
for _ in $(seq 50); do grep -q '^done$' "$sm_log" 2>/dev/null && break; sleep 0.1; done
check "start-memwatch.sh forwards set knobs" \
    "grep -qx 'MEMWATCH_RELIEF=drop_caches' '$sm_log' && grep -qx 'MEMWATCH_RELIEF_AT=3' '$sm_log' && grep -qx 'MEMWATCH_RELIEF_INTERVAL=90' '$sm_log'"
check "start-memwatch.sh fills the default MEMWATCH_RELIEF_MIN_GIB" "grep -qx 'MEMWATCH_RELIEF_MIN_GIB=1' '$sm_log'"
sm_log=$(env -u MEMWATCH_RELIEF bash "$TMP/sm/scripts/start-memwatch.sh" "$sm_name" 6)
for _ in $(seq 50); do grep -q '^done$' "$sm_log" 2>/dev/null && break; sleep 0.1; done
check "start-memwatch.sh default is relief off" "grep -qx 'MEMWATCH_RELIEF=off' '$sm_log'"

echo
echo "passed $PASS, failed $FAIL"
(( FAIL == 0 ))
