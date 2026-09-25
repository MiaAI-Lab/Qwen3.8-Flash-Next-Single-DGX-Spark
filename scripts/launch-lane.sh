# SPDX-License-Identifier: AGPL-3.0-or-later
# Split EXTRA_VLLM_ARGS into argv tokens with the same semantics start.sh uses
# (shlex, single quotes group, double quotes are data; newline is the token
# separator, so tokens containing newlines are not supported). Sets the
# nameref target to the resulting array; empty input yields an empty array.
split_extra_vllm_args() {  # <out-nameref> <value>
    local -n _out=$1
    local _val="$2" _parsed
    _out=()
    [[ -z "$_val" ]] && return 0
    _parsed=$(python3 - "$_val" <<'PY'
import sys, shlex
lx = shlex.shlex(sys.argv[1], posix=True)
lx.whitespace_split = True
lx.quotes = "'"
lx.commenters = ""
print("\n".join(lx))
PY
    ) || return 1
    if [[ -n "$_parsed" ]]; then
        local _tok
        while IFS= read -r _tok; do _out+=("$_tok"); done <<< "$_parsed"
    fi
    return 0
}
load_launch_lane() {
    START_SCRIPT="$REPO_DIR/start.sh"
    LANE_MEMWATCH_MIN_GIB=""
    LANE_MEMWATCH_MIN_FREE_GIB=""
    local file="$REPO_DIR/logs/launch-lane"
    grep -qx 'V030=true' "$file" 2>/dev/null || return 0
    START_SCRIPT="$REPO_DIR/start-v030.sh"
    LANE_MEMWATCH_MIN_GIB=$(grep -xE 'MEMWATCH_MIN_GIB=[0-9]+' "$file" | tail -1 | cut -d= -f2)
    LANE_MEMWATCH_MIN_FREE_GIB=$(grep -xE 'MEMWATCH_MIN_FREE_GIB=[0-9]+' "$file" | tail -1 | cut -d= -f2)
    return 0
}
