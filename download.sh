#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
#
# download.sh — fetch the checkpoint into the local Hugging Face cache.
#
# start.sh deliberately never downloads: it resolves the model from
# $HF_HOME/hub/models--<org>--<name> and fails fast if it is absent. This is
# the script it points you at.
#
# The checkpoint is ~99 GiB, so this takes a while and is resumable — rerun it
# after an interruption and it picks up where it stopped.
#
# Usage:
#   ./download.sh                       # stock Mia NVFP4 (ABLIT=0)
#   ABLIT=1 ./download.sh               # gated Keys ablit checkpoint (~99 GiB, same as stock).
#                                       # Set HF_TOKEN, then accept the terms on
#                                       # https://huggingface.co/drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only
#   ./download.sh Org/Some-Other-Model  # an explicit repo id
#   HF_TOKEN=hf_... ./download.sh       # for a gated repo
#   VERIFY_SHA256=0 ./download.sh       # skip sha256 verification of LFS blobs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

for arg in "$@"; do
    case "$arg" in
        -h|--help) sed -n '5,/^set -euo pipefail$/p' "$0" | sed '$d;s/^# \?//'; exit 0 ;;
    esac
done

# Environment wins over .env, same precedence rule as start.sh.
_CLI_HF_TOKEN="${HF_TOKEN:-}"
_CLI_ABLIT="${ABLIT:-}"
_CLI_TP1_MODEL_ID="${TP1_MODEL_ID:-}"
_CLI_VERIFY_SHA256="${VERIFY_SHA256:-}"
if [[ -f .env ]]; then
    # shellcheck source=.env
    source .env
fi
[[ -n "$_CLI_HF_TOKEN" ]] && HF_TOKEN="$_CLI_HF_TOKEN"
HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$_CLI_ABLIT" ]] && ABLIT="$_CLI_ABLIT"
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" == "0" || "$ABLIT" == "1" ]] || err "ABLIT must be 0 or 1 (got: '$ABLIT')"
[[ -n "$_CLI_TP1_MODEL_ID" ]] && TP1_MODEL_ID="$_CLI_TP1_MODEL_ID"
VERIFY_SHA256="${_CLI_VERIFY_SHA256:-${VERIFY_SHA256:-1}}"

STOCK_MODEL_ID="Mia-AiLab/Qwen3.8-Flash-Next-NVFP4"
ABLIT_MODEL_ID="drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only"
if [[ $# -gt 0 ]]; then
    MODEL_ID="$1"
elif [[ -n "${TP1_MODEL_ID:-}" ]]; then
    MODEL_ID="$TP1_MODEL_ID"
elif [[ "$ABLIT" == "1" ]]; then
    MODEL_ID="$ABLIT_MODEL_ID"
else
    MODEL_ID="$STOCK_MODEL_ID"
fi
if [[ "$ABLIT" == "1" && "$MODEL_ID" != "$ABLIT_MODEL_ID" ]]; then
    warn "ABLIT=1 ignored for checkpoint selection: MODEL_ID=$MODEL_ID"
fi
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
ORG="${MODEL_ID%%/*}"; NAME="${MODEL_ID##*/}"
MODEL_PATH="$HF_CACHE_DIR/hub/models--${ORG}--${NAME}"
ABLIT_PAGE="https://huggingface.co/${ABLIT_MODEL_ID}"

# Prints a snapshot hash. Exit 0 = complete, 1 = incomplete, 2 = none.
# Prefers refs/main when that snapshot is complete, else the newest complete
# snapshot, else refs/main even if incomplete (so a partial tree can resume).
resolve_snapshot() {  # <model-path>
    python3 - "$1" <<'PY'
import json, pathlib, sys

def complete(snapshot: pathlib.Path) -> bool:
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        return False
    weight_map = json.loads(index.read_text()).get("weight_map", {})
    return bool(weight_map) and all((snapshot / name).is_file()
                                    for name in set(weight_map.values()))

repo = pathlib.Path(sys.argv[1])
snap_root = repo / "snapshots"
main = (repo / "refs" / "main").read_text().strip() if (repo / "refs" / "main").is_file() else ""
if main and complete(snap_root / main):
    print(main)
    raise SystemExit(0)
complete_snaps = []
if snap_root.is_dir():
    complete_snaps = [p for p in snap_root.iterdir() if p.is_dir() and complete(p)]
if complete_snaps:
    complete_snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(complete_snaps[0].name)
    raise SystemExit(0)
if main and (snap_root / main).is_dir():
    print(main)
    raise SystemExit(1)
if snap_root.is_dir():
    cands = sorted((p for p in snap_root.iterdir() if p.is_dir()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        print(cands[0].name)
        raise SystemExit(1)
raise SystemExit(2)
PY
}

hf_token_cached() {
    [[ -s "$HF_CACHE_DIR/token" || -s "${HF_HOME:-$HOME/.cache/huggingface}/token" || -s "$HOME/.huggingface/token" ]]
}

next_start_hint() {
    if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        info "Next:  ABLIT=1 ./start.sh   (or set ABLIT=1 in .env and run ./start.sh)"
    else
        info "Next:  ./start.sh"
    fi
}

info "Model:  $MODEL_ID"
info "Cache:  $HF_CACHE_DIR"
if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    info "ABLIT=1 is gated. Set HF_TOKEN in .env (or export it) before downloading:"
    info "  HF_TOKEN=hf_... ABLIT=1 ./download.sh"
    info "Repo:  $ABLIT_PAGE"
    info "If the download returns 403: Accept the terms on that page, then retry with HF_TOKEN."
    if [[ -z "$HF_TOKEN" ]]; then
        if hf_token_cached; then
            warn "HF_TOKEN is not set; using the cached huggingface-cli login."
            warn "     For ABLIT=1, set HF_TOKEN in .env so the gated download is explicit."
        else
            err "ABLIT=1 requires HF_TOKEN.
       1. Uncomment HF_TOKEN in .env (or: export HF_TOKEN=hf_...)
       2. Open $ABLIT_PAGE
       3. Accept the terms on that page
       4. ABLIT=1 ./download.sh"
        fi
    fi
fi

# Already complete? Require every shard named by the safetensors index. A
# config.json appears early in a partial download and is not sufficient.
# A complete cache still runs the sha256 guard below: "downloaded" and
# "verified" are different claims, and a blob corrupted on disk after the
# download (the case #32 exists for) is only caught by re-running it.
if [[ -d "$MODEL_PATH" ]]; then
    SNAP=""
    SNAP_RC=0
    SNAP="$(resolve_snapshot "$MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
    if [[ "$SNAP_RC" -eq 0 && -n "$SNAP" ]]; then
        ok "Already in cache: $MODEL_PATH ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
        SKIP_ARIA=1
    else
        if [[ -n "$SNAP" ]]; then
            warn "Partial download found (snapshot $SNAP); resuming."
        fi
    fi
fi

# Both checkpoints are the same ~99 GiB: the ablit is an in-place o_proj requant
# on the same 34-shard layout, so shard byte-lengths are identical to stock. Add
# the ~27 GiB packed PLE table start.sh builds on first launch. ABLIT=1 reuses
# the stock packed table, so drop that 27 GiB only once stock has actually
# built it -- an ablit-first box still pays for the build.
AVAIL_GIB=$(df -BG --output=avail "$(dirname "$HF_CACHE_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
STOCK_PLE_DIR="$HOME/.cache/vllm/ple_cache/${STOCK_MODEL_ID%%/*}--${STOCK_MODEL_ID##*/}"
NEED_GIB=130
PLE_REUSED=0
if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]] && ls "$STOCK_PLE_DIR"/*.packed_u8 >/dev/null 2>&1; then
    NEED_GIB=105
    PLE_REUSED=1
fi
if [[ -n "$AVAIL_GIB" && "$AVAIL_GIB" -lt "$NEED_GIB" ]]; then
    warn "Only ${AVAIL_GIB} GiB free on $(dirname "$HF_CACHE_DIR")."
    warn "     The checkpoint is ~99 GiB (ABLIT=0 and ABLIT=1 are the same size)."
    if [[ "$PLE_REUSED" == "1" ]]; then
        warn "     It sits beside the stock cache, whose packed PLE table is reused,"
        warn "     so ~105 GiB free is the safe figure for this second copy."
    else
        warn "     start.sh also builds a ~27 GiB packed PLE table beside it on first"
        warn "     launch. ~130 GiB free is the safe figure."
    fi
fi

mkdir -p "$HF_CACHE_DIR"

DL_PY='
import os, sys
from huggingface_hub import snapshot_download
# No single quotes inside this block: it lives in a single-quoted bash string.
DEFAULT_CMD = "HF_TOKEN=hf_... ./download.sh"
try:
    p = snapshot_download(
        repo_id=sys.argv[1],
        token=(os.environ.get("HF_TOKEN") or None),
        max_workers=4,
    )
except Exception as e:
    name = type(e).__name__
    text = str(e)
    code = getattr(getattr(e, "response", None), "status_code", None)
    gated = (
        name == "GatedRepoError"
        or code in (401, 403)
        or "gated" in text.lower()
        or " 403" in f" {text}"
    )
    if gated:
        print(
            "403: this repo is gated.\n"
            "Accept the terms on that page:\n"
            f"  https://huggingface.co/{sys.argv[1]}\n"
            "Then retry with HF_TOKEN set:\n"
            f"  {sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CMD}",
            file=sys.stderr,
        )
        sys.exit(1)
    raise
print(p)
'

if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    RETRY_HINT="HF_TOKEN=hf_... ABLIT=1 ./download.sh"
else
    RETRY_HINT="HF_TOKEN=hf_... ./download.sh $MODEL_ID"
fi

if [[ "${SKIP_ARIA:-0}" != "1" ]]; then
    info "Downloading (resumable; interrupt and rerun to continue)..."
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        HF_HOME="$HF_CACHE_DIR" HF_TOKEN="$HF_TOKEN" python3 -c "$DL_PY" "$MODEL_ID" "$RETRY_HINT"
    else
    info "huggingface_hub not on the host; using the container image instead."
    IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
    # Drop to the invoking user's uid:gid inside the container: the cache
    # must stay owned by the host user, or the unprivileged sha256 verify
    # below (and hf/huggingface-cli on the host) cannot write their state.
    docker run --rm -i \
        -e HF_HOME=/hf -e HF_TOKEN="$HF_TOKEN" \
        -u "$(id -u):$(id -g)" \
        -v "$HF_CACHE_DIR:/hf" \
        --entrypoint python3 "$IMAGE" -c "$DL_PY" "$MODEL_ID" "$RETRY_HINT"
    fi
else
    info "Snapshot already complete; skipping the download pass (sha256 guard still runs)."
fi

# Verify exactly what start.sh will look for, so a broken download fails here.
[[ -d "$MODEL_PATH" ]] || err "Download finished but $MODEL_PATH is missing."
SNAP=""
SNAP_RC=0
SNAP="$(resolve_snapshot "$MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
[[ -n "$SNAP" ]] || err "No snapshot directory under $MODEL_PATH/snapshots"
[[ "$SNAP_RC" -eq 0 ]] || err "Snapshot is missing one or more indexed weight shards — the download is incomplete. Rerun this script."

# ---------------------------------------------------------------------------
# sha256 verification of LFS blobs (review §5.5 / jschmied A1). aria2
# preallocates to final size, so size checks pass on corrupt content; this is
# the only catch. VERIFY_SHA256=0 documents a skip but is not the default.
# ---------------------------------------------------------------------------
if [[ "$VERIFY_SHA256" == "1" ]]; then
    SNAP_DIR="$MODEL_PATH/snapshots/$SNAP"
    STATE_FILE="$MODEL_PATH/.sha256state"
    auth=(); [[ -n "$HF_TOKEN" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

    # Fetches the full LFS metadata manifest, walking the HF tree API with
    # pagination (50 entries/page via the Link: rel="next" header). The
    # manifest is <size>\t<lfs.sha256>\t<path> per line, so a truncated fetch
    # is visible in the entry count printed below.
    fetch_manifest() {  # <cache-file>
        local cache="$1"
        rm -f "$cache"
        local page=0 total=0
        local url="https://huggingface.co/api/models/$MODEL_ID/tree/main?recursive=true&expand=true&limit=50"
        while [[ -n "$url" ]]; do
            local headers="$cache.headers.$page"
            local body; body=$(curl -s -D "$headers" -m 60 "${auth[@]}" \
                -H "Accept: application/json" "$url" || true)
            [[ -n "$body" ]] || { rm -f "$cache" "$headers"; return 1; }
            # The heredoc below is python3's stdin (python3 - reads its
            # program from stdin, so a piped body would be discarded);
            # the page body travels through a file instead.
            local bodyf; bodyf=$(mktemp)
            printf '%s' "$body" > "$bodyf"
            python3 - "$cache" "$bodyf" <<'PY'
import json, sys
cache = sys.argv[1]
with open(cache, "a") as mf:
    for e in json.loads(open(sys.argv[2]).read()):
        if e.get("type") != "file":
            continue
        lfs = e.get("lfs", {})
        # lfs.oid IS the sha256 (matches the blob name HF writes into the
        # cache); the tree API has no lfs.sha256 key.
        print(f"{e.get('size', 0)}\t{lfs.get('oid', '')}\t{e.get('path', '')}", file=mf)
PY
            rm -f "$bodyf"
            total=$(grep -c . "$cache" 2>/dev/null || echo 0)
            page=$((page + 1))
            url=""
            if [[ -f "$headers" ]]; then
                url=$(tr -d '\r' < "$headers" | grep -i '^Link:' \
                    | sed -nE 's/.*<([^>]*)>; rel="next".*/\1/p' || true)
                rm -f "$headers"
            fi
            [[ -n "$url" ]] && url=$(printf '%s' "$url" | tr -d '[:space:]')
        done
        [[ -s "$cache" ]] || return 1
        echo "$total"
        return 0
    }

    MANIFEST=$(mktemp)
    ENTRY_COUNT=0
    ENTRY_COUNT=$(fetch_manifest "$MANIFEST" 2>/dev/null || echo "")
    if [[ -z "$ENTRY_COUNT" ]]; then
        warn "sha256 verification: could not fetch the LFS manifest (offline?); skipping verification."
        warn "     Rerun with network to get the corrupt-blob guard."
    else
        info "verifying sha256: ${ENTRY_COUNT} files in the remote tree manifest"
        # No fixed minimum entry count: a small repo is legitimately small,
        # and real truncation is caught by the local-file comparison below
        # (a manifest with fewer entries than LFS-sized files on disk means
        # the pagination walk came up short).
        {
            # The remote tree metadata can only be that much larger than what a
            # complete download has on disk; fewer remote entries than local LFS
            # files means we fetched a truncated manifest (jschmied: 50 of 144
            # "verified" cleanly).
            _LOCAL_LFS=$(find "$SNAP_DIR" -type f -size +1M 2>/dev/null | wc -l)
            if (( ENTRY_COUNT < _LOCAL_LFS )); then
                err "tree manifest records ${ENTRY_COUNT} files but the snapshot has ${_LOCAL_LFS} LFS-sized files — the manifest is truncated (pagination regression). Aborting verification."
            fi
            _resume=1
            if grep -q '^complete ' "$STATE_FILE" 2>/dev/null; then
                _resume=0
                { : > "$STATE_FILE"; } 2>/dev/null || true
            fi
            # Only LFS blobs carry a sha256 handle: <size>\t<64-hex>\t<path>.
            # Everything else in the tree is content-addressed JSON/metadata.
            while IFS=$'\t' read -r _sz sha path; do
                # Manifest-controlled path must stay inside the snapshot: reject
                # traversal and anything that is not a plain relative path
                # (an integrity-compromised manifest should not turn into a
                # host-path read oracle).
                case "$path" in
                    *".."*|/*|*[[:space:]]*|*[^[:print:]]*|"")
                        err "sha256 verify: manifest path '$path' is not a safe relative path; aborting."
                        ;;
                esac
                f="$SNAP_DIR/$path"
                [[ -f "$f" ]] || err "sha256 verify: $path missing from snapshot"
                # ABLIT resume: skip blobs already verified in a prior run.
                _done=0
                if [[ "$_resume" == "1" && -f "$STATE_FILE" ]]; then
                    grep -qxF "$sha  $path" "$STATE_FILE" 2>/dev/null && _done=1
                fi
                if [[ "$_done" != "1" ]]; then
                    _have=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1 || echo "")
                    if [[ "$_have" != "$sha" ]]; then
                        err "sha256 mismatch on $path (got ${_have:-no-file}, want $sha). The checkpoint is corrupt or incomplete; remove $(readlink -f "$f") and rerun ./download.sh $MODEL_ID to fetch it again."
                    fi
                    # The state file is a resume optimization, not a gate: an
                    # unwritable cache dir (root-owned snapshot from a
                    # pre-uid-fix download) must not turn a PASSING verify
                    # into a crash mid-loop. Warn once, keep verifying.
                    if ! { printf '%s  %s\n' "$sha" "$path" >> "$STATE_FILE"; } 2>/dev/null; then
                        if [[ "${_state_warned:-0}" != "1" ]]; then
                            warn "cannot write the sha256 resume state ($STATE_FILE); verification still complete, but the next run will re-hash every blob."
                            _state_warned=1
                        fi
                    fi
                fi
            done < <(awk -F'\t' 'NF >= 3 && length($2) == 64 { print }' "$MANIFEST")
            { printf 'complete %s\n' "$SNAP" >> "$STATE_FILE"; } 2>/dev/null || true
            ok "sha256 verified all LFS blobs in snapshot $SNAP."
        }
    fi
    rm -f "$MANIFEST"
fi

ok "$MODEL_ID  ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
next_start_hint
