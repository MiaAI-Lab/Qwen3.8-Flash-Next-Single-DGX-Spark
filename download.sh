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
if [[ -d "$MODEL_PATH" ]]; then
    SNAP=""
    SNAP_RC=0
    SNAP="$(resolve_snapshot "$MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
    if [[ "$SNAP_RC" -eq 0 && -n "$SNAP" ]]; then
        ok "Already in cache: $MODEL_PATH ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
        info "Nothing to do."
        next_start_hint
        exit 0
    fi
    if [[ -n "$SNAP" ]]; then
        warn "Partial download found (snapshot $SNAP); resuming."
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

info "Downloading (resumable; interrupt and rerun to continue)..."
if python3 -c "import huggingface_hub" 2>/dev/null; then
    HF_HOME="$HF_CACHE_DIR" HF_TOKEN="$HF_TOKEN" python3 -c "$DL_PY" "$MODEL_ID" "$RETRY_HINT"
else
    info "huggingface_hub not on the host; using the container image instead."
    IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
    docker run --rm -i \
        -e HF_HOME=/hf -e HF_TOKEN="$HF_TOKEN" \
        -v "$HF_CACHE_DIR:/hf" \
        --entrypoint python3 "$IMAGE" -c "$DL_PY" "$MODEL_ID" "$RETRY_HINT"
fi

# Verify exactly what start.sh will look for, so a broken download fails here.
[[ -d "$MODEL_PATH" ]] || err "Download finished but $MODEL_PATH is missing."
SNAP=""
SNAP_RC=0
SNAP="$(resolve_snapshot "$MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
[[ -n "$SNAP" ]] || err "No snapshot directory under $MODEL_PATH/snapshots"
[[ "$SNAP_RC" -eq 0 ]] || err "Snapshot is missing one or more indexed weight shards — the download is incomplete. Rerun this script."

ok "$MODEL_ID  ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
next_start_hint
