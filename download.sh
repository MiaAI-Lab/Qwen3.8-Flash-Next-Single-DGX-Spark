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
# The checkpoint is ~99 GB, so this takes a while and is resumable — rerun it
# after an interruption and it picks up where it stopped.
#
# Usage:
#   ./download.sh                       # the default checkpoint
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
        -h|--help) sed -n '5,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    esac
done

# Environment wins over .env, same precedence rule as start.sh.
_CLI_HF_TOKEN="${HF_TOKEN:-}"
if [[ -f .env ]]; then
    # shellcheck source=.env
    source .env
fi
[[ -n "$_CLI_HF_TOKEN" ]] && HF_TOKEN="$_CLI_HF_TOKEN"
HF_TOKEN="${HF_TOKEN:-}"

MODEL_ID="${1:-${TP1_MODEL_ID:-Mia-AiLab/Qwen3.8-Flash-Next-NVFP4}}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
ORG="${MODEL_ID%%/*}"; NAME="${MODEL_ID##*/}"
MODEL_PATH="$HF_CACHE_DIR/hub/models--${ORG}--${NAME}"

snapshot_complete() {  # <snapshot-dir>
    python3 - "$1" <<'PY'
import json
import pathlib
import sys

snapshot = pathlib.Path(sys.argv[1])
index = snapshot / "model.safetensors.index.json"
if not index.is_file():
    raise SystemExit(1)
weight_map = json.loads(index.read_text()).get("weight_map", {})
raise SystemExit(0 if weight_map and all((snapshot / name).is_file()
                                         for name in set(weight_map.values())) else 1)
PY
}

info "Model:  $MODEL_ID"
info "Cache:  $HF_CACHE_DIR"

# Already complete? Require every shard named by the safetensors index. A
# config.json appears early in a partial download and is not sufficient.
if [[ -d "$MODEL_PATH" ]]; then
    if [[ -f "$MODEL_PATH/refs/main" ]]; then
        SNAP="$(cat "$MODEL_PATH/refs/main")"
    else
        SNAP="$(ls -t "$MODEL_PATH/snapshots" 2>/dev/null | head -1 || true)"
    fi
    if [[ -n "$SNAP" ]] && snapshot_complete "$MODEL_PATH/snapshots/$SNAP"; then
        ok "Already in cache: $MODEL_PATH ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
        info "Nothing to do. Run ./start.sh next."
        exit 0
    fi
    warn "Partial download found; resuming."
fi

# ~99 GB plus room for the packed PLE table built on first launch (~27 GB).
AVAIL_GIB=$(df -BG --output=avail "$(dirname "$HF_CACHE_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [[ -n "$AVAIL_GIB" && "$AVAIL_GIB" -lt 130 ]]; then
    warn "Only ${AVAIL_GIB} GiB free on $(dirname "$HF_CACHE_DIR")."
    warn "     The checkpoint is ~99 GB and start.sh builds a ~27 GB packed PLE"
    warn "     table beside it on first launch. ~130 GiB free is the safe figure."
fi

mkdir -p "$HF_CACHE_DIR"

DL_PY='
import os, sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id=sys.argv[1],
    token=(os.environ.get("HF_TOKEN") or None),
    max_workers=4,
)
print(p)
'

info "Downloading (resumable; interrupt and rerun to continue)..."
if python3 -c "import huggingface_hub" 2>/dev/null; then
    HF_HOME="$HF_CACHE_DIR" HF_TOKEN="$HF_TOKEN" python3 -c "$DL_PY" "$MODEL_ID"
else
    info "huggingface_hub not on the host; using the container image instead."
    IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
    docker run --rm -i \
        -e HF_HOME=/hf -e HF_TOKEN="$HF_TOKEN" \
        -v "$HF_CACHE_DIR:/hf" \
        --entrypoint python3 "$IMAGE" -c "$DL_PY" "$MODEL_ID"
fi

# Verify exactly what start.sh will look for, so a broken download fails here.
[[ -d "$MODEL_PATH" ]] || err "Download finished but $MODEL_PATH is missing."
if [[ -f "$MODEL_PATH/refs/main" ]]; then
    SNAP="$(cat "$MODEL_PATH/refs/main")"
else
    SNAP="$(ls -t "$MODEL_PATH/snapshots" 2>/dev/null | head -1 || true)"
fi
[[ -n "$SNAP" ]] || err "No snapshot directory under $MODEL_PATH/snapshots"
snapshot_complete "$MODEL_PATH/snapshots/$SNAP" || err "Snapshot is missing one or more indexed weight shards — the download is incomplete. Rerun this script."

ok "$MODEL_ID  ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
info "Next:  ./start.sh"
