#!/usr/bin/env bash
# start-v030.sh - same single-Spark launch as start.sh on stock vLLM 0.30.0,
# serving nvidia/Qwen3.8-Flash-Next-NVFP4.
# Usage: ./start-v030.sh [--no-launch]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# V030_KV_GIB deliberately NOT defaulted here: start-v030.sh's own export
# would ride through start.sh's env-vs-.env snapshot as an environment value
# and override .env. Let .env or the caller set it; start.sh defaults to 12.
export V030=true
export IMAGE=vllm/vllm-openai:v0.30.0
export TP1_MODEL_ID="${TP1_MODEL_ID:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
export PLE_GIB="${PLE_GIB:-47.68}"
export MTP_WEIGHTS_GIB="${MTP_WEIGHTS_GIB:-2.34}"
export MTP_DISABLE_BLOCK_DROP="${MTP_DISABLE_BLOCK_DROP:-1}"
# V030_KV_GIB, MEMWATCH_MIN_GIB, MEMWATCH_MIN_FREE_GIB: NOT defaulted here.
# An export from this wrapper rides through start.sh's env-vs-.env snapshot
# as an environment value and always overrides .env. start.sh owns their
# defaults (V030_KV_GIB=12; memwatch floors 3/1) and .env can set all three.
export CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES-}"

exec "$SCRIPT_DIR/start.sh" "$@"
