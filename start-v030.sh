#!/usr/bin/env bash
# start-v030.sh - same single-Spark launch as start.sh on stock vLLM 0.30.0,
# serving nvidia/Qwen3.8-Flash-Next-NVFP4.
# Usage: ./start-v030.sh [--no-launch]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export V030=true
export IMAGE=vllm/vllm-openai:v0.30.0
export TP1_MODEL_ID="${TP1_MODEL_ID:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
export PLE_GIB="${PLE_GIB:-47.68}"
export MTP_WEIGHTS_GIB="${MTP_WEIGHTS_GIB:-2.34}"
export MTP_DISABLE_BLOCK_DROP="${MTP_DISABLE_BLOCK_DROP:-1}"
export V030_KV_GIB="${V030_KV_GIB:-12}"
export MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-3}"
export MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-1}"
export CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES-}"

exec "$SCRIPT_DIR/start.sh" "$@"
