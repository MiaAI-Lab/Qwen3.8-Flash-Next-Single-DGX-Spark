#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# ============================================================================
# tp1/start.sh — Single-node, single-GPU (TP=1) vLLM launch on ONE DGX Spark.
#
# Serves the Mia-AiLab NVFP4 checkpoint — MXFP8 attention + a 4-bit NVFP4 PLE
# table. The memory figures below were measured on the equivalent
# local-inference-lab build (98.6 GiB on disk); re-check them if this
# checkpoint's on-disk size differs. The RadixArk build (125.9 GiB) cannot fit
# one Spark and is not offered here.
#
# ---------------------------------------------------------------------------
# HOW IT FITS (measured on this box — see docs/HANDOFF-single-spark.md)
#
#   unified pool ............ 121.69 GiB   (LPDDR5X; CPU and GPU share it)
#   checkpoint on disk ......  98.57 GiB
#     of which PLE table ....  26.82 GiB   -> NOT on the GPU (see below)
#   weights on GPU ..........  71.75 GiB
#   runtime overhead ........   5.6  GiB   (non-torch 3.37 + activation 1.92
#                                          + graphs 0.12, all measured at TP1)
#   KV cache ................  whatever GMU leaves (~7-15 GiB => 250-500k tok)
#
# The PLE n-gram table is served by vLLM's CPU-offload worker from a
# MEMORY-MAPPED pre-packed file (files/build_ple_packed_table.py, built on
# first launch, ~40 s). File-backed pages are evictable page cache, so the
# non-evictable footprint of the whole deployment is ~78 GiB + KV instead of
# ~104 GiB + KV. That margin is what keeps the host alive: exhausting the
# unified pool hangs the kernel (no OOM, no logs — three times last session).
#
# Two GB10-specific bugs in vLLM's offload path are patched in
# files/patch_ple_offload.py (CUDA stream memory ops are unsupported on GB10,
# which deadlocked the GPU worker after graph capture) and
# files/patch_ple_layer.py (offload rows must carry codes AND scales).
#
# SAFETY (no sudo needed):
#   * The container runs under a hard cgroup memory cap. Measured: GPU
#     parameter allocations are NOT charged to it on GB10, so the cap bounds
#     the host-side footprint (Python procs, pinned buffers, page cache) while
#     vLLM's own --gpu-memory-utilization budget bounds the GPU side.
#   * A background watchdog (files/memwatch.sh) kills the container if host
#     MemAvailable drops below MEMWATCH_MIN_GIB.
#   * comfy-h3.service is a bash loop that launches ComfyUI (a GPU co-tenant)
#     the moment *anything* answers on port 8888. The launcher refuses 8888
#     while that service is active (disable it: sudo systemctl disable --now
#     comfy-h3.service); with it disabled the default port is 8888.
#
# Context above the native 262144 needs YaRN. MAX_MODEL_LEN is the YARN=0
# length; YARN_MAX_MODEL_LEN (default 524288) is served instead when YARN=1.
# Both live in .env, so the 0/1 flag alone switches between them. 1M does not fit.
# ---------------------------------------------------------------------------
#
# Usage:
#   ./start.sh                  # profile from .env (262k, MTP 3, port 8888)
#   ./start.sh --no-launch      # patch + print the command, don't start
#   MAX_MODEL_LEN=262144 ./start.sh
#   MTP_NUM_SPECULATIVE_TOKENS=3 ./start.sh   # re-enable MTP (1.5 GiB)
#   YARN=1 ./start.sh                         # YARN_MAX_MODEL_LEN (512k) via YaRN
#   GPU_MEMORY_UTILIZATION=0.75 ./start.sh    # pin the budget yourself
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# Precedence: environment override > tp1/.env > built-in default.
_CLI_MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
_CLI_YARN="${YARN:-}"
_CLI_YARN_MAX_MODEL_LEN="${YARN_MAX_MODEL_LEN:-}"
_CLI_GMU="${GPU_MEMORY_UTILIZATION:-}"
_CLI_MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
_CLI_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
_CLI_MTP="${MTP_NUM_SPECULATIVE_TOKENS:-}"
_CLI_REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-}"
_CLI_PLE_OFFLOAD="${PLE_OFFLOAD:-}"
_CLI_PORT="${PORT:-}"
_CLI_KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

# Knobs that are NOT read through an explicit _CLI_ variable above still have
# to honour "environment > .env": sourcing .env would otherwise overwrite them.
# Snapshot anything set in the environment, then restore it after the source.
_ENV_SNAPSHOT_VARS=(KV_TARGET_GIB HOST_SLACK_GIB OS_RESERVE_GIB MEMWATCH_MIN_GIB
                    OVERHEAD_GIB PLE_GIB CONTAINER_MEM_GIB KV_CACHE_MEMORY
                    IMAGE SERVED_MODEL_NAME CUDAGRAPH_MODE HF_TOKEN
                    EXTRA_VLLM_ARGS EXTRA_DOCKER_ARGS NATIVE_MAX_MODEL_LEN
                    YARN_CEILING_MODEL_LEN)
for _v in "${_ENV_SNAPSHOT_VARS[@]}"; do
    eval "_SNAP_$_v=\${$_v-}"
    eval "_SNAPSET_$_v=\${$_v+set}"
done

[[ -f .env ]] || err ".env not found. Copy .env.sample to .env and edit it."
# shellcheck source=.env
source .env

for _v in "${_ENV_SNAPSHOT_VARS[@]}"; do
    if [[ -n "$(eval "printf %s \"\${_SNAPSET_$_v-}\"")" ]]; then
        eval "$_v=\$_SNAP_$_v"
    fi
done

# ---------------------------------------------------------------------------
# Defaults (see tp1/.env.sample for the known-good profile).
# ---------------------------------------------------------------------------
MODEL_ID="${TP1_MODEL_ID:-Mia-AiLab/Qwen3.8-Flash-Next-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
PORT="${_CLI_PORT:-${PORT:-8888}}"            # 8888 is safe only while comfy-h3.service is disabled (it watches this port)
IMAGE="${IMAGE:?IMAGE not set in .env}"

MAX_MODEL_LEN="${_CLI_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-65536}}"
# YaRN rope scaling: 0 = off (MAX_MODEL_LEN applies, capped at native),
#                    1 = on  (YARN_MAX_MODEL_LEN applies instead).
YARN="${_CLI_YARN:-${YARN:-0}}"
NATIVE_MAX_MODEL_LEN="${NATIVE_MAX_MODEL_LEN:-262144}"   # text_config.max_position_embeddings
# The context served when YARN=1. Ignored entirely when YARN=0, so the two
# lengths can sit side by side in .env and the 0/1 flag switches between them.
YARN_MAX_MODEL_LEN="${_CLI_YARN_MAX_MODEL_LEN:-${YARN_MAX_MODEL_LEN:-524288}}"
# Validated safety ceiling for YARN_MAX_MODEL_LEN on one Spark. 1M needs
# ~28.8 GiB of KV, which drives the cgroup cap past the pool; raise only
# after re-doing the Step 2 budget arithmetic.
YARN_CEILING_MODEL_LEN="${YARN_CEILING_MODEL_LEN:-524288}"
GPU_MEMORY_UTILIZATION="${_CLI_GMU:-${GPU_MEMORY_UTILIZATION:-}}"   # empty => derived in Step 2
MAX_NUM_SEQS="${_CLI_MAX_NUM_SEQS:-${MAX_NUM_SEQS:-4}}"
MAX_NUM_BATCHED_TOKENS="${_CLI_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-2048}}"
MTP_NUM_SPECULATIVE_TOKENS="${_CLI_MTP:-${MTP_NUM_SPECULATIVE_TOKENS:-0}}"
KV_CACHE_DTYPE="${_CLI_KV_CACHE_DTYPE:-${KV_CACHE_DTYPE:-auto}}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"          # optional hard pin, bytes
# Runtime overhead on top of weights, GiB (measured at TP1: 3.37+1.92+0.12).
OVERHEAD_GIB="${OVERHEAD_GIB:-5.6}"
# KV the derived budget targets when GMU is not pinned. More KV = more UVM.
KV_TARGET_GIB="${KV_TARGET_GIB:-8.0}"
# Host-side memory the container needs beyond the GPU budget: three Python
# processes, pinned staging buffers, CPU-side torch, page cache slack.
HOST_SLACK_GIB="${HOST_SLACK_GIB:-10.0}"
# Never let the container cgroup cap come within this much of the pool.
OS_RESERVE_GIB="${OS_RESERVE_GIB:-16.0}"
# Watchdog: kill the container if host MemAvailable drops below this.
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
PLE_OFFLOAD="${_CLI_PLE_OFFLOAD:-${PLE_OFFLOAD:-true}}"
PLE_GIB="${PLE_GIB:-26.82}"
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
REQUIRE_IDLE_GPU="${_CLI_REQUIRE_IDLE_GPU:-${REQUIRE_IDLE_GPU:-true}}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
HF_TOKEN="${HF_TOKEN:-}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"   # NONE for eager debug

DO_LAUNCH=true
for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=false ;;
        -h|--help)   sed -n '1,60p' "$0"; exit 0 ;;
        *)           err "Unknown argument: $arg (try --help)" ;;
    esac
done

if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
fi
[[ "$YARN" == "0" || "$YARN" == "1" ]] || err "YARN must be 0 or 1 (got: '$YARN')"

case "$KV_CACHE_DTYPE" in
    auto|bfloat16) ;;
    fp8|fp8_e4m3)
        warn "KV_CACHE_DTYPE=$KV_CACHE_DTYPE: FP8 KV is a CAPACITY TRADE, not a free win."
        warn "     ~1.7x more KV tokens (1M context becomes reachable). The speed cost is"
        warn "     now small here (see README), but the reference implementation measured a"
        warn "     long-reasoning benchmark falling from 6/6 to 2/6. This is sparse"
        warn "     attention: quantised keys perturb which blocks the indexer selects."
        warn "     Re-validate quality on your own workload before trusting it."
        ;;
    *) err "KV_CACHE_DTYPE must be auto, bfloat16, fp8 or fp8_e4m3 (got: '$KV_CACHE_DTYPE')" ;;
esac

# The 0/1 flag picks which length is served. YARN_FACTOR stays empty unless
# YaRN is actually applied; it is the single flag the rest of the script
# keys off.
YARN_FACTOR=""
if [[ "$YARN" == "1" ]]; then
    if ! [[ "$YARN_MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
        err "YARN_MAX_MODEL_LEN must be a positive integer (got: '$YARN_MAX_MODEL_LEN')"
    fi
    if [[ "$YARN_MAX_MODEL_LEN" -gt "$YARN_CEILING_MODEL_LEN" ]]; then
        err "YARN_MAX_MODEL_LEN=$YARN_MAX_MODEL_LEN is above YARN_CEILING_MODEL_LEN=$YARN_CEILING_MODEL_LEN,
       the validated ceiling for one Spark. A 1M context needs ~28.8 GiB of KV, which
       drives the container cap past the unified pool and hangs the host.
       Raise YARN_CEILING_MODEL_LEN only after re-doing the Step 2 arithmetic."
    fi
    if [[ "$YARN_MAX_MODEL_LEN" -le "$NATIVE_MAX_MODEL_LEN" ]]; then
        warn "YARN=1 but YARN_MAX_MODEL_LEN=$YARN_MAX_MODEL_LEN is within the native"
        warn "     $NATIVE_MAX_MODEL_LEN; serving it with native rope (nothing to scale)."
    else
        # Rounded UP so original_max * factor >= the served length and vLLM's
        # own derived-length check passes.
        YARN_FACTOR=$(python3 -c "import math
print(round(math.ceil($YARN_MAX_MODEL_LEN / $NATIVE_MAX_MODEL_LEN * 10000) / 10000, 4))")
    fi
    if [[ "$MAX_MODEL_LEN" != "$YARN_MAX_MODEL_LEN" ]]; then
        info "YARN=1: serving YARN_MAX_MODEL_LEN=$YARN_MAX_MODEL_LEN (MAX_MODEL_LEN=$MAX_MODEL_LEN applies only at YARN=0)."
    fi
    MAX_MODEL_LEN="$YARN_MAX_MODEL_LEN"
elif [[ "$MAX_MODEL_LEN" -gt "$NATIVE_MAX_MODEL_LEN" ]]; then
    err "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds the native $NATIVE_MAX_MODEL_LEN and YARN=0.
       Set YARN=1 to serve YARN_MAX_MODEL_LEN with YaRN rope scaling, or lower MAX_MODEL_LEN."
fi
[[ "$PLE_OFFLOAD" == "true" ]] || err "PLE_OFFLOAD=false cannot fit one Spark (98.6 GiB of weights through UVM hung the host last session). Refusing."

# ---------------------------------------------------------------------------
# 1. Resolve the checkpoint in the local HF cache (no download, no NFS).
# ---------------------------------------------------------------------------
info "=== Step 1: Resolve checkpoint ==="
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
ORG="${MODEL_ID%%/*}"; NAME="${MODEL_ID##*/}"
MODEL_PATH="$HF_CACHE_DIR/hub/models--${ORG}--${NAME}"
[[ -d "$MODEL_PATH" ]] || err "Checkpoint not in cache: $MODEL_PATH
       Fetch it first:  ./download.sh $MODEL_ID"
if [[ -f "$MODEL_PATH/refs/main" ]]; then
    SNAPSHOT_REL="snapshots/$(cat "$MODEL_PATH/refs/main")"
else
    SNAPSHOT_REL="snapshots/$(ls -t "$MODEL_PATH/snapshots" | head -1)"
fi
[[ -f "$MODEL_PATH/$SNAPSHOT_REL/config.json" ]] || err "No snapshot under $MODEL_PATH/snapshots"
python3 - "$MODEL_PATH/$SNAPSHOT_REL" <<'PY' || err "Checkpoint snapshot is incomplete. Resume it with: ./download.sh $MODEL_ID"
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
ok "$MODEL_ID  ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"

# ---------------------------------------------------------------------------
# 2. Co-tenant guard + memory budget.
# ---------------------------------------------------------------------------
COTENANT=$(systemctl is-active comfy-h3.service 2>/dev/null || true)
if pgrep -f "ComfyUI/main.py" >/dev/null 2>&1; then
    err "ComfyUI (comfy-h3) is RUNNING and holds GPU memory. It cannot coexist
       with this deployment on unified memory. Stop it:
         sudo systemctl stop comfy-h3.service"
fi
if [[ "$COTENANT" == "active" && "$PORT" == "8888" ]]; then
    err "comfy-h3.service is active: its launcher polls http://127.0.0.1:8888/v1/models
       and starts ComfyUI (a GPU co-tenant) as soon as it answers. Serving on
       8888 would trigger it. Either use PORT=8890 or disable the service:
         sudo systemctl disable --now comfy-h3.service"
fi
if [[ "$COTENANT" == "active" ]]; then
    warn "comfy-h3.service is active but idle (waiting on port 8888). Serving on $PORT keeps it asleep; disable it to use 8888."
fi

info "=== Step 2: Memory budget ==="
KV_BYTES_PER_TOKEN=29482          # measured: 28.8 KiB/token, bf16 KV, this arch
WEIGHT_BYTES=$(du -sb "$MODEL_PATH/$SNAPSHOT_REL/" -L | cut -f1)

read -r MEM_TOTAL_GIB MEM_AVAIL_GIB <<<"$(python3 -c "
m={l.split(':')[0]:int(l.split()[1]) for l in open('/proc/meminfo') if ':' in l}
print(m['MemTotal']/1048576, m['MemAvailable']/1048576)")"

MTP_GIB=0
[[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] && MTP_GIB=1.49
KV_MULT=1.0
# FP8 halves the main KV (12 full-attn layers, ~84% of bytes/token) but the QSA
# side/compressor caches stay BF16, so the real saving is ~1.7x, not 2x.
[[ "$KV_CACHE_DTYPE" == fp8* ]] && KV_MULT=0.58

read -r WEIGHTS_GPU_GIB KV_NEED_GIB BUDGET_GIB DERIVED_GMU KV_EXPECT_GIB KV_EXPECT_TOK <<<"$(python3 -c "
w=$WEIGHT_BYTES/2**30-$PLE_GIB
kv_need=$MAX_MODEL_LEN*$KV_BYTES_PER_TOKEN*$KV_MULT/2**30
budget=w+$OVERHEAD_GIB+$MTP_GIB+max(kv_need,$KV_TARGET_GIB)
gmu=budget/$MEM_TOTAL_GIB
kv_exp=budget-w-$OVERHEAD_GIB-$MTP_GIB
print(f'{w:.2f} {kv_need:.2f} {budget:.2f} {gmu:.3f} {kv_exp:.2f} {int(kv_exp*2**30/($KV_BYTES_PER_TOKEN*$KV_MULT))}')")"

if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    warn "  caller-pinned GMU=$GPU_MEMORY_UTILIZATION (derived would be $DERIVED_GMU)"
    read -r BUDGET_GIB KV_EXPECT_GIB KV_EXPECT_TOK <<<"$(python3 -c "
b=$GPU_MEMORY_UTILIZATION*$MEM_TOTAL_GIB
kv=b-$WEIGHTS_GPU_GIB-$OVERHEAD_GIB-$MTP_GIB
print(f'{b:.2f} {kv:.2f} {int(max(kv,0)*2**30/($KV_BYTES_PER_TOKEN*$KV_MULT))}')")"
else
    GPU_MEMORY_UTILIZATION="$DERIVED_GMU"
fi
CONTAINER_MEM_GIB="${CONTAINER_MEM_GIB:-$(python3 -c "print(int($BUDGET_GIB+$HOST_SLACK_GIB))")}"
MAX_CONTAINER_GIB=$(python3 -c "print(int($MEM_TOTAL_GIB-$OS_RESERVE_GIB))")

info "  unified pool ............. ${MEM_TOTAL_GIB%.*} GiB total, ${MEM_AVAIL_GIB%.*} GiB available now"
info "  weights on GPU ........... ${WEIGHTS_GPU_GIB} GiB  (checkpoint minus ${PLE_GIB} GiB PLE table)"
info "  PLE table ................ ${PLE_GIB} GiB  memory-mapped in the CPU offload worker"
info "  runtime overhead ......... ${OVERHEAD_GIB} GiB"
[[ "$MTP_GIB" != 0 ]] && info "  MTP draft model .......... ${MTP_GIB} GiB"
info "  KV needed for ${MAX_MODEL_LEN} ...... ${KV_NEED_GIB} GiB  (kv dtype ${KV_CACHE_DTYPE})"
info "  GPU budget (GMU ${GPU_MEMORY_UTILIZATION}) ... ${BUDGET_GIB} GiB  => ~${KV_EXPECT_GIB} GiB KV (~${KV_EXPECT_TOK} tokens)"
info "  container cgroup cap ..... ${CONTAINER_MEM_GIB} GiB  (hard ceiling ${MAX_CONTAINER_GIB})"

if python3 -c "import sys; sys.exit(0 if $KV_EXPECT_GIB < $KV_NEED_GIB else 1)"; then
    err "Budget leaves ${KV_EXPECT_GIB} GiB for KV but ${MAX_MODEL_LEN} tokens need ${KV_NEED_GIB} GiB.
       Lower MAX_MODEL_LEN or raise GPU_MEMORY_UTILIZATION (FP8 KV is unsupported by this model)."
fi
if [[ "$CONTAINER_MEM_GIB" -gt "$MAX_CONTAINER_GIB" ]]; then
    err "Container cap ${CONTAINER_MEM_GIB} GiB exceeds the hard ceiling ${MAX_CONTAINER_GIB} GiB
       (pool ${MEM_TOTAL_GIB%.*} GiB minus OS_RESERVE_GIB=${OS_RESERVE_GIB}). On unified memory
       this is the line between a killed container and a hung host. Lower the budget."
fi
if $DO_LAUNCH && python3 -c "import sys; sys.exit(0 if $MEM_AVAIL_GIB < $CONTAINER_MEM_GIB+4 else 1)"; then
    err "Only ${MEM_AVAIL_GIB%.*} GiB available now but the container may use ${CONTAINER_MEM_GIB} GiB.
       Something else is holding memory (docker ps; ps --sort=-rss)."
fi
ok "  budget fits."

# ---------------------------------------------------------------------------
# 3. GPU preflight
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ "$REQUIRE_IDLE_GPU" == "true" ]]; then
    info "=== Step 3: GPU preflight ==="
    TENANTS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
              --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)
    if [[ -n "$TENANTS" ]]; then
        echo "$TENANTS"
        err "GPU is in use. Stop the 2-node server first (./stop.sh), or set REQUIRE_IDLE_GPU=false."
    fi
    ok "GPU idle."
fi

# ---------------------------------------------------------------------------
# 4. Patches + packed PLE table
# ---------------------------------------------------------------------------
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
PLE_PKG="$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ple_layer.py"
MODELOPT_PKG="$VLLM_PKG/model_executor/layers/quantization/modelopt.py"
QSA_OPS_PKG="$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
QSA_NVIDIA_PKG="$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py"

info "=== Step 4: Prepare patches ==="
if ! docker image inspect "$IMAGE" &>/dev/null; then
    info "Pulling $IMAGE ..."
    docker pull "$IMAGE"
fi

extract() {  # <path-in-image> <dest>
    if [[ ! -f "$2" ]]; then
        info "Extracting $(basename "$1") from image..."
        local tmp; tmp=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp:$1" "$2"
        docker rm "$tmp" >/dev/null 2>&1
    fi
}
PATCHED_PLE="$SCRIPT_DIR/files/ple_layer_patched.py"
extract "$PLE_PKG" "$SCRIPT_DIR/files/ple_layer_patched.py.orig"
python3 "$SCRIPT_DIR/files/patch_ple_layer.py"
[[ -f "$PATCHED_PLE" ]] || err "PLE patch missing after patch_ple_layer.py"

PATCHED_MODELOPT="$SCRIPT_DIR/files/modelopt_patched.py"
extract "$MODELOPT_PKG" "$SCRIPT_DIR/files/modelopt_patched.py.orig"
python3 "$SCRIPT_DIR/files/patch_modelopt_mxfp8.py"
[[ -f "$PATCHED_MODELOPT" ]] || err "modelopt patch missing after patch_modelopt_mxfp8.py"

# FP8 KV support for the QSA kernels. The patch is compiled out when the KV
# cache is BF16, so it is applied unconditionally and costs nothing at KV_CACHE_DTYPE=auto.
PATCHED_QSA_OPS="$SCRIPT_DIR/files/qsa_ops_patched.py"
PATCHED_QSA_NVIDIA="$SCRIPT_DIR/files/qsa_nvidia_patched.py"
extract "$QSA_OPS_PKG"    "$PATCHED_QSA_OPS.orig"
extract "$QSA_NVIDIA_PKG" "$PATCHED_QSA_NVIDIA.orig"
python3 "$SCRIPT_DIR/files/patch_qsa_fp8_kv.py"
[[ -f "$PATCHED_QSA_OPS" && -f "$PATCHED_QSA_NVIDIA" ]] || err "QSA fp8 patch missing after patch_qsa_fp8_kv.py"

OFFLOAD_DIR="$SCRIPT_DIR/files/ple_offload"
mkdir -p "$OFFLOAD_DIR/orig"
extract "$VLLM_PKG/model_executor/layers/ple_offload_layer.py" "$OFFLOAD_DIR/orig/ple_offload_layer.py"
for f in connector worker protocol; do
    extract "$VLLM_PKG/v1/ple_offload/$f.py" "$OFFLOAD_DIR/orig/$f.py"
done
python3 "$SCRIPT_DIR/files/patch_ple_offload.py"
for f in ple_offload_layer connector worker protocol; do
    [[ -f "$OFFLOAD_DIR/$f.py" ]] || err "offload patch missing: $f.py"
done
ok "Patches ready."

PLE_CACHE_HOST="$HOME/.cache/vllm/ple_cache/${ORG}--${NAME}"
PLE_CACHE_CTR="/root/.cache/vllm/ple_cache/${ORG}--${NAME}"
if ! ls "$PLE_CACHE_HOST"/*.packed_u8 >/dev/null 2>&1; then
    info "Building packed PLE table (one-time, ~40 s, <1 GiB RAM, no GPU)..."
    mkdir -p "$PLE_CACHE_HOST"
    docker run --rm --name "${CONTAINER_NAME}-plebuild" --memory 6g --cpus 8 \
        -v "$MODEL_PATH:/m:ro" -v "$HOME/.cache/vllm/ple_cache:/out" \
        -v "$SCRIPT_DIR/files/build_ple_packed_table.py:/b.py:ro" \
        --entrypoint python3 "$IMAGE" -u /b.py "/m/$SNAPSHOT_REL" "/out/${ORG}--${NAME}"
fi
ok "Packed PLE table: $(ls "$PLE_CACHE_HOST"/*.packed_u8 | head -1) ($(du -sh "$PLE_CACHE_HOST" | cut -f1))"

# ---------------------------------------------------------------------------
# 5. Build vLLM args.
# ---------------------------------------------------------------------------
VLLM_ARGS=()
VLLM_ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")
VLLM_ARGS+=("--tensor-parallel-size" "1")
VLLM_ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
VLLM_ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
VLLM_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
VLLM_ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
if [[ -n "$YARN_FACTOR" ]]; then
    # Deep-merged into text_config.rope_parameters, which is what this model
    # reads (nvidia/qsa.py) and what vLLM's max-len check scales by. The
    # existing mrope_section / rope_theta / partial_rotary_factor survive.
    VLLM_ARGS+=("--hf-overrides" "$(printf '{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":%s,\"original_max_position_embeddings\":%s}}}' "$YARN_FACTOR" "$NATIVE_MAX_MODEL_LEN")")
fi
VLLM_ARGS+=("--load-format" "safetensors")
VLLM_ARGS+=("--safetensors-load-strategy" "lazy")
VLLM_ARGS+=("--enable-chunked-prefill")
VLLM_ARGS+=("--reasoning-parser" "qwen3")
VLLM_ARGS+=("--enable-auto-tool-choice")
VLLM_ARGS+=("--tool-call-parser" "qwen3_coder")
# REQUIRED for PLE offload: only multiproc_executor spawns the offload worker.
VLLM_ARGS+=("--distributed-executor-backend" "mp")
[[ -n "$KV_CACHE_MEMORY" ]] && VLLM_ARGS+=("--kv-cache-memory" "$KV_CACHE_MEMORY")
if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
    VLLM_ARGS+=("--speculative-config" "$(printf '{\"method\":\"mtp\",\"num_speculative_tokens\":%s}' "$MTP_NUM_SPECULATIVE_TOKENS")")
fi
VLLM_ARGS+=("--compilation-config" "$(printf '{\"mode\":0,\"cudagraph_mode\":\"%s\"}' "$CUDAGRAPH_MODE")")
if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    # Word-split like a shell command line (e.g. "--api-key k --disable-log-requests").
    read -ra _EXTRA_VLLM <<< "$EXTRA_VLLM_ARGS"
    VLLM_ARGS+=("${_EXTRA_VLLM[@]}")
fi
VLLM_ARGS_STR="${VLLM_ARGS[*]}"

info ""
info "Config (single Spark, TP=1):"
info "  Model:      $MODEL_ID"
info "  Image:      $IMAGE"
if [[ -n "$YARN_FACTOR" ]]; then
info "  Context:    $MAX_MODEL_LEN tokens (YaRN factor $YARN_FACTOR over native $NATIVE_MAX_MODEL_LEN)"
else
info "  Context:    $MAX_MODEL_LEN tokens (native rope, no YaRN)"
fi
info "  GMU:        $GPU_MEMORY_UTILIZATION  (budget ${BUDGET_GIB} GiB, cgroup cap ${CONTAINER_MEM_GIB} GiB)"
info "  Max seqs:   $MAX_NUM_SEQS   Batched tokens: $MAX_NUM_BATCHED_TOKENS   KV dtype: $KV_CACHE_DTYPE"
info "  MTP:        $MTP_NUM_SPECULATIVE_TOKENS $( [[ "$MTP_NUM_SPECULATIVE_TOKENS" -eq 0 ]] && echo '(disabled)')"
info "  Graphs:     $CUDAGRAPH_MODE"
info "  Port:       $PORT"
info ""

LAUNCH_SCRIPT=$(mktemp /tmp/vllm_tp1_XXXXXX.sh)
# Note: no HF_TOKEN in the generated script — the server runs offline
# (HF_HUB_OFFLINE=1) and the token value must not persist in .last_launch.sh.
# download.sh passes it explicitly to the fetch container when needed.
cat > "$LAUNCH_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \\
    -d --name $CONTAINER_NAME \\
    --gpus all --network host --ipc host \\
    --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \\
    --memory ${CONTAINER_MEM_GIB}g --memory-swap ${CONTAINER_MEM_GIB}g \\
    -e HF_HUB_OFFLINE=1 \\
    -e TRANSFORMERS_OFFLINE=1 \\
    -e VLLM_PLE_CPU_OFFLOAD=1 \\
    -e VLLM_PLE_PACKED_TABLE_DIR=$PLE_CACHE_CTR \\
    -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \\
    -e HF_HOME=/root/.cache/huggingface \\
    -v $PATCHED_PLE:$PLE_PKG:ro \\
    -v $PATCHED_MODELOPT:$MODELOPT_PKG:ro \\
    -v $PATCHED_QSA_OPS:$QSA_OPS_PKG:ro \\
    -v $PATCHED_QSA_NVIDIA:$QSA_NVIDIA_PKG:ro \\
    -v $OFFLOAD_DIR/ple_offload_layer.py:$VLLM_PKG/model_executor/layers/ple_offload_layer.py:ro \\
    -v $OFFLOAD_DIR/connector.py:$VLLM_PKG/v1/ple_offload/connector.py:ro \\
    -v $OFFLOAD_DIR/worker.py:$VLLM_PKG/v1/ple_offload/worker.py:ro \\
    -v $OFFLOAD_DIR/protocol.py:$VLLM_PKG/v1/ple_offload/protocol.py:ro \\
    -v $HF_CACHE_DIR:/root/.cache/huggingface \\
    -v $HOME/.cache/vllm:/root/.cache/vllm \\
    $EXTRA_DOCKER_ARGS \\
    $IMAGE \\
    $MODEL_ID \\
    $VLLM_ARGS_STR \\
    --host 0.0.0.0 \\
    --port $PORT
LAUNCH_EOF
chmod +x "$LAUNCH_SCRIPT"
cp "$LAUNCH_SCRIPT" "$SCRIPT_DIR/.last_launch.sh"

if ! $DO_LAUNCH; then
    info "--no-launch: command written to .last_launch.sh"
    cat "$SCRIPT_DIR/.last_launch.sh"
    rm -f "$LAUNCH_SCRIPT"
    exit 0
fi

# ---------------------------------------------------------------------------
# 6. Launch + watchdog
# ---------------------------------------------------------------------------
info "=== Step 6: Launch ==="
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/vllm"
bash "$LAUNCH_SCRIPT"
rm -f "$LAUNCH_SCRIPT"
ok "Container $CONTAINER_NAME started."

# Kill the previous watchdog (if any) and start a fresh one.
pkill -f "memwatch.sh $CONTAINER_NAME" 2>/dev/null || true
mkdir -p "$SCRIPT_DIR/logs"
nohup bash "$SCRIPT_DIR/files/memwatch.sh" "$CONTAINER_NAME" "$MEMWATCH_MIN_GIB" \
    > "$SCRIPT_DIR/logs/memwatch-${CONTAINER_NAME}.log" 2>&1 &
ok "Watchdog running (kills container if MemAvailable < ${MEMWATCH_MIN_GIB} GiB): logs/memwatch-${CONTAINER_NAME}.log"
info "Loading weights (~3-4 min). Following logs until ready..."

docker logs -f "$CONTAINER_NAME" &
LOGPID=$!
while true; do
    sleep 10
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
        kill $LOGPID 2>/dev/null || true
        echo ""
        REASON=$(docker logs "$CONTAINER_NAME" 2>&1 \
                 | grep -oE "(ValueError|RuntimeError|TimeoutError|torch\.[A-Za-z]*Error): .*" \
                 | grep -viE "min_frames|max_frames" | tail -1 | cut -c1-400)
        [[ -n "$REASON" ]] && { echo "  vLLM reported:"; echo "    $REASON"; }
        if docker inspect "$CONTAINER_NAME" --format '{{.State.OOMKilled}}' 2>/dev/null | grep -q true; then
            echo "  Container was OOM-killed by its cgroup cap (${CONTAINER_MEM_GIB} GiB) — the host survived as designed."
        fi
        err "Container exited. Full logs: docker logs $CONTAINER_NAME"
    fi
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo "000")
    if [[ "$CODE" == "200" ]]; then
        kill $LOGPID 2>/dev/null || true
        echo ""
        ok "vLLM ready on port $PORT (TP=1, single Spark)."
        docker logs "$CONTAINER_NAME" 2>&1 | grep -iE "GPU KV cache size|Available KV cache|Maximum concurrency" | tail -3 || true
        info ""
        info "Stop:  ./stop.sh   (graceful; --force to skip the SIGTERM wait)"
        break
    fi
done
