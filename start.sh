#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# ============================================================================
# tp1/start.sh — Single-node, single-GPU (TP=1) vLLM launch on ONE DGX Spark.
#
# Serves the Mia-AiLab NVFP4 checkpoint — MXFP8 attention + a 4-bit NVFP4 PLE
# table. ABLIT=1 in .env switches to the gated Keys checkpoint
# (drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only): same Mia
# 34-shard layout, QSA self_attn.o_proj replaced at L15/19/23/27/31/35/39/43/47.
# That repo is gated — accept the Hugging Face terms, then ABLIT=1 ./download.sh.
# The memory figures below were measured on the equivalent
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
#   KV cache ................  what the host-side cap leaves (~16 GiB at the
#                                default HOST_RESERVE_GIB=26; FP8 => ~1M tok)
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
#   * The GPU budget is capped FROM THE HOST SIDE: GMU x MemTotal never exceeds
#     MemTotal - HOST_RESERVE_GIB (default 26). vLLM treats this integrated
#     GPU's "free memory" as MemAvailable (page cache included) and fills the
#     GPU side to exactly the budget, so a KV_TARGET_GIB wish that is not
#     capped comes straight out of the PLE page cache and the free pages the
#     NVIDIA driver needs. That is what killed three servers on 2026-09-04
#     (docs/memory-incident-2026-09-04.md section 7). The reserve covers, in
#     order: other containers and sessions (~7 GiB measured here), vLLM's own
#     host-side processes (~6), PLE page cache (>=6), the driver's free-page
#     reserve (>=3), and 2-3 GiB of per-request growth that is never returned.
#   * The container runs under a hard cgroup memory cap. Measured: GPU
#     parameter allocations are NOT charged to it on GB10, so the cap bounds
#     the host-side footprint (Python procs, pinned buffers, page cache) while
#     vLLM's own --gpu-memory-utilization budget bounds the GPU side. It does
#     not protect the host from the GPU side; HOST_RESERVE_GIB does.
#   * A background watchdog (files/memwatch.sh) stops the container if host
#     MemAvailable stays below MEMWATCH_MIN_GIB or MemFree stays below
#     MEMWATCH_MIN_FREE_GIB, archiving the container log first.
#   * comfy-h3.service is a bash loop that launches ComfyUI (a GPU co-tenant)
#     the moment *anything* answers on port 8888. The launcher refuses 8888
#     while that service is active (disable it: sudo systemctl disable --now
#     comfy-h3.service); with it disabled the default port is 8888.
#
# Context above the native 262144 needs YaRN. MAX_MODEL_LEN is the YARN=0
# length; YARN_MAX_MODEL_LEN (default 524288) is served instead when YARN=1.
# Both live in .env, so the 0/1 flag alone switches between them. 1M does not fit.
# ABLIT=0/1 likewise switches the checkpoint: 0 is stock Mia NVFP4, 1 is the
# gated Keys ablit snapshot (accept Hugging Face terms, then ./download.sh).
# ---------------------------------------------------------------------------
#
# Usage:
#   ./start.sh                  # profile from .env (262k, MTP 3, port 8888)
#   ./start.sh --no-launch      # patch + print the command, don't start
#   MAX_MODEL_LEN=262144 ./start.sh
#   MTP_NUM_SPECULATIVE_TOKENS=3 ./start.sh   # re-enable MTP (1.5 GiB)
#   YARN=1 ./start.sh                         # YARN_MAX_MODEL_LEN (512k) via YaRN
#   ABLIT=1 ./start.sh                        # gated ablit checkpoint (download first)
#   GPU_MEMORY_UTILIZATION=0.75 ./start.sh    # pin the budget yourself
#   HOST_RESERVE_GIB=28 ./start.sh            # more host margin, less KV
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
_CLI_ABLIT="${ABLIT:-}"
_CLI_YARN_MAX_MODEL_LEN="${YARN_MAX_MODEL_LEN:-}"
_CLI_GMU="${GPU_MEMORY_UTILIZATION:-}"
_CLI_MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
_CLI_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
_CLI_MTP="${MTP_NUM_SPECULATIVE_TOKENS:-}"
_CLI_REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-}"
_CLI_PLE_OFFLOAD="${PLE_OFFLOAD:-}"
_CLI_PORT="${PORT:-}"
_CLI_KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
_CLI_BIND="${BIND:-}"
_CLI_READY_TIMEOUT_S="${READY_TIMEOUT_S:-}"

# Knobs that are NOT read through an explicit _CLI_ variable above still have
# to honour "environment > .env": sourcing .env would otherwise overwrite them.
# Snapshot anything set in the environment, then restore it after the source.
_ENV_SNAPSHOT_VARS=(KV_TARGET_GIB HOST_RESERVE_GIB HOST_SLACK_GIB OS_RESERVE_GIB
                    MEMWATCH_MIN_GIB MEMWATCH_MIN_FREE_GIB MEMWATCH_FREE_GATE_GIB MEMWATCH_GRACE
                    OVERHEAD_GIB PLE_GIB CONTAINER_MEM_GIB KV_CACHE_MEMORY
                    MAMBA_SSM_CACHE_DTYPE
                    IMAGE SERVED_MODEL_NAME CUDAGRAPH_MODE HF_TOKEN
                    CUDAGRAPH_CAPTURE_SIZES COMPILATION_MODE MTP_K_SCHEDULE
                    MTP_DRAFT_VOCAB
                    EXTRA_VLLM_ARGS EXTRA_DOCKER_ARGS NATIVE_MAX_MODEL_LEN
                    YARN_CEILING_MODEL_LEN BIND READY_TIMEOUT_S
                    VLLM_QSA_DET_TOPK VLLM_MOE_DET_FINALIZE GDN_DECODE_KERNEL
                    MTP_DISABLE_BLOCK_DROP)
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
STOCK_MODEL_ID="Mia-AiLab/Qwen3.8-Flash-Next-NVFP4"
ABLIT_MODEL_ID="drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only"
# Abliterated weights: 0 = stock Mia NVFP4, 1 = gated Keys checkpoint
# (QSA o_proj L15/19/23/27/31/35/39/43/47). Same 0/1 pattern as YARN.
# TP1_MODEL_ID, if set, still wins and ABLIT is ignored for selection.
ABLIT="${_CLI_ABLIT:-${ABLIT:-0}}"
[[ "$ABLIT" == "0" || "$ABLIT" == "1" ]] || err "ABLIT must be 0 or 1 (got: '$ABLIT')"
if [[ -n "${TP1_MODEL_ID:-}" ]]; then
    MODEL_ID="$TP1_MODEL_ID"
    if [[ "$ABLIT" == "1" && "$MODEL_ID" != "$ABLIT_MODEL_ID" ]]; then
        warn "ABLIT=1 ignored for checkpoint selection: TP1_MODEL_ID=$MODEL_ID"
    fi
elif [[ "$ABLIT" == "1" ]]; then
    MODEL_ID="$ABLIT_MODEL_ID"
else
    MODEL_ID="$STOCK_MODEL_ID"
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
PORT="${_CLI_PORT:-${PORT:-8888}}"            # 8888 is safe only while comfy-h3.service is disabled (it watches this port)
IMAGE="${IMAGE:?IMAGE not set in .env}"
# Interface the API binds to. Default is loopback: remote clients get
# connection refused until they set BIND=0.0.0.0 (with an API key) or use an
# ssh tunnel. See the README migration note.
BIND="${_CLI_BIND:-${BIND:-127.0.0.1}}"
# BIND flows raw into the generated launch script --host argument. An
# attacker-writable .env could turn it into a shell injection; reject the
# shell-metacharacter surface — including newline/control bytes, which would
# otherwise split the heredoc into new shell statements.
for _ch in '"' "'" ';' '$' '`' '\' '|' '<' '>' '&' '(' ')' '{' '}' ' ' '*' $'\n' $'\t' $'\r'; do
    if [[ "$BIND" == *"$_ch"* ]]; then
        err "BIND='$BIND' contains a shell metacharacter; refusing to use it."
    fi
done
# Cold start is ~11 min; the first boot additionally builds the ~27 GB packed
# PLE table. Give the readiness loop this long before it archives + removes
# the wedged container and exits non-zero for the supervisor to retry.
READY_TIMEOUT_S="${_CLI_READY_TIMEOUT_S:-${READY_TIMEOUT_S:-1800}}"

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
# dtype of the GDN recurrent (SSM) state. The checkpoint asks for float32; the
# fused GDN kernel also accepts bfloat16 (FUSED_GDN_STATE_DTYPES in
# qwen_gdn_linear_attn.py). BF16 halves the ~0.23 GB per sequence the state
# costs to read and write every step, and halves the mamba page, which lets
# vLLM pick a smaller attention block. Empty keeps the checkpoint's float32.
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-}"
# Runtime overhead on top of weights, GiB (measured at TP1: 3.37+1.92+0.12).
OVERHEAD_GIB="${OVERHEAD_GIB:-5.6}"
# KV the derived budget targets when GMU is not pinned. More KV = more UVM.
# Capped from the host side by HOST_RESERVE_GIB below; the cap wins.
KV_TARGET_GIB="${KV_TARGET_GIB:-8.0}"
# Memory the GPU budget may never take: the GPU side is capped at
# MemTotal - HOST_RESERVE_GIB whatever KV_TARGET_GIB asks for. See SAFETY above
# for what the 26 GiB covers. Raise it by 2 GiB steps if the watchdog log shows
# MemAvailable idling under ~9 GiB; do not lower it to buy KV.
HOST_RESERVE_GIB="${HOST_RESERVE_GIB:-26}"
# Host-side memory the container needs beyond the GPU budget: three Python
# processes, pinned staging buffers, CPU-side torch, page cache slack.
HOST_SLACK_GIB="${HOST_SLACK_GIB:-10.0}"
# Never let the container cgroup cap come within this much of the pool.
OS_RESERVE_GIB="${OS_RESERVE_GIB:-16.0}"
# Watchdog: stop the container if host MemAvailable stays below this (GiB) ...
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
# ... or MemFree stays below this. The NVIDIA driver refuses allocations
# (NV_ERR_NO_MEMORY) at MemFree ~3 GiB while MemAvailable still reads 6+.
MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
# The MemFree floor only counts while MemAvailable is under this: with stock
# kernel watermarks MemFree sits near zero whenever the page cache is full of
# reclaimable data (measured: 0.9 GiB free, 32 GiB available, during load).
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
# Seconds the watchdog gives vLLM to exit on SIGTERM before SIGKILL.
MEMWATCH_GRACE="${MEMWATCH_GRACE:-30}"
PLE_OFFLOAD="${_CLI_PLE_OFFLOAD:-${PLE_OFFLOAD:-true}}"
# PLE_GIB is derived from the checkpoint's own PLE shard sizes, not from a
# constant: it subtracts the PLE shards from the on-disk checkpoint size to
# get GPU-resident weights (start.sh:448). The shipped stock and ablit
# snapshots are both 26.82 today; the derivation keeps a future checkpoint
# change from silently mis-sizing the budget. An explicit PLE_GIB override
# still wins.
PLE_GIB="${PLE_GIB:-}"
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
REQUIRE_IDLE_GPU="${_CLI_REQUIRE_IDLE_GPU:-${REQUIRE_IDLE_GPU:-true}}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
HF_TOKEN="${HF_TOKEN:-}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"   # NONE for eager debug
# CUDA graph capture sizes for decode. vLLM's default list is [1,2,4] plus
# multiples of 8, each rounded up to a multiple of (1+MTP) and then filtered to
# <= (1+MTP)*MAX_NUM_SEQS before it becomes a decode key. At MTP=3,
# MAX_NUM_SEQS=5 that leaves keys {4,8,16}: a full 5-sequence verify batch is 20
# tokens, matches nothing, and decodes eager. "auto" captures every
# (1+MTP)*S for S in 1..MAX_NUM_SEQS so every batch the scheduler can build has
# a graph; a comma list sets them explicitly; empty keeps the vLLM default.
# Capture costs ~1 s and a few MiB per size.
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-}"
# Batch-size schedule for the speculative token count, as
# "start:end:K,start:end:K" over inclusive batch-size (num_seqs) ranges. MTP
# multiplies tokens per step by 1+K, and every extra token in a verify batch
# drags ~10 more of the 512 experts into the step, so past a few concurrent
# sequences drafting costs more expert traffic than it returns. Empty keeps a
# constant MTP_NUM_SPECULATIVE_TOKENS at every batch size.
# Example: "1:2:3,3:6:2,7:999:0"
MTP_K_SCHEDULE="${MTP_K_SCHEDULE:-}"
# Reduced-vocabulary drafting (FR-Spec). Path on the host to a file of token
# ids, one per line, built by files/build_draft_vocab.py from a corpus of the
# model's own output. The MTP drafter reads a 1.27 GB BF16 lm_head over the
# full 248,320-token vocabulary once per draft step, three of the four
# lm_head reads in an MTP-3 engine step; a 32k-row slice is 0.16 GB. Drafts
# for tokens outside the subset are simply rejected at verification, so this
# trades acceptance for bandwidth and cannot change what the server emits.
# Empty disables it and the drafter keeps the full head. Relative paths are
# resolved against this script's directory, so the shipped
# files/draft_vocab_en_code_47k.txt works out of the box.
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-}"
if [[ -n "$MTP_DRAFT_VOCAB" && "$MTP_DRAFT_VOCAB" != /* ]]; then
    MTP_DRAFT_VOCAB="$SCRIPT_DIR/$MTP_DRAFT_VOCAB"
fi
if [[ -n "$MTP_DRAFT_VOCAB" && ! -f "$MTP_DRAFT_VOCAB" ]]; then
    err "MTP_DRAFT_VOCAB=$MTP_DRAFT_VOCAB does not exist. Empty disables reduced-vocabulary drafting."
fi
if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 && -z "$MTP_DRAFT_VOCAB" ]]; then
    warn "  MTP on with the full 248k draft head: reduced-vocabulary drafting is off."
    warn "  Set MTP_DRAFT_VOCAB (shipped default: files/draft_vocab_en_code_47k.txt)"
    warn "  for ~17% faster single-stream decode at unchanged accuracy (see CHANGELOG 2026-09-05)."
fi
# torch.compile level: 0 = none (shipped default), 3 = VLLM_COMPILE (Inductor
# fusion; adds minutes to the first launch and has not been validated against
# the PLE custom op here).
COMPILATION_MODE="${COMPILATION_MODE:-0}"
# Determinism env pass-through (review §5.15 / §6.2): VLLM_QSA_DET_TOPK needs
# a compiled kernel .so and VLLM_MOE_DET_FINALIZE needs the FlashInfer
# autotune cache-key backport — neither can ship as an env var alone. This is
# plumbing only: default unset, and the day the image carries the kernels the
# flags work. Unknown env vars are ignored harmlessly by older vLLM.
VLLM_QSA_DET_TOPK="${VLLM_QSA_DET_TOPK:-}"
VLLM_MOE_DET_FINALIZE="${VLLM_MOE_DET_FINALIZE:-}"
# GDN decode kernel (review §5.16): the default CUDA kernel deterministically
# hangs the engine at c≈32 with FP8 GDN projections. Shipped default is UNSET
# for one release (so the knob exists and README documents the stall); flip to
# triton in a later release only after a soak, so there is a bisectable state.
GDN_DECODE_KERNEL="${GDN_DECODE_KERNEL:-}"
# disable_eagle_block_drop (plan 2.4 / review §6.1): speculative-config lever
# that removes MTP's fixed prefix-cache-block back-off per turn. MTP_NUM_...
# > 0 and this knob = merge into the speculative-config JSON.
MTP_DISABLE_BLOCK_DROP="${MTP_DISABLE_BLOCK_DROP:-0}"

DO_LAUNCH=true
for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=false ;;
        -h|--help)   sed -n '1,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 0 ;;
        *)           err "Unknown argument: $arg (try --help)" ;;
    esac
done

if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
fi
[[ "$YARN" == "0" || "$YARN" == "1" ]] || err "YARN must be 0 or 1 (got: '$YARN')"
if [[ "$ABLIT" == "1" ]]; then
    warn "ABLIT=1: serving gated Keys checkpoint ($ABLIT_MODEL_ID)."
    warn "     Safety refusals are removed. MTP, PLE, experts and the chat template stay stock."
    warn "     Compatible ONLY with the Mia single-Spark NVFP4 layout (this recipe)."
fi

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
if [[ "$ABLIT" == "1" ]]; then
    [[ -d "$MODEL_PATH" ]] || err "Checkpoint not in cache: $MODEL_PATH
       Fetch it first (HF_TOKEN required):
         1. Set HF_TOKEN in .env (or: export HF_TOKEN=hf_...)
         2. Open https://huggingface.co/$ABLIT_MODEL_ID
         3. Accept the terms on that page
         4. ABLIT=1 ./download.sh"
else
    [[ -d "$MODEL_PATH" ]] || err "Checkpoint not in cache: $MODEL_PATH
       Fetch it first:  ./download.sh $MODEL_ID"
fi

# Prints a snapshot hash. Exit 0 = complete, 1 = incomplete, 2 = none.
# Prefers refs/main when that snapshot is complete, else the newest complete
# snapshot (so a leftover partial dir cannot win over a finished one).
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
SNAP=""
SNAP_RC=0
SNAP="$(resolve_snapshot "$MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
[[ -n "$SNAP" ]] || err "No snapshot under $MODEL_PATH/snapshots"
SNAPSHOT_REL="snapshots/$SNAP"
[[ -f "$MODEL_PATH/$SNAPSHOT_REL/config.json" ]] || err "No snapshot under $MODEL_PATH/snapshots"
if [[ "$SNAP_RC" -ne 0 ]]; then
    if [[ "$ABLIT" == "1" ]]; then
        err "Checkpoint snapshot is incomplete. Resume with:  ABLIT=1 ./download.sh"
    else
        err "Checkpoint snapshot is incomplete. Resume it with: ./download.sh $MODEL_ID"
    fi
fi
if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    [[ -f "$MODEL_PATH/$SNAPSHOT_REL/ABLIT_META.json" ]] || err "ABLIT=1 but $MODEL_PATH/$SNAPSHOT_REL has no ABLIT_META.json.
       Fetch the gated checkpoint first (HF_TOKEN required; accept the terms on that page):
         ABLIT=1 ./download.sh"
fi
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

# PLE_GIB from the snapshot's actual PLE shard files (index.json weight_map
# keys matching model-ple*), not from the packed table's du. An override wins.
if [[ -z "$PLE_GIB" ]]; then
    PLE_GIB=$(python3 - "$MODEL_PATH/$SNAPSHOT_REL" <<'PY'
import json, pathlib, sys
snap = pathlib.Path(sys.argv[1])
idx = snap / "model.safetensors.index.json"
try:
    wm = json.loads(idx.read_text())["weight_map"]
except Exception:
    wm = {}
gib = 0.0
for name in set(wm.values()):
    if "model-ple" in name:
        f = snap / name
        if f.is_file():
            gib += f.stat().st_size
print(f"{gib/2**30:.2f}")
PY
)
    PLE_GIB="${PLE_GIB:-26.82}"
    if [[ "$PLE_GIB" != "26.82" ]]; then
        info "  PLE_GIB derived from checkpoint PLE shards: ${PLE_GIB} GiB (not the stock 26.82)."
    fi
fi

read -r MEM_TOTAL_GIB MEM_AVAIL_GIB MEM_USED_GIB SWAP_USED_GIB <<<"$(python3 -c "
m={l.split(':')[0]:int(l.split()[1]) for l in open('/proc/meminfo') if ':' in l}
g=1048576
print(m['MemTotal']/g, m['MemAvailable']/g,
      f\"{(m['MemTotal']-m['MemAvailable'])/g:.1f}\", f\"{(m['SwapTotal']-m['SwapFree'])/g:.1f}\")")"

MTP_GIB=0
[[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] && MTP_GIB=1.49
KV_MULT=1.0
# FP8 halves the main KV (12 full-attn layers, ~84% of bytes/token) but the QSA
# side/compressor caches stay BF16, so the real saving is ~1.7x, not 2x.
[[ "$KV_CACHE_DTYPE" == fp8* ]] && KV_MULT=0.58

# The GPU side is budgeted from the host side. vLLM on this integrated GPU
# treats MemAvailable as free memory and fills the GPU side to exactly
# GMU x MemTotal, so the budget is
#   min(weights + overhead + MTP + max(kv_need, KV_TARGET_GIB),
#       MemTotal - HOST_RESERVE_GIB)
# and the KV figure is whatever the capped budget leaves. GMU is floored to
# the 3 decimals vLLM is given, so the figures below are what vLLM will do.
read -r WEIGHTS_GPU_GIB KV_NEED_GIB BUDGET_GIB DERIVED_GMU KV_EXPECT_GIB KV_EXPECT_TOK BUDGET_CAP_GIB CAP_BINDS <<<"$(python3 -c "
import math
w=$WEIGHT_BYTES/2**30-$PLE_GIB
fixed=w+$OVERHEAD_GIB+$MTP_GIB
kv_need=$MAX_MODEL_LEN*$KV_BYTES_PER_TOKEN*$KV_MULT/2**30
wish=fixed+max(kv_need,$KV_TARGET_GIB)
cap=$MEM_TOTAL_GIB-$HOST_RESERVE_GIB
budget=min(wish,cap)
gmu=math.floor(budget/$MEM_TOTAL_GIB*1000)/1000
budget=gmu*$MEM_TOTAL_GIB
kv_exp=budget-fixed
print(f'{w:.2f} {kv_need:.2f} {budget:.2f} {gmu:.3f} {kv_exp:.2f} {int(max(kv_exp,0)*2**30/($KV_BYTES_PER_TOKEN*$KV_MULT))} {cap:.2f} {int(wish>cap)}')")"

if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    warn "  caller-pinned GMU=$GPU_MEMORY_UTILIZATION (derived would be $DERIVED_GMU)"
    read -r BUDGET_GIB KV_EXPECT_GIB KV_EXPECT_TOK <<<"$(python3 -c "
b=$GPU_MEMORY_UTILIZATION*$MEM_TOTAL_GIB
kv=b-$WEIGHTS_GPU_GIB-$OVERHEAD_GIB-$MTP_GIB
print(f'{b:.2f} {kv:.2f} {int(max(kv,0)*2**30/($KV_BYTES_PER_TOKEN*$KV_MULT))}')")"
    CAP_BINDS=0
    if python3 -c "import sys; sys.exit(0 if $BUDGET_GIB > $BUDGET_CAP_GIB else 1)"; then
        warn "  pinned budget ${BUDGET_GIB} GiB is ABOVE the host-side cap ${BUDGET_CAP_GIB} GiB"
        warn "  (MemTotal - HOST_RESERVE_GIB=${HOST_RESERVE_GIB}). This is the configuration that"
        warn "  killed three servers on 2026-09-04. You asked for it; the watchdog will end it."
    fi
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
info "  host reserve ............. ${HOST_RESERVE_GIB} GiB  (HOST_RESERVE_GIB) => GPU budget cap ${BUDGET_CAP_GIB} GiB"
if [[ "$CAP_BINDS" == 1 ]]; then
    warn "  KV target ${KV_TARGET_GIB} reduced to ${KV_EXPECT_GIB} by HOST_RESERVE_GIB=${HOST_RESERVE_GIB}"
fi
info "  GPU budget (GMU ${GPU_MEMORY_UTILIZATION}) ... ${BUDGET_GIB} GiB  => ~${KV_EXPECT_GIB} GiB KV (~${KV_EXPECT_TOK} tokens)"
info "  container cgroup cap ..... ${CONTAINER_MEM_GIB} GiB  (hard ceiling ${MAX_CONTAINER_GIB}; bounds host-side memory only)"
# What the reserve already has to carry before vLLM starts: everything else on
# the box, measured as MemTotal - MemAvailable. ~7 GiB is normal here.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}\$"; then
    info "  host footprint now ....... ${MEM_USED_GIB} GiB used + ${SWAP_USED_GIB} GiB swapped, INCLUDING the running ${CONTAINER_NAME} (not a co-tenant figure)"
else
    info "  host footprint now ....... ${MEM_USED_GIB} GiB used by everything else (MemTotal - MemAvailable) + ${SWAP_USED_GIB} GiB swapped"
    if python3 -c "import sys; sys.exit(0 if $MEM_USED_GIB > 9 else 1)"; then
        warn "  co-tenants already spend ${MEM_USED_GIB} GiB of the ${HOST_RESERVE_GIB} GiB host reserve (~7 is normal here)."
        warn "  Find them: docker stats --no-stream; ps -eo rss,cmd --sort=-rss | head. Or raise HOST_RESERVE_GIB."
    fi
fi

if python3 -c "import sys; sys.exit(0 if $KV_EXPECT_GIB < $KV_NEED_GIB else 1)"; then
    if [[ "$CAP_BINDS" == 1 ]]; then
        err "HOST_RESERVE_GIB=${HOST_RESERVE_GIB} caps the GPU budget at ${BUDGET_CAP_GIB} GiB, which leaves
       ${KV_EXPECT_GIB} GiB for KV, but ${MAX_MODEL_LEN} tokens need ${KV_NEED_GIB} GiB.
       Lower MAX_MODEL_LEN or use KV_CACHE_DTYPE=fp8. Lowering HOST_RESERVE_GIB trades
       host safety for context; the incident doc explains what that bought last time."
    fi
    err "Budget leaves ${KV_EXPECT_GIB} GiB for KV but ${MAX_MODEL_LEN} tokens need ${KV_NEED_GIB} GiB.
       Lower MAX_MODEL_LEN or raise GPU_MEMORY_UTILIZATION."
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

# Kernel VM tunables. The stock values give the NVIDIA driver no free-page
# reserve (min_free_kbytes ~44 MB on a 121 GiB box) and start reclaim at 0.1%.
# files/sysctl-spark3.conf holds the values spark1 measured six crash-free runs
# with; read its header before applying (they shift MemAvailable accounting).
VM_MIN_FREE_KB=$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo 0)
VM_WSF=$(cat /proc/sys/vm/watermark_scale_factor 2>/dev/null || echo 0)
if (( VM_MIN_FREE_KB < 1048576 || VM_WSF < 100 )); then
    warn "  kernel VM tunables at defaults (vm.min_free_kbytes=${VM_MIN_FREE_KB}, vm.watermark_scale_factor=${VM_WSF}): no free-page reserve for the NVIDIA driver. Not applied by this script (sudo). See files/sysctl-spark3.conf, then: sudo sysctl -p files/sysctl-spark3.conf"
fi

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
MTP_PKG="$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"

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

# Reduced-vocabulary drafting. The patch is inert unless VLLM_MTP_DRAFT_VOCAB
# is set in the container, so it is applied unconditionally.
PATCHED_MTP="$SCRIPT_DIR/files/mtp_patched.py"
extract "$MTP_PKG" "$PATCHED_MTP.orig"
python3 "$SCRIPT_DIR/files/patch_mtp_draft_vocab.py"
[[ -f "$PATCHED_MTP" ]] || err "MTP patch missing after patch_mtp_draft_vocab.py"

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

# ---------------------------------------------------------------------------
# Quant_algo dispatch pre-flight (review §5.6 / jschmied A2b). The runtime
# reads the embedded quantization_config inside config.json, not the sidecar
# hf_quant_config.json. If the checkpoint declares a quant_algo the image's
# ModelOptMixedPrecisionConfig does not dispatch, MLMP falls through to
# UnquantizedLinearMethod and serves packed FP8 bytes as BF16 — fluent garbage
# with zero errors. Refuse to launch instead. This launches a throwaway
# container (~30 s, no GPU work); acceptable per launch.
# ---------------------------------------------------------------------------
_QUANT_PREFLIGHT_DISABLED="${QUANT_PREFLIGHT_DISABLED:-0}"
if [[ "$DO_LAUNCH" == "true" && "$_QUANT_PREFLIGHT_DISABLED" != "1" ]]; then
    _QALGO=$(python3 - "$MODEL_PATH/$SNAPSHOT_REL" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
q = None
try:
    q = json.loads((d / "config.json").read_text()).get("quantization_config")
except Exception:
    pass
side = d / "hf_quant_config.json"
if q is None and side.is_file():
    try:
        q = json.loads(side.read_text())
    except Exception:
        q = None
if not q:
    raise SystemExit(1)
# Per-layer algos are what the image dispatches (get_quant_method); the
# top-level quant_algo/quant_method (e.g. MIXED_PRECISION) names the config
# class, not a dispatchable algo, so only when no quantized_layers exist is
# the top-level value checked.
ql = (q.get("quantization") or q).get("quantized_layers", {})
algos = set()
for v in ql.values():
    a = v.get("quant_algo") if isinstance(v, dict) else None
    if a:
        algos.add(str(a).upper())
if not algos:
    top = q.get("quant_algo") or q.get("quant_method") or ""
    if isinstance(top, str):
        algos.add(top.upper())
    elif isinstance(top, list):
        algos |= {str(a).upper() for a in top}
print(" ".join(sorted(algos)))
PY
)
    if [[ -n "$_QALGO" ]]; then
        _DISPATCH=$(docker run --rm --entrypoint python3 \
            -v "$MODEL_PATH/$SNAPSHOT_REL:/m:ro" "$IMAGE" -c '
import json, pathlib, sys
cfg = json.loads(pathlib.Path("/m/config.json").read_text())
qc = cfg.get("quantization_config")
if not qc and pathlib.Path("/m/hf_quant_config.json").is_file():
    qc = json.loads(pathlib.Path("/m/hf_quant_config.json").read_text())
if not qc:
    print(json.dumps({"declared": []})); sys.exit(0)
algos = qc.get("quant_algo") or []
lst = [algos] if isinstance(algos, str) else algos
declared = sorted({str(x).upper() for x in lst})
# The engine needs a non-empty quantized_layers mapping for MIXED_PRECISION.
ql = (qc.get("quantization") or qc).get("quantized_layers", {})
present = sorted({v.get("quant_algo", "").upper() for v in ql.values() if v.get("quant_algo")})
try:
    from vllm.model_executor.layers.quantization import get_quantization_config
    clz = get_quantization_config("modelopt_mixed")
    inst = clz.from_config(qc)
    # The runtime dispatches exactly the algos found in quantized_layers;
    # get_quant_method() returns UnquantizedLinearMethod for anything else.
    supported = present
    print(json.dumps({"declared": declared, "supported": supported, "in_layers": present}))
except Exception as e:
    print(json.dumps({"declared": declared, "error": str(e)[:200]}, default=str))
    sys.exit(3)
' 2>/dev/null || echo "")
        if [[ -z "$_DISPATCH" ]]; then
            warn "quant_algo pre-flight: image introspection failed (offline / older image); skipping."
            warn "     Checkpoint declares quant_algo: $_QALGO"
        else
            _SUPPORTED=$(echo "$_DISPATCH" | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin).get("supported", [])))' 2>/dev/null || echo "")
            _ERROR=$(echo "$_DISPATCH" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("error", ""))' 2>/dev/null || echo "")
            if [[ -n "$_ERROR" ]]; then
                warn "quant_algo pre-flight: image rejected the checkpoint config ($_ERROR); skipping."
            elif [[ -n "$_SUPPORTED" ]]; then
                _MISSING=""
                for _a in $_QALGO; do
                    _up=$(echo "$_a" | tr '[:lower:]' '[:upper:]')
                    _found=0
                    for _s in $_SUPPORTED; do
                        [[ "$_s" == "$_up" ]] && _found=1
                    done
                    (( _found == 0 )) && _MISSING="$_MISSING $_a"
                done
                if [[ -n "$_MISSING" ]]; then
                    err "quant_algo $_MISSING declared by the checkpoint is not dispatched by this image."
                    err "     The image's ModelOptMixedPrecisionConfig falls back to UnquantizedLinearMethod for unrecognized algos — silent garbage. Checkpoint/image mismatch."
                else
                    info "quant_algo pre-flight: checkpoint algos ($_QALGO) dispatched by this image."
                fi
            fi
        fi
    fi
fi

# The Keys splice leaves the PLE n-gram shards stock, so the packed table is
# shared with the Mia checkpoint instead of rebuilt (~27 GiB). Read that from the
# checkpoint's own metadata rather than assuming it: if a future ablit ever
# touches PLE, build a separate table instead of poisoning the stock cache.
PLE_CACHE_ID="$MODEL_ID"
if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["recipe"]["edit_ple"] is False else 1)' \
            "$MODEL_PATH/$SNAPSHOT_REL/ABLIT_META.json" 2>/dev/null; then
        PLE_CACHE_ID="$STOCK_MODEL_ID"
        info "ABLIT=1: ABLIT_META.json reports edit_ple=false; reusing packed PLE cache for $STOCK_MODEL_ID"
        # INTERIM, plan 2.1: the edit_ple flag does NOT prove PLE identity — the
        # review found 17/34 PLE shards genuinely differ between the stock and
        # ablit snapshots, so reuse can serve wrong weights today. Cut to the
        # real identity check (download.sh 0.9 sha256 state, else sampled
        # checksum) before trusting ABLIT=1 long-term.
        warn "ABLIT=1: PLE reuse is keyed on edit_ple=false ONLY (interim). The review"
        warn "     found 17/34 PLE shards genuinely differ from stock — verify shard"
        warn "     identity before trusting ABLIT=1 (plan 2.1 replaces this check)."
    else
        warn "ABLIT=1: ABLIT_META.json does not report edit_ple=false."
        warn "     Building a separate packed PLE table for $MODEL_ID (~27 GiB)."
    fi
fi
PLE_ORG="${PLE_CACHE_ID%%/*}"; PLE_NAME="${PLE_CACHE_ID##*/}"
PLE_CACHE_HOST="$HOME/.cache/vllm/ple_cache/${PLE_ORG}--${PLE_NAME}"
PLE_CACHE_CTR="/root/.cache/vllm/ple_cache/${PLE_ORG}--${PLE_NAME}"
if ! ls "$PLE_CACHE_HOST"/*.packed_u8 >/dev/null 2>&1; then
    info "Building packed PLE table (one-time, ~40 s, <1 GiB RAM, no GPU)..."
    mkdir -p "$PLE_CACHE_HOST"
    docker run --rm --name "${CONTAINER_NAME}-plebuild" --memory 6g --cpus 8 \
        -v "$MODEL_PATH:/m:ro" -v "$HOME/.cache/vllm/ple_cache:/out" \
        -v "$SCRIPT_DIR/files/build_ple_packed_table.py:/b.py:ro" \
        --entrypoint python3 "$IMAGE" -u /b.py "/m/$SNAPSHOT_REL" "/out/${PLE_ORG}--${PLE_NAME}"
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
[[ -n "$MAMBA_SSM_CACHE_DTYPE" ]] && VLLM_ARGS+=("--mamba-ssm-cache-dtype" "$MAMBA_SSM_CACHE_DTYPE")
if [[ -n "$YARN_FACTOR" ]]; then
    # Deep-merged into text_config.rope_parameters, which is what this model
    # reads (nvidia/qsa.py) and what vLLM's max-len check scales by. The
    # existing mrope_section / rope_theta / partial_rotary_factor survive.
    VLLM_ARGS+=("--hf-overrides" "$(printf "'{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":%s,\"original_max_position_embeddings\":%s}}}'" "$YARN_FACTOR" "$NATIVE_MAX_MODEL_LEN")")
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
# MTP legality guard (review §5.3 / §6.1): legal k set derives from the
# checkpoint's attention block size and the QSA ring compress ratio —
#   capacity = compress_ratio * ceil((compress_ratio + k) / compress_ratio)
#   must divide block_size.
# Source of truth for block_size is the ENGINE's derivation, not our guess:
# introspect it with a throwaway container that imports the model module and
# prints the derived block size, cached keyed on the snapshot hash. If the
# introspection cannot run (offline / old image), fall back to the known-good
# table {0,2,3,4,9..12} for block 848 with a WARN. k=1 is rejected separately:
# it is strictly dominated (same fixed cache-block cost as k=2, half the gain).
_MTP_CACHE_DIR="$HOME/.cache/vllm/ple_cache"
if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
    _MTP_BLOCK=""; _MTP_CR=""; _MTP_INTRO_SRC="fallback"
    _MTP_INTRO_FILE="$_MTP_CACHE_DIR/mtp-ring-$(printf '%s' "$SNAP" | cut -c1-16)"
    if [[ -r "$_MTP_INTRO_FILE" ]]; then
        read -r _MTP_BLOCK _MTP_CR < "$_MTP_INTRO_FILE" && _MTP_INTRO_SRC="cache"
    fi
    if [[ -z "$_MTP_BLOCK" ]]; then
        # Best-effort; any failure falls through to the known-good table.
        _MTP_INTRO=$(docker run --rm -v "$MODEL_PATH/$SNAPSHOT_REL:/m:ro" \
            --entrypoint python3 "$IMAGE" -c '
import json, importlib, math, pathlib, sys
cfg = json.loads(pathlib.Path("/m/config.json").read_text())
tc = cfg.get("text_config", cfg)
cr = int(tc.get("qsa_compress_ratio", tc.get("compress_ratio", 4)))
bs = None
for mod in ("vllm.models.qwen3_8_flash_next.nvidia.qsa",
            "vllm.models.qwen3_8_flash_next.nvidia.attention",
            "vllm.models.qwen3_8_flash_next.nvidia.ops.qsa"):
    try:
        m = importlib.import_module(mod)
    except Exception:
        continue
    for attr in ("QSA_ATTENTION_BLOCK_SIZE", "QSA_BLOCK_SIZE", "ATTENTION_BLOCK_SIZE"):
        if getattr(m, attr, None):
            bs = getattr(m, attr); break
    qsa_cr = getattr(m, "QSA_RING_COMPRESS_RATIO", None)
    if qsa_cr: cr = int(qsa_cr)
    if bs: break
if bs is None:
    bs = 0
print(f"{int(bs)} {cr}")
' 2>/dev/null || true)
        if [[ -n "$_MTP_INTRO" ]]; then
            read -r _MTP_BLOCK _MTP_CR <<< "$_MTP_INTRO"
            [[ -n "$_MTP_BLOCK" && -n "$_MTP_CR" ]] && _MTP_INTRO_SRC="introspect" \
                || { _MTP_BLOCK=""; _MTP_CR=""; }
        fi
    fi
    if [[ -z "$_MTP_BLOCK" || -z "$_MTP_CR" || "$_MTP_BLOCK" == "0" ]]; then
        _MTP_BLOCK=848; _MTP_CR=4; _MTP_INTRO_SRC="fallback"
        warn "MTP legality: engine introspection unavailable; using the known-good table"
        warn "     for block 848, compress ratio 4: legal k = {0,2,3,4,9..12}."
    elif [[ "$_MTP_INTRO_SRC" == "introspect" ]]; then
        mkdir -p "$_MTP_CACHE_DIR"
        printf '%s %s\n' "$_MTP_BLOCK" "$_MTP_CR" > "$_MTP_INTRO_FILE" 2>/dev/null || true
    fi
    # Legal iff the ring capacity for THIS k divides block_size:
    #   capacity = compress_ratio * ceil((compress_ratio + k) / compress_ratio)
    # The single computation covers any k (no capped enumeration).
    _K="$MTP_NUM_SPECULATIVE_TOKENS"
    _CAP=$(( _MTP_CR * ((_MTP_CR + _K + _MTP_CR - 1) / _MTP_CR) ))
    _MTP_LEGAL=1
    (( _MTP_BLOCK % _CAP == 0 )) && _MTP_LEGAL=0
    if [[ "$_MTP_LEGAL" == "1" ]]; then
        err "MTP_NUM_SPECULATIVE_TOKENS=$_K is illegal for block ${_MTP_BLOCK}, compress ratio ${_MTP_CR} ($_MTP_INTRO_SRC): illegal k hard-fails the engine at config validation. Use a legal k (shipped default 3)."
    fi
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -eq 1 ]]; then
        err "MTP_NUM_SPECULATIVE_TOKENS=1 is strictly dominated: same fixed cache-block cost as k=2, half the decode gain. Use 2, 3 or 4."
    fi
    # --async-scheduling with MTP > 0 silently corrupts n-grams (jschmied:
    # "no benchmark reveals it"). Match the bare flag after the 0.2 word-split
    # and any "--async-scheduling=..." value.
    if [[ "$EXTRA_VLLM_ARGS" == *"--async-scheduling"* ]]; then
        err "EXTRA_VLLM_ARGS contains --async-scheduling while MTP is on: silent n-gram corruption (jschmied). Remove --async-scheduling."
    fi
fi
if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
    _SPEC_ARGMAX=""
    # get_top_tokens() is the only path that reads the reduced head; the
    # speculator calls it only under use_local_argmax_reduction.
    [[ -n "$MTP_DRAFT_VOCAB" ]] && _SPEC_ARGMAX=',"use_local_argmax_reduction":true'
    # disable_eagle_block_drop (vllm#53388, plan 2.4): removes MTP's fixed
    # prefix-cache-block back-off per turn. Merged the same way as the other
    # scalars; a vLLM that does not know the key ignores it harmlessly.
    [[ "$MTP_DISABLE_BLOCK_DROP" == "1" ]] && _SPEC_ARGMAX+=',"disable_eagle_block_drop":true'
    _SPEC_SCHED=""
    if [[ -n "$MTP_K_SCHEDULE" ]]; then
        _SPEC_SCHED=",\"num_speculative_tokens_per_batch_size\":[$(
            printf '%s' "$MTP_K_SCHEDULE" | awk -F, '{
                out=""
                for (i = 1; i <= NF; i++) {
                    split($i, r, ":")
                    out = out (i > 1 ? "," : "") "[" r[1] "," r[2] "," r[3] "]"
                }
                printf "%s", out
            }')]"
    fi
    VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s%s%s}'" "$MTP_NUM_SPECULATIVE_TOKENS" "$_SPEC_SCHED" "$_SPEC_ARGMAX")")
fi
_CG_SIZES="$CUDAGRAPH_CAPTURE_SIZES"
if [[ "$_CG_SIZES" == "auto" ]]; then
    # Every verify-batch width the scheduler can actually build: (1+K(S))*S for
    # S in 1..MAX_NUM_SEQS, where K(S) follows MTP_K_SCHEDULE when one is set
    # and is the constant MTP_NUM_SPECULATIVE_TOKENS otherwise. A width that is
    # not in this list has no decode graph and falls back to eager.
    _CG_SIZES=$(
        _AUTO_MAX_SEQS="$MAX_NUM_SEQS" \
        _AUTO_K="$MTP_NUM_SPECULATIVE_TOKENS" \
        _AUTO_SCHED="$MTP_K_SCHEDULE" \
        python3 -c '
import os
max_seqs = int(os.environ["_AUTO_MAX_SEQS"])
k_default = int(os.environ["_AUTO_K"])
k_of = {}
for part in filter(None, os.environ["_AUTO_SCHED"].strip().split(",")):
    lo, hi, k = (int(x) for x in part.split(":"))
    for s in range(lo, min(hi, max_seqs) + 1):
        k_of.setdefault(s, min(k, k_default))
print(",".join(str(x) for x in sorted(
    {(1 + k_of.get(s, k_default)) * s for s in range(1, max_seqs + 1)})))
'
    )
fi
if [[ -n "$_CG_SIZES" ]]; then
    VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":%s,\"cudagraph_mode\":\"%s\",\"cudagraph_capture_sizes\":[%s]}'" "$COMPILATION_MODE" "$CUDAGRAPH_MODE" "$_CG_SIZES")")
else
    VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":%s,\"cudagraph_mode\":\"%s\"}'" "$COMPILATION_MODE" "$CUDAGRAPH_MODE")")
fi
# EXTRA_VLLM_ARGS is word-split with shell-word semantics, so quoting inside
# the value is not supported (same contract as EXTRA_DOCKER_ARGS).
[[ -n "$EXTRA_VLLM_ARGS" ]] && { read -ra _EXTRA_VLLM <<< "$EXTRA_VLLM_ARGS"; VLLM_ARGS+=("${_EXTRA_VLLM[@]}"); }
VLLM_ARGS_STR="${VLLM_ARGS[*]}"

# Non-loopback bind with no api key = the whole network the box is on can
# reach an unauthenticated unfiltered model. Warn, do not refuse (this is
# exactly the override users opt into).
if [[ "$BIND" != "127.0.0.1" && "$BIND" != "::1" && "$BIND" != "localhost" ]]; then
    if ! [[ "$EXTRA_VLLM_ARGS" == *"--api-key"* ]]; then
        warn "BIND=$BIND is not loopback and no --api-key appears in EXTRA_VLLM_ARGS:"
        warn "     the API is reachable on every interface this box has:"
        warn "     $(hostname -I)"
        warn "     Serve with --api-key set, or use an ssh tunnel."
    fi
fi

info ""
info "Config (single Spark, TP=1):"
info "  Model:      $MODEL_ID"
info "  Ablit:      $ABLIT$( [[ "$ABLIT" == "1" ]] && echo ' (gated Keys o_proj L15-47)' )"
info "  Image:      $IMAGE"
if [[ -n "$YARN_FACTOR" ]]; then
info "  Context:    $MAX_MODEL_LEN tokens (YaRN factor $YARN_FACTOR over native $NATIVE_MAX_MODEL_LEN)"
else
info "  Context:    $MAX_MODEL_LEN tokens (native rope, no YaRN)"
fi
info "  GMU:        $GPU_MEMORY_UTILIZATION  (budget ${BUDGET_GIB} GiB, cgroup cap ${CONTAINER_MEM_GIB} GiB)"
info "  Max seqs:   $MAX_NUM_SEQS   Batched tokens: $MAX_NUM_BATCHED_TOKENS   KV dtype: $KV_CACHE_DTYPE"
info "  SSM state:  ${MAMBA_SSM_CACHE_DTYPE:-float32 (checkpoint)}"
info "  MTP:        $MTP_NUM_SPECULATIVE_TOKENS $( [[ "$MTP_NUM_SPECULATIVE_TOKENS" -eq 0 ]] && echo '(disabled)')"
info "  Draft vocab: ${MTP_DRAFT_VOCAB:-full (248320)}"
info "  Graphs:     $CUDAGRAPH_MODE  capture=${_CG_SIZES:-vllm-default}  compile-mode=$COMPILATION_MODE"
info "  Port:       $PORT  (bind $BIND)"
info ""

LAUNCH_SCRIPT=$(mktemp /tmp/vllm_tp1_XXXXXX.sh)
cat > "$LAUNCH_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \\
    -d --name $CONTAINER_NAME \\
    --gpus all --network host --ipc host \\
    --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \\
    --memory ${CONTAINER_MEM_GIB}g --memory-swap ${CONTAINER_MEM_GIB}g \\
    --log-opt max-size=50m --log-opt max-file=3 \\
    -e HF_HUB_OFFLINE=1 \\
    -e TRANSFORMERS_OFFLINE=1 \\
    -e VLLM_PLE_CPU_OFFLOAD=1 \\
    -e VLLM_PLE_PACKED_TABLE_DIR=$PLE_CACHE_CTR \\
    -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \\
    -e MAX_JOBS=2 \\
    -e FLASHINFER_NVCC_THREADS=1 \\
    ${VLLM_QSA_DET_TOPK:+-e VLLM_QSA_DET_TOPK=$VLLM_QSA_DET_TOPK} \\
    ${VLLM_MOE_DET_FINALIZE:+-e VLLM_MOE_DET_FINALIZE=$VLLM_MOE_DET_FINALIZE} \\
    ${GDN_DECODE_KERNEL:+-e VLLM_GDN_DECODE_KERNEL=$GDN_DECODE_KERNEL} \\
    ${MTP_DRAFT_VOCAB:+-v $MTP_DRAFT_VOCAB:/root/draft_vocab.txt:ro} \\
    ${MTP_DRAFT_VOCAB:+-e VLLM_MTP_DRAFT_VOCAB=/root/draft_vocab.txt} \\
    -e HF_HOME=/root/.cache/huggingface \\
    ${HF_TOKEN:+-e HF_TOKEN=\$HF_TOKEN} \\
    -v $PATCHED_PLE:$PLE_PKG:ro \\
    -v $PATCHED_MODELOPT:$MODELOPT_PKG:ro \\
    -v $PATCHED_QSA_OPS:$QSA_OPS_PKG:ro \\
    -v $PATCHED_QSA_NVIDIA:$QSA_NVIDIA_PKG:ro \\
    -v $PATCHED_MTP:$MTP_PKG:ro \\
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
    --host $BIND \\
    --port $PORT
LAUNCH_EOF
chmod +x "$LAUNCH_SCRIPT"
cp "$LAUNCH_SCRIPT" "$SCRIPT_DIR/.last_launch.sh"
# The copy must not be world-readable: it can still carry config data even
# though the HF_TOKEN value itself is resolved at exec time above.
chmod 600 "$SCRIPT_DIR/.last_launch.sh"

if ! $DO_LAUNCH; then
    info "--no-launch: command written to .last_launch.sh"
    cat "$SCRIPT_DIR/.last_launch.sh"
    rm -f "$LAUNCH_SCRIPT"
    exit 0
fi

# The generated launch script resolves $HF_TOKEN from ITS environment at exec
# time (0.3 hygiene), so the token must actually be exported here.
export HF_TOKEN

# ---------------------------------------------------------------------------
# 6. Launch + watchdog
# ---------------------------------------------------------------------------
info "=== Step 6: Launch ==="
mkdir -p "$SCRIPT_DIR/logs/archive"
ARCHIVE_TS=$(date '+%Y%m%dT%H%M%S')
# Keep the newest 20 sets in logs/archive/, then drop the oldest. A set is a
# timestamp prefix with -container.log / -memwatch.log / (possibly
# -probe-latency.log) members. 24/7 relauches run on a scheduled cadence, so
# without a prune the archive grows forever and threatens the checkpoint
# disk cache.
ls -1t "$SCRIPT_DIR"/logs/archive/*-container.log 2>/dev/null | tail -n +21 | while read -r f; do
    _set="${f%-container.log}"
    rm -f "${_set}-container.log" "${_set}-memwatch.log" "${_set}-probe-latency.log" "${_set}-timeout.log" 2>/dev/null || true
done
if docker inspect "$CONTAINER_NAME" &>/dev/null; then
    # The old container is removed below; keep its log for the post-mortem first.
    docker logs --tail 3000 "$CONTAINER_NAME" > "$SCRIPT_DIR/logs/archive/${CONTAINER_NAME}-${ARCHIVE_TS}-container.log" 2>&1 || true
    info "Previous container log archived: logs/archive/${CONTAINER_NAME}-${ARCHIVE_TS}-container.log"
fi
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/vllm"
bash "$LAUNCH_SCRIPT"
rm -f "$LAUNCH_SCRIPT"
ok "Container $CONTAINER_NAME started."

# Start the watchdog via the shared helper start.sh and supervise.sh both
# call, so the invocation cannot drift. It kills the previous memwatch, runs
# memwatch in the background, and echoes the log path.
MEMWATCH_LOG="$SCRIPT_DIR/logs/memwatch-${CONTAINER_NAME}.log"
if [[ -s "$MEMWATCH_LOG" ]]; then
    mv "$MEMWATCH_LOG" "$SCRIPT_DIR/logs/archive/${CONTAINER_NAME}-${ARCHIVE_TS}-memwatch.log"
    info "Previous watchdog log archived: logs/archive/${CONTAINER_NAME}-${ARCHIVE_TS}-memwatch.log"
fi
MEMWATCH_MIN_FREE_GIB="$MEMWATCH_MIN_FREE_GIB" MEMWATCH_FREE_GATE_GIB="$MEMWATCH_FREE_GATE_GIB" \
    MEMWATCH_GRACE="$MEMWATCH_GRACE" MEMWATCH_LOG="$MEMWATCH_LOG" \
    bash "$SCRIPT_DIR/scripts/start-memwatch.sh" "$CONTAINER_NAME" "$MEMWATCH_MIN_GIB"
ok "Watchdog running (stops container after 5 samples of MemAvailable < ${MEMWATCH_MIN_GIB} GiB, or MemFree < ${MEMWATCH_MIN_FREE_GIB} GiB while MemAvailable < ${MEMWATCH_FREE_GATE_GIB} GiB): logs/memwatch-${CONTAINER_NAME}.log"
info "Loading weights (~3-4 min). Following logs until ready..."

docker logs -f "$CONTAINER_NAME" &
LOGPID=$!
WAIT_START=$(date +%s)
_last_hb=0
while true; do
    sleep 10
    NOW=$(date +%s)
    ELAPSED=$((NOW - WAIT_START))
    if [[ "$ELAPSED" -gt "$READY_TIMEOUT_S" ]]; then
        kill $LOGPID 2>/dev/null || true
        echo ""
        err "Readiness timed out after ${ELAPSED}s (>READY_TIMEOUT_S=${READY_TIMEOUT_S})."
        err "Container was wedged before /health; archiving, removing, and exiting non-zero."
        docker logs --tail 100 "$CONTAINER_NAME" 2>&1 || true
        # Archive + remove the wedged container. A timed-out container left
        # Restarting in systemd's eyes would be relaunched by the supervisor
        # while the wedged one still holds the GPU/port name.
        docker logs "$CONTAINER_NAME" > "$SCRIPT_DIR/logs/archive/${CONTAINER_NAME}-${ARCHIVE_TS}-timeout.log" 2>&1 || true
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        exit 1
    fi
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
        ok "vLLM ready on port $PORT (TP=1, single Spark) after ${ELAPSED}s."
        docker logs "$CONTAINER_NAME" 2>&1 | grep -iE "GPU KV cache size|Available KV cache|Maximum concurrency" | tail -3 || true
        info ""
        info "Stop:  ./stop.sh   (graceful; --force to skip the SIGTERM wait)"
        break
    fi
    if (( NOW - _last_hb >= 60 )); then
        _last_hb=$NOW
        echo "  ...waiting for readiness: ${ELAPSED}s elapsed, last /health code $CODE"
    fi
done
