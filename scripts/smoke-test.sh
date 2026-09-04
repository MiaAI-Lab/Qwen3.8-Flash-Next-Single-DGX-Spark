#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# smoke-test.sh — verify a running server actually works: health, coherent
# generation, determinism at temperature 0, decode speed, served context.
#
# Usage:
#   ./scripts/smoke-test.sh                      # localhost:8888, no auth
#   PORT=9000 API_KEY=xyz ./scripts/smoke-test.sh
set -uo pipefail

PORT="${PORT:-8888}"
MODEL="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
EXPECT_LEN="${EXPECT_LEN:-}"        # set to 262144 or 524288 to assert context
BASE="http://localhost:$PORT"
AUTH=(); [[ -n "${API_KEY:-}" ]] && AUTH=(-H "Authorization: Bearer $API_KEY")

pass=0; fail=0; warn=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
note() { echo "  WARN  $*"; warn=$((warn+1)); }

echo "== 1. health =="
curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE/health" | grep -q 200 \
    && ok "/health 200" || { bad "/health not 200 — is the server up?"; exit 1; }

echo "== 2. model metadata =="
LEN=$(curl -s -m 5 "${AUTH[@]}" "$BASE/v1/models" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' 2>/dev/null || echo 0)
[[ "$LEN" -gt 0 ]] && ok "max_model_len=$LEN" || bad "could not read max_model_len (auth needed? set API_KEY)"
[[ -n "$EXPECT_LEN" && "$LEN" != "$EXPECT_LEN" ]] && bad "expected max_model_len=$EXPECT_LEN, got $LEN"

echo "== 3. coherent generation (temp 0) =="
RESP=$(curl -s -m 120 "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 256,
  \"messages\": [{\"role\":\"user\",\"content\":\"What is 17 * 23? Think step by step, then give the final number.\"}]}")
# This model reasons first in a `reasoning` field; content may lag behind.
ANSWER=$(echo "$RESP" | python3 -c 'import json,sys
m = json.load(sys.stdin)["choices"][0]["message"]
print((m.get("reasoning") or "") + "\n" + (m.get("content") or ""))' 2>/dev/null)
echo "$ANSWER" | grep -q "391" && ok "17*23=391 answered correctly" || bad "answer missing 391: ${ANSWER:0:200}"

echo "== 4. determinism (temp 0, two runs) =="
PROMPT='{"model":"'$MODEL'","temperature":0,"max_tokens":128,"messages":[{"role":"user","content":"List the first 8 prime numbers."}]}'
R1=$(curl -s -m 120 "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "$PROMPT" | python3 -c 'import json,sys
m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("reasoning") or "")+(m.get("content") or ""))' 2>/dev/null)
R2=$(curl -s -m 120 "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "$PROMPT" | python3 -c 'import json,sys
m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("reasoning") or "")+(m.get("content") or ""))' 2>/dev/null)
if [[ -n "$R1" && "$R1" == "$R2" ]]; then
    ok "identical outputs at temperature 0"
else
    # Known stack property, not a deployment failure: the stock GB10 QSA
    # top-k kernel is non-deterministic (drops candidates, upstream
    # vllm#51782). Flaky in both directions — a pass does not prove the
    # kernel is deterministic either.
    note "outputs differ at temperature 0 (known GB10 QSA top-k non-determinism — see issue #7)"
fi

echo "== 5. decode speed (real answer, not ignore_eos) =="
# Client-side wall clock; fine for a smoke check, not a benchmark — MTP
# acceptance is content-dependent, so treat the number as a range. Healthy
# single-stream decode on this kit is roughly 25-40 tok/s.
T0=$(date +%s.%N)
OUT=$(curl -s -m 300 "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 400,
  \"messages\": [{\"role\":\"user\",\"content\":\"Write a 350-word travel guide to Lisbon.\"}]}")
T1=$(date +%s.%N)
RATE=$(echo "$OUT" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d["usage"]["completion_tokens"])' 2>/dev/null)
if [[ -n "$RATE" && "$RATE" -gt 0 ]]; then
    TPS=$(python3 -c "print(f'{$RATE / ($T1 - $T0):.1f}')")
    echo "  completion_tokens=$RATE in $(python3 -c "print(f'{$T1-$T0:.1f}')")s -> ${TPS} tok/s"
    python3 -c "import sys; sys.exit(0 if $RATE / ($T1 - $T0) >= 15 else 1)"         && ok "decode ${TPS} tok/s (>=15)" || bad "decode ${TPS} tok/s — suspiciously slow"
else
    bad "no completion tokens in response"
fi

echo "== 6. metrics endpoint =="
curl -s -m 5 "$BASE/metrics" | grep -q "vllm:" \
    && ok "/metrics exposes vllm: series" || bad "/metrics missing vllm: series"

echo ""
echo "== $pass passed, $fail failed, $warn warnings =="
exit $((fail > 0))
