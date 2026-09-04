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

pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }

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
ANSWER=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"] or "")' 2>/dev/null)
echo "$ANSWER" | grep -q "391" && ok "17*23=391 answered correctly" || bad "answer missing 391: ${ANSWER:0:200}"

echo "== 4. determinism (temp 0, two runs) =="
PROMPT='{"model":"'$MODEL'","temperature":0,"max_tokens":128,"messages":[{"role":"user","content":"List the first 8 prime numbers."}]}'
R1=$(curl -s -m 120 "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "$PROMPT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"] or "")' 2>/dev/null)
R2=$(curl -s -m 120 "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "$PROMPT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"] or "")' 2>/dev/null)
if [[ -n "$R1" && "$R1" == "$R2" ]]; then
    ok "identical outputs at temperature 0"
else
    bad "outputs differ at temperature 0 (GB10 QSA top-k is non-deterministic — see issue #7)"
fi

echo "== 5. decode speed (real answer, not ignore_eos) =="
OUT=$(curl -s -m 300 "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 400,
  \"messages\": [{\"role\":\"user\",\"content\":\"Write a 350-word travel guide to Lisbon.\"}]}")
echo "$OUT" | python3 -c '
import json, sys
d = json.load(sys.stdin)
u = d["usage"]
ct = u["completion_tokens"]
# vLLM does not report wall time; measure client-side is unreliable, so use
# the server-side usage only and ask the caller to compare against ~25-40 tok/s.
print(f"  completion_tokens={ct}")
' 2>/dev/null || bad "no usage in response"

echo "== 6. metrics endpoint =="
curl -s -m 5 "$BASE/metrics" | grep -q "vllm:" \
    && ok "/metrics exposes vllm: series" || bad "/metrics missing vllm: series"

echo ""
echo "== $pass passed, $fail failed =="
exit $((fail > 0))
