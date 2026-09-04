#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Long-context prefill benchmark + needle retrieval.

Calibrates a prompt to an exact token target, salts it to defeat prefix
caching, and measures TTFT (= prefill wall time) via streaming.

  python3 bench/longctx.py --target 600000 --max-tokens 1024

NOTE: this build streams reasoning in delta["reasoning"], NOT
"reasoning_content". TTFT is anchored on the FIRST delta of any kind
(including the empty role chunk), which is what marks end-of-prefill.
"""
import json, time, argparse, urllib.request, uuid

BASE, MODEL = "http://localhost:8888", "qwen3.8-flash-next"
FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")   # 25 tokens/line
NEEDLES = [(0.05, "alpha", "7391-CORAL"),
           (0.50, "bravo", "2648-INDIGO"),
           (0.95, "charlie", "5127-SAFFRON")]

def post(path, payload, timeout=3600, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else json.loads(r.read())

def count_tokens(text):
    return post("/tokenize", {"model": MODEL, "prompt": text})["count"]

def build(n):
    lines = [FILLER.format(i=i) for i in range(n)]
    for frac, key, val in NEEDLES:
        idx = min(int(n * frac), n - 1)
        lines[idx] = f"Entry {idx:06d}: SECRET RECORD -- the {key} access code is {val}.\n"
    return "".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600_000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    q = ("\n\nQuestion: three SECRET RECORD lines are hidden in the log above. "
         "List the alpha, bravo and charlie access codes, one per line. "
         "Answer directly with just the three codes.")
    per_line = count_tokens(build(2000)) / 2000
    n = int(args.target / per_line)
    # unique salt at the front kills prefix-cache reuse across runs
    prompt = f"Run identifier {uuid.uuid4()} -- session log begins.\n" + build(n) + q
    print(f"[calib] {per_line:.2f} tok/line -> {n} lines; exact={count_tokens(prompt):,}", flush=True)

    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": args.max_tokens, "temperature": 0.0,
               "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    resp = post("/v1/chat/completions", payload, stream=True)
    ttft = t_last = usage = None
    text, rea = [], []
    for raw in resp:
        s = raw.decode().strip()
        if not s.startswith("data: "): continue
        d = s[6:]
        if d == "[DONE]": break
        o = json.loads(d)
        if o.get("usage"): usage = o["usage"]
        for ch in o.get("choices", []):
            if ch.get("delta") is None: continue
            if ttft is None: ttft = time.time() - t0
            piece = ch["delta"].get("content") or ""
            rp = ch["delta"].get("reasoning") or ch["delta"].get("reasoning_content") or ""
            if piece or rp:
                t_last = time.time(); text.append(piece); rea.append(rp)

    ptok = (usage or {}).get("prompt_tokens", 0)
    ctok = (usage or {}).get("completion_tokens", 0)
    print("=" * 60)
    print(f"prompt tokens  : {ptok:,}")
    print(f"TTFT (prefill) : {ttft:.2f} s")
    print(f"PREFILL SPEED  : {ptok/ttft:,.0f} tok/s")
    if t_last and ctok > 1:
        dec = t_last - (t0 + ttft)
        if dec > 0: print(f"decode speed   : {ctok/dec:,.1f} tok/s")
    print("=" * 60)
    hay = ("".join(text) + " " + "".join(rea)).upper()
    ok = True
    for _, key, val in NEEDLES:
        hit = val.upper() in hay; ok &= hit
        print(f"  {key:<8} {val:<14} {'FOUND' if hit else 'MISSING'}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")

main()
