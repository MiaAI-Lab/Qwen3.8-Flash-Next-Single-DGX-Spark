#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Decode-speed benchmark: separates CONTENT TYPE from CONTEXT LENGTH.

Uses ignore_eos so every run decodes exactly N tokens.

  python3 bench/decodebench.py --decode 600 --contexts 1000,600000 --temps 0.0

Findings that motivated the shape of this script:
  * decode is dominated by MTP acceptance, not context length
    (1k -> 600k costs only ~2-8%)
  * "copy from context" is the best case (~70 tok/s) because the MTP draft
    head predicts quoted text almost perfectly; genuine prose is ~40 tok/s.
    Do NOT quote a copy-heavy number as typical decode speed.
  * the "entropy" task at temperature 0 degenerates into repetition, which
    MTP then predicts easily -- read it only at temp 0.8.
"""
import json, time, argparse, urllib.request

BASE, MODEL = "http://localhost:8888", "qwen3.8-flash-next"
FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")
TASKS = {
 "prose":   "Write a flowing, continuous essay about the history of maritime "
            "navigation. Use ordinary narrative prose, no lists, no headings.",
 "code":    "Write a complete, heavily-commented Python implementation of a "
            "red-black tree with insert, delete and search.",
 "entropy": "Output a long list of random 12-character uppercase alphanumeric "
            "license keys, one per line, all different, no commentary.",
 "copy":    "Reproduce verbatim, in order, entries 000005 through 000034 from "
            "the log above. Output the lines exactly as they appear.",
}

def post(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def build_ctx(t):
    return "".join(FILLER.format(i=i) for i in range(max(1, int(t / 25))))

def run(ctx, task, n, temp):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": ctx + "\n\n" + task}],
               "max_tokens": n, "min_tokens": n, "ignore_eos": True,
               "temperature": temp, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.time(); resp = post("/v1/chat/completions", payload)
    ttft = t_last = usage = None
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
            t_last = time.time()
    ctok = (usage or {}).get("completion_tokens", n)
    ptok = (usage or {}).get("prompt_tokens", 0)
    win = (t_last - (t0 + ttft)) if (t_last and ttft) else None
    return ptok, ctok, ttft, ((ctok - 1) / win if win and win > 0 else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", type=int, default=600)
    ap.add_argument("--contexts", default="1000,600000")
    ap.add_argument("--temps", default="0.0,0.8")
    a = ap.parse_args()
    print(f"{'context':>9} {'temp':>5} {'content':<8} {'ptok':>9} {'ctok':>6} {'TTFT s':>9} {'dec tok/s':>10}")
    print("-" * 62)
    for c in [int(x) for x in a.contexts.split(",")]:
        ctx = build_ctx(c)
        for t in [float(x) for x in a.temps.split(",")]:
            for name, task in TASKS.items():
                p, ct, tt, dec = run(ctx, task, a.decode, t)
                print(f"{c:>9,} {t:>5.1f} {name:<8} {p:>9,} {ct:>6,} {tt:>9.2f} {dec:>10.1f}", flush=True)

main()
