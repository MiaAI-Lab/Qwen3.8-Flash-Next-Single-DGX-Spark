#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Concurrent structured-decode bench — no sparkDash required.

bench/sweep.py drives sparkDash's decode bench so vLLM's /metrics counters can
be snapshotted per level; it needs sparkDash running on :5555. This script is
the fallback for when it is not: a counting-style predictable stream (the same
workload shape sparkDash's "structured" prompt type uses), 400 completion
tokens, temperature 0, thinking off, N streams started together. Per-stream
decode tok/s comes from
first-delta -> last-byte per stream; aggregate is total tokens / wall clock.

Numbers here are NOT comparable to the README prose tables: the counting
stream is MTP's best case (near-deterministic continuation, high draft
acceptance), so it reads ~35% higher than prose single-stream. It exists so
results can be compared against structured-prompt numbers published elsewhere.

    python3 bench/structured.py                          # C1 C2 C4, 2 reps
    python3 bench/structured.py --streams 1 2 4 8 --reps 2 --port 8888
"""
import argparse, json, threading, time, urllib.request

PROMPT = ("Count from 1 to 200, one number per line, no commentary. "
          "Start now with 1.")


def one_stream(url, model, max_tokens, result, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{url}/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None; usage = {}; deltas = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                chunk = json.loads(p)
            except Exception:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if isinstance(delta.get("content"), str) or isinstance(delta.get("reasoning_content"), str):
                if first is None:
                    first = time.perf_counter()
                deltas += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    end = time.perf_counter()
    tok = int(usage.get("completion_tokens") or deltas or 1)
    decode_s = (end - first) if first else (end - start)
    result[idx] = {"tok": tok, "ttft": (first - start) if first else None,
                   "tps": tok / decode_s if decode_s > 0 else 0}


def level(url, model, max_tokens, c, reps):
    for rep in range(reps):
        results = [None] * c
        threads = [threading.Thread(target=one_stream, args=(url, model, max_tokens, results, i))
                   for i in range(c)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total_tok = sum(r["tok"] for r in results)
        per = sorted(r["tps"] for r in results)
        ttfts = [r["ttft"] for r in results if r["ttft"]]
        ttft_ms = sum(ttfts) / len(ttfts) * 1000 if ttfts else float("nan")
        print(f"C{c} rep{rep}: per-stream mean {sum(per)/len(per):.1f} (min {per[0]:.1f}) tok/s | "
              f"aggregate {total_tok/wall:.1f} tok/s | TTFT {ttft_ms:.0f} ms | "
              f"wall {wall:.1f}s | tok {total_tok}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1"
    if not a.no_warmup:
        print("warmup ...", flush=True)
        level(url, a.model, a.max_tokens, 1, 1)
    for c in a.streams:
        level(url, a.model, a.max_tokens, c, a.reps)


if __name__ == "__main__":
    main()
