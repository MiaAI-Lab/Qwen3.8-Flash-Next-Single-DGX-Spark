#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Mixed traffic: what a long prefill does to streams that are already decoding.

sparkDash measures decode-only and prefill-only; the agent harness this box
exists for does both at once (72k-token prompts arriving while earlier turns
are still streaming). Chunked prefill puts a MAX_NUM_BATCHED_TOKENS-wide chunk
in the same engine step as every co-scheduled decode, so the decoders'
inter-token latency during a prefill is set by the chunk width, not by the
decode batch.

    2 prose streams start decoding (ignore_eos, temperature 0, reasoning off).
    At --inject-after seconds one ~64k-token prompt is submitted.
    ITL of the two decoders is reported for the prefill window (from the
    injection to the big prompt's first token) and, as the control, for the
    quiet window before it.

    python3 bench/mixed.py --tag E0 --out logs/overnight-2026-09-05.jsonl

Requires an idle server: any other traffic lands in the same engine steps.
"""
import argparse, json, statistics, sys, threading, time, urllib.request

BASE, MODEL = "http://localhost:8888", "qwen3.8-flash-next"
# Same filler decodebench.py builds its contexts from: ~25 tokens per entry.
FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")
PROSE = ("Write a flowing, continuous essay about the history of maritime "
         "navigation. Use ordinary narrative prose, no lists, no headings.")
BIG_TASK = "In one short sentence, what kind of document is the log above?"


def build_ctx(t):                       # bench/decodebench.py:build_ctx
    return "".join(FILLER.format(i=i) for i in range(max(1, int(t / 25))))


def post(payload, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream(payload, marks, t_origin):
    """Append (arrival_time, delta_tokens) for every chunk that carries text."""
    resp = post(payload)
    for raw in resp:
        s = raw.decode().strip()
        if not s.startswith("data: "):
            continue
        d = s[6:]
        if d == "[DONE]":
            break
        o = json.loads(d)
        for ch in o.get("choices", []):
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning"):
                marks.append(time.time() - t_origin)


class Decoder(threading.Thread):
    def __init__(self, idx, max_tokens, t_origin):
        super().__init__(daemon=True)
        self.idx, self.max_tokens, self.t_origin = idx, max_tokens, t_origin
        self.marks, self.error = [], None

    def run(self):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": f"({self.idx}) " + PROSE}],
            "max_tokens": self.max_tokens, "min_tokens": self.max_tokens,
            "ignore_eos": True, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            stream(payload, self.marks, self.t_origin)
        except Exception as e:
            self.error = repr(e)


class BigPrompt(threading.Thread):
    def __init__(self, ctx, t_origin):
        super().__init__(daemon=True)
        self.ctx, self.t_origin = ctx, t_origin
        self.submitted = self.ttft = None
        self.prompt_tokens = 0
        self.error = None

    def run(self):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": self.ctx + "\n\n" + BIG_TASK}],
            "max_tokens": 16, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
        }
        self.submitted = time.time() - self.t_origin
        try:
            resp = post(payload)
            for raw in resp:
                s = raw.decode().strip()
                if not s.startswith("data: "):
                    continue
                d = s[6:]
                if d == "[DONE]":
                    break
                o = json.loads(d)
                if o.get("usage"):
                    self.prompt_tokens = o["usage"].get("prompt_tokens", 0)
                for ch in o.get("choices", []):
                    delta = ch.get("delta") or {}
                    if (delta.get("content") or delta.get("reasoning")) and self.ttft is None:
                        self.ttft = time.time() - self.t_origin - self.submitted
        except Exception as e:
            self.error = repr(e)


def itl_stats(marks, lo, hi):
    """Gaps between consecutive token arrivals that both fall inside [lo, hi)."""
    gaps = [(b - a) * 1000 for a, b in zip(marks, marks[1:]) if lo <= a < hi]
    if not gaps:
        return None
    gaps_sorted = sorted(gaps)

    def pct(p):
        return gaps_sorted[min(len(gaps_sorted) - 1, int(round(p / 100 * (len(gaps_sorted) - 1))))]

    return {"n": len(gaps), "mean_ms": round(statistics.fmean(gaps), 1),
            "p50_ms": round(pct(50), 1), "p95_ms": round(pct(95), 1),
            "p99_ms": round(pct(99), 1), "max_ms": round(max(gaps), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--decoders", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=800)
    ap.add_argument("--context", type=int, default=64000)
    ap.add_argument("--inject-after", type=float, default=5.0)
    ap.add_argument("--out", default="logs/overnight-2026-09-05.jsonl")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    ctx = build_ctx(a.context)
    t0 = time.time()
    decoders = [Decoder(i, a.decode_tokens, t0) for i in range(a.decoders)]
    for d in decoders:
        d.start()
    time.sleep(a.inject_after)
    big = BigPrompt(ctx, t0)
    big.start()
    big.join(timeout=1800)
    for d in decoders:
        d.join(timeout=1800)

    if big.ttft is None:
        print(f"big prompt produced no token (error={big.error})", file=sys.stderr)
    win_lo = big.submitted
    win_hi = big.submitted + (big.ttft or 0)
    quiet = [itl_stats(d.marks, 0.0, win_lo) for d in decoders]
    during = [itl_stats(d.marks, win_lo, win_hi) for d in decoders]
    after = [itl_stats(d.marks, win_hi, 1e9) for d in decoders]
    tokens_in_window = sum(sum(1 for m in d.marks if win_lo <= m < win_hi) for d in decoders)
    window_s = max(win_hi - win_lo, 1e-9)

    row = {
        "kind": "mixed", "tag": a.tag, "note": a.note,
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decoders": a.decoders, "decode_tokens": a.decode_tokens,
        "big_prompt_tokens": big.prompt_tokens, "big_ttft_s": round(big.ttft, 2) if big.ttft else None,
        "inject_at_s": round(win_lo, 2),
        "prefill_window_s": round(window_s, 2),
        "decode_tps_in_window": round(tokens_in_window / window_s, 2),
        "tokens_in_window": tokens_in_window,
        "itl_quiet": quiet, "itl_during_prefill": during, "itl_after": after,
        "errors": [d.error for d in decoders] + [big.error],
        "wall_s": round(time.time() - t0, 1),
    }
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"[{a.tag}] big prompt {big.prompt_tokens:,} tok, TTFT {row['big_ttft_s']} s "
          f"(injected at {row['inject_at_s']} s)")
    print(f"        decode during that window: {row['decode_tps_in_window']} tok/s aggregate "
          f"over {row['tokens_in_window']} tokens")
    for label, stats in (("quiet ", quiet), ("prefill", during), ("after ", after)):
        for i, st in enumerate(stats):
            print(f"        ITL {label} stream {i}: {st}")


if __name__ == "__main__":
    main()
