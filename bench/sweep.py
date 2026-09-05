#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Concurrent decode sweep through sparkDash, with /metrics deltas per level.

One sparkDash decode-bench job per concurrency level, so the vLLM counters can
be snapshotted immediately before and after each level:

    ms/step      = d(inter_token_latency_seconds_sum) / d(count) * 1000
    tokens/step  = 1 + d(spec_decode_num_accepted_tokens_total) / d(spec_decode_num_drafts_total)
    per-position = d(spec_decode_num_accepted_tokens_per_pos_total[pos]) / d(drafts)

Host memory (MemAvailable/MemFree minima) is sampled every second during the
level, and `journalctl -k` NV_ERR_NO_MEMORY lines added while the level ran are
counted (the driver's earliest out-of-memory signal; readable without sudo).
One JSON line per level is appended to --out.

    python3 bench/sweep.py --tag K3 --streams 1 2 4 --prompt prose code --repeats 3

Requires an idle server: stop anything else using :8888 first, or the counters
and the sparkDash numbers both include foreign traffic.
"""
import argparse, json, re, subprocess, sys, threading, time, urllib.error, urllib.request

DASH = "http://localhost:5555/api/sparks/spark-1/llm/bench"
METRICS = "http://localhost:8888/metrics"
COUNTERS = [
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:generation_tokens_total",
]
LINE = re.compile(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$')


def http(url, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def snapshot():
    out = {}
    with urllib.request.urlopen(METRICS, timeout=10) as r:
        for raw in r.read().decode().splitlines():
            m = LINE.match(raw)
            if not m:
                continue
            name, labels, val = m.groups()
            if name in COUNTERS:
                out[name] = float(val)
            elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
                pos = re.search(r'position="(\d+)"', labels or "").group(1)
                out[f"pos{pos}"] = float(val)
    return out


def nvrm_count(since, until):
    """NV_ERR_NO_MEMORY lines the kernel log gained in [since, until].

    Same source and framing as files/memwatch.sh: journalctl -k needs no sudo
    here, and --since/--until are inclusive at second granularity.
    """
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--since", since, "--until", until, "-q"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    return sum(1 for line in out.splitlines() if "NV_ERR_NO_MEMORY" in line)


def meminfo():
    d = {}
    for raw in open("/proc/meminfo"):
        k, v = raw.split(":")
        if k in ("MemAvailable", "MemFree"):
            d[k] = int(v.split()[0]) / 2**20
    return d


class MemMin(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.avail = self.free = 1e9

    def run(self):
        while not self.stop.is_set():
            m = meminfo()
            self.avail = min(self.avail, m["MemAvailable"])
            self.free = min(self.free, m["MemFree"])
            self.stop.wait(1)


def run_level(s, prompt, max_tokens):
    while True:
        try:
            job = http(DASH, {"port": 8888, "concurrencies": [s], "maxTokens": max_tokens, "promptType": prompt})
            break
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print("  sparkDash busy, waiting 10 s", file=sys.stderr); time.sleep(10); continue
            raise
    bid = job["benchId"]
    while True:
        time.sleep(3)
        try:
            job = http(f"{DASH}/{bid}")
        except Exception as e:                       # transient; the job keeps running
            print(f"  poll failed ({e}), retrying", file=sys.stderr); continue
        if job["status"] in ("completed", "failed", "cancelled"):
            return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="configuration label, e.g. K3 or K2")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--prompt", nargs="+", default=["prose"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", default="logs/overnight-2026-09-05.jsonl")
    ap.add_argument("--note", default="", help="free text carried into every row (config detail)")
    a = ap.parse_args()

    running = snapshot()
    order = list(a.streams)
    print(f"{'tag':<6}{'prompt':<8}{'S':>3}{'rep':>4}{'ms/step':>9}{'tok/step':>9}{'p1/p2/p3':>18}{'dash tok/s':>11}{'ttft ms':>9}{'avail':>7}{'free':>6}{'nvrm':>6}")
    for rep in range(a.repeats):
        seq = order if rep % 2 == 0 else order[::-1]
        for s in seq:
            for prompt in a.prompt:
                nv_since = time.strftime("%Y-%m-%d %H:%M:%S")
                before = snapshot(); mm = MemMin(); mm.start(); t0 = time.time()
                job = run_level(s, prompt, a.max_tokens)
                mm.stop.set(); after = snapshot()
                nvrm = nvrm_count(nv_since, time.strftime("%Y-%m-%d %H:%M:%S"))
                d = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
                res = (job.get("results") or [{}])[0]
                drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
                row = {
                    "tag": a.tag, "prompt": prompt, "S": s, "rep": rep, "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "ms_per_step": d["vllm:inter_token_latency_seconds_sum"] / max(d["vllm:inter_token_latency_seconds_count"], 1) * 1000,
                    "tok_per_step": 1 + d.get("vllm:spec_decode_num_accepted_tokens_total", 0) / drafts if drafts else 1.0,
                    "per_pos": [d.get(f"pos{i}", 0) / drafts if drafts else 0.0 for i in range(6) if f"pos{i}" in d],
                    "dash_aggregate_tps": res.get("aggregateDecodeTps"),
                    "dash_mean_tps": res.get("meanDecodeTps"),
                    "dash_ttft_ms": res.get("meanTtftMs"),
                    "streams_failed": res.get("streamsFailed"),
                    "gen_tokens": d.get("vllm:generation_tokens_total"),
                    "wall_s": time.time() - t0,
                    "mem_avail_min_gib": round(mm.avail, 2), "mem_free_min_gib": round(mm.free, 2),
                    "nvrm": nvrm, "note": a.note,
                    "benchId": job.get("benchId"), "status": job.get("status"),
                }
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                pp = "/".join(f"{x:.2f}" for x in row["per_pos"][:3])
                print(f"{a.tag:<6}{prompt:<8}{s:>3}{rep:>4}{row['ms_per_step']:>9.1f}{row['tok_per_step']:>9.2f}{pp:>18}"
                      f"{(row['dash_aggregate_tps'] or 0):>11.1f}{(row['dash_ttft_ms'] or 0):>9.0f}{mm.avail:>7.1f}{mm.free:>6.1f}"
                      f"{(nvrm if nvrm is not None else -1):>6}", flush=True)


if __name__ == "__main__":
    main()
