#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Repeatable Qwen3.8-Flash-Next latency/throughput benchmark.

The benchmark salts every prompt to avoid prefix-cache hits, forces an exact
number of output tokens, and snapshots vLLM speculative-decoding metrics.
Results are emitted as JSON for comparing server configurations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


FILLER = (
    "Entry {i:06d}: the quarterly logistics audit recorded a routine variance "
    "in the northbound depot inventory.\n"
)
TASKS = {
    "prose": (
        "Write continuous narrative prose about the history of maritime "
        "navigation. Do not use headings or lists."
    ),
    "code": (
        "Write a complete, heavily commented Python implementation of a "
        "red-black tree with insert, delete, and search."
    ),
}


@dataclass
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    elapsed_s: float
    decode_tps: float
    started_s: float
    ended_s: float
    output_sha256: str


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


class Client:
    def __init__(self, base: str, model: str, timeout: int) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def request(self, method: str, path: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def tokenize(self, text: str) -> int:
        with self.request(
            "POST", "/tokenize", {"model": self.model, "prompt": text}
        ) as response:
            return int(json.loads(response.read())["count"])

    def metrics(self) -> dict[str, float]:
        try:
            with self.request("GET", "/metrics") as response:
                lines = response.read().decode().splitlines()
        except Exception:
            return {}
        result: dict[str, float] = {}
        for line in lines:
            if line.startswith("#") or "spec_decode" not in line:
                continue
            name, _, raw = line.rpartition(" ")
            try:
                result[name] = float(raw)
            except ValueError:
                pass
        return result

    def generate(self, prompt: str, output_tokens: int) -> RequestResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        first = None
        last = None
        usage: dict = {}
        output_parts: list[str] = []
        with self.request("POST", "/v1/chat/completions", payload) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("choices"):
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    delta = event["choices"][0].get("delta", {})
                    for key in ("reasoning_content", "content"):
                        if isinstance(delta.get(key), str):
                            output_parts.append(delta[key])
        ended = time.perf_counter()
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", output_tokens))
        ttft = (first or ended) - started
        decode_window = max((last or ended) - (first or ended), 1e-9)
        decode_tps = max(completion_tokens - 1, 0) / decode_window
        return RequestResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft,
            elapsed_s=ended - started,
            decode_tps=decode_tps,
            started_s=started,
            ended_s=ended,
            output_sha256=hashlib.sha256("".join(output_parts).encode()).hexdigest(),
        )


def salted_prompt(body: str, nonce: str) -> str:
    return f"Benchmark nonce {nonce}.\n{body}"


def build_context(client: Client, target_tokens: int) -> tuple[str, int]:
    sample = "".join(FILLER.format(i=i) for i in range(256))
    tokens_per_line = client.tokenize(sample) / 256
    lines = max(1, int(target_tokens / tokens_per_line))
    context = "".join(FILLER.format(i=i) for i in range(lines))
    return context, client.tokenize(context)


def summarize_requests(results: list[RequestResult]) -> dict:
    start = min(item.started_s for item in results)
    end = max(item.ended_s for item in results)
    return {
        "requests": len(results),
        "prompt_tokens": sum(item.prompt_tokens for item in results),
        "completion_tokens": sum(item.completion_tokens for item in results),
        "wall_s": end - start,
        "aggregate_output_tps": (
            sum(item.completion_tokens for item in results) / (end - start)
        ),
        "ttft_p50_s": percentile([item.ttft_s for item in results], 0.5),
        "ttft_p95_s": percentile([item.ttft_s for item in results], 0.95),
        "e2e_p50_s": percentile([item.elapsed_s for item in results], 0.5),
        "e2e_p95_s": percentile([item.elapsed_s for item in results], 0.95),
        "output_sha256": [item.output_sha256 for item in results],
    }


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: value - before.get(key, 0.0)
        for key, value in after.items()
        if value - before.get(key, 0.0) != 0
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8888")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--single-repeats", type=int, default=2)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--requests-per-level", type=int, default=16)
    parser.add_argument("--prefill-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    client = Client(args.base, args.model, args.timeout)
    context, context_tokens = build_context(client, args.prefill_tokens)
    report: dict = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parameters": vars(args) | {"output": str(args.output)},
        "calibrated_context_tokens": context_tokens,
    }

    client.generate(salted_prompt("Reply with a short greeting.", "warmup"), 32)
    metrics_before = client.metrics()

    singles: dict[str, list[dict]] = {}
    for task_name, task in TASKS.items():
        rows = []
        for repeat in range(args.single_repeats):
            result = client.generate(
                salted_prompt(task, f"single-{task_name}-{repeat}"), args.decode_tokens
            )
            rows.append(asdict(result))
            print(
                f"single {task_name}: {result.decode_tps:.2f} tok/s, "
                f"TTFT {result.ttft_s:.3f}s",
                flush=True,
            )
        singles[task_name] = rows
    report["single_stream"] = singles

    prefill = client.generate(
        salted_prompt(
            context + "\nSummarize the log in continuous prose.", "prefill-32k"
        ),
        32,
    )
    report["prefill"] = asdict(prefill) | {
        "prefill_tps": prefill.prompt_tokens / prefill.ttft_s
    }
    print(
        f"prefill: {report['prefill']['prefill_tps']:.1f} tok/s, "
        f"TTFT {prefill.ttft_s:.3f}s",
        flush=True,
    )

    concurrency_report: dict[str, dict] = {}
    for concurrency in [int(value) for value in args.concurrency.split(",")]:
        count = max(args.requests_per_level, concurrency * 2)
        start_gate = threading.Event()

        def one_request(index: int) -> RequestResult:
            prompt = salted_prompt(
                TASKS["prose"] + f" This is independent request {index}.",
                f"concurrency-{concurrency}-{index}",
            )
            start_gate.wait()
            return client.generate(prompt, args.decode_tokens)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = [pool.submit(one_request, index) for index in range(count)]
            start_gate.set()
            results = [future.result() for future in pending]
        summary = summarize_requests(results)
        concurrency_report[str(concurrency)] = summary
        print(
            f"concurrency {concurrency}: "
            f"{summary['aggregate_output_tps']:.2f} aggregate tok/s, "
            f"TTFT p95 {summary['ttft_p95_s']:.3f}s",
            flush=True,
        )
    report["concurrency"] = concurrency_report

    metrics_after = client.metrics()
    report["spec_decode_metric_delta"] = metric_delta(metrics_before, metrics_after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
