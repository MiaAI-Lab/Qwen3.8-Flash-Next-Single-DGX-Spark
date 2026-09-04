#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Print percentage deltas between two perf_sweep.py result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def change(before: float, after: float) -> str:
    return f"{(after / before - 1) * 100:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    before = json.loads(args.baseline.read_text())
    after = json.loads(args.candidate.read_text())

    print(f"{before['label']} -> {after['label']}")
    for task in before["single_stream"]:
        a = mean([row["decode_tps"] for row in before["single_stream"][task]])
        b = mean([row["decode_tps"] for row in after["single_stream"][task]])
        print(f"single {task:8s} {a:8.2f} -> {b:8.2f} tok/s  {change(a, b)}")
    a = before["prefill"]["prefill_tps"]
    b = after["prefill"]["prefill_tps"]
    print(f"32k prefill     {a:8.1f} -> {b:8.1f} tok/s  {change(a, b)}")
    for concurrency in before["concurrency"]:
        a = before["concurrency"][concurrency]["aggregate_output_tps"]
        b = after["concurrency"][concurrency]["aggregate_output_tps"]
        print(f"concurrency {concurrency:>2s}  {a:8.2f} -> {b:8.2f} tok/s  {change(a, b)}")
        a = before["concurrency"][concurrency]["ttft_p95_s"]
        b = after["concurrency"][concurrency]["ttft_p95_s"]
        print(f"  TTFT p95       {a:8.3f} -> {b:8.3f} s      {change(a, b)}")
        a = before["concurrency"][concurrency]["e2e_p95_s"]
        b = after["concurrency"][concurrency]["e2e_p95_s"]
        print(f"  E2E p95        {a:8.3f} -> {b:8.3f} s      {change(a, b)}")

    def spec_acceptance(report: dict) -> float:
        metrics = report["spec_decode_metric_delta"]
        accepted = sum(
            value
            for name, value in metrics.items()
            if "spec_decode_num_accepted_tokens_total" in name
        )
        drafted = sum(
            value
            for name, value in metrics.items()
            if "spec_decode_num_draft_tokens_total" in name
        )
        return accepted / drafted

    a = spec_acceptance(before)
    b = spec_acceptance(after)
    print(f"MTP acceptance   {a:8.1%} -> {b:8.1%}        {change(a, b)}")

    before_hashes = [
        row["output_sha256"]
        for rows in before["single_stream"].values()
        for row in rows
    ] + [before["prefill"]["output_sha256"]]
    after_hashes = [
        row["output_sha256"]
        for rows in after["single_stream"].values()
        for row in rows
    ] + [after["prefill"]["output_sha256"]]
    for concurrency in before["concurrency"]:
        before_hashes.extend(before["concurrency"][concurrency]["output_sha256"])
        after_hashes.extend(after["concurrency"][concurrency]["output_sha256"])
    matched = sum(a == b for a, b in zip(before_hashes, after_hashes))
    print(f"exact output hashes matched: {matched}/{len(before_hashes)}")


if __name__ == "__main__":
    main()
