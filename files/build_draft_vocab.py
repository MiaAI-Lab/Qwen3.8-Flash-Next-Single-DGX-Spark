#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Build the reduced draft vocabulary for files/patch_mtp_draft_vocab.py.

Counts token frequencies over a corpus and writes the most frequent ids, one
per line. The corpus that matters is the *model's own output distribution*,
because that is what the drafter has to predict -- not a general text corpus
and not the prompts.

  python3 files/build_draft_vocab.py corpus.jsonl --out draft_vocab.txt --size 32768

Reads .jsonl with a "text" field, or plain .txt. Always keeps every special /
added token, whatever its frequency: those are cheap (a few hundred rows) and
losing one costs acceptance at exactly the structural boundaries where drafts
are otherwise easiest.

Coverage, not size, is the number to tune on. Report prints the fraction of
corpus token occurrences the chosen vocabulary covers; the tokens it misses
are not errors, they are drafts the target model will reject.
"""
import argparse
import json
import os
import sys
from collections import Counter


CHUNK = 1 << 20  # tokenize ~1 MiB at a time; the corpora are hundreds of MiB


def iter_texts(path: str):
    """Yield bounded chunks of a corpus. `path` may carry a `:N` repeat weight,
    so a small in-distribution corpus can be given the same say as a large
    generic one."""
    repeat = 1
    if ":" in path and path.rsplit(":", 1)[1].isdigit():
        path, repeat = path.rsplit(":", 1)
        repeat = int(repeat)
    for _ in range(repeat):
        if path.endswith(".jsonl"):
            with open(path) as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line).get("text", "")
        else:
            with open(path, errors="replace") as handle:
                while True:
                    block = handle.read(CHUNK)
                    if not block:
                        break
                    yield block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+",
                    help="corpus files, each optionally suffixed :N to repeat it N times")
    ap.add_argument("--model", default=os.environ.get(
        "DRAFT_VOCAB_MODEL", "Mia-AiLab/Qwen3.8-Flash-Next-NVFP4"))
    ap.add_argument("--out", default="draft_vocab.txt")
    ap.add_argument("--size", type=int, default=32768)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = len(tok)

    counts: Counter[int] = Counter()
    docs = 0
    for path in args.corpus:
        for text in iter_texts(path):
            if not text:
                continue
            counts.update(tok(text, add_special_tokens=False)["input_ids"])
            docs += 1
            if docs % 200 == 0:
                print(f"  ... {docs} chunks, {sum(counts.values()):,} tokens",
                      file=sys.stderr, flush=True)
    total = sum(counts.values())
    if not total:
        print("ERROR: corpus produced no tokens", file=sys.stderr)
        sys.exit(1)

    special = set(tok.all_special_ids or [])
    added = getattr(tok, "added_tokens_encoder", {}) or {}
    special |= {int(i) for i in added.values()}
    special = {i for i in special if 0 <= i < vocab_size}

    ranked = [tid for tid, _ in counts.most_common()]
    keep: list[int] = sorted(special)
    seen = set(keep)
    for tid in ranked:
        if len(keep) >= args.size:
            break
        if tid not in seen:
            keep.append(tid)
            seen.add(tid)
    keep = sorted(seen)

    covered = sum(counts[t] for t in seen if t in counts)
    print(f"corpus:      {docs} documents, {total:,} token occurrences, "
          f"{len(counts):,} distinct ids")
    print(f"vocabulary:  {vocab_size:,} -> {len(keep):,} "
          f"({100.0 * len(keep) / vocab_size:.1f}%), "
          f"{len(special)} special/added kept unconditionally")
    print(f"coverage:    {100.0 * covered / total:.4f}% of corpus occurrences")
    miss = total - covered
    print(f"             {miss:,} occurrences ({100.0 * miss / total:.4f}%) fall "
          f"outside; those become rejected drafts, never wrong output")

    for cut in (8192, 16384, 32768, 65536, 131072):
        sub = set(sorted(special)) | set(ranked[:max(0, cut - len(special))])
        cov = sum(counts[t] for t in sub if t in counts)
        print(f"  size {cut:>7,}: coverage {100.0 * cov / total:7.4f}%")

    if args.report_only:
        return
    with open(args.out, "w") as handle:
        handle.write("".join(f"{t}\n" for t in keep))
    print(f"wrote {len(keep):,} ids -> {args.out}")


if __name__ == "__main__":
    main()
