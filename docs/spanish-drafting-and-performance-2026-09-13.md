# Serving this recipe in Spanish: drafting, sampling and benchmarks

Notes from running `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark` on a
Spanish-language DGX Spark. Three findings, none of them visible from the
measurements already in this repository, and a harness for each.

| | |
|---|---|
| **1** | A draft vocabulary built from a dictionary destroys Spanish. Use the one this repo ships. |
| **2** | Sampling parameters differ by mode, and mixing them causes language mixing. |
| **3** | The published tok/s and the tok/s you measure are different benchmarks, not a deployment problem. |
| **4** | Spanish output carries ~40x more lexical corruption than English, and it comes from the checkpoint, not from drafting. |

> **Correction, 2026-09-14, from the fork these notes came from.** Finding 1
> stands as written: a dictionary-built vocabulary does degrade drafting, and
> rebuilding it from text is the right move. What it does not do is fix the
> Spanish. The revision of this document that reached this repository presented
> finding 1 as the *cause* of the language drift. That attribution is wrong.
> Section 4 records the measurements: the drift reproduces with the corrected
> 65k vocabulary loaded, at the sampling of section 2, and with speculative
> decoding switched off entirely. Fixing the vocabulary bought acceptance rate,
> which is speed. It did not buy correctness. The cause is in the checkpoint.

---

## 1. Build the draft vocabulary from text, never from a dictionary

`files/draft_vocab_en_code_47k.txt` ships with this repo and is English and
code. It needs no Spanish to serve Spanish correctly. **Three independent
deployments converged on this exact file** — which is itself evidence.

A locally built replacement, intended to add Spanish coverage, degraded output
badly enough to force a rollback. Reconstructing how it was built took two
passes, and the first conclusion was wrong.

**What it was built from.** Real corpora — 11.7 MB of actual session text,
Spanish and English — plus `es_dict.txt`, a Spanish dictionary: 885,412 words
on a single line, every inflected form exactly once. Re-running
`files/build_draft_vocab.py` over those inputs reproduces the vocabulary
**exactly**, 100% id-for-id, which settles what went in.

**What actually broke.** Not the dictionary. Measured against the 65k it
replaced, the vocabulary kept only **286** ids below 400 — the byte-fallback
range, the pieces BPE uses to assemble `ñ`, `á`, `é` and every multi-byte UTF-8
sequence. Upstream's shipped file keeps **376**.

But rebuilding from the corpora **without** the dictionary yields the same 286.
The dictionary is not what dropped them.

**`build_draft_vocab.py` does not pin the byte-fallback range.** It keeps
special and added tokens unconditionally and says so; byte-fallback ids get no
such protection. Any corpus that does not exercise them loses them, and 11.7 MB
of conversation logs does not. Upstream's file keeps 376 because its corpus is
513 MiB of wikitext — broad enough for those ids to earn their place on
frequency alone.

So the failure is a **missing guard in the builder**, exposed by a corpus that
was too small and too narrow. The dictionary was, if anything, harmless: the
corpora alone yielded 33,380 distinct ids, and the dictionary filled the
remaining slots up to 44,138 with rare Spanish words. It displaced nothing.

The first version of this document said the dictionary destroyed the frequency
signal and that this cost the byte-fallback tokens. That was an inference from
filenames and timestamps, and the measurement above refutes it.

### If a Spanish-aware vocabulary is ever wanted

- **Pin the byte-fallback range unconditionally**, the way special and added
  tokens already are. This is the actual fix, and it is one line. Without it the
  builder is only safe on corpora large and varied enough to exercise those ids
  by frequency, which is a property nobody checks.
- **Start from the shipped vocabulary as a floor** and only add. Rebuilding from
  scratch is what makes it possible to lose ids that already worked.
- **Use text with natural frequencies** — Spanish Wikipedia, the way
  `english.txt` uses wikitext-103. A dictionary is not fatal on its own, but it
  contributes no frequency information: every form weighs one.
- **Gate it on Spanish output** before serving it to anyone.

### On the correctness argument

The CHANGELOG argues that a reduced vocabulary cannot change output: drafting is
greedy, and the rejection sampler's greedy branch is
`accepted = target_argmax == draft_sampled`, so a draft survives only when it
equals the target's own choice. Poor coverage costs speed, never correctness.

Two boundaries on that, both worth stating:

- **The evidence behind it is MGSM in English and Chinese**, and the CHANGELOG
  says plainly that the result "does not establish universal quality
  preservation". Spanish was never tested.
- **It is the greedy branch.** Above temperature 0 the verification is
  probabilistic, and there the draft distribution can influence what is
  accepted. That is a plausible route for a reduced vocabulary to affect output
  and it is **unverified** — settling it needs an A/B with and without the
  reduced vocabulary at production temperature.

### Measured: the shipped vocabulary covers 64% of Spanish output

The speed win from reduced-vocabulary drafting is two effects pulling opposite
ways. The byte saving is **language-independent** — the draft head shrinks from
1.18 to 0.22 GiB whatever is being written. Acceptance is **not**: drafts for
tokens outside the vocabulary are rejected, and that gives the saving back.

The CHANGELOG measures this for Chinese — 50.6% coverage, throughput gain
"nothing measurable" — and concludes out-of-vocabulary traffic is break-even.
Nobody measured Spanish. Measured here, over real Spanish model output:

| | coverage | accepted/draft | tok/s |
|---|---|---|---|
| English | 98.9% | 1.77 | 45.9 |
| **Spanish** | **64.4%** | **1.01** | **33.8** |

Spanish sits closer to Chinese than to English. The drafter lands 1.01 tokens
per proposal against 1.77, and 12 tok/s go with it. **The published speed win is
largely an English win.**

### Extending the vocabulary instead of rebuilding it

`files/build_draft_vocab_extend.py` takes the shipped vocabulary as a **floor**
and only adds. It cannot lose an id that already worked, which is the failure
mode above. It also pins the byte-fallback range unconditionally, before looking
at any frequency.

Built with 668 MiB of Spanish Wikipedia (natural frequencies, as `english.txt`
uses wikitext-103) plus model output, to 65,536 rows:

| | shipped 47k | extended 65k |
|---|---|---|
| coverage on Spanish output | 64.4% | **99.1%** |
| byte-fallback ids | 376 | **400** |
| byte saving retained | 100% | **91%** |
| Spanish quality gate | clean | **clean** |

### Measured with an interleaved A/B

Prompt choice moves acceptance by 20-30% on this model, so single-prompt
comparisons are worthless. Five prompts per language, two repetitions each, the
same ten measurements in every arm, in **A B B A** order so any monotonic drift
cancels. Sixty seconds of settle after each boot, identical in all four.

| arm | vocab | ES tok/s | ES acc | EN tok/s | EN acc |
|---|---|---|---|---|---|
| A | 47k | 32.99 | 0.96 | 40.36 | 1.45 |
| B | 65k | 41.96 | 1.61 | 40.97 | 1.49 |
| B | 65k | 41.88 | 1.51 | 41.68 | 1.55 |
| A | 47k | 32.20 | 0.92 | 40.17 | 1.42 |

| | 47k | 65k | |
|---|---|---|---|
| **Spanish tok/s** | 32.60 | **41.92** | **+28.6%** |
| **Spanish accepted/draft** | 0.94 | **1.56** | **+66%** |
| English tok/s | 40.27 | 41.33 | +2.6% |
| English accepted/draft | 1.435 | 1.52 | +6% |

**Spanish gains 28.6% and English pays nothing** — it is marginally better,
though +2.6% is at the edge of what this setup resolves.

The validity check is the within-arm agreement: the two 47k arms agree to 2.4%
and the two 65k arms to 0.2%, against a 20-30% spread *between prompts*. Same
prompts in both arms is what separates signal from noise here. The two A arms
are 52 minutes apart and give the same answer, which excludes warm-up and
thermal drift.

The size trade is gentle in this range — the draft head is 0.22 GiB at 47k and
0.31 at 65k against 1.18 full — so buying coverage for a second language costs
9% of the byte saving and returns 66% of acceptance in that language.

Quality was re-gated, not assumed: a new vocabulary is a new artefact and
`bench/audit-spanish.py` was run against it in full. Zero replacement
characters, zero drift markers.

---

## 2. Sampling parameters depend on the mode

Qwen publishes **different** settings for the two modes
([model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)):

| | thinking | instruct (thinking off) |
|---|---|---|
| temperature | 1.0 | **0.7** |
| top_p | 0.95 | **0.80** |
| top_k | 20 | 20 |
| presence_penalty | 0.0 | **1.5** |
| repetition_penalty | 1.0 | 1.0 |

The checkpoint's `generation_config.json` carries the **thinking** values and
vLLM applies them automatically; the boot log says so:

```
Default vLLM sampling parameters have been overridden by the model's
generation_config.json: {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}
```

So a server left alone is correct for thinking traffic. **A client that turns
thinking off without also sending the instruct values runs the combination the
model card warns about:** *"using a higher value may occasionally result in
language mixing and a slight decrease in model performance."*

Observed here, not hypothetical: an audit pass at thinking-off with
thinking-mode sampling answered a Spanish prompt in English, on an unrelated
subject. It did not reproduce in three retries once sampling matched the mode.

Practical consequence for anyone with a proxy in front of this model: pinning
`temperature: 1.0` unconditionally is right for thinking traffic and wrong for
everything else.

---

## 3. The performance gap that is not a gap

A day was spent here asking why this host measured **37.15 tok/s** against the
48.7 published in the README — decomposing the engine step, measuring MTP
acceptance per position, sampling clocks under load. Nothing explained it.

Another public recipe for this same runtime publishes two benchmark families
side by side, and the answer is the difference between them:

| benchmark | C1 |
|---|---|
| Decode Bench (synthetic counting stream) | **66.17 tok/s** |
| Agent / tool JSON | 37.1 |
| Code edit | 42.8 |
| Generic coding | 46.9 |
| Long-context review | 37.8 |

Our 37.15 matched their "Agent / tool JSON" (37.1) and "Long-context review"
(37.8) to the decimal. **The published peaks come from a counting stream;
realistic work lands at 37-47.** Nothing was ever slow.

Both belong in an evaluation — a counting stream is a clean decode measurement,
realistic work reveals the cost of actual work. Comparing one against the other
does not.

### Measured here, their protocol

`bench/structured-protocol.py`. 400 tokens, temperature 0, top_p 1, thinking
off, warm server. Shipped 47k vocabulary, `MAX_NUM_SEQS=4`, KV pool 1,057,362
tokens, native 262,144 context.

| | this host | their publish |
|---|---|---|
| Decode Bench C1, per stream | 61.41 | 66.17 |
| Decode Bench C2, aggregate | 104.13 | 109.70 |
| Decode Bench C4, aggregate | **182.54** | 179.31 |
| Agent / tool JSON C1 | 46.6 | 37.1 |
| Code edit C1 | 44.0 | 42.8 |
| Generic coding C1 | 54.1 | 46.9 |
| Long-context review C1 | 34.7 | 37.8 |

5-7% below at C1 and C2, slightly above at C4 — while running
`MAX_NUM_SEQS=4` against their 8.

**Caveat**: their prompt *text* is not published, so only the Decode Bench row
compares cleanly. The production-suite prompts here are equivalent in spirit,
not identical, and MTP acceptance is prompt-dependent.

### Why MAX_NUM_SEQS stays at 4

`.env.sample` ships 4 while the README's decode table is measured at 8, and
`docs/overnight-2026-09-05.md` gives the reason: eight concurrent long requests
will contend and preempt, and no long-context run has been done. This host's
traffic is long sessions, which is exactly the untested case. The other recipe
runs 8 without publishing a long-context test either.

---

## 4. Cross-language leakage is the checkpoint's, not the pipeline's

Spanish output from this checkpoint contains words that do not exist in Spanish
and that no writer produces: `comenzana` for *comenzaban*, `bloqua` for
*bloquea*, `cabizajo` for *cabizbajo*. A distinct subset is spelled with letters
foreign to Spanish orthography — `generaziones`, `parezian`, `plastiko`,
`neblika`, `esperansa` — subword pieces belonging to a neighbouring language's
spelling, not random letter noise. The same signature appears when a reply slips
into Italian (`revizione` for *revisione*) and, at the extreme, when a whole
generation leaves Castilian and does not come back.

**It is specific to Spanish.** Same sampling, same day, ten matched prompts per
language:

| language | words | malformations | per 10k |
|---|---|---|---|
| Spanish | 16,792 | 29 | **17.3** |
| English | 22,236 | 1 (arguable) | **0.4** |

English is clean once contractions and dictionary gaps are removed. That rules
out generic quantization damage to the output head, which would not respect
language.

**It is not the drafter.** `MTP_NUM_SPECULATIVE_TOKENS=0`, then restored,
measuring the same battery in each state. The third arm exists because two arms
cannot separate an effect from run-to-run spread:

| arm | words | malformations | per 10k | generations drifting |
|---|---|---|---|---|
| MTP on (container A) | 13,628 | 168 | 123.3 | 2/30 |
| **MTP off** (container B) | 13,734 | 200 | **145.6** | 0/30 |
| MTP on (container C) | 13,625 | 232 | 170.3 | 1/30 |

A and C are **identical** configurations and differ by 47 points. That spread is
wider than anything switching MTP off produces, and the MTP-off arm lands
between them. Speculative decoding, the reduced draft vocabulary and the whole
drafting path are excluded.

**It is not the nucleus being skipped.** In
`vllm/v1/worker/gpu/sample/sampler.py` the only caller passing
`skip_top_k_top_p=True` is the non-speculative `sample()`, which applies the
filter itself; the speculative `_verify()` path takes the default and filters
the target logits before `rejection_sample`. Behaviourally, at `temperature 2.0`
both `top_k=1` and `top_p=0.01` return coherent prose while `top_k=0` with
`top_p=1.0` returns token salad. The filters work.

**Where the mass is.** Forcing the exact prefix that preceded each malformation
and reading the top-20 (`bench/probe-logprobs.py`):

| produced | correct | rank/p of correct | rank/p of malformed |
|---|---|---|---|
| `arancaba` | *arrancaba* | 1 / 0.479 | 8 / **0.016** |
| `nocha` | *noche* | 1 / 0.622 | 19 / **0.0017** |
| `chocoate` | *chocolate* | 1 / 0.241 | 8 / **0.016** |

Batched MoE decoding is not deterministic, so exact figures move a few points
between runs, but the shape is stable: the malformed continuation carries
**1-2% of the probability mass**, not a remote tail. `top_k=20` and `top_p=0.95`
both keep it and `temperature 1.0` samples it at that rate. Lowering temperature
sharpens the distribution and is the only sampling lever that moves the rate; it
does not remove the mass.

**What the mass is.** The per-layer-embedding n-gram table — 320,001,536 rows of
160 — is the part of this architecture that carries surface-form, sub-word
knowledge, and in the NVFP4 mirror it is quantized with everything else. A token
touches 16 of those rows, so the error does not average out across a large
matmul the way it does in a dense layer; it lands directly on the spelling of
the next piece. That is consistent with the two observations above: English,
whose n-grams are the best covered, is unaffected, and the damage shows up as
neighbouring-language subword pieces rather than noise.

Serving NVIDIA's checkpoint, which keeps that table at higher precision, on the
same host and the same sampling: **30.0 malformations per 10k against 123.3, and
0/30 generations drifting against 2/30.** That is the test, and it is the only
change between the two arms.

---

## The harnesses

`bench/audit-spanish.py` — the Spanish quality gate that did not exist. Eight
long generations (essays, narrative, a formal report, Python with Spanish
docstrings, JSON) plus a five-turn conversation. Analysis is **per paragraph**,
because a reply can begin in Castilian and drift halfway through; looking only
at the whole response hides that. Flags Unicode replacement characters and
markers for Asturian, Galician, Catalan and Portuguese separately. Selects
sampling from the mode.

Result with the shipped 47k vocabulary: across four runs, two vocabularies, two
temperatures and both sampling regimes — roughly **60,000 tokens of Spanish
output** — **zero replacement characters and zero dialect-drift markers**.
Accents and `ñ` render correctly throughout.

It also carries a b/v orthography check, and an honest note about it: **that
check has never caught a real error.** Across four audits it produced only false
positives from its own regexes — `la` and `el` from a capturing group inside a
lookahead, then the correct `anduvo`, then the correct `hacia`. The one
confirmed defect, *reescriví* for *reescribí*, was found by a user in
production. That output is impeccable Castilian, correctly accented, with no
replacement characters and no dialect markers: **it passes every language check
here.** Language identification is not spelling verification. Treat the check as
an unvalidated safety net.

`bench/structured-protocol.py` — replicates the published structured protocol
so numbers taken here can sit beside other recipes' published tables.

`bench/verify-smoke.py` — the short version, for a quick check after a boot.

**Run the audit with thinking off.** With thinking on, long prompts spend the
entire budget reasoning and `content` returns empty on a perfectly healthy
server. The README's sanity test warns about this at 200 tokens; it also happens
at 2,000 when the prompt asks for a thousand words.

---

## A measurement trap

**Reasoning budget.** Through an OpenAI-compatible proxy, `enable_thinking`
defaults on and every reply spends 150-320 tokens reasoning before writing. A
one-line greeting cost 168 tokens of which 158 were reasoning. With a tight
`max_tokens` the client receives an empty `content` and no error.
