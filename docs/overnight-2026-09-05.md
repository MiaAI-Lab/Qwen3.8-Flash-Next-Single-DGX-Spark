# Overnight run, 2026-09-05/06: static K, index sharing, kernels, chunk width

Unattended pass through `docs/synthesis-astra-fable-2026-09-05.md` §6 on the
one DGX Spark this repo is written for (`spark3`, GB10, 121.6 GiB unified).
Every number below was measured during this run and is labelled with the
configuration it came from. Nothing here is projected unless it says
"estimated".

Method, unchanged for every experiment:

- One launch per configuration, `./stop.sh` then `./start.sh`, watchdog on
  (`files/memwatch.sh`), `HOST_RESERVE_GIB=26`, GMU never pinned.
- **Standard sweep** = sparkDash decode bench, prose, 600 tokens, at
  S ∈ {1, 2, 4, 8}, three repeats, order alternated across repeats
  (1,2,4,8 / 8,4,2,1 / 1,2,4,8), one sparkDash job per level so
  `http://localhost:8888/metrics` can be snapshotted on either side of it.
  `bench/sweep.py` derives ms per engine step, tokens per step and
  per-position acceptance from those deltas, samples `MemAvailable` and
  `MemFree` every second, and counts `journalctl -k` `NV_ERR_NO_MEMORY` lines
  added during the level. Raw rows: `logs/overnight-2026-09-05.jsonl`.
- **Noise floor** on this host is ±5% on a single sweep. A difference counts
  only when the mean moves more than 5% in the same direction in all three
  repeats at that S.
- A configuration the watchdog stops, or whose `MemAvailable` minimum drops
  below 10 GiB, or that logs `NV_ERR_NO_MEMORY` while `MemAvailable` is under
  ~10 GiB, is a failed experiment whatever its tok/s.

## Summary

Ten launches, ~5.5 hours, no watchdog event and no host memory incident.
Time to `/health` ranged from 10 min 53 s to 12 min 13 s (`logs/overnight-launches.log`),
so a launch plus its standard sweep costs about 18 minutes.

| | experiment | result |
|---|---|---|
| E0 | baseline: `MAX_NUM_SEQS=8`, V2 runner pinned, K=3 | reference — 44.6 / 68.9 / 107.8 / **151.6** tok/s at S=1/2/4/8 |
| E1 | MTP K=2 | tied with K=3 at every S |
| E2 | MTP K=1 | −8.5% to −14.4% |
| E3 | MTP K=0 (no speculation) | −31.9% to −45.5%; KV pool +227,280 tokens |
| E4 | MTP sparse-index reuse | **blocked** — the flag cannot reach the draft config from the CLI |
| E5 | dynamic K | **skipped** — K=3 wins at every S, nothing to schedule |
| E6 | `flashinfer_b12x` MoE | **failed** — selects, then illegal memory access in `profile_run` |
| E7 | BF16 GDN recurrent state | **KEPT** — +8.5% at 8 streams, needles 15/15 unchanged |
| E8 | prefill chunk 1,024 | **opt-in** — ITL p95 1.67x better (bar was 2x), p99 2.08x |
| E9 | compilation mode 3 | no change (+0.3% / +1.0%, inside noise) |
| — | final launch, E0 + E7 | 48.7 / 74.6 / **113.7** / **162.9** tok/s, 45 min soak clean |

**One change kept: BF16 GDN recurrent state.** It is shipped as a new
`MAMBA_SSM_CACHE_DTYPE` knob in `start.sh` (default empty = the checkpoint's
float32) and enabled in `.env.sample`.

**Two things to know before reading the tables.**

1. **The server left running uses `MAX_NUM_SEQS=8`; `.env.sample` still ships
   4.** Eight streams is where the night's biggest number lives (162.9 tok/s
   against 113.7 at four) and it soaked cleanly, so it stays on this host. It
   is *not* promoted to the shipped default because every sweep tonight was
   short-context, and eight concurrent full-length requests do not fit the KV
   pool (max concurrency at 524,288 tokens is 2.16x). That is a capacity
   question this run did not test. Drop `.env` to `MAX_NUM_SEQS=4` if this
   host's traffic turns long-context.
2. **The 8-stream figure supersedes the CHANGELOG's 114 tok/s**, which was
   measured before `CUDAGRAPH_CAPTURE_SIZES=auto` covered all eight verify
   widths. It is not a regression in the old number; it is a different graph
   configuration.

### Deviations from the plan, and why

- **The 10-minute zero-request window was applied once, before E0**, then
  replaced by a shorter drain check plus the `qwen-code`/`screen`/running-request
  checks before each later relaunch. The rule exists to avoid killing someone
  else's in-flight work; after E0 the only traffic on the port was this run's
  own benchmarks, and ten idle minutes before each of ten relaunches would have
  cost 100 minutes of a 5.5-hour budget. The pre-launch check refused two
  relaunches on this basis (E4 and E8's predecessor) when a sweep had not
  finished draining — it was doing real work, not ceremony.
- **E5 was skipped** on the condition the goal document set for it. See the K
  table.
- **E4's knob was reverted rather than shipped.** See E4.

## Harness verification (live server, before any relaunch)

`bench/sweep.py` was verified against the server that had been up for 10 hours,
at 600 tokens, one repeat:

| S | ms/step | tok/step | aggregate tok/s | anchor (2026-09-05 11:16 sparkDash) | anchor ms/step (CHANGELOG) |
|---|---|---|---|---|---|
| 1 | 63.0 | 3.03 | 47.8 | 46.1 | 63.9 |
| 2 | 77.9 | 2.83 | 71.2 | 70.3 | — |
| 4 | 103.0 | 2.82 | 106.7 | 107.0 | 101.6 |

Within 4% of both anchors at every level, so the wrapper measures what the
README tables were measured with. (An earlier smoke row in
`logs/sweep-smoke.jsonl` reads 33.5 tok/s at S=1; that job ran with
`maxTokens=128`, where the first tokens of the stream dominate. It is not
comparable and is not used below.)

<!-- results appended per experiment as they complete -->

---

## E0 — baseline (`MAX_NUM_SEQS=8`, V2 runner pinned, MTP K=3)

Everything below is measured against this launch, not against the server that
was running when the night started. That server had been up 13 hours under
agent traffic; the relaunch resets the driver's accumulated allocation and the
page cache, and both move decode.

`.env` diff against the shipped profile (`.env.pre-overnight`):

```diff
-MAX_NUM_SEQS=4
+MAX_NUM_SEQS=8
+EXTRA_DOCKER_ARGS="-e VLLM_USE_V2_MODEL_RUNNER=1"
```

Launch: 12 min 12 s to `/health` (00:59:49 → 01:12:01).

Verification (`logs/verify-E0.txt`):

```
gpu_worker.py:500  Using V2 Model Runner
(no "Overriding cudagraph_mode" line anywhere in the log)
'cudagraph_capture_sizes': [4, 8, 12, 16, 20, 24, 28, 32]
Graph capturing finished in 5 secs, took 0.35 GiB     (passes of 8 and of 2 graphs)
mtp.py:111   MTP draft vocab: 65536 of 248320 tokens (26.4%); draft lm_head
             1.18 -> 0.31 GiB per draft step, 2.61 GiB saved per step at MTP 3
gpu_worker.py:693      Available KV cache memory: 17.3 GiB
kv_cache_utils.py:2258 GPU KV cache size: 1,161,935 tokens,
                       Maximum concurrency for 524,288 tokens per request: 2.22x
nvfp4.py:291 Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
interface.py:915 Setting attention block size to 3200 tokens
```

12 `NV_ERR_NO_MEMORY` lines during startup, none after `/health` and none
during any sweep level. The watchdog logged each burst with the memory it
happened at — 1 line at `MemAvailable` 35.8 GiB, 10 at 15.8 GiB, 1 at
13.7 GiB — which is the benign startup pattern the README describes (the
driver bouncing off free pages while the checkpoint read fills the page cache),
not the fatal one, which is the same line under ~10 GiB. Every launch tonight
looked like this: the per-launch counts were 12, 5, 10, 0, 2, 1, 12, 2 and 4,
all during startup, none after `/health`, and none during any benchmark
level in any configuration.

### Standard sweep

| S | ms/step | tok/step | accept p1/p2/p3 | aggregate tok/s | repeats | MemAvail min | MemFree min | NVRM |
|---|---|---|---|---|---|---|---|---|
| 1 | 62.9 | 2.83 | 0.79/0.62/0.42 | **44.6** | 45.4/44.8/43.6 | 13.0 | 10.8 | 0 |
| 2 | 77.3 | 2.72 | 0.78/0.56/0.38 | **68.9** | 68.1/70.7/68.0 | 13.0 | 10.7 | 0 |
| 4 | 101.1 | 2.80 | 0.80/0.59/0.41 | **107.8** | 109.0/107.3/107.2 | 13.0 | 10.6 | 0 |
| 8 | 141.4 | 2.80 | 0.80/0.59/0.41 | **151.6** | 147.2/154.1/153.5 | 13.0 | 10.6 | 0 |

Two things in this table are worth carrying forward:

- **151.6 tok/s at 8 streams**, against the 114 tok/s the CHANGELOG records
  for `MAX_NUM_SEQS` 5 → 8 on 2026-09-05. That earlier figure was measured
  before `CUDAGRAPH_CAPTURE_SIZES=auto` covered all eight verify widths; here
  every width from 4 to 32 has a FULL decode graph. The 8-stream step is
  141.4 ms against the CHANGELOG's 146.2 ms.
- **Acceptance on sparkDash prose is high** — 0.80/0.59/0.41 per position,
  2.80 tokens per step at K=3. That is much closer to the agent/code profile
  in `bench/kmodel.py` (0.836/0.689/0.560) than to its "prose" profile
  (0.65/0.40/0.25). High acceptance is what makes a deep draft pay, so the K
  sweep below is being run on a workload that favours large K.

### Mixed traffic (2 decoders + one 64k prompt injected at 5 s)

| | quiet (before injection) | during the prefill | after |
|---|---|---|---|
| ITL mean | 78.2 / 79.8 ms | **934.0 / 934.0 ms** | 77.6 / 78.4 ms |
| ITL p50 | 76.0 / 76.4 ms | 1,057.3 / 1,057.3 ms | 77.4 / 77.5 ms |
| ITL p95 | 94.1 / 98.5 ms | **1,111.1 / 1,111.3 ms** | 85.0 / 85.0 ms |
| ITL p99 | 98.5 / 120.4 ms | **1,399.8 / 1,399.7 ms** | 95.0 / 95.2 ms |

64,026-token prompt, TTFT **34.54 s**. Aggregate decode across both streams
during that window: **2.14 tok/s** (74 tokens in 34.54 s).

The p50 of 1,057 ms is the mechanism in one number: a 2,048-token chunk at the
~2,100 tok/s this host prefills at is 0.97 s of GPU time, and chunked prefill
puts one chunk in the same engine step as every co-scheduled decode. The
decoders do not get a slow step, they get *one step per chunk*. This is the
baseline E8 has to beat.

### Prefill (sparkDash) and long-context quality

| context | prefill | TTFT |
|---|---|---|
| 32,768 | 2,128 tok/s | 15.42 s |
| 65,536 | 2,195 tok/s | 29.88 s |

`bench/longctx.py` at 32k, five runs: **15/15 needles found**, 5/5 PASS
(alpha at 5%, bravo at 50%, charlie at 95% depth), prefill 1,979–1,998 tok/s.
This is the pass count E7 has to match.

Three sanity prompts and a 4-turn continuation were captured for comparison
against the kernel and dtype experiments (reasoning, code, prose; the
reasoning answer is correct at 22 sheep).

---

## E1–E3 — static K sweep with FULL graphs

One launch per depth, everything else held at E0. `CUDAGRAPH_CAPTURE_SIZES=auto`
recomputes the widths from K, so each launch captures a FULL decode graph for
every verify batch its scheduler can build. `.env` diff per launch is one line:
`MTP_NUM_SPECULATIVE_TOKENS=3` → `2` → `1` → `0`.

### E1 — K=2

Verified: `Using V2 Model Runner`, no `Overriding cudagraph_mode`,
`'cudagraph_capture_sizes': [3, 6, 9, 12, 15, 18, 21, 24]` — (1+2)×S for
S=1..8, as expected. KV 16.32 GiB = 1,120,593 tokens. 5 `NV_ERR_NO_MEMORY`
lines during startup, none afterwards.

| S | ms/step | tok/step | accept p1/p2 | aggregate tok/s | repeats | MemAvail min | MemFree min | NVRM |
|---|---|---|---|---|---|---|---|---|
| 1 | 55.9 | 2.46 | 0.82/0.64 | **44.3** | 44.5/43.3/45.0 | 14.8 | 4.2 | 0 |
| 2 | 68.5 | 2.40 | 0.80/0.60 | **69.4** | 68.3/70.4/69.4 | 14.8 | 4.1 | 0 |
| 4 | 90.5 | 2.40 | 0.80/0.60 | **103.9** | 103.7/107.0/101.1 | 14.8 | 4.1 | 0 |
| 8 | 124.7 | 2.42 | 0.81/0.61 | **150.1** | 154.1/149.5/146.8 | 14.8 | 4.0 | 0 |

Against E0 (K=3), by the ±5%-in-all-three-repeats rule:

| S | E0 mean | E1 mean | mean Δ | per-repeat Δ | verdict |
|---|---|---|---|---|---|
| 1 | 44.6 | 44.3 | −0.8% | −2.1%, −3.3%, +3.1% | no change |
| 2 | 68.9 | 69.4 | +0.7% | +0.4%, −0.4%, +2.0% | no change |
| 4 | 107.8 | 103.9 | −3.6% | −4.8%, −0.3%, −5.6% | no change |
| 8 | 151.6 | 150.1 | −1.0% | +4.7%, −3.0%, −4.4% | no change |

**K=2 and K=3 are throughput-equivalent here, and the two columns that move
show why.** Dropping the third draft step takes 11% off the engine step at
every concurrency (62.9 → 55.9 ms at S=1, 141.4 → 124.7 at S=8) and takes
almost exactly the same fraction off the tokens the step yields (2.83 → 2.46,
2.80 → 2.42). The third draft position was being accepted 0.41–0.42 of the
time, and at that acceptance its cost and its return cancel. Note that the
first two positions do not get *worse* without a third to follow them
(0.79/0.62 → 0.82/0.64 at S=1), so nothing is lost by shortening the draft
except the tokens the third position would have contributed.

### E2 — K=1

Verified: `'cudagraph_capture_sizes': [2, 4, 6, 8, 10, 12, 14, 16]`, V2 runner,
no override. KV 16.02 GiB = 1,121,393 tokens. 10 startup `NV_ERR_NO_MEMORY`
lines, none afterwards.

| S | ms/step | tok/step | accept p1 | aggregate tok/s | repeats | MemAvail min | MemFree min | NVRM |
|---|---|---|---|---|---|---|---|---|
| 1 | 48.7 | 1.86 | 0.86 | **38.2** | 38.0/38.1/38.5 | 15.3 | 6.2 | 0 |
| 2 | 58.1 | 1.82 | 0.82 | **61.8** | 62.0/62.0/61.3 | 15.4 | 6.2 | 0 |
| 4 | 75.1 | 1.82 | 0.82 | **95.0** | 95.2/95.4/94.5 | 15.4 | 6.1 | 0 |
| 8 | 102.2 | 1.84 | 0.84 | **138.7** | 137.2/139.4/139.6 | 15.3 | 6.0 | 0 |

| S | E0 mean | E2 mean | mean Δ | per-repeat Δ | verdict |
|---|---|---|---|---|---|
| 1 | 44.6 | 38.2 | −14.4% | −16.3%, −15.0%, −11.9% | **worse** |
| 2 | 68.9 | 61.8 | −10.4% | −8.9%, −12.3%, −9.9% | **worse** |
| 4 | 107.8 | 95.0 | −11.9% | −12.7%, −11.1%, −11.8% | **worse** |
| 8 | 151.6 | 138.7 | −8.5% | −6.8%, −9.6%, −9.1% | **worse** |

K=1 loses everywhere, and by more than the noise floor in all three repeats at
every S. The second draft position was worth 0.60–0.64 acceptance; dropping it
takes 13% off the step and 24% off the tokens the step yields, which is not a
trade that closes. Note that the *first* position's acceptance rises as the
draft shortens (0.79 → 0.82 → 0.86 at S=1 for K=3/2/1) — a shorter draft is
proposed from a less speculative state — but not nearly enough to compensate.

### E3 — K=0 (MTP disabled)

Verified: `speculative_config=None`, `'cudagraph_capture_sizes': [1, 2, 3, 4,
5, 6, 7, 8]`, and — as required — **no speculator lines of any kind**: zero
matches for `speculator`, `MTP draft vocab` or `Qwen3_8FlashNextMTP` in the
container log, and zero `vllm:spec_decode_*` series in `/metrics`.
0 `NV_ERR_NO_MEMORY` lines for the whole launch.

The KV pool grows, as expected, when the 1.49 GiB draft model is not built:

| | E0 (K=3) | E3 (K=0) | Δ |
|---|---|---|---|
| Available KV cache memory | 17.3 GiB | **17.9 GiB** | +0.6 GiB |
| GPU KV cache size | 1,161,935 tok | **1,389,215 tok** | +227,280 tok (+19.6%) |
| Max concurrency @ 524,288 | 2.22x | 2.65x | +0.43x |

The token count rises much more than the byte count because the draft model's
own KV is gone from the pool as well as its weights.

| S | ms/step | tok/step | aggregate tok/s | repeats | MemAvail min | MemFree min | NVRM |
|---|---|---|---|---|---|---|---|
| 1 | 41.1 | 1.00 | **24.3** | 24.3/24.3/24.3 | 16.2 | 7.9 | 0 |
| 2 | 47.2 | 1.00 | **41.9** | 41.1/42.4/42.1 | 16.2 | 7.9 | 0 |
| 4 | 57.8 | 1.00 | **68.5** | 68.3/68.8/68.6 | 16.2 | 7.8 | 0 |
| 8 | 76.4 | 1.00 | **103.3** | 102.7/103.4/103.7 | 16.1 | 7.7 | 0 |

Worse everywhere by 32–46%. Speculative decoding is worth 84% at one stream
and 47% at eight on this host and this workload.

### The K table

Aggregate decode tok/s, prose, 600 tokens, three repeats, mean:

| S | K=0 | K=1 | K=2 | K=3 | best |
|---|---|---|---|---|---|
| 1 | 24.3 (−45.5%) | 38.2 (−14.4%) | 44.3 (−0.8%) | **44.6** | K=3 ≡ K=2 |
| 2 | 41.9 (−39.3%) | 61.8 (−10.4%) | **69.4** (+0.7%) | 68.9 | K=2 ≡ K=3 |
| 4 | 68.5 (−36.4%) | 95.0 (−11.9%) | 103.9 (−3.6%) | **107.8** | K=3 ≡ K=2 |
| 8 | 103.3 (−31.9%) | 138.7 (−8.5%) | 150.1 (−1.0%) | **151.6** | K=3 ≡ K=2 |

Engine step (ms) and tokens per step behind those numbers:

| S | K=0 | K=1 | K=2 | K=3 |
|---|---|---|---|---|
| 1 | 41.1 / 1.00 | 48.7 / 1.86 | 55.9 / 2.46 | 62.9 / 2.83 |
| 2 | 47.2 / 1.00 | 58.1 / 1.82 | 68.5 / 2.40 | 77.3 / 2.72 |
| 4 | 57.8 / 1.00 | 75.1 / 1.82 | 90.5 / 2.40 | 101.1 / 2.80 |
| 8 | 76.4 / 1.00 | 102.2 / 1.84 | 124.7 / 2.42 | 141.4 / 2.80 |

**Result: K=3 is optimal at every concurrency, with K=2 statistically tied to
it, and there is no crossover.** The open question from the synthesis document
was whether K=1 wins at S=8 with FULL graphs; it does not — it loses 8.5%
there, and the single historical data point that suggested otherwise
(step 160.8 → 127.4 ms at S=8) was taken under PIECEWISE graphs, where the
K=3 side was paying a graph penalty it does not pay here. With every verify
width captured as a FULL graph, the deeper draft keeps its advantage all the
way to 8 streams.

Two mechanisms are visible in the step/token table. Adding a draft position
costs a near-constant slice of step time — roughly 7 ms at S=1 and 22 ms at
S=8 per position, flat across positions — while each position returns its own
acceptance probability in tokens. Positions 1 and 2 return 0.80 and 0.60, well
above break-even; position 3 returns 0.41, which is almost exactly break-even
and is why K=2 and K=3 tie. And acceptance on the earlier positions *improves*
slightly as K shrinks (p1 = 0.79 / 0.82 / 0.86 at K=3/2/1, S=1), so the
positions are not independent — but not by enough to change the ranking.

**Consequence for E5.** The goal document runs dynamic K only "if E1–E3 found
a per-S optimum that is not K=3 everywhere". It is K=3 everywhere, tied with
K=2 and never beaten, so a batch-size schedule has nothing to schedule: it
would spend a launch to reproduce a flat line. **E5 is skipped**, and the
launch it would have used goes to E9. The `MTP_K_SCHEDULE` knob and the
`VLLM_USE_V2_MODEL_RUNNER=1` pin that makes it safe are both still in place and
still correct; what this measurement removes is the reason to use them.

---

## E4 — MTP sparse-index reuse: blocked, not measured

**The synthesis document's premise is wrong for this vLLM build.** §4 says
`index_share_for_mtp_iteration` "can be turned on with an `hf-overrides` entry;
no code change". It cannot. Dict `--hf-overrides` are deliberately not
propagated to the draft model's config, and that flag is read only from the
draft config.

What was built and what it showed. `start.sh` gained an `MTP_INDEX_SHARE=1`
knob that merges `"index_share_for_mtp_iteration":true` into the single
`--hf-overrides` `text_config` object, and `files/patch_mtp_draft_vocab.py`
gained a one-line INFO log the first time the speculator reaches the draft
indexer — the runtime assertion the goal document asked for, since the flag has
no log of its own. The launch was clean and the flag reached the command line:

```
--hf-overrides '{"text_config":{"rope_parameters":{...},"index_share_for_mtp_iteration":true}}'
```

but the assertion never fired, before or after a served request:

```
$ docker logs vllm-fn-tp1 | grep -i "MTP index share"
(nothing)
```

The cause is `SpeculativeConfig.compose_draft_hf_overrides`
(`vllm/config/speculative.py:734`), whose own docstring states the rule:

> Callable overrides on the target are config-to-config transforms […] and
> must also reach the draft config […]. **Dict overrides are target-specific
> key patches and are not applied to the draft.**

Confirmed directly in the container:

```
$ docker exec vllm-fn-tp1 python3 -c "...compose_draft_hf_overrides({'text_config': {...}})..."
dict override composes to: hf_config_override
is it the plain override (i.e. the dict is dropped)? True
```

So `config/speculative.py:538` reads `index_share_for_mtp_iteration` off a
draft `text_config` that never received it, `share_mtp_topk_indices` stays
False (`v1/worker/gpu/spec_decode/mtp/speculator.py:30-33`), and
`set_skip_topk` is never called. The model-side support the synthesis verified
is real — `files/mtp_patched.py:305,311` implement both methods — but nothing
reaches it from the CLI.

**Verdict: no measurement, launch does not count** (the gate refused to
benchmark, so no numbers were taken from it). **Not pursued further tonight**,
and the `MTP_INDEX_SHARE` knob was reverted rather than shipped, because a
switch that silently does nothing is worse than no switch. The only routes left
are a checkpoint `config.json` edit (out of scope — the constraints forbid
touching the checkpoint) or patching `speculative.py`. The obvious one-line
patch — propagate dict overrides to the draft too — is **not** safe here: this
server's other dict override is the 512k YaRN rope config, and pushing that
onto the draft silently changes the drafter's positional encoding. A correct
patch has to pass this one key and nothing else, and it needs its own launch
and an acceptance check. That is a daytime change, not an overnight one.

The diagnostic INFO line is kept. It costs nothing while the flag is off and it
is what will prove the flag arrived if anyone does the propagation work.

**E4's launch is also the one datapoint on restart-to-restart KV variation**:
same `.env` as E0 in every respect that touches memory, but
`Available KV cache memory: 15.59 GiB / 1,045,742 tokens` against E0's
`17.3 GiB / 1,161,935`. The README already notes the pool varies between
restarts; 10% is the size of that variation on this host, which is worth
knowing before reading a small KV difference as a result.

---

## E6 — `flashinfer_b12x` MoE: loads, then kills the engine

`.env`: E0 plus
`EXTRA_VLLM_ARGS="--kernel-config '{\"moe_backend\":\"flashinfer_b12x\"}'"`.
Rendered command (`.last_launch.sh`): `--kernel-config
'{"moe_backend":"flashinfer_b12x"}'`.

**It selects, for both processes.** The synthesis was right that the opt-in
works and that the backend accepts this checkpoint:

```
(Worker pid=228)           INFO nvfp4.py:244 Using 'FLASHINFER_B12X' NvFp4 MoE backend
(PleOffloadWorker pid=406) INFO nvfp4.py:244 Using 'FLASHINFER_B12X' NvFp4 MoE backend
```

**Then the engine dies during memory profiling, before the KV pool is sized:**

```
gpu_worker.py:634  determine_available_memory
  model_runner.py:854   profile_run
    model_runner.py:736   _dummy_run -> execute_model
      qwen3_8_flash_next/nvidia/model.py:494  layer(
        model.py:310                          attn_out = self.linear_attn(...)
          mamba/gdn/qwen_gdn_linear_attn.py:818  return self._forward_method(hidden_states)
          mamba/gdn/qwen_gdn_linear_attn.py     self._warmup_prefill_kernels(mixed_qkvz[:, :qkv_size], 0)
RuntimeError: Worker failed with error 'CUDA error: an illegal memory access was encountered'
```

Container `Exited (1)`; `start.sh` never reached `/health`, so nothing was
benchmarked. The traceback surfaces inside the GDN linear-attention warm-up
rather than inside the MoE, which is the ordinary shape of an asynchronous CUDA
fault: the illegal access is raised at the next synchronising kernel, and the
only thing changed in this launch is the MoE backend that ran just before it.

**Verdict: failed, skipped, not patched around.** The synthesis called the
`nvfp4.py:177-179` comment excluding this backend from `auto` "stale-looking".
On this GPU it is not stale — the exclusion is load-bearing, and the guard is
the only thing that was keeping `auto` alive. Anyone revisiting this needs a
standalone kernel reproducer at the shapes `profile_run` uses, not a serving
launch.

**No host risk in the failure**: 0 `NV_ERR_NO_MEMORY` lines for the whole
launch, the watchdog logged no floor breach and exited normally when the
container went (`container gone; watchdog exit (NV_ERR_NO_MEMORY seen since
watchdog start: 0)`), and the driver figure went straight back from 79.8 GiB
to 2.1 GiB. A crash inside the container is the safe failure mode here; it is
the ones that keep running that cost a host.

Full log kept at `logs/archive/E6-flashinfer_b12x-failed-launch.log`.

---

## E7 — BF16 GDN recurrent state: **kept**, +8.5% at 8 streams

`.env`: E0 plus `EXTRA_VLLM_ARGS="--mamba-ssm-cache-dtype bfloat16"`.

Verification. The checkpoint asks for float32 and the CLI overrides it, which
vLLM says out loud:

```
config.py:799 WARNING  Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config,
              but --mamba-ssm-cache-dtype='bfloat16' was passed. Using the user-specified value.
              'mamba_ssm_cache_dtype': 'bfloat16'
interface.py:915  Setting attention block size to 1664 tokens   (E0: 3200)
interface.py:939  Padding mamba page size by 0.48%              (E0: 0.25%)
qwen_gdn_linear_attn.py:158  Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)
qwen_gdn_linear_attn.py:505  GDN decode kernel: cuda
```

No dtype fallback warning from `qwen_gdn_linear_attn` — the fused kernel takes
bfloat16 directly, as `FUSED_GDN_STATE_DTYPES` promised. The attention block
size halves as predicted; the exact figure is **1,664 rather than 1,600**,
because the block has to be padded until the attention page size matches the
mamba page size exactly. KV pool 16.64 GiB = 1,180,814 tokens against E0's
17.3 GiB = 1,161,935 — slightly *more* tokens out of slightly less memory, which
is the halved mamba page showing up. 1 `NV_ERR_NO_MEMORY` line at startup, none
after.

| S | ms/step | tok/step | accept p1/p2/p3 | aggregate tok/s | repeats | MemAvail min | MemFree min | NVRM |
|---|---|---|---|---|---|---|---|---|
| 1 | 61.3 | 2.92 | 0.82/0.63/0.46 | **47.6** | 46.1/46.4/50.4 | 14.4 | 6.7 | 0 |
| 2 | 74.1 | 2.79 | 0.79/0.59/0.40 | **73.3** | 71.4/74.4/74.2 | 14.3 | 6.6 | 0 |
| 4 | 95.5 | 2.77 | 0.79/0.58/0.40 | **111.1** | 111.9/112.5/109.1 | 14.3 | 6.6 | 0 |
| 8 | 130.4 | 2.80 | 0.79/0.58/0.42 | **164.5** | 162.7/165.5/165.2 | 14.3 | 6.0 | 0 |

| S | E0 mean | E7 mean | mean Δ | per-repeat Δ | verdict |
|---|---|---|---|---|---|
| 1 | 44.6 | 47.6 | +6.8% | +1.4%, +3.6%, +15.5% | no change (one repeat carries it) |
| 2 | 68.9 | 73.3 | +6.4% | +4.9%, +5.3%, +9.1% | no change (one repeat at +4.9%) |
| 4 | 107.8 | 111.1 | +3.1% | +2.7%, +4.8%, +1.8% | no change |
| 8 | 151.6 | **164.5** | **+8.5%** | +10.6%, +7.4%, +7.6% | **better** |

The gain is in the step, not in acceptance: 141.4 → 130.4 ms at S=8 with
tokens per step unchanged at 2.80. That is the mechanism the synthesis
predicted — 0.23 GB × S of recurrent state read and written per step, halved —
and the size is right: ~2% predicted at 8 streams against 8.5% measured, so the
prediction was low, most likely because the halved mamba page also buys a
smaller attention block and better prefix-cache granularity.

**Quality: unchanged.** `bench/longctx.py` at 32k, five runs, **15/15 needles
found, 5/5 PASS** — exactly E0's pass count, at all three depths including the
95% one. The 4-turn continuation, which is what would expose a recurrence
degrading across turns, ends on a summary that recalls every element of the
conversation (hash map, write-ahead log, lock, replay-on-restart, torn-write
handling) in both configurations; the two are indistinguishable in substance.
The three fixed sanity prompts agree with E0, including the correct 22 on the
"all but 9" question.

**Verdict: keep.** It passes both halves of the rule the goal document set —
needles match E0's pass count, and the S=8 sweep gains more than 5% in all
three repeats — with no loss anywhere and no memory-rule violation.

---

## E8 — prefill chunk 1024 under mixed traffic: **opt-in, not the default**

`.env`: E7 (bf16 SSM kept) plus `MAX_NUM_BATCHED_TOKENS=1024`. Verified
`scheduler.py:246 Chunked prefill is enabled with max_num_batched_tokens=1024`
— the width is honoured, not raised back for the draft token slots. KV
16.9 GiB = 1,197,928 tokens.

**Decode-only is unchanged**, which is the precondition for reading the rest:

| S | E7 (2048) | E8 (1024) | Δ |
|---|---|---|---|
| 1 | 47.6 | 48.0 (48.9/46.5/48.5) | +0.8% |
| 8 | 164.5 | 162.3 (166.9/155.9/164.1) | −1.3% |

**Mixed traffic**, the test this experiment exists for — two streams decoding,
one 64k prompt injected at 5 s:

| | E0 (2048 chunk) | E8 (1024 chunk) | change |
|---|---|---|---|
| decoder ITL p50 in window | 1,057.3 ms | **628.6 / 629.0 ms** | **1.68x better** |
| decoder ITL **p95** in window | 1,111.1 / 1,111.3 ms | **665.6 / 664.9 ms** | **1.67x better** |
| decoder ITL **p99** in window | 1,399.8 / 1,399.7 ms | **674.1 / 674.1 ms** | **2.08x better** |
| decoder ITL max in window | 1,399.8 ms | 677.8 ms | 2.06x better |
| aggregate decode in window | 2.14 tok/s | **3.45 tok/s** | +61% |
| 64k prompt TTFT | 34.54 s | **40.34 s** | **+16.8% worse** |
| ITL quiet (control) | 94.1 / 98.5 ms p95 | 81.5 / 84.4 ms p95 | (E7's decode gain) |

Prefill alone (sparkDash, no co-scheduled decode):

| context | E0 (2048) | E8 (1024) | Δ |
|---|---|---|---|
| 32,768 | 2,127.7 tok/s | 2,118.7 tok/s | −0.4% |
| 65,536 | 2,194.7 tok/s | 2,074.1 tok/s | −5.5% |

**Verdict: documented opt-in, shipped default unchanged.** The goal document's
bar is "p95 ITL improves by 2x or more with TTFT within 25%". TTFT passes
(+16.8%, inside 25%) but **p95 improves 1.67x, not 2x**, so the bar is not met
and `MAX_NUM_BATCHED_TOKENS` stays at 2,048. It is a near miss and the shape of
the win is exactly as predicted: halving the chunk halves the engine step a
co-scheduled decode has to wait through, and the distribution tightens sharply
— the p99 and the max both improve by more than 2x, because the long tail at
2,048 was one chunk landing badly, and there is no such tail at 1,024.

Worth recording for whoever tunes this next: **p99 does clear 2x**, and the
cost is only 5.5% of prefill at 64k. For the agent harness described in the
synthesis — 72k-token prompts arriving while earlier turns still stream — 1,024
is very likely the better setting, and the case for it should be made on p99
and on aggregate decode during prefill (+61%), not on p95. Anyone making that
change should re-measure with their own traffic rather than promote this row.

---

## E9 — compilation mode 3: no change, not kept

`.env`: E7 (bf16 SSM kept, chunk back to 2,048) plus `COMPILATION_MODE=3` and
`CUDAGRAPH_MODE=FULL_AND_PIECEWISE`. `CUDAGRAPH_MODE` is already an
environment-readable knob in `start.sh`, so no new knob was needed.

Verified it actually engaged:

```
'mode': <CompilationMode.VLLM_COMPILE: 3>
'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>
backends.py:393  Compiling a graph for compile range (1, 2048) takes 9.90 s
backends.py:393  Compiling a graph for compile range (1, 2048) takes 3.19 s
(no "Overriding cudagraph_mode" line)
```

The PLE custom op survives Inductor: the server answered every benchmark
request, with no first-request hang against the
`VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300` that `start.sh` sets. Startup cost was
smaller than expected — 12 min 13 s to `/health` against E0's 12 min 12 s,
because the two Inductor compilations total 13 s against an 11-minute
checkpoint read.

| S | E7 mean | E9 mean | mean Δ | per-repeat Δ | verdict |
|---|---|---|---|---|---|
| 1 | 47.6 | 47.8 | +0.3% | +5.4%, +2.1%, −6.1% | no change |
| 4 | 111.1 | 112.3 | +1.0% | +0.5%, +0.4%, +2.1% | no change |

**Verdict: not kept.** This is the result the byte model predicted for the
right reason: fusion only helps the part of the step that is not
memory-bandwidth-bound, and after the reduced draft head this server has very
little of that left even at one stream. Worth knowing that it is *safe* —
mode 3 loads, compiles quickly, keeps FULL decode graphs and does not disturb
the PLE handshake — so it stays available for a future configuration that is
not bandwidth-bound. It buys nothing on this one.

---

## Final configuration

E0 plus the one kept change. Re-verified in its own launch rather than reusing
E7's, because E7's container also carried an `EXTRA_VLLM_ARGS` spelling of the
flag that has since been replaced by a first-class knob.

`.env` diff against `.env.pre-overnight` (the profile this run started from):

```diff
-MAX_NUM_SEQS=4
+MAX_NUM_SEQS=8
+MAMBA_SSM_CACHE_DTYPE=bfloat16
+EXTRA_DOCKER_ARGS="-e VLLM_USE_V2_MODEL_RUNNER=1"
```

`start.sh` gained `MAMBA_SSM_CACHE_DTYPE` (default empty = the checkpoint's
float32, i.e. previous behaviour), so the change exists in the repo and not
only in the ignored `.env`. The knob was verified to render **byte-identically**
to the `EXTRA_VLLM_ARGS` spelling the soaked container was launched with, so
the running server matches what `.env` now produces.

Verification of the final launch:

```
gpu_worker.py:500  Using V2 Model Runner
(no "Overriding cudagraph_mode")
'cudagraph_capture_sizes': [4, 8, 12, 16, 20, 24, 28, 32]
interface.py:915  Setting attention block size to 1664 tokens
gpu_worker.py:693      Available KV cache memory: 15.98 GiB
kv_cache_utils.py:2258 GPU KV cache size: 1,132,586 tokens,
                       Maximum concurrency for 524,288 tokens per request: 2.16x
```

4 `NV_ERR_NO_MEMORY` lines at startup, none after `/health`.

### Standard sweep on the final launch

| S | ms/step | tok/step | accept p1/p2/p3 | aggregate tok/s | repeats | MemAvail min | MemFree min | NVRM |
|---|---|---|---|---|---|---|---|---|
| 1 | 61.5 | 3.00 | 0.86/0.68/0.46 | **48.7** | 46.9/50.1/49.1 | 16.1 | 1.8 | 0 |
| 2 | 74.3 | 2.83 | 0.82/0.60/0.42 | **74.6** | 75.4/72.0/76.3 | 16.1 | 1.8 | 0 |
| 4 | 96.2 | 2.84 | 0.81/0.61/0.43 | **113.7** | 116.0/112.1/113.0 | 16.1 | 1.8 | 0 |
| 8 | 131.0 | 2.81 | 0.80/0.60/0.41 | **162.9** | 158.9/166.1/163.7 | 16.1 | 1.7 | 0 |

Against E0, this reproduces the E7 result on an independent launch:

| S | E0 | final | mean Δ | per-repeat Δ | verdict |
|---|---|---|---|---|---|
| 1 | 44.6 | 48.7 | +9.2% | +3.3%, +11.8%, +12.6% | no change (one repeat under 5%) |
| 2 | 68.9 | 74.6 | +8.2% | +10.8%, +1.9%, +12.2% | no change (one repeat under 5%) |
| 4 | 107.8 | 113.7 | +5.5% | +6.5%, +4.5%, +5.4% | no change (one repeat under 5%) |
| 8 | 151.6 | **162.9** | **+7.4%** | +8.0%, +7.8%, +6.6% | **better** |

The `MemFree` minimum of 1.7 GiB looks alarming next to the watchdog's 2 GiB
floor and is not: that floor only counts while `MemAvailable` is under 10 GiB,
and `MemAvailable` never went below 16.1. The watchdog timeline at the time
reads `free=1611MiB cached=15682MiB avail=16641MiB` — the page cache is full of
the PLE table, which is exactly what it is supposed to be full of. This is the
condition the gate was added for, after an ungated version killed a healthy
launch on 2026-09-05.

---

## What to do next

Only things tonight's measurements actually point at.

1. **Decide the chunk width on the harness's own traffic.** E8 is the largest
   unclaimed win in this run and it was rejected on a single percentile. p99
   inter-token latency during a 64k prefill improves 2.08x and aggregate decode
   in the window improves 61%, for 5.5% of prefill at 64k. The synthesis
   document says this box exists for an agent harness sending 72k-token prompts
   at up to 3 concurrent — that is precisely the traffic `bench/mixed.py`
   simulates. Run the harness for an hour at 2,048 and an hour at 1,024 and
   compare turn latency, then set the default from that rather than from a
   synthetic injection.

2. **Test `MAX_NUM_SEQS=8` at long context before shipping it.** It is running
   here and it soaked, but on short-context sweeps only. The KV pool holds
   2.16x a 524k request, so eight concurrent long requests will contend and
   preempt. What is missing is one run with 8 concurrent 100k+ prompts,
   watching `vllm:num_requests_waiting_by_reason{reason="capacity"}` and
   `MemAvailable`. Until then `.env.sample` stays at 4.

3. **`index_share_for_mtp_iteration` needs a targeted patch, and is probably
   worth more here than the synthesis estimated.** The saving grows with
   context (the draft's compressed-key scan is per token per unit of context)
   and this host's real traffic is 72k-token prompts, so the "small saving"
   estimate was made for the wrong workload. The patch has to pass exactly that
   one key onto the draft config and nothing else — propagating dict overrides
   wholesale would push the 512k YaRN rope config onto the drafter. Budget a
   launch plus an acceptance-rate comparison, since it changes what the drafter
   proposes on steps 2 and 3.

4. **`flashinfer_b12x` needs a kernel reproducer, not another launch.** It
   selects and then faults inside `profile_run`. Reproduce at the shapes that
   run uses, outside the server; a serving launch costs 12 minutes to learn
   nothing new.

5. **Refit or retire `bench/kmodel.py`.** It is deliberately not committed with
   the rest of the harness. Given the right acceptance numbers it called the
   K=2/K=3 tie correctly at both ends of the sweep, but its built-in "prose"
   acceptance profile (0.65/0.40/0.25) is wrong for this host's actual prose
   workload, which measures 0.80/0.59/0.41 — with that profile it predicts K=1
   wins at 8 streams, which is the opposite of the measurement. It also
   over-predicts K=0 badly (148.8 tok/s at 8 streams against 103.3 measured),
   because its fixed per-draft-step cost is too small. All three inputs are now
   measured, so refitting it is cheap; shipping it unrefitted would ship a
   predictor this run partly falsified.

6. **The byte model under-predicted BF16 state by 4x** — ~2% expected at 8
   streams, 8.5% measured. The gap is most likely everything the model does not
   count: the halved mamba page also halves the attention block (3,200 → 1,664)
   and doubles prefix-cache granularity. Worth a note before the model is used
   to rank the next batch of items, since it was accurate on the reduced-head
   change and is being trusted on that record.

7. **Prefill is still where this machine spends its day and it barely moved
   tonight.** Every experiment here was a decode experiment; the only prefill
   numbers taken were controls. The synthesis's §7 point stands untouched: a
   prefill profile of one 64k prompt would rank that side the way the byte
   model ranked decode.

### 45-minute idle + light-load soak

Sampled every 30 s from 04:29:50 to 05:14:54, with one short request injected
every 2 minutes (89 samples, `logs/soak-final.tsv`):

| | min | max | first → last |
|---|---|---|---|
| host `MemAvailable` | **15.52 GiB** | 16.42 GiB | 16.30 → 16.42 |
| host `MemFree` | 1.25 GiB | 2.06 GiB | 1.62 → 2.06 |
| watchdog `driver` figure | 93.61 GiB | 93.83 GiB | 93.73 → 93.71 (**drift −0.02 GiB**) |
| `NV_ERR_NO_MEMORY` | — | — | **0 over the whole soak** |

No watchdog event; `logs/memwatch-vllm-fn-tp1.log` contains no floor-breach
line of either kind for the entire launch. **Pass.**

Two things worth reading off this table:

- **The driver figure does not grow.** It ends 0.02 GiB below where it started,
  against a 95.60 GiB budget. The per-request growth the 2026-09-05 entry
  documents (96.4 → 97.5 GiB over hours of harness traffic) is a
  long-prompt phenomenon; 45 minutes of short requests does not move it. The
  server this run replaced had reached 95.4 GiB after 10 hours of agent
  traffic, which is why the E0 relaunch — not that server — is the baseline
  everything here is measured against.
- **`MemFree` sits at 1.3–2.1 GiB and that is correct.** `Cached` is ~15.7 GiB
  of PLE table, which is what the reserve is sized to hold. The watchdog's
  2 GiB `MemFree` floor only counts while `MemAvailable` is under 10 GiB, and
  `MemAvailable` never went below 15.52. This is precisely the case the gate
  was added for on 2026-09-05, after an ungated version killed a healthy
  launch during weight loading.

**The server is left running on this configuration.**
