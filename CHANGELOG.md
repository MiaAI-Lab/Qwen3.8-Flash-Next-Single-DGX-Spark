# Changelog

Notable changes to this deployment kit. The repository is not versioned; entries
are grouped by date, newest first. Every measurement named here was taken on the
one DGX Spark this repo is written for — treat them as that host's numbers, not
as promises.

## 2026-09-05

### Fixed

- **The GPU budget had no term for the host, and three servers died of it**
  (2026-09-04, `KV_TARGET_GIB=22`, under a five-agent qwen-code harness sending
  370 requests averaging 72k input tokens). `start.sh` sized the GPU budget as
  `weights + overhead + MTP + KV_TARGET_GIB` and vLLM, which detects this GPU
  as integrated and treats host `MemAvailable` as free GPU memory, filled the
  GPU side to exactly that number. That left 20.7 GiB of the 121.6 GiB pool for
  a host that needs at least 22: other containers and sessions ~7 GiB, vLLM's
  host-side processes ~6, PLE page cache ≥6, free pages the NVIDIA driver needs
  ≥3 — before 2–3 GiB of per-request growth that the CUDA caching allocator
  never returns while serving (this build's UMA release valve is only called
  from the model loader). The servers idled 1–3 GiB above the 6 GiB watchdog
  floor, the driver logged `NV_ERR_NO_MEMORY` at `MemFree` ~3 GiB, and the
  watchdog fired. The previous entry's reading of deaths 1–2 as watchdog noise
  was wrong; the debounce stays, it just was not the cause.

  `start.sh` now caps the budget from the host side:
  `min(weights + overhead + MTP + max(kv_need, KV_TARGET_GIB), MemTotal −
  HOST_RESERVE_GIB)`, `HOST_RESERVE_GIB` defaulting to 26, KV derived from the
  capped budget ("KV target 22 reduced to 16.67 by HOST_RESERVE_GIB=26"), and
  refuses to launch if the capped KV is under what `MAX_MODEL_LEN` needs. It
  prints the reserve and the live non-vLLM host footprint (warns above 9 GiB),
  and warns when a pinned `GPU_MEMORY_UTILIZATION` exceeds the cap. The cgroup
  cap logic is unchanged; GPU allocations are not charged to it on GB10, so it
  never protected the host from this. `.env.sample` ships `KV_TARGET_GIB=16`
  and `HOST_RESERVE_GIB=26`.

  Measured on this host, same day, shipped profile (262k, FP8, MTP 3,
  `MAX_NUM_SEQS=5`), no kernel tunables applied:

  | | old (`KV_TARGET_GIB=22`, GMU 0.830) | new (GMU 0.780) |
  |---|---|---|
  | GPU budget | 100.9 GiB | 94.87 GiB |
  | KV pool (FP8) | 22.2 GiB | 16.46 GiB = 992,584 tok (3.79x a 262k req) |
  | time to `/health` | — | 10 min 51 s |
  | host MemAvailable idle | 6.9–8.8 GiB | 15.7 GiB at +2 min; 15.5–16.4 over 40 min |
  | host MemFree idle | ~4.9 GiB | 4.4–5.2 GiB |
  | after two ~90k prompts | ~6.9 GiB, never back | 15.2 GiB 60 s after the second (min 14.9 during; MemFree ≥ 3.5) |
  | five concurrent ~60k prompts | died under the harness | 14.57 GiB at +60 s (min 14.26; MemFree ≥ 3.24); 5/5 completed, no watchdog event |
  | `NV_ERR_NO_MEMORY` (`journalctl -k`) | 63 between 16:50 and 23:59 on 2026-09-04 | **0** across launch, both tests and 50 idle minutes |

  The per-request growth is still there and is now budgeted for, not fixed:
  the first 90k prompt moved the watchdog's `driver` figure from 95.6 to
  96.4 GiB and `MemAvailable` from 16.2 to 15.05 GiB, permanently.
  The second 90k prompt added nothing (96.4 → 96.4 GiB); five concurrent 60k
  prompts added 0.2 GiB (96.6). Session minimum over launch, both tests and
  50 idle minutes: `MemAvailable` 14.26 GiB, `MemFree` 3.24 GiB.
  The qwen-code harness that killed the old config then ran against the new
  budget for 2.5 hours (~38 requests, 19 of them 50–100k tokens, 3 over 100k,
  up to 3 concurrent): no watchdog event, no driver error, `MemAvailable`
  14.2–14.9 GiB between turns and 12.8 GiB at the low point, the driver figure
  flat at 96.6 GiB for three hours then one 0.9 GiB step to 97.5 on a 3-way
  batch — the growth mechanism, absorbed by the reserve as intended.

- **Watchdog: a second floor, richer timeline, logs archived before the stop**
  (`files/memwatch.sh`). It now also stops the container when `MemFree` stays
  under `MEMWATCH_MIN_FREE_GIB` (default 2) for 5 samples — but only while
  `MemAvailable` is under `MEMWATCH_FREE_GATE_GIB` (default 10). The gate was
  learned the hard way: the ungated version killed a healthy launch at 00:49
  because `MemFree` fell to 0.9 GiB while 32 GiB of weights streamed through
  the page cache (`MemAvailable` 32 GiB, zero driver errors). With the stock
  kernel watermarks `MemFree` is only meaningful once the cache is gone.
  Every 10 s it counts `NV_ERR_NO_MEMORY` in `journalctl -k` and logs any
  non-zero count. The 5 s timeline adds `cached`, `anon`, `shmem`, `mapped`,
  `sunreclaim` and a derived `driver` figure (`MemTotal − MemFree − Buffers −
  Cached − AnonPages − Slab − PageTables − KernelStack`), which is where the
  growth shows. Before `docker stop` it writes `docker logs --tail 3000` and a
  copy of its own log to `logs/archive/`; the grace period is 30 s
  (`MEMWATCH_GRACE`), and `start.sh` archives the previous run's container and
  watchdog logs before it relaunches. Observed: vLLM ignores SIGTERM while
  loading weights, so a stop in that phase ends in the SIGKILL fallback.

- **`stop.sh` discarded the container log.** It ran `docker rm -f` with no
  copy; a run stopped by hand had no post-mortem. It now archives `docker logs
  --tail 3000` and the watchdog log to `logs/archive/` first, like `start.sh`
  and `memwatch.sh`.

- **Stale comment in `.env`**: `MAX_NUM_BATCHED_TOKENS=2048` was labelled
  "local override: faster prefill" from the reverted 8192 experiment.

### Added

- **Reduced-vocabulary drafting for the MTP head** (`files/patch_mtp_draft_vocab.py`,
  `files/build_draft_vocab.py`, `MTP_DRAFT_VOCAB`). **The largest measured win
  on this host: -16.9% single-stream step time, and the only change so far that
  moves single stream at all.**

  The drafter carries its own BF16 `ParallelLMHead` over the whole 248,320-token
  vocabulary, 1.18 GiB, read once per draft step. At MTP 3 that is three of the
  four `lm_head` reads in an engine step -- about a third of every byte a
  single-stream step moves -- to produce one argmax. Draft sampling is greedy
  (`draft_sample_method` defaults to `"greedy"`, and the speculator only builds
  `draft_logits` for `"probabilistic"`, so *every* draft goes through
  `get_top_tokens` regardless of request temperature), so the drafter needs the
  arg max and nothing else, and ~74% of the vocabulary never wins it.

  The patch adds `get_top_tokens()` to `Qwen3_8FlashNextMTP`, reading a sliced
  BF16 head selected by token id, and leaves `compute_logits` on the full head
  so every other path keeps full-vocabulary behaviour. It engages only when
  `MTP_DRAFT_VOCAB` names a file, and `start.sh` then also passes
  `"use_local_argmax_reduction":true`, which is what routes the speculator
  through `get_top_tokens`. TP=1 only; it refuses to engage otherwise.

  At 65,536 rows the draft head is 0.31 GiB, saving **2.61 GiB per engine step**.
  Measured step time against the same server with the full head:

  | streams | full head | reduced head | delta |
  |---|---|---|---|
  | 1 | 76.9 ms  | 63.9 ms  | **-16.9%** |
  | 2 | 93.0 ms  | 79.0 ms  | -15.1% |
  | 4 | 117.3 ms | 101.6 ms | -13.4% |
  | 5 | 127.4 ms | 115.7 ms | -9.2%  |
  | 8 | 155.7 ms | 146.2 ms | -6.1%  |

  Three repeats each; the byte model predicts -18.0% at one stream and -6.2% at
  eight, so prediction and measurement agree within about a point at both ends.
  The saving is a fixed 2.61 GiB while the step grows with concurrency (each
  extra token pulls in ~10 more of the 512 MoE experts), which is why the gain
  shrinks as streams rise. Decode-phase throughput: 28.4 -> ~35 tok/s at one
  stream, 84.7 -> 101.3 at five.

  **Accuracy is unaffected, and that is structural rather than lucky.** The
  rejection sampler's greedy branch is
  `accepted = target_argmax == draft_sampled`, storing
  `draft_sampled if accepted else target_argmax`: a draft is kept only when it
  equals the target model's own choice, and otherwise the target's token is
  emitted. The patch changes only what is *proposed*; the verification path is
  untouched, so a reduced-vocabulary drafter is indistinguishable from a less
  accurate one, which is the condition rejection sampling exists to handle.
  Confirmed on MGSM (the same 250 grade-school problems in each language, exact
  numeric match, 8 concurrent, temperature 0):

  | | reduced 65k | full 248k | delta |
  |---|---|---|---|
  | en accuracy | 237/250 = 94.8% | 234/250 = 93.6% | +1.2 pts (0.57 sigma) |
  | zh accuracy | 216/250 = 86.4% | 216/250 = 86.4% | 0.0 pts (0.00 sigma) |
  | en throughput | 95.2 tok/s | 83.9 tok/s | **+13.4%** |
  | zh throughput | 84.4 tok/s | 82.4 tok/s | +2.4% |

  Chinese is the stress case: the shipped vocabulary covers 50.6% of the tokens
  the model emits in Chinese against 98.9% in English, and Chinese accuracy is
  *identical* to the problem, 216 of 250 either way. What coverage buys is
  speed, never correctness -- English gains 13%, Chinese gains nothing
  measurable because the unconditional byte saving and the acceptance the poor
  coverage costs cancel out. Out-of-vocabulary traffic comes out break-even, not
  slower, so the reduced head is safe for mixed traffic.

  Exact-text A/B is not available on this server: two passes over the same 26
  temperature-0 prompts on one unchanged config produced 0/26 identical outputs.
  Concurrent batch composition changes MoE/Marlin reduction order and flips
  near-tied logits. That nondeterminism predates this change; it is why the
  kernel and a graded task eval are the evidence here rather than a diff.

  Building the vocabulary needs a real corpus, and this is the part that nearly
  sank the item. A vocabulary fitted to the model's own generated output does
  not work: 52 generations gave 4,250 distinct tokens, so every size from 8k to
  65k was the same set at 76% held-out coverage, and generating enough would
  take a day of the server doing nothing else. Qwen's BPE id order is also a
  poor frequency proxy -- "keep every id below 32,768" covers only 80.9% of
  occurrences. What worked was 513 MiB of wikitext-103 plus 47 MiB of real
  Python source (x3) plus the model's own output (x20): 160M token occurrences,
  104,522 distinct ids.

      python3 files/build_draft_vocab.py english.txt code.txt:3 model_out.jsonl:20 \
          --size 65536 --out ~/.cache/vllm/draft_vocab/qwen38fn_en_code_65k.txt

  Corpus coverage by size: 8k 87.1%, 16k 93.2%, 32k 97.8%, 65k 99.84%. On the
  model's own English+code output, 32k covers 93.7% and 65k covers 97.3%. 65k
  ships because the crossover where the acceptance loss eats the byte saving is
  around 88-90% coverage, and 65k costs only 3 points of byte saving (18.0% vs
  21.2% of a single-stream step) to buy 3.6 points of coverage. The vocabulary
  is a generated artifact and is not tracked here; `MTP_DRAFT_VOCAB` empty
  restores full-vocabulary drafting.

  Not free in memory: the full 1.18 GiB head stays resident and the 0.31 GiB
  slice is added on top, so this spends ~0.31 GiB of GPU memory to save
  bandwidth. Under greedy draft sampling `compute_logits` is dead on the draft
  path, so the full head could be released later for another 1.18 GiB.

- **Batched page prefetch before the PLE row gather** (`files/patch_ple_layer.py`,
  `files/patch_ple_offload.py`). The offload worker gathers 16 rows per token
  out of a 26.82 GiB mmap with `torch.index_select`, which never reaches
  PyTorch's ~32k-element parallel grain at decode sizes, so every missing 4 KiB
  page was faulted in one at a time on one thread while the GPU worker spun on
  the handshake. Measured in-container against the real table, 100% cold rows:
  a 280-row gather takes 20.12 ms (71.4 us/fault) and does not improve with
  more torch threads (1 -> 17.39 ms, 4 -> 16.35, 8 -> 16.34), confirming the
  serial loop. `_ple_prefetch_rows()` now maps the gather's row ids to page
  offsets, dedups them (`torch.unique`, so the reads issue in ascending file
  order) and names them all with `posix_fadvise(WILLNEED)` through an fd kept
  beside the mmap, before the gather runs: the same 280-row gather drops to
  **1.51 ms, 13x**. Advisory only and wrapped, so any failure falls back to
  today's path. End to end the win is smaller, because only ~20% of lookups
  miss the page cache (measured 5.51 pages/generated token at 1 stream, 3.55 at
  5 -- the README's older 57 KiB/token figure implied ~87%): **-3.2% mean step
  time across 1/2/4/5/6/8 streams**, every one of the six improving. The
  page-cache confound is ruled out -- the patched run faulted *more* pages than
  the unpatched one (6.47 vs 5.51 per token at 1 stream) and was still faster,
  which is the prefetch signature. Note the fault arithmetic alone predicts
  ~0.6-2%, so ~1-2% of the measured gain is unexplained.

- **`CUDAGRAPH_CAPTURE_SIZES`** (`start.sh`, `.env.sample`, default `auto`).
  vLLM builds its decode graph list as `[1,2,4]` plus multiples of 8, rounds
  each to a multiple of `1+MTP`, then keeps only sizes `<= (1+MTP)*MAX_NUM_SEQS`.
  At MTP 3 that leaves decode keys `{4,8,16}` whatever `MAX_NUM_SEQS` is: a
  3-sequence batch (12 tokens) pads up to 16 and reads a fourth request's worth
  of experts for nothing, and at `MAX_NUM_SEQS=5` a full 5-sequence batch (20
  tokens) matches no key and decodes eager. Confirmed from the engine log
  (`cudagraph_capture_sizes: [1,2,4,8,16,24,32,40]`, 3 graphs captured), not
  inferred. `auto` captures every `(1+K(S))*S` the scheduler can build, honouring
  `MTP_K_SCHEDULE` when one is set; at `MAX_NUM_SEQS=4` that is `[4,8,12,16]`
  and 4 graphs. Worth ~4-5 ms on the 5-sequence step (the marginal cost of the
  5th stream fell 16.1 -> 10.6 ms) and nothing at 1/2/4 streams, where the
  graphs already existed -- an order of magnitude less than the +20-25% that had
  been projected by attributing the whole gap to the eager fallback. Keep it
  anyway: it is free, and it is the precondition for any `MAX_NUM_SEQS > 4`.

- **`MTP_K_SCHEDULE`** (`start.sh`, `.env.sample`, default empty) — dynamic
  speculative depth per batch size, `"start:end:K,..."`. **Measured as a
  regression on this build; ships disabled.** Setting
  `num_speculative_tokens_per_batch_size` makes vLLM override `cudagraph_mode`
  from `FULL_DECODE_ONLY` to `PIECEWISE`, so the computed capture sizes are
  never used. The byte saving is real (K=1 at 8 streams cut the step 160.8 ->
  127.4 ms) but tokens/step fell 2.29 -> 1.68 and the graph penalty ate the
  rest: 105 vs 114 tok/s. The single-stream control, where K is 3 either way,
  isolates that penalty at **79.6 -> 99.0 ms, +24%** -- far more than graphs are
  worth at 5 streams, because kernel-launch overhead hides under memory traffic
  at concurrency and is exposed without it. It also broke the memory budget:
  PIECEWISE graph memory drove the driver to 99.4 GiB against a 94.87 GiB
  budget, `MemAvailable` to 9.1 GiB and `MemFree` to 1.9 GiB, and the watchdog
  stopped the server (`MemFree under 2 GiB for 5 samples`, 7 `NV_ERR_NO_MEMORY`
  since watchdog start). The log names an untested escape hatch,
  `VLLM_USE_V2_MODEL_RUNNER=1`.

- **`COMPILATION_MODE`** (`start.sh`, `.env.sample`, default `0`) — torch.compile
  level, previously hard-coded. Untested above 0 here.

- **`files/sysctl-spark3.conf`** — `vm.min_free_kbytes=4194304`,
  `vm.watermark_scale_factor=300`, `vm.swappiness=30`, the values a sibling
  Spark measured six crash-free bring-ups with. `start.sh` warns when the box
  is at the kernel defaults (45155 / 10) and prints the `sysctl -p` command.
  **Not applied** by anything in this repo and not yet measured here: at those
  values the same physical state reads roughly 11–15 GiB lower in
  `MemAvailable` (computed from the kernel's watermark formula), so the 6 GiB
  watchdog floor has to be re-derived against a measured run first.

### Measured

- **End-to-end sweep after the optimisation pass**, sparkDash against the
  shipped profile (512k YaRN, 2,048 chunks, `MAX_NUM_SEQS=4`, MTP 3, FP8 KV,
  `KV_TARGET_GIB=20` -> 16.18 GiB = 974,768 tokens = 3.72x a 262k request).
  Same rope config and chunk width as the rows it replaces, so decode is a
  matched pair; prefill differs only in `KV_TARGET_GIB` (22 -> 20), which does
  not change the prefill rate. One run each.

  | decode, prose | before | after | change |
  |---|---|---|---|
  | 1 stream  | 36.9 tok/s | **46.3** | **+25.5%** |
  | 2 streams | 57.4 tok/s | **73.0** | +27.2% |
  | 3 streams | -          | 91.9     | - |
  | 4 streams | 85.9 tok/s | **108.1** | +25.8% |

  | prefill | before | after | change |
  |---|---|---|---|
  | 8k   | 1,646 tok/s | **1,764** (TTFT 4.67 s)   | +7.2%  |
  | 16k  | 2,052 tok/s | **2,265** (TTFT 7.25 s)   | +10.4% |
  | 32k  | 2,073 tok/s | **2,265** (TTFT 14.49 s)  | +9.3%  |
  | 64k  | 2,037 tok/s | **2,222** (TTFT 29.52 s)  | +9.1%  |
  | 128k | 1,945 tok/s | **2,110** (TTFT 62.15 s)  | +8.5%  |
  | 256k | 1,791 tok/s | **1,913** (TTFT 137.03 s) | +6.8%  |

  The single-stream decode figure independently reproduces the in-repo
  measurement taken with a different harness on different prompts (28.4 -> 35.7
  tok/s, +26%): different absolute numbers, same gain.

- **The prefill gain is the PLE prefetch, not the draft vocabulary**, which does
  not touch prefill at all. This reverses the priority the two items were given
  during the work. The PLE row gather runs per prefilled token, so a 2,048-token
  chunk gathers 16 rows per token -- about 32,768 of them -- against the ~256 a
  4-stream MTP-3 decode step gathers. That is the regime where batching the page
  faults measured 13x in isolation, and it explains why the same patch was worth
  only ~3% on decode: too few faults per step there for the fault latency to
  matter. So the gather fix is worth roughly three times more on prefill than on
  decode, and it was ranked last on decode evidence alone. Not isolated with an
  A/B -- attribution is inference from the mechanism, and one launch with
  `MTP_DRAFT_VOCAB` empty would separate the two.

- **Decode is on the memory-bandwidth wall, and that is what ranks the work.**
  Measured step time against the byte model in `docs/fable51-max.md`: 1.37x the
  floor at one stream, 1.09x at four, 1.04x at five, and 160.8 ms against a
  165 ms floor at eight. The consequence is that only *bytes removed* convert
  into time at concurrency. Reduced-vocabulary drafting was predicted at -18.0%
  and -6.2% at one and eight streams from the byte arithmetic alone and measured
  -16.9% and -6.1%, agreeing within about a point at both ends -- so the byte
  model can rank future work before a restart is spent on it. The two overhead
  items returned ~3% each; the one byte item returned 17%.

- **Run-to-run determinism.** Greedy decoding on this server is not
  reproducible: two passes over the same 26 temperature-0 prompts on one
  unchanged configuration produced 0/26 identical outputs, diverging 0.3-7% in.
  Concurrent batch composition changes MoE/Marlin reduction order and flips
  near-tied logits. This predates any change here and is why the draft-head
  comparison was settled with the rejection-sampler kernel and a graded task
  eval rather than a text diff.

## 2026-09-04

### Fixed

- **The memory watchdog killed healthy servers on a single noisy sample**
  (`c79f765`). `files/memwatch.sh` triggered on one `MemAvailable` reading below
  `MEMWATCH_MIN_GIB`. Two servers were lost to this; in both cases the samples
  either side of the trigger sat 400–700 MiB *above* the floor:

  ```
  23:32:32 avail=6542MiB          <- 398 MiB above the 6144 MiB floor
  23:32:33 MemAvailable=6090 MiB < floor -> docker kill
  ```

  `MemAvailable` moves ~107 MiB between 5 s samples here, with excursions past
  1 GiB, so a one-sample test against a fixed threshold fires on noise. The
  trigger now requires **5 consecutive** sub-floor samples; replayed against
  both recorded crash sequences it does not fire, and it still fires on a
  sustained decline. A lone excursion logs `recovered after N sub-floor
  sample(s)` and resets the counter.

  Not yet proven live: the debounce has not fired in real conditions, and the
  slow downward trend in `MemAvailable` under sustained long-context prefill is
  still unexplained.

- **The watchdog's own kill was invisible in its log** (`c79f765`). It polled
  every second but logged every fifth sample, so the reading that caused a kill
  was never in the post-mortem. It now logs every sample once within 1 GiB of
  the floor.

- **The watchdog leaked POSIX shared memory** (`c79f765`). It used `docker kill`
  (SIGKILL); with `--ipc host` that strands the container's `/dev/shm` segments
  until reboot — the leak `stop.sh` already takes care to avoid. It now sends
  SIGTERM with a 10 s grace period (`MEMWATCH_GRACE`) and falls back to SIGKILL.

- **An incomplete checkpoint got past the pre-flight check** (`b0a9f5e`).
  `start.sh` and `download.sh` treated "`config.json` exists" as "checkpoint
  complete", but `config.json` lands early in a download, so an interrupted
  fetch failed later inside vLLM instead. Both now require every shard named by
  `model.safetensors.index.json` (35 here). Catches missing and dangling-symlink
  shards, not truncated blobs. Split out of #2; co-authored with @lidaiqing.

### Changed

- **FP8 KV: per-tensor scales hoisted out of the QSA dots** (`69f7b4c`, from #2,
  co-authored with @lidaiqing). The kernels previously dequantised each tile with
  vLLM's `_cast_kv_tile`, which materialises an FP32 tile
  (`(data.to(tl.float32) * scale).to(Q.dtype)`); `block_n` was halved to keep
  that inside GB10's shared-memory budget. They now cast FP8→BF16 and apply the
  scalar scales after the dots, which removes the FP32 tile and restores the
  BF16 tile width.

  Exact before rounding: the score scale is a scalar, so it commutes with the
  `softmax_scale` multiply and the validity mask, and the output scale is
  applied to `normalized_output` above the `NUM_SPLITS` branch so it factors
  through the split-K LSE merge. Slightly *more* accurate than dequantising
  first, since FP8→BF16 is exact while rounding `scale × fp8` into BF16 is not.

  Measured by the contributor: +6.3% prefill @32k, −40.6% sparse-QSA kernel
  latency, one BF16 ULP maximum error, BF16 path bit-identical. End-to-end on
  this host the kernel alone came out around +3% at 32k against an unmatched
  baseline — inside run-to-run noise. No matched A/B has been run here.

- **`.env.sample` documents `MAX_NUM_BATCHED_TOKENS`** and keeps the default at
  2048 (`366f6df`). Raising it to 8192 measured, on this host, 32k prefill
  2,133 → 2,366 tok/s (+10.9%) and TTFT 15.38 → 13.87 s (−9.8%), with the whole
  8k–128k curve flattening. It is offered as an opt-in knob rather than a
  default because the supporting observation is minutes, not hours. It is paid
  for out of the KV pool rather than the GPU budget: 8192 chunks raise peak
  activation to 1.27 GiB, which vLLM profiles before sizing the KV cache.

- **README prefill table now carries both chunk widths side by side**, each
  labelled with the configuration it was measured at (`366f6df`). They differ in
  rope config and KV target as well as chunk width, so only the 32k pair is a
  clean A/B.

- **`start.sh`'s FP8 warning dropped a stale speed claim** (`69f7b4c`). It cited
  the reference implementation's ~30% slower prefill and ~9% slower decode; the
  quality warning (a long-reasoning benchmark falling 6/6 → 2/6) stands.

- **README: `### FP8 KV cache (opt-in)` → `(default)`**, and the patch bullet's
  "Off by default" corrected (`69f7b4c`). Both contradicted the body text and
  the shipped `KV_CACHE_DTYPE=fp8`.

### Reverted

- **`KV_TARGET_GIB` 22 → 20 → 22 → 20.** Lowered in `554f295` on the theory that
  the watchdog kills were a memory problem, restored in `c79f765` once those two
  kills turned out to be a watchdog bug, then lowered again once a third server
  died on a genuine sustained decline. Both things were true: the watchdog fired
  on noise *and* the host margin at 22 is thin. The default is 20.

- **`MAX_NUM_BATCHED_TOKENS` default 2048 → 8192 → 2048.** Defaulted in
  `677f4ab`, backed out in `366f6df` and documented as an opt-in instead. See
  above for the measurements.

### Verified

- **The debounced watchdog correctly distinguished a real event from noise.** A
  third server died at 23:53 on a monotonic descent — 7,101 MiB to 5,726 MiB in
  9 seconds, still falling — and the trigger fired after 5 consecutive sub-floor
  samples. The per-sample near-floor logging captured the whole descent, which
  the old script could not have shown. Unlike the two earlier kills, the
  container's own cgroup was *growing* through this one (10,716 → 12,348 MiB),
  so the mechanism differs from the slow drift.

  The 10 s SIGTERM grace was not enough — `docker stop` escalated to SIGKILL and
  the exit code was still 137, so the shm-leak protection did not take effect.

### Known open

- Host `MemAvailable` has two unexplained behaviours: a slow decline under
  sustained long-context prefill, and at least one burst that consumed ~1.4 GiB
  in 9 seconds. `memwatch.sh` logs `MemAvailable`, `MemFree`, `SwapFree` and
  the container's cgroup usage; adding `Mapped`/`Cached`/`AnonPages` would let
  the next descent identify its own cause.
- The shipped default (262k, `KV_TARGET_GIB=22`, FP8, 2048 chunks) has still not
  been benchmarked end to end.
- Whether the 6 GiB watchdog floor is the right threshold has never been
  examined; it was chosen during bring-up.
- `MEMWATCH_GRACE` (10 s) is too short for vLLM to shut down cleanly.
