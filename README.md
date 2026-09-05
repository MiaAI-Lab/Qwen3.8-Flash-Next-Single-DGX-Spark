<h1 align="center">Qwen3.8-Flash-Next on ONE DGX Spark (TP=1)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Self-contained recipe for serving the `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`
checkpoint (99 GB) from a single DGX Spark's 121 GiB unified memory, via vLLM
with the PLE table offloaded and memory-mapped. This is a **vision-language**
model: text, images and video all work out of the box (see below). Nothing here depends on the
2-node files it was derived from.

```
cp .env.sample .env        # edit IMAGE / HF_TOKEN if needed
./download.sh              # fetch the ~99 GB checkpoint (resumable)
./start.sh                 # ~10-12 min to /health; serves on :8888
./stop.sh                  # container + watchdog, graceful
```

`start.sh` never downloads anything — it resolves the checkpoint from the local
Hugging Face cache and fails fast if it is absent. Budget ~130 GiB of free disk:
99 GB for the checkpoint plus the ~27 GB packed PLE table built on first launch.

`./start.sh --no-launch` prints the derived memory budget and the docker
command without running anything. `./stop.sh` sends SIGTERM and waits up to
`STOP_TIMEOUT` (default 30 s) so vLLM can unlink its POSIX shared memory —
the container runs with `--ipc host`, so segments it leaves behind leak onto
the host's `/dev/shm` until reboot. `./stop.sh --force` skips the wait.

## Measured profile

`.env.sample` ships **262,144 context (YaRN off), MTP 3, `HOST_RESERVE_GIB=26`,
`KV_TARGET_GIB=16`, `KV_CACHE_DTYPE=fp8`, `MAX_NUM_SEQS=4`,
`MAX_NUM_BATCHED_TOKENS=2048`**.
Everything below was measured on this host on 2026-09-04; each row names the
configuration it came from, because the numbers move a lot between them.

| Configuration | KV pool | Prefill @400k | Decode (prose, idle) | Needles 5/50/95% |
|---|---|---|---|---|
| 262k, `KV_TARGET_GIB=20`, BF16 | 21.28 GiB = 736,837 tok (2.81x a 262k req) | — | — | — |
| 512k YaRN, `KV_TARGET_GIB=20`, BF16 | 19.2 GiB = 704,558 tok (1.34x a 512k req) | 1,537 tok/s (TTFT 260.3 s) | 28.3 tok/s (sd 2.6) | 3/3 PASS |
| 512k YaRN, `KV_TARGET_GIB=22`, BF16 | 796,196 tok (1.52x a 512k req) | 1,883 tok/s @32k | — | 12/14 (see FP8 section) |
| 512k YaRN, `KV_TARGET_GIB=22`, FP8 | 22.2 GiB = 1,431,164 tok (2.73x a 512k req) | 1,495 tok/s (TTFT 267.7 s); 1,769 tok/s @32k | 27.1 tok/s (sd 1.3) | 15/20 (see FP8 section) |

The shipped profile itself (262k, `HOST_RESERVE_GIB=26`, `KV_TARGET_GIB=16`,
FP8, `MAX_NUM_SEQS=5`) was measured on 2026-09-05:

| | |
|---|---|
| GPU budget | GMU 0.780 = 94.87 GiB |
| Available KV cache | 16.46 GiB = **992,584 tokens** (3.79x a 262k request) |
| Time to `/health` | 10 min 51 s (checkpoint read from NVMe) |
| Host MemAvailable, 2 min after `/health` | 15.7 GiB (MemFree 5.1 GiB) |
| Host MemAvailable, 40 idle minutes | 15.5–16.4 GiB (MemFree ≥ 4.4 GiB) |
| Host MemAvailable after two ~90k prompts | 16.2 → 15.05 GiB after the first, 15.2 GiB 60 s after the second (min 14.9 during prefill; MemFree ≥ 3.5 GiB) |
| Host MemAvailable, five concurrent ~60k prompts | 14.9 → 14.57 GiB at +60 s (min 14.26 during; MemFree ≥ 3.24 GiB); 5/5 completed, no watchdog event |
| 2.5 h under the qwen-code harness (~38 requests, 19 of them 50–100k tokens, up to 3 concurrent) | 14.2–14.9 GiB between turns, min 12.8 GiB at 3 concurrent; driver 96.6 → 97.5 GiB in one step |
| `NV_ERR_NO_MEMORY` in `journalctl -k` | 0 across launch and all of the above |

The idle figure used to be quoted here as ~12.9 GiB; that came from a run at
`KV_TARGET_GIB=20` before the day's co-tenants were on the box. See the safety
rules for why the number matters.

One honest gap: the **shipped default itself** (262k, `KV_TARGET_GIB=20`,
FP8) has not been benchmarked end to end — the 262k row above is at a lower KV
target and BF16, and predates the `MADV_RANDOM` mmap change. Short-context
(32k) prefill is now measured for both dtypes; see the FP8 section.

`KV_TARGET_GIB` shipped as 22, then 20, until 2026-09-05. Both lost servers:
three on 2026-09-04. The rows above at 22 are real measurements, but the host
they were taken on had 6.9–8.8 GiB of `MemAvailable` left, against a 6 GiB
watchdog floor and a GPU driver that refuses allocations before that. The
budget is now capped from the host side (`HOST_RESERVE_GIB`, see
[Safety rules](#safety-rules)); the KV pool is whatever the cap leaves, and 16
sits just under it. 20 and 22 are no longer reachable through `KV_TARGET_GIB`
alone; a pinned `GPU_MEMORY_UTILIZATION` still gets you there, with a warning.

### Prefill and decode, measured with sparkDash

Both sweeps below were measured with
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) against this server.
Benchmark scripts are not shipped in this repo; use sparkDash to reproduce
them. The two prefill columns are **not** a clean A/B — they differ in rope
config and KV target as well as chunk width — so each is labelled with what it
was measured at.

**Prefill.** At the shipped 2,048-token chunk width throughput peaks around
32-64k and falls away with context. Raising `MAX_NUM_BATCHED_TOKENS` to 8,192
flattens it from 16k out to 128k, because per-chunk overhead is amortised over
4x fewer chunks:

| context | shipped: 2,048 chunks (512k YaRN, `KV_TARGET_GIB=22`) | opt-in: 8,192 chunks (262k native, `KV_TARGET_GIB=20`) |
|---|---|---|
| 8k | 5.00 s · 1,646 tok/s | **3.69 s · 2,228 tok/s** |
| 16k | 8.00 s · 2,052 tok/s | **7.16 s · 2,293 tok/s** |
| 32k | 15.83 s · 2,073 tok/s | **13.87 s · 2,366 tok/s** |
| 64k | 32.20 s · 2,037 tok/s | **28.31 s · 2,316 tok/s** |
| 128k | 67.41 s · 1,945 tok/s | **58.88 s · 2,227 tok/s** |
| 256k | 146.40 s · 1,791 tok/s | not re-measured |

The one matched pair — same server, same kernel, only the chunk width changed —
is 32k: **2,133 → 2,366 tok/s (+10.9%), TTFT 15.38 → 13.87 s (−9.8%)**. The
rest of the right-hand column is one run each and should be read as indicative.

### Raising the prefill chunk width (opt-in)

**The 8,192 column is not the default.** If you want the faster prefill and
TTFT above, set it yourself:

```
MAX_NUM_BATCHED_TOKENS=8192 ./start.sh     # one launch
```

or edit the line in `.env` to make it stick.

It is paid for out of the KV pool, not the GPU budget. 8,192 chunks raise peak
activation to 1.27 GiB, and vLLM profiles that *before* it sizes the KV cache,
so the pool absorbs it: 1,145,289 tokens measured at `KV_TARGET_GIB=20`,
still 4.37x a full 262k request. The best matched evidence for the size of that
trade is PR #2's own pair at `KV_TARGET_GIB=22`, 1,282,724 → 1,249,637 tokens —
about **−2.6%** of pool for **+11%** prefill.

Host `MemAvailable` sat at 8.1-8.3 GiB idle and low-watered at 7.4 GiB across
the 8,192 sweep. It is offered as a knob rather than a default because the
supporting observation is minutes, not hours: if you serve long sessions near
the memory floor, measure it on your own workload before committing to it.

**Decode on prose** (2,048 chunks, 512k YaRN), by concurrent stream count:

| streams | TTFT | aggregate | per stream |
|---|---|---|---|
| 1 | 418 ms | 36.9 tok/s | 36.9 tok/s |
| 2 | 445 ms | 57.4 tok/s | 29.7 tok/s |
| 4 | 550 ms | **85.9 tok/s** | 23.4 tok/s |

Decode speed on this model is **strongly content-dependent**, because MTP
speculative decoding accepts more drafts on predictable text. Measured on this
server: mean acceptance length 2.1 of a possible 4, per-position acceptance
0.65 / 0.33 / 0.14, average draft acceptance 37-41%. Highly predictable output
(quoting text back out of the context) reaches ~41 tok/s; dense technical prose
sits lower. Treat single-stream decode as a range rather than one number.


## Multimodal (images and video)

The checkpoint is multimodal (`is_multimodal: true`, `language_model_only:
false`, a 27-layer vision tower) and the launcher enables it by default —
nothing extra to configure. The vision tower is already counted in the
"weights on GPU" figure, so images and video cost no additional GPU budget.

Verified on this host 2026-09-04 against the running server:

| Modality | Test | Result |
|---|---|---|
| Image | 336x336 PNG, three colour bands | named all three in order; 179 prompt tokens |
| Video | 4 s clip, 16 frames, one colour per second | named all four **in temporal order**; 376 prompt tokens |

Use the standard OpenAI content-part shapes — `image_url` and `video_url`,
either an `http(s)://` URL or a `data:` URI:

```
curl -s localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"qwen3.8-flash-next","max_tokens":600,"temperature":0,
 "messages":[{"role":"user","content":[
   {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}},
   {"type":"text","text":"Describe this image."}]}]}'
```

Three things to know before leaning on it:

- **MTP speculative decoding degrades on multimodal requests.** The draft model
  cannot take multimodal embeddings, so vLLM logs `using text-only draft inputs
  instead` and falls back for those requests. The answer is still correct — the
  target model sees the image — but decode runs closer to the non-speculative
  speed. Text-only requests are unaffected.
- **Video is token-hungry.** Frame count and resolution drive prompt length
  fast. At `YARN=1` you have only 1.34x a full-length request in KV across
  `MAX_NUM_SEQS=4`, so concurrent video work contends; the 262k profile
  (2.81x) has far more headroom for it.
- **Long video at 512k is untested here.** The tests above were long-text *or*
  short-multimodal, never both at once.

## Configuration

Precedence is **environment > `.env` > built-in default in `start.sh`**, so any
knob can be overridden per launch:

```
MAX_MODEL_LEN=65536 MTP_NUM_SPECULATIVE_TOKENS=0 ./start.sh
```

The safety-relevant knob is `HOST_RESERVE_GIB` (default 26): the GPU budget
is capped at `MemTotal − HOST_RESERVE_GIB` no matter what `KV_TARGET_GIB`
asks for, and `start.sh` prints "KV target X reduced to Y" when the cap binds.
`KV_TARGET_GIB` is a wish under that cap (16 gives ~1M FP8 tokens here).
`HOST_SLACK_GIB` sizes the container cgroup cap (GPU budget + this); it bounds
host-side memory only and does not protect the host from the GPU side.

### Long context beyond 262k (YaRN)

The model's native context is 262,144. Going past it needs YaRN rope scaling,
which is off by default. The two lengths live side by side in `.env` and the
`YARN` flag alone picks which one is served:

```
YARN=0                     # 0 = native rope, 1 = YaRN
MAX_MODEL_LEN=262144       # served at YARN=0; cannot exceed native 262144
YARN_MAX_MODEL_LEN=524288  # served at YARN=1; ignored entirely at YARN=0
```

So `YARN=1` is the only edit needed to go to 512k, and flipping it back to `0`
returns to 262k without touching anything else. For a single launch:
`YARN=1 ./start.sh`.

`start.sh` derives the scaling factor itself (`YARN_MAX_MODEL_LEN / 262144`,
rounded up — 2.0 for 512k) and passes it to vLLM as a `--hf-overrides`
deep-merge into `text_config.rope_parameters`, which is the field this model
actually reads. The existing `mrope_section`, `rope_theta` and
`partial_rotary_factor` are preserved, so the attention path keeps the same
`MRotaryEmbedding` and mrope stays enabled.

512k fits with **no other change**: it needs 14.4 GiB of KV, well inside what
`KV_TARGET_GIB` provides at the shipped 20 or at 22, and the GPU budget and
cgroup cap are unchanged from 262k. Measured at `YARN=1`, BF16,
`KV_TARGET_GIB=20` (2026-09-04):

| | |
|---|---|
| Available KV cache | 19.2 GiB = **704,558 tokens** (1.34x a full 524,288 request) |
| Host MemAvailable idle | ~11.3 GiB |
| Output | coherent; MTP 3 and YaRN run together without incident |

At `KV_TARGET_GIB=22` with FP8 the same context gets 2.73x headroom instead of
1.34x — see [FP8 KV cache](#fp8-kv-cache-default).

400k prefill stress test (salted to defeat prefix caching, needles planted at
5% / 50% / 95% depth):

| | |
|---|---|
| Prompt | 400,062 tokens |
| TTFT (prefill) | 260.3 s = **1,537 tok/s** |
| Needle retrieval | **3/3 PASS**, including 95% depth |
| Host MemAvailable low-water | **10.97 GiB** (watchdog floor is 6 GiB) |
| Peak container RSS | 18.7 GiB of the 103 GiB cap |

Decode measured 40 tok/s on that run, but the answer is three codes copied out
of the context — MTP's best case, not typical decode speed.

| Setting | Result |
|---|---|
| `YARN=1` | serves `YARN_MAX_MODEL_LEN`; `MAX_MODEL_LEN` is ignored (logged) |
| `YARN=0` with `MAX_MODEL_LEN` > 262144 | refused: tells you to set `YARN=1` |
| `YARN_MAX_MODEL_LEN` > `YARN_CEILING_MODEL_LEN` (524288) | refused: above the validated ceiling |
| `YARN=1` with `YARN_MAX_MODEL_LEN` at or below 262144 | warns, serves that length with native rope |
| 1M even with the ceiling raised | refused by the Step 2 budget check (cap 112 GiB vs 105 GiB ceiling) |

YaRN trades some short-context accuracy for the longer window, so leave it off
unless you need more than 262k. The 512k path serves correctly but its decode
and prefill speeds have not yet been benchmarked.

### Reasoning is on by default

This build reasons before answering, and `start.sh` passes
`--reasoning-parser qwen3`, so the thinking block arrives in a separate
`reasoning` field rather than inside `content`. The chat template enables it
whenever the flag is unset:

```jinja
{%- if enable_thinking is undefined or enable_thinking is true %}
```

Turn it off **per request** — no restart, so reasoning and non-reasoning
traffic can share one server:

```json
{"model":"qwen3.8-flash-next",
 "chat_template_kwargs":{"enable_thinking":false},
 "messages":[{"role":"user","content":"What is 17*23? One line."}]}
```

Measured on this host:

| | default | `enable_thinking: false` |
|---|---|---|
| reasoning tokens | 41 | **0** |
| completion tokens | 47 | **12** |
| `content` | `"\n\n391"` | `"17 * 23 = 391"` |

Two consequences worth knowing:

- **It is why `content` can come back empty.** With a small `max_tokens` the
  reply is often still inside its reasoning. Budget ~400+ tokens, or disable
  thinking. This is not a bug — see the sanity test above.
- **It dominates latency on simple work.** Reasoning ran to 4,841 tokens on the
  hardest task in our suite. For extraction, classification or short factual
  answers, disabling it is a large win; leave it on for anything that needs
  actual multi-step reasoning.

### FP8 KV cache (default)

`KV_CACHE_DTYPE=fp8` roughly doubles the KV pool by storing the main KV in
fp8-e4m3. The QSA Triton kernels cast FP8 tiles to BF16 for the tensor-core
dots and apply the per-tensor K/V scales once, to the score and the output
accumulator. **This is the shipped default**, on the strength of the
measurements below.

```
KV_CACHE_DTYPE=fp8    # ~2x KV pool, enables a 1M context (default)
KV_CACHE_DTYPE=auto   # BF16 KV, if you would rather not take the trade
```

Measured on this host 2026-09-04, identical prompts, `YARN=1`,
`KV_TARGET_GIB=22`, idle server:

All rows below are at matched settings (`KV_TARGET_GIB=22`, 512k YaRN) unless
noted. KV pool varies a little between restarts, so a range is given.

| | BF16 | FP8 | Δ |
|---|---|---|---|
| KV pool | 779,671–796,196 tok | **1,431,164–1,502,014 tok** | **~1.8–1.9x** |
| Concurrency @ 524,288 | 1.49–1.52x | **2.73–2.86x** | ~+85% |
| Prefill @400k | 1,537 tok/s | 1,495 tok/s | −2.7% |
| Prefill @32k (2 runs each) | 1,883 tok/s | 1,769 tok/s | −6.1% |
| Decode (prose, idle) | 28.3 tok/s (sd 2.6) | 27.1 tok/s (sd 1.3) | −4% |
| Reasoning suite (11 tasks) | **11/11** | **11/11** | same |
| Needle miss rate @32k | 2/14 (14%) | 5/20 (25%) | p=0.67, **n.s.** |

Only the 12 full-attention layers shrink (~84% of bytes/token); the QSA
side/compressor caches stay BF16, which is why the gain is ~1.85x rather than
2x, and why `KV_MULT` in `start.sh` is 0.58 rather than 0.5.

**The short-context penalty has since been removed.** The original patch
dequantised each tile with vLLM's `_cast_kv_tile`, which materialises an FP32
tile (`(data.to(tl.float32) * scale).to(Q.dtype)`), and halved `block_n` to
keep that inside GB10's shared-memory budget. That cost more on a short kernel
than a long one: −6.1% at 32k versus −2.7% at 400k, in the rows above.

Hoisting the per-tensor scales outside the dots removes the FP32 tile, so FP8
runs at the same `block_n` as BF16. The scales are scalars, so this is exact
before rounding: `(Q·K)·k_scale` for the score, and `v_scale` on the normalised
output, which factors cleanly through the split-K LSE merge. It is also
slightly *more* accurate than dequantising first — FP8→BF16 is exact, whereas
rounding `scale × fp8` into BF16's 8-bit mantissa is not.

Measured by [@lidaiqing](https://github.com/lidaiqing) on this host (#2), FP8
before vs after the hoist, at matched settings:

| | before | after | Δ |
|---|---|---|---|
| Prefill @32k | 1,827 tok/s | 1,942 tok/s | **+6.3%** |
| Decode, 1 stream | 24.06 tok/s | 23.36 tok/s | −2.9% |
| Decode, 4 streams | 56.44 tok/s | 60.91 tok/s | +7.9% |
| Decode, 8 streams | 60.16 tok/s | 60.43 tok/s | +0.4% |
| Sparse QSA kernel, 512 rows | 2.984 ms | 1.772 ms | **−40.6%** |
| Block selector kernel | 0.1392 ms | 0.1008 ms | −27.6% |

The kernel is 40% faster in isolation but attention is not the bottleneck at
these settings, so end-to-end decode barely moves; the win that survives is
short-context prefill. Single-stream decode is within this model's
content-dependent MTP variance. On identical tensors the maximum
sparse-attention error was 1.53e-5 (one BF16 ULP) and the BF16 path was
bit-identical, as the algebra predicts.

**On quality.** Both dtypes score 11/11 on the reasoning suite
(4 multi-step short tasks, 4 long chain-of-thought up to ~4,800 reasoning
tokens, 3 tasks combining three facts from a ~100k-token context). Both max it
out, so the honest reading is *no gross regression at n=11* — enough to rule
out the 6/6 → 2/6 collapse the reference measured, not enough to detect finer
drift. A suite everything passes cannot rank anything.

**A caution about needle tests on this model.** The 95%-depth needle at 32k is
flaky *regardless of KV dtype*: BF16 missed it 2/14 times, FP8 5/20, which
Fisher's exact test cannot distinguish (p=0.67). The two shallower needles were
found 34/34 times in both. So a single needle run is weak evidence here — an
isolated PASS or FAIL at 95% depth says little, and comparisons need matched
sample counts on both sides. The 3/3 results quoted elsewhere in this README
are single samples and should be read with that in mind.

**A caveat before trusting these numbers: quality is not settled.** Needle retrieval passing at 5/50/95% depth shows the
scales and dequantisation are broadly right, and short factual/arithmetic
answers were correct. It does **not** clear the failure mode that matters: the
reference measured a long-reasoning benchmark falling from **6/6 to 2/6** with
FP8 KV. This is sparse attention — quantised keys perturb which blocks the
indexer selects, not merely the attention output — so degradation can appear
as fluent, plausible, wrong reasoning while needles still pass. No
long-reasoning A/B has been run on this host, and the scale hoist has not
changed that — its one-BF16-ULP bound is a numerical result, not a quality
one. Treat FP8 as a capacity trade for workloads you have validated
yourself.

### PLE mmap access pattern

The packed PLE table is advised `MADV_RANDOM` (in `patch_ple_offload.py`).
Without it the kernel faults in a ~64 KiB window to serve each 90-byte row
lookup. Measured on this host:

| | default mmap | `MADV_RANDOM` |
|---|---|---|
| Disk read per decoded token | ~1,366 KiB | **57 KiB** (−24x) |
| Decode | 26.3 tok/s (sd 1.6) | 28.3 tok/s (sd 2.6) |
| Host MemAvailable | ~10.9 GiB | **~12.95 GiB** |

The decode difference is within noise — decode was never disk-*throughput*
bound (1.4 MiB/token at 26 tok/s is only ~36 MB/s). The real win is the
~2 GiB of unified memory no longer wasted on readahead that is thrown away,
which is what funds the KV pool `KV_TARGET_GIB` asks for.

## Safety rules

Each of these cost a hard host hang or a dead server during bring-up.

- **Budget the GPU from the host side.** vLLM detects this GPU as integrated
  and treats host `MemAvailable` — page cache included — as free GPU memory,
  then fills the GPU side to exactly `GMU × MemTotal`. Nothing in vLLM keeps
  anything back for the host. `start.sh` therefore caps the budget at
  `MemTotal − HOST_RESERVE_GIB` (26 GiB by default) and derives the KV pool
  from the remainder. What the reserve has to hold, measured here: other
  containers and sessions ~7 GiB (`start.sh` prints the live figure as "host
  footprint now" and warns above 9), vLLM's own host-side processes ~6, the PLE
  page cache that keeps decode off NVMe ≥6, free pages the NVIDIA driver needs
  to allocate at all ≥3, and 2–3 GiB of per-request growth (below). The page
  cache is not spare memory.
- **Keep host `MemAvailable` at or above ~10 GiB under load.** Exhausting the
  unified pool hangs the kernel with no OOM kill and no logs; the driver starts
  refusing allocations (`NV_ERR_NO_MEMORY` in `journalctl -k`, which works
  without sudo) well before that, at `MemFree` ~3 GiB.
- **`comfy-h3.service` must stay disabled.** It polls `127.0.0.1:8888` and
  launches ComfyUI (a GPU co-tenant) as soon as anything answers there.
  `start.sh` refuses port 8888 while that service is active.
- **Never set `PLE_OFFLOAD=false` at TP=1** — 99 GB through UVM hangs the host.
- **The stock QSA backend refuses FP8 KV** (`supported_kv_cache_dtypes =
  ["auto","bfloat16"]`). `patch_qsa_fp8_kv.py` in this repo adds it; without
  that patch `KV_CACHE_DTYPE=fp8` cannot work, and reading a quantised cache
  as BF16 would produce silent garbage rather than an error.
- **Do not raise `YARN_CEILING_MODEL_LEN` past 524288 at BF16.** A 1M context
  needs ~28.8 GiB of KV, driving the container cap to 112 GiB against a 105 GiB
  hard ceiling; `start.sh` refuses it at two independent checks. With
  `KV_CACHE_DTYPE=fp8` a 1M request needs only ~16.7 GiB and the budget does
  fit — but 1M has **never been run on this host**, at either dtype. Raising
  the ceiling means you are the one testing it.
- `docker --memory` does not bound GPU allocations on GB10, only host-side
  memory. vLLM's `--gpu-memory-utilization` is what bounds the GPU.
- **Kernel VM tunables.** The box ships with `vm.min_free_kbytes=45155` and
  `vm.watermark_scale_factor=10`: a 44 MB free-page floor and reclaim that
  starts at 0.1 %. `files/sysctl-spark3.conf` holds the values a sibling Spark
  measured six crash-free bring-ups with; `start.sh` warns when the box is at
  the defaults. They are **not applied** by anything in this repo, and the
  file's header explains why the watchdog floor must be re-derived before
  they are: at those values the same physical state reads roughly 11–15 GiB
  lower in `MemAvailable` (computed from the kernel's watermark formula, not
  measured here).

### What happened on 2026-09-04

Three servers died in one evening at `KV_TARGET_GIB=22`, all under a qwen-code
agent harness (up to five agents, 370 requests averaging 72k input tokens over
five hours, pointed at `127.0.0.1:8888`). The budget arithmetic left 20.7 GiB
of the 121.6 GiB pool for everything that is not the GPU, against the ≥22 GiB
listed above. `sar` shows the first server spending its last hour at 6.3–6.6
GiB of `MemAvailable`; the kernel log shows the driver refusing four
allocations in the eight seconds before the second death; the watchdog's own
log shows the third at `MemFree` 2.6 GiB. The earlier reading of the first two
deaths as watchdog noise was wrong: the debounce added that day is a good
change and does not touch the cause.

The growth is real and permanent. Each new largest request (70–95k tokens)
grows driver-side memory by ~2 GiB — workspaces the startup profile never
touched, held by PyTorch's caching allocator, which this build only releases
under pressure inside the model loader, never while serving. In the watchdog
log it appears as the container cgroup going *down* (PLE page cache evicted)
while `MemAvailable` goes down and `MemFree` stays flat; the new `driver`
column makes it visible directly. The reserve is sized to absorb it.

### Watchdog

`files/memwatch.sh` runs alongside the container, polls `/proc/meminfo` every
second, and stops the container on either of two floors, each debounced over
**5 consecutive** samples (a lone excursion logs `recovered after N sub-floor
sample(s)` and resets the counter — `MemAvailable` moves ~107 MiB between
samples here, with excursions past 1 GiB):

- `MemAvailable < MEMWATCH_MIN_GIB` (default 6): the page cache is gone.
- `MemFree < MEMWATCH_MIN_FREE_GIB` (default 2) **while** `MemAvailable <
  MEMWATCH_FREE_GATE_GIB` (default 10): the driver's failure point. The gate
  is not optional. With the stock watermarks `MemFree` legitimately sits near
  zero whenever the page cache is full of reclaimable data — measured during
  weight loading: `MemFree` 0.9 GiB, `MemAvailable` 32 GiB, zero driver
  errors — and an ungated version of this trigger killed a healthy launch.

Every 10 s it counts `NV_ERR_NO_MEMORY` lines in `journalctl -k` and logs any
non-zero count. Read it together with `MemAvailable`: a handful during
startup, when the driver takes the weights and then the KV pool in two large
bursts while `MemFree` is transiently ~1 GiB under the page cache from the
checkpoint read, is the driver bouncing off free pages and retrying (measured
2026-09-05 08:15–08:16: five of them at `MemAvailable` 17–34 GiB, launch
succeeded; the launch seven hours earlier had none — it depends on where
kswapd is when the burst lands). The fatal pattern is the same line with
`MemAvailable` under ~10 GiB, when there is no cache left to reclaim. The
timeline
(every 5 s, every sample once within 1 GiB of a floor) carries `avail`,
`free`, `swapfree`, the container cgroup, `cached`, `anon`, `shmem`, `mapped`,
`sunreclaim` and the derived `driver` figure (`MemTotal − MemFree − Buffers −
Cached − AnonPages − Slab − PageTables − KernelStack`: memory outside page
cache, anon and cgroup accounting, i.e. taken through the NVIDIA driver;
95.5 GiB at idle here against a 94.87 GiB budget).

Before stopping it archives `docker logs --tail 3000` and a copy of its own
log to `logs/archive/<container>-<timestamp>-{container,memwatch}.log`, then
sends SIGTERM with a 30 s grace period (`MEMWATCH_GRACE`) and falls back to
SIGKILL, so vLLM can unlink its POSIX shared memory — a hard kill leaks those
segments onto the host's `/dev/shm` until reboot, because the container runs
with `--ipc host`. vLLM does not honour SIGTERM while still loading weights;
a stop in that phase ends in the SIGKILL. `start.sh` archives the previous
container and watchdog logs the same way before it relaunches.

## Sanity test

```
curl -s localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"qwen3.8-flash-next","temperature":0,"max_tokens":400,
 "messages":[{"role":"user","content":"In one sentence, what is a DGX Spark?"}]}' \
 | python3 -c "
import json,sys
m=json.load(sys.stdin)['choices'][0]['message']
print('reasoning:', (m.get('reasoning') or '')[:200])
print('content  :', m.get('content'))"
```

This build emits reasoning **before** the answer, in a `reasoning` field rather
than `content`. Budget at least ~400 `max_tokens`: at 200 the reply is still
inside its reasoning, so `content` comes back empty on a perfectly healthy
server. Gibberish in either field means the PLE path has regressed (bf16 IPC
buffer or missing quant scales) — see the patch notes below.

## Layout

- `download.sh` — fetches the checkpoint into the Hugging Face cache
  (resumable; honours `HF_TOKEN` for gated repos). Uses the host's
  `huggingface_hub` if present, otherwise the container image.
- `start.sh` — launcher: derives the GPU budget from live memory under the
  `HOST_RESERVE_GIB` cap, builds the packed PLE table on first run,
  regenerates the patched vLLM files, archives the previous run's logs, starts
  the container and `files/memwatch.sh`.
- `stop.sh` — stops the watchdog, then the container (gracefully by default);
  reports leftover `/dev/shm` segments without deleting them.
- `files/patch_ple_layer.py`, `files/patch_modelopt_mxfp8.py`,
  `files/patch_ple_offload.py` — generators that rewrite the patched vLLM
  files from pristine `*.orig` / `orig/` copies on **every** launch. Those
  copies are not in the repo — `start.sh` extracts them from the image on
  first run. Edit the generators; edits to the generated files are overwritten.
- `files/build_ple_packed_table.py` — one-time packed PLE table builder
  (27 GB output under `~/.cache/vllm/ple_cache/`, memory-mapped at runtime).
- `files/sysctl-spark3.conf` — recommended kernel VM tunables, not applied by
  anything here; read its header first.

Benchmarks are not part of this repo. The published prefill and decode numbers
were measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash).

## What is patched and why

- **PLE layer** (`patch_ple_layer.py`): NVFP4/FP8 dispatch for the PLE table;
  offloaded rows carry codes *and* scales (90 B/head); the GPU-side placeholder
  learns its quant method from config because its constructor is skipped under
  offload; tolerates multi-call `load_weights`; slices the 2560-wide IPC buffer
  to the 1440 valid bytes.
- **ModelOpt** (`patch_modelopt_mxfp8.py`): BF16 fallback for MXFP8 shapes that
  FlashInfer rejects.
- **PLE offload** (`patch_ple_offload.py`): GB10 has no CUDA stream memory ops
  (`CAN_USE_STREAM_MEM_OPS=0`, measured), and vLLM's offload semaphore used them
  and deadlocked after graph capture. Replaced with a host-side handshake — the
  GPU worker posts a request, the CPU worker copies and writes a sequence number
  to shared memory, the GPU worker proceeds. It also attaches the memory-mapped
  packed table instead of loading 27 GB into RAM. The mmap is advised
  `MADV_RANDOM`: without it the kernel faults in a ~64 KiB window to serve each
  90-byte row lookup, and measurements here showed **24x** more disk read per
  decoded token (1,366 -> 57 KiB/token) plus ~2 GiB of page cache wasted on
  readahead that is never used.
- **FP8 KV cache** (`patch_qsa_fp8_kv.py`, via `KV_CACHE_DTYPE=fp8`): casts
  FP8 K/V tiles to BF16 for the tensor-core dots and applies the per-tensor
  scales once to the score and the normalised output, plumbs `k_scale`/
  `v_scale` into the kernels, and relaxes the four BF16-only guards and the
  inherited FlashAttention rejection. Avoiding an FP32 dequantisation tile lets
  FP8 keep the BF16 `block_n`. Raises the KV pool from ~800k to ~1.26M tokens
  at the shipped `KV_TARGET_GIB=20` (~1.38M at 22), which is what makes a 1M context
  arithmetically possible on one Spark. **On by default** and still a real
  quality trade — see the warning `start.sh` prints.
  The FP8-KV approach is credited to
  [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX)
  (Apache-2.0), reimplemented here against this image's own sources. That
  credit applies to this one patch; nothing else in this repository derives
  from that project.

## License

Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)

Licensed under the **GNU Affero General Public License v3.0 or later**
(AGPL-3.0-or-later). See `LICENSE`. Every source file carries an
`SPDX-License-Identifier: AGPL-3.0-or-later` header.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. It is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License
for more details.

Because this is AGPL and this repository exists to run a **network server**:
if you modify these scripts and offer the resulting service to users over a
network, section 13 requires you to offer those users the corresponding
source of your modified version.

### What the license does and does not cover

It covers the files in this repository — the launcher, the patch generators,
the packed-table builder and the watchdog. It does **not** relicense anything
they operate on, each of which carries its own terms:

- **vLLM** (Apache-2.0) — not redistributed here. `start.sh` extracts the
  pristine `*.orig` sources from the container image at runtime, and the patch
  generators emit modified copies onto your machine only. Those generated files
  keep vLLM's own Apache-2.0 headers and remain Apache-2.0 works.
- **The container image** `vllm/vllm-openai:qwen38-flash-next` and its
  dependencies — upstream terms apply.
- **The model checkpoint** `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` — weights are
  governed by the checkpoint's own license, not by this repository's.
