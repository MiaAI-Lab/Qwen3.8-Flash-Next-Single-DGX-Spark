# Qwen3.8-Flash-Next on ONE DGX Spark (TP=1)

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
./start.sh                 # ~10-12 min to /health; serves on :8888
./stop.sh                  # container + watchdog, graceful
```

`./start.sh --no-launch` prints the derived memory budget and the docker
command without running anything. `./stop.sh` sends SIGTERM and waits up to
`STOP_TIMEOUT` (default 30 s) so vLLM can unlink its POSIX shared memory —
the container runs with `--ipc host`, so segments it leaves behind leak onto
the host's `/dev/shm` until reboot. `./stop.sh --force` skips the wait.

## Measured profile

With the shipped `.env.sample` (262,144 native context, MTP 3,
`KV_TARGET_GIB=20`, `MAX_NUM_SEQS=4`), measured on this host 2026-09-04:

| | |
|---|---|
| Available KV cache | 21.28 GiB = **736,837 tokens** (2.81x a full 262k request) |
| Host MemAvailable under load | **~10 GiB**, with ~10 GiB page cache |
| KV dtype / size | bf16, ~28.8 KiB per token |

Speed numbers below are from the identical recipe on the sibling Spark and
have **not** yet been re-measured here: decode ~25 tok/s prose / ~30 tok/s code
(MTP 3, temp 0), prefill ~1,750 tok/s at 200k, needle at 90% depth of a 201k
prompt retrieved correctly.

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

The safety-relevant knobs are `KV_TARGET_GIB` (how much KV to target; the main
consumer of host memory) and `HOST_SLACK_GIB` (container cgroup cap = GPU
budget + this). `KV_TARGET_GIB=16` gives ~590k tokens and more host margin.

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

512k fits the shipped profile with **no other change**: it needs 14.4 GiB of KV
against the ~20 GiB the default `KV_TARGET_GIB=20` already provides, and the GPU
budget and cgroup cap are unchanged from 262k. Measured on this host at
`YARN=1` (2026-09-04):

| | |
|---|---|
| Available KV cache | 19.14 GiB = **701,554 tokens** (1.34x a full 524,288 request) |
| Host MemAvailable idle | ~11.3 GiB |
| Output | coherent; MTP 3 and YaRN run together without incident |

400k prefill stress test (`bench/longctx.py --target 400000`, salted to defeat
prefix caching, needles at 5% / 50% / 95% depth):

| | |
|---|---|
| Prompt | 400,062 tokens |
| TTFT (prefill) | 260.3 s = **1,537 tok/s** |
| Needle retrieval | **3/3 PASS**, including 95% depth |
| Host MemAvailable low-water | **10.97 GiB** (watchdog floor is 6 GiB) |
| Peak container RSS | 18.7 GiB of the 103 GiB cap |

Decode measured 40 tok/s on that run, but the answer is three codes copied out
of the context — MTP's best case. Do not quote it as typical decode speed; see
the note at the top of `bench/decodebench.py`.

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

## Safety rules

Each of these cost a hard host hang during bring-up.

- **Keep host MemAvailable at or above ~10 GiB under load.** Exhausting the
  unified pool hangs the kernel with no OOM kill and no logs. `KV_TARGET_GIB`
  is the knob that eats it. The page cache is not spare memory — it is what
  keeps PLE lookups off NVMe.
- **`comfy-h3.service` must stay disabled.** It polls `127.0.0.1:8888` and
  launches ComfyUI (a GPU co-tenant) as soon as anything answers there.
  `start.sh` refuses port 8888 while that service is active.
- **Never set `PLE_OFFLOAD=false` at TP=1** — 99 GB through UVM hangs the host.
- **FP8 KV cache is unsupported** by this model's QSA attention backend
  (`supported_kv_cache_dtypes = ["auto","bfloat16"]`).
- **Do not raise `YARN_CEILING_MODEL_LEN` past 524288.** A 1M context needs
  ~28.8 GiB of KV, which drives the container cap to 112 GiB against a
  105 GiB hard ceiling. `start.sh` refuses it twice over; don't work around it.
- `docker --memory` does not bound GPU allocations on GB10, only host-side
  memory. vLLM's `--gpu-memory-utilization` is what bounds the GPU.

`files/memwatch.sh` runs alongside the container and kills it if MemAvailable
drops below `MEMWATCH_MIN_GIB` (default 6). Its log is under `logs/`.

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

- `start.sh` — launcher: derives the GPU budget from live memory,
  builds the packed PLE table on first run, regenerates the patched vLLM
  files, starts the container and `files/memwatch.sh`.
- `stop.sh` — stops the watchdog, then the container (gracefully by default);
  reports leftover `/dev/shm` segments without deleting them.
- `files/patch_ple_layer.py`, `files/patch_modelopt_mxfp8.py`,
  `files/patch_ple_offload.py` — generators that rewrite the patched vLLM
  files from pristine `*.orig` / `orig/` copies on **every** launch. Those
  copies are not in the repo — `start.sh` extracts them from the image on
  first run. Edit the generators; edits to the generated files are overwritten.
- `files/build_ple_packed_table.py` — one-time packed PLE table builder
  (27 GB output under `~/.cache/vllm/ple_cache/`, memory-mapped at runtime).
- `bench/decodebench.py`, `bench/longctx.py` — decode and long-context/needle
  benchmarks. Both target `http://localhost:8888` hardcoded in `BASE` at the
  top of the file; edit it to point elsewhere.

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
- **FP8 KV cache** (`patch_qsa_fp8_kv.py`, opt-in via `KV_CACHE_DTYPE=fp8`):
  dequantises each K/V tile inside the QSA Triton kernels with vLLM's
  `_cast_kv_tile`, plumbs `k_scale`/`v_scale` into the kernels, relaxes the
  four BF16-only guards and the inherited FlashAttention rejection, and halves
  `block_n` under quantisation. Raises the KV pool from ~800k to ~1.38M tokens,
  which is what makes a 1M context arithmetically possible on one Spark.
  **Off by default** and a real trade — see the warning `start.sh` prints.

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
the packed-table builder, the watchdog and the benchmarks. It does **not**
relicense anything they operate on, each of which carries its own terms:

- **vLLM** (Apache-2.0) — not redistributed here. `start.sh` extracts the
  pristine `*.orig` sources from the container image at runtime, and the patch
  generators emit modified copies onto your machine only. Those generated files
  keep vLLM's own Apache-2.0 headers and remain Apache-2.0 works.
- **The container image** `vllm/vllm-openai:qwen38-flash-next` and its
  dependencies — upstream terms apply.
- **The model checkpoint** `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` — weights are
  governed by the checkpoint's own license, not by this repository's.
