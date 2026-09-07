# Opt-in FlashInfer GDN prefill on GB10

This is a small backend-selection patch, not an SGLang port. FlashInfer 0.6.17
already ships a native SM12x GDN kernel and selects `sm_121a` on GB10. The
installed vLLM resolver excludes SM121. The overlay permits that path only for
an explicit `GDN_PREFILL_BACKEND=flashinfer`, CUDA 13+, and 128-wide key/value heads.
`auto` preserves the existing Triton/FLA choice. Decode selection is unchanged.

The idea was investigated after reviewing
[Gabriel Olympie's (@gabrielolympie) SM120 patches](https://github.com/gabrielolympie/sglang-flashnext-sm120/tree/67d2f9234fa45ae1339f0d53cd37cb695e9c6493),
especially their FP32 prefill-state handling. vLLM's existing FlashInfer adapter
already converts initial state/gates to FP32, and FlashInfer converts sequence
offsets to int64. Neither of those adapters needs replacing here.

## Reproduction and scope

Measured 2026-09-06 on one DGX Spark/GB10, recipe base `ef1af5f`, with local
request telemetry and a 32 GiB host reserve. Same image, checkpoint, context,
memory reserve, and scheduling limits on both sides:

- Image `vllm/vllm-openai:qwen38-flash-next`, digest
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
- vLLM `0.1.dev20073+g8e685d198`, PyTorch `2.13.0+cu130`, FlashInfer `0.6.17`.
- `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` revision
  `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
- TP1, native 262144 context, 4 sequences, prefill batch 2048, FP8 attention KV,
  BF16 recurrent state, V2 runner, MTP3, same 65536-entry draft vocabulary.
- FULL_DECODE_ONLY graphs 4/8/12/16, compilation mode 0, GMU 0.737.
- Speech containers remained running but were not benchmarked; no MPS/caps.

Serving results are medians of three repetitions, after two excluded 32K
warmups. Decode requests generate 600 tokens each. Cold prompts have unique
early salts and reported zero cached tokens. Every request completed and
counter deltas matched the benchmark's request counts (no other LLM traffic).

| Measurement | Triton/FLA | FlashInfer |
|---|---:|---:|
| Cold ~8K prefill, tok/s | 2096.6 | 2225.9 (+6.2%) |
| Cold ~32K prefill, tok/s | 2084.1 | 2226.3 (+6.8%) |
| ~8K client TTFT | 3.867 s | 3.639 s |
| ~32K client TTFT | 15.454 s | 14.457 s |
| Solo decode, tok/s | 34.95 | 34.90 |
| 2-stream decode, tok/s per stream | 29.91 | 31.33 |
| 4-stream decode, tok/s per stream | 23.62 | 23.51 |

Do **not** interpret the 2-stream throughput increase as a decode-kernel win:
draft acceptance varied. Estimated decode-step medians were 60.90→61.03 ms
(solo), 72.75→72.48 ms (2 streams), and 91.56→92.01 ms (4 streams).
That is effectively unchanged at this sample size. Different floating-point
prefill implementations can produce different subsequent token trajectories.

The first measured ~8K FlashInfer request took 4.33 s, whereas the subsequent
ones took 3.60–3.64 s. The reported median includes it. Startup/JIT and first-use
effects are not a steady-state speed guarantee.

### Coding follow-up (2026-09-07)

Same image and inference controls, with four synthetic Python/C++ tasks at
1/2/4 streams (three repetitions each), plus cold repository-repair prompts.
Thinking off, temperature 0, natural completion; 50 requests per profile,
including six excluded warmups. Short-task decode is the mean of four per-task
medians, not aggregate throughput.

| Measurement | Triton/FLA | FlashInfer |
|---|---:|---:|
| Cold 6941-token code prefill, tok/s | 1959 | 2093 (+6.8%) |
| Cold 27120-token code prefill, tok/s | 2027 | 2155 (+6.3%) |
| Solo code decode, tok/s | 53.81 | 53.56 |
| 2-stream code decode, tok/s per request | 44.27 | 44.09 |
| 4-stream code decode, tok/s per request | 35.49 | 35.27 |
| Whole implementations passing fixed tests | 42/44 | 41/44 |

Both profiles passed 35/36 on the four short tasks, each with one incorrect
planner. Repository repairs passed 7/8 versus 6/8; those failures interpreted
"touching integer intervals" differently from the frozen grader. No prompt or
test was changed after seeing results. Even the baseline's repeated greedy
requests generated different source texts, so this is not a proof of bitwise
or model-quality equivalence. All outputs compiled/parsed and ended naturally;
no preemptions or unrelated LLM traffic were detected. KV profiling this time
was 697098→675130 tokens (10.45→10.12 GiB), illustrating boot variation.

## Correctness and costs

- Five CPU tests cover opt-in/default behavior, unsupported GPU/geometry/CUDA,
  unavailable kernels, source drift, and double application.
- Thirteen isolated GPU checks compare outputs **and final state** against
  FLA with actual 16-key/48-value-head geometry, ragged batches, short tails,
  ordinary/slow decay, and resumed 1664-token BF16-state chunks. All finite;
  maximum relative RMS differences about 0.57% (output) and 0.48% (state).
  These are numerical comparisons, not bitwise equality.
- At 2048 tokens, the isolated adapter measured 2.214→0.551 ms (~4×). Only a
  fraction of end-to-end prefill uses this kernel; the serving gain is ~7%.
- Six 32K/128K retrieval and append-only continuation requests passed, including
  reuse of 29952 and 124800 cached tokens. No broken recurrence/cache behavior
  was observed in this bounded test.
- The same four answer-level mistakes occurred in the 18-question greedy/
  temperature-0.7 synthetic suite: **14/18 correct both before and after**.
  Exact-format scores were 10/18 and 11/18. This is not a general accuracy or
  long-reasoning evaluation, and is not evidence of improved intelligence.
- Unicode and auto/forced tool checks passed; all six Chat/Responses timing/
  cache canaries passed, including final streaming usage metrics.
- A separately retained prose sample was coherent. Generated Python interval-
  merging code passed 100 edge/randomized cases without mutating its inputs,
  executed only in an unprivileged, no-network, read-only disposable container.
- KV boot capacity was **726387→705884 tokens** (10.88→10.57 GiB), about 2.8%
  lower under the same host safety budget. Treat this as an observed profiling/
  workspace cost, not a universal constant. Do not lower the reserve to hide it.
- No unexpected container restarts or engine failures during these tests.
  The kernel journal did record five recovered `NV_ERR_NO_MEMORY` allocation
  warnings during loading, before readiness; none during serving. The prior
  baseline boot also had recovered driver allocation warnings. Existing
  Transformers video-processor documentation warnings remain. This test does
  not establish that the broader deployment's startup memory behavior is fixed.

The patch validates the entire input source hash before writing the overlay.
An image/source update fails closed and needs review. No checkpoint or disk KV
cache is modified. Set `GDN_PREFILL_BACKEND=auto` and restart to return to stock.

## Tests

CPU resolver tests run in the image, without GPU or network:

```sh
docker run --rm --network none --memory 512m \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD/files:/tests:ro" -w /tests \
  --entrypoint python3 vllm/vllm-openai:qwen38-flash-next test_gdn_sm121.py
```

`bench/gdn_sm121_preflight.py` runs the isolated numerical/state comparisons.
Run it in the serving image with GPU access **while the LLM is stopped**, in a
bounded disposable container. It does not load the checkpoint or change the
serving process. First compilation took about 37 seconds in this test.
