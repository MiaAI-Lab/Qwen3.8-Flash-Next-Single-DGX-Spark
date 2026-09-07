# Opt-in FP8 reduced MTP draft head on GB10

**Experimental; disabled by default.** The
[coding follow-up](sm121-code-benchmarks.md) measured about 5% faster solo
code token production, but no four-stream batch-throughput gain or demonstrated
quality equivalence. Read its execution-test failures before enabling this.

`MTP_DRAFT_HEAD_FP8=1` adds a row-scaled E4M3 copy of the existing reduced
BF16 draft head. The Triton kernel reads FP8 weights, dequantizes tiles to
BF16, accumulates in FP32, applies FP32 row scales, and returns BF16 logits.
This is W8A16 weight-only drafting, not FP8 activation quantization.

It changes only MTP `get_top_tokens`; the target model's head, target weights,
full-vocabulary verification, tokenizer mapping, forward pass, and rejection
sampling are untouched. Original BF16 draft buffers are retained. This does
not promise bit-identical generated strings: numerical batching and speculative
paths can change trajectories even when the target weights are identical.

The kernel is adapted from Apache-2.0 kernels in
[Gabriel Olympie's (@gabrielolympie) patches 0004/0005](https://github.com/gabrielolympie/sglang-flashnext-sm120/tree/67d2f9234fa45ae1339f0d53cd37cb695e9c6493).
Attribution and the retained kernel license are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
The GB10 tactic (N tile 32, K tile 256, 3 stages, no split-K) was selected using
serialized CUDA graph timing with cold weight rings larger than L2. None of
the SGLang runtime, target-weight quantization, or relaxed acceptance is ported.

## Enable and disable

Keep the same MTP depth and draft vocabulary when comparing:

```sh
# Stop the running LLM safely first; do not start a second copy.
MTP_DRAFT_HEAD_FP8=1 ./start.sh
```

Requires a valid `MTP_DRAFT_VOCAB`, MTP enabled, TP1, SM121, and a BF16 Qwen
head with hidden width 2560. Unsupported geometry fails clearly. Set the knob
to 0 and restart to remove the override entirely. The default is 0.

Mia's reduced-vocabulary patch runs first. A separate source-exact patch then
checks its hash before generating the optional MTP overlay. On an incompatible
image or recipe update it fails closed. No checkpoint file is modified, no
FP8 weight copy is written to disk, and this is not a durable KV cache.

FP8 buffers are module-owned, nonpersistent, and prepared after loading but
before graph capture. There is no global data-pointer cache, eviction policy,
or capture-time quantization. Large batches (>32 rows) use the original BF16
head. Small-batch fallback is also retained for unsupported activation layouts.

## Isolated and serving results

Measured 2026-09-06 on one GB10; recipe base `ef1af5f` plus local telemetry
and host safety overlays. The FlashInfer GDN candidate was **disabled** for
this measurement. Both sides used:

- `vllm/vllm-openai:qwen38-flash-next`, digest
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
- vLLM `0.1.dev20073+g8e685d198`, PyTorch `2.13.0+cu130`.
- `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` revision
  `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
- Native 262144 context, TP1, 4 sequences, 2048 prefill chunk, FP8 attention
  KV, BF16 recurrent state, MTP3, identical 65536-token draft vocabulary.
- FULL_DECODE_ONLY graphs 4/8/12/16, compilation 0, V2 runner, GMU 0.737,
  32 GiB host reserve. No MPS or compute caps. Speech daemons were unchanged;
  no speech benchmark workload was generated.

Serving numbers are medians of three repetitions, 600 output tokens per
request, after two excluded 32K warmups. Request counters matched the test
traffic throughout. These are workload-specific measurements, not guarantees.

| Measurement | BF16 reduced head | FP8 reduced head |
|---|---:|---:|
| Solo prose decode, tok/s | 34.95 | 36.26 (+3.8%) |
| 2 streams, tok/s per stream | 29.91 | 30.73 (+2.8%) |
| 4 streams, tok/s per stream | 23.62 | 24.12 (+2.1%) |
| Estimated solo decode step | 60.90 ms | 58.37 ms |
| Estimated 2-stream decode step | 72.75 ms | 70.24 ms |
| Estimated 4-stream decode step | 91.56 ms | 89.49 ms |

Solo acceptance fraction was 0.385→0.373; some of the kernel gain was spent
on rejected proposals. The 2-/4-stream fractions were 0.391→0.395 and
0.388→0.387. The reduction in step cost is more consistent than raw tok/s,
which also follows the particular generated text and acceptance.

Using the **actual checkpoint head** with synthetic hidden activations:

| M | BF16 kernel | FP8 kernel |
|---|---:|---:|
| 1 | 1.909 ms | 0.699 ms (2.73×) |
| 4 | 1.426 ms | 0.706 ms (2.02×) |
| 16 | 1.454 ms | 0.716 ms (2.03×) |

The whole model is not 2–3× faster; the head is only part of each step.
Cold ~8K/~32K prefill measured 2125/2119 tok/s versus 2097/2084 before, a
small difference that should not be advertised as a prefill-kernel improvement.

## Correctness and memory

- Nine actual-head batch sizes (1,2,3,4,8,12,16,32,33) passed numerical
  checks. Kernel vs FP32 reference on the quantized weights had relative RMS
  error below 0.007%. FP8 approximation vs original BF16 was ~2.6–2.7% RMS.
  Batch 33 took an exact BF16 fallback. Synthetic top-1 agreement is not a
  substitute for actual serving acceptance or a model-quality evaluation.
- Zero rows stayed finite; nonfinite weights were rejected. Original BF16
  weights were bitwise unchanged. Extra buffers were excluded from state_dict.
- Four CPU tests verify source drift, scope, opt-in, and nonpersistent storage.
- 32K/128K retrieval plus append-only cache continuation passed all six cases.
  Unicode and both automatic/forced tool calls passed.
- All six Chat/Responses cache/timing canaries passed, including streaming.
- Natural prose was coherent. A generated interval-merging Python function
  passed 100 edge/random cases and input-nonmutation checks in a separate,
  network-disabled, read-only, resource-limited container.
- Synthetic answer-level score was 15/18 versus 14/18 in the baseline, with
  thinking off at temperatures 0 and 0.7. Exact-format score was 11/18 versus
  10/18. These small scores do **not** demonstrate increased intelligence or
  establish absence of regressions on arbitrary reasoning/multimodal tasks.
- The 65536-row FP8 copy and scales cost **0.1565 GiB** beyond the retained
  BF16 buffers. Known memory cost scales with draft-vocabulary size.
- Boot KV was 688311 tokens / 10.31 GiB versus baseline 726387 / 10.88 GiB.
  That observed pool change is larger than the buffers alone; profiling and
  workspace allocation vary across boots. Do not attribute all of it to FP8
  storage or compensate by reducing the host reserve.
- No unexpected engine restart, serving-time driver allocation failure, or
  OOM kill occurred during the measured serving workload. This does not fix
  the deployment's preexisting startup allocation-warning behavior.
- The recipe-style `VLLM_MTP_DRAFT_HEAD_FP8` environment flag produces an
  unknown-vLLM-variable warning, like the existing draft-vocabulary flag. The
  model helper reads it explicitly; its startup log confirms activation.

`bench/mtp_fp8_head_preflight.py` is a self-contained actual-head check. Run it
in a bounded GPU container while the LLM is stopped, with
`VLLM_MTP_DRAFT_HEAD_FP8=1`, the helper importable from `files/`, and explicit
`--snapshot` / `--draft-vocab` paths. Mount model files read-only and disable
network access. It does not start an API server or alter weights on disk.
