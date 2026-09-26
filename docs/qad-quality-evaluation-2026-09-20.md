# Independent QAD quality evaluation on one GB10

This is an independent operator report for
[`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4),
served with the MiaAI-Lab single-Spark runtime on one ASUS Ascent GX10. The
combination was stable enough to complete long reasoning and instruction
following runs on a single GB10, which made the newer QAD checkpoint genuinely
practical for this host.

The tested Hugging Face revision was
`7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`. The runtime came from this
repository at `e74e7af934c19799eca5a2c0dc9f97bc6decd784`; the server reported
vLLM `0.1.dev20073+g8e685d198`.

## Why these benchmarks

GPQA Diamond and IFBench were selected because NVIDIA reports both for its
official
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
checkpoint. Tool Eval Bench was added because the local-inference-lab model
card publishes a Tool Eval result. This gives useful reference points for the
QAD checkpoint without inventing a new comparison suite.

| Source | GPQA Diamond | AA-LCR | IFBench | Tool Eval |
|---|---:|---:|---:|---:|
| NVIDIA FP8 model card | 92.0 | 71.9 | 80.5 | — |
| NVIDIA NVFP4 model card | 91.5 | 74.1 | 81.0 | — |
| local-inference-lab QAD model card | 89.9 | 79.4 | — | 91 |
| Independent QAD run on Mia runtime | **91.41** | — | **82.31** | **95** |

AA-LCR 79.4 is the QAD model author's published result, not an independent
measurement from this run.

## Serving configuration

| Property | Value |
|---|---|
| Host | One ASUS Ascent GX10, GB10, 128 GB unified memory |
| Model revision | `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd` |
| Runtime revision | `e74e7af934c19799eca5a2c0dc9f97bc6decd784` |
| vLLM | `0.1.dev20073+g8e685d198` |
| Context | Native 262,144; YaRN off |
| KV / recurrent state | FP8 KV; BF16 Mamba SSM cache |
| Speculative decoding | MTP3 |
| Scheduler | `MAX_NUM_SEQS=8`, `MAX_NUM_BATCHED_TOKENS=8192` |
| CUDA graphs | Full decode only; capture widths 4 through 32 |
| PLE | Mia memory-mapped CPU offload path |

This is a measured compatibility point, not a new shipped profile. In
particular, `MAX_NUM_SEQS=8` and the 8,192-token prefill chunk differ from the
repository defaults.

## Results and protocols

### GPQA Diamond

- Result: **181 / 198 correct = 91.41%** (standard error 2.00 percentage points).
- One epoch, concurrency 3.
- Generation: temperature 1.0, top-p 0.95, `reasoning_effort=xhigh`, maximum
  131,072 output tokens, streaming enabled.
- Usage: 62,570 input tokens, 3,125,886 output tokens, **3,188,456 total**;
  3,062,529 tokens were reported as reasoning tokens.
- Wall time: 2026-09-19 20:08 to 2026-09-20 08:28, UTC+05:00.

### IFBench

- Result: **242 / 294 passed = 82.31%** (standard error 2.23 percentage points).
- The pinned 294-prompt dataset was scored with the official prompt-level
  loose constraints; one epoch, concurrency 4.
- Generation: temperature 1.0, top-p 0.95, `reasoning_effort=xhigh`, maximum
  131,072 output tokens, streaming enabled, six-hour per-response timeout.
- Usage: 39,190 input tokens, 2,444,061 output tokens, **2,483,251 total**;
  2,340,899 tokens were reported as reasoning tokens.
- Wall time: 2026-09-20 09:01 to 17:25, UTC+05:00.

### Tool Eval Bench

- Tool: `tool-eval-bench` `v2.6.1.dev72+gd84fce442`.
- Result: **165 / 174 points = 95 / 100** across all 88 hard-mode scenarios.
- Completion rate: 98.9%. TC-45 was excluded because the endpoint did not
  enforce `tool_choice=required`; it was classified as an infrastructure
  limitation rather than a model failure.
- Sequential execution, temperature 0, seed 42, `reasoning_effort=xhigh`,
  maximum 12 turns and 16,384 output tokens.

## Interpretation and limitations

These rows are useful reference points, not strict apples-to-apples claims:

- NVIDIA publishes generation parameters but not a fully identical evaluation
  harness for every score.
- This IFBench run used one response per prompt. The commonly cited Artificial
  Analysis protocol uses five repeats, so the one-pass result has higher
  sampling variance.
- The QAD model-card scores are author-reported. Only the bold row was measured
  in this independent run.
- Tool Eval versions and endpoint capabilities can change; the exact tool
  version and excluded infrastructure case are therefore part of the result.

As an operator observation, an older checkpoint sometimes degraded into
non-Latin/gibberish text or unstable responses. That behavior was not observed
across the **5,671,707 total GPQA and IFBench tokens** above. This is encouraging
field evidence, not a controlled old-versus-new A/B experiment.

## Reproducing the opt-in

The repository default is intentionally unchanged. Download and select the QAD
checkpoint explicitly:

```bash
./download.sh local-inference-lab/Qwen3.8-Flash-Next-NVFP4
TP1_MODEL_ID=local-inference-lab/Qwen3.8-Flash-Next-NVFP4 ./start.sh
```

The request to refresh the Mia mirror or document this checkpoint as a
maintainer-supported option is tracked in
[#60](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/60).
