# GB10 opt-in kernel comparison: coding workloads

Measured 2026-09-07 on one DGX Spark/GB10. Baseline is Mia's `ef1af5f`
**with MTP3, reduced drafting, and BF16 recurrent state already enabled**;
these are incremental gains, not comparisons against non-speculative decode.

The submission branch is rebased onto `78b0675` (snapshot selection and the
optional gated checkpoint). The measurements below remain from `ef1af5f` with
the stock checkpoint; the newer launcher gets source checks/dry runs, not a
fresh performance claim. The gated checkpoint was not benchmarked here.

Two independent switches, both disabled by default:

- `GDN_PREFILL_BACKEND=flashinfer`: explicit SM121 FlashInfer prefill path;
  `auto` restores the existing resolver behavior.
- `MTP_DRAFT_HEAD_FP8=1`: experimental W8A16 reduced draft head; `0` leaves
  the original BF16 head in place. Target weights and sampling are unchanged.

## Method

Same image/checkpoint/vocabulary across four sequential fresh boots:

- Image digest `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`;
  vLLM `0.1.dev20073+g8e685d198`, PyTorch `2.13.0+cu130`, FlashInfer `0.6.17`.
- `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` revision
  `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
- TP1, native 262144 context, four sequences, 2048 prefill batch, 65536-ID
  draft vocabulary, BF16 recurrent state, FP8 attention KV, V2 runner,
  FULL_DECODE_ONLY graphs 4/8/12/16, compilation 0, GMU 0.737.
- **32-GiB host reserve**, not the sample's 26 GiB. Speech daemons remained
  resident but were not exercised; no MPS or compute cap. Local request timing
  telemetry was present in every profile; it is not bundled in these patches.

Four synthetic tasks: Python deep merge/diff, Python dependency planner,
C++17 LRU, and C++17 lexical path normalization. Each ran three times at
1/2/4 streams; all two-task pairings were covered. Six cold repository-repair
prompts (three each at 6941/27120 tokens) plus two append-only follow-ups
completed the suite. Six warmup requests per boot were excluded.

200 total requests, 176 measured implementations. Temperature 0, fixed seed,
thinking off, streaming usage, natural completion (1536-token safety ceiling).
Short/cold prompt hashes matched across profiles. No foreign LLM traffic or
preemption was detected. Every measured output ended naturally and parsed or
compiled. Code was execution-tested only in non-root, no-network, read-only
disposable containers with resource/time limits. Grader fixtures were frozen
before measurement and self-tested with five positive/four negative controls.

Decode below is the mean of the four per-task medians, using engine
`(output_tokens - 1) / generation_time`, not the prefill-inclusive legacy rate.
Prefill is median uncached tokens / scheduled-to-first-token time. All generated
outputs, including functional failures, remain in these timing figures.

## Comparison

| Measurement | Baseline | GDN only | FP8 head only | Both |
|---|---:|---:|---:|---:|
| Solo code decode, tok/s | 53.81 | 53.56 | 56.60 | 56.76 |
| 2-stream code decode, tok/s per request | 44.27 | 44.09 | 46.05 | 45.85 |
| 4-stream code decode, tok/s per request | 35.49 | 35.27 | 35.74 | 35.96 |
| Cold 6941-token code prefill, tok/s | 1959 | 2093 | 1980 | 2088 |
| Cold 27120-token code prefill, tok/s | 2027 | 2155 | 2044 | 2140 |
| 27120-token client TTFT, seconds | 13.43 | 12.65 | 13.33 | 12.73 |
| 4-stream batch end-to-end aggregate, tok/s | 115.15 | 114.68 | 109.68 | 115.32 |
| Whole implementations passing fixed tests | 42/44 | 41/44 | 39/44 | 40/44 |

Combined: **+5.5% solo decode, +3.6% at two streams, +1.3% at four;
and +5.5–6.6% cold prefill.** GDN alone delivers 6.3–6.8% faster prefill with
essentially unchanged decode. FP8 alone gives 5.2% solo code decode gain;
estimated solo decode-step time falls from 60.13 to 57.57 ms, with similar
acceptance (0.761→0.769). Combined step time is 58.20 ms, acceptance 0.763.

## What the numbers do not establish

- **Four-stream batch throughput is effectively flat.** Natural completions
  have different lengths, so batches drain toward lower concurrency. Do not
  multiply per-request decode by stream count to infer aggregate throughput.
- Faster tokens did not consistently shorten completed-task latency. Solo
  task-balanced elapsed time was 9.19 s baseline versus 9.56 s combined;
  longer generated implementations can spend the token-rate improvement.
- The main four-task correctness subset scored 35/36, 35/36, 34/36, 35/36.
  Both FP8 profiles generated one path normalizer that incorrectly reduced
  `../..` to `.`; baseline/GDN instead each had one incorrect planner.
  Repository repairs scored 7/8, 6/8, 5/8, 5/8. Those failures interpreted
  "touching integer intervals" as adjacency (`start <= end + 1`), while the
  frozen grader required shared endpoints (`start <= end`). We did not change
  the scorer. A repeated warm answer is correlated with its preceding answer.
- Even baseline's identical greedy solo requests produced different source
  texts. This small suite neither establishes kernel-caused quality degradation
  nor demonstrates quality equivalence. **FP8 remains experimental.**
- Restricting decode to passing outputs yields 53.74/44.27/35.49 tok/s baseline
  versus 56.76/45.85/35.92 combined. That filtering is survivor-biased; it is
  not a substitute for reporting all failures.
- One boot per profile and three repeats are directional evidence, not tight
  confidence intervals, SWE-bench, or a full coding-agent evaluation.

## Memory and compatibility

The FP8 copy/scales cost **0.1565 GiB** beyond retained BF16 buffers. Actual
profiled KV was baseline 697098 tokens/10.45 GiB, GDN 675130/10.12,
FP8 648769/9.72, combined 675130/10.11. Combined capacity was 3.2% lower
than this baseline, but boot/workspace variation is larger than the known
buffer alone. Host reserve was not reduced. Minimum combined MemAvailable
was 14.48 GiB; no watchdog stop, unexpected restart, or OOM kill occurred.
Recovered startup NVIDIA allocation warnings numbered 2/0/1/1; none occurred
during measured serving. This is not an endurance or maximum-capacity test.

All profiles reused 4992/~7K and 24960/~27K prefix tokens on append-only
follow-ups. Final combined Unicode/tool checks passed 3/3, and Chat/Responses
cache/timing checks, including final streaming metrics, passed 6/6.

For source guards, isolated numerical tests, per-feature reproduction, and
the earlier prose A/B, see [GDN](sm121-gdn-prefill.md) and
[FP8 draft head](sm121-mtp-fp8-head.md). Credit to
[Gabriel Olympie (@gabrielolympie)](https://github.com/gabrielolympie/sglang-flashnext-sm120/tree/67d2f9234fa45ae1339f0d53cd37cb695e9c6493)
for the original SM120 work; [attribution and licensing](../THIRD_PARTY_NOTICES.md)
describe the GB10/vLLM adaptation.
