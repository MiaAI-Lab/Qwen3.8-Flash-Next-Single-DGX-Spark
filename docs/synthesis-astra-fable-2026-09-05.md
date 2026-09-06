# Cross-check of `astra-high.md` and `fable51-max.md`: what is done, what overlaps, what is still true

Written 2026-09-05 against the container running at the time (`vllm-fn-tp1`,
image `sha256:d464f3b4…` built 2026-08-26, vLLM `0.1.dev20073+g8e685d198`,
HEAD `203834c`). Every "verified" below was checked in the container source or
the engine log during this pass, not taken from either report. No config or
code was changed and no benchmark was run.

## 1. Facts that settle the disagreements between the two reports

| Claim | fable51-max | astra-high | Verified now |
|---|---|---|---|
| MoE kernel for the 48 target layers | Marlin ("GPU does not have native FP4") | `FLASHINFER_CUTLASS` for the target, Marlin only for the W4A16 MTP experts | **astra is right.** Log: `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend` at 11:31 for the target, `Using 'MARLIN'` at 11:39 when the W4A16 draft loads. Marlin also serves the NVFP4 *linear* layers (vision). |
| Model runner | V1 ("escape hatch `VLLM_USE_V2_MODEL_RUNNER=1` untested") | V2 by default for this architecture | **V2 is active** (`gpu_worker.py:500 Using V2 Model Runner`), with no env var set. |
| Why dynamic K went PIECEWISE | vLLM overrides the graph mode whenever the schedule is set | The override skips V2, so the historical run does not prove anything about V2 | **Both incomplete.** The archived dynamic-K log (`logs/archive/…T090906-container.log`) shows *both* `Using V2 Model Runner` at 05:50 and the PIECEWISE override at 05:58. The override fires from `VllmConfig.__post_init__` on the config copy that `with_model_config()` makes for the **draft** model (`config/vllm.py:851`); its architecture is `Qwen3_8FlashNextMTP`, not in the V2 default set, so `use_v2_model_runner` is False on that copy, and the copy shares the target's `compilation_config` object, which gets mutated. Pinning `VLLM_USE_V2_MODEL_RUNNER=1` short-circuits the property to True on every copy. The V2 graph manager already captures a FULL decode graph per K in the schedule (`v1/worker/gpu/cudagraph_utils.py:196-217`). |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | "zero risk, 0–10%" | Does not reach the routed MoE calls | **astra is right.** `experts/marlin_moe.py:159,226` hard-code `use_atomic_add=False, use_fp32_reduce=True`. The env var only affects ordinary Marlin linears, i.e. vision layers. And the target MoE is not Marlin anyway. |
| FP8 QSA "halves `block_n`" (fable Q) | premise for the FP8-MMA proposal | FP8 scale hoist already done | **Stale.** `files/qsa_ops_patched.py:935-939` has `block_n = 64` on every profile; the halving was removed in `69f7b4c`. Split-count tuning ("Tuned on GB300", lines 698 and 928) is still open. |
| SM12x native MoE (`flashinfer_b12x`) | not mentioned as available | exists, excluded from auto, accepts W4A16 | **Verified.** `oracle/nvfp4.py:177-179` excludes it from auto with a stale-looking SM121 MMA-guard comment; `flashinfer_b12x_moe.py:180-185` accepts `activation_key=None` (W4A16). Opt-in is `--kernel-config '{"moe_backend":"flashinfer_b12x"}'` (`engine/arg_utils.py:1645`). |
| Skinny-GEMM plans gated to SM103 | yes | yes | `low_latency_gemm.py:157 if dtype != bfloat16 or not _is_sm103(): return`. |
| Fused multi-step draft decode | unsupported for QSA state backend | same; contract is `supports_draft_decode_metadata_update` / `update_draft_decode_metadata` | Log line present; `v1/attention/backend.py:693,818` is the contract; `flash_attn.py`, `triton_attn.py`, `mla/sparse_swa.py` are reference implementations. |
| MTP sparse-index reuse (`index_share_for_mtp_iteration`) | — | hooks exist; "do not assume the model supports them" | **The model does support them.** `nvidia/mtp.py:305,311` implement `set_skip_topk` / `compact_topk_indices`; the V2 MTP speculator activates when the draft `hf_config` carries the flag (`spec_decode/mtp/speculator.py:27-33`, populated from the text config at `config/speculative.py:538-548`). The checkpoint's `config.json` does **not** set it, so it is off. It can be turned on with an `hf-overrides` entry; no code change. |
| bf16 SSM state supported by the fused GDN kernel | "if the backend supports it" | "if the backend supports it" | `FUSED_GDN_STATE_DTYPES = (float32, bfloat16)` (`qwen_gdn_linear_attn.py:90`); the checkpoint sets `mamba_ssm_dtype = float32`; the CLI knob is `--mamba-ssm-cache-dtype`. |
| Adaptive draft length (fable U) | port from dspark | — | `enable_adaptive_verification` is rejected for any method but `dspark` (`config/speculative.py:1189`). |
| n-gram drafter hybrid (astra 5.3) | — | research | `ngram`/`ngram_gpu` are unsupported by the V2 runner (`config/vllm.py:2385`), so this would also force V1. |
| Swap pressure on the engine (fable H) | 1.3 GiB swapped, engine pages at risk | not justified without measurement | 3.2 GiB of swap is in use on the host, but the vLLM processes readable from the host show `VmSwap: 0 kB`. No evidence the engine is being swapped. `vm.swappiness` is still 60, `min_free_kbytes` 45155. |
| Power cap / clocks | "spends most of its time power-capped" | not a proven bottleneck | Idle sample: SM 2190 of 3003 MHz, `sw_power_cap: Not Active`, 9.7 W. Says nothing about load; only a trace under load would. |

## 2. Already done and measured on this host (do not re-spend a launch)

| Item | Reports | Result (CHANGELOG 2026-09-05) |
|---|---|---|
| Reduced-vocabulary drafting, 65k rows | fable #1 / I; astra #1 | Active now (`MTP draft vocab: 65536 … 2.61 GiB saved per step`). Step −16.9% at 1 stream, −6.1% at 8. MGSM accuracy unchanged. |
| CUDA graph for every decode width | fable #2 / A; astra 1.2 | `CUDAGRAPH_CAPTURE_SIZES=auto`, 4 graphs captured. ~4–5 ms on the 5-stream step, nothing at 1/2/4. |
| PLE page prefetch (`posix_fadvise`) | fable #3 / K; astra 1.2 | 13x on the gather in isolation; −3.2% decode step; the real payoff was **prefill +7–10%**. |
| `MAX_NUM_SEQS` 5 → 8 | fable #8 / B; astra 11.1 | 84.7 → 114 tok/s at 8 streams, short context. Shipped at 4 because of long-context KV and the host reserve. |
| Dynamic K (as specified) | fable #4 / C | Regression (105 vs 114 tok/s) plus a memory event, **but the cause was PIECEWISE graphs, not the byte model** (see §1). |
| FP8 KV scale hoist | fable Q premise | Done in `69f7b4c`; `block_n` back to 64. |
| `MAX_NUM_BATCHED_TOKENS` 8192 | fable G (upper end) | +11% prefill at 32k, opt-in. The 512/1024 end for inter-token latency is untested. |
| Host-side budget cap, watchdog floors | fable #8 (safety half) | `HOST_RESERVE_GIB=26`; 0 `NV_ERR_NO_MEMORY` across the 2.5 h harness soak. |

Current shipped step times to plan against (reduced head, FULL graphs, `MAX_NUM_SEQS` 4 in `.env`): 63.9 ms at 1 stream, 101.6 at 4, 146.2 at 8 (from the 8-seq run). Byte-model floors: 1.37x at 1 stream, ~1.04x at 5+, i.e. at concurrency only bytes removed convert to time.

## 3. Open items both reports agree on, ranked

Percentages are byte-model planning numbers against today's step, not
measurements; the byte model predicted the reduced-head result within a point,
so they are worth about that much.

| # | Item | fable | astra | What is true today | Expected | Cost |
|---|---|---|---|---|---|---|
| 1 | **Static K sweep 0/1/2/3 with FULL graphs, at S=1/2/4/8** | C, §3.3 crossover | #2, §5.1 | Never run as a static sweep. The one data point (K=1 at S=8: step 160.8 → 127.4 ms, 2.29 → 1.68 tok/step) was taken under PIECEWISE. Break-even for K=1 at S=8 is a step under ~118 ms with FULL graphs; the graph penalty at S=1 was 19 ms, so this is genuinely open. K=2 never measured. | Unknown; could be 0 or +10% at 8 streams. Nothing at 1 stream (K=3 wins there). | 4 config launches, ~12 min each. Record resolved KV, since K=0 frees 1.5 GiB. |
| 2 | **Dynamic K on V2 with the runner pinned** | #4/C (as the mechanism), | 5.2 (pin the runner explicitly) | Root cause in §1: add `-e VLLM_USE_V2_MODEL_RUNNER=1` via `EXTRA_DOCKER_ARGS`, keep `MTP_K_SCHEDULE`, check the log has **no** "Overriding cudagraph_mode" line and `Capturing decode CUDA graphs (FULL)` counts one graph per K. The 99.4 GiB driver figure in the failed run was PIECEWISE graph memory; it should not recur with FULL graphs but the watchdog must stay on. | Whatever #1 says the per-S optimum is, delivered automatically across batch sizes. Do #1 first so the schedule is measured, not guessed. | 1 launch after #1. |
| 3 | **FP8 target `lm_head` and hyper-connection weights** | J, #7 | §12 (with the caveat that only one head read is left) | The only remaining byte item below 8 streams. Saving is 0.64 GB (head) + 0.63 GB (HC) per step. ModelOpt FP8 already accepts `ParallelLMHead` (`modelopt.py:185,202`); HC needs a one-line generator patch to stop passing `quant_config=None` (`hyperconnection.py:102,113,122`). | ~4% + ~4% at 1 stream, ~1.7% + ~1.7% at 8. | Medium: checkpoint re-export of two tensor groups. NVMe has 46 GB free against a 98.5 GiB checkpoint, so rewrite only the affected shards in place and update the index. Needs a logprob-drift and task eval. |
| 4 | **Fused multi-step draft decode for the QSA state backend** | N | #4, §7 | Only helps single stream (S=1 sits 1.37x above the floor; S≥5 has nothing to remove). The three reference backends in §1 show the contract. The metadata rebuild's cost has never been measured; astra's rule stands: if the gaps sum to 1 ms, this cannot save 10. | 0–10% at 1 stream, 0 at concurrency. | Medium-high; profile first (item 9). |
| 5 | **Native SM12x MoE (`flashinfer_b12x`)** | R (wrong premise, right target) | #3, §6.1 | Opt-in is one CLI flag. It would replace both the target CUTLASS path and the draft Marlin path. A 1-stream K=3 verify batch is 40 routed rows, inside its micro-kernel regime. May fail to JIT on SM121; may lose to CUTLASS at low rows. | Unknown. Decode gain likely small; larger-T (prefill, 8-stream verify) is where a better grouped GEMM shows. | 1 launch to see if it loads and what it does to step time; then a kernel microbench if promising. |
| 6 | **bf16 SSM state (`--mamba-ssm-cache-dtype bfloat16`)** | F | §9 (as the simple alternative) | Kernel accepts it. Halves 0.23 GB × S of state traffic per step (more if the speculative path stores per-candidate states, which astra says to check). Also halves the mamba page, which lets vLLM pick a 1,600-token attention block and doubles prefix-cache granularity for multi-turn. | ~2% at 8 streams, ~0 at 1. Prefix-cache effect matters more for the agent harness. | 1 launch plus a long-context quality check (needles, multi-turn). |
| 7 | **Prefill chunk 512/1024 for inter-token latency under mixed traffic** | G | 11.2 | Untested at the low end. A 2,048 chunk is ~1 s of GPU time at 2,200 tok/s, paid by every co-scheduled decode. This is the harness's actual traffic (72k-token prompts, up to 3 concurrent). | No change to decode-only numbers; p95 ITL under prefill could fall 2–4x at 10–20% prefill cost. | 2 launches; measure p95/p99 ITL and TTFT together. |
| 8 | **QSA split-count table for SM121** | Q (split half only) | #8, §10 | 48 SMs against a table tuned for 160. At 1 stream K=3 the 8 base programs are split 32 ways plus a merge. Attention's share of the step is unmeasured. | A few % if attention is visible in the profile; 0 otherwise. | Medium; Triton, needs the profile. |
| 9 | **A profile before any of #4, #8, #10, #11** | §5 step 2 | §13.3 | Neither report has one. torch profiler via `--profiler-config` in the image; `nsys` on the host only. NVTX ranges for target, each draft step, PLE submit/wait, sampling, metadata. | Ranks the single-stream overhead items; nothing else can. | One session on the running server, no restart. |
| 10 | **PLE overlap / handshake trimming** | M, O | §8.3, 8.1 | Prefetch already removed the fault loop; what is left is the synchronous handshake (ZMQ, `.item()` syncs, copy-stream sync, spin). Only exposed at low concurrency. Payload at S=8,K=3 is 46 KB, so this is latency, not bandwidth. | Bounded by measured handshake time; unknown until #9. | Medium; do not reintroduce stream memory ops (GB10 has none). |
| 11 | **PLE row cache** | L, S | 8.2 | Both say collect a hit-rate curve first. Miss rate is now ~20%, prefetched. On decode this is a latency-tail item. It might matter more for prefill, where a 2,048 chunk gathers ~32k rows, but that is inference, not measurement. | Small on decode. | Medium; memory competes with the host reserve. |
| 12 | **TP1 skinny BF16 GEMM plans on SM121** | P | #5, §6.3 | Gate verified. After the reduced head, the big BF16 GEMMs left per step are one 1.18 GiB target head read and the HC mixes; cuBLAS efficiency on these shapes is unmeasured. | A few % at 1 stream if cuBLAS is well under 80% of bandwidth; less after #3. | High; CuTe DSL for sm_121 plus TP1 shape plans. |
| 13 | **Compilation mode 3** | E | §6.3 (secondary) | Untested. Fusion only helps the non-bandwidth part, which exists at 1 stream only. Risk: PLE custom op and handshake under Inductor. | 0–5% at 1 stream. | 1 launch, several minutes of compile. |
| 14 | **Per-session vocabulary extension for the reduced head / fused GEMM+argmax / FP8 draft head** | I (64k variant) | §4 items 4–6 | 65k is in. Chinese coverage is 50.6% and Chinese throughput gained nothing; code identifiers are the other tail. A fixed-size per-session extension refreshed at turn boundaries is the one that targets the harness. | Recovers acceptance on multilingual/code tails; ~0 on English prose. | Medium; graph-stable buffers. |
| 15 | **GDN speculative-state traffic** | F (bytes only) | §9 (deferred commit, low-rank log) | Research. First check what the fused MTP kernel actually stores per candidate. | Only large if the profile shows state stores growing with K×S. | High. |

## 4. Worth keeping from one report only

- **`index_share_for_mtp_iteration` (astra §7, end).** Fully plumbed, off in the checkpoint config. Turning it on skips the draft layer's indexer top-k on draft steps 2 and 3 and reuses step 0's indices. Config-only: extend the `--hf-overrides` JSON (`start.sh:533` only emits one when `YARN=1`, so add it there or via `EXTRA_VLLM_ARGS`). Small saving that grows with context (the compressed-key scan is per token per unit of context); a possible acceptance cost on the draft only. Cheap enough to fold into the K-sweep launches.
- **Release the full 1.18 GiB draft head (CHANGELOG note; astra §4.2).** Under greedy draft sampling `compute_logits` on the draft is dead. Memory, not speed: 1.18 GiB back to KV or the host reserve.
- **Startup: skip PLE shard materialisation in the GPU placeholder (HANDOFF §"rough edges" #1; fable §5 end).** `Model loading took 524 s` in the current log. Worth ~7 min per launch and makes every A/B in §3 three times cheaper. Not a serving change.
- **Benchmark fixes (astra 13.2).** `bench/decodebench.py` is sequential and its TTFT/last-token marks are loose; sparkDash is what produced the concurrency numbers. Keep using sparkDash for A/B and read `/metrics` deltas for ms/step and tokens/step.
- **Scoped Marlin experiment for the draft MoE only (astra 6.2).** If anyone still wants the atomic-add test, it needs a patch to the two routed calls in `marlin_moe.py`, and it can only touch the MTP's 1.3 GiB of W4A16 experts. Low value.

## 5. Drop, or re-frame

- **`VLLM_MARLIN_USE_ATOMIC_ADD=1` as a decode experiment (fable D).** No-op for the routed MoE, and the target is CUTLASS. Drop.
- **"Five streams run without graphs" as a +20–25% item (fable #2).** Done and measured at ~3%. Closed.
- **"PLE is the wall that stops concurrency scaling" (fable #3).** Miss rate is ~20% and bytes per token fall with concurrency; closed by measurement.
- **~25 ms of removable per-step overhead at concurrency (fable #5, §5 tables).** Exists only at 1 stream. Any aggregate projection built on it is void.
- **Fable Q's FP8-MMA half.** Its `block_n` premise is stale; the split-count half survives as §3 #8.
- **A second prefill process on the same Spark, `mamba_cache_mode=all`, HMM direct GPU access to the mmap (fable T).** Astra's objections stand; the last one is research with a real hang risk and must not be tried without the memory guards.
- **Driver upgrade, SM overclock (fable §2.1 power-cap remark).** No evidence either is a bottleneck; 580.159.03 is current for Spark.
- **fable's byte model.** Keep it. It is the one thing in either report that predicted a result within a point, and astra's expert-count table is the same model. Use it to rank before every launch.

## 6. Order that spends the fewest launches

1. **One profile session on the running server** (§3 #9). No restart. Settles whether #4, #8, #10 are worth anything at 1 stream.
2. **Static K sweep** (#1), and in the same launches: `index_share_for_mtp_iteration` on, and `VLLM_USE_V2_MODEL_RUNNER=1` pinned so a later dynamic-K launch is a one-line change. Confirm each launch's log has `Using V2 Model Runner`, no "Overriding cudagraph_mode", and the expected FULL decode graph count.
3. **Dynamic K with the measured schedule** (#2). Watch `driver` in the memwatch timeline; abort on the historical signature.
4. **`flashinfer_b12x` opt-in** (#5) and **bf16 SSM** (#6): one launch each, sparkDash decode sweep plus the needle test for bf16.
5. **Chunk size 1024** (#7) measured on the harness's own traffic shape, reporting p95 ITL and TTFT, not just aggregate.
6. **FP8 head + HC re-export** (#3): the last byte item; do it once the cheap launches are banked, with an eval.
7. Kernel items (#4, #8, #12) only where the profile from step 1 shows exposed time.

Throughout: keep `HOST_RESERVE_GIB=26`, record resolved KV and graph memory per launch, and treat a configuration the watchdog stops as a failed experiment regardless of its tok/s.

## 7. What neither report covers

The agent harness that motivates this box is prefill-dominated: 72k-token
prompts at ~2,200 tok/s mean 30 s of TTFT per turn against a few seconds of
decode. Both reports scope themselves to decode. Of the items above, only the
PLE prefetch (done), the 8192 chunk (opt-in), prefix caching for stable system
prompts, and possibly `flashinfer_b12x` at large T touch prefill. A prefill
profile of one 64k prompt would rank that side the same way the byte model
ranked decode.
