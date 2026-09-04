---
name: Bug report
about: The model kit is not behaving as expected, is producing an error, or the numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README and .env.sample first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's "Measured profile" and
     "Configuration" sections list the knobs that affect behavior and the defaults
     shipped in .env.sample. -->

- Hardware: <!-- e.g. 1x DGX Spark (GB10, 128 GB unified memory) -->
- Host `MemAvailable` before launch: <!-- `grep MemAvailable /proc/meminfo` -->
- Image / vLLM version: <!-- `docker images | grep vllm`; `docker exec vllm-fn-tp1 python3 -c "import vllm;print(vllm.__version__)"` -->
- Model (`TP1_MODEL_ID`): <!-- default Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 -->
- `start.sh` invocation: <!-- e.g. `./start.sh`, or `YARN=1 ./start.sh` -->
- Relevant `.env` values: <!-- YARN, MAX_MODEL_LEN, YARN_MAX_MODEL_LEN, KV_TARGET_GIB, KV_CACHE_DTYPE, MTP_NUM_SPECULATIVE_TOKENS, MAX_NUM_SEQS, HOST_SLACK_GIB, PLE_OFFLOAD -->
- Output of `./start.sh --no-launch`: <!-- the derived budget block is usually the fastest way to diagnose a memory problem -->

---

## Steps to Reproduce

<!-- Please include full steps so that we can reproduce the problem. -->

1. Run `./start.sh` <!-- describe any overrides and what it printed up to the failure -->
2. ... <!-- describe steps to demonstrate the bug -->
3. ... <!-- for example "curl /v1/models reports a different max_model_len than .env asked for" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add any other context here: a minimal request that reproduces bad output, JSON
responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.

      NOTE: this build emits reasoning BEFORE the answer, in a "reasoning" field
      rather than "content". Budget at least ~400 max_tokens: at 200 the reply is
      often still inside its reasoning and "content" comes back empty on a
      perfectly healthy server. That is not a bug.
-->

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 400
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the backticks, and mention whether it came
      from `start.sh`, `docker logs vllm-fn-tp1`, `logs/memwatch-vllm-fn-tp1.log`,
      or a client.

      Common culprits worth checking before filing:
        * Host froze with no OOM kill and no logs -> host MemAvailable went below
          ~10 GiB. KV_TARGET_GIB is the knob that eats it (README "Safety rules").
        * Container vanished mid-run -> the watchdog killed it; check
          logs/memwatch-vllm-fn-tp1.log for the MemAvailable trace.
        * `start.sh` refuses port 8888 -> comfy-h3.service is active and will launch
          a GPU co-tenant as soon as anything answers there.
        * Gibberish output -> the PLE path has regressed (bf16 IPC buffer or missing
          quant scales). See "What is patched and why" in the README.
        * `KV_CACHE_DTYPE=fp8` refused -> the stock QSA backend only accepts bf16;
          files/patch_qsa_fp8_kv.py is what adds fp8 support.
        * Startup takes ~10-12 min -> that is expected, not a hang.
-->

```

```

</details>
