---
name: Improvement
about: Something in this kit already works, but could be faster, safer, clearer or better defaulted.
title: ""
labels: "enhancement"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README and .env.sample first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you want to propose an improvement to something that already exists, fill
     out the template below. For something that does not exist at all, please use
     the Feature request template instead.
-->

## What could be better

<!--
     Describe the current behavior and why it is not ideal. Examples that fit this
     repo:
       * a default in .env.sample that is wrong for most people
       * a memory budget rule that is too conservative or not conservative enough
       * a patch generator that breaks against a newer image
       * startup time (currently ~10-12 min; the GPU-worker placeholder still
         materialises every PLE shard while draining its weight iterator)
       * a benchmark that measures the wrong thing
-->

## Proposed change

<!--
     What should it do instead? If this touches the vLLM patches, note that
     files/*_patched.py and files/ple_offload/*.py are GENERATED on every launch
     from pristine sources extracted from the image. Edit the generators
     (files/patch_*.py), never the generated output.
-->

## Measured impact

<!--
     This kit lives or dies on measured numbers, so please include them where you
     can. The README records results in this shape:

       * KV pool (GiB and tokens) from `docker logs vllm-fn-tp1 | grep "GPU KV cache size"`
       * prefill tok/s and TTFT from `python3 bench/longctx.py --target N`
       * decode tok/s from `python3 bench/decodebench.py`
       * host MemAvailable under load, and the low-water mark during a long prefill
       * needle retrieval PASS/FAIL at 5/50/95% depth

     Before/after pairs are much more useful than a single number, and please say
     which configuration each number came from (context length, KV_CACHE_DTYPE,
     KV_TARGET_GIB) - they move a lot between settings.

     If you are comparing two configurations, please use the same sample count for
     both. A rare failure will not show up in one or two runs.
-->

## Risk

<!--
     Does this change the memory budget? On unified memory, exhausting the pool
     hangs the kernel with no OOM kill and no logs, so anything that raises the GPU
     budget or lowers host headroom needs the Step 2 arithmetic re-checked. Say so
     here if it does.
-->
