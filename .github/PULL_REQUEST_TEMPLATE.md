## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this kit, a good description usually covers:
      * which .env knob or script behavior changes
      * whether the change affects measured numbers (KV pool, TTFT, prefill/decode
        throughput, concurrency, host MemAvailable) and in which direction
      * why the change is safe on a single Spark's unified memory budget

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the [issue tracker](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues).
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this kit that usually means:

      * `bash -n start.sh stop.sh files/memwatch.sh` (syntax check)
      * `python3 -m py_compile files/*.py`
      * `shellcheck start.sh stop.sh` if available
      * `./start.sh --no-launch` to check the derived memory budget and the docker
        command without starting anything
      * an actual launch, plus
        `docker logs vllm-fn-tp1 2>&1 | grep -E "Available KV cache memory|GPU KV cache size"`
      * if behavior changed, the measured numbers with the new settings, stating
        which configuration they came from (see README "Measured profile")

    If you changed a patch generator, confirm the generated file still applies:
    every anchor must match exactly once, and start.sh regenerates the output on
    every launch.

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and `.env.sample` and kept my changes consistent with them.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (script syntax check, a dry run, a launch, or a re-measurement).
- [ ] I edited the patch **generators** (`files/patch_*.py`), not the generated `*_patched.py` / `files/ple_offload/*.py` outputs, which are overwritten on every launch.
- [ ] I updated the README and/or `.env.sample` if a `.env` knob, default, or measured number changed.
- [ ] If my change affects the memory budget, I re-checked it against the host `MemAvailable` floor in the README's safety rules.
- [ ] Defaults in `.env.sample` still work out of the box; a new knob has a sane fallback like the existing ones.
- [ ] I added any new tracked file to `.gitignore` (it is an allowlist: everything is ignored unless explicitly re-included).
