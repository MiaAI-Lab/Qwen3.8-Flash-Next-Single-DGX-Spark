# Changelog

Notable changes to this deployment kit. The repository is not versioned; entries
are grouped by date, newest first. Every measurement named here was taken on the
one DGX Spark this repo is written for — treat them as that host's numbers, not
as promises.

## 2026-09-04

### Fixed

- **The memory watchdog killed healthy servers on a single noisy sample**
  (`c79f765`). `files/memwatch.sh` triggered on one `MemAvailable` reading below
  `MEMWATCH_MIN_GIB`. Two servers were lost to this; in both cases the samples
  either side of the trigger sat 400–700 MiB *above* the floor:

  ```
  23:32:32 avail=6542MiB          <- 398 MiB above the 6144 MiB floor
  23:32:33 MemAvailable=6090 MiB < floor -> docker kill
  ```

  `MemAvailable` moves ~107 MiB between 5 s samples here, with excursions past
  1 GiB, so a one-sample test against a fixed threshold fires on noise. The
  trigger now requires **5 consecutive** sub-floor samples; replayed against
  both recorded crash sequences it does not fire, and it still fires on a
  sustained decline. A lone excursion logs `recovered after N sub-floor
  sample(s)` and resets the counter.

  Not yet proven live: the debounce has not fired in real conditions, and the
  slow downward trend in `MemAvailable` under sustained long-context prefill is
  still unexplained.

- **The watchdog's own kill was invisible in its log** (`c79f765`). It polled
  every second but logged every fifth sample, so the reading that caused a kill
  was never in the post-mortem. It now logs every sample once within 1 GiB of
  the floor.

- **The watchdog leaked POSIX shared memory** (`c79f765`). It used `docker kill`
  (SIGKILL); with `--ipc host` that strands the container's `/dev/shm` segments
  until reboot — the leak `stop.sh` already takes care to avoid. It now sends
  SIGTERM with a 10 s grace period (`MEMWATCH_GRACE`) and falls back to SIGKILL.

- **An incomplete checkpoint got past the pre-flight check** (`b0a9f5e`).
  `start.sh` and `download.sh` treated "`config.json` exists" as "checkpoint
  complete", but `config.json` lands early in a download, so an interrupted
  fetch failed later inside vLLM instead. Both now require every shard named by
  `model.safetensors.index.json` (35 here). Catches missing and dangling-symlink
  shards, not truncated blobs. Split out of #2; co-authored with @lidaiqing.

### Changed

- **FP8 KV: per-tensor scales hoisted out of the QSA dots** (`69f7b4c`, from #2,
  co-authored with @lidaiqing). The kernels previously dequantised each tile with
  vLLM's `_cast_kv_tile`, which materialises an FP32 tile
  (`(data.to(tl.float32) * scale).to(Q.dtype)`); `block_n` was halved to keep
  that inside GB10's shared-memory budget. They now cast FP8→BF16 and apply the
  scalar scales after the dots, which removes the FP32 tile and restores the
  BF16 tile width.

  Exact before rounding: the score scale is a scalar, so it commutes with the
  `softmax_scale` multiply and the validity mask, and the output scale is
  applied to `normalized_output` above the `NUM_SPLITS` branch so it factors
  through the split-K LSE merge. Slightly *more* accurate than dequantising
  first, since FP8→BF16 is exact while rounding `scale × fp8` into BF16 is not.

  Measured by the contributor: +6.3% prefill @32k, −40.6% sparse-QSA kernel
  latency, one BF16 ULP maximum error, BF16 path bit-identical. End-to-end on
  this host the kernel alone came out around +3% at 32k against an unmatched
  baseline — inside run-to-run noise. No matched A/B has been run here.

- **`.env.sample` documents `MAX_NUM_BATCHED_TOKENS`** and keeps the default at
  2048 (`366f6df`). Raising it to 8192 measured, on this host, 32k prefill
  2,133 → 2,366 tok/s (+10.9%) and TTFT 15.38 → 13.87 s (−9.8%), with the whole
  8k–128k curve flattening. It is offered as an opt-in knob rather than a
  default because the supporting observation is minutes, not hours. It is paid
  for out of the KV pool rather than the GPU budget: 8192 chunks raise peak
  activation to 1.27 GiB, which vLLM profiles before sizing the KV cache.

- **README prefill table now carries both chunk widths side by side**, each
  labelled with the configuration it was measured at (`366f6df`). They differ in
  rope config and KV target as well as chunk width, so only the 32k pair is a
  clean A/B.

- **`start.sh`'s FP8 warning dropped a stale speed claim** (`69f7b4c`). It cited
  the reference implementation's ~30% slower prefill and ~9% slower decode; the
  quality warning (a long-reasoning benchmark falling 6/6 → 2/6) stands.

- **README: `### FP8 KV cache (opt-in)` → `(default)`**, and the patch bullet's
  "Off by default" corrected (`69f7b4c`). Both contradicted the body text and
  the shipped `KV_CACHE_DTYPE=fp8`.

### Reverted

- **`KV_TARGET_GIB` 22 → 20 → 22 → 20.** Lowered in `554f295` on the theory that
  the watchdog kills were a memory problem, restored in `c79f765` once those two
  kills turned out to be a watchdog bug, then lowered again once a third server
  died on a genuine sustained decline. Both things were true: the watchdog fired
  on noise *and* the host margin at 22 is thin. The default is 20.

- **`MAX_NUM_BATCHED_TOKENS` default 2048 → 8192 → 2048.** Defaulted in
  `677f4ab`, backed out in `366f6df` and documented as an opt-in instead. See
  above for the measurements.

### Verified

- **The debounced watchdog correctly distinguished a real event from noise.** A
  third server died at 23:53 on a monotonic descent — 7,101 MiB to 5,726 MiB in
  9 seconds, still falling — and the trigger fired after 5 consecutive sub-floor
  samples. The per-sample near-floor logging captured the whole descent, which
  the old script could not have shown. Unlike the two earlier kills, the
  container's own cgroup was *growing* through this one (10,716 → 12,348 MiB),
  so the mechanism differs from the slow drift.

  The 10 s SIGTERM grace was not enough — `docker stop` escalated to SIGKILL and
  the exit code was still 137, so the shm-leak protection did not take effect.

### Known open

- Host `MemAvailable` has two unexplained behaviours: a slow decline under
  sustained long-context prefill, and at least one burst that consumed ~1.4 GiB
  in 9 seconds. `memwatch.sh` logs `MemAvailable`, `MemFree`, `SwapFree` and
  the container's cgroup usage; adding `Mapped`/`Cached`/`AnonPages` would let
  the next descent identify its own cause.
- The shipped default (262k, `KV_TARGET_GIB=22`, FP8, 2048 chunks) has still not
  been benchmarked end to end.
- Whether the 6 GiB watchdog floor is the right threshold has never been
  examined; it was chosen during bring-up.
- `MEMWATCH_GRACE` (10 s) is too short for vLLM to shut down cleanly.
