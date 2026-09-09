# Source of truth — upstream review + 24/7 stability plan

Last verified: 2026-09-09, against fork `main` at `d038090` (= upstream `main`, merged PR #39).
Status update 2026-09-09 (later the same day): the gap rows below are being closed
on branch `24-7-stability` (Phase 0 + Phase 1 + the Phase 2 plumbing items; see
`.kilo/plans/247-stability-plan.md`). Rows fixed there are struck from §2.2 and
noted here with their commits; §2.2 and §4/§5 remain the source of truth for
anything still listed as open.

Fixed on `24-7-stability` (commit → review row):

- `e950715` smoke-test.sh ported (§2.2 #18, fix-branch survivor) — health,
  coherent generation, temp-0 determinism WARN-only, ≥15 tok/s decode floor,
  tool-call round-trip with `completion_tokens > 0` asserts, /metrics series.
- `063e4bd` stop.sh reads `.env` (#24 / PR #26); touches `logs/stopping`.
- `6a14079` download.sh sha256-verify vs paginated HF tree manifest (§4.8 A1;
  entry count printed, ≥ local LFS-file count asserted, ABLIT resume state).
- `94d17a1` + `2041b43` start.sh: `BIND=127.0.0.1` default + api-key WARN
  (#8/PR #25), `READY_TIMEOUT_S` heartbeat/timeout (#10), `EXTRA_VLLM_ARGS`
  word-split (#11), HF_TOKEN exec-time resolution + `chmod 600` (#9), log
  rotation flags (§4.5), MTP legality + async-scheduling guards (§4.8), JIT
  bounds `MAX_JOBS=2`/`FLASHINFER_NVCC_THREADS=1` (§4.8 C2), quant_algo
  pre-flight (§4.8 A2b), `PLE_GIB` derived from checkpoint PLE shards
  (#16 remainder), determinism/GDN/block-drop env plumbing (§6.2, Phase 2).
- `9ff5c9d` + `42991e9` memwatch: `WATCHDOG EMERGENCY STOP` marker, alert hook,
  LEAK TREND line (§4.3).
- `1e9645f` + `9c6d66f` supervisor loop, health probe (stateless,
  `completion_tokens > 0`), shm cleanup, maintenance relaunch (Sun 04:00,
  drain via `vllm:num_requests_running` incl. labels), heartbeat, alert.sh
  (webhook, rate-limited, silent-safe), systemd user units, log rotation
  (§4.1, §4.2, §4.3, §4.4, §4.5, §4.6, §4.7 README).
- `f7a5a15` README "Unattended operation" section (§4.7 host hygiene), BIND
  migration note, knob documentation; CHANGELOG per kept change.

Still open on the branch (Phase 1 exit drills + Phase 2/3): the six Definition-
of-Done drills and the 7-day soak have not been run; ABLIT PLE identity check is
excluded (user direction — interim WARN only); `GDN_DECODE_KERNEL=triton` flip
deferred one release; `disable_eagle_block_drop` A/B unmeasured;
`VLLM_USE_DEEP_GEMM` relevance unverified on our image.

The jschmied reference repo (§6) was verified **directly** the same day: README,
`patches/MANIFEST.md`, `scripts/serve-flashnext.sh`, `tools/main/` (incl. `memguard.sh`),
`REPRODUCE.md`, `notes/failure-modes.md`, `notes/mtp-vs-prefix-cache.md`,
`notes/what-generalises.md`. Claims that came only from the earlier draft and could not be
confirmed against the source are marked as such or removed.
This document supersedes the earlier upstream-review draft. It is the single reference for:

1. what "the model runs stably 24/7" means on this hardware,
2. what is already fixed, what is broken, and where each thing lives,
3. what upstream/third-party work is worth importing,
4. the prioritized plan to close the remaining gaps.

Read section 1 for the goal, section 2 for where the code actually stands today,
section 4 for the honest gap list, and section 5 for the work order. Everything
in between is supporting context. File/line references are to commit `d038090`.

---

## 1. The goal, in plain terms

"The model stable running 24/7" on a single DGX Spark means five things, and
the current repo delivers only the first two:

| # | Property | Status today |
|---|----------|--------------|
| 1 | Correct output — model answers, follows tools, doesn't loop | **Partial** — works at sane temperature; ABLIT is broken (#34); determinism has known kernel bugs |
| 2 | Doesn't kill the host — memory watchdog + host-side budget | **Done** — memwatch + HOST_RESERVE_GIB cap, measured over ten launches |
| 3 | Detects a wedged engine (alive but not serving) and recovers | **Missing** — nothing watches the server after first `/health` 200 |
| 4 | Recovers from a stop/crash/reboot without a human | **Missing** — no restart policy, no supervisor, watchdog dies with the container |
| 5 | Survives its own byproducts — leaked shm, growing logs, slow memory leak | **Missing** — no cleanup cadence, no log rotation, leak acknowledged but unmanaged |

The hardware makes this harder than a normal server:

- **Unified memory.** CPU and GPU share 121.69 GiB. Exhausting the pool does not
  raise an OOM — it hangs the kernel (three hosts lost this way on 2026-09-04,
  see start.sh:33-34). Every availability mechanism must respect the memory
  budget; a naive `--restart` loop that ratchets leaked memory can OOM the box.
- **Cold start is expensive.** ~11 min to `/health` (≈96 GB NVMe read), plus a
  one-time ~27 GB packed-PLE-table build. Recovery is not instant; the design
  must distinguish "restarting" from "dead" for users and alerting.
- **A single watchdog, one trigger.** `files/memwatch.sh` watches host memory
  only. It protects the *host*, not the *service*.

The core missing concept: **a closed detect → stop → recover loop.** Today the
kit can detect (partially) and stop, but never recover, and parts of detection
(health probing) don't exist at all.

---

## 2. Verified state of our `main` (2026-09-09)

### 2.1 What `main` already has (do not re-do these)

These were listed as missing or "in our branches" in earlier notes; they are on
`main` right now:

- **`--reasoning-parser qwen3`** — start.sh:657
- **`--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder`** — start.sh:658-659
  (note: jschmied's repo uses `qwen3_xml` for this model — see §6.2)
- **`--distributed-executor-backend mp`** — start.sh:661 (PLE offload requires it)
- **`--cap-add SYS_NICE --cap-add SYS_PTRACE`** — start.sh:739 (SYS_PTRACE is the
  PLE-offload ~10-minute-death fix)
- **Snapshot selection via `refs/main`, newest-complete fallback** — start.sh:344,
  download.sh:72-73. This is upstream issue #15, already fixed on `main`.
- **Host-side GMU cap** (`MemTotal − HOST_RESERVE_GIB` bounds the GPU budget;
  "KV target reduced to Y" when the wish exceeds it) — start.sh Step 2 budget
  derivation, measured over ten launches, MemAvailable ≥ 13 GiB (see .env.sample
  KV_TARGET_GIB comment). This is most of upstream issue #16.
- **Watchdog graceful-then-force stop** (`docker stop -t GRACE`, fallback
  `docker kill`) with log archiving before stopping — files/memwatch.sh:75-84.
  This is most of upstream issue #13.
- **Dual watchdog floors** (MemAvailable < 6 GiB, or MemFree < 2 GiB while
  MemAvailable < 10 GiB) + NV_ERR_NO_MEMORY kernel-log counting — memwatch header
  and lines 102-129.
- **Env snapshot/restore so environment > .env works** — start.sh:92-131.
- **KV cache fp8 + Mamba SSM state bfloat16 + reduced draft vocab (47k) +
  `VLLM_USE_V2_MODEL_RUNNER=1` pin** — .env.sample defaults (these are the
  measured-good profile from the 2026-09-05/06 overnight runs).

### 2.2 What is broken or missing on `main`

Each row is verified against the code at `d038090`:

| Gap | Where | Upstream ref | Effect |
|-----|-------|--------------|--------|
| Readiness loop never times out | start.sh:818-833 (`while true; sleep 10`) | #10 | Stalled weight load hangs `start.sh` forever; under systemd this blocks the unit in `activating` until `TimeoutStartSec` |
| `EXTRA_VLLM_ARGS` not word-split | start.sh:709 (`VLLM_ARGS+=("$EXTRA_VLLM_ARGS")`) | #11 | Multi-word/quoted values (JSON configs, `--api-key`) break into wrong argv. `EXTRA_DOCKER_ARGS` is mostly fine (heredoc expansion splits it), the bug is the vLLM side |
| HF_TOKEN persisted in `.last_launch.sh` | start.sh:749 → cp at :769, no chmod | #9 | Token in a 644 file in the repo dir |
| API bound to all interfaces | start.sh:765 `--host 0.0.0.0` + `--network host` | #8 / PR #25 | Unauthenticated API on every network the box has |
| `stop.sh` ignores `.env` | stop.sh:13 (env var only, never sources .env) | #24 / PR #26 | `./stop.sh` targets the wrong container when `TP1_CONTAINER_NAME` is customized |
| No restart policy / boot integration | docker run at start.sh:730s has no `--restart`; nothing systemd | #14 | Container death or host reboot = down until a human shows up |
| Watchdog exits when container dies | memwatch.sh:92 loop condition, :139 exit | (new — see §4) | Any recovery mechanism based on `docker --restart` runs unprotected |
| No post-ready health monitoring | start.sh stops probing at first 200 | #18 (partial) + §4 | A wedged-but-alive engine serves nothing, forever, silently |
| No log rotation | memwatch writes every 1-5 s; `docker run` has no `--log-opt`; `logs/archive/` grows per stop | #17 | Unbounded disk growth; disk-full on this box also threatens the 99 GiB checkpoint cache |
| ABLIT reuses stock PLE when `edit_ple:false` | start.sh:614-619 | #34/#36 | ABLIT=1 serves wrong PLE for 17/34 shards → "fluent but wrong / ignores instructions / loops" |
| No MTP legality guard | start.sh:663-683 accepts any `MTP_NUM_SPECULATIVE_TOKENS` | jschmied | illegal k (5–8, and 13–16 at block 848) **hard-fails the engine at config validation** — a "stop" that looks like a boot failure; k with `--async-scheduling` → silent n-gram corruption |
| No JIT fan-out bounds | no `MAX_JOBS` / `FLASHINFER_NVCC_THREADS` in the container env | jschmied C2 | an unbounded compile fan-out on GB10 OOMs the **whole box**, not the container; a driver upgrade invalidates the JIT cache, so the first launch after one is exactly when this fires |
| `VLLM_GDN_DECODE_KERNEL=triton` absent | not in start.sh / .env.sample | jschmied REPRODUCE §4 | Default CUDA kernel deterministically hangs the engine at c≈32 with FP8 GDN projections. No error, requests stall (silent wedge class) |
| `PLE_GIB` hardcoded | start.sh:208 (`26.82`) | #16 remainder | Wrong PLE estimate after a checkpoint change silently mis-sizes the budget |
| No sha256 verification of downloads | download.sh checks completeness only (weight_map files exist), not content | jschmied A1 | A corrupt (preallocated-size) download yields fluent-but-wrong output invariant to every config; two size-correct corrupt shards cost them a full day and two retracted upstream issues |
| No quant_algo dispatch check | nothing validates the image's `ModelOptMixedPrecisionConfig.get_quant_method` against the checkpoint's `quant_algo` set before launch | jschmied A2b | An unrecognized `quant_algo` (e.g. `FP8_PB_WO`) falls through to `UnquantizedLinearMethod` — packed FP8 bytes loaded as BF16, whole-checkpoint fluent garbage, zero errors |
| No alerting anywhere | memwatch/stop.sh print to local logs only | (new — see §4) | 24/7 unattended with no notification = users discover downtime first |

### 2.3 The two fix branches are stale — re-port, do not merge

`fix/quick-wins` (upstream PR #21) and `fix/ops-batch` (upstream PR #22) both
branched from `665dea8` and are **20 commits behind `main`**. The diff is large
and mostly *deletions* of things `main` gained since (README, patch generators,
sysctl file, the entire ablit path). Concretely:

- Parts of `fix/quick-wins` were already landed upstream via merged PR #39
  (snapshot selection) or were independently implemented better on `main`
  (KV cap, watchdog semantics).
- Parts of `fix/ops-batch` are already subsumed (memwatch's cgroup fallback and
  graceful stop are better on `main` than the branch's versions).
- What is still worth taking from the branches: **`scripts/smoke-test.sh`**
  (health, coherent generation, temperature-0 determinism, decode speed,
  max_model_len assertion) and **`qwen38-flash.service`** (as a *starting
  point only* — its `Type=oneshot` design cannot restart a crashed container
  and does not supervise memwatch; see §4.1).

**Action: audit each branch hunk against current `main`, port the survivors as
small clean commits, then delete the branches.** Do not `git merge` them.

---

## 3. Upstream landscape (condensed)

Fork `malvavisc0/Qwen3.8-Flash-Next-Single-DGX-Spark` at `main` = upstream
`MiaAI-Lab` `main` (`d038090`, merged PR #39). We have no issues/PRs of our own;
everything below lives upstream. Both our PRs (#21, #22) are still open there.

### 3.1 Upstream PRs (8 open)

| PR | What | Verdict for us |
|----|------|----------------|
| #35 / #33 | download.sh quoting fix | **Already in `main`** (via merged #39). Nothing to do. |
| #29 | 64KB-page kernel-load hang / OOM (cloned lazy yields) | Only if we hit boot stalls on a 64K-page DGX OS image. Revisit on evidence. |
| #31 | Opt-in GB10 kernels (+5.5% solo tok/s) | Large, conflicts. Later, if ever — performance, not stability. |
| #26 | stop.sh reads `.env` | **Port** (3-line fix, still broken on our `main`). |
| #25 | Bind API to loopback | **Port** (still `0.0.0.0` on our `main`). |
| #22 / #21 | Ours (ops batch / quick wins) | Superseded: branches are stale, re-port the survivors (§2.3). |

### 3.2 Upstream issues that matter, re-triaged against our `main`

**Correctness — "wrong output / stops generating"**

- **#34 (+dup #36) — ABLIT PLE mismatch.** Our start.sh:614-619 keys PLE reuse on
  `recipe.edit_ple is False` from `ABLIT_META.json`, but the PLE shards genuinely
  differ in 17/34 shards. Until fixed, **ABLIT=1 serves wrong weights** — the
  "fluent-but-wrong / ignores prompts / loops" symptom. Highest-value correctness
  fix. Fix: honor the actual shard diff (compare shard hashes against stock),
  rebuild the packed table keyed on the real identity, not the `edit_ple` flag.
- **#7 + #28 — greedy decoding non-deterministic on GB10.** QSA top-k drops
  candidates (#7, upstream vllm#55122) and, per jschmied, the deeper defect is
  non-deterministic *indexer scores* requiring a stack of overlays (§6.2, §6.5).
  These
  are **image/kernel-level** — our kit can only (a) document the temperature-0
  caveat, (b) ship the smoke-test determinism gate (sequential-only; §6.5),
  (c) adopt the fix when the image rolls it in. Not a launcher fix.
- **#20 — "stops generating after several runs" (user report).** Triage
  conclusion (ours, confirmed upstream): default temperature 1.0 drifting out of
  tool-call format (config, not weights) + the #7 kernel issue + context overflow
  presented as a clean `finish_reason: stop`. jschmied adds a fourth mechanism
  with numbers (§6.1): on short agent turns, MTP's fixed 1,600-token cache-block
  back-off makes each turn slower and shorter-feeling than no speculation at
  all (break-even ≈68 output tokens) — read by users as "the model stopped".
  Action for us: safe defaults + the §4 detection layer + jschmied's pre-flight
  items (§4.8) + the `disable_eagle_block_drop` A/B (Phase 2).

**Ops/stability — "stops with no error / hangs / down"**

- **#10 — readiness timeout.** Still open on our `main`. Fix: `READY_TIMEOUT_S`
  (default 1800 — cold start is ~11 min, first boot longer) + heartbeat output
  while waiting; exit non-zero on timeout so a supervisor can retry.
- **#14 — restart policy / systemd.** Still open; the real design is §4.1, not
  just `--restart unless-stopped`.
- **#13 — watchdog kill hygiene / cgroup fallback.** **Largely fixed on `main`**
  (graceful SIGTERM stop with archive, cgroup read with `|| echo 0` fallback).
  Remaining nicety: memwatch logs when `cg` reads 0 instead of silently showing
  `container=0MiB`.
- **#16 — memory budget.** **Largely fixed on `main`** (host-side cap, measured).
  Remaining: derive `PLE_GIB` from the actual packed table instead of the
  hardcoded 26.82.
- **#17 — log rotation.** Open; for 24/7 this is a disk-failure risk, not
  cosmetics (§4.5).
- **#18 — smoke test + /metrics docs.** Smoke test exists in `fix/ops-batch`
  (port it); /metrics documentation still missing; continuous probing is new
  work (§4.2).
- **#19 — MAX_NUM_SEQS=4 flatlines aggregate throughput.** Perf, not stability;
  safe measured value is 8 (2026-09-05/06 sweeps). Ship as default change with
  the measured caveats.
- **#24 — stop.sh reads .env.** Open; port PR #26.
- **#15 — snapshot selection.** **Fixed on `main`.** No action.

**Security**

- **#9 — HF_TOKEN leak.** Open on `main` (start.sh:749 → :769, no chmod).
- **#8 / PR #25 — API on 0.0.0.0.** Open on `main`. Fix both halves: default
  `BIND=127.0.0.1` knob + a startup warning when bound non-loopback without an
  `--api-key`.

**Later / optional:** #12 (hybrid checkpoint, +20% decode), #6 (safe prefix
caching — needs vllm#50729 Mamba block-size fix), #4 (prefill measurements),
#29/#31 (kernels), #27/#37 (NVFP4 GDN corruption — checkpoint-level; revisit
if we switch checkpoints), #38/#30/#23 (informational).

---

## 4. The 24/7 gap analysis — what nothing in the repo does today

This is the section the earlier review missed entirely. Each subsection is a
work item in §5.

### 4.1 Nothing supervises the service (the big one)

The current architecture has a hole exactly where 24/7 needs it closed:

- **memwatch's loop runs only while the container exists** (memwatch.sh:92:
  `while docker ps | grep container`). When the container dies — watchdog stop,
  crash, OOM-cgroup kill — memwatch exits (line 139). It cannot restart anything.
- **`docker --restart unless-stopped` alone is wrong**: the restarted container
  comes back healthy-looking but **unwatched** (memwatch is dead), so the first
  memory emergency after it becomes a host hang.
- **The ops-batch systemd unit is insufficient by design**: `Type=oneshot` +
  `RemainAfterExit` means systemd runs `start.sh` once and never watches the
  container; its own comments concede "a crash of the already-running container
  is vLLM's/docker's business". For 24/7 it is exactly our business.
- **The failure chain is unattended**: watchdog stops container → memwatch
  archives logs and exits → server down → no alert, no recovery.

**Required design** (implement once, as one supervisor):

1. A systemd unit (`Restart=always`) runs a **supervisor loop** script that owns
   the whole lifecycle: ensure container up (docker run / docker start), ensure
   memwatch running, run the health probe, execute recovery with backoff.
2. Recovery policy: on container death or watchdog stop → clean our leaked
   `/dev/shm` segments (see 4.4) → relaunch, with **exponential backoff and a
   cap on consecutive emergency restarts** (e.g. 3 in a row → stop trying,
   alert loudly, stay down — a crash loop on this box can hang the *host*).
3. memwatch stays exactly as-is (it exits after stopping the container — that
   is correct), but its stop must be *observable* by the supervisor: exit code 2
   + a marker line in its log the supervisor greps, so emergency stops get
   backoff treatment while clean exits do not.

### 4.2 Nothing watches the service after first readiness

`start.sh` polls `/health` until the first 200 (start.sh:826) and then stops
probing. memwatch watches *memory*. Between them, nothing notices:

- **Silent wedges**: GDN decode-kernel stall at c≈32 (no error, requests hang —
  jschmied §6.2), PLE offload semaphore leaks (jschmied finding 138: with CUDA
  graphs the unpatched offload consumes the *previous* step's PLE outputs —
  real traffic never hides it), engine thread death with healthy
  memory. The server is "up" and serves nothing.
- **Slow degradation**: short/truncated agent turns read as "model stopped" long
  before any memory floor trips — per jschmied's measurements, MTP below the
  ~68-token break-even is *slower per turn than no speculation* (fixed
  1,600-token cache-block back-off per turn), and MTP timings vary up to 1.83×
  per server start. Detection: probe latency trends; fix: the
  `disable_eagle_block_drop` lever and per-workload MTP defaults (§6.1).

**Required**: a continuous probe (a small loop or systemd timer, ~1/min):
`/health` + a **1-token generation request** (health alone can stay 200 through
several of these failure modes); N consecutive failures (e.g. 5) → graceful
stop → supervisor recovery path. Log probe latency as a free trend metric.
This is the *detection* half of the loop; 4.1 is the *recovery* half.

### 4.3 The slow leak is documented but unmanaged

The repo's own docs quantify it: "2-3 GiB of per-request growth that is never
returned" (start.sh:51), ~3 GiB driver growth over 13 h of agent traffic
(docs/goal-overnight-2026-09-05.md, memory rules). At that rate a long-lived
server eats its 13 GiB MemAvailable margin in days and the watchdog stops it —
correct, but it converts a known, scheduled maintenance item into an
unscheduled outage.

**Required**: a **scheduled graceful relaunch** (systemd timer, weekly, at a
chosen low-traffic hour: stop → shm check → start), plus a memwatch *trend
alert* on the `driver` figure (its own header documents that a permanent step
up in `driver` is CUDA-allocator growth). Scheduled restarts are ~11 min of
planned downtime; watchdog stops are the same 11 min but unplanned. Also
document for users: the maintenance window, and that in-flight requests are
drained (stop.sh SIGTERM path).

### 4.4 `/dev/shm` ratchets across emergency cycles

With `--ipc host`, a SIGKILL-ed container leaks `psm_*`/`sem.mp-*` segments onto
the host until reboot (stop.sh:53-59). Each emergency cycle therefore *shrinks
the usable pool*, and an auto-recovery loop without cleanup ratchets the host
toward the hang we built everything to avoid.

**Required**: the supervisor cleans our own segments between cycles — same test
stop.sh already uses (`psm_*`, `sem.mp-*`), executed only when no vLLM/sglang
container is running (stop.sh's comment already states the rule). Cheap, and it
converts the ratchet into a flat line.

### 4.5 Disk grows forever

memwatch logs every 1–5 s for eternity; `docker run` uses the default json-file
log driver with **no size cap** (start.sh:730s); every stop writes up to 3000
container-log lines plus a memwatch copy into `logs/archive/`. Weeks of 24/7
operation fill the NVMe that also holds the ~99 GiB checkpoint — disk-full
during weight loading is how you get a corrupted-looking cache.

**Required**: `--log-opt max-size=50m --log-opt max-file=3` on docker run (or
journald driver); size-capped/truncated memwatch log (it is timeline data;
truncate on rotation, keep archives); a pruning rule for `logs/archive/`
(e.g. keep last 20); optionally a disk-floor check in memwatch alongside the
memory floors (same debounce pattern, `df` on the cache volume).

### 4.6 No alerting

Every failure above is silent outside the box. 24/7 unattended without a
notification path means the first reporter of downtime is a user.

**Required**: one tiny `alert.sh` (ntfy/webhook/telegram — pick one) called from:
supervisor on any restart, memwatch stop_container (it already logs a reason —
add the hook call), systemd `OnFailure=` on the unit, and a daily heartbeat
("alive, N restarts this week, MemAvailable X"). Also one **negative** test:
send a deliberately failing alert once, so we know the path works.

### 4.7 Host/boot hygiene

For a box that must come back after power events on its own:

- `loginctl enable-linger <user>` (user units start at boot without a login).
- Docker service enabled (DGX Spark OS default; verify once).
- **`comfy-h3.service` must stay disabled** — it grabs port 8888 the moment
  anything answers there (start.sh:60-63); a reboot with it enabled means our
  server can't take its port. Add a boot-time check to the supervisor.
- Unattended-upgrades: disable automatic reboots (an Ubuntu auto-reboot at 02:00
  with no linger/systemd setup = down until morning) and pin the NVIDIA driver
  (a driver bump under a running server is a forced outage).
- NTP on (log correlation across memwatch/journal/archives is worthless without
  synced clocks).
- `REQUIRE_IDLE_GPU=1` behavior at supervisor boot: on reboot the GPU is idle by
  definition, but after a *crash-loop* restart the previous engine's memory may
  not be released yet — the supervisor's backoff must be ≥ the release window.

### 4.8 The wedge-class config items (prevention beats detection)

Detection (4.2) is the safety net; these cheap settings remove known wedge
triggers entirely — all verified absent on `main`:

- **`VLLM_GDN_DECODE_KERNEL=triton`** on the container (jschmied: default CUDA
  kernel hangs the engine at c≈32, no error). Add to the docker env; verify
  against our image first (it is version-dependent; if the env var is unknown
  to our vLLM it is silently ignored, which is safe).
- **MTP legality guard** in start.sh: validate `MTP_NUM_SPECULATIVE_TOKENS` against
  the ring-capacity formula from the checkpoint's block size (jschmied:
  `capacity = compress_ratio × cdiv(compress_ratio + n, compress_ratio)` must
  divide `block_size`; at our 848 that legalizes 0–4 and 9–12, outlaws 5–8 and
  13–16; illegal k hard-fails the engine at config validation). Also reject k=1
  (strictly dominated — same fixed cache-block cost as k=2, half the decode
  gain), and refuse `--async-scheduling` in EXTRA_VLLM_ARGS while MTP > 0
  (silent n-gram corruption, jschmied: "no benchmark reveals it").
- **Tool-call parser check**: we ship `qwen3_coder` (start.sh:659); jschmied's
  field notes say `qwen3_xml` for this model. One of them is wrong for *our
  image*. Verify with the ported smoke test's tool-call check; keep whichever
  passes and record the evidence in CHANGELOG.
- **sha256 verification** in download.sh (jschmied failure-mode A1: aria2
  preallocates to final size, so size checks pass on corrupt content; symptom is
  fluent-but-wrong output invariant to every config — unfixable from the
  launcher, preventable here). Use HF's `lfs.sha256`; make it
  `VERIFY_SHA256=1` default-on with a documented skip. **Paginate the HF tree
  API** (50 entries/page, `Link: rel="next"`) and print the entry count — a
  non-paginating fetcher writes a manifest of a truncated file list that then
  "verifies" cleanly (jschmied hit exactly this: 50 of 144 files recorded,
  all-present files verified, a third of a checkpoint certified complete).

---

## 5. The plan

Ordered by "unattended availability first, correctness second, performance
last". Phases 0–1 are the 24/7 closure; each item is small and independently
shippable.

### Phase 0 — unblock the base (an evening)

1. **Re-port the fix-branch survivors** (§2.3): smoke-test script, `.env`-aware
   stop.sh (PR #26), `EXTRA_VLLM_ARGS` word-split (`read -ra`, #11), HF_TOKEN
   hygiene (#9: don't bake the token into `.last_launch.sh` — pass at runtime
   or `chmod 600`), `BIND=127.0.0.1` default + warning (#8/#25), log-rotation
   flags (#17, §4.5). Delete the branches after.
2. **Readiness timeout** (#10): `READY_TIMEOUT_S=1800`, heartbeat during wait,
   non-zero exit on timeout (so the future supervisor can retry).
3. **MTP legality + async-scheduling guards** (§4.8).
4. **JIT fan-out bounds**: `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1` into the container
   env (jschmied C2 — unbounded compile fan-out OOMs the whole box, and a driver
   upgrade is the trigger because it invalidates the JIT cache).
5. **sha256 verification in download.sh** (§4.8, jschmied A1) — with HF tree API
   **pagination** (50 entries/page) and a printed entry count, or the manifest itself
   is silently truncated and "verified".
6. **Quant_algo dispatch pre-flight** (§4.8, jschmied A2b) — refuse to launch when
   the image's `get_quant_method` does not dispatch every `quant_algo` the checkpoint
   declares.
7. **PLE_GIB derived** from the actual packed table (#16 remainder).

### Phase 1 — close the 24/7 loop (the core work)

8. **Supervisor loop + systemd unit** (§4.1): one script owning
   ensure-container/ensure-watchdog/probe/recover, `Restart=always`,
   backoff, 3-emergency-stops circuit breaker, boot-time comfy-h3 and
   linger checks. This replaces the ops-batch oneshot unit design.
   (jschmied's bare-metal SYS_PTRACE note applies here: a systemd unit
   running the engine needs `AmbientCapabilities=CAP_SYS_PTRACE` —
   `CapabilityBoundingSet` alone is not sufficient. Ours runs in Docker
   with `--cap-add=SYS_PTRACE`, so this only bites if we ever move to
   bare-metal/systemd serving.)
9. **Continuous health probe** (§4.2): 1-token generation probe ~1/min,
   5-failure trigger, latency trend logged. Probe must assert
   non-empty `completion_tokens` (jschmied: an empty output "passes" a
   naive check while the model is still inside its thinking block).
10. **shm cleanup between cycles** (§4.4) — supervisor step.
11. **Scheduled graceful relaunch** (§4.3): weekly timer + driver-trend alert
   in memwatch.
12. **Alerting** (§4.6): alert.sh + hookups + heartbeat + one negative test.
13. **Host hygiene pass** (§4.7): linger, comfy-h3, unattended-upgrades, NTP —
    document the exact commands in README's new "Unattended operation" section.

### Phase 2 — correctness

14. **ABLIT PLE identity fix** (#34/#36, start.sh:614-619): key the packed table
    on actual shard identity; until then, mark ABLIT=1 as known-broken in
    start.sh's own output (warn loudly), not just in docs.
15. **Determinism** (#7/#28): ship the smoke-test determinism gate as part of
    every launch verification (post-ready, one probe, **sequential** —
    concurrency perturbs 2,503/2,504 positions even with MTP off, per jschmied,
    so a concurrent gate would flap); document the temperature-0 caveat. The
    end-state on their side is *five* jointly-necessary fixes, and the two
    env-gated ones (`VLLM_QSA_DET_TOPK` + `VLLM_MOE_DET_FINALIZE`) **cannot be
    shipped as env vars alone** — det-topk needs a compiled `.so`, finalize
    needs the FlashInfer autotune cache-key backport or the server dies. What we
    can do now: track whether our image rolls in vllm#55122/#55375, and add the
    env plumbing behind a disabled default so the day the image carries the
    kernels, the flags work.
16. **GDN kernel env + parser verification** (§4.8): flip on `triton` after a
    soak; settle `qwen3_coder` vs `qwen3_xml` with a tool-call round-trip that
    asserts HTTP 200 AND non-empty `completion_tokens` (jschmied: an all-empty
    cell "passes" a naive check because the model can still be inside its
    thinking block when the budget runs out).
17. **`disable_eagle_block_drop` A/B on our image** (§6.1): the vllm#53388
    speculative-config lever that removes MTP's fixed 1,600-token cache-block
    cost per turn — the direct config-level lever for the short-turn half of
    the #20 symptom class. Measure per-turn latency at k=2/3 with and without.

### Phase 3 — performance, after stability is boring

18. MAX_NUM_SEQS=8 as default (#19, measured safe) + PLE prewarm option.
19. Hybrid checkpoint mode (#12, +20% decode) and safe prefix caching (#6,
    needs vllm#50729) — both need careful memory re-budgeting on this host.
    Guidance from jschmied for the checkpoint half: quantize what speculation
    runs per draft token (lm_head complements MTP), not what is latency-bound
    (hyper-connections got *slower* despite halving bytes).
20. Kernel PRs #29/#31 on evidence (boot stalls / speed), checkpoint swap for
    #27 if we change builds. **For #31 specifically**: jschmied records that a
    family-gate-widening "SM121 fix" circulating in the field was retracted
    upstream (sglang#36806) for silently corrupting long-context output —
    validate any kernel we adopt with long-context needle checks, not just
    throughput (§6.3).

### Definition of done — "stable 24/7"

All of the following, demonstrated, not assumed:

- [ ] `docker kill` of the container from an idle host: back online unattended,
      watchdog re-armed, alert received, total outage ≈ one cold start.
- [ ] Forced memory emergency (watchdog stop): alert received, shm cleaned,
      recovery attempted with backoff, no host hang, no ratchet over 3 forced
      emergencies in a day.
- [ ] Host reboot: server serving again with zero human action.
- [ ] Wedge simulation (SIGSTOP the engine process): detected within ~5 probe
      intervals, recovered.
- [ ] 7-day soak on the shipped profile: zero manual interventions, disk usage
      flat, MemAvailable trend flat (no leak), scheduled relaunch executed once
      cleanly, daily heartbeats received.
- [ ] Alert negative-test passed (a deliberately failed alert was seen).

---

## 6. Third-party reference: `jschmied/qwen38-flash-next-gb10`

Found 2026-09-09; **verified directly** the same day (README, `patches/MANIFEST.md`,
`scripts/serve-flashnext.sh`, `tools/main/` incl. `memguard.sh`, `REPRODUCE.md`,
`notes/failure-modes.md`, `notes/mtp-vs-prefix-cache.md`, `notes/what-generalises.md`).
Same model, same hardware (GB10, sm_121, 128 GB unified, aarch64), but the
**nightly-vLLM side** (`0.1.dev20073+g8e685d198` + the #53899 PLE-offload tree +
venv overlays) serving a different checkpoint (RadixArk NVFP4 + their own FP8
mixed-precision overlay). Not drop-in for our pinned image — the patches are cut
against one exact nightly and `apply.sh` refuses another version — but the failure
modes and the launcher hardening transfer directly.

### 6.1 Corrections to our earlier reading (checked against the source)

Three claims carried over from the earlier draft were **wrong or unsupported**:

1. **"Bimodal MTP acceptance, ~4.5 vs ~1.4, 40–55% collapsed turns, per-start
   bias 12–55%" — not in their notes.** What `mtp-vs-prefix-cache.md` actually
   measures is different and more useful:
   - MTP (any Eagle-family drafter) makes the scheduler back off **one full
     prefix-cache block (1,600 tokens) per turn** — `scheduler.py`:
     `if self.use_eagle: last_cache_position -= block_size`. The cost is **fixed,
     not proportional**: 50% of a 2k-turn's cacheable prefix, 5.6% at 16k, 4.1% at 32k.
   - **Break-even ≈68 output tokens** (measured, and predicted from code+depth-curve
     before the measurement): below it, MTP k=2 is *slower per agent turn than no
     speculation*; above it, MTP wins (+24.2% at 400-token turns). **k=1 is strictly
     dominated** — same block cost, half the decode gain — and should be rejected
     by our legality guard as pointless, not just 5–8.
   - What *is* bimodal per server start is **MTP timing** (within-config spread up
     to 1.83×, no-spec stable at 1.01×), which they attribute to acceptance varying
     per start. Consequence for us: never judge an MTP change from one launch —
     already our overnight rule, now with their numbers behind it.
   - **`disable_eagle_block_drop` exists** (`"disable_eagle_block_drop":true` in
     their speculative-config, credited to vllm#53388): it keeps the trailing
     prefix-cache block. That is a config lever for exactly our #20 agentic
     symptom — worth testing on our image before any kernel work.
2. **Illegal MTP k does not cause HTTP 400s — it hard-fails the engine at config
   validation** (`QSA ring capacity must divide the attention block size`). The
   legality set is block-size-derived: `capacity = compress_ratio ×
   cdiv(compress_ratio + n, compress_ratio)`; with our 848 block, legal k is
   0–4 and 9–12, illegal 5–8 **and 13–16** (jschmied widened the ring to reach
   5–8 — their patch — but measured n=6 as *worse than no speculation* on agent
   work, so the widened band is a decode-benchmark lever only). Our guard should
   validate against this formula from the checkpoint's block size, not a
   hardcoded list.
3. **"31,115 chars of thinking" / max-model-len**: real, but their headline
   default is 32768 (the 8192 was a benchmarking choice), and — the part that
   matters for our config — context on this architecture is **admission/memory
   cost, not decode cost** (QSA decode is flat 4k→60k, their strongest
   replicated result). We already ship 262144; the note is for users who
   reduce it.

### 6.2 What jschmied runs that our `main` does not (verified in serve-flashnext.sh)

| Setting | Their rationale (from REPRODUCE.md §4 / serve script comments) | Ours |
|---|---|---|
| `VLLM_GDN_DECODE_KERNEL=triton` | default CUDA kernel **deterministically hangs** the engine at c≈32 with FP8 GDN projections — no error, requests stall | **absent — add (§4.8)** |
| `VLLM_USE_DEEP_GEMM=0` | DeepGEMM gates on family-120 which sm_121 satisfies, then faults (`unspecified launch failure`, vllm#54125) | absent; **check whether our image's default path even selects it** before adding |
| `VLLM_QSA_DET_TOPK=1` + `VLLM_QSA_DET_LIB=<path>/_C_det.so` | deterministic `persistent_topk` (vllm#55122 port); **needs a compiled .so** from their `patches/kernel-det`, not just the env var | absent; **cannot ship as an env var alone** — needs the kernel binary or the image rolling in #55122 |
| `VLLM_MOE_DET_FINALIZE=1` | bit-stable MoE finalize; **requires the FlashInfer autotune cache-key backport** or the server dies with `Invalid gemm2 profile id` (vllm#54945) | absent; same coupling caveat |
| `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1` | unbounded JIT fan-out OOMs the whole box (C2); driver upgrades invalidate the JIT cache | **absent — add (§4.8)** |
| `--cap-add=SYS_PTRACE` | PLE offload `rebuild_cuda_tensor` needs `pidfd_getfd`; yama `ptrace_scope=1` (the OS default) blocks sibling workers; **fails ~10 min in** with only `Failed core proc(s): {}` | already on `main` — and their bare-metal note gives the systemd equivalent we'll need for §4.1: `AmbientCapabilities=CAP_SYS_PTRACE` (a bounding set alone is **not** sufficient) |
| tool-call parser | `qwen3_xml` + `--enable-auto-tool-choice` (every request with tools 400s without them) | we ship `qwen3_coder` — **one is wrong for some image; verify by tool-call round-trip, don't port blindly** |
| sha256 manifest at download (`hfget.sh`, `SOURCE.json`) | A1; plus tree-API pagination (50/page) and provenance recovery via `lfs.oid` intersection | **absent — add (§4.8)** |

Their memguard (`tools/main/memguard.sh`) is a **second opinion worth having when
we touch memwatch triggers**: it stops a unit only on *real thrash* — PSI
`full avg10 > 60%` **AND** MemAvailable < 3 GiB, 3 consecutive samples. Their v1
used PSI `some` alone and **killed a healthy load** (paging to swap raises `some`
without any stall). Two lessons for our floors: (a) PSI `full` is the stall
signal, `some` is not; (b) a trigger must be debounced against healthy-but-busy
memory phases — our MemFree/MemAvailable gating already embodies (b); their
combined check is a cheaper expression of the same idea if we ever add swap.

### 6.3 Also verified, previously unrecorded

- **`patches/MANIFEST.md` is a process lesson for our `files/patch_*` generators**:
  their patched-venv snapshot mechanism drifted (4 of 5 copies stale, one patch
  missing entirely) and "a patched venv with a hand-maintained snapshot is a
  machine nobody can rebuild." We already generate patches (files/patch_*.py) —
  the transferable rules are: an idempotent apply that reports
  applied/already/FAILED, post-apply assertions for the two fixes that fail
  *silently*, and a grep for `PROBE|_dq_calls|# temporary` before trusting any
  benchmark. Their apply.sh asserts exactly those; our start.sh patch
  generation has no equivalent post-asserts.
- **The retracted "SM121 QSA kernel-guard fix" warning (REPRODUCE.md)**: a
  family-gate-widening fix circulating in DGX Spark repos was **retracted
  upstream (sglang#36806)** — it corrupts output at long context (1/4 wrong at
  120k, 4/4 at 210k, HTTP 200 throughout). **Directly relevant to upstream PR
  #31 (opt-in GB10 kernels) and anything we port from it: family-gate widening
  is exactly the shape of that PR's changes. Test long-context output, not
  just throughput, before adopting any of it.**
- **The "invariant to every config" triage rule** (failure-modes.md, their most
  useful line): when output is fluent but wrong and *nothing you change alters
  it*, stop varying configuration — invariance is evidence of corrupt **data**.
  Four distinct causes produce "loads but wrong output" (corrupt shard A1;
  shadowed scale A2; unrecognized quant_algo A2b; SM100-only kernels emitting
  NaN on SM121 A3 — `!!!!`-forever is the NaN signature). Distinguish before
  eliminating hypotheses. Worth a README troubleshooting table of our own.
- **Two measurement traps that would invalidate our own soak criteria (§5)**:
  an all-empty output cell "passes" a determinism check (assert non-empty before
  comparing: the model can still be inside its thinking block when the budget
  runs out), and `max_tokens` is a ceiling, not a target (a "130-token turn"
  that emitted 45 tokens is not comparable). Our smoke test and the §4.2 probe
  must assert `completion_tokens > 0`.
- **Version pins are load-bearing (REPRODUCE.md §0)**: their FlashInfer 0.6.18
  pin story (0.6.18 dropped SM121a JIT cubins → runtime JIT → the C2 box-OOM)
  is the pattern to check whenever our image bumps a pinned dependency: does the
  new version silently drop a prebuilt path we rely on?

### 6.4 What is deliberately NOT portable

- Their venv overlay stack (det-topk `.so`, MoE finalize, autotune cache-key,
  PLE semaphore reset, ring-widening, FP8_PB_WO dispatch line, fp8-KV QSA
  patch): all cut against `0.1.dev20073+g8e685d198`; `apply.sh` refuses other
  versions because upstream has since renamed the model package and refactored
  `ple_layer.py`. None of it can cross into our container image without the
  image itself carrying it.
- Their checkpoint work (FP8 dense projections, FP8 lm_head, pruned variants):
  different weights from ours; the *mechanisms* (quantize what the drafter runs
  per-token; don't quantize latency-bound tiny GEMMs) transfer as guidance for
  our Phase 3, the numbers do not.
- Their `--max-num-seqs 16` and GMU 0.90: their box serves a different, lighter
  configuration (32k context, MTP k=2, their checkpoint); our host-side budget
  derivation at HOST_RESERVE_GIB=26 is stricter, measured on our hardware, and
  should not move to match theirs.

### 6.5 Verdict

Higher value than the upstream issue tracker for the *failure-mode* half of our
24/7 problem: it independently hit and documented the silent-wedge class (GDN
stall, PLE semaphore one-step-ahead, yama/ptrace sibling failure), the
corrupt-data class (A1/A2b with working offline checks), and the
config-legality class (MTP k formula, async-scheduling interaction) — plus the
launcher hygiene (JIT bounds, PSI-full trigger design, SYS_PTRACE systemd
equivalent) our §4/§5 items need. Its determinism work is ahead of upstream's
issue tracker (five jointly-necessary fixes, vllm#55375 merged 2026-09-05, the
rest as venv overlays we cannot ship) and sets the expectation for what "the
image rolls it in" will eventually mean. Concurrency determinism remains open
even there (batch composition perturbs 2,503/2,504 positions with MTP fully
off) — so our smoke-test determinism gate must run **sequentially** or it will
flap.

---

## 7. Caveats — what "fixing" here means

- **Launcher-side fixes** (start.sh, stop.sh, download.sh, memwatch, supervisor,
  systemd, alerting): fully ours, ship whenever.
- **Kernel-level fixes** (#7/#28 determinism, #6 prefix caching, #29/#31
  kernels): live in the vLLM image or upstream. Until the image rolls them in,
  our kit can only carry env knobs, guards, documentation, and detection.
- **Checkpoint-level facts** (#34 PLE shards, #27 GDN quant): fixable by us
  (#34 is our packed-table keying) or require a checkpoint swap (#27).
- **The two clusters behind most user-visible "it stopped" reports**: default
  sampling config (fixed by defaults + docs) and the wedge class (fixed by
  §4.8 prevention + §4.2 detection). Neither is a weight bug.

Maintainers' rule for this document: when a row above is fixed, move it to
§2.1 with the commit hash and delete it from the plan; keep the file the
single source of truth rather than spawning new review documents per session.
