# Third-party kernel attribution

The Triton W8A16 kernel in `files/spark_mtp_fp8_head.py` is adapted from
**Gabriel Olympie (@gabrielolympie)** and contributors' work in
[sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120/tree/67d2f9234fa45ae1339f0d53cd37cb695e9c6493),
pinned at commit `67d2f9234fa45ae1339f0d53cd37cb695e9c6493`:

- `patches/0004-sm120-lowm-triton-gemm.patch`
- `patches/0005-sm120-fp8-weight-only.patch`

Those kernels carry `SPDX-License-Identifier: Apache-2.0`. The adapted helper
retains that identifier; a copy of the license is provided in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). The recipe's existing
top-level license is unchanged.

Raymond Lucke's adaptation restricts the optimization to vLLM's reduced MTP
draft head, uses module-owned nonpersistent buffers instead of a global pointer
cache, and tunes a split-free N=32/K=256/stages=3 tactic for SM121/GB10. Target
weights and rejection sampling are not changed.

The investigation into FlashInfer GDN prefill was also inspired by Gabriel's
prefill-state work. That resolver patch uses vLLM's existing FlashInfer adapter;
it is not a transplant of the SGLang runtime. Thanks to Gabriel for sharing
the original optimization work and measurements.
