# SPDX-License-Identifier: Apache-2.0
"""Opt-in FlashInfer GDN prefill on GB10; no automatic backend change.

Applied to an exact image source in a disposable CPU container. The installed
FlashInfer must provide its SM12x CuTe DSL kernel. Numeric/state and serving
tests are recorded separately; this patch never changes the decode backend.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
EXPECTED_SHA256 = "81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0"
ANCHOR = '''        supports_flashinfer = True
        supports_cutedsl = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
'''
REPLACEMENT = '''        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        backend == "flashinfer"
        and current_platform.is_device_capability((12, 1))
        and head_k_dim == 128
        and getattr(vllm_config.model_config.hf_text_config, "linear_value_head_dim", None) == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        # GB10 is deliberately opt-in. FlashInfer 0.6.17 includes a native
        # SM12x path; the existing adapter supplies FP32 state/gates and the
        # FlashInfer wrapper converts cu_seqlens to int64. Do not enable the
        # separate SM10x-only in-tree CuteDSL backend here.
        from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm120

        if chunk_gated_delta_rule_sm120 is None:
            raise RuntimeError("FlashInfer SM12x GDN prefill kernel is unavailable")
        supports_flashinfer = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
'''


def patch(source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_SHA256 or source.count(ANCHOR) != 1:
        raise RuntimeError(f"GDN source drift ({digest}); refusing unvalidated patch")
    result = source.replace(ANCHOR, REPLACEMENT)
    ast.parse(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(patch(args.source.read_text()))
    print(json.dumps({"patch": "gdn_sm121", "source_sha256": EXPECTED_SHA256,
                      "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
