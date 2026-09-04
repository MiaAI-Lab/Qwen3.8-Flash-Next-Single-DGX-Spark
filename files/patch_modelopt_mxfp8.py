#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Patch modelopt.py so MXFP8 linear layers the FlashInfer kernel cannot run
fall back to the BF16 emulation kernel.

FlashInfer's mm_mxfp8 (FlashInferCutlassMxfp8LinearKernel) only accepts weights
whose per-partition dims satisfy N >= 128, N % 32 == 0, K >= 128, K % 32 == 0.
Measured on GB10 (sm_121) with vllm/vllm-openai:qwen38-flash-next; the limits
are independent of the token count M:

    N=128/160/192/224/256/288  ok      N=136/144/152/176/208/240  ValueError
    K=128/160/192/1152/2560    ok      K=2152/4304                AssertionError

Two shapes in RadixArk/Qwen3.8-Flash-Next-NVFP4 violate this:

    language_model.layers.*.linear_attn.in_proj_a/b   [48, 2560]    N < 128
    visual.blocks.*.mlp.linear_fc1                    [4304, 1152]  N % 32 == 16

The first is fatal at engine start ("mm_mxfp8 requires N >= 128, got N=48"),
the second at the vision profile run ("Problem size is not supported").

Rather than special-casing shapes, ModelOptMxFp8LinearMethod.create_weights now
inspects the real per-partition (N, K) — which is what the kernel sees, after
the TP split — and swaps its kernel for EmulationMxfp8LinearKernel when the
native path cannot take them. Emulation dequantizes MXFP8 -> BF16 once at load
time, so those layers then run as plain BF16 linears (in_proj_a/b are 48x2560:
~17 MB of extra BF16 weight across all 36 linear-attention layers).

The visual.* prefix rule is kept on top of the shape check: it is the verified
multimodal configuration, and dequantizing only visual.* (rather than every
MXFP8 layer) is what avoids the global BF16 dequant OOM.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "modelopt_patched.py.orig")
OUT = os.path.join(HERE, "modelopt_patched.py")

EMULATION_CLASS = '''

def _mxfp8_emulation_kernel():
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )

    return EmulationMxfp8LinearKernel(Mxfp8LinearLayerConfig())


class ModelOptMxFp8EmulationLinearMethod(ModelOptMxFp8LinearMethod):
    """MXFP8 linear layers forced to BF16 emulation (vision-only dispatch)."""

    def __init__(self, quant_config: ModelOptMxFp8Config) -> None:
        self.quant_config = quant_config
        if not self.quant_config.is_checkpoint_mxfp8_serialized:
            raise ValueError(
                "MXFP8 currently only supports serialized checkpoints. "
                "Dynamic quantization is not supported."
            )

        self.kernel = _mxfp8_emulation_kernel()


def _mxfp8_use_vision_emulation(prefix: str) -> bool:
    return ".visual." in prefix or prefix.startswith("visual.")


def _mxfp8_native_kernel_supports(n: int, k: int) -> bool:
    """Can FlashInfer mm_mxfp8 run an [N, K] weight?

    Measured on GB10/sm_121: N >= 128 and N % 32 == 0 (else "Problem size is
    not supported"), K >= 128 and K % 32 == 0 (asserted by the kernel). Both
    are token-count independent.
    """
    return n >= 128 and n % 32 == 0 and k >= 128 and k % MXFP8_BLOCK_SIZE == 0
'''

# Inserted into ModelOptMxFp8LinearMethod.create_weights, where the dims are the
# post-TP-split ones the kernel will actually be handed.
SHAPE_FALLBACK = '''
        # Downgrade to BF16 emulation for shapes the native MXFP8 GEMM rejects
        # (e.g. linear_attn.in_proj_a/b, [48, 2560] -> N < 128). Each layer gets
        # its own ModelOptMxFp8LinearMethod, so this is per-layer.
        if not _mxfp8_native_kernel_supports(
            output_size_per_partition, input_size_per_partition
        ):
            from vllm.model_executor.kernels.linear.mxfp8.emulation import (
                EmulationMxfp8LinearKernel,
            )

            if not isinstance(self.kernel, EmulationMxfp8LinearKernel):
                logger.warning_once(
                    "MXFP8 layer [N=%d, K=%d] is not supported by %s "
                    "(needs N,K >= 128 and divisible by 32); falling back to "
                    "BF16 emulation for this shape.",
                    output_size_per_partition,
                    input_size_per_partition,
                    type(self.kernel).__name__,
                )
                self.kernel = _mxfp8_emulation_kernel()
'''


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise AssertionError(f"modelopt: {what} anchor missing (count={src.count(old)})")
    return src.replace(old, new)


def patch() -> None:
    src = open(ORIG).read()

    anchor = (
        "        return self.kernel.apply_weights(layer, x, bias)\n\n\n"
        "class ModelOptMxFp8FusedMoE(FusedMoEMethodBase):\n"
    )
    src = _replace_once(
        src,
        anchor,
        anchor.replace(
            "\n\nclass ModelOptMxFp8FusedMoE",
            EMULATION_CLASS + "\n\nclass ModelOptMxFp8FusedMoE",
        ),
        "EmulationClass",
    )

    src = _replace_once(
        src,
        "        layer.output_size_per_partition = output_size_per_partition\n\n"
        "        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:\n",
        "        layer.output_size_per_partition = output_size_per_partition\n"
        + SHAPE_FALLBACK
        + "\n        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:\n",
        "create_weights shape fallback",
    )

    src = _replace_once(
        src,
        '            if quant_algo == "MXFP8":\n'
        "                return ModelOptMxFp8LinearMethod(self.mxfp8_config)\n",
        '            if quant_algo == "MXFP8":\n'
        "                if _mxfp8_use_vision_emulation(prefix):\n"
        "                    return ModelOptMxFp8EmulationLinearMethod(self.mxfp8_config)\n"
        "                return ModelOptMxFp8LinearMethod(self.mxfp8_config)\n",
        "MXFP8 dispatch",
    )

    open(OUT, "w").write(src)
    print("ok", OUT)


if __name__ == "__main__":
    if not os.path.isfile(ORIG):
        print(f"ERROR: missing {ORIG}", file=sys.stderr)
        sys.exit(1)
    patch()
