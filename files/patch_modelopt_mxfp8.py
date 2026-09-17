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

Second, unrelated patch in the same file: bridge NVIDIA's MTP quantized_layers
declaration to the module prefixes vLLM actually queries, and add a dispatch
branch for MoE-routed FP8_PB_WO on RoutedExperts.

NVIDIA's hf_quant_config.json declares the MTP draft model's MoE experts as
mtp.layers.0.mlp.experts (local index -- MTP has exactly one layer of its
own), quant_algo "FP8_BLOCK_SCALES", group_size 128. mtp.py's
remap_weight_names() renumbers the checkpoint's weight tensors from that local
index to the global one the mounted module sits at (e.g. mtp.layers.48...,
after the main model's own layers), but nothing renumbers quantized_layers to
match, so _resolve_quant_algo("mtp.layers.48.mlp.experts") never finds the
declaration and get_quant_method falls through to None: the layer is built
unquantized, and loading its checkpoint's weight_scale_inv then fails with
"has no parameter 'w2_weight_scale_inv'".

Separately, even a correct prefix match would still return None: FP8 is the
only MoE-routed FP8 variant get_quant_method knows about (via
ModelOptFp8MoEMethod, which assumes per-tensor static scale), and it has no
branch for whatever this checkpoint's block-scaled variant resolves to. Debug
logging during bring-up showed _resolve_quant_algo actually returns
"FP8_PB_WO" here, not the raw checkpoint string "FP8_BLOCK_SCALES" -- ModelOpt
config parsing normalizes it internally. FP8_PB_WO already has a Linear
handler in this file (ModelOptFp8PbWoLinearMethod, _WEIGHT_BLOCK_SIZE =
(128, 128)) but no RoutedExperts/MoE counterpart. Verified against the actual
checkpoint that the block size matches: for mtp.layers.0.mlp.experts.0.
down_proj, weight is F8_E4M3 [2560, 640] and weight_scale_inv is BF16
[20, 5] -- exactly [2560/128, 640/128]. vLLM's own (non-ModelOpt)
Fp8MoEMethod/Fp8Config already implement that layout; DeepSeek-V3 checkpoints
use the same class. This patch routes FP8_PB_WO MoE experts to it rather than
adding a new ModelOpt-native MoE method.
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

# Inserted into _quantized_layer_prefix_candidates, before its final return.
MTP_PREFIX_BRIDGE = '''
        # NVIDIA's quantized_layers declares the MTP draft layer at its own
        # local index ("mtp.layers.0...."); the mounted module sits at a
        # global index after the main model's layers (e.g. "mtp.layers.48
        # ..."). remap_weight_names() in mtp.py renumbers the weight tensors
        # to match, but never touches quantized_layers, so bridge it here.
        # MTP has exactly one local layer, so any numbered mtp layer prefix
        # may also resolve against index 0.
        if prefix.startswith("mtp.layers."):
            rest = prefix[len("mtp.layers.") :]
            dot = rest.find(".")
            if dot != -1 and rest[:dot].isdigit() and rest[:dot] != "0":
                candidates.append("mtp.layers.0." + rest[dot + 1 :])
'''

# Inserted into get_quant_method's RoutedExperts branch, after the FP8 case.
FP8_PB_WO_MOE_DISPATCH = '''
            if quant_algo == "FP8_PB_WO":
                # NVIDIA's hf_quant_config.json declares this layer as
                # "FP8_BLOCK_SCALES" (group_size 128); ModelOpt's own config
                # parsing normalizes that to its internal "FP8_PB_WO" name --
                # confirmed live (debug logging during bring-up) that
                # quant_algo actually arrives here as "FP8_PB_WO", not the
                # raw checkpoint string. ModelOptFp8PbWoLinearMethod already
                # handles this for Linear layers with _WEIGHT_BLOCK_SIZE =
                # (128, 128) -- same 128x128 block size verified independently
                # against this checkpoint's tensors (down_proj weight
                # [2560, 640] F8_E4M3, weight_scale_inv [20, 5] BF16). There is
                # no equivalent RoutedExperts method in this vLLM build, so
                # route to vLLM's native (non-ModelOpt) block-scaled
                # Fp8MoEMethod instead -- the same class DeepSeek-V3
                # checkpoints use for identical weight/weight_scale_inv
                # layout.
                from vllm.model_executor.layers.quantization.fp8 import (
                    Fp8Config,
                    Fp8MoEMethod,
                )

                return Fp8MoEMethod(
                    Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        weight_block_size=[128, 128],
                    ),
                    layer,
                )
'''

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

    src = _replace_once(
        src,
        "                \"language_model.model.\" + prefix[len(\"model.language_model.\") :]\n"
        "            )\n"
        "\n"
        "        return tuple(dict.fromkeys(candidates))\n",
        "                \"language_model.model.\" + prefix[len(\"model.language_model.\") :]\n"
        "            )\n"
        + MTP_PREFIX_BRIDGE +
        "\n        return tuple(dict.fromkeys(candidates))\n",
        "MTP prefix bridge",
    )

    src = _replace_once(
        src,
        "        if isinstance(layer, RoutedExperts):\n"
        "            if quant_algo == \"FP8\":\n"
        "                return ModelOptFp8MoEMethod(\n"
        "                    quant_config=self.fp8_config,\n"
        "                    moe_config=layer.moe_config,\n"
        "                )\n"
        "            if quant_algo == \"NVFP4\":\n",
        "        if isinstance(layer, RoutedExperts):\n"
        "            if quant_algo == \"FP8\":\n"
        "                return ModelOptFp8MoEMethod(\n"
        "                    quant_config=self.fp8_config,\n"
        "                    moe_config=layer.moe_config,\n"
        "                )\n"
        + FP8_PB_WO_MOE_DISPATCH +
        "            if quant_algo == \"NVFP4\":\n",
        "FP8_PB_WO MoE dispatch",
    )

    open(OUT, "w").write(src)
    print("ok", OUT)


if __name__ == "__main__":
    if not os.path.isfile(ORIG):
        print(f"ERROR: missing {ORIG}", file=sys.stderr)
        sys.exit(1)
    patch()
