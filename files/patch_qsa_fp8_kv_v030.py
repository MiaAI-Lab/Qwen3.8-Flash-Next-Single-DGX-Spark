#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Backport vllm-project/vllm#55557 (Qwen4Exp fp8_e4m3 main KV cache on the QSA
path) onto the vLLM 0.30.0 nvidia/qsa.py and nvidia/ops/qsa.py.

Delete this file, its test and its overlay in start.sh once the image moves to
vLLM 0.31 or later, which ships #55557 natively.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = {
    "models/qwen4_exp/nvidia/qsa.py": "qsa.py",
    "models/qwen4_exp/nvidia/ops/qsa.py": "ops/qsa.py",
}
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

UPSTREAM_DIFF = r'''diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa.py b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
index d3986ba38ae0..6d0df0e741b7 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -4,15 +4,24 @@
 
 from __future__ import annotations
 
+from functools import lru_cache
+
 import torch
 
 from vllm.model_executor.warmup.jit_warmup_triton_helper import (
     TritonWarmupTensor,
     triton_scalar_specialization_rep,
 )
+from vllm.platforms import current_platform
 from vllm.triton_utils import HAS_TRITON, tl, triton
 
 
+@lru_cache(maxsize=1)
+def _is_sm120() -> bool:
+    """True on sm_120 (RTX PRO 6000 Blackwell): selects the sm_120 tuning table."""
+    return current_platform.get_device_capability() == (12, 0)
+
+
 @triton.jit(do_not_specialize=["num_rows", "num_requests"])
 def _qsa_sparse_paged_gqa_splitk_kernel(
     q_ptr,
@@ -24,6 +33,8 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     partial_output_ptr,
     partial_lse_ptr,
     output_ptr,
+    softmax_scale,
+    output_scale,
     output_gate_ptr,
     stride_q_row,
     stride_q_head,
@@ -52,6 +63,7 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     NUM_TILES: tl.constexpr,
     BLOCK_M: tl.constexpr,
     BLOCK_N: tl.constexpr,
+    IS_FP8: tl.constexpr,
 ) -> None:
     row = tl.program_id(0)
     kv_head = tl.program_id(1)
@@ -81,7 +93,10 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
     normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
     accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
-    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
+    # softmax_scale is the host-side attention scale (1/sqrt(head_dim), with the
+    # fp8 K dequant scale already pre-multiplied in); convert to log2 units once
+    # here for the exp2-based online softmax.
+    score_scale = softmax_scale * 1.4426950408889634
 
     tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))
 
@@ -129,15 +144,26 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
             mask=valid[:, None],
             other=0.0,
         )
+        if IS_FP8:
+            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
+            # measured slower here and less accurate).
+            keys = keys.to(query.dtype)
         scores = tl.dot(query, keys)
-        # Scaling scores avoids re-quantizing a scaled query to BF16.
-        scores *= softmax_scale_log2
+        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
+        # caches the K dequant scale is already folded into softmax_scale on the
+        # host.
+        scores *= score_scale
         scores = tl.where(valid[None, :], scores, -1.0e20)
         next_max = tl.maximum(max_value, tl.max(scores, axis=1))
         alpha = tl.math.exp2(max_value - next_max)
         probabilities = tl.where(
             valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
         )
+        if IS_FP8:
+            # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
+            # softmax) so fp16 has the range, its wider mantissa is more
+            # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
+            values = values.to(tl.float16)
         accumulator = tl.dot(
             probabilities.to(values.dtype),
             values,
@@ -147,9 +173,14 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
         max_value = next_max
 
     has_values = normalizer > 0
+    # Fold the fp8 V dequant scale (output_scale, 1.0 for bf16) into a per-row
+    # reciprocal normalizer, so the output is a per-row multiply rather than a
+    # HEAD_DIM-wide scale. The split-K LSE below keeps the unscaled normalizer,
+    # so the merge stays correct.
+    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
     normalized_output = tl.where(
         has_values[:, None],
-        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
+        accumulator * inv_normalizer[:, None],
         0.0,
     )
     output_mask = head_offsets[:, None] < GROUP_SIZE
@@ -439,18 +470,65 @@ def _compress_qsa_groups_kernel(
     )
 
 
+def _select_sm120_config(
+    base_programs: int, use_prefill_config: bool, is_fp8: bool
+) -> tuple[int, int, int]:
+    """(block_n, target_splits, num_warps) retuned on sm_120 (RTX PRO 6000
+    Blackwell), split by cache dtype: bf16 and fp8 favour different configs on
+    sm_120, most visibly on the large-prefill region. Each entry is the fastest
+    config that stays correct on its own path; the main win is more warps."""
+    if is_fp8:
+        if base_programs > 2048:
+            return (32, 2, 1) if use_prefill_config else (64, 1, 2)
+        if base_programs <= 24:
+            return 64, 64, 8
+        if base_programs <= 32:
+            return 128, 8, 4
+        if base_programs <= 64:
+            return 64, 8, 4
+        if base_programs <= 128:
+            return 32, 4, 4
+        if base_programs <= 256:
+            return 128, 4, 4
+        if base_programs <= 512:
+            return 64, 4, 4
+        return 32, 2, 1
+    # bf16 K/V cache.
+    if base_programs > 2048:
+        return 64, 2, 2
+    if base_programs <= 24:
+        return 64, 64, 8
+    if base_programs <= 64:
+        return 64, 8, 8
+    if base_programs <= 128:
+        return 32, 8, 8
+    if base_programs <= 256:
+        return 32, 8, 1
+    if base_programs <= 512:
+        return 64, 4, 2
+    return 32, 4, 1
+
+
 def _select_config(
-    num_rows: int, num_kv_heads: int, use_prefill_config: bool, num_columns: int
+    num_rows: int,
+    num_kv_heads: int,
+    use_prefill_config: bool,
+    num_columns: int,
+    is_fp8: bool = False,
 ) -> tuple[int, int, int, int]:
     """Select (block_n, num_warps, num_tiles, num_splits) for the kernel.
 
-    Tuned on GB300 for the Qwen3.8-Flash-Next TP1/TP2/TP4 shapes, keyed on
-    base_programs = num_rows * num_kv_heads. The bp > 2048 region splits on
-    use_prefill_config (capture-stable: at FULL-graph capture max_query_len is the
-    uniform decode/verify length).
+    Keyed on base_programs = num_rows * num_kv_heads. The bp > 2048 region splits
+    on use_prefill_config (capture-stable: at FULL-graph capture max_query_len is
+    the uniform decode/verify length). This default table was tuned on GB300;
+    sm_120 (RTX PRO 6000 Blackwell) dispatches to _select_sm120_config instead.
     """
     base_programs = num_rows * num_kv_heads
-    if base_programs > 2048:
+    if _is_sm120():
+        BLOCK_N, target_splits, num_warps = _select_sm120_config(
+            base_programs, use_prefill_config, is_fp8
+        )
+    elif base_programs > 2048:
         BLOCK_N, target_splits, num_warps = (
             (32, 1, 1) if use_prefill_config else (64, 1, 2)
         )
@@ -483,10 +561,17 @@ def qsa_sparse_paged_attention(
     token_to_req: torch.Tensor,
     use_prefill_config: bool,
     out: torch.Tensor | None = None,
+    k_scale: float | None = None,
+    v_scale: float | None = None,
     *,
     output_gate: torch.Tensor,
 ) -> torch.Tensor:
-    """Run sparse GQA directly over paged BF16 K/V caches.
+    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.
+
+    With fp8 caches, k_scale/v_scale are the layer's per-tensor dequant scales
+    as host floats (e.g. layer._k_scale_float/_v_scale_float): k_scale is
+    pre-multiplied into the softmax scale and v_scale becomes the kernel's
+    output scale, so the kernel needs no device scale buffers.
 
     logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
     with the trailing column holding each row's valid-entry count (written by
@@ -510,7 +595,19 @@ def qsa_sparse_paged_attention(
         raise ValueError("QSA sparse attention requires valid grouped-query heads")
     head_dim = q.shape[2]
     assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
-    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
+    assert q.dtype == torch.bfloat16
+    assert k_cache.dtype == v_cache.dtype
+    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
+    if is_fp8:
+        assert k_scale is not None and v_scale is not None
+        # Host pre-multiply: fold the K dequant scale into the attention scale
+        # and pass V's dequant scale as the kernel's output scale.
+        softmax_scale = (head_dim**-0.5) * float(k_scale)
+        output_scale = float(v_scale)
+    else:
+        assert k_cache.dtype == torch.bfloat16
+        softmax_scale = head_dim**-0.5
+        output_scale = 1.0
     assert logical_indices.dtype == block_table.dtype == torch.int32
     assert token_to_req.dtype == torch.int32
     assert q.device == k_cache.device == v_cache.device
@@ -535,7 +632,7 @@ def qsa_sparse_paged_attention(
     block_m = triton.next_power_of_2(group_size)
     selection_width = logical_indices.shape[1] - 1  # trailing column is the count
     block_n, partial_warps, num_tiles, num_splits = _select_config(
-        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width
+        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width, is_fp8
     )
 
     # Split=1 writes output directly and compiles out all workspace accesses.
@@ -565,6 +662,8 @@ def qsa_sparse_paged_attention(
         partial_output,
         partial_lse,
         out,
+        softmax_scale,
+        output_scale,
         output_gate_view,
         q.stride(0),
         q.stride(1),
@@ -593,6 +692,7 @@ def qsa_sparse_paged_attention(
         NUM_TILES=num_tiles,
         BLOCK_M=block_m,
         BLOCK_N=block_n,
+        IS_FP8=is_fp8,
         num_warps=partial_warps,
         num_stages=2,
     )
@@ -630,13 +730,18 @@ def warmup_qsa_sparse_paged_attention(
 
     head_dim = kv_cache.shape[-1] // 2
     key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
+    # An fp8 cache is allocated as uint8 and viewed as e4m3 at attention time.
+    is_fp8 = kv_cache.dtype == torch.uint8
+    cache_dtype = torch.float8_e4m3fn if is_fp8 else key_cache.dtype
     num_kv_heads = key_cache.shape[2]
     group_size = num_query_heads // num_kv_heads
     block_m = triton.next_power_of_2(group_size)
 
     # Every config the dispatch can pick for this group size.
     profiles = {
-        _select_config(num_rows, num_kv_heads, use_prefill_config, selection_width)
+        _select_config(
+            num_rows, num_kv_heads, use_prefill_config, selection_width, is_fp8
+        )
         for num_rows in range(1, 8193)
         for use_prefill_config in (False, True)
     }
@@ -650,12 +755,12 @@ def warmup_qsa_sparse_paged_attention(
         torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
     )
     k_cache_ptr = TritonWarmupTensor(
-        key_cache.dtype,
+        cache_dtype,
         shape=tuple(key_cache.shape),
         strides=tuple(key_cache.stride()),
     )
     v_cache_ptr = TritonWarmupTensor(
-        value_cache.dtype,
+        cache_dtype,
         shape=tuple(value_cache.shape),
         strides=tuple(value_cache.stride()),
     )
@@ -701,6 +806,8 @@ def warmup_qsa_sparse_paged_attention(
             partial_output_ptr,
             partial_lse_ptr,
             output_ptr,
+            1.0,
+            1.0,
             output_gate_ptr,
             row_stride,
             head_stride,
@@ -729,6 +836,7 @@ def warmup_qsa_sparse_paged_attention(
             NUM_TILES=num_tiles,
             BLOCK_M=block_m,
             BLOCK_N=block_n,
+            IS_FP8=is_fp8,
             num_warps=warps,
             num_stages=2,
             grid=(num_rows, num_kv_heads, num_splits),
diff --git a/vllm/models/qwen4_exp/nvidia/qsa.py b/vllm/models/qwen4_exp/nvidia/qsa.py
index 1c2808a6c47f..e480142a2218 100644
--- a/vllm/models/qwen4_exp/nvidia/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/qsa.py
@@ -24,6 +24,7 @@
 from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
 from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
 from vllm.platforms import current_platform
+from vllm.platforms.interface import DeviceCapability
 from vllm.transformers_utils.configs.qwen4_exp import (
     Qwen4ExpTextConfig,
 )
@@ -64,7 +65,38 @@ class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
     """FullAttentionSpec backend used by the merged QSA owner."""
 
     supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
-    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]
+    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
+    # with the layer's per-tensor scales and dequantized on load inside the QSA
+    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
+    # does not apply (see supports_kv_cache_dtype and the impl constructor).
+    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
+        "auto",
+        "bfloat16",
+        "fp8",
+        "fp8_e4m3",
+    ]
+
+    @classmethod
+    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
+        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes
+
+    @classmethod
+    def supports_combination(
+        cls,
+        head_size: int,
+        dtype: torch.dtype,
+        kv_cache_dtype: CacheDType | None,
+        block_size: int | None,
+        use_mla: bool,
+        has_sink: bool,
+        use_sparse: bool,
+        use_mm_prefix: bool,
+        device_capability: DeviceCapability,
+    ) -> str | None:
+        # QSA dequantizes the fp8 KV in its own Triton kernel and never runs
+        # flash-attn over the quantized cache, so the parent's fp8-KV rejection
+        # does not apply and every combination it is handed is accepted here.
+        return None
 
     @staticmethod
     def get_name() -> str:
@@ -98,16 +130,53 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
     supports_dcp: bool = False
     supports_pcp: bool = False
 
-    def __init__(self, *args, **kwargs) -> None:
-        super().__init__(*args, **kwargs)
+    def __init__(
+        self,
+        num_heads: int,
+        head_size: int,
+        scale: float,
+        num_kv_heads: int,
+        alibi_slopes: list[float] | None,
+        sliding_window: int | None,
+        kv_cache_dtype: str,
+        logits_soft_cap: float | None = None,
+        attn_type: AttentionType = AttentionType.DECODER,
+        kv_sharing_target_layer_name: str | None = None,
+        sinks: torch.Tensor | None = None,
+    ) -> None:
+        # The parent constructor probes flash-attn for quantized-KV support and
+        # raises where it is unavailable (sm120), but QSA dequantizes fp8 inside
+        # its own Triton kernel and never runs flash-attn over the cache. Hand
+        # the parent "auto" for that probe and restore the real dtype afterwards:
+        # the parent only uses it there, and do_kv_cache_update reads the
+        # attribute at call time.
+        real_kv_cache_dtype = kv_cache_dtype
+        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
+            kv_cache_dtype = "auto"
+        super().__init__(
+            num_heads,
+            head_size,
+            scale,
+            num_kv_heads,
+            alibi_slopes,
+            sliding_window,
+            kv_cache_dtype,
+            logits_soft_cap,
+            attn_type,
+            kv_sharing_target_layer_name,
+            sinks,
+        )
+        self.kv_cache_dtype = real_kv_cache_dtype
         if not is_flash_attn_varlen_func_available():
             raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
         if self.dcp_world_size != 1:
             raise NotImplementedError(
                 "Qwen4Exp QSA does not support decode context parallelism"
             )
-        if self.kv_cache_dtype not in ("auto", "bfloat16"):
-            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
+        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
+            raise NotImplementedError(
+                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
+            )
         self.supports_quant_query_input = False
 
     def forward_qsa(
@@ -144,8 +213,23 @@ def forward_qsa(
         logical_indices = topk_buffer[:num_tokens]
         token_to_req = token_to_req[:num_tokens]
         key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
-        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
-            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")
+        k_scale = v_scale = None
+        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
+            # The cache is allocated as uint8; reinterpret the e4m3 bytes
+            # (same itemsize, so shape and strides are preserved).
+            key_cache = key_cache.view(torch.float8_e4m3fn)
+            value_cache = value_cache.view(torch.float8_e4m3fn)
+            # Host-side per-tensor dequant scales (Python floats), as used by
+            # other host-scale backends; folded into the kernel's scales.
+            k_scale = layer._k_scale_float
+            v_scale = layer._v_scale_float
+        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
+            torch.bfloat16,
+            torch.float8_e4m3fn,
+        ):
+            raise NotImplementedError(
+                "Qwen4Exp QSA requires BF16 Q and BF16 or FP8-e4m3 K/V"
+            )
 
         from .ops.qsa import qsa_sparse_paged_attention
 
@@ -158,6 +242,8 @@ def forward_qsa(
             token_to_req,
             use_prefill_config,
             output[:num_tokens],
+            k_scale=k_scale,
+            v_scale=v_scale,
             output_gate=output_gate[:num_tokens],
         )
         return output
@@ -185,8 +271,15 @@ def __init__(
             raise ValueError("Qwen4Exp QSA requires a paged KV cache")
         if model_config.dtype != torch.bfloat16:
             raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
-        if cache_config.cache_dtype not in ("auto", "bfloat16"):
-            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
+        if cache_config.cache_dtype not in (
+            "auto",
+            "bfloat16",
+            "fp8",
+            "fp8_e4m3",
+        ):
+            raise NotImplementedError(
+                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
+            )
         if getattr(quant_config, "kv_cache_scheme", None) is not None:
             raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
         parallel_config = vllm_config.parallel_config
@@ -285,8 +378,10 @@ def __init__(
         self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
             self.kv_cache_dtype, model_config
         )
-        if self.kv_cache_torch_dtype != torch.bfloat16:
-            raise NotImplementedError("Qwen4Exp QSA requires BF16 cache storage")
+        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
+            raise NotImplementedError(
+                "Qwen4Exp QSA requires BF16 or FP8-e4m3 (uint8) cache storage"
+            )
         self.kv_sharing_target_layer_name = None
         self.kv_cache = torch.tensor([])
         set_default_quant_scales(self, register_buffer=True)
'''


def parse_diff(text: str) -> dict[str, list[tuple[list[str], list[str]]]]:
    files: dict[str, list[tuple[list[str], list[str]]]] = {}
    hunks: list[tuple[list[str], list[str]]] = []
    old_left = new_left = 0
    for line in text.splitlines(keepends=True):
        if old_left or new_left:
            if line[0] != "+":
                hunks[-1][0].append(line[1:])
                old_left -= 1
            if line[0] != "-":
                hunks[-1][1].append(line[1:])
                new_left -= 1
        elif line.startswith("+++ b/vllm/"):
            hunks = files.setdefault(line[len("+++ b/vllm/") :].rstrip("\n"), [])
        elif line.startswith("@@"):
            old_left, new_left = (int(n or 1) for n in HUNK_HEADER.match(line).groups())
            hunks.append(([], []))
    if old_left or new_left:
        raise SystemExit("vendored vllm#55557 diff is truncated")
    return files


def find(lines: list[str], block: list[str], start: int) -> int:
    for i in range(start, len(lines) - len(block) + 1):
        if lines[i : i + len(block)] == block:
            return i
    return -1


def apply(lines: list[str], hunks: list[tuple[list[str], list[str]]]) -> list[str] | None:
    out: list[str] = []
    cursor = 0
    for old, new in hunks:
        at = find(lines, old, cursor)
        if at < 0:
            return None
        out += lines[cursor:at] + new
        cursor = at + len(old)
    return out + lines[cursor:]


def already_applied(lines: list[str], hunks: list[tuple[list[str], list[str]]]) -> bool:
    cursor = 0
    for _, new in hunks:
        at = find(lines, new, cursor)
        if at < 0:
            return False
        cursor = at + len(new)
    return True


def main(argv: list[str]) -> int:
    if argv[:1] == ["--list"]:
        print("\n".join(TARGETS))
        return 0
    orig_dir = argv[0] if argv else os.path.join(HERE, "v030_fp8kv", "orig")
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, "v030_fp8kv")
    diff = parse_diff(UPSTREAM_DIFF)
    sources = {}
    for rel, local in TARGETS.items():
        with open(os.path.join(orig_dir, local), encoding="utf-8") as f:
            sources[rel] = f.read().splitlines(keepends=True)
    if all(already_applied(sources[rel], diff[rel]) for rel in TARGETS):
        print("vllm#55557 already present in the originals, copying them unchanged")
        results = sources
    else:
        results = {}
        for rel in TARGETS:
            patched = apply(sources[rel], diff[rel])
            if patched is None:
                print(f"vllm#55557 does not apply to {rel}, nothing written", file=sys.stderr)
                return 1
            results[rel] = patched
    texts = {rel: "".join(lines) for rel, lines in results.items()}
    for rel, text in texts.items():
        ast.parse(text, filename=rel)
    for rel, local in TARGETS.items():
        path = os.path.join(out_dir, local)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write(texts[rel])
        os.replace(path + ".tmp", path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
