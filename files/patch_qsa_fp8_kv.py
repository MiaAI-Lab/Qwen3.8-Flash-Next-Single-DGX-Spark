#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Teach the Qwen3.8-Flash-Next QSA kernels to read an FP8-e4m3 KV cache.

Credit, for this file only: the FP8-KV approach comes from
lancelind/qwen3.8-Flash-DGX (Apache-2.0). Nothing else in this repository
derives from that project. Reimplemented here against this image's own
sources: dequantise each K/V tile inside the Triton kernels using vLLM's
canonical ``_cast_kv_tile`` rather than duplicating the maths, so upstream
fixes are inherited.

Why FP8 KV is worth having: the main KV is 12 full-attention layers x 2 KV
heads x 256 dims x (K+V) x 2 B = 24,576 B/token, ~84% of the 29,294 B/token
this model spends. Halving it lifts a 19 GiB pool from ~700k to ~1.2M tokens,
which is what makes a 1M-token context fit on one Spark.

Why it is OFF by default: on the reference implementation a long-reasoning
benchmark regressed from 6/6 to 2/6 with FP8 KV, prefill lost ~30% and decode
~9%. This is sparse attention -- quantising keys perturbs which blocks the
indexer selects, not merely the attention output. Treat it as a capacity
trade, not a free win, and re-validate quality on your own workload.

Inert unless --kv-cache-dtype is fp8: KV_QUANT_MODE is a tl.constexpr, so the
dequantisation branches are eliminated at Triton compile time and the BF16
path emits the same code as before.

Inputs:  files/qsa_ops_patched.py.orig     (nvidia/ops/qsa.py from the image)
         files/qsa_nvidia_patched.py.orig  (nvidia/qsa.py from the image)
Outputs: files/qsa_ops_patched.py, files/qsa_nvidia_patched.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def patch(name: str, edits: list[tuple[str, str]]) -> None:
    orig = os.path.join(HERE, f"{name}.orig")
    dest = os.path.join(HERE, name)
    if not os.path.exists(orig):
        sys.exit(f"{name}: missing {orig} (start.sh extracts it from the image)")
    src = open(orig).read()
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(
                f"{name}: anchor {i} not unique/missing (count={count}):\n{old[:180]}"
            )
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"{name}: patched source does not parse: {exc}")
    open(dest, "w").write(src)
    print(f"patched {name}")


# ---------------------------------------------------------------------------
# Helpers injected into ops/qsa.py.
# ---------------------------------------------------------------------------
HELPERS = '''

_QSA_DEFAULT_SCALE: dict[torch.device, torch.Tensor] = {}


def _qsa_scale_ptr(scale, device):
    """Return a 0-d fp32 scale the kernels can always dereference.

    ``_cast_kv_tile`` loads the scale unconditionally in FP8 mode, so a real
    tensor must exist even when the caller passes nothing.
    """
    if scale is not None:
        return scale.to(device=device, dtype=torch.float32).reshape(())
    cached = _QSA_DEFAULT_SCALE.get(device)
    if cached is None:
        cached = torch.ones((), dtype=torch.float32, device=device)
        _QSA_DEFAULT_SCALE[device] = cached
    return cached


def _qsa_as_fp8(cache, kv_quant_mode, what):
    """Reinterpret a uint8 KV allocation as fp8-e4m3 without copying.

    vLLM allocates a quantised KV cache as uint8. Reading it as uint8 in the
    kernel would silently produce garbage, so refuse that combination loudly
    rather than returning plausible nonsense.
    """
    quantised_dtypes = (torch.uint8, torch.float8_e4m3fn)
    if kv_quant_mode:
        if cache.dtype == torch.uint8:
            return cache.view(torch.float8_e4m3fn)
        return cache
    if cache.dtype in quantised_dtypes:
        raise RuntimeError(
            f"QSA {what} cache is {cache.dtype} but KV dequantisation is off; "
            "refusing to read quantised bytes as BF16"
        )
    return cache

'''

patch(
    "qsa_ops_patched.py",
    [
        # -- import the canonical dequantiser + local helpers ----------------
        (
            "from vllm.triton_utils import HAS_TRITON, tl, triton\n",
            "from vllm.triton_utils import HAS_TRITON, tl, triton\n"
            "from vllm.v1.attention.ops.triton_unified_attention import _cast_kv_tile\n",
        ),
        (
            "\n\ndef _validate_mqa(q: torch.Tensor) -> None:",
            HELPERS + "\ndef _validate_mqa(q: torch.Tensor) -> None:",
        ),
        # -- MQA (block-selection) kernel: signature -------------------------
        (
            "    num_requests,\n"
            "    score_divisor,\n"
            "    PAGE_SIZE: tl.constexpr,\n",
            "    num_requests,\n"
            "    score_divisor,\n"
            "    k_scale_ptr,\n"
            "    PAGE_SIZE: tl.constexpr,\n",
        ),
        (
            "    MAX_N: tl.constexpr,\n"
            "    COMPRESS_RATIO: tl.constexpr,\n"
            ") -> None:\n",
            "    MAX_N: tl.constexpr,\n"
            "    COMPRESS_RATIO: tl.constexpr,\n"
            "    KV_QUANT_MODE: tl.constexpr,\n"
            ") -> None:\n",
        ),
        # -- MQA kernel: dequantise keys before scoring ----------------------
        (
            "        scores = tl.dot(keys, query, out_dtype=tl.float32)\n",
            "        keys = _cast_kv_tile(keys, query, k_scale_ptr, KV_QUANT_MODE)\n"
            "        scores = tl.dot(keys, query, out_dtype=tl.float32)\n",
        ),
        # -- decode/prefill sparse GQA kernel: signature ---------------------
        (
            "    num_rows,\n"
            "    num_cache_blocks,\n"
            "    num_requests,\n"
            "    TOPK: tl.constexpr,\n",
            "    num_rows,\n"
            "    num_cache_blocks,\n"
            "    num_requests,\n"
            "    k_scale_ptr,\n"
            "    v_scale_ptr,\n"
            "    TOPK: tl.constexpr,\n",
        ),
        (
            "    BLOCK_M: tl.constexpr,\n"
            "    BLOCK_N: tl.constexpr,\n"
            ") -> None:\n"
            "    row = tl.program_id(0)\n"
            "    kv_head = tl.program_id(1)\n",
            "    BLOCK_M: tl.constexpr,\n"
            "    BLOCK_N: tl.constexpr,\n"
            "    KV_QUANT_MODE: tl.constexpr,\n"
            ") -> None:\n"
            "    row = tl.program_id(0)\n"
            "    kv_head = tl.program_id(1)\n",
        ),
        # -- decode kernel: dequantise K and V before the dots ---------------
        (
            "        scores = tl.dot(query, keys)\n"
            "        # Scaling scores avoids re-quantizing a scaled query to BF16.\n",
            "        keys = _cast_kv_tile(keys, query, k_scale_ptr, KV_QUANT_MODE)\n"
            "        values = _cast_kv_tile(values, query, v_scale_ptr, KV_QUANT_MODE)\n"
            "        scores = tl.dot(query, keys)\n"
            "        # Scaling scores avoids re-quantizing a scaled query to BF16.\n",
        ),
        # -- MQA wrapper: accept a scale + mode, reinterpret the cache -------
        (
            "    num_columns: int | None = None,\n"
            "    score_scale: float | None = None,\n"
            ") -> tuple[torch.Tensor, torch.Tensor]:\n",
            "    num_columns: int | None = None,\n"
            "    score_scale: float | None = None,\n"
            "    k_scale: torch.Tensor | None = None,\n"
            "    kv_quant_mode: int = 0,\n"
            ") -> tuple[torch.Tensor, torch.Tensor]:\n",
        ),
        (
            "    _validate_mqa(q)\n",
            "    _validate_mqa(q)\n"
            "    k_cache = _qsa_as_fp8(k_cache, kv_quant_mode, \"selector\")\n",
        ),
        (
            "        float(score_divisor),\n"
            "        PAGE_SIZE=k_cache.shape[1],\n",
            "        float(score_divisor),\n"
            "        _qsa_scale_ptr(k_scale, q.device),\n"
            "        PAGE_SIZE=k_cache.shape[1],\n",
        ),
        (
            "        COMPRESS_RATIO=compress_ratio,\n        num_warps=2,\n",
            "        COMPRESS_RATIO=compress_ratio,\n"
            "        KV_QUANT_MODE=kv_quant_mode,\n"
            "        num_warps=2,\n",
        ),
        # -- sparse attention wrapper: scales, mode, reinterpret, tile width --
        (
            "    out: torch.Tensor | None = None,\n"
            ") -> torch.Tensor:\n"
            '    """Run sparse GQA directly over paged BF16 K/V caches."""\n',
            "    out: torch.Tensor | None = None,\n"
            "    k_scale: torch.Tensor | None = None,\n"
            "    v_scale: torch.Tensor | None = None,\n"
            "    kv_quant_mode: int = 0,\n"
            ") -> torch.Tensor:\n"
            '    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches."""\n',
        ),
        (
            "    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:\n"
            '        raise ValueError("QSA sparse attention metadata has invalid shapes")\n',
            "    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:\n"
            '        raise ValueError("QSA sparse attention metadata has invalid shapes")\n'
            '    k_cache = _qsa_as_fp8(k_cache, kv_quant_mode, "key")\n'
            '    v_cache = _qsa_as_fp8(v_cache, kv_quant_mode, "value")\n',
        ),
        # -- the runtime dtype assert: fp8 caches are legal now -------------
        (
            "    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16\n",
            "    assert q.dtype == torch.bfloat16\n"
            "    if kv_quant_mode:\n"
            "        assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn\n"
            "    else:\n"
            "        assert k_cache.dtype == v_cache.dtype == torch.bfloat16\n",
        ),
        (
            "    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)\n",
            "    if kv_quant_mode:\n"
            "        # Dequantised tiles need BF16 staging in shared memory; halve the\n"
            "        # tile width so occupancy stays inside GB10's SMEM budget.\n"
            "        block_n = max(16, block_n // 2)\n"
            "    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)\n",
        ),
        (
            "        block_table.shape[0],\n"
            "        TOPK=logical_indices.shape[1],\n",
            "        block_table.shape[0],\n"
            "        _qsa_scale_ptr(k_scale, q.device),\n"
            "        _qsa_scale_ptr(v_scale, q.device),\n"
            "        TOPK=logical_indices.shape[1],\n",
        ),
        (
            "        BLOCK_N=block_n,\n        num_warps=partial_warps,\n",
            "        BLOCK_N=block_n,\n"
            "        KV_QUANT_MODE=kv_quant_mode,\n"
            "        num_warps=partial_warps,\n",
        ),
    ],
)

patch(
    "qsa_nvidia_patched.py",
    [
        # -- advertise fp8 on the main attention backend ---------------------
        (
            '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]\n'
            "\n"
            "    @staticmethod\n"
            "    def get_name() -> str:\n"
            '        return "QWEN38_FLASH_NEXT_QSA_TRITON"\n',
            "    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n"
            '        "auto",\n        "bfloat16",\n        "fp8",\n        "fp8_e4m3",\n'
            "    ]\n"
            "\n"
            "    @staticmethod\n"
            "    def get_name() -> str:\n"
            '        return "QWEN38_FLASH_NEXT_QSA_TRITON"\n',
        ),
        # -- allow an fp8 cache with a bf16 query ----------------------------
        (
            "        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:\n"
            '            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")\n',
            "        kv_quant_mode = (\n"
            "            1 if key_cache.dtype in (torch.uint8, torch.float8_e4m3fn) else 0\n"
            "        )\n"
            "        if query.dtype != torch.bfloat16:\n"
            '            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires a BF16 query")\n'
            "        if not kv_quant_mode and key_cache.dtype != torch.bfloat16:\n"
            "            raise NotImplementedError(\n"
            '                "Qwen3.8-Flash-Next QSA requires a BF16 or FP8-e4m3 KV cache"\n'
            "            )\n",
        ),
        # -- hand the layer's scales to the kernel ---------------------------
        (
            "        qsa_sparse_paged_attention(\n"
            "            query[:num_tokens],\n"
            "            key_cache,\n"
            "            value_cache,\n"
            "            logical_indices,\n"
            "            attn_metadata.block_table,\n"
            "            token_to_req,\n"
            "            output[:num_tokens],\n"
            "        )\n",
            "        qsa_sparse_paged_attention(\n"
            "            query[:num_tokens],\n"
            "            key_cache,\n"
            "            value_cache,\n"
            "            logical_indices,\n"
            "            attn_metadata.block_table,\n"
            "            token_to_req,\n"
            "            output[:num_tokens],\n"
            '            k_scale=getattr(layer, "_k_scale", None),\n'
            '            v_scale=getattr(layer, "_v_scale", None),\n'
            "            kv_quant_mode=kv_quant_mode,\n"
            "        )\n",
        ),
        # -- neutralise the parent FlashAttention fp8 rejection --------------
        # FlashAttentionImpl.__init__ refuses a quantised KV cache because the
        # FlashAttention kernel cannot read one. That does not apply here: this
        # layer's only attention entry point is forward_qsa (via the
        # qwen3_8_flash_next_qsa_with_output op), which runs the QSA Triton
        # kernels this patch teaches to dequantise. The inherited
        # FlashAttention forward is never called for these layers -- verified
        # against this build: the Impl subclass defines only __init__ and
        # forward_qsa. So hand the parent an unquantised dtype, then restore.
        (
            "    def __init__(self, *args, **kwargs) -> None:\n"
            "        super().__init__(*args, **kwargs)\n"
            "        if not is_flash_attn_varlen_func_available():\n",
            "    def __init__(self, *args, **kwargs) -> None:\n"
            "        _real_kv_dtype = None\n"
            "        _kv_pos = None\n"
            '        if "kv_cache_dtype" in kwargs:\n'
            '            _real_kv_dtype = kwargs["kv_cache_dtype"]\n'
            "        else:\n"
            "            import inspect as _inspect\n"
            "\n"
            "            _names = list(\n"
            "                _inspect.signature(FlashAttentionImpl.__init__).parameters\n"
            "            )[1:]\n"
            '            if "kv_cache_dtype" in _names:\n'
            '                _idx = _names.index("kv_cache_dtype")\n'
            "                if _idx < len(args):\n"
            "                    _kv_pos = _idx\n"
            "                    _real_kv_dtype = args[_idx]\n"
            "        if _real_kv_dtype is not None and _real_kv_dtype not in (\n"
            '            "auto",\n            "bfloat16",\n'
            "        ):\n"
            "            if _kv_pos is not None:\n"
            '                args = args[:_kv_pos] + ("auto",) + args[_kv_pos + 1 :]\n'
            "            else:\n"
            '                kwargs = {**kwargs, "kv_cache_dtype": "auto"}\n'
            "            super().__init__(*args, **kwargs)\n"
            "            self.kv_cache_dtype = _real_kv_dtype\n"
            "        else:\n"
            "            super().__init__(*args, **kwargs)\n"
            "        if not is_flash_attn_varlen_func_available():\n",
        ),
        # -- impl __init__: second copy of the cache_dtype guard -------------
        (
            '        if self.kv_cache_dtype not in ("auto", "bfloat16"):\n'
            "            raise NotImplementedError(\n"
            '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
            "            )\n"
            "        self.supports_quant_query_input = False\n",
            "        if self.kv_cache_dtype not in (\n"
            '            "auto",\n            "bfloat16",\n            "fp8",\n            "fp8_e4m3",\n'
            "        ):\n"
            "            raise NotImplementedError(\n"
            '                "Qwen3.8-Flash-Next QSA supports a BF16 or FP8-e4m3 main KV cache"\n'
            "            )\n"
            "        self.supports_quant_query_input = False\n",
        ),
        # -- storage dtype: fp8 is stored as uint8 / float8_e4m3fn -----------
        (
            "        if self.kv_cache_torch_dtype != torch.bfloat16:\n"
            "            raise NotImplementedError(\n"
            '                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"\n'
            "            )\n",
            "        if self.kv_cache_torch_dtype not in (\n"
            "            torch.bfloat16,\n            torch.uint8,\n            torch.float8_e4m3fn,\n"
            "        ):\n"
            "            raise NotImplementedError(\n"
            '                "Qwen3.8-Flash-Next QSA cache storage must be BF16 or FP8-e4m3 "\n'
            '                f"(got {self.kv_cache_torch_dtype})"\n'
            "            )\n",
        ),
        # -- stop __init__ rejecting an fp8 cache_dtype ----------------------
        (
            '        if cache_config.cache_dtype not in ("auto", "bfloat16"):\n'
            "            raise NotImplementedError(\n"
            '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
            "            )\n",
            "        if cache_config.cache_dtype not in (\n"
            '            "auto",\n            "bfloat16",\n            "fp8",\n            "fp8_e4m3",\n'
            "        ):\n"
            "            raise NotImplementedError(\n"
            '                "Qwen3.8-Flash-Next QSA supports a BF16 or FP8-e4m3 main KV cache"\n'
            "            )\n",
        ),
    ],
)

print("ok")
