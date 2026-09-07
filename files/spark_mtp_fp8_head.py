# SPDX-License-Identifier: Apache-2.0
"""Opt-in GB10 W8A16 reduced MTP head; target logits remain untouched.

Kernel adapted by Raymond Lucke from Gabriel Olympie's (@gabrielolympie)
sglang-flashnext-sm120, patches 0004/0005,
commit 67d2f9234fa45ae1339f0d53cd37cb695e9c6493 (Apache-2.0 kernels).
See LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md in this recipe.
The (N=32, K=256, stages=3) tactic was measured on SM121 with cold weights
under serialized CUDA graph replay, not copied from RTX timings.

Original BF16 buffers are retained. FP8/scales are nonpersistent module-owned
buffers created after loading, never a data_ptr-keyed global cache or a lazy
capture-time quantization. No target-model, tokenizer, or sampler patch.
"""
import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def _head(X, W, S, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          XM: tl.constexpr, BM: tl.constexpr):
    rn = tl.program_id(0)*32 + tl.arange(0, 32)
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, 32), tl.float32)
    for k0 in tl.range(0, K, 256, num_stages=3):
        rk = k0 + tl.arange(0, 256)
        x = tl.load(X + rm[:, None]*XM + rk[None, :],
                    (rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None]*K + rk[None, :],
                    (rn[:, None] < N) & (rk[None, :] < K), other=0.0).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    scale = tl.load(S + rn, rn < N, other=0.0)
    acc = acc * scale[None, :]
    tl.store(O + rm[:, None]*N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


def _logits(x: torch.Tensor, bf16: torch.Tensor, fp8: torch.Tensor,
            scale: torch.Tensor) -> torch.Tensor:
    if (x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda
            or not 0 < x.shape[0] <= 32 or x.stride(1) != 1):
        return torch.nn.functional.linear(x.to(bf16.dtype), bf16)
    m, k = x.shape
    n = fp8.shape[0]
    out = x.new_empty((m, n))
    _head[(triton.cdiv(n, 32),)](x, fp8, scale, out, m, n, k, x.stride(0),
                               max(16, triton.next_power_of_2(m)), num_warps=4)
    return out


def _fake(x: torch.Tensor, bf16: torch.Tensor, fp8: torch.Tensor,
          scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], bf16.shape[0]), dtype=bf16.dtype)


direct_register_custom_op(op_name="spark_mtp_fp8_head", op_func=_logits, fake_impl=_fake)


@torch.no_grad()
def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row E4M3 weights, FP32 scales; <=32 MiB FP32 chunk."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("FP8 draft head requires a contiguous 2D BF16 head")
    packed = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    rows = max(1, (32*1024*1024)//(weight.shape[1]*4))
    for start in range(0, weight.shape[0], rows):
        chunk = weight[start:start+rows].float()
        if not torch.isfinite(chunk).all():
            raise ValueError("Non-finite BF16 draft-head weights")
        s = (chunk.abs().amax(dim=1)/448).clamp_min(1e-12)
        packed[start:start+rows] = (chunk/s[:, None]).clamp(-448, 448).to(packed.dtype)
        scale[start:start+rows] = s
    return packed, scale


@torch.no_grad()
def attach_fp8_draft_head(model: torch.nn.Module) -> None:
    if os.environ.get("VLLM_MTP_DRAFT_HEAD_FP8", "0") != "1":
        return
    weight = getattr(model, "_draft_lm_head_weight", None)
    if (weight is None or not weight.is_cuda or weight.shape[1] != 2560
            or torch.cuda.get_device_capability(weight.device) != (12, 1)
            or getattr(model.lm_head, "tp_size", 1) != 1):
        raise RuntimeError("MTP_DRAFT_HEAD_FP8 requires SM121, TP1, and a loaded reduced Qwen head")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FP8 draft head must be prepared before graph capture")
    packed, scales = quantize_rows(weight)
    model.register_buffer("_draft_lm_head_fp8", packed, persistent=False)
    model.register_buffer("_draft_lm_head_fp8_scale", scales, persistent=False)
    # The real graph warmup covers all actual M values. Explicitly compile
    # the basic tactic here so no lazy quantization/JIT is hidden in capture.
    for m in (1, 2, 4, 8, 12, 16):
        _logits(torch.zeros((m, weight.shape[1]), dtype=weight.dtype, device=weight.device),
                weight, packed, scales)
    logger.info("MTP FP8 draft head: %d rows; %.3f GiB extra immutable buffers; "
                "target weights and rejection sampling unchanged",
                weight.shape[0], (packed.numel()+scales.numel()*4)/2**30)
