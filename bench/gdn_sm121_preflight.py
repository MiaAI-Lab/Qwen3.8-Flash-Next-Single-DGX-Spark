"""SM121 FlashInfer GDN vs installed FLA: numerics, state continuation, timing.

Synthetic inputs with the live Qwen geometry (Hq=16, Hv=48, K=V=128),
including ragged batches, nonzero BF16 initial states, and resumed chunks.
Runs outside the serving engine. Requires Qwen stopped to leave memory headroom.
"""
import json
import statistics
import time

import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule as fi,
    fla_chunk_gated_delta_rule as fla,
)


def emit(row):
    print(json.dumps(row), flush=True)


def errors(actual, reference):
    a, b = actual.float(), reference.float()
    rms = b.square().mean().sqrt().clamp_min(1e-8)
    return {"finite": bool(torch.isfinite(a).all()),
            "relative_rmse": float((a-b).square().mean().sqrt()/rms),
            "max_absolute": float((a-b).abs().max()),
            "reference_rms": float(rms)}


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    # Includes adapter/copies and launch overhead, unlike just timing the core.
    for _ in range(9):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end)/8)
    return statistics.median(samples)


def make(lengths, seed, g_scale=1):
    torch.manual_seed(seed)
    total = sum(lengths)
    device = "cuda"
    def r(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32)
    return dict(q=r(1, total, 16, 128), k=r(1, total, 16, 128), v=r(1, total, 48, 128),
        g=-torch.rand(1, total, 48, device=device)*g_scale,
        beta=torch.rand(1, total, 48, device=device, dtype=torch.bfloat16),
        initial_state=r(len(lengths), 48, 128, 128)*0.1,
        output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)


@torch.inference_mode()
def main():
    assert torch.cuda.get_device_capability() == (12, 1)
    emit({"begin": True, "torch": torch.__version__, "device": torch.cuda.get_device_name(),
          "time": time.time()})
    for index, lengths in enumerate(([31], [65], [1664], [2048], [17, 63, 129, 257], [512]*4)):
        for g_scale in (1, 0.01):
            inputs = make(lengths, 610+index, g_scale)
            started = time.monotonic()
            ref_o, ref_s = fla(**inputs)
            out, state = fi(**inputs)
            torch.cuda.synchronize()
            eo, es = errors(out, ref_o), errors(state, ref_s)
            row = {"lengths": lengths, "g_scale": g_scale, "output_error": eo, "state_error": es,
                   "first_call_s": time.monotonic()-started}
            # A relative-RMS guard, not arbitrary elementwise allclose near zeros.
            row["passed"] = all(e["finite"] and e["relative_rmse"] < .03 for e in (eo, es))
            if g_scale == 1:
                row.update(fla_ms=timed(lambda: fla(**inputs)), fi_ms=timed(lambda: fi(**inputs)))
            emit(row)
            assert row["passed"], "GDN output/state failed numerical guard"
            del inputs, ref_o, ref_s, out, state
    # Two consecutive 1664-token cache chunks with BF16 state write-back.
    inputs = make([3328], 777, .01)
    full_o, full_s = fla(**inputs)
    states = {"fla": inputs["initial_state"], "fi": inputs["initial_state"]}
    outputs = {"fla": [], "fi": []}
    for start in (0, 1664):
        for name, fn in (("fla", fla), ("fi", fi)):
            kwargs = {**inputs, "initial_state": states[name].to(torch.bfloat16),
                "cu_seqlens": torch.tensor([0, 1664], device="cuda", dtype=torch.int32)}
            for key in ("q", "k", "v", "g", "beta"):
                kwargs[key] = inputs[key][:, start:start+1664].contiguous()
            o, states[name] = fn(**kwargs)
            outputs[name].append(o)
    eo = errors(torch.cat(outputs["fi"], 1), torch.cat(outputs["fla"], 1))
    es = errors(states["fi"], states["fla"])
    passed = all(e["finite"] and e["relative_rmse"] < .03 for e in (eo, es))
    emit({"continuation": True, "output_error": eo, "state_error": es,
          "fla_split_vs_full": errors(torch.cat(outputs["fla"], 1), full_o), "passed": passed})
    assert passed


if __name__ == "__main__":
    main()
