# SPDX-License-Identifier: Apache-2.0
"""Actual-checkpoint FP8 draft-head tests. Run with the LLM stopped.

Use the serving image in a bounded, network-disabled disposable container.
Make files/spark_mtp_fp8_head.py importable (e.g. PYTHONPATH=/recipe/files).
Mount the complete HF cache tree read-only so snapshot symlinks resolve.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import statistics

import torch
from safetensors import safe_open

import spark_mtp_fp8_head as candidate


def error(actual, reference):
    a, b = actual.float(), reference.float()
    return {"finite": bool(torch.isfinite(a).all()),
            "relative_rmse": float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)),
            "max_absolute": float((a-b).abs().max())}


def graph_ms(fn, weights):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for w in weights: fn(w)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [fn(w) for w in weights]
    for _ in range(3): graph.replay()
    samples = []
    for _ in range(9):
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(8): graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b)/8/len(weights))
    return statistics.median(samples)


@torch.inference_mode()
def main(snapshot, vocab):
    assert torch.cuda.get_device_capability() == (12,1)
    torch.manual_seed(853)
    torch.set_float32_matmul_precision("highest")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    ids = torch.tensor(sorted({int(x) for x in vocab.read_text().splitlines() if x.strip()}))
    with safe_open(snapshot / index["weight_map"]["lm_head.weight"], framework="pt", device="cpu") as handle:
        full = handle.get_tensor("lm_head.weight")
        assert ids.numel() and int(ids.min()) >= 0 and int(ids.max()) < full.shape[0]
        weight = full.index_select(0,ids).to("cuda")
    original = weight.clone()
    packed,scale = candidate.quantize_rows(weight)
    assert torch.equal(weight,original)
    count = max(4, (128*1024**2 + packed.numel()-1)//packed.numel())
    weights = [weight] + [weight.clone() for _ in range(count-1)]
    fp8s = [(packed,scale)] + [(packed.clone(),scale.clone()) for _ in range(count-1)]
    for m in (1,2,3,4,8,12,16,32,33):
        x = torch.randn(m,weight.shape[1],device="cuda",dtype=torch.bfloat16)
        bf = torch.nn.functional.linear(x,weight)
        out = torch.ops.vllm.spark_mtp_fp8_head(x,weight,packed,scale)
        ref = (x.float() @ packed.float().T * scale[None,:]).to(torch.bfloat16) if m <= 32 else bf
        numerical = error(out,ref)
        assert numerical["finite"] and numerical["relative_rmse"] < .006
        slow = graph_ms(lambda w: torch.nn.functional.linear(x,w),weights)
        fast = graph_ms(lambda ps: torch.ops.vllm.spark_mtp_fp8_head(x,weight,ps[0],ps[1]),fp8s)
        print(json.dumps({"m":m,"passed":True,"kernel_error":numerical,
            "vs_bf16_error":error(out,bf),"top1_matches":int((out.argmax(-1)==bf.argmax(-1)).sum()),
            "top1_total":m,"bf16_ms":slow,"fp8_ms":fast,"speedup":slow/fast}),flush=True)
    zeros = torch.zeros(32,2560,device="cuda",dtype=torch.bfloat16)
    zp,zs = candidate.quantize_rows(zeros)
    assert torch.isfinite(zs).all() and torch.count_nonzero(zp.float()) == 0
    zeros[0,0] = float("nan")
    try: candidate.quantize_rows(zeros)
    except ValueError: pass
    else: raise AssertionError("Nonfinite weights accepted")
    module = torch.nn.Module()
    module.lm_head = SimpleNamespace(tp_size=1)
    module.register_buffer("_draft_lm_head_weight",weight,persistent=False)
    candidate.attach_fp8_draft_head(module)
    assert "_draft_lm_head_fp8" in dict(module.named_buffers())
    assert "_draft_lm_head_fp8" not in module.state_dict()
    assert torch.equal(original,module._draft_lm_head_weight)
    print(json.dumps({"buffer_and_invalid_input_checks":True,"passed":True}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--draft-vocab",type=Path,required=True)
    args = parser.parse_args()
    main(args.snapshot,args.draft_vocab)
