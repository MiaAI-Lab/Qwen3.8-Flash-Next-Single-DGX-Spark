"""CPU-sized regression checks; run in the pinned vLLM image, not on weights.

The optional PLE_SNAPSHOT / PLE_CACHE paths enable sampled checks of the real
NVIDIA cache. No model-sized allocations and no source writes are performed.
"""
import ast
import json
import logging
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
from vllm.models.qwen3_8_flash_next.nvidia.ple_layer import (
    _get_ple_embedding_quant_method,
    Qwen3_8FlashNextPLEFp8EmbeddingMethod as FP8,
    Qwen3_8FlashNextPLENVFp4EmbeddingMethod as FP4,
)

ROOT = Path(__file__).resolve().parents[1]
HF = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
RT = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def config(algo):
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None, exclude_modules=[],
        quantized_layers={RT: {"quant_algo": algo}},
        fp8_config=None, nvfp4_config=None, w4a16_nvfp4_config=None,
        mxfp8_config=None,
    )


def safetensors(path, tensors):
    header, body = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(body)
        body.extend(data)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(body)])
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


def read_header(path):
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length)), length + 8


class NvidiaPLETests(unittest.TestCase):
    def test_mixed_dispatch(self):
        self.assertIsInstance(_get_ple_embedding_quant_method(config("FP8"), RT), FP8)
        self.assertIsInstance(_get_ple_embedding_quant_method(config("NVFP4"), RT), FP4)
        self.assertIsNone(_get_ple_embedding_quant_method(config("FP8"), RT + ".absent"))
        with self.assertRaises(ValueError):
            _get_ple_embedding_quant_method(config("MXFP8"), RT)

    def test_existing_declared_dtype(self):
        self.assertIsInstance(_get_ple_embedding_quant_method(None, RT, "fp8"), FP8)
        self.assertIsInstance(_get_ple_embedding_quant_method(None, RT, "nvfp4"), FP4)

    def test_fp8_global_scale_slot(self):
        # create_weights builds ModelWeightParameter, which resolves the TP
        # rank; initialize a single-rank gloo world for it.
        import socket
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ps.init_distributed_environment(world_size=1, rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
        with set_current_vllm_config(VllmConfig()):
            ps.initialize_model_parallel(1, 1)
            layer = torch.nn.Module()
            FP8().create_weights(layer, 4, [2], 4, 2, torch.bfloat16)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_scale.numel(), 1)
        self.assertEqual(layer.weight_scale.dtype, torch.bfloat16)

    def fixture(self, root, fp8):
        snap, out = root / "snapshot-a", root / "out"
        snap.mkdir()
        data = {}
        for shard in range(2):
            raw = bytes([0x38 + shard, 0xB8, 0x40, 0x00])
            data[f"{HF}.shard_{shard}.weight"] = ("F8_E4M3" if fp8 else "U8", [2, 2], raw)
            if not fp8:
                data[f"{HF}.shard_{shard}.weight_scale"] = ("F8_E4M3", [2, 1], bytes([56, 64]))
        if fp8:
            data[HF + ".weight_scale"] = ("BF16", [1], bytes([0x80, 0x3F]))
        safetensors(snap / "model.safetensors", data)
        (snap / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in data}}))
        return snap, out

    def build(self, snap, out):
        return subprocess.run([sys.executable, str(ROOT / "files/build_ple_packed_table.py"),
                               str(snap), str(out)], capture_output=True, text=True)

    def test_fp8_builder_and_stale_cache(self):
        with tempfile.TemporaryDirectory() as d:
            snap, out = self.fixture(Path(d), True)
            result = self.build(snap, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            target = out / (RT + ".packed_u8")
            self.assertEqual(target.read_bytes(), bytes([56, 184, 64, 0, 57, 184, 64, 0]))
            self.assertEqual(self.build(snap, out).returncode, 0)
            meta_path = Path(str(target) + ".json")
            meta = json.loads(meta_path.read_text()); meta["snapshot"] = "other"
            meta_path.write_text(json.dumps(meta))
            self.assertNotEqual(self.build(snap, out).returncode, 0)

    def test_original_nvfp4_builder(self):
        with tempfile.TemporaryDirectory() as d:
            snap, out = self.fixture(Path(d), False)
            result = self.build(snap, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((out / (RT + ".packed_u8")).read_bytes(),
                             bytes([56, 184, 56, 64, 0, 64, 57, 184, 56, 64, 0, 64]))

    def test_packed_fp8_attach_and_bit_exact_gather(self):
        # Exercise the generated worker's actual mmap attachment method without
        # starting its multiprocessing / CUDA machinery.
        tree = ast.parse((ROOT / "files/ple_offload/worker.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_attach_packed_table")
        method.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), method], type_ignores=[])
        ns = dict(torch=torch, json=json, os=os, logger=logging.getLogger(__name__))
        exec(compile(ast.fix_missing_locations(module), "worker-method", "exec"), ns)
        with tempfile.TemporaryDirectory() as d:
            snap, out = self.fixture(Path(d), True)
            self.assertEqual(self.build(snap, out).returncode, 0)
            emb = torch.nn.Module()
            emb.weight = torch.nn.Parameter(torch.empty((4, 2), dtype=torch.float8_e4m3fn), requires_grad=False)
            emb.weight_scale = torch.nn.Parameter(torch.tensor([2.], dtype=torch.bfloat16), requires_grad=False)
            emb.quant_method = FP8()
            ns["_attach_packed_table"](RT, SimpleNamespace(ngram_embedding=emb), str(out / (RT + ".packed_u8")))
            output = torch.empty((2, 2), dtype=torch.float8_e4m3fn)
            torch.index_select(emb._packed_table, 0, torch.tensor([0, 1]),
                               out=output.view(torch.uint8))
            torch.testing.assert_close(output.to(torch.bfloat16) * emb.weight_scale,
                                       torch.tensor([[2., -2.], [4., 0.]], dtype=torch.bfloat16))
            self.assertEqual(emb.weight_scale.numel(), 1)
            os.close(emb._packed_table_fd)

    @unittest.skipUnless(os.environ.get("PLE_SNAPSHOT"), "real snapshot not supplied")
    def test_real_nvidia_cache_samples(self):
        snap, cache = Path(os.environ["PLE_SNAPSHOT"]), Path(os.environ["PLE_CACHE"])
        idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        target = cache / (RT + ".packed_u8")
        meta = json.loads(Path(str(target) + ".json").read_text())
        self.assertEqual(meta["snapshot"], snap.name)
        self.assertEqual(meta["shard_dtype"], "F8_E4M3")
        self.assertEqual(target.stat().st_size, meta["total_rows"] * meta["row_width"])
        # Start/middle/end of every shard, plus bytes crossing row boundaries.
        count = 0
        with target.open("rb") as packed:
            for shard in range(meta["num_shards"]):
                key = f"{HF}.shard_{shard}.weight"
                h, base = read_header(snap / idx[key]); shape = h[key]["shape"]
                self.assertEqual(h[key]["dtype"], "F8_E4M3")
                for row in (0, shape[0] // 2, shape[0] - 2):
                    offset = row * shape[1]
                    with (snap / idx[key]).open("rb") as source:
                        source.seek(base + h[key]["data_offsets"][0] + offset)
                        raw = source.read(shape[1] * 2)
                    packed.seek((shard * shape[0] + row) * shape[1])
                    self.assertEqual(packed.read(len(raw)), raw)
                    count += 1
        print(f"Real NVIDIA packed cache: {count} sampled spans match original bytes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
