"""CPU regression check for the PLE offload staging buffer width.

forward_impl slices the worker's pinned staging buffer to [:T, :row_width].
At the ple_embed_dim width (2560) that slice is not contiguous for T > 1, so
reshape() returns a copy and index_select writes the rows into the copy. The
GPU then gets the stale staging bytes for every forward with more than one
token (each MTP verify step, each prefill chunk). patch_ple_offload.py sizes
the buffer to the packed row width (16 heads x 90 bytes = 1440).

The test runs the production forward_impl from files/ple_layer_patched.py and
the production _staging_width from files/ple_offload/worker.py. start.sh
generates both files; run ./start.sh --dry-run or the patch scripts first.
Only the hash table sizes are small. The code path is unchanged.

    python3 -m unittest tests/test_ple_staging_width.py
"""
import ast
import logging
import os
from pathlib import Path
import types
import unittest

import torch

REPO = Path(__file__).resolve().parent.parent
PLE_SRC = REPO / "files" / "ple_layer_patched.py"
WORKER_SRC = REPO / "files" / "ple_offload" / "worker.py"
# Checkpoint values (Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 config.json).
NGRAM_SIZE, HEADS_PER_NGRAM, EMBED_DIM = 3, 8, 2560
VOCAB, EOS = 248320, 248044
HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM              # 16
HEAD_DIM = EMBED_DIM // HEADS                           # 160 FP4 values
ROW_WIDTH = HEAD_DIM // 2 + HEAD_DIM // 16              # 80 code bytes + 10 scale bytes


def _load_forward_impl():
    tree = ast.parse(PLE_SRC.read_text())
    ns = {"torch": torch, "os": os}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_splitmix64", "_ple_prefetch_rows"):
            keep.append(node)
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "").startswith(("_MASK64", "_SPLITMIX", "_PLE_")) for t in node.targets):
            keep.append(node)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(PLE_SRC), "exec"), ns)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Qwen3_8FlashNextNGramEmbedding")
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    cns = dict(ns, is_offload_process=lambda: True)
    for name in ("_shift_precompute", "_shift_apply", "forward_impl"):
        node = methods[name]
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(PLE_SRC), "exec"), cns)
    return ns, cns


def _load_staging_width():
    tree = ast.parse(WORKER_SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_staging_width")
    ns = {"torch": torch, "logger": logging.getLogger("ple-staging-test")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(WORKER_SRC), "exec"), ns)
    return ns["_staging_width"]


class _PackedEmbedding:
    """Stands in for the worker's embedding: a packed uint8 table, rows looked up by id."""

    def __init__(self, table):
        self._packed_table = table
        self._packed_table_fd = None

    def __call__(self, ids):
        rows = self._packed_table.index_select(0, ids.reshape(-1))
        return rows.view(*ids.shape, rows.shape[-1])


def _make_layer(ns, cns, rows_per_head=1009, max_tokens=512):
    layer = types.SimpleNamespace(ngram_size=NGRAM_SIZE, heads_per_ngram=HEADS_PER_NGRAM,
                                  ngram_heads=HEADS, eos_token_id=EOS)
    half = max(1, (((1 << 63) - 1) // VOCAB) // 2)
    seed = 1234 + 10007
    layer.layer_multipliers = torch.tensor(
        [2 * (ns["_splitmix64"](seed + ns["_SPLITMIX_GAMMA"] * (i + 1)) % half) + 1
         for i in range(NGRAM_SIZE)], dtype=torch.long)
    primes, p = [], rows_per_head
    while len(primes) < HEADS:
        p += 1
        if all(p % d for d in range(2, int(p ** 0.5) + 1)):
            primes.append(p)
    offsets = [sum(primes[:i]) for i in range(HEADS)]
    layer.ngram_heads_vocab_sizes = torch.tensor(primes, dtype=torch.long)
    layer.ngram_heads_offsets = torch.tensor(offsets, dtype=torch.long)
    table = torch.randint(0, 256, (sum(primes), ROW_WIDTH), dtype=torch.uint8,
                          generator=torch.Generator().manual_seed(0))
    layer.ngram_embedding = _PackedEmbedding(table)
    layer.positions_buffer = torch.arange(max_tokens, dtype=torch.int64)
    layer.padded_buffer = torch.full((16, max_tokens), EOS, dtype=torch.int64)
    layer._shift_precompute = cns["_shift_precompute"]
    layer._shift_apply = cns["_shift_apply"]
    layer.forward_impl = types.MethodType(cns["forward_impl"], layer)
    return layer


def _inputs(nt, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (nt,), dtype=torch.int32, generator=g)
    qsl = torch.tensor([0, nt], dtype=torch.int32)
    ctx = torch.randint(0, VOCAB, (1, 2), dtype=torch.int32, generator=g)
    return ids, qsl, ctx


@unittest.skipUnless(PLE_SRC.exists() and WORKER_SRC.exists(),
                     "generated files missing: run the patch scripts (start.sh) first")
class StagingWidthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.cns = _load_forward_impl()
        cls.staging_width = staticmethod(_load_staging_width())

    def _rows(self, layer, nt, seed, buf):
        ids, qsl, ctx = _inputs(nt, seed)
        ref = layer.forward_impl(ids, ids, qsl, ctx, output_buffer=None)
        out = layer.forward_impl(ids, ids, qsl, ctx, output_buffer=buf)
        return ref, out

    def test_embed_dim_buffer_loses_multi_token_rows(self):
        """Documents the bug: at width 2560 the rows reach the buffer only when T == 1."""
        layer = _make_layer(self.ns, self.cns)
        for nt in (1, 2, 4, 8, 16):
            buf = torch.zeros(512, EMBED_DIM, dtype=torch.uint8)
            ref, _ = self._rows(layer, nt, nt, buf)
            self.assertEqual(tuple(ref.shape), (nt, HEADS * ROW_WIDTH))
            self.assertEqual(torch.equal(buf[:nt, :ref.shape[1]], ref), nt == 1, f"nt={nt}")

    def test_packed_width_buffer_keeps_all_rows(self):
        layer = _make_layer(self.ns, self.cns)
        width = self.staging_width(layer, EMBED_DIM)
        self.assertEqual(width, HEADS * ROW_WIDTH)
        self.assertEqual(width, 1440)
        for nt in (1, 2, 4, 8, 16, 256):
            buf = torch.zeros(512, width, dtype=torch.uint8)
            ref, out = self._rows(layer, nt, 100 + nt, buf)
            self.assertTrue(torch.equal(out, ref), f"nt={nt}")
            self.assertTrue(torch.equal(buf[:nt], ref), f"nt={nt}")
            self.assertEqual(out.data_ptr(), buf.data_ptr(), f"nt={nt}")

    def test_width_follows_forward_impl_branches(self):
        sw = self.staging_width
        emb = types.SimpleNamespace
        packed = emb(_packed_table=torch.zeros(3, 90, dtype=torch.uint8))
        self.assertEqual(sw(emb(ngram_heads=16, ngram_embedding=packed), 2560), 1440)
        split = emb(weight=torch.zeros(3, 80, dtype=torch.uint8), weight_scale=torch.zeros(3, 10, dtype=torch.uint8))
        self.assertEqual(sw(emb(ngram_heads=16, ngram_embedding=split), 2560), 1440)
        plain = emb(weight=torch.zeros(3, 160, dtype=torch.bfloat16))
        self.assertEqual(sw(emb(ngram_heads=16, ngram_embedding=plain), 2560), 2560)

    def test_unknown_or_bad_width_keeps_embed_dim(self):
        sw = self.staging_width
        emb = types.SimpleNamespace
        self.assertEqual(sw(emb(ngram_heads=16), 2560), 2560)
        self.assertEqual(sw(emb(), 2560), 2560)
        emptied = emb(weight=torch.zeros(0, dtype=torch.uint8))  # weight released, no packed table
        self.assertEqual(sw(emb(ngram_heads=16, ngram_embedding=emptied), 2560), 2560)
        too_wide = emb(_packed_table=torch.zeros(3, 200, dtype=torch.uint8))
        self.assertEqual(sw(emb(ngram_heads=16, ngram_embedding=too_wide), 2560), 2560)


if __name__ == "__main__":
    unittest.main()
