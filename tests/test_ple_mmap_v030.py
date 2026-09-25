import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "files" / "patch_ple_mmap_v030.py"
ORIG = Path(
    os.environ.get(
        "PLE_MMAP_ORIG",
        "/w/v030/single/ngram_embedding.py.orig",
    )
)
HAVE_ORIG = ORIG.is_file()
IDENTITY = {"model": "nvidia/Qwen3.8-Flash-Next-NVFP4", "revision": None, "snapshot": "fab0aec"}


def _run_patch(orig, out):
    return subprocess.run(
        [sys.executable, str(PATCH), str(orig), str(out)],
        capture_output=True,
        text=True,
    )


def _load_generated(tmp):
    out = Path(tmp) / "ngram_embedding_gen.py"
    result = _run_patch(ORIG, out)
    assert result.returncode == 0, result.stderr
    name = "vllm.models.qwen4_exp.nvidia.ngram_embedding_gen"
    spec = importlib.util.spec_from_file_location(name, out)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE_ORIG, f"original not found at {ORIG}")
class Generator(unittest.TestCase):
    def test_applies_and_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.py"
            result = _run_patch(ORIG, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            compile(out.read_text(), str(out), "exec")
            self.assertIn("_lookup_ple_rows_from_host_addr_kernel", out.read_text())

    def test_moved_anchor_fails_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = Path(tmp) / "orig.py"
            orig.write_text(ORIG.read_text().replace("    requires_device_loading: bool = False\n", ""))
            out = Path(tmp) / "out.py"
            result = _run_patch(orig, out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor", result.stderr)
            self.assertFalse(out.exists())

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.py", Path(tmp) / "b.py"
            self.assertEqual(_run_patch(ORIG, first).returncode, 0)
            self.assertEqual(_run_patch(ORIG, second).returncode, 0)
            self.assertEqual(first.read_text(), second.read_text())
            self.assertNotEqual(_run_patch(first, Path(tmp) / "c.py").returncode, 0)


@unittest.skipUnless(HAVE_ORIG, f"original not found at {ORIG}")
class PersistentTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._gen_dir = tempfile.TemporaryDirectory()
        cls.mod = _load_generated(cls._gen_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._gen_dir.cleanup()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "ple.bin")
        self.sidecar = os.path.join(self._tmp.name, "ple.json")
        self.shape = (64, 16)
        self.dtype = torch.float8_e4m3fn
        self.fingerprint = self.mod._ple_mmap_fingerprint(IDENTITY, self.shape, self.dtype)

    def tearDown(self):
        self._tmp.cleanup()

    def _open(self, fingerprint=None):
        return self.mod._ple_mmap_open(self.path, self.shape, self.dtype, fingerprint or self.fingerprint)

    def _fill(self, weight):
        data = torch.arange(weight.numel(), dtype=torch.int32).remainder(251).to(torch.uint8)
        weight.view(torch.uint8).view(-1).copy_(data)
        return data

    def _layer(self, weight, ready, shards, expected):
        return SimpleNamespace(
            _mmap_path=self.path,
            _mmap_ready=ready,
            _mmap_shards=set(shards),
            _mmap_expected_shards=expected,
            _mmap_fingerprint=self.fingerprint,
            weight=weight,
        )

    def test_build_then_reuse(self):
        weight, ready = self._open()
        self.assertFalse(ready)
        data = self._fill(weight)
        layer = self._layer(weight, False, range(4), 4)
        self.mod.Qwen4ExpPLEPinnedHostEmbedding.commit_mmap_table(layer)
        self.assertTrue(layer._mmap_ready)
        with open(self.sidecar) as handle:
            self.assertEqual(json.load(handle), self.fingerprint)
        self.assertFalse(os.path.exists(self.sidecar + ".tmp"))
        del weight
        weight, ready = self._open()
        self.assertTrue(ready)
        self.assertTrue(torch.equal(weight.view(torch.uint8).view(-1), data))

    def test_fingerprint_mismatch_rebuilds(self):
        weight, _ = self._open()
        self.mod.Qwen4ExpPLEPinnedHostEmbedding.commit_mmap_table(self._layer(weight, False, [0], 1))
        other = self.mod._ple_mmap_fingerprint({**IDENTITY, "snapshot": "other"}, self.shape, self.dtype)
        _, ready = self._open(other)
        self.assertFalse(ready)
        self.assertFalse(os.path.exists(self.sidecar))

    def test_same_snapshot_other_model_path_reuses(self):
        weight, _ = self._open()
        self.mod.Qwen4ExpPLEPinnedHostEmbedding.commit_mmap_table(self._layer(weight, False, [0], 1))
        other = self.mod._ple_mmap_fingerprint({**IDENTITY, "model": "/elsewhere/snapshot"}, self.shape, self.dtype)
        _, ready = self._open(other)
        self.assertTrue(ready)

    def test_no_snapshot_keys_on_model_path(self):
        base = {**IDENTITY, "snapshot": None}
        fp = self.mod._ple_mmap_fingerprint(base, self.shape, self.dtype)
        moved = self.mod._ple_mmap_fingerprint({**base, "model": "/elsewhere"}, self.shape, self.dtype)
        self.assertNotEqual(self.mod._ple_mmap_identity_key(fp), self.mod._ple_mmap_identity_key(moved))

    def test_size_mismatch_drops_sidecar_before_write(self):
        weight, _ = self._open()
        self.mod.Qwen4ExpPLEPinnedHostEmbedding.commit_mmap_table(self._layer(weight, False, [0], 1))
        del weight
        with open(self.path, "ab") as handle:
            handle.truncate(4096 * 1024)
        weight, ready = self._open()
        self.assertFalse(ready)
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertEqual(os.path.getsize(self.path), 64 * 16)

    def test_crash_before_sidecar_rebuilds(self):
        weight, ready = self._open()
        self.assertFalse(ready)
        self._fill(weight)
        del weight
        _, ready = self._open()
        self.assertFalse(ready)

    def test_incomplete_load_is_not_persisted(self):
        weight, _ = self._open()
        layer = self._layer(weight, False, range(3), 4)
        self.mod.Qwen4ExpPLEPinnedHostEmbedding.commit_mmap_table(layer)
        self.assertFalse(layer._mmap_ready)
        self.assertFalse(os.path.exists(self.sidecar))

    def test_sync_failure_leaves_no_sidecar(self):
        weight, _ = self._open()
        libc = mock.Mock()
        libc.msync.return_value = -1
        with mock.patch.object(self.mod.ctypes, "CDLL", return_value=libc):
            with self.assertRaises(OSError):
                self.mod._ple_mmap_commit(self.path, weight, self.fingerprint)
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertFalse(os.path.exists(self.sidecar + ".tmp"))

    def test_identity_uses_local_snapshot_dir(self):
        snapshot = os.path.join(self._tmp.name, "snapshots", "abc123")
        os.makedirs(snapshot)
        config = SimpleNamespace(model_config=SimpleNamespace(model=snapshot, model_weights="", revision=None))
        with mock.patch.object(self.mod, "get_current_vllm_config", return_value=config):
            identity = self.mod._ple_checkpoint_identity()
        self.assertEqual(identity["snapshot"], "abc123")
        self.assertEqual(identity["model"], os.path.realpath(snapshot))

    def _location(self, identity, env=True):
        environ = {"VLLM_PLE_MMAP_DIR": self._tmp.name if env else ""}
        with mock.patch.dict(os.environ, environ), mock.patch.object(
            self.mod, "_ple_checkpoint_identity", return_value=identity
        ):
            return self.mod._ple_mmap_location("model.layers.1.ple/ngram_embedding", 0)

    def test_location_disabled_without_env(self):
        self.assertIsNone(self._location(IDENTITY, env=False))

    def test_location_falls_back_to_model_path_hash(self):
        path, _ = self._location({"model": "/models/local", "revision": None, "snapshot": None})
        other, _ = self._location({"model": "/models/other", "revision": None, "snapshot": None})
        self.assertNotEqual(path, other)
        self.assertEqual(path, self._location({"model": "/models/local", "revision": None, "snapshot": None})[0])
        self.assertTrue(path.endswith(".etp0.bin"))

    def test_two_snapshots_keep_separate_tables(self):
        tables = {}
        for value, snapshot in enumerate(("fab0aecb760cec45227f", "925d7be6c14c6c9442ef"), 1):
            path, identity = self._location({**IDENTITY, "snapshot": snapshot})
            self.assertIn(f".{snapshot[:12]}.etp0.bin", path)
            fingerprint = self.mod._ple_mmap_fingerprint(identity, self.shape, self.dtype)
            weight, ready = self.mod._ple_mmap_open(path, self.shape, self.dtype, fingerprint)
            self.assertFalse(ready)
            data = torch.full((weight.numel(),), value, dtype=torch.uint8)
            weight.view(torch.uint8).view(-1).copy_(data)
            layer = self._layer(weight, False, [0], 1)
            layer._mmap_path, layer._mmap_fingerprint = path, fingerprint
            self.mod.Qwen4ExpPLEPinnedHostEmbedding.commit_mmap_table(layer)
            tables[path] = (fingerprint, data)
            del weight, layer
        self.assertEqual(len(tables), 2)
        for path, (fingerprint, data) in tables.items():
            weight, ready = self.mod._ple_mmap_open(path, self.shape, self.dtype, fingerprint)
            self.assertTrue(ready)
            self.assertTrue(torch.equal(weight.view(torch.uint8).view(-1), data))

    def test_post_load_hook_commits_only_mmap_layers(self):
        commit = mock.Mock()
        scale = torch.ones(1)
        self.mod.Qwen4ExpPLEFp8EmbeddingMethod().process_weights_after_loading(
            SimpleNamespace(weight_scale=scale, commit_mmap_table=commit)
        )
        self.mod.Qwen4ExpPLEUnquantizedEmbeddingMethod().process_weights_after_loading(
            SimpleNamespace(commit_mmap_table=commit)
        )
        self.assertEqual(commit.call_count, 2)
        self.mod.Qwen4ExpPLEFp8EmbeddingMethod().process_weights_after_loading(SimpleNamespace(weight_scale=scale))


@unittest.skipUnless(HAVE_ORIG, f"original not found at {ORIG}")
class LoadWeightsSkip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._gen_dir = tempfile.TemporaryDirectory()
        cls.mod = _load_generated(cls._gen_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._gen_dir.cleanup()

    def _run(self, device="cpu", **embedding_attrs):
        loader = mock.Mock()
        embedding = SimpleNamespace(
            org_vocab_size=10,
            embedding_dim=4,
            weight=SimpleNamespace(weight_loader=loader),
            **embedding_attrs,
        )
        owner = SimpleNamespace(
            layer_multipliers=torch.zeros(2, dtype=torch.long),
            ngram_heads_offsets=torch.zeros(2, dtype=torch.long),
            ngram_heads_vocab_sizes=torch.zeros(2, dtype=torch.long),
            split_ngram_parts=2,
            ngram_embedding=embedding,
        )
        weights = [(f"ngram_embedding.shard_{i}.weight", torch.zeros(5, 4, device=device)) for i in range(2)]
        loaded = self.mod.Qwen4ExpNGramEmbedding.load_weights(owner, weights)
        self.assertEqual(loaded, {"ngram_embedding.weight"})
        return loader, embedding

    def test_ready_table_skips_copy(self):
        loader, embedding = self._run(device="meta", _mmap_path="/x.bin", _mmap_ready=True, _mmap_shards=set())
        loader.assert_not_called()
        self.assertEqual(embedding._mmap_shards, {0, 1})
        self.assertEqual(embedding._mmap_expected_shards, 2)

    def test_building_table_copies(self):
        loader, embedding = self._run(_mmap_path="/x.bin", _mmap_ready=False, _mmap_shards=set())
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(embedding._mmap_shards, {0, 1})

    def test_upstream_layer_unchanged(self):
        loader, embedding = self._run()
        self.assertEqual(loader.call_count, 2)
        self.assertFalse(hasattr(embedding, "_mmap_shards"))


@unittest.skipUnless(HAVE_ORIG and torch.cuda.is_available(), "needs CUDA and the original source")
class HostAddressGather(unittest.TestCase):
    def test_gather_returns_file_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            mod = _load_generated(tmp)
            rows, dim = 4096, 96
            path = os.path.join(tmp, "table.bin")
            with open(path, "wb") as handle:
                handle.truncate(rows * dim)
            host = torch.from_file(path, shared=True, size=rows * dim, dtype=torch.uint8).view(rows, dim)
            host.copy_(torch.randint(0, 256, (rows, dim), dtype=torch.uint8))
            start, end = 1000, 3000
            ids = torch.tensor([1000, 2999, 1500, 5, 3000, 2048], device="cuda")
            out = torch.full((ids.numel(), dim), 7, dtype=torch.uint8, device="cuda")
            mod._lookup_ple_rows_from_host_addr_kernel[(ids.numel(),)](
                host.data_ptr(), ids, out, dim, start, end, BLOCK_B=128
            )
            torch.cuda.synchronize()
            want = torch.zeros(ids.numel(), dim, dtype=torch.uint8)
            for i, gid in enumerate(ids.tolist()):
                if start <= gid < end:
                    want[i] = host[gid - start]
            self.assertTrue(torch.equal(out.cpu(), want))


if __name__ == "__main__":
    unittest.main(verbosity=2)
