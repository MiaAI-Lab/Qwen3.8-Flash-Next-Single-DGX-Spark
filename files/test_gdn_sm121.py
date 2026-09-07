# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests. Run in the serving image; no GPU or downloads needed.

GDN_TEST_SOURCE may override the installed-image source for local testing.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

import patch_gdn_sm121 as patcher


class Platform:
    def __init__(self, cap): self.cap = cap
    def is_cuda(self): return self.cap is not None
    def is_device_capability(self, want):
        actual = self.cap[0]*10+self.cap[1] if self.cap else -1
        return actual == (want[0]*10+want[1] if isinstance(want, tuple) else want)
    def is_device_capability_family(self, want): return bool(self.cap and self.cap[0] == want//10)
    def get_cuda_runtime_major(self): return 13


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get("GDN_TEST_SOURCE", str(patcher.SOURCE))).read_text()

    def resolve(self, cap, backend="auto", head_dim=128, value_dim=128, cuda=13, available=True):
        tree = ast.parse(patcher.patch(self.source))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_gdn_prefill_backend")
        fn.returns = None
        fn.args.args[0].annotation = None
        platform = Platform(cap)
        platform.get_cuda_runtime_major = lambda: cuda
        ns = {"current_platform": platform}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "resolver", "exec"), ns)
        config = SimpleNamespace(additional_config={"gdn_prefill_backend": backend},
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(
                linear_key_head_dim=head_dim, linear_value_head_dim=value_dim)))
        kernels = SimpleNamespace(chunk_gated_delta_rule_sm120=(lambda: None) if available else None)
        with mock_patch.dict("sys.modules", {"flashinfer.gdn_kernels": kernels}):
            return ns["_resolve_gdn_prefill_backend"](config)[1]

    def test_opt_in(self):
        self.assertEqual(self.resolve((12,1)), "triton")
        self.assertEqual(self.resolve((12,1), "flashinfer"), "flashinfer")
        self.assertEqual(self.resolve((12,1), "cutedsl"), "triton")
    def test_other_platforms(self):
        self.assertEqual(self.resolve(None, "flashinfer"), "triton")
        self.assertEqual(self.resolve((12,0), "flashinfer"), "triton")
        self.assertEqual(self.resolve((9,0)), "flashinfer")
        self.assertEqual(self.resolve((10,3)), "flashinfer")
    def test_geometry_and_cuda(self):
        self.assertEqual(self.resolve((12,1), "flashinfer", head_dim=64), "triton")
        self.assertEqual(self.resolve((12,1), "flashinfer", value_dim=256), "triton")
        self.assertEqual(self.resolve((12,1), "flashinfer", cuda=12), "triton")
    def test_missing_kernel(self):
        with self.assertRaises(RuntimeError):
            self.resolve((12,1), "flashinfer", available=False)
    def test_source_drift(self):
        for bad in (self.source + "\n", patcher.patch(self.source)):
            with self.assertRaises(RuntimeError): patcher.patch(bad)


if __name__ == "__main__":
    unittest.main()
