# SPDX-License-Identifier: Apache-2.0
"""CPU source-scope checks, against Mia's generated reduced-vocabulary MTP.

Run after start.sh --no-launch (which generates files/mtp_patched.py), or set
MTP_TEST_SOURCE to the validated source. Does not load the model or GPU.
"""
import ast
import os
from pathlib import Path
import unittest

import patch_mtp_fp8_head as patcher


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get("MTP_TEST_SOURCE",
                          str(Path(__file__).with_name("mtp_patched.py")))).read_text()

    def test_only_draft_selection_and_load_change(self):
        def bodies(source):
            tree = ast.parse(source)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3_8FlashNextMTP")
            return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = bodies(self.source), bodies(patcher.patch(self.source))
        self.assertEqual({k for k in before if before[k] != after[k]}, {"get_top_tokens", "load_weights"})
        for unchanged in ("compute_logits", "forward", "__init__", "embed_input_ids"):
            self.assertEqual(before[unchanged], after[unchanged])

    def test_other_top_level_functions_unchanged(self):
        def functions(source):
            return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(functions(self.source), functions(patcher.patch(self.source)))

    def test_source_drift(self):
        for bad in (self.source + "\n", patcher.patch(self.source)):
            with self.assertRaises(RuntimeError): patcher.patch(bad)

    def test_runtime_has_explicit_opt_in_and_nonpersistent_buffers(self):
        source = Path(__file__).with_name("spark_mtp_fp8_head.py").read_text()
        tree = ast.parse(source)
        attach = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attach_fp8_draft_head")
        self.assertIsInstance(attach.body[0], ast.If)
        self.assertIn("VLLM_MTP_DRAFT_HEAD_FP8", ast.unparse(attach.body[0].test))
        self.assertIsInstance(attach.body[0].body[0], ast.Return)
        registrations = [n for n in ast.walk(attach) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "register_buffer"]
        self.assertEqual(len(registrations), 2)
        for call in registrations:
            self.assertTrue(any(k.arg == "persistent" and isinstance(k.value, ast.Constant)
                                and k.value.value is False for k in call.keywords))


if __name__ == "__main__": unittest.main()
