import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "files"
PATCH = FILES / "patch_determinism.py"
QSA_ORIG = FILES / "qsa_ops_patched.py.orig"
MOE_ORIG = FILES / "determinism" / "orig" / "flashinfer_cutlass_moe.py"
HAVE_SOURCES = QSA_ORIG.is_file() and MOE_ORIG.is_file()


def _patch_module():
    ns = {"__name__": "patch_determinism", "__file__": str(PATCH)}
    exec(compile(PATCH.read_text(), str(PATCH), "exec"), ns)
    return ns


def _sorted_topk(blocks):
    ns = _patch_module()
    inserted = ns["QSA_EDITS"][-1][1].split("        if _SORTED_TOPK:\n", 1)[1]
    body = "\n".join(line[12:] for line in inserted.splitlines())
    scope = {"torch": torch, "blocks": blocks, "_INT32_MAX": torch.iinfo(torch.int32).max}
    exec(body, scope)
    return blocks


class SortedTopk(unittest.TestCase):
    def test_full_row_is_sorted_and_keeps_the_set(self):
        g = torch.Generator().manual_seed(0)
        blocks = torch.randperm(4096, generator=g)[:512].int().view(1, 512).repeat(3, 1)
        blocks[1] = blocks[1][torch.randperm(512, generator=g)]
        blocks[2] = blocks[2].flip(0)
        want = blocks[0].sort().values
        out = _sorted_topk(blocks.clone())
        for row in out:
            self.assertTrue(torch.equal(row, want))

    def test_short_row_keeps_minus_one_tail(self):
        row = torch.cat([torch.arange(300, dtype=torch.int32), torch.full((212,), -1, dtype=torch.int32)])
        out = _sorted_topk(row.view(1, -1).clone())
        self.assertTrue(torch.equal(out[0], row))

    def test_interleaved_minus_one_moves_to_the_tail(self):
        row = torch.tensor([[5, -1, 2, -1, 9]], dtype=torch.int32)
        self.assertEqual(_sorted_topk(row).tolist(), [[2, 5, 9, -1, -1]])


@unittest.skipUnless(HAVE_SOURCES, "no extracted image sources; run ./start.sh --no-launch first")
class Generator(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "determinism" / "orig").mkdir(parents=True)
        shutil.copy(PATCH, self.tmp / PATCH.name)
        shutil.copy(QSA_ORIG, self.tmp / "qsa_ops_patched.py")
        shutil.copy(MOE_ORIG, self.tmp / "determinism" / "orig" / MOE_ORIG.name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run([sys.executable, str(self.tmp / PATCH.name)], capture_output=True, text=True)

    def test_patches_both_files_and_is_idempotent(self):
        self.assertEqual(self.run_patch().returncode, 0)
        qsa = (self.tmp / "qsa_ops_patched.py").read_text()
        moe = (self.tmp / "determinism" / "flashinfer_cutlass_moe.py").read_text()
        ast.parse(qsa)
        ast.parse(moe)
        self.assertIn('os.getenv("VLLM_QSA_DET_TOPK", "0") == "1"', qsa)
        self.assertIn('use_fused_finalize=os.getenv("VLLM_MOE_DET_FINALIZE", "0") != "1"', moe)
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.tmp / "qsa_ops_patched.py").read_text(), qsa)
        self.assertEqual(qsa.count("if _SORTED_TOPK:"), 1)

    def test_missing_moe_source_fails(self):
        os.unlink(self.tmp / "determinism" / "orig" / MOE_ORIG.name)
        result = self.run_patch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("start.sh extracts it", result.stderr)

    def test_moved_anchor_fails_without_writing(self):
        qsa = self.tmp / "qsa_ops_patched.py"
        qsa.write_text(qsa.read_text().replace("topk_op(logits,", "topk_op( logits,"))
        before = qsa.read_text()
        self.assertNotEqual(self.run_patch().returncode, 0)
        self.assertEqual(qsa.read_text(), before)


if __name__ == "__main__":
    unittest.main()
