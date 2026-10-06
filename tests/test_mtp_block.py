"""CPU check for files/mtp_block.py, the per-k block of the start.sh MTP guard.

The expected blocks are engine log values ("Setting attention block size to")
with the .env.sample dtypes (MAMBA_SSM_CACHE_DTYPE=bfloat16, KV_CACHE_DTYPE=fp8):
1664 at k=3, 1680 at k=4 and 1728 at k=6.

    python3 -m unittest tests/test_mtp_block.py
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "files" / "mtp_block.py"
spec = importlib.util.spec_from_file_location("mtp_block", SRC)
mtp_block = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtp_block)

# Checkpoint values (Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 config.json).
CFG = {"text_config": {
    "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_value_head_dim": 128, "linear_num_value_heads": 48,
    "linear_conv_kernel_dim": 4, "hidden_size": 2560, "hc_count": 4,
    "ple_conv_kernel_size": 4, "ngram_size": 3, "num_key_value_heads": 2,
    "head_dim": 256, "indexer_compress_ratio": 4}}


def legal(k, block):
    return block % mtp_block.ring_capacity(k, 4) == 0


class DerivedBlock(unittest.TestCase):
    def test_engine_log_values(self):
        for k, want in ((3, 1664), (4, 1680), (6, 1728)):
            self.assertEqual(mtp_block.derived_block(CFG, k, "bfloat16", "fp8"), want, k)

    def test_k6_is_legal_and_k5_is_not(self):
        # The old guard used one block (848) for every k and rejected k=6 and 16.
        blocks = {k: mtp_block.derived_block(CFG, k, "bfloat16", "fp8") for k in range(17)}
        legal_ks = [k for k, b in blocks.items() if legal(k, b)]
        self.assertEqual(legal_ks, [0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 16])

    def test_block_grows_with_k(self):
        blocks = [mtp_block.derived_block(CFG, k, "bfloat16", "fp8") for k in range(17)]
        self.assertEqual(blocks, sorted(blocks))
        self.assertLess(blocks[0], blocks[16])

    def test_old_fallback_block_is_one_k(self):
        # The 848 fallback table is the block of k=4 at SSM bfloat16, KV auto.
        self.assertEqual(mtp_block.derived_block(CFG, 4, "bfloat16", "auto"), 848)
        self.assertNotEqual(mtp_block.derived_block(CFG, 6, "bfloat16", "auto"), 848)

    def test_dtypes_change_the_block(self):
        fp8 = mtp_block.derived_block(CFG, 4, "bfloat16", "fp8")
        self.assertLess(mtp_block.derived_block(CFG, 4, "bfloat16", "auto"), fp8)
        self.assertGreater(mtp_block.derived_block(CFG, 4, "", "fp8"), fp8)

    def test_cli_output_and_missing_keys(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d, "good.json"), Path(d, "bad.json")
            good.write_text(json.dumps(CFG))
            bad.write_text(json.dumps({"hidden_size": 2560}))
            out = subprocess.run([sys.executable, SRC, good, "6", "bfloat16", "fp8"],
                                 capture_output=True, text=True, check=True)
            self.assertEqual(out.stdout.strip(), "1728 4")
            rc = subprocess.run([sys.executable, SRC, bad, "6"], capture_output=True, text=True)
            self.assertEqual(rc.returncode, 1)
            self.assertEqual(rc.stdout, "")


if __name__ == "__main__":
    unittest.main()
