#!/usr/bin/env python3
"""CPU-only base-argv check; never execute the launcher or contact a GPU."""
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PromptTokenDetailsTests(unittest.TestCase):
    def test_base_args(self):
        source = (ROOT / "start.sh").read_text()
        blocks = re.findall(
            r'^\s*VLLM_ARGS=\(\)\n((?:[ \t]*VLLM_ARGS\+=\([^\n]*\)\n)+)',
            source, re.M,
        )
        self.assertEqual(len(blocks), 1)
        argv = subprocess.check_output(
            ["bash", "--noprofile", "--norc", "-c",
             'VLLM_ARGS=()\n' + blocks[0] + '\nprintf "%s\\0" "${VLLM_ARGS[@]}"'],
            env={"PATH": os.defpath},
        ).decode().split("\0")[:-1]
        self.assertEqual(argv.count("--enable-prompt-tokens-details"), 1)
        self.assertNotIn("--enable-prompt-token-details", source)
        self.assertNotIn("--no-enable-prompt-tokens-details", source)
        self.assertEqual(source.count('VLLM_ARGS_STR="${VLLM_ARGS[*]}"'), 1)
        self.assertEqual(source.count('$VLLM_ARGS_STR \\'), 1)


if __name__ == "__main__":
    unittest.main()
