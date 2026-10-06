#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
import ast
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "mtp_v030_patched.py.orig")
OUT = os.path.join(HERE, "mtp_v030_patched.py")


def _blocks():
    spec = importlib.util.spec_from_file_location("patch_mtp_draft_vocab", os.path.join(HERE, "patch_mtp_draft_vocab.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DRAFT_VOCAB_BLOCK, module.GET_TOP_TOKENS


def main() -> None:
    if not os.path.isfile(ORIG):
        sys.exit(f"ERROR: missing {ORIG} (start.sh extracts it from the image)")
    src = open(ORIG).read()
    if "_attach_draft_vocab" in src:
        sys.exit(f"ERROR: {ORIG} is already patched")
    draft_vocab_block, get_top_tokens = _blocks()
    edits = [
        ("import torch\nfrom torch import nn\n", "import os\n\nimport torch\nfrom torch import nn\n"),
        ("from vllm.distributed import get_pp_group\n",
         "from vllm.distributed import get_pp_group\n"
         "from vllm.logger import init_logger\n"),
        ("\nclass Qwen4ExpMultiTokenPredictor(nn.Module):\n",
         "\nlogger = init_logger(__name__)\n" + draft_vocab_block + "\nclass Qwen4ExpMultiTokenPredictor(nn.Module):\n"),
        ("        return self.logits_processor(self.lm_head, hidden_states)\n",
         "        return self.logits_processor(self.lm_head, hidden_states)\n" + get_top_tokens),
        ("        return loader.load_weights(remap_weight_names(), mapper=mapper)\n",
         "        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)\n"
         "        _attach_draft_vocab(self)\n"
         "        return loaded\n"),
    ]
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(f"mtp_v030_patched: anchor {i} not unique/missing (count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"mtp_v030_patched: patched source does not parse: {exc}")
    open(OUT, "w").write(src)
    print("patched mtp_v030_patched.py")


if __name__ == "__main__":
    main()
