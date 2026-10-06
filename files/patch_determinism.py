#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = "VLLM_QSA_DET_TOPK"


def apply(src: str, name: str, edits: list[tuple[str, str]]) -> str:
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(f"{name}: anchor {i} not unique/missing (count={count}):\n{old[:180]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"{name}: patched source does not parse: {exc}")
    return src


QSA_EDITS = [
    (
        "_TOPK_WORKSPACE_BYTES = 1024 * 1024\n",
        "_TOPK_WORKSPACE_BYTES = 1024 * 1024\n"
        "_SORTED_TOPK = os.getenv(\"VLLM_QSA_DET_TOPK\", \"0\") == \"1\"\n"
        "_INT32_MAX = torch.iinfo(torch.int32).max\n",
    ),
    (
        "        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n",
        "        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n"
        "        if _SORTED_TOPK:\n"
        "            keyed = torch.where(blocks < 0, _INT32_MAX, blocks).sort(dim=1).values\n"
        "            blocks.copy_(torch.where(keyed == _INT32_MAX, -1, keyed))\n",
    ),
]

MOE_EDITS = [
    (
        "import torch\n",
        "import os\n\nimport torch\n",
    ),
    (
        "            use_w4_group_scaling=use_w4_group_scaling,\n        )\n",
        "            use_w4_group_scaling=use_w4_group_scaling,\n"
        "            use_fused_finalize=os.getenv(\"VLLM_MOE_DET_FINALIZE\", \"0\") != \"1\",\n"
        "        )\n",
    ),
]


def main() -> None:
    qsa = os.path.join(HERE, "qsa_ops_patched.py")
    if not os.path.exists(qsa):
        sys.exit("qsa_ops_patched.py missing: run patch_qsa_fp8_kv.py first")
    src = open(qsa).read()
    if MARKER not in src:
        if "import os\n" not in src:
            QSA_EDITS.insert(0, ("import torch\n", "import os\n\nimport torch\n"))
        patched = apply(src, "qsa_ops_patched.py", QSA_EDITS)
        open(qsa, "w").write(patched)
        print("patched qsa_ops_patched.py (sorted top-k)")

    moe_orig = os.path.join(HERE, "determinism", "orig", "flashinfer_cutlass_moe.py")
    moe_dest = os.path.join(HERE, "determinism", "flashinfer_cutlass_moe.py")
    if not os.path.exists(moe_orig):
        sys.exit(f"missing {moe_orig} (start.sh extracts it from the image)")
    src = open(moe_orig).read()
    if "use_fused_finalize" in src:
        print("flashinfer_cutlass_moe.py: image already has use_fused_finalize; mounting stock file")
        open(moe_dest, "w").write(src)
        return
    patched = apply(src, "flashinfer_cutlass_moe.py", MOE_EDITS)
    open(moe_dest, "w").write(patched)
    print("patched flashinfer_cutlass_moe.py (VLLM_MOE_DET_FINALIZE)")


if __name__ == "__main__":
    main()
