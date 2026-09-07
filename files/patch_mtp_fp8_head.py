# SPDX-License-Identifier: Apache-2.0
"""Source-exact optional overlay AFTER Mia's reduced-vocabulary MTP patch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "8da66d9f48bd93c935d74e2635c45483dc999c87b94bc4ce828e727eb1713349"
EDITS = (
    ("from .low_latency_gemm import enable_qwen38next_low_latency_gemm\n",
     "from .low_latency_gemm import enable_qwen38next_low_latency_gemm\n"
     "from .spark_mtp_fp8_head import attach_fp8_draft_head\n"),
    ("        _attach_draft_vocab(self)\n        return loaded\n",
     "        _attach_draft_vocab(self)\n        attach_fp8_draft_head(self)\n        return loaded\n"),
    ("        logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)\n",
     "        fp8 = getattr(self, \"_draft_lm_head_fp8\", None)\n"
     "        if fp8 is None:\n"
     "            logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)\n"
     "        else:\n"
     "            logits = torch.ops.vllm.spark_mtp_fp8_head(\n"
     "                hidden_states, weight, fp8, self._draft_lm_head_fp8_scale\n"
     "            )\n"),
)


def patch(source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Reduced MTP source drift ({digest}); refusing unvalidated patch")
    for old, new in EDITS:
        if source.count(old) != 1:
            raise RuntimeError("Expected exactly one MTP source anchor")
        source = source.replace(old, new)
    ast.parse(source)
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = patch(args.source.read_text())
    args.output.write_text(result)
    print(json.dumps({"patch": "mtp_fp8_head", "source_sha256": EXPECTED_SHA256,
                      "output_sha256": hashlib.sha256(result.encode()).hexdigest()}))
