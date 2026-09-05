#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Patch nvidia/mtp.py for reduced-vocabulary drafting (FR-Spec style).

The MTP drafter carries its own BF16 ParallelLMHead over the full 248,320-token
vocabulary: 1.27 GB, read once per draft step. At MTP 3 that is three of the
four lm_head reads in an engine step, 3.8 GB of the ~15.6 GB a single-stream
step moves.

Draft sampling is greedy (`draft_sample_method` defaults to "greedy", and the
speculator only builds `draft_logits` for the "probabilistic" method), so the
drafter needs an argmax and nothing else. An argmax over a frequency-ranked
subset of the vocabulary is wrong only for tokens outside the subset, and a
wrong draft is *rejected by the target model at verification*, exactly like any
other bad draft. So this trades acceptance for bandwidth and cannot change what
the server emits.

The reduced head is used only through `get_top_tokens`, which the speculator
calls when `use_local_argmax_reduction` is set. `compute_logits` is left on the
full head, so every other path keeps full-vocabulary behaviour.

TP=1 only: the reduced head is a plain matmul, not a vocab-parallel one. This
repo is a single-Spark deployment, and the patch refuses to engage otherwise.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "mtp_patched.py.orig")
OUT = os.path.join(HERE, "mtp_patched.py")

DRAFT_VOCAB_BLOCK = '''

def _attach_draft_vocab(model: nn.Module) -> None:
    """Slice the drafter's lm_head down to VLLM_MTP_DRAFT_VOCAB's token ids.

    The file is one integer token id per line (see files/build_draft_vocab.py).
    Leaves the full head in place and untouched; only get_top_tokens below
    reads the slice.
    """
    path = os.environ.get("VLLM_MTP_DRAFT_VOCAB", "").strip()
    if not path:
        return
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2:
        logger.warning("MTP draft vocab: no 2-D lm_head weight; skipping.")
        return
    if getattr(lm_head, "tp_size", 1) != 1:
        logger.warning("MTP draft vocab: TP>1 is unsupported; skipping.")
        return
    scale = getattr(model.logits_processor, "scale", 1.0)
    if scale <= 0.0:
        logger.warning("MTP draft vocab: non-positive logit scale; skipping.")
        return

    org_vocab = int(getattr(lm_head, "org_vocab_size", weight.shape[0]))
    with open(path) as handle:
        ids = sorted({int(line) for line in handle if line.strip()})
    ids = [i for i in ids if 0 <= i < org_vocab]
    if not ids or len(ids) >= org_vocab:
        logger.warning(
            "MTP draft vocab: %d usable ids against a %d vocabulary; skipping.",
            len(ids), org_vocab,
        )
        return

    index = torch.tensor(ids, dtype=torch.long, device=weight.device)
    model.register_buffer(
        "_draft_lm_head_weight",
        weight.data.index_select(0, index).contiguous(),
        persistent=False,
    )
    model.register_buffer(
        "_draft_id_to_target_id",
        index.to(torch.int32),
        persistent=False,
    )
    full_gib = weight.numel() * weight.element_size() / 2**30
    cut_gib = model._draft_lm_head_weight.numel() * weight.element_size() / 2**30
    logger.info(
        "MTP draft vocab: %d of %d tokens (%.1f%%); draft lm_head %.2f -> %.2f "
        "GiB per draft step, %.2f GiB saved per step at MTP %d",
        len(ids), org_vocab, 100.0 * len(ids) / org_vocab, full_gib, cut_gib,
        (full_gib - cut_gib) * 3, 3,
    )

'''

GET_TOP_TOKENS = '''
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft token ids, over the reduced vocabulary when present.

        Positive logit scaling and the tanh soft cap are both monotonic, so the
        argmax is unchanged by skipping them; _attach_draft_vocab refuses to
        engage when the scale is not positive.
        """
        weight = getattr(self, "_draft_lm_head_weight", None)
        if weight is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
        return self._draft_id_to_target_id[logits.argmax(dim=-1)].to(torch.long)

'''


def patch(name: str, edits: list[tuple[str, str]]) -> None:
    src = open(ORIG).read()
    for old, new in edits:
        count = src.count(old)
        if count != 1:
            raise AssertionError(
                f"{name}: anchor not unique/missing (count={count}):\n{old[:200]}"
            )
        src = src.replace(old, new)
    open(OUT, "w").write(src)
    print("patched", name)


def main() -> None:
    if not os.path.isfile(ORIG):
        print(f"ERROR: missing {ORIG}", file=sys.stderr)
        sys.exit(1)

    patch("mtp_patched", [
        # os + a logger, and the slicing helper at module scope.
        (
            "from vllm.compilation.decorators import support_torch_compile\n",
            "import os\n\n"
            "from vllm.compilation.decorators import support_torch_compile\n"
            "from vllm.logger import init_logger\n",
        ),
        # Module scope, before the first helper: the MTP class itself sits
        # under a @support_torch_compile decorator, so nothing can go between.
        (
            "def _remap_ignored_layers(\n",
            "logger = init_logger(__name__)\n"
            + DRAFT_VOCAB_BLOCK
            + "\ndef _remap_ignored_layers(\n",
        ),
        # get_top_tokens beside compute_logits, which stays full-vocabulary.
        (
            "    def compute_logits(\n"
            "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
            "    ) -> torch.Tensor | None:\n"
            "        return self.logits_processor(self.lm_head, hidden_states)\n",
            "    def compute_logits(\n"
            "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
            "    ) -> torch.Tensor | None:\n"
            "        return self.logits_processor(self.lm_head, hidden_states)\n"
            + GET_TOP_TOKENS,
        ),
        # Slice once the real weights are in.
        (
            "        return loader.load_weights(remap_weight_names())\n",
            "        loaded = loader.load_weights(remap_weight_names())\n"
            "        _attach_draft_vocab(self)\n"
            "        return loaded\n",
        ),
    ])
    print("ok")


if __name__ == "__main__":
    main()
