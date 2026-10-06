#!/usr/bin/env python3
"""Attention block size that the engine derives for one MTP k.

The engine sizes the attention block so that one block holds one mamba page.
The GDN conv state in that page has (conv_kernel - 1 + k) rows, so the block
grows with k. The formula follows platforms/interface.py:849-921 of the pinned
image (mamba_cache_mode "align"). It gives the engine log values
"Setting attention block size to 1664 / 1680 / 1728" at k = 3 / 4 / 6.

    python3 mtp_block.py <config.json> <k> [ssm_dtype] [kv_dtype]

It prints "<block> <compress_ratio>" for the start.sh MTP legality guard. It
exits 1 when config.json does not have the keys of a Qwen3.8-Flash-Next
checkpoint; start.sh then uses its old path.
"""
import json
import sys

# Bytes per element. An empty SSM dtype means the checkpoint state (float32).
# KV "auto" means the model dtype (bfloat16).
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4,
               "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}
KERNEL_BLOCK_ALIGN = 16
CONV_BYTES = 2  # the conv states use the model dtype (bfloat16)


def ring_capacity(k, compress_ratio):
    """QSA ring rows (models/qwen3_8_flash_next/common/qsa_cache.py:773-785)."""
    return compress_ratio * -(-(compress_ratio + k) // compress_ratio)


def derived_block(cfg, k, ssm_dtype="", kv_dtype="auto"):
    t = cfg.get("text_config", cfg)
    ssm = DTYPE_BYTES.get(ssm_dtype, 4) if ssm_dtype else 4
    kv = 2 if kv_dtype in ("", "auto") else DTYPE_BYTES[kv_dtype]
    kd, nk = t["linear_key_head_dim"], t["linear_num_key_heads"]
    vd, nv = t["linear_value_head_dim"], t["linear_num_value_heads"]
    gdn_conv_dim = kd * nk * 2 + vd * nv
    gdn_page = (gdn_conv_dim * (t["linear_conv_kernel_dim"] - 1 + k) * CONV_BYTES
                + nv * vd * kd * ssm)
    ple_conv_dim = t["hidden_size"] * t.get("hc_count", 4)
    ple_state_len = (t.get("ple_conv_kernel_size", 4) - 1) * t.get("ngram_size", 3)
    ple_page = ple_conv_dim * (ple_state_len + k) * CONV_BYTES
    attn_bytes_per_token = 2 * t["num_key_value_heads"] * t["head_dim"] * kv
    page = max(gdn_page, ple_page)
    return KERNEL_BLOCK_ALIGN * -(-page // (KERNEL_BLOCK_ALIGN * attn_bytes_per_token))


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    with open(argv[1]) as f:
        cfg = json.load(f)
    k = int(argv[2])
    ssm = argv[3] if len(argv) > 3 else ""
    kv = argv[4] if len(argv) > 4 else "auto"
    try:
        block = derived_block(cfg, k, ssm, kv)
    except KeyError as e:
        print(f"config.json has no {e}", file=sys.stderr)
        return 1
    ratio = int(cfg.get("text_config", cfg).get("indexer_compress_ratio", 4))
    print(f"{block} {ratio}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
