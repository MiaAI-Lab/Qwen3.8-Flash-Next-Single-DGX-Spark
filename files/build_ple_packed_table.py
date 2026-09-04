#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Build a packed NVFP4 PLE table file for memory-mapped CPU offload.

The checkpoint stores the PLE n-gram table as 128 row shards, each split into
4-bit codes (uint8 [rows, head_dim/2]) and FP8 block scales ([rows, head_dim/16]).
vLLM's NVFP4 PLE lookup returns, per row, ``cat(codes, scales.view(uint8))``.
This script writes exactly that layout as one flat file: [total_rows, 90] uint8,
row r of shard i at index i*shard_rows + r. The offload worker memory-maps it,
so the 26.8 GiB table lives in the (evictable) page cache instead of RSS.

Streams shard-by-shard with numpy memmaps; peak RAM is well under 1 GiB.

Usage: build_ple_packed_table.py <snapshot_dir> <out_dir>
"""
import json, os, struct, sys, time
import numpy as np

snap, out_dir = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
prefix_of = {}
for k in idx:
    if ".ngram_embedding.shard_0.weight" in k and not k.endswith("weight_scale"):
        prefix_of[k[: k.index(".shard_0.weight")]] = True
if not prefix_of:
    sys.exit("no PLE ngram_embedding shards in index")

headers = {}
def header(fname):
    if fname not in headers:
        with open(os.path.join(snap, fname), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[fname] = (json.loads(f.read(n)), 8 + n)
    return headers[fname]

def view(name):
    fname = idx[name]
    h, base = header(fname)
    meta = h[name]
    start, end = meta["data_offsets"]
    mm = np.memmap(os.path.join(snap, fname), dtype=np.uint8, mode="r",
                   offset=base + start, shape=(end - start,))
    return mm.reshape(meta["shape"]), meta["dtype"]

os.makedirs(out_dir, exist_ok=True)
for prefix in prefix_of:
    # vLLM name: strip leading "model." and map language_model -> language_model.model
    vname = prefix
    if vname.startswith("model.language_model."):
        vname = "language_model.model." + vname[len("model.language_model."):]
    shards = sorted({int(k[len(prefix) + len(".shard_"):].split(".")[0])
                     for k in idx if k.startswith(prefix + ".shard_")})
    assert shards == list(range(len(shards))), shards
    w0, dt = view(f"{prefix}.shard_0.weight"); assert dt == "U8", dt
    s0, dt = view(f"{prefix}.shard_0.weight_scale"); assert dt == "F8_E4M3", dt
    rows, cw = w0.shape; sw = s0.shape[1]
    width = cw + sw
    out_name = os.path.join(out_dir, vname + ".packed_u8")
    meta = {"rows_per_shard": rows, "num_shards": len(shards), "row_width": width,
            "codes_width": cw, "scales_width": sw, "total_rows": rows * len(shards),
            "snapshot": os.path.basename(os.path.normpath(snap))}
    if os.path.exists(out_name) and os.path.getsize(out_name) == rows * len(shards) * width:
        print("exists:", out_name); continue
    print(f"building {out_name}: {len(shards)} shards x {rows} rows x {width} B = "
          f"{rows*len(shards)*width/2**30:.2f} GiB", flush=True)
    t0 = time.time()
    tmp = out_name + ".tmp"
    CH = 1 << 19
    with open(tmp, "wb") as out:
        for i in shards:
            w, _ = view(f"{prefix}.shard_{i}.weight")
            s, _ = view(f"{prefix}.shard_{i}.weight_scale")
            assert w.shape == (rows, cw) and s.shape == (rows, sw), (i, w.shape, s.shape)
            for c in range(0, rows, CH):
                np.concatenate([w[c:c + CH], s[c:c + CH]], axis=1).tofile(out)
            if i % 8 == 0:
                print(f"  shard {i}/{len(shards)} {time.time()-t0:.0f}s", flush=True)
    assert os.path.getsize(tmp) == rows * len(shards) * width
    os.rename(tmp, out_name)
    json.dump(meta, open(out_name + ".json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
