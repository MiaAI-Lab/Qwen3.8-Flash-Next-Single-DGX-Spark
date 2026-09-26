#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Patch vLLM's DefaultModelLoader for faster cold starts (FAST_LOAD=1).

Measured on this Spark with the stock Mia checkpoint, weight loading was ~75%
of the ~11 min to /health, and it was not the disk:

1. Main-model pass, ~450-510 s. The 14 routed-expert shards (4.8 GB each,
   ~15,600 tensors apiece) took ~30 s each on the lazy mmap path. py-spy put
   81% of the pass in `expert_data.copy_(loaded_weight)` in
   fused_moe/routed_experts.py (_load_w13/_load_w2): one small synchronous
   pageable host->device copy per expert tensor, each behind page faults on
   the mmap. A raw sequential read of the same shard takes 4.4 s, and a warm
   page cache barely helps (27.9 s cold vs 26.8 s warm), so the per-copy cost
   is the bottleneck, not I/O.

   Fix: read each of those shards with one sequential host read and one
   host->device copy, and hand out tensors as views of that device buffer, so
   every per-expert copy_() becomes device-to-device (~5 s per shard cold,
   bound by the read). What shaped it on this hardware:

   - Not fastsafetensors (vLLM's --load-format fastsafetensors). It is fast,
     but its C++ reader leaves 8 GiB of pinned host memory (Shmem) allocated
     for the life of the process, whatever the bounce-buffer settings. On
     unified memory vLLM's profile run counts that as used, and the KV pool
     fell from 16.6 to ~5.6 GiB. Its ParallelLoader read-ahead also kept
     ~15 GiB of shard buffers reserved. Plain host memory here, no pinning.
   - One shard at a time. The views keep the shard buffer alive through
     torch's refcount; the last tensor of each shard is yielded as a copy so
     no view pins it once the caller moves on, and it is released with
     torch.cuda.empty_cache() before the next shard is read.
   - Shards holding PLE embedding tensors stay on the lazy path. The PLE
     embedding loader (patch_ple_layer.py) keeps references to some loaded
     tensors (packed codes/scales, regular weights, the global scale) until
     the end of its load_weights(); a retained view would pin a whole shard
     buffer on the device. Those 19 shards were already fast on the lazy
     path (~1 s each). Shards with a dtype the bulk path does not map also
     stay lazy.

2. MTP drafter pass, ~69 s. The drafter is loaded through the same loader,
   so it opened all 35 shards to pick out its mtp.* tensors (one shard) and
   embed_tokens/lm_head (another). The file list is now narrowed through
   model.safetensors.index.json to the shards that hold those names (~25 s).

The PLE offload worker calls get_all_weights() directly rather than
load_weights(), so it never sees either change and keeps the lazy path.

Inert unless VLLM_FAST_LOAD=1 is set in the container (start.sh does that for
FAST_LOAD=1), so the patched file is safe to mount unconditionally.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "default_loader_patched.py.orig")
OUT = os.path.join(HERE, "default_loader_patched.py")

HELPERS = '''def _files_with_keys(
    hf_folder: str, files: list[str], key_substrs: tuple[str, ...]
) -> list[str]:
    """Shards holding a tensor whose name contains one of key_substrs."""
    import json

    index_path = os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_path):
        return []
    with open(index_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    wanted = {
        shard
        for name, shard in weight_map.items()
        if any(s in name for s in key_substrs)
    }
    return [p for p in files if os.path.basename(p) in wanted]


# Checkpoint names the MTP drafter reads: its own mtp.* layers plus the shared
# embedding and head (see _remap_mtp_weight_name in the model's mtp.py).
_FAST_LOAD_DRAFT_KEYS = ("mtp.", "embed_tokens.", "lm_head.", "shared_head.")
# Shards holding these stay on the lazy path (see patch_default_loader.py).
_FAST_LOAD_LAZY_KEYS = ("ple_embedding",)


_FAST_LOAD_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def _fast_load_bulk_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """One shard at a time: one host read, one device copy, then views.

    The yielded tensors are views of the shard's device buffer, so the MoE
    loader's per-expert copy_() runs device-to-device. Views hold the buffer
    through torch's refcount, so a caller that keeps one keeps the buffer
    alive instead of reading freed memory. The last tensor of each shard is
    yielded as a copy, so no view pins a buffer once the caller moves on, and
    the buffer is released before the next shard is read. Plain host memory,
    not pinned: pinned staging stays reserved and on unified memory it comes
    out of the KV pool.
    """
    import json
    import struct

    from tqdm import tqdm

    from vllm.model_executor.model_loader.weight_utils import (
        _natural_sort_key,
        enable_tqdm,
    )

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for path in tqdm(
        sorted(hf_weights_files, key=_natural_sort_key),
        desc="Loading safetensors shards in bulk (FAST_LOAD)",
        disable=not enable_tqdm(use_tqdm_on_load),
    ):
        with open(path, "rb", buffering=0) as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
            header.pop("__metadata__", None)
            size = os.fstat(f.fileno()).st_size - 8 - header_len
            host = torch.empty(size, dtype=torch.uint8)
            view = memoryview(host.numpy())
            got = 0
            while got < size:
                n = f.readinto(view[got:])
                if not n:
                    raise EOFError(f"{path}: truncated safetensors data section")
                got += n
            del view
        buf = host.to(device)
        del host
        names = list(header)
        for i, name in enumerate(names):
            info = header[name]
            begin, end = info["data_offsets"]
            dtype = _FAST_LOAD_DTYPES[info["dtype"]]
            raw = buf[begin:end]
            if begin % dtype.itemsize or i == len(names) - 1:
                raw = raw.clone()
            yield name, raw.view(dtype).reshape(info["shape"])
        del buf, raw
        torch.cuda.empty_cache()


def _fast_load_supported(path: str) -> bool:
    """True when every tensor in the shard has a dtype the bulk path maps."""
    import json
    import struct

    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    return all(v["dtype"] in _FAST_LOAD_DTYPES for v in header.values())


'''

ANCHOR_CLASS = "class DefaultModelLoader(BaseModelLoader):\n"
CLASS_ATTRS = (
    "    # FAST_LOAD pass state (patch_default_loader.py); set in load_weights().\n"
    "    _fast_load_draft: bool = False\n"
    "    _fast_load_main: bool = False\n"
    "\n"
)

ANCHOR_PREPARE = """        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
"""
PREPARE_ADD = """        if self._fast_load_draft and use_safetensors:
            kept = _files_with_keys(hf_folder, hf_weights_files, _FAST_LOAD_DRAFT_KEYS)
            if kept:
                logger.info(
                    "FAST_LOAD: drafter reads %d of %d shard(s)",
                    len(kept),
                    len(hf_weights_files),
                )
                hf_weights_files = kept
        if self._fast_load_main and use_safetensors:
            lazy_files = _files_with_keys(
                hf_folder, hf_weights_files, _FAST_LOAD_LAZY_KEYS
            )
            lazy_files += [
                p
                for p in hf_weights_files
                if p not in lazy_files and not _fast_load_supported(p)
            ]
            fast_files = [p for p in hf_weights_files if p not in lazy_files]
            logger.info(
                "FAST_LOAD: main model, bulk load for %d shard(s), lazy mmap "
                "for %d shard(s)",
                len(fast_files),
                len(lazy_files),
            )
            if self.counter_before_loading_weights == 0.0:
                self.counter_before_loading_weights = time.perf_counter()

            def _fast_load_iter():
                if lazy_files:
                    yield from safetensors_weights_iterator(
                        lazy_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                        safetensors_prefetch_num_threads=(
                            self.load_config.safetensors_prefetch_num_threads
                        ),
                        safetensors_prefetch_block_size=(
                            self.load_config.safetensors_prefetch_block_size
                        ),
                    )
                if fast_files:
                    yield from _fast_load_bulk_iterator(
                        fast_files, self.load_config.use_tqdm_on_load
                    )

            return (
                (source.prefix + name, tensor)
                for (name, tensor) in _fast_load_iter()
            )
"""

ANCHOR_LOAD = (
    "        loaded_weights = model.load_weights("
    "self.get_all_weights(model_config, model))\n"
)
LOAD_REPLACEMENT = """        # FAST_LOAD (patch_default_loader.py). Only this path sets the flags;
        # the PLE offload worker calls get_all_weights() directly and keeps
        # the stock lazy path.
        _fast = os.environ.get("VLLM_FAST_LOAD") == "1"
        _is_draft = type(model).__name__.endswith("MTP")
        self._fast_load_draft = _fast and _is_draft
        self._fast_load_main = _fast and not _is_draft
        try:
            loaded_weights = model.load_weights(
                self.get_all_weights(model_config, model)
            )
        finally:
            self._fast_load_draft = False
            self._fast_load_main = False
"""


def main() -> int:
    if not os.path.isfile(ORIG):
        print(f"patch_default_loader: missing {ORIG}", file=sys.stderr)
        return 1
    src = open(ORIG).read()
    for name, anchor in (
        ("class", ANCHOR_CLASS),
        ("_prepare_weights call", ANCHOR_PREPARE),
        ("load_weights call", ANCHOR_LOAD),
    ):
        count = src.count(anchor)
        if count != 1:
            print(
                f"patch_default_loader: anchor '{name}' matched {count} times "
                "(expected 1); the image's default_loader.py changed",
                file=sys.stderr,
            )
            return 1
    src = src.replace(ANCHOR_CLASS, HELPERS + ANCHOR_CLASS + CLASS_ATTRS)
    src = src.replace(ANCHOR_PREPARE, ANCHOR_PREPARE + PREPARE_ADD)
    src = src.replace(ANCHOR_LOAD, LOAD_REPLACEMENT)
    with open(OUT, "w") as f:
        f.write(src)
    print(f"patch_default_loader: wrote {os.path.basename(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
