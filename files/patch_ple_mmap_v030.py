#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "v030_ple", "orig", "ngram_embedding.py")
OUT = os.path.join(HERE, "v030_ple", "ngram_embedding.py")

HELPERS = '''
@triton.jit
def _lookup_ple_rows_from_host_addr_kernel(
    weight_addr,
    ids_ptr,
    output_ptr,
    row_bytes,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_B: tl.constexpr,
):
    weight_ptr = weight_addr.to(tl.pointer_type(tl.uint8))
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_B)
    store_mask = offsets < row_bytes
    values = tl.load(
        weight_ptr + local_idx * row_bytes + offsets,
        mask=store_mask & in_range,
        other=0,
    )
    tl.store(output_ptr + row_id * row_bytes + offsets, values, mask=store_mask)


_PLE_MMAP_FORMAT = "ple-mmap-v1"




def _ple_mmap_sidecar(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def _ple_fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ple_checkpoint_identity() -> dict:
    model_config = get_current_vllm_config().model_config
    model = str(model_config.model_weights or model_config.model)
    snapshot = None
    if os.path.isdir(model):
        model = os.path.realpath(model)
        snapshot = os.path.basename(model)
    else:
        from huggingface_hub import try_to_load_from_cache

        try:
            cached = try_to_load_from_cache(
                model, "config.json", revision=model_config.revision
            )
        except ValueError:
            cached = None
        if isinstance(cached, str):
            snapshot = os.path.basename(os.path.dirname(cached))
    return {"model": model, "revision": model_config.revision, "snapshot": snapshot}


def _ple_mmap_location(prefix: str, rank: int) -> tuple[str, dict] | None:
    root = os.environ.get("VLLM_PLE_MMAP_DIR", "").strip()
    if not root:
        return None
    identity = _ple_checkpoint_identity()
    key = identity["snapshot"] or hashlib.sha256(identity["model"].encode()).hexdigest()
    os.makedirs(root, exist_ok=True)
    name = f"{prefix.replace('/', '_')}.{key[:12]}.etp{rank}.bin"
    return os.path.join(root, name), identity


def _ple_mmap_fingerprint(identity: dict, shape: tuple[int, int], dtype) -> dict:
    return {
        "format": _PLE_MMAP_FORMAT,
        **identity,
        "shape": list(shape),
        "dtype": str(dtype),
    }


def _ple_mmap_matches(path: str, fingerprint: dict, nbytes: int) -> bool:
    try:
        with open(_ple_mmap_sidecar(path)) as handle:
            stored = json.load(handle)
        return stored == fingerprint and os.path.getsize(path) == nbytes
    except (OSError, ValueError):
        return False


def _ple_mmap_open(
    path: str, shape: tuple[int, int], dtype, fingerprint: dict
) -> tuple[torch.Tensor, bool]:
    nbytes = shape[0] * shape[1] * dtype.itemsize
    ready = _ple_mmap_matches(path, fingerprint, nbytes)
    if not ready:
        sidecar = _ple_mmap_sidecar(path)
        if os.path.exists(sidecar):
            os.unlink(sidecar)
            _ple_fsync_dir(os.path.dirname(path))
        with open(path, "ab") as handle:
            handle.truncate(nbytes)
    raw = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    advice = int(os.environ.get("VLLM_PLE_MMAP_ADVICE", "1"))
    if advice:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        start = raw.data_ptr() & ~(mmap.PAGESIZE - 1)
        length = raw.data_ptr() + nbytes - start
        rc = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), advice)
        logger.info("PLE mmap %s: madvise(%d) rc=%d", path, advice, rc)
    return raw.view(dtype).view(*shape), ready


def _ple_mmap_commit(path: str, weight: torch.Tensor, fingerprint: dict) -> None:
    nbytes = weight.numel() * weight.element_size()
    start = weight.data_ptr() & ~(mmap.PAGESIZE - 1)
    length = weight.data_ptr() + nbytes - start
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.msync(ctypes.c_void_p(start), ctypes.c_size_t(length), 4):
        raise OSError(ctypes.get_errno(), f"msync failed for {path}")
    sidecar = _ple_mmap_sidecar(path)
    with open(sidecar + ".tmp", "w") as handle:
        json.dump(fingerprint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(sidecar + ".tmp", sidecar)
    _ple_fsync_dir(os.path.dirname(path))

'''

EDITS = [
    (
        "from abc import ABC, abstractmethod\n",
        "import ctypes\nimport hashlib\nimport json\nimport mmap\nimport os\n"
        "from abc import ABC, abstractmethod\n",
    ),
    (
        "    requires_device_loading: bool = False\n",
        "    requires_device_loading: bool = False\n"
        "\n"
        "    def process_weights_after_loading(self, layer: nn.Module) -> None:\n"
        "        commit = getattr(layer, \"commit_mmap_table\", None)\n"
        "        if commit is not None:\n"
        "            commit()\n",
    ),
    (
        "            raise ValueError(\"FP8 PLE checkpoint is missing its global scale\")\n",
        "            raise ValueError(\"FP8 PLE checkpoint is missing its global scale\")\n"
        "        super().process_weights_after_loading(layer)\n",
    ),
    (
        "class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):\n",
        HELPERS.lstrip("\n") + "\nclass Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):\n",
    ),
    (
        "        if not is_uva_available():\n"
        "            raise RuntimeError(\"Engram CPU offload requires UVA support\")\n",
        "        location = _ple_mmap_location(prefix, get_etp_group().rank_in_group)\n"
        "        self._mmap_path, self._mmap_identity = location or (None, None)\n"
        "        self._mmap_ready = False\n"
        "        self._mmap_shards: set[int] = set()\n"
        "        if self._mmap_path is None and not is_uva_available():\n"
        "            raise RuntimeError(\"Engram CPU offload requires UVA support\")\n",
    ),
    (
        "        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)\n"
        "        self._block_d = triton.next_power_of_2(self.embedding_dim)\n"
        "        self._prefetch_stream = torch.cuda.Stream(device=self._uva_weight.device)\n",
        "        if self._mmap_path is None:\n"
        "            self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)\n"
        "            device = self._uva_weight.device\n"
        "        else:\n"
        "            self._uva_weight = None\n"
        "            device = torch.device(\"cuda\", torch.cuda.current_device())\n"
        "            logger.info(\n"
        "                \"PLE %s: %s file-backed table %s (%.2f GiB), read over ATS\",\n"
        "                prefix,\n"
        "                \"reused\" if self._mmap_ready else \"building\",\n"
        "                self._mmap_path,\n"
        "                self.weight.numel() * self.weight.element_size() / 2**30,\n"
        "            )\n"
        "        self._block_d = triton.next_power_of_2(self.embedding_dim)\n"
        "        self._row_bytes = self.embedding_dim * self.weight.element_size()\n"
        "        self._block_b = triton.next_power_of_2(self._row_bytes)\n"
        "        self._prefetch_stream = torch.cuda.Stream(device=device)\n",
    ),
    (
        "            device=self._uva_weight.device,\n        )\n        self._output_dim",
        "            device=device,\n        )\n        self._output_dim",
    ),
    (
        "        \"\"\"Allocate the complete PLE weight directly in pinned CPU memory.\"\"\"\n"
        "        return torch.empty(\n",
        "        \"\"\"Allocate the complete PLE weight directly in pinned CPU memory.\"\"\"\n"
        "        if self._mmap_path is not None:\n"
        "            self._mmap_fingerprint = _ple_mmap_fingerprint(\n"
        "                self._mmap_identity, (num_embeddings, embedding_dim), dtype\n"
        "            )\n"
        "            weight, self._mmap_ready = _ple_mmap_open(\n"
        "                self._mmap_path,\n"
        "                (num_embeddings, embedding_dim),\n"
        "                dtype,\n"
        "                self._mmap_fingerprint,\n"
        "            )\n"
        "            return weight\n"
        "        return torch.empty(\n",
    ),
    (
        "    def _lookup(\n        self,\n        input_ids: torch.Tensor,\n",
        "    def commit_mmap_table(self) -> None:\n"
        "        if self._mmap_path is None or self._mmap_ready:\n"
        "            return\n"
        "        expected = getattr(self, \"_mmap_expected_shards\", None)\n"
        "        if expected is None or len(self._mmap_shards) != expected:\n"
        "            logger.warning(\n"
        "                \"PLE %s: loaded %d/%s shards, table not persisted\",\n"
        "                self._mmap_path,\n"
        "                len(self._mmap_shards),\n"
        "                expected,\n"
        "            )\n"
        "            return\n"
        "        _ple_mmap_commit(self._mmap_path, self.weight, self._mmap_fingerprint)\n"
        "        self._mmap_ready = True\n"
        "        logger.info(\"PLE %s: built persistent table\", self._mmap_path)\n"
        "\n"
        "    def _lookup(\n        self,\n        input_ids: torch.Tensor,\n",
    ),
    (
        "            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](\n"
        "                self._uva_weight,\n",
        "            if self._uva_weight is None:\n"
        "                _lookup_ple_rows_from_host_addr_kernel[(flat_ids.numel(),)](\n"
        "                    self.weight.data_ptr(),\n"
        "                    flat_ids,\n"
        "                    output.view(torch.uint8),\n"
        "                    self._row_bytes,\n"
        "                    self.shard_indices.org_vocab_start_index,\n"
        "                    self.shard_indices.org_vocab_end_index,\n"
        "                    BLOCK_B=self._block_b,\n"
        "                )\n"
        "                return output\n"
        "            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](\n"
        "                self._uva_weight,\n",
    ),
    (
        "                embedding.weight.weight_loader(\n"
        "                    embedding.weight,\n"
        "                    loaded_weight,\n"
        "                    checkpoint_start=checkpoint_start,\n"
        "                )\n",
        "                if getattr(embedding, \"_mmap_path\", None) is not None:\n"
        "                    embedding._mmap_shards.add(shard_index)\n"
        "                    embedding._mmap_expected_shards = (\n"
        "                        embedding.org_vocab_size + shard_size - 1\n"
        "                    ) // shard_size\n"
        "                if not getattr(embedding, \"_mmap_ready\", False):\n"
        "                    embedding.weight.weight_loader(\n"
        "                        embedding.weight,\n"
        "                        loaded_weight,\n"
        "                        checkpoint_start=checkpoint_start,\n"
        "                    )\n",
    ),
]


def patch(src: str) -> str:
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"ngram_embedding.py: anchor {i} not unique/missing (count={count}):\n{old[:180]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"ngram_embedding.py: patched source does not parse: {exc}")
    return src


def main() -> None:
    orig = sys.argv[1] if len(sys.argv) > 1 else ORIG
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    with open(orig) as handle:
        src = patch(handle.read())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(src)
    print(f"patched {out}")


if __name__ == "__main__":
    main()
