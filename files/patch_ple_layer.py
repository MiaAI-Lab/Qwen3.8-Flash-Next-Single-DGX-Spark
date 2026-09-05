#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Patch ple_layer.py for mixed NVFP4 + FP8 PLE checkpoint loading.

Ports the PLE quant dispatch from vLLM PR #53899 (qwen4_exp) onto the
qwen3_8_flash_next NVIDIA ple_layer shipped in vllm/vllm-openai:qwen38-flash-next.

Checkpoints declare their PLE table format in text_config.ple_embedding_dtype:
  - "nvfp4"  -> uint8 packed rows (dim/2) + fp8 block scales + fp32 global scale
  - "float8_e4m3fn" / fp8 -> FP8 rows (full dim) + one global bf16/fp32 scale

The stock image only selects the FP8 path when the whole checkpoint is Fp8Config,
which breaks NVFP4-body checkpoints (both in-family NVFP4 PLE and excluded FP8 PLE).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "ple_layer_patched.py.orig")
OUT = os.path.join(HERE, "ple_layer_patched.py")

PREFETCH_BLOCK = '''
_PLE_PAGE_SHIFT = 12
_PLE_PAGE = 1 << _PLE_PAGE_SHIFT


def _ple_prefetch_rows(fd, ids, row_width) -> None:
    """Queue every page a row gather will touch, in one batch.

    torch.index_select over the 27 GiB packed table is one CPU thread walking
    a list of unrelated 90-byte rows. PyTorch only parallelises index_select
    past a ~32k-element grain, and a decode batch gathers 64-320 rows, so each
    missing 4 KiB page is faulted in on its own at queue depth 1 (~77 us on
    this NVMe) while the GPU worker spins waiting for the handshake. Naming
    all of the pages up front lets the device see them at once; the gather
    that follows finds them resident or already in flight.

    Advisory only: any failure just leaves the original fault-per-row path.
    """
    if fd is None or ids is None or ids.numel() == 0:
        return
    try:
        offsets = ids.to(torch.int64) * row_width
        pages = torch.cat(
            (
                offsets >> _PLE_PAGE_SHIFT,
                (offsets + row_width - 1) >> _PLE_PAGE_SHIFT,
            )
        )
        # unique() sorts, so the reads are issued in ascending file order.
        for page in torch.unique(pages).tolist():
            os.posix_fadvise(
                fd, page << _PLE_PAGE_SHIFT, _PLE_PAGE, os.POSIX_FADV_WILLNEED
            )
    except Exception:  # pragma: no cover - advisory prefetch only
        pass

'''

NVFP4_BLOCK = """
_NVFP4_BLOCK_SIZE = 16


class Qwen3_8FlashNextPLENVFp4EmbeddingMethod(QuantizeMethodBase):
    \"\"\"NVFP4 PLE embedding kept packed at runtime.\"\"\"

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, extra_weight_attrs
        if input_size_per_partition % _NVFP4_BLOCK_SIZE:
            raise ValueError(
                "NVFP4 PLE embedding requires the embedding dim to be a "
                f"multiple of {_NVFP4_BLOCK_SIZE}, got {input_size_per_partition}"
            )
        self.params_dtype = params_dtype
        self.head_dim = input_size_per_partition
        self.packed_row_width = (
            input_size_per_partition // 2
            + input_size_per_partition // _NVFP4_BLOCK_SIZE
        )
        rows = sum(output_partition_sizes)
        weight = nn.Parameter(
            torch.empty(rows, input_size_per_partition // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)

        weight_scale = nn.Parameter(
            torch.empty(
                rows,
                input_size_per_partition // _NVFP4_BLOCK_SIZE,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight_scale", weight_scale)

        weight_scale_2 = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)
        self._lut: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        ws2 = layer.weight_scale_2.data.reshape(())
        if ws2.numel() == 0 or not torch.isfinite(ws2).all():
            raise ValueError(
                "NVFP4 PLE checkpoint is missing ngram_embedding.weight_scale_2"
            )
        self._lut = torch.tensor(
            _FP4_VALUES, dtype=torch.float32, device=layer.weight.device
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE NVFP4 weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        codes = F.embedding(input_, layer.weight)
        scales = F.embedding(input_, layer.weight_scale)
        return torch.cat((codes, scales.view(torch.uint8)), dim=-1)

    def dequantize(
        self,
        packed_rows: torch.Tensor,
        scale_2: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        lut = self._lut
        if lut is None or lut.device != packed_rows.device:
            lut = torch.tensor(
                _FP4_VALUES, dtype=torch.float32, device=packed_rows.device
            )
            self._lut = lut
        return _dequant_nvfp4_rows(
            packed_rows, self.head_dim, scale_2, output_dtype, lut
        )


def _ple_dtype_is_fp8(ple_embedding_dtype: object) -> bool:
    if ple_embedding_dtype is None:
        return False
    if isinstance(ple_embedding_dtype, torch.dtype):
        return ple_embedding_dtype == torch.float8_e4m3fn
    text = str(ple_embedding_dtype).rsplit(".", 1)[-1].lower()
    return text in {"float8_e4m3fn", "fp8"}


def _ple_dtype_is_nvfp4(ple_embedding_dtype: object) -> bool:
    if ple_embedding_dtype is None:
        return False
    if isinstance(ple_embedding_dtype, torch.dtype):
        return False
    text = str(ple_embedding_dtype).rsplit(".", 1)[-1].lower()
    return text in {"nvfp4", "fp4", "w4a16_nvfp4"}


_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant_nvfp4_codes(
    packed: torch.Tensor,
    scale: torch.Tensor,
    scale_2: torch.Tensor,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    half = packed.shape[-1]
    codes = torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=-1).reshape(
        *packed.shape[:-1], half * 2
    )
    if lut is None:
        lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = lut[(codes & 0x7).long()]
    sign = ((codes >> 3) & 1).to(torch.float32)
    fp4 = (magnitude * (1 - 2 * sign)).reshape(
        *packed.shape[:-1], -1, _NVFP4_BLOCK_SIZE
    )
    output = fp4 * scale.float().unsqueeze(-1) * scale_2.float()
    return output.reshape(*packed.shape[:-1], half * 2)


def _dequant_nvfp4_rows(
    packed_rows: torch.Tensor,
    head_dim: int,
    scale_2: torch.Tensor,
    output_dtype: torch.dtype,
    lut: torch.Tensor,
) -> torch.Tensor:
    half = head_dim // 2
    codes = packed_rows[..., :half]
    scales = packed_rows[..., half:].contiguous().view(torch.float8_e4m3fn)
    return _dequant_nvfp4_codes(codes, scales, scale_2, lut).to(output_dtype)


def _get_shared_nvfp4_outer_scale(
    outer_scales: dict[int, torch.Tensor],
) -> torch.Tensor:
    first_index, reference = next(iter(outer_scales.items()))
    reference = reference.reshape(())
    for shard_index, outer_scale in outer_scales.items():
        if not torch.equal(outer_scale.reshape(()), reference):
            raise ValueError(
                "NVFP4 PLE shards must share the same global scale, but "
                f"shards {first_index} and {shard_index} differ"
            )
    return reference
"""

GET_QUANT_METHOD = '''
def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
    ple_embedding_dtype: object = None,
) -> QuantizeMethodBase | None:
    """Select a packed PLE embedding method for quantized checkpoint shards."""

    # MIXED_PRECISION checkpoints (modelopt_mixed) declare the PLE table
    # format in text_config.ple_embedding_dtype, not via ModelOptNvFp4Config.
    if _ple_dtype_is_nvfp4(ple_embedding_dtype):
        logger.info_once(
            "PLE embedding %s uses the runtime NVFP4 method (ple_embedding_dtype)",
            prefix,
        )
        return Qwen3_8FlashNextPLENVFp4EmbeddingMethod()
    if _ple_dtype_is_fp8(ple_embedding_dtype):
        logger.info_once(
            "PLE embedding %s uses the runtime FP8 method (ple_embedding_dtype)",
            prefix,
        )
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()

    if isinstance(quant_config, Fp8Config):
        if not quant_config.is_checkpoint_fp8_serialized:
            return None
        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return None
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return None
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()

    if isinstance(quant_config, ModelOptNvFp4Config):
        if not quant_config.is_checkpoint_nvfp4_serialized:
            return None
        if not quant_config.is_layer_excluded(prefix):
            logger.info_once("PLE embedding %s uses the runtime NVFP4 method", prefix)
            return Qwen3_8FlashNextPLENVFp4EmbeddingMethod()
        if _ple_dtype_is_fp8(ple_embedding_dtype):
            logger.info_once(
                "Excluded ModelOpt PLE embedding %s uses the runtime FP8 method",
                prefix,
            )
            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()

    return None
'''

LOAD_WEIGHTS = r'''
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            retained: set[str] = set()
            quant_method = getattr(
                getattr(self, "ngram_embedding", None), "quant_method", None
            )
            if isinstance(quant_method, Qwen3_8FlashNextPLEFp8EmbeddingMethod):
                for name, loaded_weight in weights:
                    if name != "ngram_embedding.weight_scale":
                        continue
                    self.register_buffer(
                        "_offload_weight_scale",
                        loaded_weight.to(
                            device=torch.accelerator.current_accelerator()
                        ),
                        persistent=False,
                    )
                    retained.add(name)
            elif isinstance(quant_method, Qwen3_8FlashNextPLENVFp4EmbeddingMethod):
                scale_2 = None
                for name, loaded_weight in weights:
                    if name != "ngram_embedding.weight_scale_2":
                        continue
                    scale_2 = loaded_weight.reshape(()).to(
                        device=torch.accelerator.current_accelerator()
                    )
                    retained.add(name)
                    # no break: drain the iterator (generic loader must not see the rest)
                self._offload_quant_method = quant_method
                if scale_2 is None:
                    # load_weights is invoked once per tensor group; the
                    # global scale may arrive in a later call. The GPU-side
                    # dequant raises if it is still missing at use time.
                    return retained
                self.register_buffer(
                    "_offload_weight_scale_2", scale_2, persistent=False
                )
                self.register_buffer(
                    "_offload_nvfp4_lut",
                    torch.tensor(
                        _FP4_VALUES, dtype=torch.float32, device=scale_2.device
                    ),
                    persistent=False,
                )
            else:
                for _ in weights:
                    pass
            return retained

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"
        quant_method = getattr(self.ngram_embedding, "quant_method", None)
        nvfp4_runtime = isinstance(quant_method, Qwen3_8FlashNextPLENVFp4EmbeddingMethod)
        packed_codes: dict[int, torch.Tensor] = {}
        packed_scales: dict[int, torch.Tensor] = {}
        global_outer_scale: torch.Tensor | None = None

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue

            if nvfp4_runtime and (
                name == "ngram_embedding.weight_scale_2"
                or name.endswith(".ngram_embedding.weight_scale_2")
                or leaf_name == "weight_scale_2"
            ):
                global_outer_scale = loaded_weight
                continue

            if nvfp4_runtime and name.startswith(shard_prefix):
                suffix = name[len(shard_prefix) :]
                for ending, sink in (
                    (".weight_scale", packed_scales),
                    (".weight", packed_codes),
                ):
                    if not suffix.endswith(ending):
                        continue
                    shard_text = suffix[: -len(ending)]
                    shard_index = int(shard_text)
                    if shard_index >= self.split_ngram_parts:
                        raise ValueError(
                            f"PLE embedding shard index {shard_index} exceeds "
                            f"split_ngram_parts={self.split_ngram_parts}"
                        )
                    sink[shard_index] = loaded_weight
                    break
                else:
                    regular_weights.append((name, loaded_weight))
                continue

            if (
                not nvfp4_runtime
                and name.startswith(shard_prefix)
                and name.endswith(".weight")
            ):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if nvfp4_runtime and packed_codes:
            for shard_index in packed_codes:
                if shard_index not in packed_scales:
                    raise ValueError(
                        f"NVFP4 PLE shard {shard_index} is missing its block scales"
                    )

        for shard_index, codes in packed_codes.items():
            embedding = self.ngram_embedding
            shard_size = (
                embedding.org_vocab_size + self.split_ngram_parts - 1
            ) // self.split_ngram_parts
            checkpoint_start = shard_index * shard_size
            expected_rows = max(
                0, min(shard_size, embedding.org_vocab_size - checkpoint_start)
            )
            expected_codes_shape = (expected_rows, embedding.embedding_dim // 2)
            if tuple(codes.shape) != expected_codes_shape:
                raise ValueError(
                    f"Shape mismatch for packed PLE shard {shard_index}: "
                    f"expected {expected_codes_shape}, got {tuple(codes.shape)}"
                )
            tp_bounds = {
                "checkpoint_start": checkpoint_start,
                "tp_start": embedding.shard_indices.org_vocab_start_index,
                "tp_end": embedding.shard_indices.org_vocab_end_index,
            }
            scales = packed_scales[shard_index]
            expected_scales_shape = (
                expected_rows,
                embedding.embedding_dim // _NVFP4_BLOCK_SIZE,
            )
            if tuple(scales.shape) != expected_scales_shape:
                raise ValueError(
                    f"Scale shape mismatch for packed PLE shard {shard_index}: "
                    f"expected {expected_scales_shape}, got {tuple(scales.shape)}"
                )
            copy_ple_embedding_shard_(embedding.weight.data, codes, **tp_bounds)
            copy_ple_embedding_shard_(embedding.weight_scale.data, scales, **tp_bounds)

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))

        if nvfp4_runtime:
            if global_outer_scale is not None:
                self.ngram_embedding.weight_scale_2.data.copy_(
                    global_outer_scale.reshape(())
                )
                loaded.add("ngram_embedding.weight_scale_2")
                self._nvfp4_global_scale_loaded = True
            if packed_codes:
                loaded.update(
                    {
                        "ngram_embedding.weight",
                        "ngram_embedding.weight_scale",
                    }
                )
        return loaded
'''

DEQUANT = '''
    def _dequantize_embeddings(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize PLE lookup output."""

        quant_method = getattr(
            getattr(self.ple_embedding, "ngram_embedding", None),
            "quant_method",
            None,
        )
        offload_quant_method = getattr(
            self.ple_embedding, "_offload_quant_method", None
        )
        if isinstance(offload_quant_method, Qwen3_8FlashNextPLENVFp4EmbeddingMethod):
            offload_scale_2 = getattr(
                self.ple_embedding, "_offload_weight_scale_2", None
            )
            if offload_scale_2 is None:
                raise RuntimeError("NVFP4 PLE offload is missing its global scale")
            packed_row_width = (
                self.ple_head_dim // 2 + self.ple_head_dim // _NVFP4_BLOCK_SIZE
            )
            packed_rows = embeddings.unflatten(
                -1, (self.ple_ngram_heads, packed_row_width)
            )
            dequantized = _dequant_nvfp4_rows(
                packed_rows,
                self.ple_head_dim,
                offload_scale_2,
                output_dtype,
                self.ple_embedding._offload_nvfp4_lut,
            )
            return dequantized.flatten(-2)
        if isinstance(quant_method, Qwen3_8FlashNextPLENVFp4EmbeddingMethod):
            packed_rows = embeddings.unflatten(
                -1,
                (self.ple_embedding.ngram_heads, quant_method.packed_row_width),
            )
            dequantized = quant_method.dequantize(
                packed_rows,
                self.ple_embedding.ngram_embedding.weight_scale_2,
                output_dtype,
            )
            return dequantized.flatten(-2)
        if not is_fp8(embeddings):
            return embeddings
        weight_scale = self._get_embedding_weight_scale()
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)
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

    patch("ple_layer_patched", [
        (
            "from vllm.forward_context import get_forward_context\n",
            "from vllm.forward_context import get_forward_context\n"
            "import os\n"
            "from vllm.logger import init_logger\n"
            "from vllm.model_executor.layers.quantization.modelopt import "
            "ModelOptNvFp4Config\n"
            "from vllm.model_executor.utils import set_weight_attrs\n\n"
            "logger = init_logger(__name__)\n",
        ),
        (
            "    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:\n"
            "        return F.embedding(input_, layer.weight)\n\n\n"
            "def _get_ple_embedding_quant_method(\n",
            "    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:\n"
            "        return F.embedding(input_, layer.weight)\n\n\n"
            + PREFETCH_BLOCK
            + NVFP4_BLOCK
            + "\n"
            + GET_QUANT_METHOD
            + "\n\n"
            "def _get_ple_embedding_quant_method_REMOVED(\n",
        ),
        (
            "def _get_ple_embedding_quant_method_REMOVED(\n"
            "    quant_config: QuantizationConfig | None,\n"
            "    prefix: str,\n"
            ") -> QuantizeMethodBase | None:\n"
            '    """Select global-scale FP8 only for quantized PLE checkpoint shards."""\n\n'
            "    if not isinstance(quant_config, Fp8Config):\n"
            "        return None\n"
            "    if not quant_config.is_checkpoint_fp8_serialized:\n"
            "        return None\n\n"
            "    ignored_layers = quant_config.ignored_layers\n"
            "    if is_layer_skipped(\n"
            "        prefix,\n"
            "        ignored_layers,\n"
            "        quant_config.packed_modules_mapping,\n"
            "        match_mode=quant_config.ignored_layers_match_mode,\n"
            "    ):\n"
            "        return None\n"
            "    # PLE checkpoint shards form one runtime embedding parameter.\n"
            '    shard_prefix = f"{prefix}.shard_"\n'
            "    if any(name.startswith(shard_prefix) for name in ignored_layers):\n"
            "        return None\n"
            "    return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n\n\n",
            "",
        ),
        (
            "            quant_method=_get_ple_embedding_quant_method(\n"
            "                quant_config, f\"{prefix}.ngram_embedding\"\n"
            "            ),\n",
            "            quant_method=_get_ple_embedding_quant_method(\n"
            "                quant_config,\n"
            "                f\"{prefix}.ngram_embedding\",\n"
            "                getattr(config, \"ple_embedding_dtype\", None),\n"
            "            ),\n",
        ),
        (
            "    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:\n"
            '        """Keep quantized lookup results in their embedding storage dtype."""\n'
            "        embedding = getattr(self, \"ngram_embedding\", None)\n"
            "        weight = getattr(embedding, \"weight\", None)\n"
            "        if weight is not None:\n"
            "            return weight.dtype\n"
            "        if hasattr(self, \"_offload_weight_scale\"):\n"
            "            return torch.float8_e4m3fn\n"
            "        return default_dtype\n\n"
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
            '        """Load hash buffers and checkpoint-split embedding rows."""\n\n'
            "        # GPU workers retain only the global FP8 scale. The CPU process owns the\n"
            "        # embedding weight and returns its quantized lookup output unchanged.\n"
            "        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():\n"
            "            retained: set[str] = set()\n"
            "            for name, loaded_weight in weights:\n"
            "                if name != \"ngram_embedding.weight_scale\":\n"
            "                    continue\n"
            "                self.register_buffer(\n"
            "                    \"_offload_weight_scale\",\n"
            "                    loaded_weight.to(device=torch.accelerator.current_accelerator()),\n"
            "                    persistent=False,\n"
            "                )\n"
            "                retained.add(name)\n"
            "            return retained\n\n"
            "        persistent_buffers = {\n"
            "            \"layer_multipliers\": self.layer_multipliers,\n"
            "            \"ngram_heads_offsets\": self.ngram_heads_offsets,\n"
            "            \"ngram_heads_vocab_sizes\": self.ngram_heads_vocab_sizes,\n"
            "        }\n"
            "        loaded: set[str] = set()\n"
            "        regular_weights: list[tuple[str, torch.Tensor]] = []\n"
            '        shard_prefix = "ngram_embedding.shard_"\n\n'
            "        for name, loaded_weight in weights:\n"
            "            leaf_name = name.rsplit(\".\", 1)[-1]\n"
            "            if leaf_name.startswith(\"hashstats_\") or leaf_name == \"token_lookup\":\n"
            "                continue\n"
            "            if name in persistent_buffers:\n"
            "                buffer = persistent_buffers[name]\n"
            "                if buffer.shape != loaded_weight.shape:\n"
            "                    raise ValueError(\n"
            '                        f"Shape mismatch for {name}: expected "\n'
            '                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"\n'
            "                    )\n"
            "                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))\n"
            "                loaded.add(name)\n"
            "                continue\n"
            "            if name.startswith(shard_prefix) and name.endswith(\".weight\"):\n"
            "                shard_text = name[len(shard_prefix) : -len(\".weight\")]\n"
            "                if not shard_text.isdigit():\n"
            "                    regular_weights.append((name, loaded_weight))\n"
            "                    continue\n"
            "                shard_index = int(shard_text)\n"
            "                if shard_index >= self.split_ngram_parts:\n"
            "                    raise ValueError(\n"
            '                        f"PLE embedding shard index {shard_index} exceeds "\n'
            '                        f"split_ngram_parts={self.split_ngram_parts}"\n'
            "                    )\n"
            "                embedding = self.ngram_embedding\n"
            "                shard_size = (\n"
            "                    embedding.org_vocab_size + self.split_ngram_parts - 1\n"
            "                ) // self.split_ngram_parts\n"
            "                checkpoint_start = shard_index * shard_size\n"
            "                expected_rows = max(\n"
            "                    0,\n"
            "                    min(shard_size, embedding.org_vocab_size - checkpoint_start),\n"
            "                )\n"
            "                expected_shape = (expected_rows, embedding.embedding_dim)\n"
            "                if tuple(loaded_weight.shape) != expected_shape:\n"
            "                    raise ValueError(\n"
            '                        f"Shape mismatch for PLE embedding shard {shard_index}: "\n'
            '                        f"expected {expected_shape}, got "\n'
            '                        f"{tuple(loaded_weight.shape)}"\n'
            "                    )\n"
            "                copy_ple_embedding_shard_(\n"
            "                    embedding.weight.data,\n"
            "                    loaded_weight,\n"
            "                    checkpoint_start=checkpoint_start,\n"
            "                    tp_start=embedding.shard_indices.org_vocab_start_index,\n"
            "                    tp_end=embedding.shard_indices.org_vocab_end_index,\n"
            "                )\n"
            "                loaded.add(\"ngram_embedding.weight\")\n"
            "                continue\n"
            "            regular_weights.append((name, loaded_weight))\n\n"
            "        if regular_weights:\n"
            "            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))\n"
            "        return loaded\n",
            LOAD_WEIGHTS,
        ),
        (
            "        self.prefix = prefix\n"
            "        self.hidden_size = int(config.hidden_size)\n"
            "        self.hc_count = config.hc_count\n",
            "        self.prefix = prefix\n"
            "        self.hidden_size = int(config.hidden_size)\n"
            "        self.ple_embedding_dim = int(config.ple_embed_dim)\n"
            "        self.ple_ngram_heads = (int(config.ngram_size) - 1) * int(\n"
            "            config.heads_per_ngram\n"
            "        )\n"
            "        self.ple_head_dim = self.ple_embedding_dim // self.ple_ngram_heads\n"
            "        self.hc_count = config.hc_count\n",
        ),
        (
            "    def _dequantize_embeddings(\n"
            "        self,\n"
            "        embeddings: torch.Tensor,\n"
            "        output_dtype: torch.dtype,\n"
            "    ) -> torch.Tensor:\n"
            '        """Dequantize PLE lookup output."""\n\n'
            "        if not is_fp8(embeddings):\n"
            "            return embeddings\n"
            "        weight_scale = self._get_embedding_weight_scale()\n"
            "        if weight_scale is None:\n"
            "            raise RuntimeError(\"FP8 PLE embedding is missing its global scale\")\n"
            "        if weight_scale.device != embeddings.device:\n"
            "            raise RuntimeError(\"FP8 PLE embedding scale must be on the output device\")\n"
            "        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)\n",
            DEQUANT,
        ),
        # --- offload-path fixes (NVFP4 PLE) ---------------------------------
        # 1. Re-add get_offload_output_dtype. The upstream offload fast path
        #    does torch.index_select(weight, ..., out=buffer), which requires
        #    buffer.dtype == weight.dtype. For an NVFP4 table the weight is
        #    packed uint8, so the buffer must be uint8 too; the GPU side then
        #    dequantizes via _dequantize_embeddings. Without this the buffer
        #    defaults to bf16 and index_select raises
        #    "self and result must have the same scalar type".
        (
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
            '        """Load hash buffers and checkpoint-split embedding rows."""\n',
            "    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:\n"
            '        """Keep quantized lookup results in their storage dtype."""\n'
            "        embedding = getattr(self, \"ngram_embedding\", None)\n"
            "        weight = getattr(embedding, \"weight\", None)\n"
            "        if weight is not None:\n"
            "            return weight.dtype\n"
            "        if hasattr(self, \"_offload_weight_scale_2\"):\n"
            "            return torch.uint8\n"
            "        if hasattr(self, \"_offload_weight_scale\"):\n"
            "            return torch.float8_e4m3fn\n"
            "        return default_dtype\n\n"
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
            '        """Load hash buffers and checkpoint-split embedding rows."""\n',
        ),
        # 2. Record the offload quant method on the GPU-side placeholder so
        #    _dequantize_embeddings takes its NVFP4 branch (it reads
        #    _offload_quant_method, which nothing previously assigned).
        (
            "                self.register_buffer(\n"
            "                    \"_offload_nvfp4_lut\",\n"
            "                    torch.tensor(\n"
            "                        _FP4_VALUES, dtype=torch.float32, device=scale_2.device\n"
            "                    ),\n"
            "                    persistent=False,\n"
            "                )\n",
            "                self.register_buffer(\n"
            "                    \"_offload_nvfp4_lut\",\n"
            "                    torch.tensor(\n"
            "                        _FP4_VALUES, dtype=torch.float32, device=scale_2.device\n"
            "                    ),\n"
            "                    persistent=False,\n"
            "                )\n"
            "                self._offload_quant_method = quant_method\n",
        ),

        # 3. Generalize the offload fast path to the packed row width. Upstream
        #    assumes each embedding row is head_dim wide, but an NVFP4 table
        #    stores head_dim/2 code bytes + head_dim/16 scale bytes per row.
        #    Slicing the shared buffer to embedding_dim and reshaping to
        #    head_dim gives index_select a wrongly-shaped out= tensor, so torch
        #    RESIZES it -- silently allocating a new tensor instead of writing
        #    into the IPC buffer. The GPU side then waits forever. Derive the
        #    width from the weight itself (identical to upstream when
        #    unquantized, since row_width == head_dim there).
        (
            "        if output_buffer is not None:\n"
            "            output = output_buffer[:num_tokens, : self.embedding_dim]\n"
            "            torch.index_select(\n"
            "                self.ngram_embedding.weight,\n"
            "                0,\n"
            "                ngram_ids.reshape(-1),\n"
            "                out=output.reshape(-1, self.head_dim),\n"
            "            )\n"
            "            return output\n",
            "        if output_buffer is not None:\n"
            "            # Offload fast path. The GPU side expects, per head, the\n"
            "            # same packed row the on-device lookup produces:\n"
            "            # NVFP4 -> cat(codes[head_dim/2], scales[head_dim/16]).\n"
            "            # Prefer a memory-mapped pre-packed table (built by\n"
            "            # files/build_ple_packed_table.py); otherwise assemble\n"
            "            # the row from the separate code/scale parameters.\n"
            "            emb = self.ngram_embedding\n"
            "            ids = ngram_ids.reshape(-1)\n"
            "            packed = getattr(emb, \"_packed_table\", None)\n"
            "            scales = getattr(emb, \"weight_scale\", None)\n"
            "            if packed is not None:\n"
            "                row_width = packed.shape[-1]\n"
            "                total_width = ngram_ids.shape[-1] * row_width\n"
            "                output = output_buffer[:num_tokens, :total_width]\n"
            "                _ple_prefetch_rows(\n"
            "                    getattr(emb, \"_packed_table_fd\", None), ids, row_width\n"
            "                )\n"
            "                torch.index_select(\n"
            "                    packed, 0, ids, out=output.reshape(-1, row_width)\n"
            "                )\n"
            "                return output\n"
            "            if scales is not None and scales.dim() == 2:\n"
            "                codes = emb.weight\n"
            "                scales_u8 = scales.view(torch.uint8)\n"
            "                cw, sw = codes.shape[-1], scales_u8.shape[-1]\n"
            "                row_width = cw + sw\n"
            "                total_width = ngram_ids.shape[-1] * row_width\n"
            "                output = output_buffer[:num_tokens, :total_width]\n"
            "                rows = output.reshape(-1, row_width)\n"
            "                rows[:, :cw].copy_(codes.index_select(0, ids))\n"
            "                rows[:, cw:].copy_(scales_u8.index_select(0, ids))\n"
            "                return output\n"
            "            weight = emb.weight\n"
            "            row_width = weight.shape[-1]\n"
            "            total_width = ngram_ids.shape[-1] * row_width\n"
            "            output = output_buffer[:num_tokens, :total_width]\n"
            "            torch.index_select(\n"
            "                weight, 0, ids, out=output.reshape(-1, row_width)\n"
            "            )\n"
            "            return output\n",
        ),

        # ---- GB10 offload placeholder fixes (see files/patch_ple_offload.py) ----
        # (a) The GPU-worker placeholder skips its constructor, so it has no
        #     ngram_embedding/quant_method. Give it the quant method from config
        #     so load_weights() can capture the global scale + LUT and
        #     get_offload_output_dtype() reports uint8 (packed bytes), not bf16.
        (
            "                quant_config=quant_config,\n"
            "                params_dtype=model_config.dtype,\n"
            "            )\n"
            "        self.key_proj = ReplicatedLinear(\n",
            "                quant_config=quant_config,\n"
            "                params_dtype=model_config.dtype,\n"
            "            )\n"
            "        if not hasattr(self.ple_embedding, \"ngram_embedding\"):\n"
            "            # Offload placeholder on the GPU worker.\n"
            "            self.ple_embedding._offload_quant_method = (\n"
            "                _get_ple_embedding_quant_method(\n"
            "                    quant_config,\n"
            "                    f\"{prefix}.ple_embedding.ngram_embedding\",\n"
            "                    getattr(config, \"ple_embedding_dtype\", None),\n"
            "                )\n"
            "            )\n"
            "        self.key_proj = ReplicatedLinear(\n",
        ),
        (
            "        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():\n"
            "            retained: set[str] = set()\n"
            "            quant_method = getattr(\n"
            "                getattr(self, \"ngram_embedding\", None), \"quant_method\", None\n"
            "            )\n",
            "        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():\n"
            "            retained: set[str] = set()\n"
            "            quant_method = getattr(\n"
            "                getattr(self, \"ngram_embedding\", None), \"quant_method\", None\n"
            "            )\n"
            "            if quant_method is None:\n"
            "                quant_method = getattr(self, \"_offload_quant_method\", None)\n",
        ),
        (
            "        if hasattr(self, \"_offload_weight_scale_2\"):\n"
            "            return torch.uint8\n",
            "        if hasattr(self, \"_offload_weight_scale_2\"):\n"
            "            return torch.uint8\n"
            "        if isinstance(\n"
            "            getattr(self, \"_offload_quant_method\", None),\n"
            "            Qwen3_8FlashNextPLENVFp4EmbeddingMethod,\n"
            "        ):\n"
            "            return torch.uint8\n"
            "        if isinstance(\n"
            "            getattr(self, \"_offload_quant_method\", None),\n"
            "            Qwen3_8FlashNextPLEFp8EmbeddingMethod,\n"
            "        ):\n"
            "            return torch.float8_e4m3fn\n",
        ),
        # (b) The IPC output buffer is ple_embed_dim wide; packed NVFP4 rows only
        #     fill heads*packed_row_width of it. Slice before unflatten.
        (
            "            packed_row_width = (\n"
            "                self.ple_head_dim // 2 + self.ple_head_dim // _NVFP4_BLOCK_SIZE\n"
            "            )\n"
            "            packed_rows = embeddings.unflatten(\n"
            "                -1, (self.ple_ngram_heads, packed_row_width)\n"
            "            )\n",
            "            packed_row_width = (\n"
            "                self.ple_head_dim // 2 + self.ple_head_dim // _NVFP4_BLOCK_SIZE\n"
            "            )\n"
            "            packed_rows = embeddings[\n"
            "                ..., : self.ple_ngram_heads * packed_row_width\n"
            "            ].unflatten(-1, (self.ple_ngram_heads, packed_row_width))\n",
        ),
    ])
    print("ok")


if __name__ == "__main__":
    main()
