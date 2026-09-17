"""CPU-only regression check for the MTP FP8_PB_WO dispatch fix.

Exercises _resolve_quant_algo's MTP local-index bridging directly against a
synthetic ModelOptMixedPrecisionConfig -- no RoutedExperts/FusedMoEConfig
construction (heavy vLLM distributed/config machinery not worth mocking here;
the real dispatch and Fp8MoEMethod construction are exercised by an actual
GPU boot instead).
"""
import unittest

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)

RT_LOCAL = "mtp.layers.0.mlp.experts"
RT_GLOBAL = "mtp.layers.48.mlp.experts"


def config(entries):
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None, exclude_modules=[],
        quantized_layers=entries,
        fp8_config=None, nvfp4_config=None, w4a16_nvfp4_config=None,
        mxfp8_config=None,
    )


class MtpQuantAlgoBridgeTests(unittest.TestCase):
    def test_global_index_resolves_via_local_declaration(self):
        c = config({RT_LOCAL: {"quant_algo": "FP8_PB_WO", "group_size": 128}})
        self.assertEqual(c._resolve_quant_algo(RT_GLOBAL), "FP8_PB_WO")

    def test_local_index_still_resolves_directly(self):
        c = config({RT_LOCAL: {"quant_algo": "FP8_PB_WO", "group_size": 128}})
        self.assertEqual(c._resolve_quant_algo(RT_LOCAL), "FP8_PB_WO")

    def test_other_global_layer_indices_also_bridge(self):
        # MTP always has exactly one local layer; any global mount index for
        # an *mtp.layers.<N>* prefix should resolve against local index 0.
        c = config({RT_LOCAL: {"quant_algo": "FP8_PB_WO", "group_size": 128}})
        for n in (1, 12, 48, 100):
            self.assertEqual(c._resolve_quant_algo(f"mtp.layers.{n}.mlp.experts"), "FP8_PB_WO")

    def test_non_mtp_prefix_unaffected(self):
        # Guard against over-matching: a non-MTP prefix with a numeric
        # segment elsewhere must not spuriously bridge to layers.0.
        c = config({"model.language_model.layers.5.mlp.experts": {"quant_algo": "FP8"}})
        self.assertIsNone(c._resolve_quant_algo("mtp.layers.48.mlp.experts"))
        self.assertEqual(
            c._resolve_quant_algo("model.language_model.layers.5.mlp.experts"), "FP8"
        )

    def test_unresolved_prefix_still_none(self):
        c = config({RT_LOCAL: {"quant_algo": "FP8_PB_WO", "group_size": 128}})
        self.assertIsNone(c._resolve_quant_algo("mtp.layers.48.mlp.gate"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
