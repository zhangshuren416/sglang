# SPDX-License-Identifier: Apache-2.0
"""Guard sparse-layer indexing and ordinary-SwiGLU model admission."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.moe.dfc_model_adapter import DfcModelAdapter
from sglang.srt.hardware_backend.npu.moe.weight_offload import MemFabricMoEOffloader
from sglang.srt.runtime_context import get_context, get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ModelSlimW8A8Int8MoE:
    pass


class _Experts(torch.nn.Module):
    """Metadata only: admission must not allocate or invoke an NPU operation."""

    def __init__(self):
        super().__init__()
        device = SimpleNamespace(type="npu", index=0)
        self.w13_weight = SimpleNamespace(
            dtype=torch.int8, ndim=3, shape=(2, 64, 64), device=device
        )
        self.w2_weight = SimpleNamespace(
            dtype=torch.int8, ndim=3, shape=(2, 64, 32), device=device
        )
        self.w13_scheme = ModelSlimW8A8Int8MoE()
        self.w2_scheme = ModelSlimW8A8Int8MoE()
        self.quant_method = object()
        self.num_fused_shared_experts = 0
        self.moe_ep_size = 2
        self.moe_tp_size = 1
        self.num_experts = 4
        self.top_k = 2
        self.moe_runner_config = SimpleNamespace(
            activation="silu", is_gated=True, apply_router_weight_on_input=False
        )


class _Moe(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.experts = _Experts()

    def forward(self):
        return object()


class _Decoder(torch.nn.Module):
    def __init__(self, model_type, layer_id, sparse=True):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, hidden_act="silu")
        self.layer_id = layer_id
        self.is_layer_sparse = sparse
        self.mlp = _Moe(layer_id) if sparse else torch.nn.Identity()

    def forward(self):
        return self.mlp()


def _decoder(model_type, layer_id, sparse=True):
    name = {
        "qwen3_moe": "Qwen3MoeDecoderLayer",
        "deepseek_v4": "DeepseekV4DecoderLayer",
    }.get(model_type, "DeepseekV2DecoderLayer")
    return type(name, (_Decoder,), {})(model_type, layer_id, sparse)


class TestDfcModelAdapter(CustomTestCase):
    def test_dense_prefix_does_not_take_the_resident_moe_index(self):
        """Skipping dense decoders must leave the first actual expert layer resident."""
        for model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
            with self.subTest(model_type=model_type):
                adapter = DfcModelAdapter()
                for layer_id in range(3):
                    self.assertIsNone(
                        adapter._register_layer(_decoder(model_type, layer_id, False))
                    )
                first, second = _decoder(model_type, 3), _decoder(model_type, 5)
                self.assertEqual(adapter._register_layer(first), 0)
                self.assertEqual(adapter._register_layer(second), 1)
                self.assertEqual(adapter.layers, [first, second])

    def test_qwen_routed_layers_do_not_require_deepseek_shared_attributes(self):
        """Qwen3 has neither moe.config nor moe.num_fused_shared_experts."""
        adapter = DfcModelAdapter()
        self.assertEqual(adapter._register_layer(_decoder("qwen3_moe", 0)), 0)

    def test_clipping_and_modified_ffn_cannot_pass_plain_swiglu_admission(self):
        for limit in (0.0, 10.0):
            with self.subTest(limit=limit):
                decoder = _decoder("deepseek_v3", 3)
                decoder.config.swiglu_limit = limit
                with self.assertRaisesRegex(ValueError, "clipping"):
                    DfcModelAdapter()._register_layer(decoder)
        decoder = _decoder("qwen3_moe", 0)
        decoder.mlp.experts.moe_runner_config.gemm1_alpha = 1.702
        with self.assertRaisesRegex(ValueError, "unmodified gated SwiGLU"):
            DfcModelAdapter()._register_layer(decoder)

    def test_v4_and_glm_flash_are_not_admitted_as_plain_swiglu_models(self):
        for model_type in ("deepseek_v4", "glm5_next", "glm5_next_text"):
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, "no adapter"):
                    DfcModelAdapter()._register_layer(_decoder(model_type, 0))

    def test_ragged_layer_geometry_cannot_reuse_the_first_layer_capacity(self):
        adapter = DfcModelAdapter()
        adapter._register_layer(_decoder("qwen3_moe", 0))
        second = _decoder("qwen3_moe", 1)
        second.mlp.experts.top_k = 1
        with self.assertRaisesRegex(ValueError, "uniform"):
            adapter._register_layer(second)

    def test_offload_wrap_preserves_dense_layers_and_collects_later_experts(self):
        """Exercise the real wrapper before any checkpoint/pool native operation."""
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "pool.json"
            config.write_text(json.dumps({"store_url": "tcp://h:1"}))
            with get_context().override_server_args(
                moe_weight_offload_backend="memfabric",
                moe_weight_offload_config=str(config),
            ):
                offloader = MemFabricMoEOffloader()
                dense = _decoder("deepseek_v3", 0, False)
                first, second = _decoder("deepseek_v3", 3), _decoder("deepseek_v3", 5)
                try:
                    self.assertEqual(
                        offloader.wrap_modules(iter((dense, first, second))),
                        [dense, first, second],
                    )
                    self.assertEqual(offloader.layers, [first, second])
                    self.assertNotIn(id(first.mlp.experts), offloader.components)
                    self.assertEqual(
                        offloader.components[id(second.mlp.experts)], (1, "experts")
                    )
                    self.assertEqual(len(offloader.hooks), 8)
                    self.assertFalse(dense._forward_pre_hooks)
                finally:
                    for hook in offloader.hooks:
                        hook.remove()
                    get_resources().buffers.pop("moe_weight_offloader", None)


class TestMemFabricModelAdmission(CustomTestCase):
    def test_v4_activation_and_hash_routing_do_not_require_dfc(self):
        """Weight transport must leave native SiTU/clipping/router metadata intact."""
        with get_context().override_server_args():
            offloader = MemFabricMoEOffloader()
        for layer_id, activation in enumerate(("swigluoai", "situ")):
            decoder = _decoder("deepseek_v4", layer_id)
            decoder.config.hidden_act = activation
            decoder.config.swiglu_limit = 10.0
            runner = decoder.mlp.experts.moe_runner_config
            runner.activation = activation
            runner.gemm1_alpha = 1.702
            runner.gemm1_clamp_limit = 10.0
            # Router metadata can differ while the physical expert shapes agree.
            decoder.mlp.experts.top_k = layer_id + 1
            self.assertEqual(offloader._register_layer(decoder), layer_id)
            self.assertIs(decoder.mlp.experts.moe_runner_config, runner)
            self.assertEqual(runner.gemm1_clamp_limit, 10.0)
        self.assertIsNone(offloader.execution_adapter)

    def test_v4_prefetch_does_not_make_v4_a_dfc_model(self):
        with get_context().override_server_args():
            offloader = MemFabricMoEOffloader(execution_adapter=DfcModelAdapter())
        with self.assertRaisesRegex(ValueError, "no adapter"):
            offloader._register_layer(_decoder("deepseek_v4", 0))
        self.assertFalse(offloader.layers)

    def test_nonuniform_weight_geometry_cannot_reuse_the_same_slots(self):
        with get_context().override_server_args():
            offloader = MemFabricMoEOffloader()
        offloader._register_layer(_decoder("deepseek_v4", 0))
        second = _decoder("deepseek_v4", 1)
        second.mlp.experts.w2_weight.shape = (2, 64, 64)
        with self.assertRaisesRegex(ValueError, "uniform"):
            offloader._register_layer(second)

    def test_resident_single_layer_post_init_does_not_create_dfc_runtime(self):
        with get_context().override_server_args():
            offloader = MemFabricMoEOffloader()
        offloader.enabled = True
        offloader.layers = [_decoder("deepseek_v4", 0)]
        resources = get_resources().buffers
        before = dict(resources)
        offloader.post_init()
        self.assertTrue(offloader.ready_for_forward)
        self.assertIsNone(offloader.offload)
        self.assertEqual(resources, before)


if __name__ == "__main__":
    unittest.main()
