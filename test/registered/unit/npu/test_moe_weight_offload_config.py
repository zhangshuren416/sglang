# SPDX-License-Identifier: Apache-2.0
"""Keep transport selection and eager configuration independent of DFC."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.dfc_hook import handle_dfc_config
from sglang.srt.arg_groups.moe_weight_offload_hook import (
    handle_moe_weight_offload_config,
)
from sglang.srt.hardware_backend.npu.moe.dfc_model_adapter import DfcModelAdapter
from sglang.srt.hardware_backend.npu.moe.weight_offload import MemFabricMoEOffloader
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.offloader import NoopOffloader, create_offloader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMoEWeightOffloadConfig(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config_path = Path(self.directory.name) / "pool.json"
        self.config_path.write_text(json.dumps({"store_url": "tcp://h:1"}))

    def _config(self, **updates):
        fields = dict(
            device="npu",
            dtype="bfloat16",
            quantization="modelslim",
            moe_a2a_backend="deepep",
            moe_weight_offload_backend="memfabric",
            moe_weight_offload_config=str(self.config_path),
            dfc_max_output_tokens=None,
            disaggregation_mode="null",
            pp_size=1,
            moe_dp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            dwdp_size=1,
            nnodes=1,
            tp_size=16,
            max_ep_size=None,
            elastic_ep_initial_size=None,
            speculative_algorithm=None,
            ep_num_redundant_experts=0,
            ep_join_mode=None,
            cpu_offload_gb=0,
            offload_group_size=-1,
            load_format="auto",
            weight_cache_mode="off",
            enable_hierarchical_cache=False,
            cuda_graph_config=None,
        )
        fields.update(updates)
        return SimpleNamespace(**fields)

    def _resolve_transport(self, cfg):
        def declare(_args, _source, **updates):
            vars(cfg).update(updates)

        with (
            patch(
                "sglang.srt.arg_groups.moe_weight_offload_hook.resolving_view",
                return_value=cfg,
            ),
            patch(
                "sglang.srt.arg_groups.moe_weight_offload_hook.declare_resolution",
                side_effect=declare,
            ),
        ):
            handle_moe_weight_offload_config(object())

    def test_deepep_transport_allows_colocated_prefill_and_decode_without_zbal(self):
        for mode in ("null", "prefill", "decode"):
            with self.subTest(mode=mode):
                cfg = self._config(disaggregation_mode=mode)
                # No call to the DFC-only ZBAL requirement is permitted here.
                with patch(
                    "sglang.srt.arg_groups.dfc_hook.resolving_view", return_value=cfg
                ):
                    handle_dfc_config(object())
                self._resolve_transport(cfg)
                self.assertTrue(cfg.disable_shared_experts_fusion)
                self.assertTrue(cfg.disable_overlap_schedule)
                for phase in (Phase.PREFILL, Phase.DECODE):
                    self.assertEqual(
                        cfg.cuda_graph_config[phase].backend, Backend.DISABLED
                    )

    def test_dfc_capacity_option_stays_compute_specific(self):
        cfg = self._config(dfc_max_output_tokens=1024)
        with patch("sglang.srt.arg_groups.dfc_hook.resolving_view", return_value=cfg):
            with self.assertRaisesRegex(ValueError, "dfc-max-output-tokens"):
                handle_dfc_config(object())

    def test_invalid_transport_combinations_are_rejected(self):
        cases = (
            dict(moe_weight_offload_backend="none"),
            dict(moe_weight_offload_config=None),
            dict(device="cuda"),
            dict(speculative_algorithm="EAGLE"),
            dict(cpu_offload_gb=1),
            dict(enable_two_batch_overlap=True),
            dict(enable_torch_compile=True),
            dict(moe_a2a_backend="ascend_fuseep"),
        )
        for updates in cases:
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    self._resolve_transport(self._config(**updates))

    def test_factory_composes_dfc_only_when_selected(self):
        for backend in ("deepep", "ascend_dfc"):
            with self.subTest(backend=backend):
                with get_context().override_server_args(
                    moe_a2a_backend=backend,
                    moe_weight_offload_backend="memfabric",
                    moe_weight_offload_config=str(self.config_path),
                ):
                    offloader = create_offloader()
                self.assertIsInstance(offloader, MemFabricMoEOffloader)
                if backend == "deepep":
                    self.assertIsNone(offloader.execution_adapter)
                else:
                    self.assertIsInstance(offloader.execution_adapter, DfcModelAdapter)

    def test_resident_baselines_do_not_create_a_weight_pool(self):
        for backend in ("deepep", "ascend_dfc"):
            with self.subTest(backend=backend):
                with get_context().override_server_args(
                    moe_a2a_backend=backend, moe_weight_offload_backend="none"
                ):
                    offloader = create_offloader()
                expected = NoopOffloader if backend == "deepep" else DfcModelAdapter
                self.assertIsInstance(offloader, expected)


if __name__ == "__main__":
    unittest.main()
