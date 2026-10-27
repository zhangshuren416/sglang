# SPDX-License-Identifier: Apache-2.0
"""Model validation and lifecycle for resident Ascend DFC execution."""

from __future__ import annotations

from sglang.srt.hardware_backend.npu.moe.dfc import DfcRuntime
from sglang.srt.runtime_context import get_parallel, get_resources
from sglang.srt.utils.offloader import BaseOffloader


class DfcModelAdapter(BaseOffloader):
    """Use existing model construction hooks without requiring a weight pool."""

    def __init__(self):
        self.layers = []
        self.device = None
        self.closed = False

    def _register_layer(self, decoder):
        if type(decoder).__name__ != "DeepseekV4DecoderLayer":
            raise ValueError(
                "ascend_dfc currently supports DeepSeek-V4 decoder layers only"
            )
        moe = decoder.mlp
        if self.device is None and moe.experts.w13_weight.device.type != "npu":
            raise ValueError("ascend_dfc requires an NPU model")
        if any(
            type(getattr(moe.experts, f"{prefix}_scheme", None)).__name__
            != "ModelSlimW8A8Int8MoE"
            for prefix in ("w13", "w2")
        ):
            raise ValueError(
                "ascend_dfc requires ModelSlim W8A8 for both routed projections"
            )
        if getattr(moe.config, "swiglu_limit", None) not in (None, 0):
            raise ValueError(
                "Installed DFC API cannot express this model's swiglu_limit"
            )
        if moe.num_fused_shared_experts:
            raise ValueError("ascend_dfc requires disabled shared expert fusion")
        index = len(self.layers)
        self.layers.append(decoder)
        if self.device is None:
            self.device = moe.experts.w13_weight.device
        return index

    def _publish(self):
        if not self.layers:
            raise ValueError("DFC model has no local MoE layers")
        get_resources().buffers["ascend_dfc_model_adapter"] = self

    def wrap_modules(self, all_modules_generator, **kwargs):
        for decoder in all_modules_generator:
            self._register_layer(decoder)
        self._publish()
        return list(self.layers)

    def post_init(self):
        from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

        parallel = get_parallel()
        lazy_init_zbal_gva_mem(
            "npu",
            self.device.index,
            parallel.world_group.rank_in_group,
            parallel.launch_world_size,
            parallel.world_group.cpu_group,
        )
        get_resources().buffers["ascend_dfc_runtime"] = DfcRuntime(
            self.device, self.layers[0].mlp.experts
        )

    def close(self):
        if self.closed:
            return
        runtime = get_resources().buffers.pop("ascend_dfc_runtime", None)
        if runtime is not None:
            runtime.close()
        self.layers.clear()
        self.closed = True
