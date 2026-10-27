# SPDX-License-Identifier: Apache-2.0
"""Model validation and lifecycle for resident Ascend DFC execution."""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.npu.moe.dfc import DfcRuntime
from sglang.srt.runtime_context import get_parallel, get_resources
from sglang.srt.utils.offloader import BaseOffloader


# Model types identify mathematical semantics; decoder classes identify the
# construction/execution paths audited for this eager adapter.
DFC_MODEL_DECODERS = {
    "qwen3_moe": "Qwen3MoeDecoderLayer",
    "deepseek_v3": "DeepseekV2DecoderLayer",  # includes R1 and V3.1
    "deepseek_v32": "DeepseekV2DecoderLayer",
    "glm_moe_dsa": "DeepseekV2DecoderLayer",  # ordinary GLM-5.3
}


class DfcModelAdapter(BaseOffloader):
    """Use existing model construction hooks without requiring a weight pool."""

    def __init__(self):
        self.layers = []
        self.device = None
        self.closed = False
        self.geometry = None

    def _register_layer(self, decoder):
        config = decoder.config
        model_type = getattr(config, "model_type", None)
        expected = DFC_MODEL_DECODERS.get(model_type)
        if expected is None or type(decoder).__name__ != expected:
            raise ValueError(
                f"ascend_dfc has no adapter for {model_type!r}/"
                f"{type(decoder).__name__}; supported model types: "
                f"{sorted(DFC_MODEL_DECODERS)}"
            )
        if getattr(config, "hidden_act", None) != "silu":
            raise ValueError("DFC requires ordinary SiLU/SwiGLU expert activation")
        if getattr(config, "swiglu_limit", None) is not None:
            raise ValueError("Installed DFC cannot express configured SwiGLU clipping")
        moe = decoder.mlp
        experts = getattr(moe, "experts", None)
        if experts is None:
            if not decoder.is_layer_sparse:
                # DeepSeek/GLM have an initial dense prefix. Keep it on the
                # original path and count residency/slots only over actual MoEs.
                return None
            raise ValueError("Sparse DFC decoder is missing its routed experts")
        if any(
            type(getattr(experts, f"{prefix}_scheme", None)).__name__
            != "ModelSlimW8A8Int8MoE"
            for prefix in ("w13", "w2")
        ):
            raise ValueError(
                "ascend_dfc requires ModelSlim W8A8 for both routed projections"
            )
        runner = experts.moe_runner_config
        if (
            runner.activation != "silu"
            or not runner.is_gated
            or runner.apply_router_weight_on_input
            or any(
                getattr(runner, name, None) is not None
                for name in (
                    "swiglu_limit",
                    "gemm1_alpha",
                    "gemm1_beta",
                    "gemm1_clamp_limit",
                )
            )
        ):
            raise ValueError("DFC requires an unmodified gated SwiGLU expert FFN")
        if experts.num_fused_shared_experts:
            raise ValueError("ascend_dfc requires disabled shared expert fusion")
        w13, w2 = experts.w13_weight, experts.w2_weight
        if w13.dtype != torch.int8 or w2.dtype != torch.int8:
            raise ValueError("DFC requires INT8 routed expert weights")
        if (
            w13.device.type != "npu"
            or w2.device != w13.device
            or (self.device is not None and w13.device != self.device)
        ):
            raise ValueError("DFC expert weights must share one NPU device")
        if w13.ndim != 3 or w2.ndim != 3:
            raise ValueError("DFC requires rank-3 routed expert weights")
        local_experts, hidden, intermediate = w2.shape
        if (
            tuple(w13.shape) != (local_experts, 2 * intermediate, hidden)
            or hidden <= 0
            or hidden % 32
            or intermediate <= 0
            or (2 * intermediate) % 64
            or experts.moe_tp_size != 1
            or not 1 <= experts.moe_ep_size <= 16
            or not 1 <= experts.num_experts <= 4096
            or experts.num_experts != local_experts * experts.moe_ep_size
            or not 1 <= experts.top_k <= experts.num_experts
        ):
            raise ValueError(
                "DFC expert geometry does not match the native EP/INT8 ABI"
            )
        geometry = (
            local_experts,
            hidden,
            intermediate,
            experts.num_experts,
            experts.top_k,
        )
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("DFC currently requires uniform routed expert geometry")
        self.geometry = geometry
        index = len(self.layers)
        if self.layers and decoder.layer_id <= self.layers[-1].layer_id:
            raise ValueError("DFC MoE layers must be registered in increasing order")
        self.layers.append(decoder)
        if self.device is None:
            self.device = w13.device
        return index

    def _publish(self):
        if not self.layers:
            raise ValueError("DFC model has no local MoE layers")
        get_resources().buffers["ascend_dfc_model_adapter"] = self

    def wrap_modules(self, all_modules_generator, **kwargs):
        modules = []
        for decoder in all_modules_generator:
            modules.append(decoder)
            self._register_layer(decoder)
        self._publish()
        return modules

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
