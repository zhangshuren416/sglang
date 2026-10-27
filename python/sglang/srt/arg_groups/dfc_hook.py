# SPDX-License-Identifier: Apache-2.0
"""Validation for the opt-in eager A3 DFC prefill path."""

from __future__ import annotations

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    with_phase,
)


def handle_dfc_config(server_args):
    cfg = resolving_view(server_args)
    enabled = cfg.moe_weight_offload_backend == "memfabric"
    if cfg.moe_a2a_backend != "ascend_dfc":
        if (
            enabled
            or cfg.moe_weight_offload_config is not None
            or cfg.dfc_max_output_tokens is not None
        ):
            raise ValueError(
                "DFC weight offload/capacity options require "
                "--moe-a2a-backend ascend_dfc"
            )
        return
    if cfg.device not in (None, "npu") or cfg.quantization != "modelslim":
        raise ValueError("ascend_dfc requires NPU and ModelSlim W8A8 quantization")
    if cfg.dtype not in ("auto", "bfloat16"):
        raise ValueError("ascend_dfc requires BF16 activations")
    if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() <= 0:
        raise ValueError(
            "ascend_dfc requires the existing "
            "SGLANG_ZBAL_LOCAL_MEM_SIZE runtime configuration"
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "ascend_dfc is a prefill backend; "
            "keep the existing backend on decode workers"
        )
    if (
        cfg.pp_size != 1
        or cfg.moe_dp_size != 1
        or cfg.attn_cp_size != 1
        or cfg.dcp_size != 1
    ):
        raise ValueError(
            "ascend_dfc requires PP1, MoE-DP1, no CP/DCP, and EP spanning TP"
        )
    if (
        cfg.nnodes < 1
        or cfg.tp_size < 1
        or cfg.nnodes > cfg.tp_size
        or cfg.tp_size % cfg.nnodes
    ):
        raise ValueError(
            "DFC requires an equal positive number of TP ranks on every node"
        )
    if (
        cfg.tp_size > 16
        or cfg.max_ep_size not in (None, cfg.tp_size)
        or cfg.elastic_ep_initial_size not in (None, cfg.tp_size)
    ):
        raise ValueError("Native A3 DFC requires fixed EP spanning TP, with EP <=16")
    unsupported = (
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "enable_eplb",
        "enable_waterfill",
        "enforce_shared_experts_fusion",
        "enable_prefill_cp",
        "enable_lora",
        "enable_memory_saver",
        "enable_elastic_expert_backup",
        "enable_weights_cpu_backup",
        "enable_torch_compile",
        "enable_pd_role_switch",
    )
    for name in unsupported:
        if getattr(cfg, name, False):
            raise ValueError(f"ascend_dfc does not support {name}")
    if cfg.speculative_algorithm or cfg.ep_num_redundant_experts or cfg.ep_join_mode:
        raise ValueError(
            "ascend_dfc does not support speculative decoding "
            "or dynamic/redundant experts"
        )
    if cfg.dwdp_size != 1 or cfg.cpu_offload_gb > 0 or cfg.offload_group_size > 0:
        raise ValueError(
            "ascend_dfc cannot be combined with DWDP or the generic weight offloaders"
        )
    if cfg.dfc_max_output_tokens is not None and cfg.dfc_max_output_tokens <= 0:
        raise ValueError("--dfc-max-output-tokens must be positive")
    if cfg.load_format not in ("auto", "safetensors"):
        raise ValueError(
            "ascend_dfc currently uses the native local auto/safetensors loader"
        )
    if cfg.weight_cache_mode != "off":
        raise ValueError("ascend_dfc cannot attach to an IPC weight cache")
    if enabled:
        if cfg.disaggregation_mode != "prefill":
            raise ValueError(
                "MemFabric MoE weight offload requires a PD prefill instance"
            )
        if not cfg.moe_weight_offload_config:
            raise ValueError(
                "MemFabric MoE weight offload requires --moe-weight-offload-config"
            )
        if cfg.enable_hierarchical_cache:
            from sglang.srt.mem_cache.pool_host.npu_memfabric import (
                memfabric_host_memory_enabled,
            )

            if memfabric_host_memory_enabled():
                raise ValueError(
                    "DFC weights and HiCache cannot share the MemFabric offload entity"
                )
        from sglang.srt.hardware_backend.npu.moe.dfc_config import WeightPoolConfig

        WeightPoolConfig.read(cfg.moe_weight_offload_config)
    elif cfg.moe_weight_offload_config is not None:
        raise ValueError(
            "--moe-weight-offload-config requires "
            "--moe-weight-offload-backend memfabric"
        )
    # This hook also runs before the generic graph parser; the raw default is None.
    graph = cfg.cuda_graph_config or CudaGraphConfig()
    graph = with_phase(graph, Phase.PREFILL, backend=Backend.DISABLED)
    graph = with_phase(graph, Phase.DECODE, backend=Backend.DISABLED)
    declare_resolution(
        server_args,
        "handle_dfc_config",
        disable_shared_experts_fusion=True,
        cuda_graph_config=graph,
    )
