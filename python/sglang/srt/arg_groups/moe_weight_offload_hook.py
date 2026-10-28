# SPDX-License-Identifier: Apache-2.0
"""Validation for MemFabric weight transport, independent of MoE execution."""

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.hardware_backend.npu.moe.weight_offload_config import WeightPoolConfig
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    with_phase,
)


def handle_moe_weight_offload_config(server_args):
    cfg = resolving_view(server_args)
    if cfg.moe_weight_offload_backend != "memfabric":
        if cfg.moe_weight_offload_config is not None:
            raise ValueError(
                "--moe-weight-offload-config requires "
                "--moe-weight-offload-backend memfabric"
            )
        return
    if cfg.device not in (None, "npu") or cfg.quantization != "modelslim":
        raise ValueError("MemFabric weight offload requires NPU and ModelSlim W8A8")
    if cfg.dtype not in ("auto", "bfloat16"):
        raise ValueError("MemFabric weight offload requires BF16 activations")
    if cfg.moe_a2a_backend not in ("deepep", "ascend_dfc"):
        raise ValueError(
            "MemFabric weight offload currently supports DeepEP or ascend_dfc execution"
        )
    if not cfg.moe_weight_offload_config:
        raise ValueError(
            "MemFabric weight offload requires --moe-weight-offload-config"
        )
    WeightPoolConfig.read(cfg.moe_weight_offload_config)
    if (
        cfg.pp_size != 1
        or cfg.moe_dp_size != 1
        or cfg.attn_cp_size != 1
        or cfg.dcp_size != 1
        or cfg.dwdp_size != 1
    ):
        raise ValueError(
            "MemFabric weight offload requires PP1, MoE-DP1, no CP/DCP/DWDP"
        )
    if cfg.nnodes < 1 or cfg.tp_size < cfg.nnodes or cfg.tp_size % cfg.nnodes:
        raise ValueError(
            "Weight offload requires equal positive TP rank counts per node"
        )
    if (
        cfg.max_ep_size not in (None, cfg.tp_size)
        or cfg.elastic_ep_initial_size not in (None, cfg.tp_size)
        or cfg.speculative_algorithm
        or cfg.ep_num_redundant_experts
        or cfg.ep_join_mode
    ):
        raise ValueError(
            "MemFabric weight offload requires fixed EP without speculative decoding "
            "or dynamic/redundant experts"
        )
    if cfg.cpu_offload_gb > 0 or cfg.offload_group_size > 0:
        raise ValueError("MemFabric weight offload cannot use generic CPU offloaders")
    for name in (
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
    ):
        if getattr(cfg, name, False):
            raise ValueError(f"MemFabric weight offload does not support {name}")
    if cfg.load_format not in ("auto", "safetensors") or cfg.weight_cache_mode != "off":
        raise ValueError(
            "MemFabric weight offload requires native local checkpoint loading"
        )
    if cfg.enable_hierarchical_cache:
        from sglang.srt.mem_cache.pool_host.npu_memfabric import (
            memfabric_host_memory_enabled,
        )

        if memfabric_host_memory_enabled():
            raise ValueError(
                "MoE weights and HiCache cannot share the MemFabric entity"
            )

    # Hooks fence eager module calls. PD prefill, decode and colocated workers
    # can use the same ring; the selected compute backend adds its own limits.
    graph = cfg.cuda_graph_config or CudaGraphConfig()
    graph = with_phase(graph, Phase.PREFILL, backend=Backend.DISABLED)
    graph = with_phase(graph, Phase.DECODE, backend=Backend.DISABLED)
    declare_resolution(
        server_args,
        "handle_moe_weight_offload_config",
        disable_shared_experts_fusion=True,
        disable_overlap_schedule=True,
        cuda_graph_config=graph,
    )
