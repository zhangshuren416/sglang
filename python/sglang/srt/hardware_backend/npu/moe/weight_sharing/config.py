# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    is_weight_sharing_enabled,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

PREFETCH_THREAD_JOIN_TIMEOUT_SECS = 30
MEMFABRIC_CONTROL_TIMEOUT_SECS = 120


@dataclass(frozen=True)
class WeightShareConfig:
    group_size: int
    instance_rank: int
    store_url: str
    pool_bytes_override: Optional[int]
    tp_size: int
    pp_size: int

    @property
    def world_size(self) -> int:
        return self.group_size * self.tp_size * self.pp_size

    def group_rank(self, pp_rank: int, tp_rank: int) -> int:
        return self.group_rank_for(self.instance_rank, pp_rank, tp_rank)

    def group_rank_for(self, instance_rank: int, pp_rank: int, tp_rank: int) -> int:
        return (
            instance_rank * self.tp_size * self.pp_size
            + pp_rank * self.tp_size
            + tp_rank
        )


_config: Optional[WeightShareConfig] = None
_config_source = None


def get_gloo_init_method(store_url: str) -> str:
    address = store_url.removeprefix("tcp://")
    if "://" in address:
        raise ValueError("--moe-weight-share-store-url must use a TCP store")
    store = NetworkAddress.parse(address)
    if not 0 < store.port < 65535:
        raise ValueError(
            "--moe-weight-share-store-url must use a port from 1 to 65534 "
            "because the next port is used for Gloo"
        )
    host = store.host.split("/", 1)[0]  # Strip the optional MemFabric subnet mask.
    return NetworkAddress(host, store.port + 1).to_tcp()


def validate_weight_share_args(server_args) -> None:
    size = server_args.moe_weight_share_size
    if size < 2:
        raise ValueError("--moe-weight-share-size must be at least 2")
    if not 0 <= server_args.moe_weight_share_rank < size:
        raise ValueError(
            f"--moe-weight-share-rank must be in [0, {size}), got "
            f"{server_args.moe_weight_share_rank}"
        )
    if not is_npu():
        raise ValueError("MoE weight sharing requires Ascend NPU")
    if server_args.moe_a2a_backend not in ("none", "deepep"):
        raise ValueError(
            "MoE weight sharing requires --moe-a2a-backend none or deepep, got "
            f"{server_args.moe_a2a_backend}"
        )
    if server_args.enable_eplb:
        raise ValueError("MoE weight sharing does not support EPLB")
    if server_args.dp_size > 1:
        raise ValueError(
            "MoE weight sharing does not support data-parallel model replicas "
            f"(--dp-size {server_args.dp_size})"
        )
    if server_args.dwdp_size != 1:
        raise ValueError("MoE weight sharing does not support DWDP")
    if server_args.enable_two_batch_overlap:
        raise ValueError("MoE weight sharing does not support two-batch overlap")
    if server_args.startup_weight_load_mode == "overlap":
        raise ValueError(
            "MoE weight sharing does not support overlapped startup weight loading"
        )
    if not server_args.disable_shared_experts_fusion:
        raise ValueError(
            "MoE weight sharing requires --disable-shared-experts-fusion"
        )
    graph = server_args.cuda_graph_config
    if graph is None or graph.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "MoE weight sharing requires prefill CUDA graphs to be disabled"
        )
    if os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND", "").lower() == "memfabric":
        raise ValueError(
            "MoE weight sharing cannot share the MemFabric offload entity with "
            "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric"
        )

    store_url = server_args.moe_weight_share_store_url
    if not store_url:
        raise ValueError("--moe-weight-share-store-url is required")
    get_gloo_init_method(store_url)
    pd_store_url = os.environ.get("ASCEND_MF_STORE_URL")
    if pd_store_url and store_url == pd_store_url:
        raise ValueError(
            "--moe-weight-share-store-url must differ from ASCEND_MF_STORE_URL"
        )

    override = server_args.moe_weight_share_bytes
    if override is not None and override <= 0:
        raise ValueError("--moe-weight-share-bytes must be positive")


def get_weight_share_config() -> Optional[WeightShareConfig]:
    global _config, _config_source

    if not is_weight_sharing_enabled():
        return None
    moe = get_exec().moe
    if _config_source is moe:
        return _config

    parallel = get_parallel()
    _config = WeightShareConfig(
        group_size=moe.moe_weight_share_size,
        instance_rank=moe.moe_weight_share_rank,
        store_url=moe.moe_weight_share_store_url,
        pool_bytes_override=moe.moe_weight_share_bytes,
        tp_size=parallel.tp_size,
        pp_size=parallel.pp_size,
    )
    _config_source = moe
    logger.info(
        "MoE weight sharing enabled: instance %d/%d, tp_size=%d, pp_size=%d, store=%s",
        _config.instance_rank,
        _config.group_size,
        _config.tp_size,
        _config.pp_size,
        _config.store_url,
    )
    return _config


def reset_weight_share_config_for_test() -> None:
    global _config, _config_source
    _config = None
    _config_source = None
