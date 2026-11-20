# SPDX-License-Identifier: Apache-2.0
"""Server-argument handling for cross-instance MoE weight sharing."""

from __future__ import annotations

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.hardware_backend.npu.moe.weight_sharing.config import (
    validate_weight_share_args,
)


def handle_moe_weight_sharing(server_args) -> None:
    cfg = resolving_view(server_args)
    if not getattr(cfg, "moe_weight_share_size", 0):
        return
    if cfg.disaggregation_mode != "prefill":
        raise ValueError(
            "MoE weight sharing requires --disaggregation-mode prefill, got "
            f"{cfg.disaggregation_mode}"
        )
    if cfg.enable_pd_role_switch:
        raise ValueError("MoE weight sharing does not support --enable-pd-role-switch")

    declare_resolution(
        server_args,
        "handle_moe_weight_sharing",
        disable_shared_experts_fusion=True,
    )
    validate_weight_share_args(cfg)
