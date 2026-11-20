# SPDX-License-Identifier: Apache-2.0
"""Cross-instance MoE expert weight sharing for Ascend prefill workers."""

from sglang.srt.hardware_backend.npu.moe.weight_sharing.config import (
    WeightShareConfig,
    get_weight_share_config,
)
from sglang.srt.hardware_backend.npu.moe.weight_sharing.layout import (
    owned_range,
)

__all__ = ["WeightShareConfig", "get_weight_share_config", "owned_range"]
