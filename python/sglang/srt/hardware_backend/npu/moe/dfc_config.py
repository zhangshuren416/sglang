# SPDX-License-Identifier: Apache-2.0
"""Host-only configuration and layout checks for the eager A3 DFC path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightPoolConfig:
    pool_id: str
    store_url: str
    reserve_bytes_per_rank: int | str = "auto"
    sdma_channels: int = 4

    @classmethod
    def read(cls, path: str) -> WeightPoolConfig:
        data = json.loads(Path(path).read_text())
        allowed = {
            "schema_version",
            "pool_id",
            "store_url",
            "reserve_bytes_per_rank",
            "sdma_channels",
        }
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("Invalid DFC weight pool configuration keys")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("DFC weight pool schema_version must be 1")
        for name in ("pool_id", "store_url"):
            if not isinstance(data.get(name), str) or not data[name].strip():
                raise ValueError(f"DFC weight pool requires a non-empty {name}")
        if not data["store_url"].startswith(("tcp://", "etcd://", "reg://")):
            raise ValueError("Unsupported DFC weight pool store_url scheme")
        if len(data["store_url"].encode()) >= 64:
            raise ValueError(
                "DFC weight pool store_url must fit the native 64-byte field"
            )
        reserve = data.get("reserve_bytes_per_rank", "auto")
        if reserve != "auto" and (type(reserve) is not int or reserve <= 0):
            raise ValueError(
                "reserve_bytes_per_rank must be 'auto' or a positive integer"
            )
        channels = data.get("sdma_channels", 4)
        if type(channels) is not int or not 1 <= channels <= 8:
            raise ValueError("DFC SDMA channels must be an integer in [1, 8]")
        return cls(data["pool_id"], data["store_url"], reserve, channels)


def slot_for_layer(index: int) -> int:
    if index < 1:
        raise ValueError("The first MoE layer is resident and has no offload slot")
    return (index - 1) % 2


def physical_layout(tensor, npu_format: int) -> tuple:
    """Reject ambiguous views before exposing storage to a raw DMA interface."""
    storage = tensor.untyped_storage()
    if tensor.storage_offset() != 0 or tensor.data_ptr() != storage.data_ptr():
        raise ValueError("DFC offload requires a tensor at its physical storage base")
    if storage.nbytes() < tensor.numel() * tensor.element_size():
        raise ValueError("DFC tensor physical storage is smaller than its logical data")
    if npu_format not in (2, 29):
        raise ValueError(
            "DFC offload currently supports ND and FRACTAL_NZ storage only"
        )
    if npu_format == 2 and storage.nbytes() != tensor.numel() * tensor.element_size():
        raise ValueError("DFC offload cannot export a partial ND storage view")
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tuple(tensor.stride()),
        npu_format,
        storage.nbytes(),
    )


def pool_capacity(required_bytes: int, configured: int | str) -> int:
    if required_bytes <= 0:
        raise ValueError("DFC pool must contain at least one nonresident weight")
    gib = 1 << 30
    requested = required_bytes if configured == "auto" else configured
    if type(requested) is not int or requested < required_bytes:
        raise ValueError(
            f"DFC pool needs {required_bytes} bytes, configured {configured}"
        )
    return ((requested + gib - 1) // gib) * gib
