# SPDX-License-Identifier: Apache-2.0
"""Backend-independent MemFabric expert weight pool and raw layout checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightPoolConfig:
    store_url: str
    sdma_backend: str
    reserve_bytes: int | str = "auto"
    sdma_channels: int = 4

    @classmethod
    def read(cls, path: str) -> WeightPoolConfig:
        data = json.loads(Path(path).read_text())
        allowed = {
            "store_url",
            "reserve_bytes",
            "sdma_channels",
            "sdma_backend",
        }
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("Invalid MoE weight pool configuration keys")
        if not isinstance(data.get("store_url"), str) or not data["store_url"].strip():
            raise ValueError("MoE weight pool requires a non-empty store_url")
        if not data["store_url"].startswith(("tcp://", "etcd://", "reg://")):
            raise ValueError("Unsupported MoE weight pool store_url scheme")
        if len(data["store_url"].encode()) >= 64:
            raise ValueError(
                "MoE weight pool store_url must fit the native 64-byte field"
            )
        reserve = data.get("reserve_bytes", "auto")
        if reserve != "auto" and (type(reserve) is not int or reserve <= 0):
            raise ValueError(
                "reserve_bytes must be 'auto' or a positive integer"
            )
        channels = data.get("sdma_channels", 4)
        if type(channels) is not int or not 1 <= channels <= 8:
            raise ValueError("MoE weight SDMA channels must be an integer in [1, 8]")
        backend = data.get("sdma_backend")
        if backend not in ("aiv", "aicpu"):
            raise ValueError(
                "MoE weight pool requires explicit sdma_backend: 'aiv' or 'aicpu'"
            )
        return cls(
            store_url=data["store_url"],
            sdma_backend=backend,
            reserve_bytes=reserve,
            sdma_channels=channels,
        )


def slot_for_layer(index: int) -> int:
    if index < 1:
        raise ValueError("The first MoE layer is resident and has no offload slot")
    return (index - 1) % 2


def physical_layout(tensor, npu_format: int) -> tuple:
    """Reject ambiguous views before exposing storage to a raw DMA interface."""
    storage = tensor.untyped_storage()
    if tensor.storage_offset() != 0 or tensor.data_ptr() != storage.data_ptr():
        raise ValueError(
            "MoE weight offload requires a tensor at its physical storage base"
        )
    if storage.nbytes() < tensor.numel() * tensor.element_size():
        raise ValueError(
            "MoE weight tensor physical storage is smaller than its logical data"
        )
    if npu_format not in (2, 29):
        raise ValueError(
            "MoE weight offload currently supports ND and FRACTAL_NZ storage only"
        )
    if npu_format == 2 and storage.nbytes() != tensor.numel() * tensor.element_size():
        raise ValueError("MoE weight offload cannot export a partial ND storage view")
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tuple(tensor.stride()),
        npu_format,
        storage.nbytes(),
    )


def pool_capacity(required_bytes: int, configured: int | str) -> int:
    if required_bytes <= 0:
        raise ValueError("MoE weight pool must contain at least one nonresident weight")
    gib = 1 << 30
    requested = required_bytes if configured == "auto" else configured
    if type(requested) is not int or requested < required_bytes:
        raise ValueError(
            f"MoE weight pool needs {required_bytes} bytes, configured {configured}"
        )
    return ((requested + gib - 1) // gib) * gib
