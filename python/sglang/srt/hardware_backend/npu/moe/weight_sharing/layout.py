# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from sglang.srt.utils import round_up

_SHARED_PREFIXES = ("w13_", "w2_")


def owned_range(rank: int, num_experts: int, group_size: int) -> Tuple[int, int]:
    base, remainder = divmod(num_experts, group_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def owned_max(num_experts: int, group_size: int) -> int:
    return (num_experts + group_size - 1) // group_size


@dataclass(frozen=True)
class ExpertLayout:
    per_expert_bytes: int
    base_shape: Tuple[int, ...]
    base_stride: Tuple[int, ...]
    inverse_perm: Tuple[int, ...]
    dtype: torch.dtype
    npu_format: int

    def nbytes(self, num_experts: int) -> int:
        return self.per_expert_bytes * num_experts


def derive_expert_layout(tensor: torch.Tensor, npu_format: int) -> ExpertLayout:
    if tensor.storage_offset() != 0:
        raise ValueError(
            f"Shared tensor {tuple(tensor.shape)} must start at storage offset 0"
        )
    shape = tuple(tensor.shape)
    if not shape or shape[0] == 0:
        raise ValueError(f"Shared tensor {shape} must have a nonempty expert dimension")
    storage_bytes = tensor.untyped_storage().nbytes()
    if storage_bytes % shape[0] != 0:
        raise ValueError(
            f"Shared tensor {shape} storage {storage_bytes} is not expert aligned"
        )
    per_expert_bytes = storage_bytes // shape[0]
    element_bytes = tensor.element_size()
    if per_expert_bytes % element_bytes != 0:
        raise ValueError(f"Shared tensor {shape} expert bytes are not element aligned")

    stride = tensor.stride()
    if len(shape) == 1 and (stride != (1,) or per_expert_bytes != element_bytes):
        raise ValueError(
            f"Shared tensor {shape} must store one contiguous scalar per expert"
        )
    order = sorted(
        range(1, len(shape)), key=lambda d: (-stride[d], -shape[d], d)
    )

    base_shape = (shape[0],) + tuple(shape[d] for d in order)
    inner_stride = [1] * len(order)
    for i in range(len(order) - 2, -1, -1):
        inner_stride[i] = inner_stride[i + 1] * base_shape[i + 2]
    for i, d in enumerate(order):
        if shape[d] > 1 and stride[d] != inner_stride[i]:
            raise ValueError(
                f"Shared tensor {shape} stride {stride} is not permuted contiguous"
            )

    inverse = [0] * len(shape)
    for i, d in enumerate(order):
        inverse[d] = i + 1
    return ExpertLayout(
        per_expert_bytes=per_expert_bytes,
        base_shape=base_shape,
        base_stride=(0,) + tuple(inner_stride),
        inverse_perm=tuple(inverse),
        dtype=tensor.dtype,
        npu_format=npu_format,
    )


def nz_upper_bytes_per_expert(
    shape: Tuple[int, ...], element_bytes: int, *, column_alignment: int = 16
) -> int:
    """Bound NZ storage for either orientation of the expert matrix."""
    dims = shape[1:]
    if len(dims) < 2:
        return max(1, torch.Size(dims).numel()) * element_bytes
    outer = 1
    for dim in dims[:-2]:
        outer *= dim
    rows, cols = dims[-2:]
    padded_elements = max(
        round_up(rows, 16) * round_up(cols, column_alignment),
        round_up(cols, 16) * round_up(rows, column_alignment),
    )
    return outer * padded_elements * element_bytes


def reserve_bytes_per_expert(tensor: torch.Tensor, *, is_weight: bool) -> int:
    shape = tuple(tensor.shape)
    element_bytes = tensor.element_size()
    pre_bytes = tensor.untyped_storage().nbytes() // shape[0]
    if not is_weight:
        return 2 * pre_bytes + 4096
    return max(
        pre_bytes,
        nz_upper_bytes_per_expert(
            shape, element_bytes, column_alignment=max(16, 32 // element_bytes)
        ),
    )


def shared_tensor_items(layer: torch.nn.Module, num_experts: int) -> Dict[str, torch.Tensor]:
    items: Dict[str, torch.Tensor] = {}
    candidates = list(layer.named_parameters(recurse=False))
    candidates.extend(layer.named_buffers(recurse=False))
    candidates.extend(vars(layer).items())
    for name, value in candidates:
        if not name.startswith(_SHARED_PREFIXES) or name in items:
            continue
        if not isinstance(value, torch.Tensor) or value.dim() == 0:
            continue
        if value.shape[0] == num_experts:
            items[name] = value
    return items


def allocate_experts(allocator, layout: ExpertLayout, num_experts: int):
    element_bytes = layout.dtype.itemsize
    if layout.per_expert_bytes % element_bytes != 0:
        raise ValueError("Expert byte stride is not element aligned")
    base_shape = (num_experts,) + layout.base_shape[1:]
    base_stride = (
        layout.per_expert_bytes // element_bytes,
    ) + layout.base_stride[1:]
    base = allocator.allocate(
        nbytes=layout.nbytes(num_experts),
        dtype=layout.dtype,
        shape=base_shape,
        stride=base_stride,
        npu_format=layout.npu_format,
    )
    if layout.inverse_perm == tuple(range(len(layout.inverse_perm))):
        return base, base
    return base, base.permute(*layout.inverse_perm)
