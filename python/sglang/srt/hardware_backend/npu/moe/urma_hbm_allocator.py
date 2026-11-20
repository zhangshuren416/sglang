# SPDX-License-Identifier: Apache-2.0

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat
from sglang.srt.utils import round_up


class UrmaHbmAllocator:
    """Suballocate tensors from one rank's registered MemFabric HBM window.

    Allocate during initialization and reuse the tensors during inference. The
    tensors borrow the window's storage. Keep the allocator alive and drain all
    local and remote users before destroying the window.
    """

    def __init__(self, window, *, device: torch.device, local_mem_size: int):
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("URMA HBM allocation requires an explicit NPU device")

        symmetric_size = getattr(window, "symmetric_size", None)
        if symmetric_size is None:
            raise RuntimeError("MemFabric ShareMemory.symmetric_size is required")
        rank = window.local_rank
        gva = window.gva
        if not (gva > 0 and 0 <= rank < window.rank_size):
            raise ValueError("Invalid MemFabric HBM window address or local rank")
        if not 0 < local_mem_size <= symmetric_size:
            raise ValueError("Local HBM capacity must be within the rank window")

        self._make_storage = getattr(
            torch_npu._C, "_construct_storage_from_data_pointer", None
        )
        self._make_tensor = getattr(
            torch_npu._C, "_construct_NPU_Tensor_From_Storage_And_Metadata", None
        )
        if self._make_storage is None or self._make_tensor is None:
            raise RuntimeError(
                "TorchNPU external storage and tensor bindings are required"
            )

        self._window = window
        self._base = gva + rank * symmetric_size
        self._limit = self._base + local_mem_size
        self._next_ptr = self._base

    @property
    def base_address(self) -> int:
        return self._base

    @property
    def reserved_end_address(self) -> int:
        return self._next_ptr

    def reserve(self, nbytes: int, *, alignment: int = 512) -> int:
        """Reserve physical bytes and return the aligned window address."""
        if nbytes <= 0 or alignment <= 0:
            raise ValueError("Allocation size and alignment must be positive")
        ptr = round_up(self._next_ptr, alignment)
        if ptr + nbytes > self._limit:
            raise RuntimeError(
                f"URMA HBM window has {max(0, self._limit - ptr)} bytes available "
                f"after alignment, requested {nbytes} bytes"
            )
        self._next_ptr = ptr + nbytes
        return ptr

    def allocate(
        self,
        *,
        nbytes: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        npu_format: int,
        alignment: int = 512,
    ) -> torch.Tensor:
        """Reserve bytes and expose them as a tensor with the requested metadata."""
        ptr = self.reserve(nbytes, alignment=alignment)
        return self.wrap(ptr, nbytes, dtype, shape, stride, npu_format)

    def wrap(
        self,
        ptr: int,
        nbytes: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        npu_format: int,
    ) -> torch.Tensor:
        """Expose already reserved window bytes as a tensor with new metadata."""
        if nbytes <= 0:
            raise ValueError("Allocation size must be positive")
        if not (self._base <= ptr and ptr + nbytes <= self._next_ptr):
            raise ValueError("Requested tensor exceeds the reserved URMA HBM range")
        return self._wrap(ptr, nbytes, dtype, shape, stride, npu_format)

    def empty_like(
        self, template: torch.Tensor, *, alignment: int = 512
    ) -> torch.Tensor:
        """Reserve storage with the same physical layout as a base weight tensor."""
        storage = template.untyped_storage()
        if (
            template.device != self.device
            or template.storage_offset() != 0
            or template.data_ptr() != storage.data_ptr()
        ):
            raise ValueError("Expected a base weight tensor on the allocator's NPU")

        npu_format = torch_npu.get_npu_format(template)
        if npu_format not in (
            NPUACLFormat.ACL_FORMAT_ND,
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
        ):
            raise ValueError(f"Unsupported URMA weight format: {npu_format}")
        if (
            npu_format == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ
            and not template.is_contiguous()
        ):
            raise ValueError("Expected a contiguous FRACTAL_NZ weight template")

        nbytes = storage.nbytes()
        if nbytes <= 0 or alignment <= 0:
            raise ValueError("Allocation size and alignment must be positive")
        ptr = round_up(self._next_ptr, alignment)
        if ptr + nbytes > self._limit:
            raise RuntimeError(
                f"URMA HBM window has {max(0, self._limit - ptr)} bytes available "
                f"after alignment, requested {nbytes} bytes"
            )

        tensor = self._wrap(
            ptr,
            nbytes,
            template.dtype,
            tuple(template.shape),
            template.stride(),
            npu_format,
        )
        if torch_npu.get_storage_size(tensor) != torch_npu.get_storage_size(template):
            raise ValueError("Reconstructed tensor has a different NPU storage size")
        self._next_ptr = ptr + nbytes
        return tensor

    def byte_view(
        self, tensor: torch.Tensor, byte_offset: int, nbytes: int
    ) -> torch.Tensor:
        """Expose an allocated tensor's physical byte range as a URMA destination."""
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if not (
            tensor.device == self.device
            and self._base <= ptr < ptr + storage.nbytes() <= self._next_ptr
        ):
            raise ValueError("Expected tensor storage within this URMA HBM allocation")
        if not 0 <= byte_offset < byte_offset + nbytes <= storage.nbytes():
            raise ValueError("URMA byte range exceeds the tensor's physical storage")

        return self._wrap(
            ptr + byte_offset,
            nbytes,
            torch.uint8,
            (nbytes,),
            (1,),
            NPUACLFormat.ACL_FORMAT_ND,
        )

    def _wrap(
        self,
        ptr: int,
        nbytes: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        npu_format: int,
    ) -> torch.Tensor:
        storage = self._make_storage(ptr, self.device, nbytes)
        tensor = self._make_tensor(
            {
                "dtype": dtype,
                "size": shape,
                "stride": stride,
                "storage_offset": 0,
                "npu_format": int(npu_format),
            },
            storage,
        )
        if (
            tensor.data_ptr() != ptr
            or tensor.device != self.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or tensor.stride() != stride
            or tensor.storage_offset() != 0
            or tensor.untyped_storage().nbytes() != nbytes
            or torch_npu.get_npu_format(tensor) != npu_format
        ):
            raise RuntimeError(
                "TorchNPU did not preserve the requested URMA tensor layout"
            )
        return tensor
