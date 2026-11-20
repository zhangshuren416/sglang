# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch

from sglang.srt.hardware_backend.npu.moe.weight_sharing.config import (
    MEMFABRIC_CONTROL_TIMEOUT_SECS,
    WeightShareConfig,
)
from sglang.srt.utils import check_return_code, is_npu

if is_npu():
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.moe.urma_hbm_allocator import (
        UrmaHbmAllocator,
    )

logger = logging.getLogger(__name__)

_OFFLOAD_DRAM_BYTES = 1 << 30


class WeightShareResources:
    def __init__(
        self,
        config: WeightShareConfig,
        *,
        device: torch.device,
        local_mem_size: int,
        group_rank: int,
    ):
        import memfabric_hybrid as mf
        from memfabric_hybrid import offload, shm

        self._mf = mf
        self._offload = offload
        self._shm = shm
        self._closed = False
        self._mf_ready = False
        self._offload_ready = False
        self._shm_ready = False
        self.shm_window = None
        self.allocator = None

        try:
            check_return_code(mf.initialize(), "memfabric_hybrid.initialize")
            self._mf_ready = True
            shm_config = shm.ShmConfig()
            shm_config.operation_timeout = MEMFABRIC_CONTROL_TIMEOUT_SECS
            shm_config.start_store = group_rank == 0
            check_return_code(
                shm.initialize(
                    config.store_url,
                    config.world_size,
                    group_rank,
                    device.index,
                    shm_config,
                ),
                "shm.initialize",
            )
            self._shm_ready = True

            self.shm_window = shm.create(
                id=0,
                rank_size=config.world_size,
                rank_id=group_rank,
                local_mem_size=local_mem_size,
                data_op_type=shm.ShmDataOpType.AIV_URMA,
            )
            check_return_code(self.shm_window.barrier(), "shm.barrier")

            symmetric_size = getattr(self.shm_window, "symmetric_size", None)
            if symmetric_size is None or local_mem_size > symmetric_size:
                raise RuntimeError(
                    f"MemFabric window exposes {symmetric_size} bytes per rank, "
                    f"requested {local_mem_size}"
                )

            offload_config = offload.OffloadConfig()
            offload_config.device_id = device.index
            offload_config.scene = offload.Scene.LOCAL
            offload_config.reserve_size = _OFFLOAD_DRAM_BYTES
            offload_config.alloc_size = _OFFLOAD_DRAM_BYTES
            check_return_code(offload.initialize(offload_config), "offload.initialize")
            self._offload_ready = True

            self.allocator = UrmaHbmAllocator(
                self.shm_window, device=device, local_mem_size=local_mem_size
            )
        except BaseException:
            self._release()
            raise

    @property
    def rank_size(self) -> int:
        return self.shm_window.rank_size

    def create_copy_plan(self, entries, *, device):
        return self._offload.UrmaCopyPlan(entries, device=device)

    def peer_address(self, peer_rank: int, local_ptr: int) -> int:
        symmetric_size = self.shm_window.symmetric_size
        local_base = self.shm_window.gva + self.shm_window.local_rank * symmetric_size
        peer_base = self.shm_window.gva + peer_rank * symmetric_size
        return peer_base + (local_ptr - local_base)

    def close(self) -> None:
        if self._closed:
            return
        if self.shm_window is not None:
            torch.npu.synchronize()
            check_return_code(self.shm_window.barrier(), "shm.barrier during teardown")
        self._release()

    def _release(self) -> None:
        self._closed = True
        if self._offload_ready:
            self._offload.uninitialize()
            self._offload_ready = False
        if self.shm_window is not None:
            self.shm_window.destroy()
            self.shm_window = None
        if self._shm_ready:
            self._shm.uninitialize(0)
            self._shm_ready = False
        if self._mf_ready:
            self._mf.uninitialize()
            self._mf_ready = False
        logger.info("MoE weight sharing window closed")
