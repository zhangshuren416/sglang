# SPDX-License-Identifier: Apache-2.0
"""Local and two-rank allocator smoke tests for Ascend A5.

The local case needs one NPU and ShareMemory.symmetric_size. The two-rank
case also needs a second NPU and UrmaCopyPlan from the MemFabric wheel.
"""

import unittest
from contextlib import ExitStack

import acl
import memfabric_hybrid as mf
import torch
import torch.multiprocessing as mp
import torch_npu
from memfabric_hybrid import offload, shm

from sglang.srt.hardware_backend.npu.moe.urma_hbm_allocator import UrmaHbmAllocator
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat
from sglang.srt.utils.network import get_open_port
from sglang.test.test_utils import CustomTestCase

_WORLD_SIZE = 2
_WINDOW_BYTES = 1 << 30
_OFFLOAD_DRAM_BYTES = 1 << 30
_ACL_MEMCPY_DEVICE_TO_DEVICE = 3
_MATRIX_SIZE = 128
_FORMATS = (NPUACLFormat.ACL_FORMAT_ND, NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)


def _check_ret(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} failed with return code {ret}")


def _make_template(rank, device, npu_format):
    pattern = torch.arange(_MATRIX_SIZE**2, dtype=torch.int32)
    pattern = pattern.remainder(31).reshape(_MATRIX_SIZE, _MATRIX_SIZE).half()
    template = (pattern + rank * 32).to(device)
    if npu_format == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ:
        template = torch_npu.npu_format_cast(template, int(npu_format))
    if torch_npu.get_npu_format(template) != npu_format:
        raise AssertionError(f"[rank {rank}] cannot create {npu_format.name} template")
    return template, pattern


def _copy_storage(template, *destinations):
    # A5's Tensor.copy_ does not accept FRACTAL_NZ. Copy the packed bytes.
    nbytes = template.untyped_storage().nbytes()
    torch.npu.synchronize(template.device)
    for tensor in destinations:
        _check_ret(
            acl.rt.memcpy(
                tensor.data_ptr(),
                nbytes,
                template.data_ptr(),
                nbytes,
                _ACL_MEMCPY_DEVICE_TO_DEVICE,
            ),
            f"acl.rt.memcpy D2D on {template.device} ({nbytes} bytes)",
        )
    return nbytes


def _compute_result(identity, tensor):
    # Read the reconstructed NZ layout with format cast before ordinary mm.
    if torch_npu.get_npu_format(tensor) == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ:
        tensor = torch_npu.npu_format_cast(tensor, int(NPUACLFormat.ACL_FORMAT_ND))
    return torch.mm(identity, tensor).cpu()


def _check_format(rank, window, allocator, copy_stream, sync, npu_format):
    device = allocator.device
    peer = 1 - rank
    template, pattern = _make_template(rank, device, npu_format)

    source = allocator.empty_like(template)
    whole = allocator.empty_like(template)
    split = allocator.empty_like(template)

    # Publish the actual device layout and prefill both destinations with this
    # rank's pattern so a missing or partial transfer cannot pass by chance.
    nbytes = _copy_storage(template, source, whole, split)

    local_base = window.gva + rank * window.symmetric_size
    peer_base = window.gva + peer * window.symmetric_size
    # Both ranks use the same allocation order and tensor layouts.
    peer_source = peer_base + (source.data_ptr() - local_base)
    half = nbytes // 2
    copies = [
        (peer_source, whole, nbytes),
        (peer_source, allocator.byte_view(split, 0, half), half),
        (
            peer_source + half,
            allocator.byte_view(split, half, nbytes - half),
            nbytes - half,
        ),
    ]
    sync.wait()
    try:
        with offload.UrmaCopyPlan(copies, device=device) as plan:
            with torch.npu.stream(copy_stream):
                plan.enqueue()
    finally:
        # Keep both source windows live until both workers leave the copy phase.
        sync.wait()

    identity = torch.eye(_MATRIX_SIZE, dtype=template.dtype, device=device)
    expected = pattern + peer * 32
    for label, tensor in (("whole tensor", whole), ("split byte views", split)):
        torch.testing.assert_close(
            _compute_result(identity, tensor),
            expected,
            rtol=0,
            atol=0,
            msg=lambda message: f"[rank {rank}] {npu_format.name} {label}\n{message}",
        )
    print(f"[rank {rank}] {npu_format.name} whole and split copies passed", flush=True)


def _open_window(cleanup, rank, world_size, store_url, data_op_type):
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True

    _check_ret(mf.initialize(), "memfabric_hybrid.initialize")
    cleanup.callback(mf.uninitialize)

    shm_config = shm.ShmConfig()
    shm_config.start_store = rank == 0
    _check_ret(
        shm.initialize(store_url, world_size, rank, device.index, shm_config),
        "shm.initialize",
    )
    cleanup.callback(shm.uninitialize, 0)
    window = shm.create(
        id=0,
        rank_size=world_size,
        rank_id=rank,
        local_mem_size=_WINDOW_BYTES,
        data_op_type=data_op_type,
    )
    cleanup.callback(window.destroy)
    _check_ret(window.barrier(), "shm.barrier")
    return window, device


def _run_local(rank, store_url):
    with ExitStack() as cleanup:
        # AIV_URMA requires a peer context during mmap, so use the local HBM path.
        window, device = _open_window(
            cleanup, rank, 1, store_url, shm.ShmDataOpType.MTE
        )
        cleanup.callback(torch.npu.synchronize, device)
        allocator = UrmaHbmAllocator(
            window, device=device, local_mem_size=_WINDOW_BYTES
        )
        identity = torch.eye(_MATRIX_SIZE, dtype=torch.float16, device=device)
        for npu_format in _FORMATS:
            template, expected = _make_template(rank, device, npu_format)
            tensor = allocator.empty_like(template)
            ptr = tensor.data_ptr()
            _copy_storage(template, tensor)
            if tensor.data_ptr() != ptr:
                raise AssertionError(f"{npu_format.name} copy changed the address")
            torch.testing.assert_close(
                _compute_result(identity, tensor), expected, rtol=0, atol=0
            )
            print(f"[local] {npu_format.name} copy and compute passed", flush=True)


def _run_rank(rank, store_url, sync):
    with ExitStack() as cleanup:
        window, device = _open_window(
            cleanup, rank, _WORLD_SIZE, store_url, shm.ShmDataOpType.AIV_URMA
        )

        config = offload.OffloadConfig()
        config.device_id = device.index
        config.scene = offload.Scene.LOCAL
        config.reserve_size = _OFFLOAD_DRAM_BYTES
        config.alloc_size = _OFFLOAD_DRAM_BYTES
        _check_ret(offload.initialize(config), "offload.initialize")
        cleanup.callback(offload.uninitialize)
        cleanup.callback(torch.npu.synchronize, device)

        allocator = UrmaHbmAllocator(
            window, device=device, local_mem_size=_WINDOW_BYTES
        )
        copy_stream = torch.npu.Stream(device=device)
        for npu_format in _FORMATS:
            _check_format(rank, window, allocator, copy_stream, sync, npu_format)
        sync.wait()


class TestUrmaHbmAllocator(CustomTestCase):
    def test_local_copy_and_compute(self):
        self.assertTrue(
            hasattr(shm.ShareMemory, "symmetric_size"),
            "The MemFabric wheel needs ShareMemory.symmetric_size",
        )
        self.assertGreaterEqual(torch.npu.device_count(), 1, "One NPU is required")
        store_url = f"tcp://127.0.0.1:{get_open_port()}"
        mp.spawn(_run_local, args=(store_url,), nprocs=1)

    def test_dual_rank_copy_and_compute(self):
        self.assertTrue(
            hasattr(offload, "UrmaCopyPlan")
            and hasattr(shm.ShareMemory, "symmetric_size"),
            "The MemFabric wheel needs UrmaCopyPlan and ShareMemory.symmetric_size",
        )
        self.assertGreaterEqual(
            torch.npu.device_count(), _WORLD_SIZE, "Two Ascend A5 devices are required"
        )
        store_url = f"tcp://127.0.0.1:{get_open_port()}"
        sync = mp.get_context("spawn").Barrier(_WORLD_SIZE, timeout=120)
        mp.spawn(_run_rank, args=(store_url, sync), nprocs=_WORLD_SIZE)


if __name__ == "__main__":
    unittest.main()
