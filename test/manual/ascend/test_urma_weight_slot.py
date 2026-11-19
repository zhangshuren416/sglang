# SPDX-License-Identifier: Apache-2.0
"""Device checks for the Ascend MoE weight-sharing slot layout.

The manager that will consume these results is not implemented yet. Each case
covers an assumption that cannot be settled by reading the MemFabric source:

1. One UrmaCopyPlan can be prepared once and enqueued for several rounds with
   fresh peer data.
2. Local D2D fill and remote URMA fill can target disjoint expert ranges of one
   full-layer slot tensor.
3. The MXFP8 layout (FRACTAL_NZ base with a transposed [E, K, N] view) can be
   allocated at full expert count and filled by expert byte offsets.

Every rank creates a 1 GiB AIV_URMA window and a 1 GiB local DRAM offload pool.
"""

import unittest

import acl
import memfabric_hybrid as mf
import torch
import torch.multiprocessing as mp
import torch_npu
from memfabric_hybrid import offload, shm

from sglang.srt.hardware_backend.npu.moe.urma_hbm_allocator import UrmaHbmAllocator
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat
from sglang.srt.utils import round_up
from sglang.srt.utils.network import get_open_port
from sglang.test.test_utils import CustomTestCase

_WORLD_SIZE = 2
_WINDOW_BYTES = 1 << 30
_OFFLOAD_DRAM_BYTES = 1 << 30
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_ACL_MEMCPY_DEVICE_TO_DEVICE = 3
_EXPERTS = 4
_MATRIX_ROWS = 128
_MATRIX_COLS = 64


def _check_ret(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} failed with return code {ret}")


def _open_runtime(rank, store_url):
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True

    _check_ret(mf.initialize(), "memfabric_hybrid.initialize")

    shm_config = shm.ShmConfig()
    shm_config.start_store = rank == 0
    _check_ret(
        shm.initialize(store_url, _WORLD_SIZE, rank, device.index, shm_config),
        "shm.initialize",
    )
    window = shm.create(
        id=0,
        rank_size=_WORLD_SIZE,
        rank_id=rank,
        local_mem_size=_WINDOW_BYTES,
        data_op_type=shm.ShmDataOpType.AIV_URMA,
    )
    _check_ret(window.barrier(), "shm.barrier")

    config = offload.OffloadConfig()
    config.device_id = device.index
    config.scene = offload.Scene.LOCAL
    config.reserve_size = _OFFLOAD_DRAM_BYTES
    config.alloc_size = _OFFLOAD_DRAM_BYTES
    _check_ret(offload.initialize(config), "offload.initialize")
    return window, device


def _close_runtime(window, device):
    torch.npu.synchronize(device)
    window.barrier()
    offload.uninitialize()
    window.destroy()
    shm.uninitialize(0)
    mf.uninitialize()


def _base_pattern(dtype=torch.float16):
    pattern = torch.arange(_MATRIX_ROWS * _MATRIX_COLS, dtype=torch.int32)
    return pattern.remainder(31).reshape(_MATRIX_ROWS, _MATRIX_COLS).to(dtype)


def _hbm_bytes(tensor):
    nbytes = tensor.untyped_storage().nbytes()
    host = torch.empty(nbytes, dtype=torch.uint8)
    _check_ret(
        acl.rt.memcpy(
            host.data_ptr(),
            nbytes,
            tensor.data_ptr(),
            nbytes,
            _ACL_MEMCPY_DEVICE_TO_HOST,
        ),
        "acl.rt.memcpy D2H",
    )
    return host


def _copy_d2d(dst_ptr, nbytes, src_ptr, stream):
    if hasattr(acl.rt, "memcpy_async"):
        try:
            ret = acl.rt.memcpy_async(
                dst_ptr,
                nbytes,
                src_ptr,
                nbytes,
                _ACL_MEMCPY_DEVICE_TO_DEVICE,
                stream.npu_stream,
            )
            if ret == 0:
                return True
        except TypeError:
            pass
    _check_ret(
        acl.rt.memcpy(
            dst_ptr, nbytes, src_ptr, nbytes, _ACL_MEMCPY_DEVICE_TO_DEVICE
        ),
        "acl.rt.memcpy D2D",
    )
    return False


def _alloc(allocator, shape, stride, dtype, npu_format, nbytes, alignment=512):
    ptr = round_up(allocator._next_ptr, alignment)
    if ptr + nbytes > allocator._limit:
        raise RuntimeError(
            f"slot test needs {nbytes} bytes at {ptr}, "
            f"window ends at {allocator._limit}"
        )
    tensor = allocator._wrap(ptr, nbytes, dtype, shape, stride, npu_format)
    if tensor.untyped_storage().nbytes() != nbytes:
        raise RuntimeError("allocated slot storage size does not match the request")
    allocator._next_ptr = ptr + nbytes
    return tensor


def _derive_layout(template):
    """Split a packed [E, ...] tensor into (per-expert bytes, base layout)."""
    shape = tuple(template.shape)
    stride = template.stride()
    storage_bytes = template.untyped_storage().nbytes()
    if len(shape) < 2 or storage_bytes % shape[0] != 0:
        raise ValueError(
            f"unsupported template shape={shape}, storage={storage_bytes}"
        )
    per_expert_bytes = storage_bytes // shape[0]

    order = sorted(range(len(shape)), key=lambda d: stride[d], reverse=True)
    if order[0] != 0:
        raise ValueError(f"expert dim is not the outer stride: shape={shape}")
    base_shape = tuple(shape[d] for d in order)
    base_stride = [1] * len(base_shape)
    for i in reversed(range(len(base_shape) - 1)):
        base_stride[i] = base_stride[i + 1] * base_shape[i + 1]
    if any(stride[d] != base_stride[i] for i, d in enumerate(order)):
        raise ValueError(
            f"template is not a permuted contiguous tensor: "
            f"shape={shape}, stride={stride}"
        )

    inverse = [0] * len(shape)
    for i, d in enumerate(order):
        inverse[d] = i
    return per_expert_bytes, base_shape, tuple(base_stride), inverse


def _alloc_experts(allocator, template, num_experts):
    """Allocate a packed [num_experts, ...] tensor with the template's layout."""
    per_expert_bytes, base_shape, base_stride, inverse = _derive_layout(template)
    element_bytes = template.element_size()
    if per_expert_bytes % element_bytes != 0:
        raise ValueError("per-expert bytes are not element aligned")

    shape = (num_experts,) + base_shape[1:]
    stride = (per_expert_bytes // element_bytes,) + base_stride[1:]
    nbytes = per_expert_bytes * num_experts
    base = _alloc(
        allocator,
        shape,
        stride,
        template.dtype,
        torch_npu.get_npu_format(template),
        nbytes,
    )
    view = base.permute(*inverse) if inverse != list(range(len(inverse))) else base
    return base, view, per_expert_bytes


def _owned_range(rank, num_experts, world_size):
    base, remainder = divmod(num_experts, world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _peer_gva(window, rank, tensor, peer):
    local_base = window.gva + rank * window.symmetric_size
    peer_base = window.gva + peer * window.symmetric_size
    return peer_base + (tensor.data_ptr() - local_base)


def _run_plan_reuse(rank, store_url, sync):
    window, device = _open_runtime(rank, store_url)
    try:
        allocator = UrmaHbmAllocator(
            window, device=device, local_mem_size=_WINDOW_BYTES
        )
        copy_stream = torch.npu.Stream(device=device)
        template = (_base_pattern() + rank * 32).npu()
        source = allocator.empty_like(template)
        destination = allocator.empty_like(template)
        nbytes = template.untyped_storage().nbytes()

        peer = 1 - rank
        plan = offload.UrmaCopyPlan(
            [(_peer_gva(window, rank, source, peer), destination, nbytes)],
            device=device,
        )
        try:
            for round_id in range(1, 3):
                pattern = (
                    _base_pattern() + rank * 32 + round_id * 2048
                ).npu()
                _check_ret(
                    acl.rt.memcpy(
                        source.data_ptr(),
                        nbytes,
                        pattern.data_ptr(),
                        nbytes,
                        _ACL_MEMCPY_DEVICE_TO_DEVICE,
                    ),
                    "acl.rt.memcpy source fill",
                )
                torch.npu.synchronize(device)
                sync.wait()
                with torch.npu.stream(copy_stream):
                    plan.enqueue()
                sync.wait()
                got = _hbm_bytes(destination)
                host_expected = _base_pattern() + peer * 32 + round_id * 2048
                expected_bytes = host_expected.view(torch.uint8).flatten()
                if not torch.equal(got, expected_bytes):
                    raise AssertionError(
                        f"[rank {rank}] plan reuse round {round_id} mismatch"
                    )
            print(f"[rank {rank}] UrmaCopyPlan reuse passed", flush=True)
        finally:
            torch.npu.synchronize(device)
            sync.wait()
            plan.close()
    finally:
        _close_runtime(window, device)


def _run_merge(rank, store_url, sync, npu_format_value, transposed):
    npu_format = NPUACLFormat(npu_format_value)
    window, device = _open_runtime(rank, store_url)
    try:
        allocator = UrmaHbmAllocator(
            window, device=device, local_mem_size=_WINDOW_BYTES
        )
        copy_stream = torch.npu.Stream(device=device)
        peer = 1 - rank
        start, end = _owned_range(rank, _EXPERTS, _WORLD_SIZE)
        peer_start, peer_end = _owned_range(peer, _EXPERTS, _WORLD_SIZE)

        experts = torch.stack(
            [_base_pattern() + e * 1000 for e in range(_EXPERTS)]
        ).to(device)
        if npu_format == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ:
            full_base = torch_npu.npu_format_cast(experts, int(npu_format))
        else:
            full_base = experts
        full_view = full_base.transpose(1, 2) if transposed else full_base
        if npu_format == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ:
            if torch_npu.get_npu_format(full_view) != npu_format:
                raise AssertionError("transposed view lost the FRACTAL_NZ format")
        full_bytes = _hbm_bytes(full_base)

        shard_base, _, per_expert_bytes = _alloc_experts(
            allocator, full_view, end - start
        )
        _check_ret(
            acl.rt.memcpy(
                shard_base.data_ptr(),
                (end - start) * per_expert_bytes,
                full_base.data_ptr() + start * per_expert_bytes,
                (end - start) * per_expert_bytes,
                _ACL_MEMCPY_DEVICE_TO_DEVICE,
            ),
            "acl.rt.memcpy shard fill",
        )

        slot_base, _, slot_expert_bytes = _alloc_experts(
            allocator, full_view, _EXPERTS
        )
        if slot_expert_bytes != per_expert_bytes:
            raise AssertionError("slot and shard disagree on per-expert bytes")

        destination = allocator.byte_view(
            slot_base,
            peer_start * per_expert_bytes,
            (peer_end - peer_start) * per_expert_bytes,
        )
        plan = offload.UrmaCopyPlan(
            [(_peer_gva(window, rank, shard_base, peer), destination, destination.numel())],
            device=device,
        )
        try:
            torch.npu.synchronize(device)
            sync.wait()
            async_local = None
            with torch.npu.stream(copy_stream):
                async_local = _copy_d2d(
                    slot_base.data_ptr() + start * per_expert_bytes,
                    (end - start) * per_expert_bytes,
                    shard_base.data_ptr(),
                    copy_stream,
                )
                plan.enqueue()
            torch.npu.synchronize(device)

            got = _hbm_bytes(slot_base)
            if not torch.equal(got, full_bytes):
                raise AssertionError(
                    f"[rank {rank}] {npu_format.name} transposed={transposed} "
                    "slot bytes differ from the full reference"
                )
            print(
                f"[rank {rank}] {npu_format.name} transposed={transposed} "
                f"slot merge passed (async_local={async_local})",
                flush=True,
            )
        finally:
            torch.npu.synchronize(device)
            sync.wait()
            plan.close()
    finally:
        _close_runtime(window, device)


class TestUrmaWeightSlot(CustomTestCase):
    def _spawn(self, target, *args):
        self.assertGreaterEqual(
            torch.npu.device_count(),
            _WORLD_SIZE,
            "Two Ascend A5 devices are required",
        )
        self.assertTrue(
            hasattr(offload, "UrmaCopyPlan") and hasattr(shm.ShareMemory, "symmetric_size"),
            "The MemFabric wheel needs UrmaCopyPlan and ShareMemory.symmetric_size",
        )
        store_url = f"tcp://127.0.0.1:{get_open_port()}"
        sync = mp.get_context("spawn").Barrier(_WORLD_SIZE, timeout=180)
        mp.spawn(target, args=(store_url, sync, *args), nprocs=_WORLD_SIZE)

    def test_plan_reuse(self):
        self._spawn(_run_plan_reuse)

    def test_slot_merge_nd(self):
        self._spawn(_run_merge, int(NPUACLFormat.ACL_FORMAT_ND), False)

    def test_slot_merge_nz(self):
        self._spawn(
            _run_merge, int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ), False
        )

    def test_slot_merge_transposed_nz(self):
        self._spawn(
            _run_merge, int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ), True
        )


if __name__ == "__main__":
    unittest.main()
