# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.hardware_backend.npu.moe.weight_sharing.config import (
    PREFETCH_THREAD_JOIN_TIMEOUT_SECS,
    WeightShareConfig,
    get_gloo_init_method,
    get_weight_share_config,
)
from sglang.srt.hardware_backend.npu.moe.weight_sharing.layout import (
    ExpertLayout,
    allocate_experts,
    derive_expert_layout,
    owned_max,
    owned_range,
    reserve_bytes_per_expert,
    shared_tensor_items,
)
from sglang.srt.hardware_backend.npu.moe.weight_sharing.window import (
    WeightShareResources,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A4Int4MoEMethod,
    NPUW4A4MXFP4MoEMethod,
    NPUW4A8Int8MoEMethod,
    NPUW4A8MXFP4MoEMethod,
    NPUW8A8Int8MoEMethod,
    NPUMXFP8MoEMethod,
    NPUUnquantMoEMethod,
)
from sglang.srt.runtime_context import get_parallel, process_model_config
from sglang.srt.utils import (
    check_return_code,
    init_custom_process_group,
    is_npu,
    round_up,
)

if is_npu():
    import acl
    import torch_npu

logger = logging.getLogger(__name__)

_GIB = 1 << 30
_WINDOW_LIMIT = 64 * _GIB
_ALIGNMENT = 512
_ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def _copy_d2d(dst_ptr: int, nbytes: int, src_ptr: int, stream) -> None:
    if nbytes <= 0:
        return

    ret = acl.rt.memcpy_async(
        dst_ptr,
        nbytes,
        src_ptr,
        nbytes,
        _ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream.npu_stream,
    )
    if ret == 0:
        return

    # The synchronous fallback must also wait for the producer or slot event.
    stream.synchronize()
    check_return_code(
        acl.rt.memcpy(dst_ptr, nbytes, src_ptr, nbytes, _ACL_MEMCPY_DEVICE_TO_DEVICE),
        "acl.rt.memcpy D2D",
    )


def _set_layer_tensor(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in layer._parameters:
        layer._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
    elif name in layer._buffers:
        layer._buffers[name] = tensor
    else:
        setattr(layer, name, tensor)


def _deferred_estimate_per_expert(reserve_per_expert: int) -> int:
    return max(1, reserve_per_expert // 2 + reserve_per_expert // 32) + 4096


@dataclass
class _TensorEntry:
    name: str
    bind_pre: bool = False
    reserve_ptr: Optional[int] = None
    reserve_bytes: int = 0
    resident_budget_bytes: int = 0
    slot_estimate_per_expert: int = 0
    resident: Optional[torch.Tensor] = None
    layout: Optional[ExpertLayout] = None
    slot_bases: List[torch.Tensor] = field(default_factory=list)
    slot_views: List[torch.Tensor] = field(default_factory=list)


@dataclass
class SharedLayer:
    layer: Any
    layer_id: int
    num_local_experts: int
    owned_start: int
    owned_count: int
    needs_deferred_resident: bool
    tensors: Dict[str, _TensorEntry] = field(default_factory=dict)
    ready: threading.Event = field(default_factory=threading.Event)
    plans: List[List[Any]] = field(default_factory=list)
    slot_index: int = 0

    def local_to_shard(self, expert_id: int) -> int:
        if 0 <= expert_id - self.owned_start < self.owned_count:
            return expert_id - self.owned_start
        return -1

    def before_forward(self) -> None:
        get_manager().before_layer(self)

    def after_forward(self) -> None:
        get_manager().after_layer(self)


class _Manager:
    def __init__(self) -> None:
        self.config: Optional[WeightShareConfig] = None
        self.layers: Dict[int, SharedLayer] = {}
        self.ordered: List[SharedLayer] = []
        self.resources = None
        self.prepared = False
        self.finalized = False
        self.slot_count = 1
        self._failure: Optional[BaseException] = None
        self._busy = False
        self._condition = threading.Condition()
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._copy_stream = None
        self._index_of: Dict[int, int] = {}
        self._device: Optional[torch.device] = None
        self._slot_done: List[Optional[Any]] = []
        self._slot_used: List[bool] = []
        self.window_bytes = 0

    def register_layer(self, layer: Any) -> Optional[SharedLayer]:
        config = get_weight_share_config()
        if config is None:
            return None
        if getattr(layer, "is_draft", False):
            return None
        if layer.layer_id <= 0:
            return None
        if getattr(layer, "num_fused_shared_experts", 0):
            raise ValueError(
                "MoE weight sharing requires fused shared experts to be disabled"
            )
        if getattr(layer, "is_shared_fused_moe", False):
            raise ValueError("MoE weight sharing does not apply to shared experts")

        num_local = layer.num_local_experts
        if num_local < config.group_size:
            raise ValueError(
                f"MoE weight sharing needs at least {config.group_size} local "
                f"experts per worker, got {num_local}"
            )
        if layer.layer_id in self.layers:
            raise ValueError(
                f"Duplicate MoE layer id {layer.layer_id} under weight sharing"
            )

        start, end = owned_range(config.instance_rank, num_local, config.group_size)
        needs_deferred = (
            type(layer.quant_method).__name__ == "NPUMXFP8OnlineMoEMethod"
        )
        share = SharedLayer(
            layer=layer,
            layer_id=layer.layer_id,
            num_local_experts=num_local,
            owned_start=start,
            owned_count=end - start,
            needs_deferred_resident=needs_deferred,
        )
        self.layers[layer.layer_id] = share
        self.config = config
        return share

    def _collect_tensor_entries(self, share: SharedLayer) -> None:
        items = shared_tensor_items(share.layer, share.owned_count)
        if not items:
            raise RuntimeError(
                f"No shareable expert tensors found on MoE layer {share.layer_id}"
            )
        for name in sorted(items):
            tensor = items[name]
            layout = derive_expert_layout(tensor, torch_npu_format(tensor))
            reserve_per_expert = reserve_bytes_per_expert(
                tensor, is_weight=name.endswith("_weight")
            )
            tensor_entry = _TensorEntry(name=name)
            tensor_entry.slot_estimate_per_expert = reserve_per_expert
            max_owned_expert_count = owned_max(
                share.num_local_experts, self.config.group_size
            )
            if share.needs_deferred_resident:
                tensor_entry.slot_estimate_per_expert = _deferred_estimate_per_expert(
                    reserve_per_expert
                )
                tensor_entry.resident_budget_bytes = (
                    tensor_entry.slot_estimate_per_expert * max_owned_expert_count
                )
            elif name.endswith("_weight"):
                tensor_entry.bind_pre = True
                tensor_entry.layout = layout
                tensor_entry.reserve_bytes = (
                    reserve_per_expert * max_owned_expert_count
                )
                tensor_entry.resident_budget_bytes = tensor_entry.reserve_bytes
            else:
                tensor_entry.resident_budget_bytes = (
                    reserve_per_expert * max_owned_expert_count
                )
            share.tensors[name] = tensor_entry

    def _release_pre_shard_storage(self) -> None:
        for share in self.ordered:
            items = shared_tensor_items(share.layer, share.owned_count)
            for name, tensor_entry in share.tensors.items():
                if not tensor_entry.bind_pre:
                    continue
                source = items[name]
                empty = torch.empty(0, dtype=source.dtype, device=source.device)
                if name in share.layer._parameters:
                    share.layer._parameters[name].data = empty
                elif name in share.layer._buffers:
                    share.layer._buffers[name] = empty
                else:
                    setattr(share.layer, name, empty)

    def prepare(self) -> None:
        config = get_weight_share_config()
        if config is None or self.prepared:
            return
        self.config = config
        self._validate_peer_parallel_config()
        self.ordered = list(self.layers.values())

        empty_stage = None if self.ordered else self._pp_rank()
        stages = (
            get_world_group().all_gather_object(empty_stage)
            if config.pp_size > 1
            else [empty_stage]
        )
        empty_stages = sorted({rank for rank in stages if rank is not None})
        if empty_stages:
            raise ValueError(
                "MoE weight sharing requires at least one shareable MoE layer per "
                f"PP stage. PP stages {empty_stages} have none."
            )

        resident_bytes = 0
        layer_slot_bytes: List[int] = []
        num_local = {share.num_local_experts for share in self.ordered}
        if len(num_local) != 1:
            raise ValueError(
                f"MoE weight sharing requires identical expert counts, got {num_local}"
            )
        logger.info(
            "MoE weight sharing layers %s, instance %d owns experts [%d, %d)",
            [share.layer_id for share in self.ordered],
            config.instance_rank,
            self.ordered[0].owned_start,
            self.ordered[0].owned_start + self.ordered[0].owned_count,
        )

        for share in self.ordered:
            _require_supported_kernel(share.layer)
            self._collect_tensor_entries(share)
            resident_bytes += sum(
                tensor_entry.resident_budget_bytes for tensor_entry in share.tensors.values()
            )
            layer_slot_bytes.append(
                sum(
                    tensor_entry.slot_estimate_per_expert * share.num_local_experts
                    for tensor_entry in share.tensors.values()
                )
            )

        self.slot_count = 1 if len(self.ordered) == 1 else 2
        slot_bytes = self.slot_count * max(layer_slot_bytes)
        base_bytes = resident_bytes + slot_bytes
        margin = max(_GIB // 4, base_bytes // 20)
        required = base_bytes + margin
        override = config.pool_bytes_override
        if override is not None and override < base_bytes:
            raise ValueError(
                f"--moe-weight-share-bytes {override} is below the required "
                f"{base_bytes} bytes"
            )
        local_window = round_up(max(required, override or 0), _GIB)

        window_bytes = self._negotiate_window_bytes(config, local_window)
        if window_bytes > _WINDOW_LIMIT:
            raise ValueError(
                f"MoE weight sharing needs {window_bytes} bytes, above the "
                f"{_WINDOW_LIMIT}-byte MemFabric window limit"
            )
        self._release_pre_shard_storage()
        torch.npu.empty_cache()
        free_bytes, _ = torch.npu.mem_get_info()
        peak_extra = max(layer_slot_bytes) + window_bytes
        if peak_extra > free_bytes:
            raise RuntimeError(
                f"MoE weight sharing needs {window_bytes} bytes plus at most "
                f"{max(layer_slot_bytes)} bytes of "
                f"load temporaries, but only {free_bytes} bytes are free"
            )

        device = torch.device("npu", torch.npu.current_device())
        group_rank = config.group_rank(self._pp_rank(), self._tp_rank())
        self._device = device
        self.window_bytes = window_bytes
        self.resources = WeightShareResources(
            config,
            device=device,
            local_mem_size=window_bytes,
            group_rank=group_rank,
        )
        self._allocate_resident_shards()
        self.prepared = True
        logger.info(
            "MoE weight sharing pool: layers=%d, resident=%d MiB, slots=%d x %d MiB, "
            "window=%d MiB, free=%d MiB",
            len(self.ordered),
            resident_bytes >> 20,
            self.slot_count,
            (max(layer_slot_bytes) >> 20),
            window_bytes >> 20,
            free_bytes >> 20,
        )

    def _negotiate_window_bytes(
        self, config: WeightShareConfig, local_bytes: int
    ) -> int:
        if config.pp_size == 1:
            return local_bytes
        sizes = get_world_group().all_gather_object(local_bytes)
        return max(sizes)

    def _allocate_resident_shards(self) -> None:
        allocator = self.resources.allocator
        for share in self.ordered:
            for name in sorted(share.tensors):
                tensor_entry = share.tensors[name]
                if not tensor_entry.bind_pre:
                    continue
                layout = tensor_entry.layout
                ptr = allocator.reserve(tensor_entry.reserve_bytes, alignment=_ALIGNMENT)
                tensor_entry.reserve_ptr = ptr
                _, shard = _wrap_experts(allocator, ptr, layout, share.owned_count)
                tensor_entry.resident = shard
                if name in share.layer._parameters:
                    share.layer._parameters[name].data = shard
                else:
                    _set_layer_tensor(share.layer, name, shard)

    def after_module_processed(self, layer: torch.nn.Module) -> None:
        share = self.layers.get(getattr(layer, "layer_id", None))
        if (
            share is None
            or share.layer is not layer
            or not self.prepared
            or self.finalized
        ):
            return
        if not share.tensors:
            self._collect_tensor_entries(share)

        items = shared_tensor_items(layer, share.owned_count)
        max_owned_expert_count = owned_max(
            share.num_local_experts, self.config.group_size
        )
        allocator = self.resources.allocator
        for name in list(share.tensors):
            if name not in items:
                del share.tensors[name]
        for name in sorted(items):
            if name not in share.tensors:
                share.tensors[name] = _TensorEntry(name=name)
        # Quantization and the copy back run in order on the loading stream.
        stream = torch.npu.current_stream(self._device)
        for name, tensor_entry in share.tensors.items():
            tensor = items[name]
            npu_format = torch_npu_format(tensor)
            layout = derive_expert_layout(tensor, npu_format)
            required_bytes = layout.nbytes(max_owned_expert_count)
            ptr = tensor_entry.reserve_ptr
            if ptr is None:
                ptr = allocator.reserve(required_bytes, alignment=_ALIGNMENT)
            elif required_bytes > tensor_entry.reserve_bytes:
                raise RuntimeError(
                    f"MoE layer {share.layer_id} tensor {name} needs {required_bytes} "
                    f"bytes after postprocessing, but {tensor_entry.reserve_bytes} "
                    "bytes were reserved before loading"
                )
            _, resident = _wrap_experts(allocator, ptr, layout, share.owned_count)
            if resident.data_ptr() != tensor.data_ptr():
                _copy_d2d(
                    resident.data_ptr(),
                    layout.nbytes(share.owned_count),
                    tensor.data_ptr(),
                    stream,
                )
                # The next assignment may drop the layer's reference to this storage.
                tensor.record_stream(stream)
            tensor_entry.layout = layout
            tensor_entry.resident = resident
            _set_layer_tensor(layer, name, resident)

    def finalize(self) -> None:
        if not self.prepared or self.finalized:
            return
        allocator = self.resources.allocator
        first = self.ordered[0]
        names = sorted(first.tensors)
        for share in self.ordered:
            if sorted(share.tensors) != names:
                raise ValueError(
                    f"MoE layer {share.layer_id} has a different shared tensor set"
                )
            for name in names:
                layout = share.tensors[name].layout
                expected = first.tensors[name].layout
                if (
                    layout.per_expert_bytes != expected.per_expert_bytes
                    or layout.base_shape != expected.base_shape
                    or layout.base_stride != expected.base_stride
                    or layout.dtype != expected.dtype
                    or layout.npu_format != expected.npu_format
                ):
                    raise ValueError(
                        f"MoE layer {share.layer_id} tensor {name} does not match "
                        "the other shared layers"
                    )
                while len(share.tensors[name].slot_bases) < self.slot_count:
                    share.tensors[name].slot_bases.append(None)
                    share.tensors[name].slot_views.append(None)

        for slot in range(self.slot_count):
            for name in names:
                layout = first.tensors[name].layout
                base, view = allocate_experts(
                    allocator, layout, first.num_local_experts
                )
                for share in self.ordered:
                    tensor_entry = share.tensors[name]
                    tensor_entry.slot_bases[slot] = base
                    tensor_entry.slot_views[slot] = view

        for index, share in enumerate(self.ordered):
            share.slot_index = index % self.slot_count
            share.plans = [
                self._build_plans(share, slot) for slot in range(self.slot_count)
            ]
            self._index_of[share.layer_id] = index
        self._slot_done = [None] * self.slot_count
        self._slot_used = [False] * self.slot_count

        total = allocator.reserved_end_address - allocator.base_address
        logger.info(
            "MoE weight sharing allocated %d MiB of %d MiB, %d MiB left",
            total >> 20,
            self.window_bytes >> 20,
            (self.window_bytes - total) >> 20,
        )
        torch.npu.synchronize()
        check_return_code(
            self.resources.shm_window.barrier(), "shm.barrier during weight publication"
        )
        self._start_prefetch_thread()
        self.finalized = True

    def _validate_peer_parallel_config(self) -> None:
        world = get_world_group()
        result = None
        if world.rank_in_group == 0:
            try:
                result = (self._gather_peer_parallel_configs(), None)
            except Exception as exc:
                result = (None, str(exc))
        peer_configs, error = world.broadcast_object(result)
        if error is not None:
            raise RuntimeError(
                f"MoE weight sharing could not exchange parallel configurations: {error}"
            )
        if any(peer != peer_configs[0] for peer in peer_configs[1:]):
            raise ValueError(
                "MoE weight sharing requires identical parallel configurations "
                f"across Prefill instances, got {peer_configs}"
            )

    def _gather_peer_parallel_configs(self) -> list:
        config = self.config
        parallel = get_parallel()
        pp_layer_ranges = ()
        if config.pp_size > 1:
            num_hidden_layers = process_model_config().num_hidden_layers
            pp_layer_ranges = tuple(
                get_pp_indices(num_hidden_layers, pp_rank, config.pp_size)
                for pp_rank in range(config.pp_size)
            )
        parallel_config = dict(
            tp_size=config.tp_size,
            pp_size=config.pp_size,
            pp_layer_ranges=pp_layer_ranges,
            moe_ep_size=parallel.moe_ep_size,
            moe_dp_size=parallel.moe_dp_size,
            attn_dp_size=parallel.attn_dp_size,
            attn_cp_size=parallel.attn_cp_size,
            attn_dcp_size=parallel.attn_dcp_size,
        )
        group = init_custom_process_group(
            backend="gloo",
            init_method=get_gloo_init_method(config.store_url),
            timeout=timedelta(seconds=parallel.dist_timeout or 120),
            world_size=config.group_size,
            rank=config.instance_rank,
            group_name="moe_weight_share",
        )
        peer_configs = [None] * config.group_size
        try:
            torch.distributed.all_gather_object(
                peer_configs, parallel_config, group=group
            )
        finally:
            torch.distributed.destroy_process_group(group)
        return peer_configs

    def _build_plans(self, share: SharedLayer, slot: int) -> List[Any]:
        config = self.config
        tp_rank = self._tp_rank()
        pp_rank = self._pp_rank()
        device = self._device
        plans = []
        for peer in range(config.group_size):
            if peer == config.instance_rank:
                continue
            peer_rank = config.group_rank_for(peer, pp_rank, tp_rank)
            start, end = owned_range(peer, share.num_local_experts, config.group_size)
            entries = []
            for name in sorted(share.tensors):
                tensor_entry = share.tensors[name]
                per_expert_bytes = tensor_entry.layout.per_expert_bytes
                nbytes = (end - start) * per_expert_bytes
                if nbytes == 0:
                    continue
                destination = self.resources.allocator.byte_view(
                    tensor_entry.slot_bases[slot], start * per_expert_bytes, nbytes
                )
                source = self.resources.peer_address(
                    peer_rank, tensor_entry.resident.data_ptr()
                )
                entries.append((source, destination, nbytes))
            if entries:
                plans.append(
                    self.resources.create_copy_plan(entries, device=device)
                )
        return plans

    def _start_prefetch_thread(self) -> None:
        self._queue = queue.Queue()
        self._copy_stream = torch.npu.Stream(device=self._device)
        self._thread = threading.Thread(
            target=self._prefetch_loop, name="moe-weight-share-prefetch", daemon=True
        )
        self._thread.start()

    def _prefetch_loop(self) -> None:
        torch.npu.set_device(self._device)
        while True:
            layer_id = self._queue.get()
            if layer_id is None:
                return
            share = self.layers[layer_id]
            try:
                self._prefetch_layer(share)
            except BaseException as exc:
                self._failure = exc
            finally:
                with self._condition:
                    share.ready.set()
                    self._busy = False
                    self._condition.notify_all()

    def _prefetch_layer(self, share: SharedLayer) -> None:
        slot = share.slot_index
        with torch.npu.stream(self._copy_stream):
            if self._slot_used[slot]:
                self._copy_stream.wait_event(self._slot_done[slot])
            for name in sorted(share.tensors):
                tensor_entry = share.tensors[name]
                count = share.owned_count
                if count == 0:
                    continue
                destination = tensor_entry.slot_views[slot].narrow(
                    0, share.owned_start, count
                )
                _copy_d2d(
                    destination.data_ptr(),
                    count * tensor_entry.layout.per_expert_bytes,
                    tensor_entry.resident.data_ptr(),
                    self._copy_stream,
                )
            for plan in share.plans[slot]:
                plan.enqueue()
        self._copy_stream.synchronize()

    def on_forward_start(self) -> None:
        if not self.finalized:
            return
        with self._condition:
            while self._busy:
                self._condition.wait()
            if self._failure is not None:
                raise RuntimeError("MoE weight prefetch failed") from self._failure
            for share in self.ordered:
                share.ready.clear()
        self._submit(self.ordered[0])

    def _submit(self, share: SharedLayer) -> None:
        with self._condition:
            if self._busy:
                raise RuntimeError("MoE weight prefetch already in flight")
            share.ready.clear()
            self._busy = True
        self._queue.put(share.layer_id)

    def before_layer(self, share: SharedLayer) -> None:
        if not self.finalized:
            raise RuntimeError("MoE weight sharing was not finalized before forward")
        share.ready.wait()
        if self._failure is not None:
            raise RuntimeError("MoE weight prefetch failed") from self._failure
        share.ready.clear()
        slot = share.slot_index
        for name, tensor_entry in share.tensors.items():
            _set_layer_tensor(share.layer, name, tensor_entry.slot_views[slot])

    def after_layer(self, share: SharedLayer) -> None:
        if not self.finalized:
            return
        slot = share.slot_index
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        self._slot_done[slot] = event
        self._slot_used[slot] = True
        index = self._index_of[share.layer_id]
        if index + 1 < len(self.ordered):
            self._submit(self.ordered[index + 1])

    def cleanup(self) -> None:
        self.finalized = False
        thread = self._thread
        if thread is not None:
            self._queue.put(None)
            thread.join(timeout=PREFETCH_THREAD_JOIN_TIMEOUT_SECS)
            if thread.is_alive():
                raise TimeoutError(
                    "MoE weight prefetch thread did not stop within "
                    f"{PREFETCH_THREAD_JOIN_TIMEOUT_SECS} seconds"
                )
            self._thread = None
        if self.resources is not None:
            for share in self.ordered:
                for plans in share.plans:
                    for plan in plans:
                        try:
                            plan.close()
                        except Exception as exc:
                            logger.warning(
                                "Failed to close URMA copy plan: %s", exc
                            )
            self.resources.close()
            self.resources = None
        self.layers.clear()
        self.ordered.clear()
        self.prepared = False
        self._failure = None
        self._busy = False

    def _tp_rank(self) -> int:
        return get_parallel().tp_rank

    def _pp_rank(self) -> int:
        return get_parallel().pp_rank


def _require_supported_kernel(layer) -> None:
    kernel = getattr(layer, "w13_kernel", None)
    if kernel is None:
        if type(layer.quant_method).__name__ in (
            "NPUMXFP8OnlineMoEMethod",
            "UnquantizedFusedMoEMethod",
        ):
            return
        raise ValueError(
            f"MoE weight sharing is not supported by "
            f"{type(layer.quant_method).__name__} on layer {layer.layer_id}"
        )

    supported = (
        NPUW4A8Int8MoEMethod,
        NPUW8A8Int8MoEMethod,
        NPUMXFP8MoEMethod,
        NPUW4A8MXFP4MoEMethod,
        NPUW4A4MXFP4MoEMethod,
        NPUW4A4Int4MoEMethod,
        NPUUnquantMoEMethod,
    )
    if not isinstance(kernel, supported):
        raise ValueError(
            f"MoE weight sharing does not support kernel "
            f"{type(kernel).__name__} on layer {layer.layer_id}"
        )


def _wrap_experts(allocator, ptr: int, layout: ExpertLayout, num_experts: int):
    element_bytes = layout.dtype.itemsize
    if layout.per_expert_bytes % element_bytes != 0:
        raise ValueError("Expert byte stride is not element aligned")
    base_shape = (num_experts,) + layout.base_shape[1:]
    base_stride = (
        layout.per_expert_bytes // element_bytes,
    ) + layout.base_stride[1:]
    base = allocator.wrap(
        ptr,
        layout.nbytes(num_experts),
        layout.dtype,
        base_shape,
        base_stride,
        layout.npu_format,
    )
    if layout.inverse_perm == tuple(range(len(layout.inverse_perm))):
        return base, base
    return base, base.permute(*layout.inverse_perm)


_manager: Optional[_Manager] = None


def get_manager() -> _Manager:
    global _manager
    if _manager is None:
        _manager = _Manager()
    return _manager


def torch_npu_format(tensor: torch.Tensor) -> int:
    return int(torch_npu.get_npu_format(tensor))


def maybe_register_layer(layer) -> Optional[SharedLayer]:
    if get_weight_share_config() is None:
        return None
    return get_manager().register_layer(layer)


def prepare_weight_sharing() -> None:
    if get_weight_share_config() is None:
        return
    get_manager().prepare()


def after_module_quantized(module: torch.nn.Module) -> None:
    if get_weight_share_config() is None:
        return
    get_manager().after_module_processed(module)


def finalize_weight_sharing() -> None:
    if get_weight_share_config() is None:
        return
    get_manager().finalize()


def on_model_forward_start() -> None:
    if get_weight_share_config() is None:
        return
    get_manager().on_forward_start()


def cleanup_weight_sharing() -> None:
    global _manager
    if _manager is None:
        return
    _manager.cleanup()
    _manager = None
