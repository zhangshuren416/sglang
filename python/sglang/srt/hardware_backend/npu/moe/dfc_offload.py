# SPDX-License-Identifier: Apache-2.0
"""Eager SGLang layer scheduling over MemFabric's prepared A3 SDMA plans."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager

import torch
import torch.distributed as dist

from sglang.srt.hardware_backend.npu.moe.dfc_config import (
    WeightPoolConfig,
    physical_layout,
    pool_capacity,
    slot_for_layer,
)
from sglang.srt.hardware_backend.npu.moe.dfc_model_adapter import DfcModelAdapter
from sglang.srt.runtime_context import get_exec, get_parallel, get_stream

logger = logging.getLogger(__name__)


def _module_tensors(module):
    """Include quant-method caches that are ordinary attributes, as well as state."""
    result = dict(module._parameters)
    result.update(module._buffers)
    result.update(
        {k: v for k, v in vars(module).items() if isinstance(v, torch.Tensor)}
    )
    return {name: value for name, value in result.items() if value is not None}


def _bind(module, name, tensor):
    if name in module._parameters:
        module._parameters[name].data = tensor
    elif name in module._buffers:
        module._buffers[name] = tensor
    else:
        setattr(module, name, tensor)


class DfcOffloader(DfcModelAdapter):
    """Park native loader targets on CPU; export one processed MoE layer at a time.

    Shared-expert projections participate in the same bundle and consumed event.
    The native checkpoint loader still owns all name mapping and TP/EP slicing.
    """

    def __init__(self):
        super().__init__()
        self.enabled = get_exec().offload.moe_weight_offload_backend == "memfabric"
        self.config = (
            WeightPoolConfig.read(get_exec().offload.moe_weight_offload_config)
            if self.enabled
            else None
        )
        self.components = {}
        self.expected = set()
        self.processed = set()
        self.slots = [{}, {}]
        self.layouts = {}
        self.pending = []
        self.sources = {}
        self.plans = {}
        self.offload = None
        self.stream = None
        self.compute = None
        self.last_use = [None, None]
        self.inflight = deque()
        self.events = {}
        self.current = None
        self.next_index = 0
        self.epoch = 0
        self.failed = False
        self.ready_for_forward = False

    def wrap_modules(self, all_modules_generator, **kwargs):
        modules = []
        for decoder in all_modules_generator:
            index = self._register_layer(decoder)
            modules.append(decoder)
            if not self.enabled:
                continue
            decoder._dfc_weight_offloader = self
            decoder._derived_weight_cache_error = (
                "Online weight updates are unsupported with immutable "
                "MemFabric DFC weight bundles"
            )
            if index == 0:
                continue
            for root_name in ("experts", "shared_experts"):
                root = getattr(moe, root_name, None)
                if root is None:
                    continue
                for relative, owner in root.named_modules():
                    if getattr(owner, "quant_method", None) is None:
                        continue
                    path = ".".join(part for part in (root_name, relative) if part)
                    self.components[id(owner)] = (index, path)
                    self.expected.add((index, path))
                    # Preserve Parameter subclasses and native loader attributes.
                    for name, value in _module_tensors(owner).items():
                        _bind(owner, name, value.detach().to("cpu"))
        self._publish()
        return modules

    def postprocess_module(self, module, quant_method, target_device):
        component = self.components.get(id(module))
        if component is None:
            return False
        if component in self.processed:
            raise RuntimeError("DFC offloaded weights must not be postprocessed twice")
        index, path = component
        for name, value in _module_tensors(module).items():
            _bind(module, name, value.detach().to(target_device))
        quant_method.process_weights_after_loading(module)
        for name, value in _module_tensors(module).items():
            key = f"{path}.{name}"
            self._capture(index, key, module, name, value.detach())
        self.processed.add(component)
        if self.offload is None and all(
            item in self.processed for item in self.expected if item[0] == 1
        ):
            self._initialize_pool()
        return True

    def _capture(self, index, key, owner, name, source):
        import torch_npu

        layout = physical_layout(source, torch_npu.get_npu_format(source))
        if index == 1:
            if key in self.layouts:
                raise RuntimeError(f"Duplicate DFC bundle tensor: {key}")
            self.layouts[key] = layout
            for slot in self.slots:
                value = torch.empty_strided(
                    source.shape,
                    source.stride(),
                    dtype=source.dtype,
                    device=self.device,
                )
                if layout[3] != torch_npu.get_npu_format(value):
                    value = torch_npu.npu_format_cast(value, layout[3])
                if physical_layout(value, torch_npu.get_npu_format(value)) != layout:
                    raise ValueError(
                        f"Cannot allocate identical DFC physical slot for {key}"
                    )
                slot[key] = value
        elif self.layouts.get(key) != layout:
            raise ValueError(
                f"DFC offload requires uniform per-layer physical layout: {key}"
            )
        if self.offload is None:
            self.pending.append((index, key, source))
        else:
            self._export(index, key, source)
        _bind(owner, name, self.slots[slot_for_layer(index)][key])

    def _initialize_pool(self):
        from memfabric_hybrid import offload

        for name in (
            "OffloadConfig",
            "Scene",
            "initialize",
            "uninitialize",
            "empty",
            "free",
            "sparse_copy",
            "SdmaCopyPlan",
        ):
            if not hasattr(offload, name):
                raise RuntimeError(f"Installed MemFabric lacks required offload.{name}")
        if not self.layouts:
            raise RuntimeError("DFC offload layer contains no execution tensors")
        parallel = get_parallel()
        required = (len(self.layers) - 1) * sum(
            ((layout[-1] + 15) // 16) * 16 for layout in self.layouts.values()
        )
        reserve = pool_capacity(required, self.config.reserve_bytes_per_rank)
        # Agree before native initialization to reject mismatched jobs/layouts.
        fingerprint = (self.config, reserve, sorted(self.layouts.items()))
        peers = [None] * parallel.tp_size
        dist.all_gather_object(peers, fingerprint, group=parallel.tp_group.cpu_group)
        if any(peer != fingerprint for peer in peers):
            raise ValueError(
                "DFC pool configuration or execution layout differs across ranks"
            )
        config = offload.OffloadConfig()
        config.device_id = self.device.index
        config.scene = offload.Scene.SHARED
        config.world_size = parallel.tp_size
        config.rank_id = parallel.tp_rank
        config.local_world_size = parallel.tp_size // parallel.nnodes
        config.reserve_size = reserve
        config.alloc_size = reserve
        config.store_url = self.config.store_url
        if offload.initialize(config) != 0:
            raise RuntimeError("MemFabric SHARED weight pool initialization failed")
        self.offload = offload
        self.stream = get_stream("moe_weight_prefetch")
        for index, key, source in self.pending:
            self._export(index, key, source)
        self.pending.clear()
        logger.info(
            "DFC weight pool %s: local payload=%d bytes reserve=%d bytes",
            self.config.pool_id,
            required,
            reserve,
        )

    def _export(self, index, key, source):
        """Load-time raw NZ/ND bytes; no logical Tensor copy or format conversion."""
        size = self.layouts[key][-1]
        destination = self.offload.empty((size,), dtype=torch.uint8)
        self.sources[index, key] = destination
        # sparse_copy lengths are uint8 element counts. Split below signed int32 limits.
        chunks = [min(8 << 20, size - offset) for offset in range(0, size, 8 << 20)]
        offsets = list(range(0, size, 8 << 20))
        source_ptrs = torch.tensor(
            [source.data_ptr() + offset for offset in offsets],
            dtype=torch.int64,
            device=self.device,
        )
        destination_ptrs = torch.tensor(
            [destination.data_ptr() + offset for offset in offsets],
            dtype=torch.int64,
            device=self.device,
        )
        lengths = torch.tensor(chunks, dtype=torch.int32, device=self.device)
        count = torch.tensor(len(chunks), dtype=torch.int32, device=self.device)
        if (
            self.offload.sparse_copy(
                source_ptrs, destination_ptrs, lengths, count, self.device
            )
            != 0
        ):
            self.failed = True
            raise RuntimeError(f"DFC weight export failed: layer={index} tensor={key}")
        # Initialization only: source scratch and descriptors live until this fence.
        torch.npu.current_stream(self.device).synchronize()

    def post_init(self):
        if self.enabled and self.processed != self.expected:
            raise RuntimeError(
                "DFC expert components were not loaded: "
                f"{self.expected - self.processed}"
            )
        if self.enabled and len(self.layers) > 1:
            for index in range(1, len(self.layers)):
                keys = {key for layer, key in self.sources if layer == index}
                if keys != set(self.layouts):
                    raise RuntimeError(
                        f"Incomplete DFC execution bundle at layer {index}"
                    )
                self.plans[index] = self.offload.SdmaCopyPlan(
                    [
                        (
                            self.sources[index, key],
                            self.slots[slot_for_layer(index)][key],
                            layout[-1],
                        )
                        for key, layout in self.layouts.items()
                    ],
                    self.device,
                    self.config.sdma_channels,
                )
        super().post_init()
        self.ready_for_forward = True

    def on_layer_enter(self, layer_id):
        if self.closed or self.failed or not self.ready_for_forward:
            raise RuntimeError(
                "DFC offloader is not ready or a previous forward failed"
            )
        index = self.next_index
        if self.current is not None or self.layers[index].layer_id != layer_id:
            self.failed = True
            raise RuntimeError("DFC layers must execute once in their declared order")
        compute = torch.npu.current_stream(self.device)
        if self.compute is not None and compute != self.compute:
            if index != 0:
                raise RuntimeError("DFC compute stream changed inside a model forward")
            # Warmup and serving can use different streams. Handoff serializes
            # the entire old forward without synchronizing the host.
            compute.wait_stream(self.compute)
        self.compute = compute
        if index == 0:
            while self.inflight and self.inflight[0][0].query():
                self.inflight.popleft()
            self.events = {}
        self.current = index
        if index + 1 < len(self.layers):
            target = index + 1
            slot = slot_for_layer(target)
            submitted, ready = torch.npu.Event(), torch.npu.Event()
            self.events[target] = (submitted, ready)
            with torch.npu.stream(self.stream):
                if self.last_use[slot] is not None:
                    self.stream.wait_event(self.last_use[slot])
                self.plans[target].enqueue(submitted)
                ready.record(self.stream)

    @contextmanager
    def moe_scope(self, layer_id):
        index = self.current
        if index is None or self.layers[index].layer_id != layer_id:
            raise RuntimeError("DFC MoE entered without its layer prefetch boundary")
        try:
            if index > 0:
                self.compute.wait_event(self.events[index][1])
            if index + 1 in self.events:
                self.compute.wait_event(self.events[index + 1][0])
            yield
            alt = self.layers[index].mlp.alt_stream
            if alt is not None:
                self.compute.wait_stream(alt)
            consumed = torch.npu.Event()
            consumed.record(self.compute)
            if index > 0:
                self.last_use[slot_for_layer(index)] = consumed
            self.current = None
            self.next_index = index + 1
            if self.next_index == len(self.layers):
                self.inflight.append((consumed, self.events))
                self.next_index = 0
                self.epoch += 1
        except BaseException:
            self.failed = True
            raise

    def close(self):
        """Collective shutdown before process-group destruction, not a finalizer."""
        if self.closed:
            return
        # Also cover a failed load-time export on the initialization stream.
        with torch.npu.device(self.device):
            torch.npu.synchronize()
        if self.compute is not None:
            self.compute.synchronize()
        if self.stream is not None:
            self.stream.synchronize()
        for plan in self.plans.values():
            plan.close()
        if self.offload is not None:
            group = get_parallel().tp_group.cpu_group
            dist.barrier(group=group)
            for source in self.sources.values():
                self.offload.free(source.data_ptr())
            dist.barrier(group=group)
            rank = get_parallel().tp_rank
            if rank != 0:
                self.offload.uninitialize()
            dist.barrier(group=group)
            if rank == 0:
                self.offload.uninitialize()
        self.plans.clear()
        self.sources.clear()
        self.pending.clear()
        self.slots = [{}, {}]
        self.events.clear()
        self.inflight.clear()
        self.last_use = [None, None]
        super().close()
