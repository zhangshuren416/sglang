# SPDX-License-Identifier: Apache-2.0
"""Expert weight prefetch over MemFabric SDMA, independent of MoE execution."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager

import torch
import torch.distributed as dist

from sglang.srt.hardware_backend.npu.moe.weight_offload_config import (
    WeightPoolConfig,
    physical_layout,
    pool_capacity,
    slot_for_layer,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_resources, get_stream
from sglang.srt.utils.offloader import BaseOffloader

logger = logging.getLogger(__name__)

WEIGHT_OFFLOAD_MODEL_DECODERS = {
    "qwen3_moe": "Qwen3MoeDecoderLayer",
    "deepseek_v3": "DeepseekV2DecoderLayer",
    "deepseek_v32": "DeepseekV2DecoderLayer",
    "glm_moe_dsa": "DeepseekV2DecoderLayer",
    "deepseek_v4": "DeepseekV4DecoderLayer",
}


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


class MemFabricMoEOffloader(BaseOffloader):
    """Park native loader targets on CPU; export one processed MoE layer at a time.

    Only routed-expert projections are parked, exported and prefetched; shared
    experts stay resident on the device. The native checkpoint loader still owns
    all name mapping and TP/EP slicing.
    """

    def __init__(self, execution_adapter=None):
        # DFC's optional adapter owns only its admission and compute runtime.
        # Native DeepEP/V4 never constructs it or changes its activation/layout.
        self.execution_adapter = execution_adapter
        self.layers = []
        self.device = None
        self.closed = False
        self.geometry = None
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
        self.pool_base = 0
        self.pool_allocation = 0
        self.rank_payload_bytes = 0
        self.layer_payload_bytes = 0
        self.tensor_offsets = {}
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
        self.hooks = []
        self.moe_context = None

    def _register_layer(self, decoder):
        model_type = getattr(decoder.config, "model_type", None)
        if type(decoder).__name__ != WEIGHT_OFFLOAD_MODEL_DECODERS.get(model_type):
            raise ValueError(
                f"MemFabric weight offload has no adapter for {model_type!r}/"
                f"{type(decoder).__name__}"
            )
        experts = getattr(decoder.mlp, "experts", None)
        if experts is None:
            if not getattr(decoder, "is_layer_sparse", True):
                return None
            raise ValueError("Weight offload decoder is missing routed experts")
        if any(
            type(getattr(experts, f"{prefix}_scheme", None)).__name__
            != "ModelSlimW8A8Int8MoE"
            for prefix in ("w13", "w2")
        ):
            raise ValueError("Weight offload requires ModelSlim W8A8 routed experts")
        if experts.num_fused_shared_experts:
            raise ValueError("Weight offload requires disabled shared expert fusion")
        w13, w2 = experts.w13_weight, experts.w2_weight
        if (
            w13.dtype != torch.int8
            or w2.dtype != torch.int8
            or w13.ndim != 3
            or w2.ndim != 3
            or experts.moe_tp_size != 1
        ):
            raise ValueError(
                "Weight offload requires INT8 rank-3 expert shards and MoE TP1"
            )
        if (
            w13.device.type != "npu"
            or w2.device != w13.device
            or (self.device is not None and w13.device != self.device)
        ):
            raise ValueError("Offloaded expert weights must share one NPU device")
        # The ring needs uniform storage, not uniform routing or activation.
        # V4 hash routing, clipping and SiTU stay in the native DeepEP path.
        geometry = (tuple(w13.shape), tuple(w2.shape))
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError(
                "Weight offload requires uniform per-layer expert geometry"
            )
        if self.layers and decoder.layer_id <= self.layers[-1].layer_id:
            raise ValueError(
                "Offloaded MoE layers must be registered in increasing order"
            )
        index = len(self.layers)
        if self.execution_adapter is not None:
            if self.execution_adapter._register_layer(decoder) != index:
                raise ValueError(
                    "Execution adapter and weight offloader layer indices differ"
                )
        self.geometry = geometry
        self.device = w13.device
        self.layers.append(decoder)
        return index

    def _publish(self):
        if not self.layers:
            raise ValueError("Weight offload model has no local MoE layers")
        get_resources().buffers["moe_weight_offloader"] = self

    def wrap_modules(self, all_modules_generator, **kwargs):
        modules = []
        for decoder in all_modules_generator:
            index = self._register_layer(decoder)
            modules.append(decoder)
            if index is None or not self.enabled:
                continue
            moe = decoder.mlp
            self._install_hooks(decoder)
            decoder._derived_weight_cache_error = (
                "Online weight updates are unsupported with immutable "
                "MemFabric MoE weight bundles"
            )
            if index == 0:
                continue
            for relative, owner in moe.experts.named_modules():
                if getattr(owner, "quant_method", None) is None:
                    continue
                path = ".".join(part for part in ("experts", relative) if part)
                self.components[id(owner)] = (index, path)
                self.expected.add((index, path))
                # Preserve Parameter subclasses and native loader attributes.
                for name, value in _module_tensors(owner).items():
                    _bind(owner, name, value.detach().to("cpu"))
        self._publish()
        return modules

    def _install_hooks(self, decoder):
        self.hooks.extend(
            (
                decoder.register_forward_pre_hook(self._before_decoder),
                decoder.register_forward_hook(self._after_decoder, always_call=True),
                decoder.mlp.register_forward_pre_hook(self._before_moe),
                decoder.mlp.register_forward_hook(self._after_moe, always_call=True),
            )
        )

    def _before_decoder(self, decoder, args):
        try:
            self.on_layer_enter(decoder.layer_id)
        except BaseException:
            self.failed = True
            raise

    def _after_decoder(self, decoder, args, output):
        if output is None:
            # Attention can fail before the MoE is entered. Stop the ring even
            # though no MoE exit hook ran; PyTorch preserves the original error.
            self.failed = True

    def _before_moe(self, moe, args):
        if self.moe_context is not None:
            self.failed = True
            raise RuntimeError(
                "Weight offload does not support nested/interleaved MoE forwards"
            )
        context = self.moe_scope(moe.layer_id)
        try:
            context.__enter__()
        except BaseException:
            self.failed = True
            raise
        self.moe_context = context

    def _after_moe(self, moe, args, output):
        context, self.moe_context = self.moe_context, None
        if context is None:
            self.failed = True
            return
        if output is None:
            error = RuntimeError("Offloaded MoE forward failed")
            context.__exit__(RuntimeError, error, None)
        else:
            context.__exit__(None, None, None)

    def postprocess_module(self, module, quant_method, target_device):
        component = self.components.get(id(module))
        if component is None:
            return False
        if component in self.processed:
            raise RuntimeError("Offloaded weights must not be postprocessed twice")
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
                raise RuntimeError(f"Duplicate MoE weight bundle tensor: {key}")
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
                        f"Cannot allocate identical MoE weight physical slot for {key}"
                    )
                slot[key] = value
        elif self.layouts.get(key) != layout:
            raise ValueError(
                f"MoE weight offload requires uniform per-layer physical layout: {key}"
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
            "malloc",
            "free",
            "sparse_copy",
            "SdmaCopyPlan",
        ):
            if not hasattr(offload, name):
                raise RuntimeError(f"Installed MemFabric lacks required offload.{name}")
        if not self.layouts:
            raise RuntimeError("MoE weight offload layer contains no execution tensors")
        parallel = get_parallel()
        offset = 0
        for key, layout in sorted(self.layouts.items()):
            self.tensor_offsets[key] = offset
            offset += ((layout[-1] + 15) // 16) * 16
        self.layer_payload_bytes = offset
        self.rank_payload_bytes = (len(self.layers) - 1) * offset
        required = self.rank_payload_bytes * parallel.tp_size
        reserve = pool_capacity(required, self.config.reserve_bytes)
        # Agree before native initialization to reject mismatched jobs/layouts.
        fingerprint = (
            self.config,
            self.rank_payload_bytes,
            reserve,
            sorted(self.layouts.items()),
        )
        peers = [None] * parallel.tp_size
        dist.all_gather_object(peers, fingerprint, group=parallel.tp_group.cpu_group)
        if any(peer != fingerprint for peer in peers):
            raise ValueError(
                "MoE weight pool configuration or execution layout differs across ranks"
            )
        config = offload.OffloadConfig()
        config.device_id = self.device.index
        config.scene = offload.Scene.SHARED
        config.world_size = parallel.tp_size
        config.rank_id = parallel.tp_rank
        config.local_world_size = parallel.tp_size // parallel.nnodes
        # SHARED requires a common virtual slot stride. Only TP rank 0 backs
        # its slot with physical DRAM; all EP shards live in that one pool.
        config.reserve_size = reserve
        config.alloc_size = reserve if parallel.tp_rank == 0 else 0
        config.store_url = self.config.store_url
        if offload.initialize(config) != 0:
            raise RuntimeError("MemFabric SHARED weight pool initialization failed")
        self.offload = offload
        if parallel.tp_rank == 0:
            self.pool_allocation = offload.malloc(required)
        addresses = [self.pool_allocation]
        dist.broadcast_object_list(
            addresses,
            src=parallel.tp_group.ranks[0],
            group=parallel.tp_group.cpu_group,
        )
        self.pool_base = addresses[0]
        if not self.pool_base:
            raise RuntimeError(
                "MemFabric shared weight arena allocation failed on TP rank 0"
            )
        self.stream = get_stream("moe_weight_prefetch")
        for index, key, source in self.pending:
            self._export(index, key, source)
        self.pending.clear()
        logger.info(
            "MoE weight pool %s: owner=TP0 local payload=%d bytes "
            "total payload=%d bytes reserve=%d bytes",
            self.config.store_url,
            self.rank_payload_bytes,
            required,
            reserve,
        )

    def _export(self, index, key, source):
        """Load-time raw NZ/ND bytes; no logical Tensor copy or format conversion."""
        size = self.layouts[key][-1]
        # Imported SHARED GVAs can be used directly by sparse_copy and SDMA.
        # Non-owner ranks must never call malloc/empty on their zero-sized slot.
        destination = (
            self.pool_base
            + get_parallel().tp_rank * self.rank_payload_bytes
            + (index - 1) * self.layer_payload_bytes
            + self.tensor_offsets[key]
        )
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
            [destination + offset for offset in offsets],
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
            raise RuntimeError(
                f"MoE weight export failed: layer={index} tensor={key}"
            )
        # Initialization only: source scratch and descriptors live until this fence.
        torch.npu.current_stream(self.device).synchronize()

    def post_init(self):
        if self.enabled and self.processed != self.expected:
            raise RuntimeError(
                "Offloaded expert components were not loaded: "
                f"{self.expected - self.processed}"
            )
        if self.enabled and len(self.layers) > 1:
            for index in range(1, len(self.layers)):
                keys = {key for layer, key in self.sources if layer == index}
                if keys != set(self.layouts):
                    raise RuntimeError(
                        f"Incomplete MoE weight execution bundle at layer {index}"
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
        if self.execution_adapter is not None:
            self.execution_adapter.post_init()
        self.ready_for_forward = True

    def on_layer_enter(self, layer_id):
        if self.closed or self.failed or not self.ready_for_forward:
            raise RuntimeError(
                "MoE weight offloader is not ready or a previous forward failed"
            )
        index = self.next_index
        if self.current is not None or self.layers[index].layer_id != layer_id:
            self.failed = True
            raise RuntimeError(
                "Offloaded MoE layers must execute once in their declared order"
            )
        compute = torch.npu.current_stream(self.device)
        if self.compute is not None and compute != self.compute:
            if index != 0:
                raise RuntimeError(
                    "Weight offload compute stream changed inside a model forward"
                )
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
            raise RuntimeError(
                "Offloaded MoE entered without its layer prefetch boundary"
            )
        try:
            if index > 0:
                self.compute.wait_event(self.events[index][1])
            if index + 1 in self.events:
                self.compute.wait_event(self.events[index + 1][0])
            yield
            moe = self.layers[index].mlp
            for name in ("alt_stream", "routed_quant_stream"):
                stream = getattr(moe, name, None)
                if stream is not None:
                    self.compute.wait_stream(stream)
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
        if self.device is not None:
            with torch.npu.device(self.device):
                torch.npu.synchronize()
        if self.compute is not None:
            self.compute.synchronize()
        if self.stream is not None:
            self.stream.synchronize()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for plan in self.plans.values():
            plan.close()
        if self.offload is not None:
            group = get_parallel().tp_group.cpu_group
            dist.barrier(group=group)
            if self.pool_allocation:
                self.offload.free(self.pool_allocation)
                self.pool_allocation = 0
            dist.barrier(group=group)
            rank = get_parallel().tp_rank
            if rank != 0:
                self.offload.uninitialize()
            dist.barrier(group=group)
            if rank == 0:
                self.offload.uninitialize()
        self.plans.clear()
        self.sources.clear()
        self.pool_base = 0
        self.pool_allocation = 0
        self.pending.clear()
        self.slots = [{}, {}]
        self.events.clear()
        self.inflight.clear()
        self.last_use = [None, None]
        if self.execution_adapter is not None:
            self.execution_adapter.close()
        if get_resources().buffers.get("moe_weight_offloader") is self:
            get_resources().buffers.pop("moe_weight_offloader")
        self.layers.clear()
        self.closed = True
