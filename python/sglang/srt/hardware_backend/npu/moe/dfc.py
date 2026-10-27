# SPDX-License-Identifier: Apache-2.0
"""ZBAL DFC execution; expert routing and shared experts remain in SGLang."""

from __future__ import annotations

import torch

from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
)


def workspace_budget_bytes(rows, hidden, intermediate, experts, ep, topk, capacity):
    """Upper bound for the native DFC workspace in MemFabric 00b7195d.

    Match ComputeDispatchFFNCombineWorkspaceSize, with conservative A3 core
    bounds (256 AIC / 512 AIV). Keep this reservation in sync when upgrading
    that binding: its API allocates scratch internally and has no size query.
    """

    def aligned(size, alignment=512):
        return ((size + alignment - 1) // alignment) * alignment

    sections = (
        aligned(rows, 256) * topk * 4,  # expanded row indices
        ep * experts * 4,  # expert counts prefix sum
        ep * aligned(experts, 128) * 4,  # token counts
        capacity * 4,  # first per-token scale
        capacity * 4,  # second per-token scale
        capacity * (2 * intermediate) * 2,  # GMM1 output
        capacity * hidden * 2,  # GMM2 output
        capacity * hidden,  # quantized input
        capacity * aligned(intermediate),  # permuted tokens
        ep * 4 * 16,  # rank prefix sum
        256 * 4 * 16,  # AIC progress flags
        capacity * (hidden + 512),  # routing output
        capacity * 4,  # routing per-token scales
        capacity * hidden * 2,  # combine output
    )
    init_routing = rows * topk * 32 + 512 * 2 * 4 + (16 << 20)
    return aligned((16 << 20) + sum(aligned(s) for s in sections) + init_routing)


class DfcRuntime:
    def __init__(self, device: torch.device, layer):
        import torch.distributed as dist
        from zbal.zbal.deepep_adaptor import Buffer

        parallel = get_parallel()
        if not hasattr(Buffer, "dispatch_ffn_combine"):
            raise RuntimeError(
                "Installed ZBAL was built without A3 dispatch_ffn_combine"
            )
        schedule = get_schedule()
        self.max_source_tokens = max(
            schedule.max_prefill_tokens,
            schedule.chunked_prefill_size or 0,
        )
        ranks = dist.get_process_group_ranks(parallel.moe_ep_group.device_group)
        if (
            not 1 <= len(ranks) <= 16
            or not 1 <= layer.num_experts <= 4096
            or layer.num_experts % len(ranks)
        ):
            raise ValueError(
                "Native A3 DFC requires EP1..16 and divisible experts <=4096"
            )
        required = self.max_source_tokens * layer.top_k * len(ranks)
        configured = get_exec().moe.dfc_max_output_tokens
        if configured is not None and configured < required:
            raise ValueError(
                f"DFC requires capacity >= {required} expanded rows for the configured "
                "prefill limits; reduce the prefill limits or increase "
                "--dfc-max-output-tokens"
            )
        self.capacity = configured or required
        if self.capacity >= 1 << 32:
            raise ValueError("DFC expanded capacity exceeds the native uint32 limit")
        self.num_experts = layer.num_experts
        self.workspace_budget_bytes = workspace_budget_bytes(
            self.max_source_tokens,
            layer.w13_weight.shape[1],
            layer.w2_weight.shape[1],
            self.num_experts,
            len(ranks),
            layer.top_k,
            self.capacity,
        )
        self.device = device
        self.last_stream = None
        self.group = dist.new_group(ranks=ranks, backend="zbal")
        try:
            backend = self.group._get_backend(device)
            backend.init_zbal_comm_meta()
            self.buffer = Buffer(
                self.group.rank(),
                self.group.size(),
                0,
                0,
                False,
                backend.get_zbal_comm_name(),
            )
        except BaseException:
            dist.destroy_process_group(self.group)
            raise

    def forward(self, layer, hidden_states, topk_output):
        stream = torch.npu.current_stream(self.device)
        if self.last_stream is not None and self.last_stream != stream:
            stream.wait_stream(self.last_stream)
        self.last_stream = stream
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("The SGLang A3 DFC path requires BF16 activations")
        if hidden_states.shape[0] > self.max_source_tokens:
            raise ValueError(
                "DFC input exceeds the configured per-rank prefill capacity"
            )
        if layer.num_experts != self.num_experts or layer.num_fused_shared_experts:
            raise ValueError(
                "DFC requires fixed routed expert geometry without fused shared experts"
            )
        for prefix in ("w13", "w2"):
            if getattr(layer, f"{prefix}_weight_scale").dtype != torch.int64:
                raise RuntimeError(
                    "DFC scales must be packed once during weight loading"
                )
        hidden_states = hidden_states.contiguous()
        ids = topk_output.topk_ids.to(torch.int32).contiguous()
        # SGLang may mark padding routes with -1. DFC receives valid expert IDs
        # and zero weights for those rows, rather than dereferencing expert -1.
        weights = topk_output.topk_weights.to(torch.float32).masked_fill(ids < 0, 0)
        output, _, workspace = self.buffer.dispatch_ffn_combine(
            hidden_states,
            ids.clamp_min(0),
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            weights.contiguous(),
            self.capacity,
            self.num_experts,
            1,
        )
        # The native binding returns scratch ownership separately from output.
        workspace.record_stream(stream)
        return output

    def close(self):
        import torch.distributed as dist

        if self.last_stream is not None:
            self.last_stream.synchronize()
        self.buffer = None
        dist.destroy_process_group(self.group)


def forward_dfc(layer, hidden_states, topk_output):
    runtime = get_resources().buffers.get("ascend_dfc_runtime")
    if runtime is None:
        raise RuntimeError("DFC runtime was not initialized before model execution")
    return runtime.forward(layer, hidden_states, topk_output)


def process_dfc_weights(layer):
    """Build INT8 NZ [E,H,2I]/[E,I,H] and bit-packed per-channel scales once."""
    if getattr(layer, "_dfc_weights_processed", False):
        return
    import torch_npu

    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight")
        scale = getattr(layer, f"{prefix}_weight_scale")
        offset = getattr(layer, f"{prefix}_weight_offset", None)
        if weight.dtype != torch.int8 or weight.device.type != "npu":
            raise ValueError("DFC requires NPU ModelSlim W8A8 expert weights")
        if weight.ndim != 3 or scale.numel() != weight.shape[0] * weight.shape[1]:
            raise ValueError("DFC requires one scale per expert output channel")
        if offset is not None and torch.count_nonzero(offset).item() != 0:
            raise ValueError("DFC does not support nonzero expert weight offsets")
        # Load-time checks/conversions may synchronize; none occur in forward.
        packed = scale.detach().float().contiguous().view(torch.int32).to(torch.int64)
        packed = packed.reshape(-1)
        converted = torch_npu.npu_format_cast(
            weight.detach().transpose(1, 2).contiguous(), 29
        )
        if torch_npu.get_npu_format(converted) != 29:
            raise RuntimeError("DFC weight conversion did not produce FRACTAL_NZ")
        weight.data = converted
        setattr(
            layer,
            f"{prefix}_weight_scale",
            torch.nn.Parameter(packed, requires_grad=False),
        )
    layer._dfc_weights_processed = True
