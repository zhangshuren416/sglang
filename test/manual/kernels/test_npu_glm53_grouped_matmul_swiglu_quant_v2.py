"""Time Ascend A5 MXFP8 fused GMM1 and GMM2 with GLM-5.3 DFC counts.

Each of the eight EP ranks supplies 8192 tokens to DFC before routing.

Run with the DFC-enabled zbal package installed:
    torchrun --standalone --nproc-per-node=8 \
        test/manual/kernels/test_npu_glm53_grouped_matmul_swiglu_quant_v2.py
"""

import os
import unittest
from functools import partial

import torch
import torch.distributed as dist

from sglang.test.test_utils import CustomTestCase


HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
GATE_UP_SIZE = 2 * INTERMEDIATE_SIZE
TOKENS_PER_RANK = 8192
GLOBAL_EXPERTS = 256
EP_SIZE = 8
TOPK = 8
GLOBAL_TOKENS = TOKENS_PER_RANK * EP_SIZE
LOCAL_EXPERTS = GLOBAL_EXPERTS // EP_SIZE
FP8 = torch.float8_e4m3fn
WARMUP = 5
ITERATIONS = 20


def make_mxfp8_weight(out_features, in_features, device):
    weight = torch.randn(
        LOCAL_EXPERTS, out_features, in_features, device=device, dtype=torch.bfloat16
    )
    weight.mul_(in_features**-0.5)
    return torch.ops.npu.npu_dynamic_mx_quant(weight, dst_type=FP8)


def time_npu_op(op):
    for _ in range(WARMUP):
        op()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        output = op()
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end) * 1000 / ITERATIONS


class TestNpuGlm53GroupedMatmulSwigluQuantV2(CustomTestCase):
    @torch.no_grad()
    def test_glm53_ep8(self):
        self.assertEqual(
            int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
            EP_SIZE,
            "Launch this test with torchrun --standalone --nproc-per-node=8",
        )
        self.assertEqual(int(os.environ["WORLD_SIZE"]), EP_SIZE)

        os.environ["ZBAL_DFC_BALANCE_TOKEN"] = "0"
        import torch_npu
        import zbal

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("npu", local_rank)
        torch.npu.set_device(device)
        torch.npu.manual_seed(rank)
        torch_npu.npu.config.allow_internal_format = True

        e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
        if e8m0 is None:
            e8m0 = torch.float8_e8m0fnu

        max_output_size = GLOBAL_TOKENS * TOPK
        workspace_bytes = zbal.Buffer.get_dfc_workspace_size(
            TOKENS_PER_RANK,
            HIDDEN_SIZE,
            GATE_UP_SIZE,
            LOCAL_EXPERTS,
            EP_SIZE,
            TOPK,
            max_output_size=max_output_size,
        )
        store_port = os.environ.get("ZBAL_STORE_PORT", "23350")
        self.assertTrue(
            zbal.zbal_init_static_pool(
                EP_SIZE,
                local_rank,
                rank,
                workspace_bytes + (32 << 20),
                comm_meta_space_size=8192,
                ip_port=f"tcp://127.0.0.1:{store_port}",
            )
        )
        self.addCleanup(zbal.zbal_uninit)
        dist.init_process_group("zbal")
        self.addCleanup(dist.destroy_process_group)

        # Seed each source rank separately. Every process can regenerate the
        # original DFC inputs for all ranks and regroup them after the call.
        generator = torch.Generator(device="cpu")
        all_tokens = torch.empty(
            GLOBAL_TOKENS,
            HIDDEN_SIZE,
            device="cpu",
            dtype=torch.bfloat16,
        )
        router_logits = torch.empty(
            GLOBAL_TOKENS,
            GLOBAL_EXPERTS,
            device="cpu",
            dtype=torch.float32,
        )
        for source_rank in range(EP_SIZE):
            start = source_rank * TOKENS_PER_RANK
            rows = slice(start, start + TOKENS_PER_RANK)
            generator.manual_seed(source_rank)
            all_tokens[rows].normal_(generator=generator)
            router_logits[rows].normal_(generator=generator)
        router_logits[:, ::4].add_(0.5)
        all_topk_logits, all_topk_idx = router_logits.topk(TOPK, dim=-1)
        input_slice = slice(rank * TOKENS_PER_RANK, (rank + 1) * TOKENS_PER_RANK)
        tokens = all_tokens[input_slice].to(device)
        topk_idx = all_topk_idx[input_slice].to(device=device, dtype=torch.int32)
        topk_weights = all_topk_logits[input_slice].softmax(dim=-1).to(device)
        weight_nd, weight_scale = make_mxfp8_weight(GATE_UP_SIZE, HIDDEN_SIZE, device)
        down_weight, down_scale = make_mxfp8_weight(HIDDEN_SIZE, INTERMEDIATE_SIZE, device)

        buffer = zbal.Buffer(dist.group.WORLD)
        buffer.malloc_dfc_workspace(workspace_bytes)
        dist.barrier()
        # DFC consumes [E, N, K] ND weights and flat scale bytes per expert.
        combined, expert_counts = buffer.dispatch_ffn_combine(
            x=tokens,
            topk_idx=topk_idx,
            weight1=weight_nd,
            scale1=weight_scale.view(torch.uint8).reshape(LOCAL_EXPERTS, -1),
            weight2=down_weight,
            scale2=down_scale.view(torch.uint8).reshape(LOCAL_EXPERTS, -1),
            topk_weights=topk_weights,
            max_output_size=max_output_size,
            num_experts=GLOBAL_EXPERTS,
            mode="prefill",
        )
        torch.npu.synchronize()
        del buffer, combined

        # Rebuild the grouped input from the same tokens and routing decisions.
        # DFC exposes counts and combined output, without the intermediate rows.
        routed_experts = all_topk_idx - rank * LOCAL_EXPERTS
        token_rows, slots = torch.where(
            (routed_experts >= 0) & (routed_experts < LOCAL_EXPERTS)
        )
        local_ids = routed_experts[token_rows, slots]
        expected_counts = torch.bincount(local_ids, minlength=LOCAL_EXPERTS)
        self.assertEqual(expert_counts.dtype, torch.int32)
        counts_cpu = expert_counts.cpu().to(torch.int64)
        torch.testing.assert_close(counts_cpu, expected_counts, rtol=0, atol=0)
        self.assertGreater(counts_cpu.max().item(), counts_cpu.min().item())
        print(f"rank={rank} DFC expert counts={counts_cpu.tolist()}", flush=True)

        row_order = token_rows[local_ids.argsort(stable=True)]
        x = all_tokens.index_select(0, row_order).to(device)
        del all_tokens
        expert_counts = expert_counts.to(torch.int64)
        group_list = expert_counts.cumsum(0)
        local_rows = x.shape[0]
        x, x_scale = torch.ops.npu.npu_dynamic_mx_quant(x, dst_type=FP8)

        # Match SGLang's layout: cast [E, N, K] to FRACTAL_NZ (29), then
        # transpose both weight and scale without materializing the views.
        weight_nz = torch.ops.npu.npu_format_cast(weight_nd, 29).transpose(1, 2)
        weight_scale = weight_scale.transpose(1, 2)
        down_weight_nz = torch.ops.npu.npu_format_cast(down_weight, 29).transpose(1, 2)
        down_scale = down_scale.transpose(1, 2)

        fused_gmm = partial(
            torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2,
            x=x,
            weight=[weight_nz],
            weight_scale=[weight_scale],
            x_scale=x_scale,
            group_list=group_list,
            group_list_type=0,
            dequant_mode=2,
            dequant_dtype=torch.float32,
            quant_mode=2,
            quant_dtype=FP8,
            weight_scale_dtype=e8m0,
            x_scale_dtype=e8m0,
        )
        (y, y_scale), fused_gmm_us = time_npu_op(fused_gmm)
        self.assertEqual(y.shape, (local_rows, INTERMEDIATE_SIZE))
        self.assertEqual(y.dtype, FP8)
        self.assertEqual(y_scale.shape, (local_rows, INTERMEDIATE_SIZE // 64, 2))
        self.assertEqual(y_scale.element_size(), 1)

        gmm2 = partial(
            torch.ops.npu.npu_grouped_matmul,
            x=[y],
            weight=[down_weight_nz],
            scale=[down_scale],
            per_token_scale=[y_scale],
            scale_dtype=e8m0,
            per_token_scale_dtype=e8m0,
            split_item=2,
            group_type=0,
            group_list_type=1,
            group_list=expert_counts,
            output_dtype=torch.bfloat16,
        )
        (projected,), gmm2_us = time_npu_op(gmm2)
        self.assertEqual(projected.shape, (local_rows, HIDDEN_SIZE))
        self.assertEqual(projected.dtype, torch.bfloat16)

        # Decode each E8M0 byte as 2**(byte - 127) for its 32 FP8 values.
        payload = y.view(torch.uint8).cpu().view(FP8).float()
        scale_exponents = y_scale.view(torch.uint8).cpu().float() - 127
        output = payload.reshape(local_rows, -1, 32) * scale_exponents.exp2().reshape(
            local_rows, -1, 1
        )
        self.assertTrue(torch.isfinite(output).all().item())
        print(
            f"rank={rank}, input_tokens={TOKENS_PER_RANK}, M={local_rows}, "
            f"hidden={HIDDEN_SIZE}, ffn={GATE_UP_SIZE}, "
            f"E={LOCAL_EXPERTS}, warmup={WARMUP}, iterations={ITERATIONS}, "
            f"gmm1_fused={fused_gmm_us:.2f} us, gmm2={gmm2_us:.2f} us",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
