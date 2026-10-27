# A3 DFC integration

This path replaces the routed MoE dispatch/FFN/combine operation with the
native ZBAL `Buffer.dispatch_ffn_combine` binding. Routing and the separate
shared-expert MLP remain in SGLang. The interface was checked against
MemFabric `offload-layer` commit `00b7195d`.

## Resident DFC

Use the existing Ascend image, installed ZBAL wheel and ModelSlim W8A8
deployment configuration. Set `SGLANG_ZBAL_LOCAL_MEM_SIZE` (MiB) and
`SGLANG_ZBAL_BOOTSTRAP_URL` as required by that deployment, then select:

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/DeepSeek-V4-w8a8 \
    --device npu --dtype bfloat16 --quantization modelslim \
    --attention-backend dsv4 \
    --moe-a2a-backend ascend_dfc \
    --tp-size 8 --ep-size 8 --enable-dp-attention --dp-size 8 \
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --page-size 128 --mem-fraction-static 0.8
```

Add the existing PD prefill arguments when using disaggregation. All expert
weights remain resident in HBM in this mode; no MemFabric offload pool is
created. The command is a configuration example, not a verified deployment.

## Implementation

- `dfc.py`: native Buffer lifecycle, forward adaptation, one-time INT8 NZ
  conversion, packed scales and a conservative native workspace budget.
- `dfc_model_adapter.py`: model compatibility checks and initialization through
  the existing layer construction/post-load hooks, independent of weight
  offloading.
- NPU initialization: reserve 8192 KiB of ZBAL communication metadata and
  initialize mixed allocator metadata before KV cache sizing.
- KV sizing: subtract the native DFC workspace reservation before allocating
  the KV pool, in addition to the existing activation headroom.

Capacity defaults to `max(max_prefill_tokens, chunked_prefill_size) * top_k * EP`
expanded rows. `--dfc-max-output-tokens` can increase that capacity. Large
capacities consume substantial temporary HBM even with a small input batch.
The workspace estimate follows the checked native version and must be reviewed
when upgrading that dependency.

The initial adapter targets A3, DSV4, ModelSlim W8A8, BF16, fixed EP spanning TP
and eager execution. Native EP is limited to 16 and expert count to 4096, divisible
by EP. Shared-expert fusion and the combinations rejected by `dfc_hook.py` have
not been adapted. In particular, speculative execution remains gated in this
version; this does not establish a fundamental incompatibility with DFC.

The current API cannot represent routed SwiGLU clipping or nonzero expert
weight offsets. Loading validates the implemented compatibility boundaries.
Online updates and checkpoint export of the packed execution layout are rejected.

## Validation status

Only source/static checks have been performed. No compilation, CPU simulation,
A3 execution, model accuracy or performance validation has been run. Before
using the deployment, compare routed scaling, gate/up ordering, shared outputs,
padding and zero-token ranks against the existing backend with the same model.

Weight offloading is an additional change layered on this resident DFC path;
it is not required to evaluate DFC correctness independently.
