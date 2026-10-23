#!/usr/bin/env bash
# Usage: bash launch_dsv4_fuseep_tp8_dp8.sh /path/to/DeepSeek-V4-Flash-w8a8
# Based on docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/
# deepseek_v4_flash.mdx and the Ascend FuseEP mode-2 deployment examples.
set -eo pipefail

MODEL_PATH="${1:?Pass the path to an Ascend W8A8 DeepSeek V4 checkpoint}"
PORT="${PORT:-6688}"

# Skip sources absent from containers that already configure these environments.
for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
    if [[ -f "${env_script}" ]]; then
        source "${env_script}"
    fi
done

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_HCCL_BUFFSIZE="${DEEPEP_HCCL_BUFFSIZE:-800}"
# FuseEP mode 2 uses this as the capacity for received, top-k-expanded rows.
# Use the capacity from the documented mode-2 prefill deployment example.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=131072
export HCCL_BUFFSIZE=8
export HCCL_OP_EXPANSION_MODE=AIV
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --device npu \
    --dtype bfloat16 \
    --quantization modelslim \
    --attention-backend dsv4 \
    --page-size 128 \
    --tp-size 8 \
    --ep-size 8 \
    --attn-dp-size 8 \
    --chunked-prefill-size 8192 \
    --moe-a2a-backend ascend_fuseep \
    --fuseep-mode 2 \
    --enable-dp-lm-head \
    --kv-cache-dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --max-prefill-tokens 8192 \
    --max-running-requests 64 \
    --load-balance-method round_robin \
    --disable-radix-cache \
    --disable-cuda-graph \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 \
    --port "${PORT}"
