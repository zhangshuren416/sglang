#!/usr/bin/env bash
# Run in the user's existing Ascend A3 image with DSV4/CANN custom ops installed.
# Reference: SGLang's DeepSeek-V4-Flash Ascend tutorial and single-node best practice.
# https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash
# https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash#single-node-pd-mixed
# Edit the settings below, then run: bash launch_dsv4_flash_deepep_single_node.sh
set -euo pipefail

# ---- Launch settings: DeepSeek-V4-Flash ModelSlim W8A8, not V4-Pro/FP8. ----
# The official tutorial links Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp weights.
# MODEL_PATH and DRY_RUN can also be supplied as environment variables.
MODEL_PATH="${MODEL_PATH:-/path/to/DeepSeek-V4-Flash-w8a8-mtp}"
TP_SIZE=16
DP_SIZE=16
PREFILL_TOKENS=8192
CONTEXT_LENGTH=16384
MAX_RUNNING_REQUESTS=1
MEM_FRACTION_STATIC=0.7
HOST=127.0.0.1
PORT=30000
LOG_DIR="./logs/dsv4-flash-deepep"
DRY_RUN="${DRY_RUN:-0}"  # 1 prints settings/command without loading CANN or the model.

# ---- Runtime settings: one full A3 node (8 physical cards, 16 logical dies). ----
VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
SOCKET_IFNAME=lo

for value in "${TP_SIZE}" "${DP_SIZE}" "${PREFILL_TOKENS}" \
    "${CONTEXT_LENGTH}" "${MAX_RUNNING_REQUESTS}" "${PORT}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo 'Parallel sizes, token limits, request limit and port must be positive integers' >&2
        exit 2
    fi
done
IFS=',' read -r -a visible_devices <<< "${VISIBLE_DEVICES}"
if (( TP_SIZE > ${#visible_devices[@]} || TP_SIZE % DP_SIZE != 0 || PORT > 65535 )); then
    echo 'Require enough visible devices, DP_SIZE dividing TP_SIZE, and PORT <= 65535' >&2
    exit 2
fi
if (( PREFILL_TOKENS < DP_SIZE || PREFILL_TOKENS % DP_SIZE != 0 || PREFILL_TOKENS > CONTEXT_LENGTH )); then
    echo 'Require DP_SIZE <= PREFILL_TOKENS <= CONTEXT_LENGTH and PREFILL_TOKENS divisible by DP_SIZE' >&2
    exit 2
fi
if [[ "${DRY_RUN}" != 0 && "${DRY_RUN}" != 1 ]]; then
    echo 'DRY_RUN must be 0 or 1' >&2
    exit 2
fi

command=(python3 -m sglang.launch_server
    --model-path "${MODEL_PATH}" --trust-remote-code
    --device npu --attention-backend dsv4 --page-size 128
    --dtype bfloat16 --quantization modelslim --kv-cache-dtype bfloat16
    --nnodes 1 --node-rank 0
    --tp-size "${TP_SIZE}" --ep-size "${TP_SIZE}"
    --dp-size "${DP_SIZE}" --enable-dp-attention --enable-dp-lm-head
    --moe-a2a-backend deepep --deepep-mode normal
    --moe-weight-offload-backend none
    --chunked-prefill-size "${PREFILL_TOKENS}"
    --max-prefill-tokens "${PREFILL_TOKENS}"
    --context-length "${CONTEXT_LENGTH}"
    --max-total-tokens "${CONTEXT_LENGTH}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --prefill-max-requests "${MAX_RUNNING_REQUESTS}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --disable-prefill-cuda-graph --disable-decode-cuda-graph
    --disable-overlap-schedule --disable-radix-cache
    --watchdog-timeout 9000
    --host "${HOST}" --port "${PORT}")

printf 'Model: %s; TP/EP: %s; attention DP: %s; mem fraction: %s\n' \
    "${MODEL_PATH}" "${TP_SIZE}" "${DP_SIZE}" "${MEM_FRACTION_STATIC}"
printf 'DeepEP normal; ZBAL disabled; graphs, overlap and MTP disabled.\n'
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then
    exit 0
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Missing ${MODEL_PATH}/config.json; set MODEL_PATH to local V4-Flash ModelSlim W8A8 weights" >&2
    exit 2
fi

# Use the existing image's environment scripts when present. Nothing is installed
# or built here. Some vendor scripts reference unset variables, so suspend nounset.
set +u
for setup_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
    if [[ -f "${setup_script}" ]]; then
        source "${setup_script}"
    fi
done
set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
export HCCL_BUFFSIZE=8
export HCCL_OP_EXPANSION_MODE=AIV
export DEEPEP_HCCL_BUFFSIZE=2500
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export INF_NAN_MODE_FORCE_DISABLE=1
export STREAMS_PER_DEVICE=32

# Select native deep_ep dispatch/combine + separate FFN, without DFC/FuseEP.
# Clear the old DFC/ZBAL environment even when this shell previously ran V3.2.
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
unset SGLANG_ZBAL_BOOTSTRAP_URL ZBAL_NPU_ALLOC_CONF ZBAL_ENABLE_GRAPH

# Ascend-compatible V4 settings from the official tutorial/current NPU CI.
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0

mkdir -p "${LOG_DIR}"
log_file="${LOG_DIR}/deepep-$(date +%Y%m%d-%H%M%S)-$$.log"
{
    printf 'Log: %s\n' "${log_file}"
    printf 'Devices: %s; backend: native DeepEP normal; ZBAL: disabled\n' "${VISIBLE_DEVICES}"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}"
} 2>&1 | tee "${log_file}"
