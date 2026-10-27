#!/usr/bin/env bash
# Run inside the existing Ascend A3 image with the DFC-enabled ZBAL wheel.
# Edit the configuration below, then run: bash launch_dsv32_dfc_single_node.sh
set -euo pipefail

# ---- Edit launch settings here; no command-line arguments are accepted. ----
MODEL_PATH="/path/to/DeepSeek-V3.2-W8A8"
MOE_BACKEND="ascend_dfc"  # Change to deepep for the baseline.
TP_SIZE=16
DP_SIZE=16
PORT=30000
PREFILL_TOKENS=128
CONTEXT_LENGTH=2048
MEM_FRACTION_STATIC=0.8
DFC_LOG_DIR="./logs/dsv32-dfc"
DRY_RUN=0  # Set to 1 to print the command without launching the model.

# ---- Edit runtime environment here, including the visible logical NPUs. ----
export PYTHONPATH="path/to/sgl/python:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
if [[ "${MOE_BACKEND}" == "ascend_dfc" ]]; then
    # MiB: adjust to the existing machine's ZBAL budget. This is not the DFC
    # workspace size. Mixed allocation avoids a separate large pool at startup.
    export SGLANG_ZBAL_LOCAL_MEM_SIZE=58368
    export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:24669"
    export ZBAL_NPU_ALLOC_CONF="use_vmm_for_static_memory:True"
else
    # The DeepEP baseline uses its existing allocator rather than ZBAL.
    export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
    unset SGLANG_ZBAL_BOOTSTRAP_URL ZBAL_NPU_ALLOC_CONF
fi

# Use max_new_tokens=1 initially to exercise prefill without a decode step.
# No PD decode worker, router or MemFabric weight pool is required.
backend_args=()

case "${MOE_BACKEND}" in
    ascend_dfc)
        if [[ ! "${SGLANG_ZBAL_LOCAL_MEM_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
            echo 'SGLANG_ZBAL_LOCAL_MEM_SIZE must be a positive integer in MiB' >&2
            exit 2
        fi
        ;;
    deepep)
        backend_args+=(--deepep-mode normal)
        ;;
    *)
        echo 'MOE_BACKEND must be ascend_dfc or deepep' >&2
        exit 2
        ;;
esac

for positive_integer in "${TP_SIZE}" "${DP_SIZE}" "${PREFILL_TOKENS}" "${CONTEXT_LENGTH}"; do
    if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo 'TP_SIZE, DP_SIZE, PREFILL_TOKENS and CONTEXT_LENGTH must be positive integers' >&2
        exit 2
    fi
done
if (( TP_SIZE > 16 || PREFILL_TOKENS > CONTEXT_LENGTH )); then
    echo 'Require TP_SIZE <= 16 and PREFILL_TOKENS <= CONTEXT_LENGTH' >&2
    exit 2
fi
if (( TP_SIZE % DP_SIZE != 0 || PREFILL_TOKENS < DP_SIZE )); then
    echo 'DP_SIZE must divide TP_SIZE; PREFILL_TOKENS must be >= DP_SIZE' >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Missing local checkpoint config: ${MODEL_PATH}/config.json; edit MODEL_PATH in this script" >&2
    exit 2
fi

command=(python3 -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --trust-remote-code
    --device npu --attention-backend ascend
    --dtype bfloat16 --quantization modelslim
    --nnodes 1 --node-rank 0
    --tp-size "${TP_SIZE}" --ep-size "${TP_SIZE}"
    --enable-dp-attention --dp-size "${DP_SIZE}"
    --moe-a2a-backend "${MOE_BACKEND}" --moe-weight-offload-backend none
    --disable-shared-experts-fusion
    --disable-prefill-cuda-graph --disable-decode-cuda-graph
    --disable-overlap-schedule --disable-radix-cache
    --chunked-prefill-size "${PREFILL_TOKENS}"
    --max-prefill-tokens "${PREFILL_TOKENS}"
    --context-length "${CONTEXT_LENGTH}"
    --max-total-tokens "${CONTEXT_LENGTH}" --max-running-requests 1
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --watchdog-timeout 900
    --host 127.0.0.1 --port "${PORT}"
    "${backend_args[@]}")

printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

mkdir -p "${DFC_LOG_DIR}"
log_file="${DFC_LOG_DIR}/${MOE_BACKEND}-$(date +%Y%m%d-%H%M%S)-$$.log"
{
    printf 'Backend: %s; TP/EP: %s; attention DP: %s; prefill tokens: %s\n' "${MOE_BACKEND}" "${TP_SIZE}" "${DP_SIZE}" "${PREFILL_TOKENS}"
    printf 'Log: %s\n' "${log_file}"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}"
} 2>&1 | tee "${log_file}"
