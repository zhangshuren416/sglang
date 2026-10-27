#!/usr/bin/env bash
# Run inside the user's existing Ascend image with built ZBAL/MemFabric wheels.
# Usage: bash launch_dsv4_dfc_prefill.sh MODEL_PATH POOL_JSON [serving arguments...]
set -euo pipefail

model_path="${1:?Pass the local ModelSlim W8A8 checkpoint path}"
pool_config="${2:?Pass the MemFabric SHARED weight pool JSON path}"
shift 2

: "${SGLANG_ZBAL_LOCAL_MEM_SIZE:?Set the existing ZBAL HBM budget in MiB}"
: "${SGLANG_ZBAL_BOOTSTRAP_URL:?Set a ZBAL rendezvous URL for this prefill job}"

exec python3 -m sglang.launch_server \
    --model-path "${model_path}" \
    --device npu \
    --dtype bfloat16 \
    --quantization modelslim \
    --attention-backend dsv4 \
    --disaggregation-mode prefill \
    --moe-a2a-backend ascend_dfc \
    --moe-weight-offload-backend memfabric \
    --moe-weight-offload-config "${pool_config}" \
    "$@"
