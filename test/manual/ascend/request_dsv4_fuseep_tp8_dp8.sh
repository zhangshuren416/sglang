#!/usr/bin/env bash
# Run after the server reports it is ready.
# Send 8 concurrent requests, each with exactly 8192 input token IDs.
# Each request is explicitly routed to one of the 8 attention-DP ranks.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:6688}"
OUTPUT_DIR="${OUTPUT_DIR:-$(mktemp -d /tmp/sglang-fuseep-requests.XXXXXX)}"
mkdir -p "${OUTPUT_DIR}"

send_request() {
    local rank="$1"
    # Synthetic token IDs: no tokenizer or model files needed on the client.
    python3 - "${rank}" <<'PY' | curl --silent --show-error --fail-with-body \
        "${BASE_URL}/generate" \
        -H 'Content-Type: application/json' \
        --data-binary @- > "${OUTPUT_DIR}/dp${rank}.json"
import json
import sys

rank = int(sys.argv[1])
print(json.dumps({
    "input_ids": [100 + (i + rank * 37) % 4000 for i in range(8192)],
    "routed_dp_rank": rank,
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 1,
        "ignore_eos": True,
    },
    "stream": False,
}))
PY
}

pids=()
for rank in {0..7}; do
    send_request "${rank}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

echo "Responses: ${OUTPUT_DIR}/dp0.json ... dp7.json"
exit "${status}"
