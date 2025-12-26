# Use the Docker Hub mirror so we do not rely on GHCR auth
# Fully pinned for reproducibility
# Base image: vllm/vllm-openai:latest@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c
# Contains: vllm 0.11.0, torch 2.8.0, numpy 2.2.6, setuptools 79.0.1
FROM vllm/vllm-openai:v0.11.0

ENV MODEL_ID="datalab-to/chandra" \
    MODEL_DOWNLOAD_DIR="/tmp/chandra/models" \
    HF_HOME="/tmp/chandra/hf" \
    PORT=7888 \
    VLLM_ATTENTION_BACKEND="TORCH_SDPA" \
    VLLM_MAX_NUM_SEQS=16 \
    VLLM_MAX_BATCHED_TOKENS=32768 \
    VLLM_GPU_MEMORY_UTILIZATION=0.85 \
    VLLM_DTYPE="float16" \
    VLLM_SERVED_MODEL_NAME="chandra" \
    MAX_MODEL_LEN=16384

# Create user first
RUN useradd -m -u 1000 user

# Create chandra_vllm script (needs root for /usr/local/bin)
RUN cat <<'EOF' >/usr/local/bin/chandra_vllm
#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-datalab-to/chandra}"
PORT="${PORT:-7888}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
DOWNLOAD_DIR="${MODEL_DOWNLOAD_DIR:-/tmp/chandra/models}"
API_KEY="${VLLM_API_KEY:-}"
ADDITIONAL_ARGS="${VLLM_EXTRA_ARGS:-}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${VLLM_DTYPE:-float16}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-chandra}"
ATTN_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"

mkdir -p "${DOWNLOAD_DIR}"

export VLLM_ATTENTION_BACKEND="${ATTN_BACKEND}"

ARGS=(
  --model "${MODEL_ID}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --download-dir "${DOWNLOAD_DIR}"
  --no-enforce-eager
  --max-num-seqs "${MAX_NUM_SEQS}"
  --dtype "${DTYPE}"
  --max_num_batched_tokens "${MAX_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --served-model-name "${SERVED_MODEL_NAME}"
)

if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi

if [[ -n "${ADDITIONAL_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${ADDITIONAL_ARGS})
  ARGS+=("${EXTRA_ARR[@]}")
fi

echo "[chandra_vllm] Starting vLLM OpenAI server with model '${MODEL_ID}' on port ${PORT}"
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
EOF

RUN chmod +x /usr/local/bin/chandra_vllm

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && chown user:user /app/entrypoint.sh

# Switch to user and install packages (avoids root pip warning)
USER user

# Install chandra-ocr with pinned versions for full reproducibility
# Set PATH so pip doesn't warn about scripts location
ENV PATH="/home/user/.local/bin:${PATH}"
# Pin versions from successful build: chandra-ocr 0.1.8, torch 2.8.0, vllm 0.11.0
# Constraints prevent transitive dependencies from upgrading incompatible versions
RUN { \
        echo "numpy<2.3.0"; \
        echo "setuptools<80"; \
        echo "torch>=2.8.0,<2.9.0"; \
        echo "vllm==0.11.0"; \
    } > /tmp/constraints.txt && \
    pip install --no-cache-dir --user --ignore-installed \
        "blinker==1.9.0" \
        "chandra-ocr==0.1.8" \
        --constraint /tmp/constraints.txt && \
    rm /tmp/constraints.txt

EXPOSE 7888

ENTRYPOINT ["/app/entrypoint.sh"]
