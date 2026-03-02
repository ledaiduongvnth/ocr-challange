FROM vllm/vllm-openai:v0.16.0

ENV MODEL_ID="datalab-to/chandra" \
    MODEL_DOWNLOAD_DIR="/tmp/chandra/models" \
    HF_HOME="/tmp/chandra/hf" \
    PORT=7888 \
    VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    VLLM_MAX_NUM_SEQS=1 \
    VLLM_MAX_BATCHED_TOKENS=32768 \
    VLLM_GPU_MEMORY_UTILIZATION=0.85 \
    VLLM_DTYPE="bfloat16" \
    VLLM_SERVED_MODEL_NAME="chandra" \
    MAX_MODEL_LEN=4096

RUN useradd -m -u 1000 user

# Thin launcher for vLLM OpenAI server with safer defaults.
RUN cat <<'EOS' >/usr/local/bin/chandra_vllm
#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-datalab-to/chandra}"
PORT="${PORT:-7888}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DOWNLOAD_DIR="${MODEL_DOWNLOAD_DIR:-/tmp/chandra/models}"
API_KEY="${VLLM_API_KEY:-}"
ADDITIONAL_ARGS="${VLLM_EXTRA_ARGS:---trust-remote-code}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-chandra}"
ATTN_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

mkdir -p "${DOWNLOAD_DIR}"
export VLLM_ATTENTION_BACKEND="${ATTN_BACKEND}"

ARGS=(
  --model "${MODEL_ID}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --download-dir "${DOWNLOAD_DIR}"
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
EOS

RUN chmod +x /usr/local/bin/chandra_vllm

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && chown user:user /app/entrypoint.sh

USER user
ENV PATH="/home/user/.local/bin:${PATH}"

EXPOSE 7888
ENTRYPOINT ["/app/entrypoint.sh"]
