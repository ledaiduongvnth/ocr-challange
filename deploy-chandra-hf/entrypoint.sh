#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/.local/bin:${PATH}"

echo "[entrypoint] Launching server via chandra_vllm helper"
exec /usr/local/bin/chandra_vllm "$@"
