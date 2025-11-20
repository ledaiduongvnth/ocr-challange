#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/.local/bin:${PATH}"

if ! python3 -c "import chandra" >/dev/null 2>&1; then
  echo "[entrypoint] Installing chandra-ocr package (fallback - should not normally run)"
  # Use same pinned versions as Dockerfile for reproducibility
  { \
    echo "numpy<2.3.0"; \
    echo "setuptools<80"; \
    echo "torch>=2.8.0,<2.9.0"; \
    echo "vllm==0.11.0"; \
  } > /tmp/constraints.txt
  python3 -m pip install --no-cache-dir --user --ignore-installed \
    "blinker==1.9.0" \
    "chandra-ocr==0.1.8" \
    --constraint /tmp/constraints.txt
  rm -f /tmp/constraints.txt
fi

echo "[entrypoint] Launching server via chandra_vllm helper"
exec /usr/local/bin/chandra_vllm "$@"
