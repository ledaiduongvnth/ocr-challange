# Lightweight GPU-enabled image for running `pipeline/app.py`
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r /app/pipeline/requirements.txt

# Copy source
COPY pipeline /app/pipeline

# Default port used by app.py
EXPOSE 7860

CMD ["uvicorn", "pipeline.app:app", "--host", "0.0.0.0", "--port", "7860"]
