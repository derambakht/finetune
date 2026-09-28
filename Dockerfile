# syntax=docker/dockerfile:1
#
# Unsloth FineTune Studio — two build targets:
#   gpu (default) : full training stack (CUDA + Unsloth). Needs an NVIDIA GPU + NVIDIA Container Toolkit.
#   cpu           : lightweight UI-only image (configure runs, preview datasets, browse results).
#
#   docker compose up -d --build                         # GPU server (see docker-compose.yml)
#   docker compose -f docker-compose.cpu.yml up -d --build   # UI only, no GPU
#
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# --------------------------------------------------------------------------- #
# cpu: UI only (no system packages needed)
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    FTSTUDIO_OUTPUT_ROOT=/app/outputs \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# transformers is optional here: it lets the dataset preview use the real chat template.
RUN pip install -e . transformers
COPY app.py ./
COPY configs ./configs
COPY data ./data

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=4)"]
CMD ["ftstudio", "ui", "--host", "0.0.0.0", "--port", "7860"]

# --------------------------------------------------------------------------- #
# gpu: full training stack
# --------------------------------------------------------------------------- #
FROM ${CUDA_IMAGE} AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    FTSTUDIO_OUTPUT_ROOT=/app/outputs \
    GRADIO_ANALYTICS_ENABLED=False \
    TOKENIZERS_PARALLELISM=false \
    PATH=/opt/venv/bin:$PATH

# build tools + curl libs are needed by Unsloth to compile llama.cpp for GGUF export
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git cmake build-essential curl libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# Pin a version for reproducible builds, e.g. --build-arg UNSLOTH_SPEC="unsloth==2026.9.1 unsloth_zoo==2026.9.1"
ARG UNSLOTH_SPEC="unsloth unsloth_zoo"
RUN pip install ${UNSLOTH_SPEC}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .
COPY app.py ./
COPY configs ./configs
COPY data ./data

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=4)"]
CMD ["ftstudio", "ui", "--host", "0.0.0.0", "--port", "7860"]
