# GPU image for Unsloth FineTune Studio
#   docker build -t ftstudio .
#   docker run --gpus all -p 7860:7860 -v $PWD/outputs:/app/outputs -e HF_TOKEN=$HF_TOKEN ftstudio
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git cmake build-essential curl libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /app
RUN python -m pip install --upgrade pip && python -m pip install unsloth unsloth_zoo

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install -e .

COPY app.py ./
COPY configs ./configs
COPY data ./data

EXPOSE 7860
CMD ["ftstudio", "ui", "--host", "0.0.0.0", "--port", "7860"]
