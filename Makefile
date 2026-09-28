# Shortcuts for running Unsloth FineTune Studio with Docker Compose.
#   make up            build & start the GPU stack   (http://<server>:7860)
#   make up-cpu        build & start the UI-only stack (no GPU)
#   make train CONFIG=configs/persian_qwen3_chat.yaml   headless training in a one-off container
#   make logs | ps | down | restart | shell | gpu-check | test

COMPOSE     ?= docker compose
COMPOSE_CPU ?= docker compose -f docker-compose.cpu.yml
CONFIG      ?= configs/persian_qwen3_chat.yaml

.PHONY: help env up up-cpu down down-cpu restart logs ps shell train gpu-check build pull-base test

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

env:
	@test -f .env || (cp .env.example .env && echo "Created .env - edit FTSTUDIO_AUTH and HF_TOKEN before exposing the server.")

up: env
	$(COMPOSE) up -d --build
	@echo "Studio is starting on port $${FTSTUDIO_PORT:-7860}  (make logs to follow)"

up-cpu: env
	$(COMPOSE_CPU) up -d --build

down:
	$(COMPOSE) down

down-cpu:
	$(COMPOSE_CPU) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

shell:
	$(COMPOSE) exec studio bash

build: env
	$(COMPOSE) build

# Headless training from a YAML config (outputs land in ./outputs like UI runs).
train: env
	$(COMPOSE) run --rm --no-deps studio ftstudio train --config $(CONFIG)

# Verify that Docker can see the GPU (NVIDIA Container Toolkit installed & working).
gpu-check: env
	$(COMPOSE) run --rm --no-deps studio nvidia-smi

test:
	python -m pytest -q
