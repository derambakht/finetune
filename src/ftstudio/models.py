"""Model catalog, chat templates and a VRAM estimator.

The catalog is only a convenience for the UI dropdown - any Hugging Face model id
supported by Unsloth (or a local directory) can be typed in directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    family: str
    params_b: float
    chat_template: str
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.model_id}  ·  {self.params_b:g}B  ·  {self.family}"


MODEL_CATALOG: list[ModelInfo] = [
    # Qwen
    ModelInfo("unsloth/Qwen3-0.6B", "Qwen3", 0.6, "qwen3", "Tiny, great for smoke tests"),
    ModelInfo("unsloth/Qwen3-1.7B", "Qwen3", 1.7, "qwen3"),
    ModelInfo("unsloth/Qwen3-4B-Instruct-2507", "Qwen3", 4.0, "qwen3-instruct"),
    ModelInfo("unsloth/Qwen3-4B", "Qwen3", 4.0, "qwen3", "Hybrid thinking model"),
    ModelInfo("unsloth/Qwen3-8B", "Qwen3", 8.0, "qwen3"),
    ModelInfo("unsloth/Qwen3-14B", "Qwen3", 14.0, "qwen3"),
    ModelInfo("unsloth/Qwen3-30B-A3B-Instruct-2507", "Qwen3 MoE", 30.0, "qwen3-instruct"),
    ModelInfo("unsloth/Qwen3-32B", "Qwen3", 32.0, "qwen3"),
    ModelInfo("unsloth/Qwen2.5-7B-Instruct", "Qwen2.5", 7.0, "qwen-2.5"),
    ModelInfo("unsloth/Qwen2.5-Coder-7B-Instruct", "Qwen2.5", 7.0, "qwen-2.5", "Code"),
    # Llama
    ModelInfo("unsloth/Llama-3.2-1B-Instruct", "Llama 3.2", 1.0, "llama-3.1"),
    ModelInfo("unsloth/Llama-3.2-3B-Instruct", "Llama 3.2", 3.0, "llama-3.1"),
    ModelInfo("unsloth/Meta-Llama-3.1-8B-Instruct", "Llama 3.1", 8.0, "llama-3.1"),
    ModelInfo("unsloth/Llama-3.3-70B-Instruct", "Llama 3.3", 70.0, "llama-3.1"),
    # Gemma
    ModelInfo("unsloth/gemma-3-1b-it", "Gemma 3", 1.0, "gemma-3"),
    ModelInfo("unsloth/gemma-3-4b-it", "Gemma 3", 4.0, "gemma-3"),
    ModelInfo("unsloth/gemma-3-12b-it", "Gemma 3", 12.0, "gemma-3"),
    ModelInfo("unsloth/gemma-3-27b-it", "Gemma 3", 27.0, "gemma-3"),
    ModelInfo("unsloth/gemma-3n-E4B-it", "Gemma 3n", 8.0, "gemma-3"),
    # Mistral
    ModelInfo("unsloth/mistral-7b-instruct-v0.3", "Mistral", 7.0, "mistral"),
    ModelInfo("unsloth/Mistral-Nemo-Instruct-2407", "Mistral", 12.0, "mistral"),
    ModelInfo("unsloth/Mistral-Small-3.2-24B-Instruct-2506", "Mistral", 24.0, "mistral"),
    # Phi
    ModelInfo("unsloth/Phi-4", "Phi-4", 14.0, "phi-4"),
    ModelInfo("unsloth/Phi-4-mini-instruct", "Phi-4", 3.8, "phi-4"),
    # OpenAI open weights
    ModelInfo("unsloth/gpt-oss-20b", "gpt-oss", 20.0, "gpt-oss", "MoE, reasoning"),
    # DeepSeek distills
    ModelInfo("unsloth/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1", 7.0, "qwen-2.5", "Reasoning"),
    ModelInfo("unsloth/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1", 8.0, "llama-3.1", "Reasoning"),
]

MODEL_IDS = [m.model_id for m in MODEL_CATALOG]


# Chat templates understood by ``unsloth.chat_templates.get_chat_template``.
# "auto" keeps the tokenizer's own template; "alpaca" uses a plain prompt format.
CHAT_TEMPLATES = [
    "auto",
    "chatml",
    "llama-3.1",
    "llama-3",
    "qwen-2.5",
    "qwen3",
    "qwen3-instruct",
    "gemma-3",
    "mistral",
    "phi-4",
    "phi-3",
    "gpt-oss",
    "zephyr",
    "vicuna",
    "alpaca",
]

# Markers used by ``train_on_responses_only`` to mask everything except the answers.
RESPONSE_MARKERS: dict[str, tuple[str, str]] = {
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen-2.5": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen3": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen3-instruct": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama-3.1": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "llama-3": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "gemma-3": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "mistral": ("[INST]", "[/INST]"),
    "phi-4": ("<|im_start|>user<|im_sep|>", "<|im_start|>assistant<|im_sep|>"),
    "phi-3": ("<|user|>\n", "<|assistant|>\n"),
    "gpt-oss": ("<|start|>user<|message|>", "<|start|>assistant<|channel|>final<|message|>"),
    "zephyr": ("<|user|>\n", "<|assistant|>\n"),
    "vicuna": ("USER: ", "ASSISTANT: "),
    "alpaca": ("### Instruction:\n", "### Response:\n"),
}

ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


def get_model_info(model_id: str) -> ModelInfo | None:
    for m in MODEL_CATALOG:
        if m.model_id.lower() == model_id.strip().lower():
            return m
    return None


def guess_params_b(model_id: str) -> float | None:
    """Best-effort parameter count (in billions) from the catalog or the model name."""
    info = get_model_info(model_id)
    if info:
        return info.params_b
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_id)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+)\s*[mM](?![a-zA-Z])", model_id)
    if match:
        return float(match.group(1)) / 1000.0
    return None


def guess_chat_template(model_id: str) -> str:
    info = get_model_info(model_id)
    if info:
        return info.chat_template
    name = model_id.lower()
    rules = [
        ("gpt-oss", "gpt-oss"),
        ("qwen3", "qwen3"),
        ("qwen2", "qwen-2.5"),
        ("llama-3", "llama-3.1"),
        ("llama3", "llama-3.1"),
        ("gemma", "gemma-3"),
        ("mistral", "mistral"),
        ("phi-4", "phi-4"),
        ("phi-3", "phi-3"),
    ]
    for key, template in rules:
        if key in name:
            return template
    return "auto"


def response_markers(template: str) -> tuple[str, str] | None:
    return RESPONSE_MARKERS.get(template)


# --------------------------------------------------------------------------- #
# VRAM estimate
# --------------------------------------------------------------------------- #

_BYTES_PER_PARAM = {"4bit": 0.56, "8bit": 1.05, "16bit": 2.0, "full": 2.0}


def estimate_vram_gb(
    params_b: float,
    quantization: str,
    lora_r: int,
    batch_size: int,
    max_seq_length: int,
    optim: str = "adamw_8bit",
    gradient_checkpointing: str = "unsloth",
) -> dict[str, float]:
    """Rough training VRAM estimate in GB.

    It is intentionally conservative and is meant to guide choices, not to be exact.
    """
    weights = params_b * _BYTES_PER_PARAM.get(quantization, 2.0)

    if quantization == "full":
        # grads (bf16) + optimizer states + fp32 master copy heuristics
        optim_bytes = 2.0 if "8bit" in optim else 8.0
        trainable = params_b * (2.0 + optim_bytes)
    else:
        # LoRA params ~ proportional to rank and model width
        lora_params_b = params_b * 0.0012 * (lora_r / 16)
        trainable = lora_params_b * (2 + 2 + (2 if "8bit" in optim else 8))

    # Activation memory (heavily reduced by Unsloth's gradient checkpointing)
    hidden_scale = max(params_b, 0.5) ** 0.5
    act_factor = {"unsloth": 0.035, "true": 0.05, "false": 0.35}.get(gradient_checkpointing, 0.05)
    activations = batch_size * (max_seq_length / 1024) * hidden_scale * act_factor * 4

    overhead = 0.8  # CUDA context, kernels, fragmentation
    total = weights + trainable + activations + overhead
    return {
        "weights": round(weights, 2),
        "trainable": round(trainable, 2),
        "activations": round(activations, 2),
        "overhead": overhead,
        "total": round(total, 1),
    }


def recommend_gpu(total_gb: float) -> str:
    tiers = [
        (6, "Any 8 GB GPU (RTX 3060 Ti / 4060)"),
        (11, "12 GB GPU (RTX 3060 12GB / 4070)"),
        (15, "16 GB GPU (T4 / RTX 4060 Ti 16GB / 4080)"),
        (22, "24 GB GPU (RTX 3090 / 4090 / A10G / L4)"),
        (38, "40 GB GPU (A100 40GB)"),
        (46, "48 GB GPU (A6000 / L40S)"),
        (78, "80 GB GPU (A100 80GB / H100)"),
    ]
    for limit, name in tiers:
        if total_gb <= limit:
            return name
    return "Multiple 80 GB GPUs or a smaller model / 4-bit quantization"
