"""Typed configuration for a fine-tuning run.

The same :class:`FinetuneConfig` object is produced by the UI, stored as YAML in
every run directory and consumed by the training subprocess, so a run is always
fully reproducible from its ``config.yaml``.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Choice constants (shared with the UI)
# --------------------------------------------------------------------------- #

QUANTIZATION_CHOICES = {
    "4bit": "QLoRA 4-bit (bitsandbytes NF4) - lowest VRAM",
    "8bit": "LoRA 8-bit (bitsandbytes int8)",
    "16bit": "LoRA 16-bit (bf16/fp16) - best adapter quality",
    "full": "Full fine-tuning (all weights, highest VRAM)",
}

DTYPE_CHOICES = ["auto", "bfloat16", "float16"]

DATASET_SOURCES = ["huggingface", "local"]

DATASET_FORMATS = {
    "auto": "Auto-detect from columns",
    "alpaca": "Alpaca (instruction / input / output)",
    "sharegpt": "ShareGPT (conversations: from / value)",
    "messages": "OpenAI / ChatML (messages: role / content)",
    "prompt_completion": "Prompt / Completion pairs",
    "text": "Raw text (continued pre-training)",
    "custom": "Custom column mapping",
}

OPTIMIZERS = [
    "adamw_8bit",
    "paged_adamw_8bit",
    "adamw_torch",
    "adamw_torch_fused",
    "paged_adamw_32bit",
    "adafactor",
    "sgd",
]

LR_SCHEDULERS = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "constant",
    "constant_with_warmup",
    "polynomial",
    "inverse_sqrt",
]

GRADIENT_CHECKPOINTING = ["unsloth", "true", "false"]

LORA_BIAS = ["none", "all", "lora_only"]

REPORT_TO = ["none", "tensorboard", "wandb"]

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

GGUF_QUANT_METHODS = {
    "q4_k_m": "Q4_K_M - recommended balance of size & quality",
    "q5_k_m": "Q5_K_M - higher quality, larger",
    "q8_0": "Q8_0 - near lossless, fast conversion",
    "q6_k": "Q6_K - high quality",
    "q4_0": "Q4_0 - legacy 4-bit",
    "q5_0": "Q5_0 - legacy 5-bit",
    "q3_k_m": "Q3_K_M - small",
    "q2_k": "Q2_K - smallest, lowest quality",
    "f16": "F16 - no quantization (16-bit)",
    "bf16": "BF16 - no quantization (bfloat16)",
}

EXPORT_METHODS = {
    "lora": "LoRA adapters only (small, needs base model)",
    "merged_16bit": "Merged 16-bit model (vLLM / HF transformers)",
    "merged_4bit": "Merged 4-bit model (bitsandbytes)",
    "gguf": "GGUF (llama.cpp / Ollama / LM Studio)",
}


# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048
    quantization: str = "4bit"
    dtype: str = "auto"
    chat_template: str = "auto"
    trust_remote_code: bool = False
    hf_token: str = ""


@dataclass
class DatasetConfig:
    source: str = "huggingface"
    path: str = "yahma/alpaca-cleaned"
    subset: str = ""
    split: str = "train"
    format: str = "auto"
    # Column mapping used by "custom" / "prompt_completion" / "text" formats
    prompt_column: str = "prompt"
    response_column: str = "completion"
    system_column: str = ""
    text_column: str = "text"
    system_prompt: str = ""
    max_samples: int = 0  # 0 = use all
    shuffle: bool = True
    eval_split: str = ""  # name of an existing eval split (takes precedence)
    eval_ratio: float = 0.0  # hold-out fraction when no eval split is given
    num_proc: int = 2


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    bias: str = "none"
    use_rslora: bool = False
    use_loftq: bool = False
    gradient_checkpointing: str = "unsloth"


@dataclass
class TrainingConfig:
    run_name: str = "my-finetune"
    output_root: str = "outputs"
    num_train_epochs: float = 1.0
    max_steps: int = 60  # > 0 overrides epochs
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0
    warmup_steps: int = 5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 3407
    packing: bool = False
    train_on_responses_only: bool = True
    instruction_part: str = ""  # auto-filled from chat template when empty
    response_part: str = ""
    neftune_noise_alpha: float = 0.0
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 2
    eval_steps: int = 50
    report_to: str = "none"
    resume_from_checkpoint: str = ""


@dataclass
class ExportConfig:
    export_after_training: bool = False
    methods: list[str] = field(default_factory=lambda: ["lora"])
    gguf_quantizations: list[str] = field(default_factory=lambda: ["q4_k_m"])
    push_to_hub: bool = False
    hub_repo_id: str = ""
    hub_private: bool = True


@dataclass
class FinetuneConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # ------------------------------------------------------------------ io
    def to_dict(self, redact_secrets: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if redact_secrets and data["model"].get("hf_token"):
            data["model"]["hf_token"] = ""
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FinetuneConfig:
        data = data or {}
        kwargs = {}
        for f in fields(cls):
            section_cls = f.default_factory  # type: ignore[misc]
            section_data = data.get(f.name) or {}
            kwargs[f.name] = _build_section(section_cls, section_data)
        return cls(**kwargs)

    def to_yaml(self, redact_secrets: bool = True) -> str:
        return yaml.safe_dump(
            self.to_dict(redact_secrets=redact_secrets), sort_keys=False, allow_unicode=True
        )

    @classmethod
    def from_yaml(cls, text: str) -> FinetuneConfig:
        return cls.from_dict(yaml.safe_load(text) or {})

    def save(self, path: str | Path, redact_secrets: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(redact_secrets=redact_secrets), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FinetuneConfig:
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------ helpers
    @property
    def effective_batch_size(self) -> int:
        return self.training.per_device_train_batch_size * self.training.gradient_accumulation_steps

    @property
    def is_full_finetune(self) -> bool:
        return self.model.quantization == "full"


def _build_section(section_cls: type, data: dict[str, Any]):
    """Instantiate a dataclass section, ignoring unknown keys and coercing types."""
    known = {f.name: f for f in fields(section_cls)}
    defaults = section_cls()
    kwargs = {}
    for name, value in data.items():
        if name not in known:
            continue
        default_value = getattr(defaults, name)
        kwargs[name] = _coerce(value, default_value)
    return dataclasses.replace(defaults, **kwargs)


def _coerce(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value)
    if isinstance(default, str):
        return str(value)
    return value


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"- ❌ **Error:** {e}")
        for w in self.warnings:
            lines.append(f"- ⚠️ {w}")
        if not lines:
            lines.append("- ✅ Configuration looks good.")
        return "\n".join(lines)


_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_config(cfg: FinetuneConfig) -> ValidationResult:
    res = ValidationResult()
    m, d, lo, t, ex = cfg.model, cfg.dataset, cfg.lora, cfg.training, cfg.export

    if not m.model_name.strip():
        res.errors.append("A model name is required.")
    if m.quantization not in QUANTIZATION_CHOICES:
        res.errors.append(f"Unknown quantization '{m.quantization}'.")
    if m.dtype not in DTYPE_CHOICES:
        res.errors.append(f"Unknown dtype '{m.dtype}'.")
    if not 128 <= m.max_seq_length <= 262144:
        res.errors.append("max_seq_length must be between 128 and 262144.")

    if d.source not in DATASET_SOURCES:
        res.errors.append(f"Unknown dataset source '{d.source}'.")
    if not d.path.strip():
        res.errors.append("A dataset path / Hugging Face id is required.")
    elif d.source == "local" and not Path(d.path).exists():
        res.errors.append(f"Local dataset file not found: {d.path}")
    if d.format not in DATASET_FORMATS:
        res.errors.append(f"Unknown dataset format '{d.format}'.")
    if d.format == "custom" and not (d.prompt_column and d.response_column):
        res.errors.append("Custom format requires both prompt and response columns.")
    if not 0.0 <= d.eval_ratio < 0.5:
        res.errors.append("eval_ratio must be in [0, 0.5).")
    if d.max_samples < 0:
        res.errors.append("max_samples cannot be negative.")

    if not cfg.is_full_finetune:
        if lo.r <= 0:
            res.errors.append("LoRA rank must be positive.")
        if lo.alpha <= 0:
            res.errors.append("LoRA alpha must be positive.")
        if not 0.0 <= lo.dropout < 1.0:
            res.errors.append("LoRA dropout must be in [0, 1).")
        if not lo.target_modules:
            res.errors.append("Select at least one LoRA target module.")
        if lo.dropout > 0 or lo.bias != "none":
            res.warnings.append(
                "Unsloth's fast path is optimised for dropout=0 and bias='none'; other values are slower."
            )
        if lo.use_loftq and m.quantization == "4bit":
            res.warnings.append("LoftQ initialises from full-precision weights; use it with 16-bit loading.")
    if lo.gradient_checkpointing not in GRADIENT_CHECKPOINTING:
        res.errors.append(f"Unknown gradient checkpointing mode '{lo.gradient_checkpointing}'.")

    if not t.run_name.strip() or not _RUN_NAME_RE.match(t.run_name.strip()):
        res.errors.append("Run name may only contain letters, digits, '.', '_' and '-'.")
    if t.max_steps <= 0 and t.num_train_epochs <= 0:
        res.errors.append("Set either max_steps > 0 or num_train_epochs > 0.")
    if t.per_device_train_batch_size <= 0 or t.gradient_accumulation_steps <= 0:
        res.errors.append("Batch size and gradient accumulation must be positive.")
    if not 0 < t.learning_rate < 1:
        res.errors.append("Learning rate must be in (0, 1).")
    if t.optim not in OPTIMIZERS:
        res.errors.append(f"Unknown optimizer '{t.optim}'.")
    if t.lr_scheduler_type not in LR_SCHEDULERS:
        res.errors.append(f"Unknown LR scheduler '{t.lr_scheduler_type}'.")
    if not 0.0 <= t.warmup_ratio < 1.0:
        res.errors.append("warmup_ratio must be in [0, 1).")
    if t.report_to not in REPORT_TO:
        res.errors.append(f"Unknown report_to '{t.report_to}'.")
    if t.learning_rate > 5e-4:
        res.warnings.append("Learning rate above 5e-4 often destabilises LoRA training.")
    if cfg.is_full_finetune and t.learning_rate > 5e-5:
        res.warnings.append("Full fine-tuning usually needs a much smaller learning rate (1e-5 - 5e-5).")
    if t.packing and t.train_on_responses_only:
        res.warnings.append("Packing combined with train-on-responses-only can mask samples imperfectly.")
    if d.format == "text" and t.train_on_responses_only:
        res.warnings.append("Train-on-responses-only is ignored for raw text datasets.")

    for method in ex.methods:
        if method not in EXPORT_METHODS:
            res.errors.append(f"Unknown export method '{method}'.")
    if "gguf" in ex.methods and not ex.gguf_quantizations:
        res.errors.append("Select at least one GGUF quantization method.")
    for q in ex.gguf_quantizations:
        if q not in GGUF_QUANT_METHODS:
            res.errors.append(f"Unknown GGUF quantization '{q}'.")
    if ex.push_to_hub and not ex.hub_repo_id.strip():
        res.errors.append("Push to Hub requires a repository id (user/model-name).")
    if ex.push_to_hub and not m.hf_token and not _env_hf_token():
        res.warnings.append("No Hugging Face token set; pushing to the Hub will fail unless you are logged in.")

    return res


def _env_hf_token() -> str:
    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
