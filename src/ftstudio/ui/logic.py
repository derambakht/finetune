"""UI-independent logic behind the Gradio app (kept separate so it is unit-testable)."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import FinetuneConfig
from ..models import estimate_vram_gb, get_model_info, guess_chat_template, guess_params_b, recommend_gpu
from ..reporting import format_duration, list_checkpoints, list_runs, read_metrics, read_status

# Ordered registry of every config field that has a UI component.
FIELDS: list[tuple[str, str]] = [
    # model
    ("model", "model_name"),
    ("model", "quantization"),
    ("model", "dtype"),
    ("model", "max_seq_length"),
    ("model", "chat_template"),
    ("model", "trust_remote_code"),
    ("model", "hf_token"),
    # dataset
    ("dataset", "source"),
    ("dataset", "path"),
    ("dataset", "subset"),
    ("dataset", "split"),
    ("dataset", "format"),
    ("dataset", "prompt_column"),
    ("dataset", "response_column"),
    ("dataset", "system_column"),
    ("dataset", "text_column"),
    ("dataset", "system_prompt"),
    ("dataset", "normalize_persian"),
    ("dataset", "max_samples"),
    ("dataset", "shuffle"),
    ("dataset", "eval_split"),
    ("dataset", "eval_ratio"),
    ("dataset", "num_proc"),
    # lora
    ("lora", "r"),
    ("lora", "alpha"),
    ("lora", "dropout"),
    ("lora", "target_modules"),
    ("lora", "bias"),
    ("lora", "use_rslora"),
    ("lora", "use_loftq"),
    ("lora", "gradient_checkpointing"),
    # training
    ("training", "run_name"),
    ("training", "output_root"),
    ("training", "num_train_epochs"),
    ("training", "max_steps"),
    ("training", "per_device_train_batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("training", "learning_rate"),
    ("training", "lr_scheduler_type"),
    ("training", "warmup_ratio"),
    ("training", "warmup_steps"),
    ("training", "optim"),
    ("training", "weight_decay"),
    ("training", "max_grad_norm"),
    ("training", "seed"),
    ("training", "packing"),
    ("training", "train_on_responses_only"),
    ("training", "instruction_part"),
    ("training", "response_part"),
    ("training", "neftune_noise_alpha"),
    ("training", "logging_steps"),
    ("training", "save_steps"),
    ("training", "save_total_limit"),
    ("training", "eval_steps"),
    ("training", "report_to"),
    ("training", "resume_from_checkpoint"),
    # export
    ("export", "export_after_training"),
    ("export", "methods"),
    ("export", "gguf_quantizations"),
    ("export", "push_to_hub"),
    ("export", "hub_repo_id"),
    ("export", "hub_private"),
]

SECRET_FIELDS = {("model", "hf_token")}


def values_to_config(values: list[Any] | tuple[Any, ...]) -> FinetuneConfig:
    data: dict[str, dict[str, Any]] = {}
    for (section, name), value in zip(FIELDS, values):
        if value is None:
            continue
        data.setdefault(section, {})[name] = value
    return FinetuneConfig.from_dict(data)


def config_to_values(cfg: FinetuneConfig, keep_secrets: bool = True) -> list[Any]:
    """Values for every component in :data:`FIELDS`.

    Secret fields that are empty in ``cfg`` are returned as ``None`` when
    ``keep_secrets`` is set so the caller can leave the component untouched.
    """
    out = []
    for section, name in FIELDS:
        value = getattr(getattr(cfg, section), name)
        if keep_secrets and (section, name) in SECRET_FIELDS and not value:
            value = None
        out.append(value)
    return out


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "⚡ Quick test (30 steps)": {
        "training": {"max_steps": 30, "num_train_epochs": 1, "per_device_train_batch_size": 2,
                     "gradient_accumulation_steps": 4, "learning_rate": 2e-4, "warmup_steps": 5,
                     "lr_scheduler_type": "linear", "save_steps": 1000},
        "dataset": {"max_samples": 2000, "eval_ratio": 0.0},
    },
    "✅ Standard (1 epoch)": {
        "training": {"max_steps": 0, "num_train_epochs": 1, "per_device_train_batch_size": 2,
                     "gradient_accumulation_steps": 4, "learning_rate": 2e-4, "warmup_ratio": 0.03,
                     "warmup_steps": 0, "lr_scheduler_type": "linear", "save_steps": 200},
        "lora": {"r": 16, "alpha": 16},
        "dataset": {"max_samples": 0, "eval_ratio": 0.02},
    },
    "🎯 High quality (3 epochs)": {
        "training": {"max_steps": 0, "num_train_epochs": 3, "per_device_train_batch_size": 4,
                     "gradient_accumulation_steps": 4, "learning_rate": 1e-4, "warmup_ratio": 0.05,
                     "warmup_steps": 0, "lr_scheduler_type": "cosine", "weight_decay": 0.01,
                     "save_steps": 250, "eval_steps": 100},
        "lora": {"r": 32, "alpha": 32, "use_rslora": False},
        "dataset": {"max_samples": 0, "eval_ratio": 0.05},
    },
    "💾 Low VRAM (≤ 8 GB)": {
        "model": {"quantization": "4bit", "max_seq_length": 1024},
        "training": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
                     "optim": "adamw_8bit", "packing": False},
        "lora": {"r": 8, "alpha": 16, "gradient_checkpointing": "unsloth"},
    },
    "🧬 Full fine-tuning": {
        "model": {"quantization": "full"},
        "training": {"learning_rate": 2e-5, "optim": "adamw_8bit", "per_device_train_batch_size": 1,
                     "gradient_accumulation_steps": 8, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
                     "warmup_steps": 0},
    },
    "🇮🇷 Persian chat (فارسی)": {
        "model": {"model_name": "unsloth/Qwen3-4B-Instruct-2507", "chat_template": "qwen3-instruct",
                  "quantization": "4bit"},
        "dataset": {"normalize_persian": True,
                    "system_prompt": "تو یک دستیار هوشمند فارسی‌زبان هستی که پاسخ‌های دقیق، مؤدبانه و روان می‌دهی."},
        "training": {"train_on_responses_only": True, "learning_rate": 2e-4},
        "lora": {"r": 16, "alpha": 32},
    },
    "📖 Continued pre-training (raw text)": {
        "dataset": {"format": "text"},
        "training": {"train_on_responses_only": False, "packing": True, "learning_rate": 5e-5},
        "lora": {"r": 64, "alpha": 64, "use_rslora": True,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                    "down_proj", "embed_tokens", "lm_head"]},
    },
}


SAMPLE_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# (label, relative path under data/, format) — bundled example datasets
SAMPLE_DATASETS: list[tuple[str, str, str]] = [
    ("🇮🇷 فارسی · گفتگو (messages) · 30", "persian/fa_chat_messages.jsonl", "messages"),
    ("🇮🇷 فارسی · دستورالعمل (Alpaca) · 15", "persian/fa_alpaca.jsonl", "alpaca"),
    ("🇮🇷 فارسی · گفتگوی چندمرحله‌ای (ShareGPT) · 6", "persian/fa_sharegpt.jsonl", "sharegpt"),
    ("🇮🇷 فارسی · پرسش و پاسخ (CSV) · 12", "persian/fa_qa.csv", "prompt_completion"),
    ("🇮🇷 فارسی · متن خام برای پیش‌آموزش (text) · 8", "persian/fa_corpus.jsonl", "text"),
    ("🌐 Mixed EN/FA chat (messages) · 8", "sample_chat.jsonl", "messages"),
]


def sample_dataset_choices() -> list[tuple[str, str]]:
    """Dropdown choices for bundled samples that exist on disk (value = 'abs_path|format')."""
    out = []
    for label, rel, fmt in SAMPLE_DATASETS:
        path = SAMPLE_DATA_DIR / rel
        if path.exists():
            out.append((label, f"{path}|{fmt}"))
    return out


def apply_preset(cfg: FinetuneConfig, preset: str) -> FinetuneConfig:
    patch = PRESETS.get(preset)
    if not patch:
        return cfg
    data = cfg.to_dict()
    for section, values in patch.items():
        data[section].update(copy.deepcopy(values))
    return FinetuneConfig.from_dict(data)


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #


def model_info_markdown(model_id: str) -> str:
    info = get_model_info(model_id)
    if info:
        md = f"**{info.family}** · **{info.params_b:g}B** parameters · chat template `{info.chat_template}`"
        if info.notes:
            md += f" · _{info.notes}_"
        return md
    params = guess_params_b(model_id)
    guess = f"~{params:g}B parameters (guessed from name)" if params else "unknown size"
    return f"Custom model · {guess} · suggested template `{guess_chat_template(model_id)}`"


def vram_markdown(model_id: str, quantization: str, lora_r: int, batch_size: int, max_seq_length: int,
                  optim: str, gradient_checkpointing: str, available_gb: float | None = None) -> str:
    params = guess_params_b(model_id or "")
    if not params:
        return "VRAM estimate unavailable - model size unknown."
    est = estimate_vram_gb(params, quantization, int(lora_r or 16), int(batch_size or 1),
                           int(max_seq_length or 2048), optim, gradient_checkpointing)
    fit = ""
    if available_gb:
        if est["total"] <= available_gb * 0.95:
            fit = f" · ✅ fits your GPU ({available_gb:g} GB)"
        else:
            fit = f" · ❌ likely exceeds your GPU ({available_gb:g} GB)"
    return (
        f"### ≈ {est['total']} GB VRAM{fit}\n"
        f"weights {est['weights']} GB · trainable+optimizer {est['trainable']} GB · "
        f"activations {est['activations']} GB · overhead {est['overhead']} GB  \n"
        f"Recommended: **{recommend_gpu(est['total'])}** _(rough estimate)_"
    )


# --------------------------------------------------------------------------- #
# Monitoring helpers
# --------------------------------------------------------------------------- #

STATE_BADGES = {
    "idle": ("⚪", "Idle"),
    "starting": ("🟡", "Starting"),
    "loading_model": ("🟡", "Loading model"),
    "preparing_data": ("🟡", "Preparing data"),
    "training": ("🟢", "Training"),
    "evaluating": ("🔵", "Evaluating"),
    "stopping": ("🟠", "Stopping"),
    "saving": ("🔵", "Saving"),
    "exporting": ("🔵", "Exporting"),
    "completed": ("✅", "Completed"),
    "stopped": ("⏹️", "Stopped"),
    "failed": ("❌", "Failed"),
}


def status_markdown(status: dict[str, Any]) -> str:
    state = status.get("state", "idle")
    icon, label = STATE_BADGES.get(state, ("⚪", state))
    if state == "idle":
        return f"## {icon} {label}\nConfigure your run and press **Start training**."
    lines = [f"## {icon} {label}" + (f" · {status['kind']}" if status.get("kind") else "")]
    if status.get("message"):
        lines.append(f"{status['message']}")
    details = []
    if status.get("run_dir"):
        details.append(f"📁 `{status['run_dir']}`")
    if status.get("gpu"):
        details.append(f"🖥️ {status['gpu']} ({status.get('gpu_total_gb', '?')} GB)")
    if status.get("trainable_params"):
        tp, tot = status["trainable_params"], max(status.get("total_params") or 1, 1)
        details.append(f"🧮 trainable {tp:,} / {tot:,} ({100 * tp / tot:.2f}%)")
    ds = status.get("dataset")
    if isinstance(ds, dict):
        details.append(f"📚 {ds.get('format')} · train {ds.get('train_rows')} · eval {ds.get('eval_rows')}")
    if details:
        lines.append("  \n".join(details))
    if state == "failed" and status.get("traceback"):
        tb = status["traceback"].strip().splitlines()[-12:]
        lines.append("```\n" + "\n".join(tb) + "\n```")
    return "\n\n".join(lines)


def progress_html(status: dict[str, Any]) -> str:
    step = int(status.get("step") or 0)
    total = int(status.get("max_steps") or 0)
    pct = 100.0 * step / total if total else 0.0
    if status.get("state") == "completed":
        pct = 100.0
    elapsed = status.get("elapsed")
    if elapsed is None and status.get("started_at") and status.get("state") not in (None, "idle"):
        elapsed = (status.get("updated_at") or time.time()) - status["started_at"]
    eta = status.get("eta")
    epoch = status.get("epoch") or 0.0
    return f"""
<div class="fts-progress">
  <div class="fts-progress-bar"><div class="fts-progress-fill" style="width:{pct:.1f}%"></div></div>
  <div class="fts-progress-meta">
    <span>Step <b>{step}</b> / {total or '?'} ({pct:.1f}%)</span>
    <span>Epoch <b>{epoch:.2f}</b></span>
    <span>Elapsed <b>{format_duration(elapsed)}</b></span>
    <span>ETA <b>{format_duration(eta) if status.get('state') == 'training' else '-'}</b></span>
  </div>
</div>"""


def metrics_frames(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build (loss_df, lr_df, latest) for plotting."""
    loss_rows, lr_rows = [], []
    latest: dict[str, Any] = {}
    for r in records:
        step = r.get("step")
        if step is None:
            continue
        if "loss" in r:
            loss_rows.append({"step": step, "value": float(r["loss"]), "series": "train loss"})
            latest["loss"] = r["loss"]
        if "eval_loss" in r:
            loss_rows.append({"step": step, "value": float(r["eval_loss"]), "series": "eval loss"})
            latest["eval_loss"] = r["eval_loss"]
        if "learning_rate" in r:
            lr_rows.append({"step": step, "value": float(r["learning_rate"]), "series": "learning rate"})
            latest["learning_rate"] = r["learning_rate"]
        if "grad_norm" in r:
            latest["grad_norm"] = r["grad_norm"]
        if "gpu_mem_gb" in r:
            latest["gpu_mem_gb"] = r["gpu_mem_gb"]
        if "train_runtime" in r:
            latest["train_runtime"] = r["train_runtime"]
    cols = ["step", "value", "series"]
    return pd.DataFrame(loss_rows, columns=cols), pd.DataFrame(lr_rows, columns=cols), latest


def latest_metrics_markdown(latest: dict[str, Any]) -> str:
    if not latest:
        return ""
    parts = []
    labels = [("loss", "Train loss", "{:.4f}"), ("eval_loss", "Eval loss", "{:.4f}"),
              ("learning_rate", "LR", "{:.2e}"), ("grad_norm", "Grad norm", "{:.3f}"),
              ("gpu_mem_gb", "Peak VRAM", "{:.2f} GB")]
    for key, label, fmt in labels:
        if key in latest:
            try:
                parts.append(f"**{label}** {fmt.format(float(latest[key]))}")
            except (TypeError, ValueError):
                continue
    return " · ".join(parts)


def runs_table(output_root: str) -> pd.DataFrame:
    rows = []
    for run in list_runs(output_root):
        status = read_status(run)
        records = read_metrics(run)
        losses = [r["loss"] for r in records if "loss" in r]
        evals = [r["eval_loss"] for r in records if "eval_loss" in r]
        cfg = _safe_load_cfg(run)
        rows.append({
            "run": run.name,
            "state": status.get("state", "?"),
            "model": cfg.model.model_name if cfg else "?",
            "quant": cfg.model.quantization if cfg else "?",
            "steps": f"{status.get('step', 0)}/{status.get('max_steps', '?')}",
            "final loss": round(losses[-1], 4) if losses else None,
            "eval loss": round(evals[-1], 4) if evals else None,
            "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(status.get("started_at", run.stat().st_mtime))),
        })
    cols = ["run", "state", "model", "quant", "steps", "final loss", "eval loss", "started"]
    return pd.DataFrame(rows, columns=cols)


def _safe_load_cfg(run: Path) -> FinetuneConfig | None:
    try:
        return FinetuneConfig.load(run / "config.yaml")
    except Exception:  # noqa: BLE001
        return None


def model_choices(output_root: str) -> list[str]:
    """Every loadable trained model directory across runs."""
    out = []
    for run in list_runs(output_root):
        out.extend(str(p) for p in list_checkpoints(run))
    return out
