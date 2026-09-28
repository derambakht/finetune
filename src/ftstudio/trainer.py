"""Training engine built on Unsloth + TRL's SFTTrainer.

This module is executed inside a dedicated subprocess (see :mod:`ftstudio.runner`)
so that GPU memory is fully released when a run ends and the UI stays responsive.

IMPORTANT: ``unsloth`` must be imported before ``transformers`` / ``trl`` / ``peft``
so that its patches are applied - all heavy imports therefore happen lazily inside
the functions of this module.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import time
import traceback
from pathlib import Path
from typing import Any

from .config import FinetuneConfig, validate_config
from .data import TEXT_FIELD, prepare_datasets
from .models import guess_chat_template, response_markers
from .reporting import RunReporter

log = logging.getLogger("ftstudio.trainer")

FINAL_DIR = "final_model"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def resolve_token(cfg: FinetuneConfig) -> str | None:
    import os

    return cfg.model.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name)


def _gc_value(mode: str):
    return {"unsloth": "unsloth", "true": True, "false": False}.get(mode, "unsloth")


def filter_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by ``target`` - makes us robust across TRL versions."""
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def load_model_and_tokenizer(cfg: FinetuneConfig, model_name: str | None = None, for_training: bool = True):
    import unsloth  # noqa: F401  (must be first)
    from unsloth import FastLanguageModel

    m = cfg.model
    kwargs: dict[str, Any] = dict(
        model_name=model_name or m.model_name,
        max_seq_length=m.max_seq_length,
        dtype=_torch_dtype(m.dtype),
        load_in_4bit=m.quantization == "4bit",
        load_in_8bit=m.quantization == "8bit",
        full_finetuning=m.quantization == "full" and for_training,
        token=resolve_token(cfg),
        trust_remote_code=m.trust_remote_code,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    return model, tokenizer


def apply_lora(model, cfg: FinetuneConfig):
    from unsloth import FastLanguageModel

    lo = cfg.lora
    kwargs: dict[str, Any] = dict(
        r=lo.r,
        target_modules=lo.target_modules,
        lora_alpha=lo.alpha,
        lora_dropout=lo.dropout,
        bias=lo.bias,
        use_gradient_checkpointing=_gc_value(lo.gradient_checkpointing),
        random_state=cfg.training.seed,
        use_rslora=lo.use_rslora,
        loftq_config=None,
        max_seq_length=cfg.model.max_seq_length,
    )
    if lo.use_loftq:
        from peft import LoftQConfig

        kwargs["init_lora_weights"] = "loftq"
        kwargs["loftq_config"] = LoftQConfig(loftq_bits=4)
    return FastLanguageModel.get_peft_model(model, **kwargs)


def apply_chat_template(tokenizer, template: str):
    if template in ("auto", "alpaca"):
        return tokenizer
    from unsloth.chat_templates import get_chat_template

    return get_chat_template(tokenizer, chat_template=template)


def count_parameters(model) -> tuple[int, int]:
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        # bitsandbytes 4-bit params are packed two per byte
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


def build_sft_config(cfg: FinetuneConfig, run_dir: Path, has_eval: bool):
    import torch
    from trl import SFTConfig

    t = cfg.training
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if cfg.model.dtype == "float16":
        bf16, fp16 = False, True
    elif cfg.model.dtype == "bfloat16":
        bf16, fp16 = True, False
    else:
        bf16, fp16 = bf16_ok, not bf16_ok

    kwargs: dict[str, Any] = dict(
        output_dir=str(run_dir),
        run_name=t.run_name,
        dataset_text_field=TEXT_FIELD,
        max_seq_length=cfg.model.max_seq_length,  # TRL < 0.20
        max_length=cfg.model.max_seq_length,  # TRL >= 0.20
        packing=t.packing,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps if t.max_steps > 0 else -1,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        warmup_steps=t.warmup_steps,
        optim=t.optim,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        seed=t.seed,
        logging_steps=t.logging_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        report_to=t.report_to if t.report_to != "none" else "none",
        bf16=bf16,
        fp16=fp16,
        dataset_num_proc=cfg.dataset.num_proc if cfg.dataset.num_proc > 1 else None,
    )
    if t.neftune_noise_alpha > 0:
        kwargs["neftune_noise_alpha"] = t.neftune_noise_alpha
    if t.report_to == "tensorboard":
        kwargs["logging_dir"] = str(run_dir / "tensorboard")
    if has_eval:
        kwargs["eval_strategy"] = "steps"  # transformers >= 4.41
        kwargs["evaluation_strategy"] = "steps"  # older transformers
        kwargs["eval_steps"] = t.eval_steps
    return SFTConfig(**filter_kwargs(SFTConfig, kwargs))


def build_trainer(model, tokenizer, train_ds, eval_ds, sft_config, callbacks):
    from trl import SFTTrainer

    kwargs: dict[str, Any] = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        callbacks=callbacks,
    )
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def make_callback(reporter: RunReporter):
    from transformers import TrainerCallback

    class StudioCallback(TrainerCallback):
        """Streams progress/metrics to the run directory and honours stop requests."""

        def __init__(self):
            self.t0 = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()
            reporter.state("training", "Training started", max_steps=state.max_steps, step=0)

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - (self.t0 or time.time())
            step = state.global_step
            eta = (elapsed / step) * (state.max_steps - step) if step else None
            reporter.update(step=step, max_steps=state.max_steps, epoch=state.epoch or 0.0, elapsed=elapsed, eta=eta)
            if reporter.stop_requested():
                reporter.state("stopping", "Stop requested - saving checkpoint...")
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {"step": state.global_step, "epoch": state.epoch}
            record.update(logs)
            try:
                import torch

                if torch.cuda.is_available():
                    record["gpu_mem_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
            except Exception:  # noqa: BLE001
                pass
            reporter.log_metrics(record)

    return StudioCallback()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_training(cfg: FinetuneConfig, run_dir: str | Path) -> dict[str, Any]:
    """Run a complete fine-tuning job. Returns a summary dict."""
    run_dir = Path(run_dir)
    reporter = RunReporter(run_dir)
    try:
        return _run(cfg, run_dir, reporter)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        log.error("Training failed:\n%s", tb)
        hint = _error_hint(exc)
        reporter.state("failed", f"{type(exc).__name__}: {exc}{hint}", traceback=tb)
        raise


def _run(cfg: FinetuneConfig, run_dir: Path, reporter: RunReporter) -> dict[str, Any]:
    validation = validate_config(cfg)
    if not validation.ok:
        raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))
    for w in validation.warnings:
        log.warning(w)

    # ------------------------------------------------------------ model
    reporter.state("loading_model", f"Loading {cfg.model.model_name} ({cfg.model.quantization})")
    log.info("Loading model %s with quantization=%s", cfg.model.model_name, cfg.model.quantization)
    import unsloth  # noqa: F401
    import torch

    model, tokenizer = load_model_and_tokenizer(cfg)

    template = cfg.model.chat_template
    tokenizer = apply_chat_template(tokenizer, template)

    if not cfg.is_full_finetune:
        reporter.state("loading_model", "Attaching LoRA adapters")
        model = apply_lora(model, cfg)

    trainable, total = count_parameters(model)
    log.info("Parameters: trainable=%s total=%s (%.3f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1))
    reporter.update(trainable_params=trainable, total_params=total)

    # ------------------------------------------------------------ data
    reporter.state("preparing_data", "Loading and formatting dataset")
    train_ds, eval_ds, data_info = prepare_datasets(
        cfg.dataset, tokenizer, template, seed=cfg.training.seed, token=resolve_token(cfg) or ""
    )
    log.info("Dataset: %s", json.dumps(data_info))
    reporter.update(dataset=data_info)
    if len(train_ds) == 0:
        raise ValueError("The training dataset is empty after formatting. Check the format / column mapping.")
    log.info("Sample training text:\n%s\n%s", train_ds[0][TEXT_FIELD][:1500], "-" * 60)

    # ------------------------------------------------------------ trainer
    sft_config = build_sft_config(cfg, run_dir, has_eval=eval_ds is not None and len(eval_ds) > 0)
    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, sft_config, [make_callback(reporter)])

    if cfg.training.train_on_responses_only and cfg.dataset.format != "text" and data_info["format"] != "text":
        trainer = _apply_response_masking(trainer, cfg, template)

    steps_per_epoch = math.ceil(len(train_ds) / cfg.effective_batch_size)
    planned = cfg.training.max_steps if cfg.training.max_steps > 0 else math.ceil(
        steps_per_epoch * cfg.training.num_train_epochs
    )
    reporter.update(max_steps=planned, steps_per_epoch=steps_per_epoch)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        reporter.update(gpu=props.name, gpu_total_gb=round(props.total_memory / 1024**3, 2),
                        gpu_start_reserved_gb=round(torch.cuda.max_memory_reserved() / 1024**3, 2))

    # ------------------------------------------------------------ train
    reporter.state("training", "Training", max_steps=planned)
    resume = cfg.training.resume_from_checkpoint.strip() or None
    if resume and resume.lower() == "latest":
        resume = True
    stats = trainer.train(resume_from_checkpoint=resume)
    stopped = reporter.stop_requested()

    metrics = dict(stats.metrics) if stats is not None else {}
    if eval_ds is not None and len(eval_ds) > 0 and not stopped:
        reporter.state("evaluating", "Final evaluation")
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)
        reporter.log_metrics({"step": trainer.state.global_step, **eval_metrics})

    # ------------------------------------------------------------ save
    reporter.state("saving", "Saving final model")
    final_dir = run_dir / FINAL_DIR
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "ftstudio_meta.json").write_text(
        json.dumps(
            {
                "base_model": cfg.model.model_name,
                "quantization": cfg.model.quantization,
                "chat_template": template,
                "max_seq_length": cfg.model.max_seq_length,
                "full_finetune": cfg.is_full_finetune,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Saved final model to %s", final_dir)

    summary = {
        "final_model": str(final_dir),
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in metrics.items()},
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2) if torch.cuda.is_available() else None,
        "dataset": data_info,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ export
    if cfg.export.export_after_training and not stopped:
        from .export import export_loaded_model

        reporter.state("exporting", "Exporting model")
        export_loaded_model(model, tokenizer, cfg, run_dir / "export", log_fn=log.info)

    if stopped:
        reporter.state("stopped", "Stopped by user - model saved", final_model=str(final_dir))
    else:
        reporter.state("completed", "Training completed", final_model=str(final_dir), summary=summary)
    return summary


def _apply_response_masking(trainer, cfg: FinetuneConfig, template: str):
    """Mask user/system tokens so the loss is computed on assistant answers only."""
    from unsloth.chat_templates import train_on_responses_only

    instruction_part = cfg.training.instruction_part
    response_part = cfg.training.response_part
    if not (instruction_part and response_part):
        guess = template if template != "auto" else guess_chat_template(cfg.model.model_name)
        markers = response_markers(guess)
        if markers is None:
            log.warning(
                "Could not determine chat markers for template %r - set instruction_part/response_part "
                "manually. Training on full sequences instead.",
                guess,
            )
            return trainer
        instruction_part, response_part = markers
    log.info("Train on responses only: instruction=%r response=%r", instruction_part, response_part)
    return train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)


def _error_hint(exc: Exception) -> str:
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda oom" in msg:
        return (
            "\n\nHint: out of GPU memory. Try 4-bit quantization, a smaller batch size with more "
            "gradient accumulation, a shorter max_seq_length, or a smaller model."
        )
    if "401" in msg or "gated" in msg or "access to model" in msg:
        return "\n\nHint: this model is gated/private. Provide a Hugging Face token with access."
    if "no module named 'unsloth'" in msg:
        return "\n\nHint: install the training stack: pip install unsloth unsloth_zoo"
    return ""
