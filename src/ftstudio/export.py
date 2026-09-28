"""Model export: LoRA adapters, merged 16/4-bit weights, GGUF and Hugging Face Hub."""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import FinetuneConfig
from .reporting import RunReporter

log = logging.getLogger("ftstudio.export")


def read_meta(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "ftstudio_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def export_loaded_model(
    model,
    tokenizer,
    cfg: FinetuneConfig,
    out_dir: str | Path,
    log_fn: Callable[[str], None] = log.info,
) -> dict[str, str]:
    """Export an already-loaded (PEFT or full) model with the methods in ``cfg.export``."""
    ex = cfg.export
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from .trainer import resolve_token

    token = resolve_token(cfg)
    results: dict[str, str] = {}
    is_peft = hasattr(model, "peft_config")

    for method in ex.methods:
        if method == "lora":
            if not is_peft:
                log_fn("Skipping LoRA export: model has no adapters (full fine-tune).")
                continue
            target = out_dir / "lora"
            log_fn(f"Saving LoRA adapters -> {target}")
            model.save_pretrained(str(target))
            tokenizer.save_pretrained(str(target))
            results["lora"] = str(target)
            if ex.push_to_hub:
                repo = f"{ex.hub_repo_id}-lora"
                log_fn(f"Pushing LoRA adapters -> hf.co/{repo}")
                model.push_to_hub(repo, token=token, private=ex.hub_private)
                tokenizer.push_to_hub(repo, token=token, private=ex.hub_private)

        elif method in ("merged_16bit", "merged_4bit"):
            save_method = "merged_16bit" if method == "merged_16bit" else "merged_4bit_forced"
            target = out_dir / method
            log_fn(f"Saving {method} model -> {target} (this can take a while)")
            model.save_pretrained_merged(str(target), tokenizer, save_method=save_method)
            results[method] = str(target)
            if ex.push_to_hub:
                repo = ex.hub_repo_id if method == "merged_16bit" else f"{ex.hub_repo_id}-bnb-4bit"
                log_fn(f"Pushing {method} -> hf.co/{repo}")
                model.push_to_hub_merged(repo, tokenizer, save_method=save_method, token=token,
                                         private=ex.hub_private)

        elif method == "gguf":
            target = out_dir / "gguf"
            quants = list(ex.gguf_quantizations)
            log_fn(f"Converting to GGUF {quants} -> {target} (builds llama.cpp on first use)")
            model.save_pretrained_gguf(str(target), tokenizer, quantization_method=quants)
            results["gguf"] = str(target)
            if ex.push_to_hub:
                repo = f"{ex.hub_repo_id}-GGUF"
                log_fn(f"Pushing GGUF -> hf.co/{repo}")
                model.push_to_hub_gguf(repo, tokenizer, quantization_method=quants, token=token,
                                       private=ex.hub_private)
        else:
            raise ValueError(f"Unknown export method: {method}")

    (out_dir / "export_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log_fn(f"Export finished: {json.dumps(results)}")
    return results


def run_export(cfg: FinetuneConfig, source_dir: str | Path, out_dir: str | Path) -> dict[str, str]:
    """Load a trained model from ``source_dir`` and export it (used by the CLI / UI)."""
    out_dir = Path(out_dir)
    reporter = RunReporter(out_dir)
    try:
        from .trainer import apply_chat_template, load_model_and_tokenizer

        meta = read_meta(source_dir)
        if meta:
            cfg.model.max_seq_length = int(meta.get("max_seq_length", cfg.model.max_seq_length))
            # Merging is done from 16-bit weights when possible for best quality.
            if meta.get("full_finetune"):
                cfg.model.quantization = "16bit"
        reporter.state("loading_model", f"Loading {source_dir}")
        model, tokenizer = load_model_and_tokenizer(cfg, model_name=str(source_dir), for_training=False)
        template = meta.get("chat_template", cfg.model.chat_template)
        if template not in ("auto", "alpaca") and not getattr(tokenizer, "chat_template", None):
            tokenizer = apply_chat_template(tokenizer, template)

        reporter.state("exporting", f"Exporting: {', '.join(cfg.export.methods)}")
        results = export_loaded_model(model, tokenizer, cfg, out_dir, log_fn=log.info)
        reporter.state("completed", "Export completed", results=results)
        return results
    except Exception as exc:  # noqa: BLE001
        reporter.state("failed", f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
