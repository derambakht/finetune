"""Gradio web UI for Unsloth FineTune Studio."""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .. import __version__
from ..config import (
    DATASET_FORMATS,
    DEFAULT_TARGET_MODULES,
    DTYPE_CHOICES,
    EXPORT_METHODS,
    GGUF_QUANT_METHODS,
    GRADIENT_CHECKPOINTING,
    LORA_BIAS,
    LR_SCHEDULERS,
    OPTIMIZERS,
    QUANTIZATION_CHOICES,
    REPORT_TO,
    FinetuneConfig,
    validate_config,
)
from ..data import preview_dataset
from ..models import CHAT_TEMPLATES, MODEL_CATALOG, guess_chat_template
from ..reporting import list_checkpoints, list_runs, read_metrics, read_status, tail_file
from ..runner import JobManager
from ..system import system_markdown, total_gpu_memory_gb
from . import logic

EXTRA_TARGET_MODULES = ["embed_tokens", "lm_head"]

POPULAR_DATASETS = [
    ("yahma/alpaca-cleaned · Alpaca, 52k", "yahma/alpaca-cleaned|train|"),
    ("mlabonne/FineTome-100k · ShareGPT, 100k", "mlabonne/FineTome-100k|train|"),
    ("HuggingFaceH4/ultrachat_200k · messages", "HuggingFaceH4/ultrachat_200k|train_sft|"),
    ("databricks/databricks-dolly-15k · custom", "databricks/databricks-dolly-15k|train|"),
    ("openai/gsm8k · math Q/A", "openai/gsm8k|train|main"),
    ("roneneldan/TinyStories · raw text", "roneneldan/TinyStories|train|"),
]

CSS = """
.fts-header {display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding: 18px 22px; border-radius: 14px; margin-bottom: 8px;
  background: linear-gradient(120deg, #0f766e 0%, #4338ca 55%, #7c3aed 100%); color: white;}
.fts-header h1 {margin:0; font-size: 1.6rem; color:white;}
.fts-header p {margin:4px 0 0 0; opacity:.9; color:white;}
.fts-badge {background: rgba(255,255,255,.18); padding: 4px 10px; border-radius: 999px; font-size: .85rem;}
.fts-progress-bar {height: 14px; background: var(--block-border-color, #e5e7eb); border-radius: 999px; overflow:hidden;}
.fts-progress-fill {height: 100%; background: linear-gradient(90deg,#10b981,#6366f1); transition: width .6s ease;}
.fts-progress-meta {display:flex; flex-wrap:wrap; gap: 18px; margin-top: 8px; font-size: .92rem;}
.fts-log textarea {font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important; font-size: 12px !important;}
footer {display:none !important;}
"""


GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6


def _style_kwargs() -> dict:
    """Theme + CSS (Blocks() kwargs in Gradio 4/5, launch() kwargs in Gradio 6+)."""
    return {"theme": gr.themes.Soft(primary_hue="indigo"), "css": CSS}


def _chatbot(**kwargs):
    """Chatbot in 'messages' format across Gradio 4/5/6."""
    try:
        return gr.Chatbot(type="messages", **kwargs)
    except TypeError:
        return gr.Chatbot(**kwargs)


def _line_plot(title: str, y_title: str):
    return gr.LinePlot(
        value=pd.DataFrame({"step": [], "value": [], "series": []}),
        x="step",
        y="value",
        color="series",
        title=title,
        x_title="step",
        y_title=y_title,
        height=280,
    )


def build_app(output_root: str = "outputs") -> gr.Blocks:
    manager = JobManager()
    chat_engine_holder: dict[str, Any] = {}
    gpu_total = total_gpu_memory_gb()
    defaults = FinetuneConfig()
    defaults.training.output_root = output_root
    comps: dict[tuple[str, str], gr.components.Component] = {}

    def reg(section: str, name: str, component):
        comps[(section, name)] = component
        return component

    style = {} if GRADIO_6 else _style_kwargs()
    with gr.Blocks(title="Unsloth FineTune Studio", **style) as app:
        gr.HTML(
            f"""<div class="fts-header"><div><h1>🦥 Unsloth FineTune Studio</h1>
            <p>Fine-tune LLMs 2x faster with 70% less VRAM · QLoRA / LoRA / Full · GGUF export · live monitoring</p>
            <p dir="rtl">استودیوی فاین‌تیون مدل‌های زبانی با Unsloth — مدل، کوانتیزه‌سازی، دیتاست و هایپرپارامترها را انتخاب کنید و آموزش را زنده دنبال کنید.</p>
            </div><span class="fts-badge">v{__version__}</span></div>"""
        )

        with gr.Tabs():
            # ================================================================ MODEL
            with gr.Tab("🧠 Model & Quantization"):
                with gr.Row():
                    with gr.Column(scale=3):
                        reg("model", "model_name", gr.Dropdown(
                            choices=[(m.label, m.model_id) for m in MODEL_CATALOG],
                            value=defaults.model.model_name, allow_custom_value=True,
                            label="Base model",
                            info="Pick from the catalog or type any Hugging Face id / local path.",
                        ))
                        model_info = gr.Markdown(logic.model_info_markdown(defaults.model.model_name))
                        reg("model", "quantization", gr.Radio(
                            choices=[(v, k) for k, v in QUANTIZATION_CHOICES.items()],
                            value=defaults.model.quantization, label="Training precision / quantization",
                        ))
                        with gr.Row():
                            reg("model", "max_seq_length", gr.Slider(
                                256, 32768, value=defaults.model.max_seq_length, step=256,
                                label="Max sequence length",
                                info="Longer = more VRAM. Unsloth handles RoPE scaling automatically.",
                            ))
                            reg("model", "dtype", gr.Dropdown(DTYPE_CHOICES, value=defaults.model.dtype,
                                                              label="Compute dtype",
                                                              info="auto = bf16 on Ampere+, else fp16"))
                        with gr.Row():
                            reg("model", "chat_template", gr.Dropdown(
                                CHAT_TEMPLATES, value=guess_chat_template(defaults.model.model_name),
                                label="Chat template",
                                info="auto = keep the tokenizer's template. Auto-selected when the model changes.",
                            ))
                            reg("model", "trust_remote_code", gr.Checkbox(value=False, label="Trust remote code"))
                        reg("model", "hf_token", gr.Textbox(
                            label="Hugging Face token (optional)", type="password", placeholder="hf_...",
                            info="Needed for gated models (Llama, Gemma) and pushing to the Hub. Never saved to disk.",
                        ))
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Estimated training VRAM")
                        vram_md = gr.Markdown()
                        gr.Markdown("### 🖥️ System")
                        sys_md = gr.Markdown(system_markdown())
                        refresh_sys = gr.Button("🔄 Refresh system info", size="sm")
                        gr.Markdown(
                            "**Tips**\n"
                            "- **QLoRA 4-bit** fits 8B models on ~8-10 GB GPUs.\n"
                            "- **16-bit LoRA** gives slightly better quality with ~2-3x the VRAM.\n"
                            "- **Full fine-tuning** updates every weight - use small models or large GPUs.\n"
                            "- Instruct models + their chat template are the easiest starting point."
                        )

            # ================================================================ DATASET
            with gr.Tab("📚 Dataset"):
                with gr.Row():
                    with gr.Column(scale=3):
                        reg("dataset", "source", gr.Radio(
                            [("🤗 Hugging Face Hub", "huggingface"), ("📁 Local file", "local")],
                            value=defaults.dataset.source, label="Source",
                        ))
                        popular = gr.Dropdown(POPULAR_DATASETS, value=None, label="Popular datasets (quick fill)")
                        reg("dataset", "path", gr.Textbox(
                            value=defaults.dataset.path, label="Dataset id or file path",
                            info="e.g. yahma/alpaca-cleaned, or /data/train.jsonl (json, jsonl, csv, tsv, parquet, txt)",
                        ))
                        upload = gr.File(label="…or upload a dataset file",
                                         file_types=[".json", ".jsonl", ".csv", ".tsv", ".parquet", ".txt"],
                                         type="filepath")
                        with gr.Row():
                            reg("dataset", "subset", gr.Textbox(value="", label="Subset / config name"))
                            reg("dataset", "split", gr.Textbox(value="train", label="Train split"))
                            reg("dataset", "eval_split", gr.Textbox(value="", label="Eval split (optional)"))
                        reg("dataset", "format", gr.Dropdown(
                            [(v, k) for k, v in DATASET_FORMATS.items()], value=defaults.dataset.format,
                            label="Dataset format",
                        ))
                        with gr.Accordion("Column mapping (custom / prompt-completion / text formats)", open=False):
                            with gr.Row():
                                reg("dataset", "prompt_column", gr.Textbox(value=defaults.dataset.prompt_column,
                                                                           label="Prompt column"))
                                reg("dataset", "response_column", gr.Textbox(value=defaults.dataset.response_column,
                                                                             label="Response column"))
                            with gr.Row():
                                reg("dataset", "system_column", gr.Textbox(value="", label="System column (optional)"))
                                reg("dataset", "text_column", gr.Textbox(value="text", label="Text column (raw text)"))
                        reg("dataset", "system_prompt", gr.Textbox(
                            value="", lines=2, label="System prompt (optional)",
                            info="Prepended to every conversation that has no system message.",
                        ))
                        with gr.Row():
                            reg("dataset", "max_samples", gr.Number(value=0, precision=0,
                                                                    label="Max samples (0 = all)"))
                            reg("dataset", "eval_ratio", gr.Slider(0.0, 0.3, value=0.0, step=0.01,
                                                                   label="Eval hold-out ratio"))
                            reg("dataset", "num_proc", gr.Number(value=2, precision=0, label="Preprocessing workers"))
                            reg("dataset", "shuffle", gr.Checkbox(value=True, label="Shuffle"))
                        preview_btn = gr.Button("🔍 Load & preview dataset", variant="secondary")
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "### Supported formats\n"
                            "| Format | Expected columns |\n|---|---|\n"
                            "| Alpaca | `instruction`, `input`, `output` |\n"
                            "| ShareGPT | `conversations: [{from, value}]` |\n"
                            "| Messages | `messages: [{role, content}]` |\n"
                            "| Prompt/Completion | `prompt`/`question` + `completion`/`answer` |\n"
                            "| Raw text | `text` |\n"
                            "| Custom | any two columns you map |\n\n"
                            "Everything is converted to chat messages and rendered with the model's chat "
                            "template, so the fine-tuned model works out of the box in Ollama, vLLM, etc."
                        )
                preview_info = gr.Markdown()
                preview_df = gr.Dataframe(label="Raw rows", wrap=True, interactive=False)
                preview_text = gr.Textbox(label="Formatted training text (what the model sees)", lines=16,
                                          elem_classes=["fts-log"])

            # ================================================================ LORA
            with gr.Tab("🔧 LoRA / PEFT"):
                full_note = gr.Markdown(visible=False)
                with gr.Row():
                    reg("lora", "r", gr.Dropdown([4, 8, 16, 32, 64, 128, 256], value=16, label="Rank (r)",
                                                 allow_custom_value=True,
                                                 info="Capacity of the adapter. 16-32 is a good default."))
                    reg("lora", "alpha", gr.Dropdown([4, 8, 16, 32, 64, 128, 256, 512], value=16,
                                                     allow_custom_value=True, label="Alpha",
                                                     info="Scaling. Common: alpha = r or 2·r."))
                    reg("lora", "dropout", gr.Slider(0.0, 0.3, value=0.0, step=0.01, label="Dropout",
                                                     info="0 is optimised by Unsloth"))
                reg("lora", "target_modules", gr.CheckboxGroup(
                    DEFAULT_TARGET_MODULES + EXTRA_TARGET_MODULES, value=DEFAULT_TARGET_MODULES,
                    label="Target modules",
                    info="Attention + MLP projections recommended. embed_tokens/lm_head for new languages / pre-training.",
                ))
                with gr.Row():
                    reg("lora", "bias", gr.Dropdown(LORA_BIAS, value="none", label="Bias"))
                    reg("lora", "gradient_checkpointing", gr.Dropdown(
                        GRADIENT_CHECKPOINTING, value="unsloth", label="Gradient checkpointing",
                        info="'unsloth' = 30% less VRAM, supports very long contexts"))
                    reg("lora", "use_rslora", gr.Checkbox(value=False, label="Rank-stabilized LoRA (rsLoRA)"))
                    reg("lora", "use_loftq", gr.Checkbox(value=False, label="LoftQ initialisation"))

            # ================================================================ TRAINING
            with gr.Tab("⚙️ Training"):
                with gr.Row():
                    preset = gr.Dropdown(list(logic.PRESETS), value=None, label="Presets", scale=3)
                    apply_preset_btn = gr.Button("Apply preset", scale=1)
                with gr.Row():
                    reg("training", "run_name", gr.Textbox(value=defaults.training.run_name, label="Run name"))
                    reg("training", "output_root", gr.Textbox(value=output_root, label="Output directory"))
                    reg("training", "seed", gr.Number(value=3407, precision=0, label="Seed"))
                gr.Markdown("#### Duration & batch")
                with gr.Row():
                    reg("training", "num_train_epochs", gr.Number(value=1, label="Epochs"))
                    reg("training", "max_steps", gr.Number(value=60, precision=0, label="Max steps",
                                                           info="> 0 overrides epochs"))
                    reg("training", "per_device_train_batch_size", gr.Slider(1, 64, value=2, step=1,
                                                                             label="Batch size / device"))
                    reg("training", "gradient_accumulation_steps", gr.Slider(1, 64, value=4, step=1,
                                                                             label="Gradient accumulation"))
                eff_batch = gr.Markdown()
                gr.Markdown("#### Optimisation")
                with gr.Row():
                    reg("training", "learning_rate", gr.Number(value=2e-4, label="Learning rate",
                                                               info="LoRA: 1e-4 - 3e-4 · Full: 1e-5 - 5e-5"))
                    reg("training", "lr_scheduler_type", gr.Dropdown(LR_SCHEDULERS, value="linear",
                                                                     label="LR scheduler"))
                    reg("training", "optim", gr.Dropdown(OPTIMIZERS, value="adamw_8bit", label="Optimizer"))
                with gr.Row():
                    reg("training", "warmup_steps", gr.Number(value=5, precision=0, label="Warmup steps"))
                    reg("training", "warmup_ratio", gr.Slider(0.0, 0.3, value=0.0, step=0.01,
                                                              label="Warmup ratio (if steps = 0)"))
                    reg("training", "weight_decay", gr.Number(value=0.01, label="Weight decay"))
                    reg("training", "max_grad_norm", gr.Number(value=1.0, label="Max grad norm"))
                gr.Markdown("#### Data handling")
                with gr.Row():
                    reg("training", "train_on_responses_only", gr.Checkbox(
                        value=True, label="Train on responses only",
                        info="Mask user prompts from the loss (usually +quality)."))
                    reg("training", "packing", gr.Checkbox(value=False, label="Sequence packing",
                                                           info="Faster for short samples"))
                    reg("training", "neftune_noise_alpha", gr.Number(value=0, label="NEFTune noise α (0 = off)"))
                with gr.Accordion("Response masking markers (leave empty = auto from chat template)", open=False):
                    with gr.Row():
                        reg("training", "instruction_part", gr.Textbox(value="", label="Instruction part"))
                        reg("training", "response_part", gr.Textbox(value="", label="Response part"))
                gr.Markdown("#### Logging, checkpoints & evaluation")
                with gr.Row():
                    reg("training", "logging_steps", gr.Number(value=1, precision=0, label="Logging steps"))
                    reg("training", "save_steps", gr.Number(value=100, precision=0, label="Save every N steps"))
                    reg("training", "save_total_limit", gr.Number(value=2, precision=0, label="Keep last N checkpoints"))
                    reg("training", "eval_steps", gr.Number(value=50, precision=0, label="Eval every N steps"))
                with gr.Row():
                    reg("training", "report_to", gr.Dropdown(REPORT_TO, value="none", label="Report to"))
                    reg("training", "resume_from_checkpoint", gr.Textbox(
                        value="", label="Resume from checkpoint",
                        info="Path to a checkpoint-* dir, or 'latest' (within the new run dir)."))

            # ================================================================ EXPORT
            with gr.Tab("📦 Export"):
                gr.Markdown("### Export settings")
                with gr.Row():
                    reg("export", "methods", gr.CheckboxGroup(
                        [(v, k) for k, v in EXPORT_METHODS.items()], value=["lora"], label="Export formats"))
                    reg("export", "gguf_quantizations", gr.CheckboxGroup(
                        [(v, k) for k, v in GGUF_QUANT_METHODS.items()], value=["q4_k_m"],
                        label="GGUF quantization methods"))
                with gr.Row():
                    reg("export", "push_to_hub", gr.Checkbox(value=False, label="Push to Hugging Face Hub"))
                    reg("export", "hub_repo_id", gr.Textbox(value="", label="Hub repo id",
                                                            placeholder="username/my-model"))
                    reg("export", "hub_private", gr.Checkbox(value=True, label="Private repo"))
                reg("export", "export_after_training", gr.Checkbox(
                    value=False, label="Automatically export with these settings when training finishes"))
                gr.Markdown("---\n### Export an existing run")
                with gr.Row():
                    export_source = gr.Dropdown([], label="Trained model / checkpoint", allow_custom_value=True,
                                                scale=4)
                    refresh_models_btn = gr.Button("🔄", scale=0, min_width=60)
                export_out = gr.Textbox(value="", label="Output directory (empty = <run>/export)")
                export_btn = gr.Button("📦 Start export", variant="primary")
                export_msg = gr.Markdown()

            # ================================================================ TRAIN & MONITOR
            with gr.Tab("🚀 Train & Monitor"):
                with gr.Row():
                    validate_btn = gr.Button("✔️ Validate config")
                    start_btn = gr.Button("🚀 Start training", variant="primary")
                    stop_btn = gr.Button("⏹️ Graceful stop", variant="secondary")
                    kill_btn = gr.Button("☠️ Force kill", variant="stop")
                validation_md = gr.Markdown()
                status_md = gr.Markdown(logic.status_markdown({"state": "idle"}))
                progress = gr.HTML(logic.progress_html({}))
                latest_md = gr.Markdown()
                with gr.Row():
                    loss_plot = _line_plot("Loss", "loss")
                    lr_plot = _line_plot("Learning rate", "lr")
                log_box = gr.Textbox(label="Live log", lines=18, max_lines=18, autoscroll=True,
                                     elem_classes=["fts-log"], interactive=False)
                timer = gr.Timer(2.0)

            # ================================================================ PLAYGROUND
            with gr.Tab("💬 Playground"):
                with gr.Row():
                    chat_model = gr.Dropdown([], allow_custom_value=True, scale=4,
                                             label="Model to load (trained output, checkpoint or any HF id)")
                    chat_refresh = gr.Button("🔄", scale=0, min_width=60)
                with gr.Row():
                    chat_quant = gr.Dropdown(["4bit", "8bit", "16bit"], value="4bit", label="Load precision")
                    chat_template = gr.Dropdown(CHAT_TEMPLATES, value="auto", label="Chat template")
                    chat_seq = gr.Number(value=2048, precision=0, label="Max seq length")
                with gr.Row():
                    load_btn = gr.Button("⬆️ Load model", variant="primary")
                    unload_btn = gr.Button("⬇️ Unload")
                chat_status = gr.Markdown("No model loaded.")
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = _chatbot(label="Chat", height=460)
                        with gr.Row():
                            chat_input = gr.Textbox(placeholder="Type a message and press Enter…", scale=5,
                                                    show_label=False)
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_btn = gr.Button("🧹 Clear chat")
                    with gr.Column(scale=1):
                        chat_system = gr.Textbox(label="System prompt", lines=3)
                        max_new = gr.Slider(16, 4096, value=512, step=16, label="Max new tokens")
                        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                        top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="Top-p")
                        top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                        rep_pen = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition penalty")

            # ================================================================ RUNS
            with gr.Tab("🗂️ Runs"):
                runs_refresh = gr.Button("🔄 Refresh")
                runs_df = gr.Dataframe(value=logic.runs_table(output_root), interactive=False, wrap=True)
                run_select = gr.Dropdown([r.name for r in list_runs(output_root)], label="Inspect run")
                run_status = gr.Markdown()
                with gr.Row():
                    run_loss_plot = _line_plot("Loss", "loss")
                    run_lr_plot = _line_plot("Learning rate", "lr")
                run_log = gr.Textbox(label="Log", lines=14, elem_classes=["fts-log"], interactive=False)

            # ================================================================ CONFIG
            with gr.Tab("📝 Config (YAML)"):
                gr.Markdown(
                    "Save and load complete configurations. Runs can also be launched headless:\n"
                    "```bash\nftstudio train --config my_run.yaml\n```"
                )
                with gr.Row():
                    to_yaml_btn = gr.Button("⬇️ Generate YAML from UI")
                    from_yaml_btn = gr.Button("⬆️ Apply YAML to UI", variant="primary")
                yaml_box = gr.Code(language="yaml", label="config.yaml", lines=30)
                with gr.Row():
                    yaml_upload = gr.File(label="Load YAML file", file_types=[".yaml", ".yml"], type="filepath")
                    yaml_download = gr.File(label="Download YAML", interactive=False)
                yaml_msg = gr.Markdown()

        # ==================================================================== wiring
        field_inputs = [comps[k] for k in logic.FIELDS]

        def _cfg(*values) -> FinetuneConfig:
            return logic.values_to_config(values)

        def _to_updates(cfg: FinetuneConfig) -> list:
            return [gr.update() if v is None else gr.update(value=v)
                    for v in logic.config_to_values(cfg, keep_secrets=True)]

        # --- model
        def on_model_change(model_id):
            return logic.model_info_markdown(model_id or ""), gr.update(value=guess_chat_template(model_id or ""))

        comps[("model", "model_name")].change(on_model_change, comps[("model", "model_name")],
                                              [model_info, comps[("model", "chat_template")]])

        vram_inputs = [comps[("model", "model_name")], comps[("model", "quantization")], comps[("lora", "r")],
                       comps[("training", "per_device_train_batch_size")], comps[("model", "max_seq_length")],
                       comps[("training", "optim")], comps[("lora", "gradient_checkpointing")]]

        def on_vram(*args):
            try:
                return logic.vram_markdown(*args, available_gb=gpu_total)
            except (TypeError, ValueError):
                return "VRAM estimate unavailable."

        for c in vram_inputs:
            c.change(on_vram, vram_inputs, vram_md)
        app.load(on_vram, vram_inputs, vram_md)
        refresh_sys.click(lambda: system_markdown(), None, sys_md)

        def on_quant(q):
            full = q == "full"
            note = ("### 🧬 Full fine-tuning selected\nLoRA settings are ignored - all model weights are "
                    "trained. Use a low learning rate (≈2e-5).") if full else ""
            return gr.update(value=note, visible=full)

        comps[("model", "quantization")].change(on_quant, comps[("model", "quantization")], full_note)

        def on_eff_batch(bs, ga):
            try:
                return f"Effective batch size: **{int(bs) * int(ga)}** (batch × accumulation)"
            except (TypeError, ValueError):
                return ""

        eff_inputs = [comps[("training", "per_device_train_batch_size")],
                      comps[("training", "gradient_accumulation_steps")]]
        for c in eff_inputs:
            c.change(on_eff_batch, eff_inputs, eff_batch)
        app.load(on_eff_batch, eff_inputs, eff_batch)

        # --- dataset
        def on_popular(choice):
            if not choice:
                return gr.update(), gr.update(), gr.update(), gr.update()
            path, split, subset = choice.split("|")
            return gr.update(value="huggingface"), gr.update(value=path), gr.update(value=split), gr.update(value=subset)

        popular.change(on_popular, popular, [comps[("dataset", "source")], comps[("dataset", "path")],
                                             comps[("dataset", "split")], comps[("dataset", "subset")]])

        def on_upload(file_path):
            if not file_path:
                return gr.update(), gr.update()
            dest_dir = Path("data/uploads")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(file_path).name
            shutil.copy(file_path, dest)
            return gr.update(value="local"), gr.update(value=str(dest.resolve()))

        upload.upload(on_upload, upload, [comps[("dataset", "source")], comps[("dataset", "path")]])

        def on_preview(*values):
            cfg = _cfg(*values)
            tokenizer, note = None, ""
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.model_name, token=cfg.model.hf_token or None,
                    trust_remote_code=cfg.model.trust_remote_code,
                )
                if cfg.model.chat_template not in ("auto", "alpaca"):
                    note = (f"_Preview uses the tokenizer's native template; `{cfg.model.chat_template}` is "
                            "applied by Unsloth at training time._")
            except Exception as exc:  # noqa: BLE001
                note = f"_Tokenizer unavailable ({type(exc).__name__}); showing a plain rendering._"
            try:
                res = preview_dataset(cfg.dataset, tokenizer,
                                      "alpaca" if cfg.model.chat_template == "alpaca" else "auto",
                                      n=5, token=cfg.model.hf_token)
            except Exception as exc:  # noqa: BLE001
                return f"❌ **Could not load dataset:** `{type(exc).__name__}: {exc}`", None, traceback.format_exc()
            lengths = res["token_lengths"]
            len_md = ""
            if lengths:
                len_md = (f" · tokens in preview: min {min(lengths)} / avg {sum(lengths) // len(lengths)} / "
                          f"max {max(lengths)} (max_seq_length {cfg.model.max_seq_length})")
            info = (f"✅ **{res['num_rows']:,} rows** · detected format **{res['format']}** · columns "
                    f"`{', '.join(res['columns'])}`{len_md}  \n"
                    + (f"Resolved columns: `{res['resolved_columns']}`  \n" if res["resolved_columns"] else "")
                    + note)
            sep = "\n\n" + "═" * 80 + "\n\n"
            return info, res["dataframe"], sep.join(res["rendered"])

        preview_btn.click(on_preview, field_inputs, [preview_info, preview_df, preview_text])

        # --- presets
        def on_preset(name, *values):
            if not name:
                return _to_updates(_cfg(*values))
            return _to_updates(logic.apply_preset(_cfg(*values), name))

        apply_preset_btn.click(on_preset, [preset, *field_inputs], field_inputs)

        # --- training control
        def on_validate(*values):
            return validate_config(_cfg(*values)).to_markdown()

        validate_btn.click(on_validate, field_inputs, validation_md)

        def on_start(*values):
            cfg = _cfg(*values)
            res = validate_config(cfg)
            if not res.ok:
                return res.to_markdown()
            try:
                job = manager.start_training(cfg)
            except Exception as exc:  # noqa: BLE001
                return f"❌ {exc}"
            return res.to_markdown() + f"\n\n🚀 Training started in `{job.run_dir}`"

        start_btn.click(on_start, field_inputs, validation_md)
        stop_btn.click(lambda: manager.stop(force=False), None, validation_md)
        kill_btn.click(lambda: manager.stop(force=True), None, validation_md)

        plot_cache: dict[str, Any] = {"key": None}

        def on_tick():
            status = manager.state()
            if manager.job is None:
                return (logic.status_markdown(status), logic.progress_html(status), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False))
            run_dir = manager.job.run_dir
            records = read_metrics(run_dir)
            key = (str(run_dir), len(records))
            if key != plot_cache["key"]:
                plot_cache["key"] = key
                loss_df, lr_df, latest = logic.metrics_frames(records)
                loss_u, lr_u, latest_u = loss_df, lr_df, logic.latest_metrics_markdown(latest)
            else:
                loss_u, lr_u, latest_u = gr.update(), gr.update(), gr.update()
            running = status.get("running", False)
            return (logic.status_markdown(status), logic.progress_html(status), latest_u, loss_u, lr_u,
                    tail_file(manager.job.log_path), gr.update(interactive=not running),
                    gr.update(interactive=running))

        timer.tick(on_tick, None, [status_md, progress, latest_md, loss_plot, lr_plot, log_box, start_btn, stop_btn])

        # --- export
        def refresh_models(root):
            choices = logic.model_choices(root or output_root)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        refresh_models_btn.click(refresh_models, comps[("training", "output_root")], export_source)
        chat_refresh.click(refresh_models, comps[("training", "output_root")], chat_model)
        app.load(refresh_models, comps[("training", "output_root")], export_source)
        app.load(refresh_models, comps[("training", "output_root")], chat_model)

        def on_export(source, out, *values):
            cfg = _cfg(*values)
            if not source:
                return "❌ Select a trained model directory first."
            if not cfg.export.methods:
                return "❌ Select at least one export format."
            res = validate_config(cfg)
            export_errors = [e for e in res.errors if "export" in e.lower() or "gguf" in e.lower() or "hub" in e.lower()]
            if export_errors:
                return "\n".join(f"- ❌ {e}" for e in export_errors)
            src = Path(source)
            out_dir = Path(out) if out else (src.parent / "export" / src.name)
            try:
                job = manager.start_export(cfg, src, out_dir)
            except Exception as exc:  # noqa: BLE001
                return f"❌ {exc}"
            return f"📦 Export started → `{job.run_dir}`. Follow progress in **🚀 Train & Monitor**."

        export_btn.click(on_export, [export_source, export_out, *field_inputs], export_msg)

        # --- playground
        def get_engine():
            if "engine" not in chat_engine_holder:
                from ..inference import ChatEngine

                chat_engine_holder["engine"] = ChatEngine()
            return chat_engine_holder["engine"]

        def on_load(path, quant, template, seq, token):
            if not path:
                return "❌ Choose a model."
            if manager.busy:
                return "⚠️ A training/export job is using the GPU. Wait for it to finish before loading a model."
            try:
                return get_engine().load(path, max_seq_length=int(seq or 2048), quantization=quant,
                                         chat_template=template, hf_token=token or "")
            except Exception as exc:  # noqa: BLE001
                return f"❌ `{type(exc).__name__}: {exc}`"

        load_btn.click(on_load, [chat_model, chat_quant, chat_template, chat_seq, comps[("model", "hf_token")]],
                       chat_status)

        def on_unload():
            if "engine" in chat_engine_holder:
                chat_engine_holder["engine"].unload()
            return "Model unloaded."

        unload_btn.click(on_unload, None, chat_status)

        def on_chat(message, history, system, mx, temp, tp, tk, rp):
            history = list(history or [])
            if not message or not message.strip():
                yield history, ""
                return
            engine = chat_engine_holder.get("engine")
            history.append({"role": "user", "content": message})
            if engine is None or not engine.loaded:
                history.append({"role": "assistant", "content": "⚠️ Load a model first."})
                yield history, ""
                return
            messages = ([{"role": "system", "content": system}] if system and system.strip() else [])
            messages += [{"role": m["role"], "content": m["content"]} for m in history
                         if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
            history.append({"role": "assistant", "content": ""})
            try:
                for partial in engine.stream(messages, max_new_tokens=int(mx), temperature=float(temp),
                                             top_p=float(tp), top_k=int(tk), repetition_penalty=float(rp)):
                    history[-1]["content"] = partial
                    yield history, ""
            except Exception as exc:  # noqa: BLE001
                history[-1]["content"] = f"❌ {type(exc).__name__}: {exc}"
                yield history, ""

        chat_inputs = [chat_input, chatbot, chat_system, max_new, temperature, top_p, top_k, rep_pen]
        chat_input.submit(on_chat, chat_inputs, [chatbot, chat_input])
        send_btn.click(on_chat, chat_inputs, [chatbot, chat_input])
        clear_btn.click(lambda: [], None, chatbot)

        # --- runs
        def on_runs_refresh(root):
            root = root or output_root
            names = [r.name for r in list_runs(root)]
            return logic.runs_table(root), gr.update(choices=names)

        runs_refresh.click(on_runs_refresh, comps[("training", "output_root")], [runs_df, run_select])

        def on_run_select(name, root):
            if not name:
                return "", gr.update(), gr.update(), ""
            run_dir = Path(root or output_root) / name
            status = read_status(run_dir)
            status["run_dir"] = str(run_dir)
            loss_df, lr_df, latest = logic.metrics_frames(read_metrics(run_dir))
            ckpts = list_checkpoints(run_dir)
            md = logic.status_markdown(status) + "\n\n" + logic.latest_metrics_markdown(latest)
            if ckpts:
                md += "\n\n**Saved models:** " + ", ".join(f"`{c.name}`" for c in ckpts)
            return md, loss_df, lr_df, tail_file(run_dir / "train.log")

        run_select.change(on_run_select, [run_select, comps[("training", "output_root")]],
                          [run_status, run_loss_plot, run_lr_plot, run_log])

        # --- yaml
        def on_to_yaml(*values):
            cfg = _cfg(*values)
            text = cfg.to_yaml(redact_secrets=True)
            tmp = Path(tempfile.mkdtemp()) / f"{cfg.training.run_name}.yaml"
            tmp.write_text(text, encoding="utf-8")
            return text, str(tmp), "✅ YAML generated (HF token is never included)."

        to_yaml_btn.click(on_to_yaml, field_inputs, [yaml_box, yaml_download, yaml_msg])

        def on_from_yaml(text):
            try:
                cfg = FinetuneConfig.from_yaml(text or "")
            except Exception as exc:  # noqa: BLE001
                return [gr.update() for _ in field_inputs] + [f"❌ Invalid YAML: {exc}"]
            return _to_updates(cfg) + ["✅ Applied to UI.\n\n" + validate_config(cfg).to_markdown()]

        from_yaml_btn.click(on_from_yaml, yaml_box, [*field_inputs, yaml_msg])

        def on_yaml_upload(path):
            if not path:
                return gr.update()
            return Path(path).read_text(encoding="utf-8")

        yaml_upload.upload(on_yaml_upload, yaml_upload, yaml_box)

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False, output_root: str = "outputs") -> None:
    app = build_app(output_root=output_root)
    style = _style_kwargs() if GRADIO_6 else {}
    app.queue(default_concurrency_limit=4).launch(server_name=host, server_port=port, share=share,
                                                   show_error=True, **style)


if __name__ == "__main__":
    launch()
