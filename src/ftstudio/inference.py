"""Streaming chat inference with a fine-tuned (or base) model via Unsloth."""

from __future__ import annotations

import gc
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import FinetuneConfig
from .export import read_meta


class ChatEngine:
    """Holds a single model in GPU memory for interactive chat."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path: str | None = None
        self.template = "auto"
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, model_path: str, max_seq_length: int = 2048, quantization: str = "4bit",
             chat_template: str = "auto", hf_token: str = "") -> str:
        with self._lock:
            self.unload()
            from .trainer import apply_chat_template, load_model_and_tokenizer

            meta = read_meta(model_path) if Path(model_path).exists() else {}
            cfg = FinetuneConfig()
            cfg.model.max_seq_length = int(meta.get("max_seq_length", max_seq_length))
            cfg.model.quantization = "16bit" if quantization == "full" else quantization
            cfg.model.hf_token = hf_token
            template = chat_template if chat_template != "auto" else meta.get("chat_template", "auto")

            model, tokenizer = load_model_and_tokenizer(cfg, model_name=model_path, for_training=False)
            if template not in ("auto", "alpaca"):
                tokenizer = apply_chat_template(tokenizer, template)
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)
            self.model, self.tokenizer, self.model_path, self.template = model, tokenizer, model_path, template
            base = f" (base: {meta['base_model']})" if meta.get("base_model") else ""
            return f"✅ Loaded `{model_path}`{base} · template: `{template}`"

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_path = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        from .data import render_alpaca

        tok = self.tokenizer
        if self.template == "alpaca" or not getattr(tok, "chat_template", None):
            # Leave the last assistant slot open for generation.
            prompt = render_alpaca(messages + [{"role": "assistant", "content": ""}])
            return prompt
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def stream(self, messages: list[dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.7,
               top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.05) -> Iterator[str]:
        if not self.loaded:
            raise RuntimeError("No model loaded")
        from transformers import TextIteratorStreamer

        prompt = self.build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            use_cache=True,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            gen_kwargs.update(do_sample=False)

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        text = ""
        for chunk in streamer:
            text += chunk
            yield text
        thread.join()
