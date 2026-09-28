"""Dataset loading, format detection and conversion to training text.

Every supported input format is first normalised into a list of chat messages
(``[{"role": ..., "content": ...}]``) and then rendered with the model's chat
template into a single ``text`` field consumed by the SFT trainer.
Raw text datasets (continued pre-training) bypass the chat template.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .models import ALPACA_PROMPT

TEXT_FIELD = "text"

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}

_LOCAL_LOADERS = {
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".txt": "text",
    ".arrow": "arrow",
}

PROMPT_COLUMN_CANDIDATES = ["prompt", "question", "query", "input", "instruction", "user"]
RESPONSE_COLUMN_CANDIDATES = ["completion", "response", "answer", "output", "chosen", "assistant"]


# --------------------------------------------------------------------------- #
# Format detection & conversion (pure python, easy to unit-test)
# --------------------------------------------------------------------------- #


def detect_format(columns: Iterable[str], sample: dict[str, Any] | None = None) -> str:
    cols = {c.lower() for c in columns}
    if "messages" in cols:
        return "messages"
    if "conversations" in cols:
        conv = (sample or {}).get("conversations")
        if isinstance(conv, list) and conv and isinstance(conv[0], dict) and "role" in conv[0]:
            return "messages"
        return "sharegpt"
    if {"instruction", "output"} <= cols:
        return "alpaca"
    if any(c in cols for c in PROMPT_COLUMN_CANDIDATES) and any(
        c in cols for c in RESPONSE_COLUMN_CANDIDATES
    ):
        return "prompt_completion"
    if "text" in cols:
        return "text"
    return "custom"


def _find_column(columns: Iterable[str], candidates: list[str], preferred: str = "") -> str | None:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    if preferred and preferred in cols:
        return preferred
    if preferred and preferred.lower() in lower:
        return lower[preferred.lower()]
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def resolve_columns(columns: Iterable[str], cfg: DatasetConfig, fmt: str) -> dict[str, str | None]:
    """Resolve the concrete column names used for a format."""
    columns = list(columns)
    if fmt == "prompt_completion":
        return {
            "prompt": _find_column(columns, PROMPT_COLUMN_CANDIDATES, cfg.prompt_column),
            "response": _find_column(columns, RESPONSE_COLUMN_CANDIDATES, cfg.response_column),
            "system": cfg.system_column if cfg.system_column in columns else None,
        }
    if fmt == "custom":
        return {
            "prompt": cfg.prompt_column if cfg.prompt_column in columns else None,
            "response": cfg.response_column if cfg.response_column in columns else None,
            "system": cfg.system_column if cfg.system_column in columns else None,
        }
    if fmt == "text":
        return {"text": _find_column(columns, ["text", "content", "body"], cfg.text_column)}
    return {}


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalise_turns(turns: list[dict[str, Any]], role_key: str, content_key: str) -> list[dict[str, str]]:
    messages = []
    for turn in turns or []:
        if not isinstance(turn, dict):
            continue
        role = _ROLE_MAP.get(str(turn.get(role_key, "")).strip().lower())
        if role is None:
            continue
        content = turn.get(content_key)
        if isinstance(content, list):  # multimodal style [{"type": "text", "text": ...}]
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        messages.append({"role": role, "content": _as_str(content)})
    return messages


def row_to_messages(
    row: dict[str, Any],
    fmt: str,
    columns: dict[str, str | None] | None = None,
    system_prompt: str = "",
) -> list[dict[str, str]] | None:
    """Convert a single dataset row to chat messages. Returns ``None`` for unusable rows."""
    columns = columns or {}
    messages: list[dict[str, str]]

    if fmt == "alpaca":
        instruction = _as_str(row.get("instruction")).strip()
        extra = _as_str(row.get("input")).strip()
        output = _as_str(row.get("output")).strip()
        user = f"{instruction}\n\n{extra}" if extra else instruction
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
    elif fmt == "sharegpt":
        messages = _normalise_turns(row.get("conversations") or [], "from", "value")
    elif fmt == "messages":
        turns = row.get("messages") if "messages" in row else row.get("conversations")
        messages = _normalise_turns(turns or [], "role", "content")
    elif fmt in ("prompt_completion", "custom"):
        p_col, r_col = columns.get("prompt"), columns.get("response")
        if not p_col or not r_col:
            return None
        prompt, response = row.get(p_col), row.get(r_col)
        # TRL conversational prompt/completion: lists of messages
        if isinstance(prompt, list) and isinstance(response, list):
            messages = _normalise_turns(prompt, "role", "content") + _normalise_turns(response, "role", "content")
        else:
            messages = [
                {"role": "user", "content": _as_str(prompt).strip()},
                {"role": "assistant", "content": _as_str(response).strip()},
            ]
        s_col = columns.get("system")
        if s_col and _as_str(row.get(s_col)).strip():
            messages.insert(0, {"role": "system", "content": _as_str(row.get(s_col)).strip()})
    else:
        raise ValueError(f"Format '{fmt}' cannot be converted to messages")

    if system_prompt.strip() and not (messages and messages[0]["role"] == "system"):
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})

    if not any(m["role"] == "assistant" and m["content"].strip() for m in messages):
        return None
    if not any(m["role"] == "user" and m["content"].strip() for m in messages):
        return None
    return messages


def render_alpaca(messages: list[dict[str, str]], eos_token: str = "") -> str:
    """Render messages with the classic Alpaca prompt (for base models without a chat template)."""
    parts = []
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    pending_user = None
    for m in messages:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user is not None:
            instruction = f"{system}\n\n{pending_user}" if system and not parts else pending_user
            parts.append(ALPACA_PROMPT.format(instruction=instruction, response=m["content"]) + eos_token)
            pending_user = None
    return "\n\n".join(parts)


def render_messages(messages: list[dict[str, str]], tokenizer: Any = None, template: str = "auto") -> str:
    """Render chat messages to text using the tokenizer's chat template."""
    eos = getattr(tokenizer, "eos_token", "") or ""
    if template == "alpaca":
        return render_alpaca(messages, eos)
    if tokenizer is None:
        return render_plain(messages)
    if not getattr(tokenizer, "chat_template", None):
        # Base models without a chat template fall back to the Alpaca prompt.
        return render_alpaca(messages, eos)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Gemma/Llama templates add a BOS token which the tokenizer will add again.
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def render_plain(messages: list[dict[str, str]]) -> str:
    """Human-readable rendering used for previews when no tokenizer is available."""
    return "\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages)


# --------------------------------------------------------------------------- #
# Loading with 🤗 datasets
# --------------------------------------------------------------------------- #


def load_raw_dataset(cfg: DatasetConfig, split: str | None = None, token: str | None = None):
    """Load the raw dataset split described by ``cfg``."""
    from datasets import load_dataset

    split = split or cfg.split or "train"
    if cfg.source == "local":
        path = Path(cfg.path)
        if path.is_dir():
            files = [p for p in path.iterdir() if p.suffix.lower() in _LOCAL_LOADERS]
            if not files:
                raise FileNotFoundError(f"No supported data files in {path}")
            loader = _LOCAL_LOADERS[files[0].suffix.lower()]
            data_files = [str(p) for p in files if _LOCAL_LOADERS[p.suffix.lower()] == loader]
        else:
            loader = _LOCAL_LOADERS.get(path.suffix.lower())
            if loader is None:
                raise ValueError(
                    f"Unsupported file type '{path.suffix}'. Use one of: {', '.join(_LOCAL_LOADERS)}"
                )
            data_files = str(path)
        kwargs: dict[str, Any] = {}
        if path.suffix.lower() == ".tsv":
            kwargs["delimiter"] = "\t"
        # Local files have a single "train" split; the eval split is carved out later.
        return load_dataset(loader, data_files=data_files, split="train", **kwargs)

    return load_dataset(cfg.path, name=cfg.subset or None, split=split, token=token or None)


def effective_format(ds, cfg: DatasetConfig) -> str:
    if cfg.format != "auto":
        return cfg.format
    sample = ds[0] if len(ds) else {}
    return detect_format(ds.column_names, sample)


def prepare_datasets(cfg: DatasetConfig, tokenizer: Any, template: str, seed: int = 3407, token: str = ""):
    """Load, convert and split the dataset.

    Returns ``(train_ds, eval_ds_or_None, info_dict)`` where the datasets contain a
    single ``text`` column ready for SFT.
    """
    raw = load_raw_dataset(cfg, token=token)
    fmt = effective_format(raw, cfg)

    if cfg.shuffle:
        raw = raw.shuffle(seed=seed)
    if cfg.max_samples and cfg.max_samples < len(raw):
        raw = raw.select(range(cfg.max_samples))

    eval_raw = None
    if cfg.eval_split and cfg.source == "huggingface":
        eval_raw = load_raw_dataset(cfg, split=cfg.eval_split, token=token)
    elif cfg.eval_ratio > 0:
        parts = raw.train_test_split(test_size=cfg.eval_ratio, seed=seed)
        raw, eval_raw = parts["train"], parts["test"]

    train = convert_dataset(raw, cfg, fmt, tokenizer, template)
    evald = convert_dataset(eval_raw, cfg, fmt, tokenizer, template) if eval_raw is not None else None

    info = {
        "format": fmt,
        "raw_rows": len(raw) + (len(eval_raw) if eval_raw is not None else 0),
        "train_rows": len(train),
        "eval_rows": len(evald) if evald is not None else 0,
    }
    return train, evald, info


def convert_dataset(ds, cfg: DatasetConfig, fmt: str, tokenizer: Any, template: str):
    """Map a raw dataset to a single ``text`` column, dropping unusable rows."""
    columns = resolve_columns(ds.column_names, cfg, fmt)
    eos = getattr(tokenizer, "eos_token", "") or ""

    if fmt == "text":
        text_col = columns.get("text")
        if not text_col:
            raise ValueError(f"No text column found. Available columns: {ds.column_names}")

        def _map_text(batch):
            out = []
            for t in batch[text_col]:
                t = _as_str(t)
                out.append(t if (not eos or t.endswith(eos)) else t + eos)
            return {TEXT_FIELD: out}

        mapped = ds.map(_map_text, batched=True, remove_columns=ds.column_names, num_proc=_nproc(cfg, ds))
    else:
        if fmt in ("prompt_completion", "custom") and (not columns.get("prompt") or not columns.get("response")):
            raise ValueError(
                f"Could not resolve prompt/response columns for format '{fmt}'. "
                f"Available columns: {ds.column_names}"
            )

        def _map_chat(batch):
            keys = list(batch.keys())
            n = len(batch[keys[0]]) if keys else 0
            texts = []
            for i in range(n):
                row = {k: batch[k][i] for k in keys}
                messages = row_to_messages(row, fmt, columns, cfg.system_prompt)
                texts.append(render_messages(messages, tokenizer, template) if messages else "")
            return {TEXT_FIELD: texts}

        # Tokenizers are not always picklable across processes -> single process for chat rendering.
        mapped = ds.map(_map_chat, batched=True, remove_columns=ds.column_names)

    return mapped.filter(lambda r: bool(r[TEXT_FIELD] and r[TEXT_FIELD].strip()))


def _nproc(cfg: DatasetConfig, ds) -> int | None:
    if cfg.num_proc <= 1 or len(ds) < 5000:
        return None
    return cfg.num_proc


# --------------------------------------------------------------------------- #
# Preview helpers (used by the UI, no GPU required)
# --------------------------------------------------------------------------- #


def preview_dataset(cfg: DatasetConfig, tokenizer: Any = None, template: str = "auto", n: int = 5, token: str = ""):
    """Return a dict with a pandas preview, detected format and rendered examples."""
    raw = load_raw_dataset(cfg, token=token)
    fmt = effective_format(raw, cfg)
    columns = resolve_columns(raw.column_names, cfg, fmt)
    n = min(n, len(raw))
    head = raw.select(range(n))

    rendered = []
    lengths = []
    for row in head:
        if fmt == "text":
            text = _as_str(row.get(columns.get("text") or "", ""))
        else:
            messages = row_to_messages(row, fmt, columns, cfg.system_prompt)
            if not messages:
                rendered.append("⚠️ Row skipped: no user/assistant content could be extracted.")
                continue
            text = render_messages(messages, tokenizer, template) if tokenizer else render_plain(messages)
        rendered.append(text)
        if tokenizer is not None:
            try:
                lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            except Exception:  # noqa: BLE001 - preview only
                pass

    df = head.to_pandas()
    for col in df.columns:
        df[col] = df[col].map(lambda v: _truncate(_as_str(v), 300))

    return {
        "num_rows": len(raw),
        "columns": raw.column_names,
        "format": fmt,
        "resolved_columns": columns,
        "dataframe": df,
        "rendered": rendered,
        "token_lengths": lengths,
    }


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …"
