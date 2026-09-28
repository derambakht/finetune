import json

import pytest

from ftstudio.config import DatasetConfig
from ftstudio.data import (
    TEXT_FIELD,
    convert_dataset,
    detect_format,
    render_alpaca,
    render_messages,
    resolve_columns,
    row_to_messages,
)


class FakeTokenizer:
    """Minimal ChatML tokenizer stand-in."""

    eos_token = "<|im_end|>"
    bos_token = None
    chat_template = "chatml"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


@pytest.mark.parametrize(
    "columns,sample,expected",
    [
        (["messages"], None, "messages"),
        (["conversations"], {"conversations": [{"from": "human", "value": "hi"}]}, "sharegpt"),
        (["conversations"], {"conversations": [{"role": "user", "content": "hi"}]}, "messages"),
        (["instruction", "input", "output"], None, "alpaca"),
        (["question", "answer"], None, "prompt_completion"),
        (["text"], None, "text"),
        (["foo", "bar"], None, "custom"),
    ],
)
def test_detect_format(columns, sample, expected):
    assert detect_format(columns, sample) == expected


def test_alpaca_row_with_input_and_system_prompt():
    msgs = row_to_messages(
        {"instruction": "Translate", "input": "hello", "output": "salam"}, "alpaca", system_prompt="Be brief"
    )
    assert msgs == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Translate\n\nhello"},
        {"role": "assistant", "content": "salam"},
    ]


def test_sharegpt_role_mapping():
    row = {"conversations": [
        {"from": "system", "value": "sys"},
        {"from": "human", "value": "q"},
        {"from": "gpt", "value": "a"},
        {"from": "weird", "value": "ignored"},
    ]}
    msgs = row_to_messages(row, "sharegpt")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_messages_multimodal_content_is_flattened():
    row = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]},
        {"role": "assistant", "content": "hello"},
    ]}
    assert row_to_messages(row, "messages")[0]["content"] == "hi there"


def test_rows_without_answer_are_skipped():
    assert row_to_messages({"instruction": "x", "output": ""}, "alpaca") is None
    assert row_to_messages({"messages": [{"role": "user", "content": "q"}]}, "messages") is None


def test_prompt_completion_column_resolution():
    cfg = DatasetConfig(prompt_column="prompt", response_column="completion")
    cols = resolve_columns(["Question", "Answer"], cfg, "prompt_completion")
    assert cols["prompt"] == "Question" and cols["response"] == "Answer"
    msgs = row_to_messages({"Question": "2+2?", "Answer": 4}, "prompt_completion", cols)
    assert msgs[-1] == {"role": "assistant", "content": "4"}


def test_custom_mapping_with_system_column():
    cfg = DatasetConfig(prompt_column="q", response_column="a", system_column="s")
    cols = resolve_columns(["q", "a", "s"], cfg, "custom")
    msgs = row_to_messages({"q": "hi", "a": "yo", "s": "persona"}, "custom", cols)
    assert msgs[0] == {"role": "system", "content": "persona"}


def test_render_with_chat_template():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    text = render_messages(msgs, FakeTokenizer(), "auto")
    assert text == "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n"


def test_render_alpaca_appends_eos():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    text = render_alpaca(msgs, "</s>")
    assert "### Instruction:\nq" in text and text.endswith("### Response:\na</s>")


def test_convert_dataset_end_to_end(tmp_path):
    datasets = pytest.importorskip("datasets")
    rows = [
        {"instruction": "a", "input": "", "output": "b"},
        {"instruction": "c", "input": "d", "output": ""},  # dropped: empty answer
    ]
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    ds = datasets.load_dataset("json", data_files=str(path), split="train")
    out = convert_dataset(ds, DatasetConfig(), "alpaca", FakeTokenizer(), "auto")
    assert out.column_names == [TEXT_FIELD]
    assert len(out) == 1
    assert "<|im_start|>assistant\nb" in out[0][TEXT_FIELD]


def test_convert_text_dataset_adds_eos():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"text": ["hello world", "done<|im_end|>"]})
    out = convert_dataset(ds, DatasetConfig(), "text", FakeTokenizer(), "auto")
    assert out[0][TEXT_FIELD] == "hello world<|im_end|>"
    assert out[1][TEXT_FIELD] == "done<|im_end|>"


def test_prepare_datasets_local_split(tmp_path):
    pytest.importorskip("datasets")
    from ftstudio.data import prepare_datasets

    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]})
        for i in range(20)
    ))
    cfg = DatasetConfig(source="local", path=str(path), format="auto", eval_ratio=0.2, max_samples=10)
    train, evald, info = prepare_datasets(cfg, FakeTokenizer(), "auto")
    assert info["format"] == "messages"
    assert len(train) == 8 and len(evald) == 2
