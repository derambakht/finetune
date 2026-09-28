import json
from pathlib import Path

import pytest

from ftstudio.config import DatasetConfig, FinetuneConfig, validate_config
from ftstudio.data import ZWNJ, normalize_persian, prepare_datasets, row_to_messages
from ftstudio.ui import logic

from test_data import FakeTokenizer

ROOT = Path(__file__).resolve().parents[1]
PERSIAN_DIR = ROOT / "data" / "persian"


def test_normalize_arabic_letters_and_kashida():
    assert normalize_persian("كتاب علي") == "کتاب علی"
    assert normalize_persian("ســـلام") == "سلام"


def test_normalize_half_spaces():
    assert normalize_persian("می روم") == f"می{ZWNJ}روم"
    assert normalize_persian("نمی دانم") == f"نمی{ZWNJ}دانم"
    assert normalize_persian(f"کتاب {ZWNJ} ها") == f"کتاب{ZWNJ}ها"
    assert normalize_persian(f"می{ZWNJ}{ZWNJ}شود") == f"می{ZWNJ}شود"
    # words that merely end with «می» are untouched
    assert normalize_persian("دائمی است") == "دائمی است"


def test_normalize_collapses_persian_spaces_only():
    text = "Hello  world\nسلام   دنیا"
    assert normalize_persian(text) == "Hello  world\nسلام دنیا"


def test_normalize_preserves_code_indentation():
    code = "```python\ndef f():\n    return 1  # یک\n```"
    assert normalize_persian(code) == code


def test_row_to_messages_normalizes():
    msgs = row_to_messages({"instruction": "كجا مي روي؟", "output": "مي روم خانه"}, "alpaca", normalize=True)
    assert msgs[0]["content"] == f"کجا می{ZWNJ}روی؟"
    assert msgs[1]["content"] == f"می{ZWNJ}روم خانه"


def _count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


@pytest.mark.parametrize("name", ["fa_chat_messages.jsonl", "fa_alpaca.jsonl", "fa_sharegpt.jsonl",
                                  "fa_corpus.jsonl"])
def test_jsonl_samples_are_valid_json(name):
    for line in (PERSIAN_DIR / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


@pytest.mark.parametrize(
    "name,fmt,expected_format",
    [
        ("fa_chat_messages.jsonl", "auto", "messages"),
        ("fa_alpaca.jsonl", "auto", "alpaca"),
        ("fa_sharegpt.jsonl", "auto", "sharegpt"),
        ("fa_qa.csv", "auto", "prompt_completion"),
        ("fa_corpus.jsonl", "auto", "text"),
    ],
)
def test_persian_samples_convert_without_losing_rows(name, fmt, expected_format):
    pytest.importorskip("datasets")
    path = PERSIAN_DIR / name
    cfg = DatasetConfig(source="local", path=str(path), format=fmt, normalize_persian=True, shuffle=False)
    train, evald, info = prepare_datasets(cfg, FakeTokenizer(), "auto")
    assert info["format"] == expected_format
    expected_rows = _count_lines(path) - (1 if path.suffix == ".csv" else 0)
    assert len(train) == expected_rows, "every Persian sample row should survive formatting"
    assert evald is None
    text = train[0]["text"]
    assert any("؀" <= ch <= "ۿ" for ch in text), "Persian characters expected"
    # normalisation leaves no Arabic yeh / kaf behind
    for row in train:
        assert "ي" not in row["text"] and "ك" not in row["text"]


def test_multiturn_sharegpt_keeps_all_turns():
    pytest.importorskip("datasets")
    cfg = DatasetConfig(source="local", path=str(PERSIAN_DIR / "fa_sharegpt.jsonl"), format="sharegpt",
                        shuffle=False)
    train, _, _ = prepare_datasets(cfg, FakeTokenizer(), "auto")
    first = train[0]["text"]
    assert first.count("<|im_start|>assistant") == 3
    assert first.startswith("<|im_start|>system")


def test_persian_configs_are_valid(monkeypatch):
    monkeypatch.chdir(ROOT)  # configs use repo-relative data paths
    for name in ["persian_qwen3_chat.yaml", "persian_gemma3_pretrain.yaml"]:
        cfg = FinetuneConfig.load(ROOT / "configs" / name)
        res = validate_config(cfg)
        assert res.ok, (name, res.errors)
        assert cfg.dataset.normalize_persian


def test_bundled_samples_listed_in_ui():
    choices = logic.sample_dataset_choices()
    assert len(choices) == len(logic.SAMPLE_DATASETS)
    for _label, value in choices:
        path, fmt = value.rsplit("|", 1)
        assert Path(path).exists() and fmt


def test_persian_preset_valid():
    cfg = logic.apply_preset(FinetuneConfig(), "🇮🇷 Persian chat (فارسی)")
    assert cfg.dataset.normalize_persian
    assert validate_config(cfg).ok


def test_numpy_values_keep_persian_half_spaces():
    np = pytest.importorskip("numpy")
    from ftstudio.data import _as_str

    value = np.array([{"from": "human", "value": f"می{ZWNJ}روم"}], dtype=object)
    out = _as_str(value)
    assert ZWNJ in out and "\\u200c" not in out
