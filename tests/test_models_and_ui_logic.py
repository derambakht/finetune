import json

from ftstudio.config import FinetuneConfig
from ftstudio.models import (
    MODEL_CATALOG,
    RESPONSE_MARKERS,
    estimate_vram_gb,
    guess_chat_template,
    guess_params_b,
)
from ftstudio.reporting import RunReporter, list_checkpoints, read_metrics, read_status, request_stop
from ftstudio.ui import logic


def test_guess_params_and_template():
    assert guess_params_b("unsloth/Meta-Llama-3.1-8B-Instruct") == 8.0
    assert guess_params_b("someone/tiny-135M") == 0.135
    assert guess_params_b("org/model-without-size") is None
    assert guess_chat_template("someone/Qwen2.5-3B") == "qwen-2.5"
    assert guess_chat_template("x/Llama-3.2-1B") == "llama-3.1"
    assert guess_chat_template("x/unknown") == "auto"


def test_catalog_templates_have_markers():
    for m in MODEL_CATALOG:
        assert m.chat_template in RESPONSE_MARKERS, m.model_id


def test_vram_estimate_ordering():
    args = dict(lora_r=16, batch_size=2, max_seq_length=2048)
    q4 = estimate_vram_gb(8, "4bit", **args)["total"]
    q16 = estimate_vram_gb(8, "16bit", **args)["total"]
    full = estimate_vram_gb(8, "full", **args)["total"]
    assert q4 < q16 < full


def test_fields_cover_every_config_attribute():
    cfg = FinetuneConfig()
    covered = set(logic.FIELDS)
    for section in ("model", "dataset", "lora", "training", "export"):
        for name in vars(getattr(cfg, section)):
            assert (section, name) in covered, f"{section}.{name} missing from UI"


def test_values_roundtrip():
    cfg = FinetuneConfig()
    cfg.training.learning_rate = 1e-5
    cfg.model.hf_token = "hf_x"
    values = logic.config_to_values(cfg, keep_secrets=False)
    back = logic.values_to_config(values)
    assert back.to_dict() == cfg.to_dict()
    # empty secrets are returned as None so the UI keeps the typed token
    cfg.model.hf_token = ""
    assert None in logic.config_to_values(cfg, keep_secrets=True)


def test_presets_are_valid():
    from ftstudio.config import validate_config

    for name in logic.PRESETS:
        cfg = logic.apply_preset(FinetuneConfig(), name)
        assert validate_config(cfg).ok, name


def test_reporter_and_monitor_helpers(tmp_path):
    rep = RunReporter(tmp_path)
    rep.state("training", "go", max_steps=10)
    rep.update(step=5, epoch=0.5)
    rep.log_metrics({"step": 1, "loss": 2.0, "learning_rate": 1e-4})
    rep.log_metrics({"step": 5, "loss": 1.5, "eval_loss": 1.7})
    with open(tmp_path / "metrics.jsonl", "a") as fh:
        fh.write('{"partial": ')  # simulate a half-written line
    assert read_status(tmp_path)["step"] == 5
    records = read_metrics(tmp_path)
    assert len(records) == 2
    loss_df, lr_df, latest = logic.metrics_frames(records)
    assert set(loss_df["series"]) == {"train loss", "eval loss"}
    assert len(lr_df) == 1
    assert latest["loss"] == 1.5
    assert "50.0%" in logic.progress_html(read_status(tmp_path))
    assert "Training" in logic.status_markdown(read_status(tmp_path))
    assert not rep.stop_requested()
    request_stop(tmp_path)
    assert rep.stop_requested()
    json.dumps(read_status(tmp_path))


def test_list_checkpoints_order(tmp_path):
    for name in ["checkpoint-10", "checkpoint-200", "final_model", "other"]:
        (tmp_path / name).mkdir()
    assert [p.name for p in list_checkpoints(tmp_path)] == ["final_model", "checkpoint-200", "checkpoint-10"]


def test_parse_auth():
    import pytest

    from ftstudio.ui.app import parse_auth

    assert parse_auth(None) is None and parse_auth("  ") is None
    assert parse_auth("admin:s3cret") == [("admin", "s3cret")]
    assert parse_auth("a:1, b:p:w") == [("a", "1"), ("b", "p:w")]
    with pytest.raises(ValueError):
        parse_auth("no-password")
