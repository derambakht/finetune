from ftstudio.config import FinetuneConfig, validate_config


def test_default_config_is_valid():
    res = validate_config(FinetuneConfig())
    assert res.ok, res.errors


def test_yaml_roundtrip_and_secret_redaction(tmp_path):
    cfg = FinetuneConfig()
    cfg.model.hf_token = "hf_secret"
    cfg.lora.r = 64
    cfg.export.methods = ["lora", "gguf"]
    path = cfg.save(tmp_path / "c.yaml")
    text = path.read_text()
    assert "hf_secret" not in text
    loaded = FinetuneConfig.load(path)
    assert loaded.lora.r == 64
    assert loaded.export.methods == ["lora", "gguf"]
    assert loaded.model.hf_token == ""


def test_from_dict_coerces_types_and_ignores_unknown_keys():
    cfg = FinetuneConfig.from_dict({
        "model": {"max_seq_length": "4096", "unknown": 1},
        "training": {"learning_rate": "1e-4", "packing": "true", "max_steps": 10.0},
        "lora": {"target_modules": "q_proj, v_proj"},
        "bogus_section": {},
    })
    assert cfg.model.max_seq_length == 4096
    assert cfg.training.learning_rate == 1e-4
    assert cfg.training.packing is True
    assert cfg.training.max_steps == 10
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]


def test_validation_catches_errors():
    cfg = FinetuneConfig()
    cfg.model.quantization = "3bit"
    cfg.training.run_name = "bad name!"
    cfg.training.learning_rate = 0
    cfg.dataset.format = "custom"
    cfg.dataset.prompt_column = ""
    cfg.export.push_to_hub = True
    cfg.export.hub_repo_id = ""
    res = validate_config(cfg)
    joined = " ".join(res.errors)
    assert not res.ok
    assert "quantization" in joined
    assert "Run name" in joined
    assert "Learning rate" in joined
    assert "Custom format" in joined
    assert "repository id" in joined


def test_local_dataset_must_exist():
    cfg = FinetuneConfig()
    cfg.dataset.source = "local"
    cfg.dataset.path = "/does/not/exist.jsonl"
    assert any("not found" in e for e in validate_config(cfg).errors)


def test_full_finetune_warns_about_lr():
    cfg = FinetuneConfig()
    cfg.model.quantization = "full"
    cfg.training.learning_rate = 2e-4
    res = validate_config(cfg)
    assert res.ok
    assert any("Full fine-tuning" in w for w in res.warnings)
