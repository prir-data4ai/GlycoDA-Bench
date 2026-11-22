from __future__ import annotations

from pathlib import Path

import yaml

from src.config import ModelConfig, load_model_config
from src.llm.adapter import LLMAdapter


def test_reported_model_inventory_has_exactly_23_public_configs() -> None:
    manifest_path = Path("configs/models/reported_models.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    models = manifest["models"]

    assert len(models) == 23
    assert len({item["name"] for item in models}) == 23
    assert len({item["config"] for item in models}) == 23
    for item in models:
        config_path = manifest_path.parent / item["config"]
        assert config_path.is_file(), config_path
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["provider"] == "openai_compatible"
        assert config["api_key_env"] == "GLYCODA_API_KEY"
        assert config["base_url_env"] == "GLYCODA_BASE_URL"
        assert config["temperature"] == 0.7
        assert not ({"api_key", "base_url", "ss_auth_token", "ss_wsid"} & set(config))


def test_model_configs_load_with_strict_public_fields() -> None:
    paths = sorted(Path("configs/models").glob("*.yaml"))
    paths = [path for path in paths if path.name != "reported_models.yaml"]

    assert len(paths) == 23
    for path in paths:
        config = load_model_config(path)
        assert config.name
        assert config.model
        assert config.provider == "openai_compatible"


def test_thinking_mode_is_forwarded_to_openai_compatible_client(monkeypatch) -> None:
    captured: dict = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("src.llm.adapter.ChatOpenAI", fake_chat_openai)
    monkeypatch.setenv("GLYCODA_API_KEY", "test-only-placeholder")
    config = ModelConfig(
        name="qwen-thinking",
        provider="openai_compatible",
        model="qwen-thinking",
        api_key_env="GLYCODA_API_KEY",
        base_url_env="GLYCODA_BASE_URL",
        enable_thinking=True,
        thinking_budget=4096,
    )

    LLMAdapter(config)._build_llm()

    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 4096}
    }

