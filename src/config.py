from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
import os


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str = Field(description="openai_compatible | local_openai_http | stream_server_http | mock")
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str | None = None
    allow_missing_api_key: bool = False
    temperature: float = 0.0
    supports_tool_calling: bool = True
    enable_thinking: bool | None = None
    thinking_budget: int | None = None
    # Optional generic stream-server fields. Values must come from public config
    # placeholders or environment variables, never repository defaults.
    ss_wsid: str = ""
    ss_auth_token: str = ""


class BenchmarkConfig(BaseModel):
    seeds: list[int] = [42]
    output_dir: str = "outputs"
    datasets: list[str] = []
    datasets_root: str | None = None
    dataset_glob: str = "gf_case_*"
    max_datasets: int | None = None
    allowed_task_types: list[str] = ["workflow_recommendation"]


class ApiRuntimeConfig(BaseModel):
    """Runtime API settings loaded from ``configs/api.yaml`` (or similar)."""

    model_config = ConfigDict(extra="ignore")

    provider: str = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model: str | None = None
    # LLM HTTP: reduce flaky runs (proxy/TLS/EOF) without changing benchmark logic
    llm_timeout_sec: float | None = 600.0
    llm_http_max_retries: int = 2
    llm_outer_attempts: int = 4
    llm_retry_backoff_sec: float = 1.5


def repo_root() -> Path:
    """Repository root (contains ``configs/``, ``data/``, ``src/``)."""
    return Path(__file__).resolve().parent.parent


def try_load_dotenv() -> None:
    """Load ``.env`` from repo root (and optionally cwd) if python-dotenv is installed.

    Uses ``override=True`` so values from ``.env`` replace empty or stale variables
    (``load_dotenv`` defaults to *not* overriding existing keys, which breaks when
    the shell has ``OPENAI_API_KEY=`` set).
    """
    try:
        from dotenv import load_dotenv

        env_path = repo_root() / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=True)
        cwd_env = Path.cwd() / ".env"
        if cwd_env.is_file() and cwd_env.resolve() != env_path.resolve():
            load_dotenv(cwd_env, override=True)
    except Exception:
        pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"YAML file is empty: {p}")
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping/object: {p}")
    return raw


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(load_yaml(path))


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(load_yaml(path))


def load_api_runtime_config(path: str | Path) -> ApiRuntimeConfig | None:
    p = Path(path)
    if not p.exists():
        return None
    return ApiRuntimeConfig.model_validate(load_yaml(path))


def apply_api_runtime_config(cfg: ApiRuntimeConfig | None) -> None:
    if cfg is None:
        return
    # When api.yaml provides api_key, always apply it so subprocesses and empty env vars behave consistently.
    if cfg.api_key and str(cfg.api_key).strip():
        os.environ[cfg.api_key_env] = str(cfg.api_key).strip()
    if cfg.base_url and not os.getenv(cfg.base_url_env):
        os.environ[cfg.base_url_env] = cfg.base_url
    if cfg.model and not os.getenv("GLYCOAGENT_MODEL"):
        if ":" not in cfg.model and ("openai" in cfg.provider):
            os.environ["GLYCOAGENT_MODEL"] = f"openai:{cfg.model}"
        else:
            os.environ["GLYCOAGENT_MODEL"] = cfg.model
    # LLM adapter reads these (see ``src.llm.adapter``).
    if cfg.llm_timeout_sec is not None:
        os.environ["GLYCOAGENT_LLM_TIMEOUT_SEC"] = str(float(cfg.llm_timeout_sec))
    else:
        os.environ.pop("GLYCOAGENT_LLM_TIMEOUT_SEC", None)
    os.environ["GLYCOAGENT_LLM_HTTP_MAX_RETRIES"] = str(int(cfg.llm_http_max_retries))
    os.environ["GLYCOAGENT_LLM_OUTER_ATTEMPTS"] = str(int(cfg.llm_outer_attempts))
    os.environ["GLYCOAGENT_LLM_RETRY_BACKOFF_SEC"] = str(float(cfg.llm_retry_backoff_sec))
