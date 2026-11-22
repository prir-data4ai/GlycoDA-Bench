from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.validate_release import validate_repository


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validator_rejects_credentials_internal_hosts_and_local_paths(tmp_path: Path) -> None:
    _write(tmp_path, "configs/models/bad.yaml", "api_key: sk-live-example-not-a-placeholder\n")
    _write(tmp_path, "src/bad.py", "URL = 'http://service.production.polaris/v1'\n")
    _write(tmp_path, "docs/bad.md", r"Run from D:\private\experiment" + "\n")

    findings = validate_repository(tmp_path)

    assert any("credential-like value" in item for item in findings)
    assert any("internal host" in item for item in findings)
    assert any("absolute Windows path" in item for item in findings)


def test_validator_rejects_forbidden_release_directories(tmp_path: Path) -> None:
    _write(tmp_path, "outputs/run/report.json", "{}\n")
    _write(tmp_path, "data/glycoforge_benchmark_648/gf_case_0001/task.json", "{}\n")
    _write(tmp_path, "tmp_eval_case5/result.csv", "x\n1\n")

    findings = validate_repository(tmp_path)

    assert any("forbidden path" in item and "outputs" in item for item in findings)
    assert any("forbidden path" in item and "data" in item for item in findings)
    assert any("forbidden path" in item and "tmp_eval_case5" in item for item in findings)


def test_validator_accepts_environment_placeholders(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "configs/api.example.yaml",
        "api_key_env: GLYCODA_API_KEY\nbase_url_env: GLYCODA_BASE_URL\n",
    )
    _write(tmp_path, ".env.example", "GLYCODA_API_KEY=\nGLYCODA_BASE_URL=\n")
    _write(tmp_path, "src/example.py", "value = os.getenv('GLYCODA_API_KEY', '')\n")
    _write(tmp_path, "src/data/dataset.py", "DATASET_KIND = 'public-input-loader'\n")

    assert validate_repository(tmp_path) == []


def test_runtime_dataset_loader_is_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "src/data/dataset.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, "src/data/dataset.py is required by the runner and must be published"

