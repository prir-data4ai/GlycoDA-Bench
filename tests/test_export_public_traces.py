from __future__ import annotations

import json
from pathlib import Path

from scripts.export_public_traces import export_public_trace, sanitize_for_publication


def test_sanitizer_removes_credentials_endpoints_and_absolute_paths() -> None:
    windows_path = "D:" + r"\private\case\abundance.csv"
    unix_path = "/" + "home/user/run/result.csv"
    payload = {
        "api_key": "secret-value",
        "base_url": "https://internal.example/v1",
        "llm_total_tokens": 17,
        "input_csv": windows_path,
        "nested": [{"output": unix_path, "status": "ok"}],
    }

    sanitized = sanitize_for_publication(payload)

    assert "api_key" not in sanitized
    assert "base_url" not in sanitized
    assert sanitized["llm_total_tokens"] == 17
    assert sanitized["input_csv"] == "artifact://abundance.csv"
    assert sanitized["nested"][0]["output"] == "artifact://result.csv"


def test_export_keeps_plan_trace_and_metrics_but_excludes_raw_calls_and_path_fields(tmp_path: Path) -> None:
    trace_path = "D:" + r"\run\trace.json"
    inspect_path = "D:" + r"\run\inspect.csv"
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "dataset_id": "gf_case_0001",
                "scenario_type": "single_workflow",
                "task_type": "workflow_recommendation",
                "model_name": "model-a",
                "seed": 42,
                "planning_mode": "full",
                "plan_valid": True,
                "plan": {"steps": [{"tool": "inspect_dataset"}]},
                "execution_trace": [{"tool": "inspect_dataset", "output": inspect_path}],
                "decision_log": [{"decision": "inspect first"}],
                "metrics": {"plan_score": 0.8},
                "oracle": {"verdict": "correct"},
                "llm_calls": [{"raw_provider_message": "do not publish"}],
                "execution_trace_path": trace_path,
                "preflight_inspect_path": inspect_path,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "public.json"

    export_public_trace(source, output)

    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["dataset_id"] == "gf_case_0001"
    assert exported["plan"]["steps"][0]["tool"] == "inspect_dataset"
    assert exported["execution_trace"][0]["output"] == "artifact://inspect.csv"
    assert exported["metrics"]["plan_score"] == 0.8
    assert "llm_calls" not in exported
    assert "execution_trace_path" not in exported
    assert "preflight_inspect_path" not in exported
