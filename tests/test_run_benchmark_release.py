from __future__ import annotations

import json
import logging
from pathlib import Path

from src.app.run_benchmark import (
    _configure_logging,
    _discover_model_files,
    _read_dataset_manifest_meta,
    _stratified_by_scenario,
)
from src.planner.planner import _fallback_parse_plan


def _case(root: Path, name: str, scenario: str) -> Path:
    case = root / name
    case.mkdir()
    (case / "manifest.json").write_text(
        json.dumps({"dataset_id": name, "scenario_type": scenario, "task_type": "workflow_recommendation"}),
        encoding="utf-8",
    )
    return case


def test_model_discovery_excludes_the_reported_inventory_manifest(tmp_path: Path) -> None:
    (tmp_path / "reported_models.yaml").write_text("models: []", encoding="utf-8")
    model = tmp_path / "qwen3-8b-instruct.yaml"
    model.write_text("name: qwen3-8b-instruct", encoding="utf-8")
    (tmp_path / "README.md").write_text("not a model", encoding="utf-8")

    assert _discover_model_files(tmp_path) == [model]


def test_stratified_selection_is_deterministic_and_balanced(tmp_path: Path) -> None:
    cases = []
    for scenario in ("standard", "batch_only", "missing_only"):
        for idx in range(4):
            cases.append(_case(tmp_path, f"{scenario}_{idx}", scenario))

    first = _stratified_by_scenario(cases, limit=6, seed=42)
    second = _stratified_by_scenario(cases, limit=6, seed=42)
    scenarios = [_read_dataset_manifest_meta(path)[1] for path in first]

    assert first == second
    assert {name: scenarios.count(name) for name in set(scenarios)} == {
        "batch_only": 2,
        "missing_only": 2,
        "standard": 2,
    }


def test_manifest_metadata_falls_back_to_task_card(tmp_path: Path) -> None:
    case = tmp_path / "gf_case_0001"
    case.mkdir()
    (case / "manifest.json").write_text('{"dataset_id":"gf_case_0001"}', encoding="utf-8")
    (case / "task.json").write_text(
        '{"known_constraints":{"scenario_type":"batch_only"}}', encoding="utf-8"
    )

    assert _read_dataset_manifest_meta(case) == (
        "gf_case_0001",
        "batch_only",
        "workflow_recommendation",
    )


def test_fallback_parser_ignores_thinking_blocks_before_json() -> None:
    text = (
        '<think>{"irrelevant": true}</think>\n'
        '{"workflow_name":"clean","steps":['
        '{"tool":"inspect_dataset","purpose":"inspect"},'
        '{"tool":"clr_transform","purpose":"transform"}]}'
    )

    plan, reason = _fallback_parse_plan(text, ["inspect_dataset", "clr_transform"])

    assert reason == "fallback_json_parsed"
    assert plan.workflow_name == "clean"
    assert [step.tool for step in plan.steps] == ["inspect_dataset", "clr_transform"]


def test_configure_logging_can_write_plain_text_file(tmp_path: Path) -> None:
    log_path = tmp_path / "driver.log"
    logger = _configure_logging(verbose=False, log_file=str(log_path))
    logger.info("release-log-message")
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "release-log-message" in text
    assert "\x1b[" not in text

