from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.build_public_result_artifacts import (
    SUMMARY_METRICS,
    build_public_result_artifacts,
)


def _write_fixture(tmp_path: Path, *, dataset_ids: list[str]) -> tuple[Path, Path, Path]:
    results_dir = tmp_path / "source_results"
    results_dir.mkdir()
    scenario_path = tmp_path / "scenario_index.csv"
    pd.DataFrame(
        {
            "dataset_id": ["case_1", "case_2", "case_3", "case_4"],
            "task_family": ["single", "single", "paired", "paired"],
            "complexity_tier": ["simple", "hard", "simple", "hard"],
            "signal_regime": ["weak", "weak", "strong", "strong"],
        }
    ).to_csv(scenario_path, index=False)
    pd.DataFrame(
        {
            "model_name": ["raw-a", "raw-a-alias", "raw-a", "raw-a"],
            "dataset_id": dataset_ids,
            "run_status": ["completed", "completed", "skipped", "completed"],
            "failure_type": ["none", "none", "skipped", "invalid_plan"],
            "plan_score": [0.2, 0.4, 0.6, 0.8],
            "execution_score": [0.3, 0.5, 0.7, 0.9],
            "decision_score": [0.4, 0.6, 0.8, 1.0],
            "result_score": [0.5, 0.7, 0.9, 1.0],
            "batch_reduction_score": [0.1, 0.2, 0.3, 0.4],
            "bio_preservation_score": [0.9, 0.8, 0.7, 0.6],
            "de_recovery_score": [0.3, 0.4, 0.5, 0.6],
            "de_f1": [0.2, 0.3, 0.4, 0.5],
        }
    ).to_csv(results_dir / "model-a.csv", index=False)
    manifest_path = tmp_path / "reported_models.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "model-a",
                        "source_result_file": "model-a.csv",
                        "evaluation_timestamp": "2026-04-10T10:00:00",
                        "reported_model_id": "provider/model-a",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return results_dir, scenario_path, manifest_path


def test_builder_normalizes_aliases_and_uses_completed_rows_for_means(tmp_path: Path) -> None:
    results_dir, scenario_path, manifest_path = _write_fixture(
        tmp_path, dataset_ids=["case_1", "case_2", "case_3", "case_4"]
    )
    output_dir = tmp_path / "public"

    build_public_result_artifacts(
        results_dir=results_dir,
        scenario_index=scenario_path,
        model_manifest=manifest_path,
        output_dir=output_dir,
        expected_model_count=1,
        expected_case_count=4,
    )

    summary = pd.read_csv(output_dir / "model_summary.csv")
    assert summary.loc[0, "model_name"] == "model-a"
    assert summary.loc[0, "n_cases_total"] == 4
    assert summary.loc[0, "n_cases_completed"] == 3
    assert summary.loc[0, "mean_plan_score"] == pytest.approx((0.2 + 0.4 + 0.8) / 3)

    manifest = json.loads((output_dir / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_release"] == "manuscript-v1"
    assert manifest["metric_schema_id"] == "legacy-v1-report"
    assert manifest["dataset_revision"] == "manuscript-v1.0.0"
    assert manifest["dataset_commit"] == "0c3aacd0045ec3e87476689e8187376c3e41cd5c"
    assert manifest["current_evaluator_schema_id"] == "aspect-v3-eleven-metric"
    assert manifest["models"][0]["normalized_model_name"] == "model-a"
    assert manifest["models"][0]["raw_model_name_counts"] == {"raw-a": 3, "raw-a-alias": 1}

    strata = pd.read_csv(output_dir / "stratified_summary.csv")
    assert set(strata["stratum_type"]) == {"task_family", "complexity_tier", "signal_regime"}
    paired = strata[(strata["stratum_type"] == "task_family") & (strata["stratum_value"] == "paired")].iloc[0]
    assert paired["n_cases_total"] == 2
    assert paired["n_cases_completed"] == 1


def test_builder_rejects_result_case_ids_that_do_not_match_scenario_index(tmp_path: Path) -> None:
    results_dir, scenario_path, manifest_path = _write_fixture(
        tmp_path, dataset_ids=["case_1", "case_2", "case_3", "wrong_case"]
    )

    with pytest.raises(ValueError, match="case IDs do not match"):
        build_public_result_artifacts(
            results_dir=results_dir,
            scenario_index=scenario_path,
            model_manifest=manifest_path,
            output_dir=tmp_path / "public",
            expected_model_count=1,
            expected_case_count=4,
        )


@pytest.mark.parametrize("column", ["run_status", *SUMMARY_METRICS])
def test_builder_rejects_missing_required_result_columns(tmp_path: Path, column: str) -> None:
    results_dir, scenario_path, manifest_path = _write_fixture(
        tmp_path, dataset_ids=["case_1", "case_2", "case_3", "case_4"]
    )
    source = results_dir / "model-a.csv"
    frame = pd.read_csv(source).drop(columns=[column])
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="missing required result columns"):
        build_public_result_artifacts(
            results_dir=results_dir,
            scenario_index=scenario_path,
            model_manifest=manifest_path,
            output_dir=tmp_path / "public",
            expected_model_count=1,
            expected_case_count=4,
        )


def test_builder_rejects_unknown_run_status(tmp_path: Path) -> None:
    results_dir, scenario_path, manifest_path = _write_fixture(
        tmp_path, dataset_ids=["case_1", "case_2", "case_3", "case_4"]
    )
    source = results_dir / "model-a.csv"
    frame = pd.read_csv(source)
    frame.loc[0, "run_status"] = "mystery"
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="invalid run_status"):
        build_public_result_artifacts(
            results_dir=results_dir,
            scenario_index=scenario_path,
            model_manifest=manifest_path,
            output_dir=tmp_path / "public",
            expected_model_count=1,
            expected_case_count=4,
        )


def test_builder_rejects_non_numeric_metric(tmp_path: Path) -> None:
    results_dir, scenario_path, manifest_path = _write_fixture(
        tmp_path, dataset_ids=["case_1", "case_2", "case_3", "case_4"]
    )
    source = results_dir / "model-a.csv"
    frame = pd.read_csv(source)
    frame["plan_score"] = frame["plan_score"].astype(object)
    frame.loc[0, "plan_score"] = "not-a-number"
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="non-numeric metric"):
        build_public_result_artifacts(
            results_dir=results_dir,
            scenario_index=scenario_path,
            model_manifest=manifest_path,
            output_dir=tmp_path / "public",
            expected_model_count=1,
            expected_case_count=4,
        )


def test_builder_rejects_metric_outside_unit_interval(tmp_path: Path) -> None:
    results_dir, scenario_path, manifest_path = _write_fixture(
        tmp_path, dataset_ids=["case_1", "case_2", "case_3", "case_4"]
    )
    source = results_dir / "model-a.csv"
    frame = pd.read_csv(source)
    frame.loc[0, "plan_score"] = 1.5
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        build_public_result_artifacts(
            results_dir=results_dir,
            scenario_index=scenario_path,
            model_manifest=manifest_path,
            output_dir=tmp_path / "public",
            expected_model_count=1,
            expected_case_count=4,
        )


def test_builder_rejects_sources_with_no_completed_rows(tmp_path: Path) -> None:
    results_dir, scenario_path, manifest_path = _write_fixture(
        tmp_path, dataset_ids=["case_1", "case_2", "case_3", "case_4"]
    )
    source = results_dir / "model-a.csv"
    frame = pd.read_csv(source)
    frame["run_status"] = "skipped"
    for metric in SUMMARY_METRICS:
        frame[metric] = float("nan")
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="no completed rows"):
        build_public_result_artifacts(
            results_dir=results_dir,
            scenario_index=scenario_path,
            model_manifest=manifest_path,
            output_dir=tmp_path / "public",
            expected_model_count=1,
            expected_case_count=4,
        )
