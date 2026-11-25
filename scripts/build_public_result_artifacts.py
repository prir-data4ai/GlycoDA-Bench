from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SUMMARY_METRICS = (
    "plan_score",
    "execution_score",
    "decision_score",
    "result_score",
    "batch_reduction_score",
    "bio_preservation_score",
    "de_recovery_score",
    "de_f1",
)
STRATA = ("task_family", "complexity_tier", "signal_regime")
ALLOWED_RUN_STATUSES = {"completed", "skipped"}
METRIC_SCHEMA_ID = "legacy-v1-report"
CURRENT_EVALUATOR_SCHEMA_ID = "aspect-v3-eleven-metric"
DATASET_REVISION = "manuscript-v1.0.0"
DATASET_REPOSITORY = "prirdata4ai/GlycoDA-Bench"
DATASET_COMMIT = "987e7b792c36eefbb7b6cff11e04f685e720b53d"
SOURCE_DATASET_REPOSITORY = "xiaokui17/glycoforge_benchmark_648"
SOURCE_DATASET_COMMIT = "0c3aacd0045ec3e87476689e8187376c3e41cd5c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["run_status"].astype(str).str.strip().str.lower() == "completed"]


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise")
    return float(values.mean())


def _validate_result_schema(frame: pd.DataFrame, *, source_name: str) -> None:
    required = {"dataset_id", "run_status", *SUMMARY_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name}: missing required result columns: {missing}")

    statuses = frame["run_status"].astype(str).str.strip().str.lower()
    invalid_statuses = sorted(set(statuses) - ALLOWED_RUN_STATUSES)
    if invalid_statuses:
        raise ValueError(f"{source_name}: invalid run_status values: {invalid_statuses}")

    completed = statuses == "completed"
    if not completed.any():
        raise ValueError(f"{source_name}: no completed rows available for aggregation")
    for metric in SUMMARY_METRICS:
        try:
            values = pd.to_numeric(frame[metric], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source_name}: non-numeric metric values in {metric}") from exc
        missing_completed = completed & values.isna()
        if missing_completed.any():
            raise ValueError(f"{source_name}: completed rows have missing metric values in {metric}")
        finite_completed = values[completed].map(math.isfinite)
        if not finite_completed.all():
            raise ValueError(f"{source_name}: non-finite metric values in {metric}")
        outside = values[completed].lt(0) | values[completed].gt(1)
        if outside.any():
            raise ValueError(f"{source_name}: metric values outside [0, 1] in {metric}")


def _summary_row(frame: pd.DataFrame, *, model_name: str) -> dict[str, Any]:
    done = _completed(frame)
    row: dict[str, Any] = {
        "model_name": model_name,
        "n_cases_total": int(len(frame)),
        "n_cases_completed": int(len(done)),
        "n_cases_skipped": int(len(frame) - len(done)),
    }
    for metric in SUMMARY_METRICS:
        row[f"mean_{metric}"] = _mean(done, metric)
    return row


def build_public_result_artifacts(
    *,
    results_dir: Path,
    scenario_index: Path,
    model_manifest: Path,
    output_dir: Path,
    expected_model_count: int = 22,
    expected_case_count: int = 648,
) -> None:
    results_dir = Path(results_dir)
    scenario_index = Path(scenario_index)
    model_manifest = Path(model_manifest)
    output_dir = Path(output_dir)

    manifest = yaml.safe_load(model_manifest.read_text(encoding="utf-8"))
    models = list(manifest.get("models") or [])
    if len(models) != expected_model_count:
        raise ValueError(f"expected {expected_model_count} models, found {len(models)}")

    scenarios = pd.read_csv(scenario_index)
    required_scenario_columns = {"dataset_id", *STRATA}
    missing_columns = sorted(required_scenario_columns - set(scenarios.columns))
    if missing_columns:
        raise ValueError(f"scenario index missing columns: {missing_columns}")
    if len(scenarios) != expected_case_count or scenarios["dataset_id"].nunique() != expected_case_count:
        raise ValueError(
            f"scenario index must contain {expected_case_count} unique cases; "
            f"rows={len(scenarios)} unique={scenarios['dataset_id'].nunique()}"
        )
    expected_ids = set(scenarios["dataset_id"].astype(str))

    model_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    release_models: list[dict[str, Any]] = []

    for item in models:
        normalized_name = str(item["name"])
        source_name = str(item["source_result_file"])
        source_path = results_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source result CSV: {source_name}")
        frame = pd.read_csv(source_path)
        _validate_result_schema(frame, source_name=source_name)
        actual_ids = set(frame["dataset_id"].astype(str))
        if len(frame) != expected_case_count or frame["dataset_id"].nunique() != expected_case_count:
            raise ValueError(
                f"{source_name}: expected {expected_case_count} rows and unique case IDs; "
                f"rows={len(frame)} unique={frame['dataset_id'].nunique()}"
            )
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)[:5]
            extra = sorted(actual_ids - expected_ids)[:5]
            raise ValueError(f"{source_name}: case IDs do not match scenario index; missing={missing} extra={extra}")

        row = _summary_row(frame, model_name=normalized_name)
        row.update(
            {
                "reported_model_id": item.get("reported_model_id", ""),
                "evaluation_timestamp": item.get("evaluation_timestamp", ""),
                "source_result_file": source_name,
            }
        )
        model_rows.append(row)

        joined = frame.merge(
            scenarios[["dataset_id", *STRATA]],
            on="dataset_id",
            how="left",
            validate="one_to_one",
        )
        for stratum in STRATA:
            for value, group in joined.groupby(stratum, dropna=False, sort=True):
                stratum_row = _summary_row(group, model_name=normalized_name)
                stratum_row.update({"stratum_type": stratum, "stratum_value": str(value)})
                stratum_rows.append(stratum_row)

        raw_counts = (
            frame["model_name"].astype(str).value_counts().sort_index().astype(int).to_dict()
            if "model_name" in frame.columns
            else {}
        )
        failure_counts = (
            frame["failure_type"].fillna("").astype(str).value_counts().sort_index().astype(int).to_dict()
            if "failure_type" in frame.columns
            else {}
        )
        release_models.append(
            {
                "normalized_model_name": normalized_name,
                "reported_model_id": item.get("reported_model_id", ""),
                "evaluation_timestamp": item.get("evaluation_timestamp", ""),
                "source_result_file": source_name,
                "source_sha256": _sha256(source_path),
                "row_count": int(len(frame)),
                "unique_case_ids": int(frame["dataset_id"].nunique()),
                "raw_model_name_counts": raw_counts,
                "failure_type_counts": failure_counts,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_rows).to_csv(output_dir / "model_summary.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(output_dir / "stratified_summary.csv", index=False)
    release_manifest = {
        "schema_version": 2,
        "dataset_release": "manuscript-v1",
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "dataset_commit": DATASET_COMMIT,
        "source_dataset_repository": SOURCE_DATASET_REPOSITORY,
        "source_dataset_commit": SOURCE_DATASET_COMMIT,
        "metric_schema_id": METRIC_SCHEMA_ID,
        "current_evaluator_schema_id": CURRENT_EVALUATOR_SCHEMA_ID,
        "metric_schema_note": (
            "These compact tables summarize the unchanged historical manuscript result CSVs. "
            "They are not outputs of the current eleven-metric aspect evaluator; see "
            "docs/RESULT_SCHEMAS.md."
        ),
        "scenario_index_file": scenario_index.name,
        "scenario_index_sha256": _sha256(scenario_index),
        "model_count": len(release_models),
        "case_count_per_model": expected_case_count,
        "aggregation_rule": (
            "Means use rows with run_status=completed; skipped rows remain in counts. "
            "Model identifiers are normalized by configs/models/reported_models.yaml; "
            "raw model_name values remain recorded in raw_model_name_counts."
        ),
        "metrics": list(SUMMARY_METRICS),
        "strata": list(STRATA),
        "models": release_models,
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build public compact artifacts from per-model result CSVs.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scenario-index", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, default=Path("configs/models/reported_models.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--expected-model-count", type=int, default=22)
    parser.add_argument("--expected-case-count", type=int, default=648)
    args = parser.parse_args()
    build_public_result_artifacts(
        results_dir=args.results_dir,
        scenario_index=args.scenario_index,
        model_manifest=args.model_manifest,
        output_dir=args.output_dir,
        expected_model_count=args.expected_model_count,
        expected_case_count=args.expected_case_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
