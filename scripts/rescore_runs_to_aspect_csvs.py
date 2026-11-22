from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.rescore_outputs import _rescore_run
from src.metrics.aspect_evaluator import ASPECT_NAMES


def _iter_model_dirs(runs_dir: Path, models: list[str]) -> list[Path]:
    wanted = set(models)
    dirs = [p for p in sorted(runs_dir.iterdir()) if p.is_dir()]
    if wanted:
        dirs = [p for p in dirs if p.name in wanted]
    return dirs


def _aspect_row_from_payload(payload: dict, report_path: Path) -> dict[str, object]:
    metrics = payload.get("metrics") or {}
    row: dict[str, object] = {
        "model_name": str(payload.get("model_name") or report_path.parent.parent.name),
        "dataset_id": str(payload.get("dataset_id") or ""),
    }
    for name in ASPECT_NAMES:
        row[name] = float(metrics.get(name, 0.0))
    return row


def _rescore_report_to_row(report_path: Path) -> tuple[str, dict] | tuple[None, dict[str, str]]:
    try:
        payload = _rescore_run(report_path)
        row = _aspect_row_from_payload(payload, report_path)
        dataset_id = str(row.get("dataset_id") or "")
        if not dataset_id:
            raise ValueError(f"missing dataset_id in {report_path}")
        return dataset_id, row
    except Exception as exc:  # noqa: BLE001
        return None, {"report_json": str(report_path), "error": str(exc)}


def _latest_rows_for_model(model_dir: Path, workers: int) -> tuple[list[dict], list[dict[str, str]]]:
    reports = sorted(model_dir.glob("**/report.json"), key=lambda p: p.stat().st_mtime)
    latest_by_dataset: dict[str, dict] = {}
    failed: list[dict[str, str]] = []
    if workers <= 1:
        for report_path in reports:
            dataset_id, result = _rescore_report_to_row(report_path)
            if dataset_id is None:
                failed.append(result)
            else:
                latest_by_dataset[dataset_id] = result
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_rescore_report_to_row, report_path): report_path for report_path in reports}
            for fut in concurrent.futures.as_completed(futures):
                dataset_id, result = fut.result()
                if dataset_id is None:
                    failed.append(result)
                else:
                    latest_by_dataset[dataset_id] = result
    rows = [latest_by_dataset[k] for k in sorted(latest_by_dataset)]
    return rows, failed


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "dataset_id", *ASPECT_NAMES]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rescore outputs/runs by the current evaluator and write CSVs containing only the 11 aspect metrics."
    )
    ap.add_argument("--runs-dir", default="outputs/runs", help="Root folder containing per-model run directories.")
    ap.add_argument(
        "--out-dir",
        default="outputs/leaderboards/aspects_only",
        help="Directory where per-model CSV files will be written.",
    )
    ap.add_argument(
        "--all-models-out",
        default="outputs/leaderboards/aspects_only_all_models.csv",
        help="Optional combined CSV across all processed models.",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional list of model directory names under outputs/runs. Default: process all models.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads to use per model while rescoring report.json files.",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_failures: dict[str, list[dict[str, str]]] = {}
    model_dirs = _iter_model_dirs(runs_dir, args.models)
    if not model_dirs:
        raise SystemExit(f"No model directories found under {runs_dir}")

    for model_dir in model_dirs:
        rows, failed = _latest_rows_for_model(model_dir, workers=max(1, int(args.workers)))
        out_csv = out_dir / f"{model_dir.name}.csv"
        _write_csv(out_csv, rows)
        all_rows.extend(rows)
        if failed:
            all_failures[model_dir.name] = failed
        print(f"{model_dir.name}: rows={len(rows)} failed={len(failed)} csv={out_csv}")

    if args.all_models_out:
        _write_csv(Path(args.all_models_out), all_rows)
        print(f"all_models_csv={args.all_models_out} rows={len(all_rows)}")

    if all_failures:
        fail_path = out_dir / "rescore_failures.json"
        fail_path.write_text(json.dumps(all_failures, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"failures_json={fail_path}")


if __name__ == "__main__":
    main()
