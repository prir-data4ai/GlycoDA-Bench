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

from src.app.rescore_outputs import _load_json, _recover_paths
from src.metrics.aspect_evaluator import _bio_preservation_score, _load_truth_cached, _result_metrics, _safe_unit_interval


def _iter_model_dirs(runs_dir: Path, models: list[str]) -> list[Path]:
    wanted = set(models)
    dirs = [p for p in sorted(runs_dir.iterdir()) if p.is_dir()]
    if wanted:
        dirs = [p for p in dirs if p.name in wanted]
    return dirs


def _compute_one(report_path: Path) -> tuple[str, dict] | tuple[None, dict[str, str]]:
    try:
        run_dir = report_path.parent
        payload = _load_json(report_path)
        state = _load_json(run_dir / "run_state.json")
        dataset_id = str(payload.get("dataset_id") or run_dir.name.split("__")[-1])
        raw_csv, processed_csv, de_csv, metadata_csv, hidden_truth = _recover_paths(payload, state, run_dir)
        res = _result_metrics(raw_csv, processed_csv, de_csv, metadata_csv, _load_truth_cached(hidden_truth))
        bio_truth = float(res.get("bio_truth_retention_score", 0.0))
        bio_group = float(res.get("bio_group_separation_score", 0.0))
        bio_struct = _safe_unit_interval(max(float(res.get("corr_structure_preservation", 0.0)), 0.0))
        bio_score = _bio_preservation_score(bio_truth, bio_group, bio_struct)
        return dataset_id, {
            "model_name": str(payload.get("model_name") or run_dir.parent.name),
            "dataset_id": dataset_id,
            "biological_signal_preservation": float(bio_score),
        }
    except Exception as exc:  # noqa: BLE001
        return None, {"report_json": str(report_path), "error": str(exc)}


def _rows_for_model(model_dir: Path, workers: int) -> tuple[list[dict], list[dict[str, str]]]:
    reports = sorted(model_dir.glob("**/report.json"), key=lambda p: p.stat().st_mtime)
    latest_by_dataset: dict[str, dict] = {}
    failed: list[dict[str, str]] = []
    if workers <= 1:
        for report_path in reports:
            dataset_id, result = _compute_one(report_path)
            if dataset_id is None:
                failed.append(result)
            else:
                latest_by_dataset[dataset_id] = result
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_compute_one, report_path): report_path for report_path in reports}
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
    fieldnames = ["model_name", "dataset_id", "biological_signal_preservation"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute only biological_signal_preservation for outputs/runs and write one CSV per model.")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--out-dir", default="outputs/leaderboards/biological_signal_preservation_only")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=[])
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_failures: dict[str, list[dict[str, str]]] = {}
    model_dirs = _iter_model_dirs(runs_dir, args.models)
    if not model_dirs:
        raise SystemExit(f"No model directories found under {runs_dir}")

    for model_dir in model_dirs:
        rows, failed = _rows_for_model(model_dir, workers=max(1, int(args.workers)))
        out_csv = out_dir / f"{model_dir.name}.csv"
        _write_csv(out_csv, rows)
        if failed:
            all_failures[model_dir.name] = failed
        print(f"{model_dir.name}: rows={len(rows)} failed={len(failed)} csv={out_csv}")

    if all_failures:
        fail_path = out_dir / "failures.json"
        fail_path.write_text(json.dumps(all_failures, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"failures_json={fail_path}")


if __name__ == "__main__":
    main()
