from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset import resolve_hidden_truth_path
from src.metrics.evaluator import evaluate_full, top_failures
from src.oracle.rules import evaluate_oracle
from src.planner.schema import WorkflowPlan
from src.reporting.reporter import leaderboard_row_from_report_payload, write_leaderboard, write_run_report


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _find_dataset_dir(dataset_id: str) -> Path:
    if not dataset_id:
        return Path()
    candidates = [p for p in Path("data").glob(f"**/{dataset_id}") if p.is_dir()]
    if not candidates:
        return Path()
    preferred = [p for p in candidates if (p / "abundance.csv").exists() and (p / "sample_metadata.csv").exists()]
    if preferred:
        candidates = preferred
    candidates = sorted(candidates, key=lambda p: len(p.parts), reverse=True)
    return candidates[0]


def _resolve_existing_path(path_str: str, dataset_id: str) -> Path:
    raw = str(path_str or "").strip()
    if not raw:
        p = Path()
    else:
        p = Path(raw)
    if p.exists():
        return p
    dataset_dir = _find_dataset_dir(dataset_id)
    if not raw:
        if dataset_dir:
            return dataset_dir
        return p
    if p.parts:
        tail = Path(*p.parts[-3:])
        if tail.exists():
            return tail
    if dataset_dir:
        direct = dataset_dir / p.name
        if direct.exists():
            return direct
    matches = list(Path("data").glob(f"**/{dataset_id}/{p.name}"))
    if matches:
        matches = sorted(matches, key=lambda x: len(x.parts), reverse=True)
        return matches[0]
    return p


def _recover_paths(payload: dict, state: dict, run_dir: Path) -> tuple[str, str, str, str, str]:
    trace = payload.get("execution_trace") or state.get("execution_trace") or []
    artifacts = payload.get("artifacts") or state.get("artifacts") or {}
    dataset_id = str(payload.get("dataset_id") or run_dir.name.split("__")[-1])
    raw_csv = ""
    for step in trace:
        if step.get("tool") == "inspect_dataset" and step.get("input"):
            raw_csv = str(step["input"])
            break
    if not raw_csv and trace:
        raw_csv = str(trace[0].get("input") or "")
    processed_csv = str(artifacts.get("abundance") or raw_csv)
    de_csv = str(artifacts.get("run_differential_expression") or processed_csv)
    metadata_csv = str(artifacts.get("metadata") or "")
    if not metadata_csv:
        for step in trace:
            meta = (step.get("params") or {}).get("metadata_csv")
            if meta:
                metadata_csv = str(meta)
                break
    raw_csv = str(_resolve_existing_path(raw_csv, dataset_id))
    processed_csv = str(_resolve_existing_path(processed_csv, dataset_id))
    de_csv = str(_resolve_existing_path(de_csv, dataset_id))
    metadata_csv = str(_resolve_existing_path(metadata_csv, dataset_id))
    if raw_csv and Path(raw_csv).is_dir():
        raw_csv = ""
    if processed_csv and Path(processed_csv).is_dir():
        processed_csv = ""
    if de_csv and Path(de_csv).is_dir():
        de_csv = ""
    if metadata_csv and Path(metadata_csv).is_dir():
        metadata_csv = ""
    dataset_dir = _find_dataset_dir(dataset_id)
    if (not raw_csv or not Path(raw_csv).exists()) and dataset_dir:
        raw_csv = str(dataset_dir / "abundance.csv")
        if not metadata_csv or not Path(metadata_csv).exists():
            metadata_csv = str(dataset_dir / "sample_metadata.csv")
    if dataset_dir:
        if not processed_csv or not Path(processed_csv).exists():
            processed_csv = str(dataset_dir / "abundance.csv")
        if not de_csv or not Path(de_csv).exists():
            de_csv = str(dataset_dir / "abundance.csv")
    dataset_root = dataset_dir if dataset_dir else Path(raw_csv).resolve().parent
    hidden_truth = str(resolve_hidden_truth_path(dataset_root))
    if not Path(hidden_truth).exists():
        dataset_root = _find_dataset_dir(dataset_id) or _resolve_existing_path(str(dataset_root), dataset_id)
        hidden_truth = str(resolve_hidden_truth_path(dataset_root))
    return raw_csv, processed_csv, de_csv, metadata_csv, hidden_truth


def _rescore_run(report_path: Path) -> dict:
    run_dir = report_path.parent
    payload = _load_json(report_path)
    state = _load_json(run_dir / "run_state.json")
    plan_payload = payload.get("plan") or state.get("plan") or {}
    plan = WorkflowPlan.model_validate(plan_payload)
    trace = payload.get("execution_trace") or state.get("execution_trace") or []
    decision_log = payload.get("decision_log") or state.get("decision_log") or []
    raw_csv, processed_csv, de_csv, metadata_csv, hidden_truth = _recover_paths(payload, state, run_dir)
    metrics = evaluate_full(
        plan,
        trace,
        raw_csv=raw_csv,
        processed_csv=processed_csv,
        de_csv=de_csv,
        metadata_csv=metadata_csv,
        hidden_truth_json=hidden_truth,
        had_hard_errors=any(step.get("status") != "ok" for step in trace),
        plan_valid=bool(payload.get("plan_valid", state.get("status") == "completed")),
        decision_log=decision_log,
    )
    payload["metrics"] = metrics
    payload["top_failures"] = top_failures(metrics)
    payload["execution_trace"] = trace
    payload["decision_log"] = decision_log
    payload["oracle"] = payload.get("oracle") or evaluate_oracle(plan, hidden_truth)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_run_report(str(run_dir), payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Rescore saved outputs/runs/*/report.json with the new aspect-based evaluator.")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--leaderboard-out", default="outputs/leaderboards/leaderboard.csv")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    reports = sorted(runs_dir.glob("**/report.json"))
    rows = []
    failed: list[dict[str, str]] = []
    for report_path in reports:
        try:
            payload = _rescore_run(report_path)
            rows.append(leaderboard_row_from_report_payload(payload))
        except Exception as exc:  # noqa: BLE001
            failed.append({"report_json": str(report_path), "error": str(exc)})
    write_leaderboard(args.leaderboard_out, rows)
    if failed:
        fail_path = Path(args.leaderboard_out).with_name("rescore_failures.json")
        fail_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rescored={len(rows)} failed={len(failed)} leaderboard={args.leaderboard_out}")


if __name__ == "__main__":
    main()
