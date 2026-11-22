"""Aggregate per-run report.json files into one CSV (leaderboard-style)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.reporting.reporter import leaderboard_row_from_report_payload, write_leaderboard


def _row_from_report(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = leaderboard_row_from_report_payload(data)
    row["run_dir"] = path.parent.name
    row["report_json"] = str(path)
    return row


def aggregate(runs_dir: Path, name_prefix: str, out_csv: Path) -> int:
    files = sorted(runs_dir.glob("**/report.json"))
    if name_prefix:
        files = [p for p in files if p.relative_to(runs_dir).parts[0].startswith(name_prefix)]
    if not files:
        return 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [_row_from_report(p) for p in files]
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def aggregate_benchmark_layout(runs_dir: Path, name_prefix: str, out_csv: Path) -> int:
    """Rebuild run_benchmark-style leaderboard CSV from report.json files (full plan/exec sub-columns).

    When multiple reports exist for the same (model_name, dataset_id), the newest file by mtime wins.
    """
    files = sorted(runs_dir.glob("**/report.json"), key=lambda p: p.stat().st_mtime)
    if name_prefix:
        files = [p for p in files if p.relative_to(runs_dir).parts[0].startswith(name_prefix)]
    if not files:
        return 0
    by_key: dict[tuple[str, str], dict] = {}
    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        mn = str(data.get("model_name") or "")
        did = str(data.get("dataset_id") or "")
        by_key[(mn, did)] = data
    rows = [leaderboard_row_from_report_payload(d) for d in by_key.values()]
    write_leaderboard(str(out_csv), rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge outputs/runs/*/report.json into one CSV.")
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/runs"),
        help="Root under which run folders live (flat or runs/<model_name>/...)",
    )
    ap.add_argument(
        "--name-prefix",
        default="",
        help="Only include run folders whose name starts with this (e.g. 20260329)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/aggregated_leaderboard.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--benchmark-layout",
        action="store_true",
        help=(
            "Write the same columns as run_benchmark's leaderboard (plan/exec sub-metrics included). "
            "Dedupes by (model_name, dataset_id); latest report.json by mtime wins. "
            "Example: --benchmark-layout --out outputs/leaderboards/leaderboard.csv"
        ),
    )
    args = ap.parse_args()
    if args.benchmark_layout:
        n = aggregate_benchmark_layout(args.runs_dir, args.name_prefix, args.out)
    else:
        n = aggregate(args.runs_dir, args.name_prefix, args.out)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
