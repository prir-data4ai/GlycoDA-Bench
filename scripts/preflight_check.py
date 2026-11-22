"""Pre-flight validation for GlycoForge benchmark dataset directories.

Usage:
    python scripts/preflight_check.py data/glycoforge_benchmark_500 --max 10

Checks every dataset dir for:
  1. Required files exist (abundance.csv, clr.csv, sample_metadata.csv, manifest.json).
  2. Evaluator ground truth exists (legacy ``hidden_truth.json`` in the case dir, or
     ``<pack>/_private_hidden_truth/<case>.json`` — see ``resolve_hidden_truth_path``).
  3. sample_metadata rows == sample columns in abundance (accounting for glycan_index wide layout).
  4. hidden_truth feature IDs use G### format consistent with the glycan count.
  5. manifest n_samples / n_features agree with the actual matrix.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import resolve_hidden_truth_path  # noqa: E402

REQUIRED = ["abundance.csv", "clr.csv", "sample_metadata.csv", "manifest.json"]


def check_one(ds_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED:
        if not (ds_dir / name).exists():
            errors.append(f"missing {name}")

    if errors:
        return errors

    truth_path = resolve_hidden_truth_path(ds_dir)
    if not truth_path.is_file():
        errors.append(
            "missing evaluator ground truth (case/hidden_truth.json or _private_hidden_truth/<case>.json)"
        )
        return errors

    abundance = pd.read_csv(ds_dir / "abundance.csv")
    meta = pd.read_csv(ds_dir / "sample_metadata.csv")

    if "glycan_index" in abundance.columns:
        n_samples = abundance.shape[1] - 1
        n_features = abundance.shape[0]
    else:
        n_samples = abundance.shape[0]
        n_features = abundance.shape[1]

    if len(meta) != n_samples:
        errors.append(
            f"ALIGNMENT: metadata rows={len(meta)} != sample columns={n_samples}"
        )

    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    ds = manifest.get("dataset_summary", {})
    if ds.get("n_samples") and ds["n_samples"] != n_samples:
        errors.append(
            f"manifest n_samples={ds['n_samples']} != actual {n_samples}"
        )
    if ds.get("n_features") and ds["n_features"] != n_features:
        errors.append(
            f"manifest n_features={ds['n_features']} != actual {n_features}"
        )

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    td = truth.get("true_differential", {})
    feats = td.get("feature_ids", [])
    for f in feats:
        s = str(f).strip()
        if not (re.fullmatch(r"\d+", s) or re.fullmatch(r"[Gg]\d+", s)):
            errors.append(f"hidden_truth feature '{f}' is not numeric/G###")
            break

    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Benchmark root (e.g. data/glycoforge_benchmark_500)")
    ap.add_argument("--max", type=int, default=None, help="Check at most N dirs")
    ap.add_argument("--glob", default="gf_case_*", help="Dir glob pattern")
    args = ap.parse_args()

    root = Path(args.root)
    dirs = sorted(p for p in root.glob(args.glob) if p.is_dir())
    if args.max:
        dirs = dirs[: args.max]

    total = len(dirs)
    bad = 0
    for d in dirs:
        errs = check_one(d)
        if errs:
            bad += 1
            print(f"FAIL {d.name}: {'; '.join(errs)}")

    ok = total - bad
    print(f"\n{ok}/{total} passed, {bad} failed.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
