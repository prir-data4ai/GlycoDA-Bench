from __future__ import annotations

import argparse

import pandas as pd
from src.metrics.aspect_evaluator import ASPECT_NAMES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--leaderboard", default="outputs/leaderboards/leaderboard.csv")
    p.add_argument(
        "--detail",
        action="store_true",
        help="Also print flattened aspect detail columns.",
    )
    args = p.parse_args()
    df = pd.read_csv(args.leaderboard)
    if df.empty:
        print("No leaderboard rows.")
        return
    cols = [c for c in ["model_name", "dataset_id", *ASPECT_NAMES, "oracle_verdict"] if c in df.columns]
    if args.detail:
        for c in df.columns:
            if "__" in c:
                if c not in cols:
                    cols.append(c)
    sort_cols = [c for c in ["model_name", "dataset_id"] if c in df.columns]
    shown = df.sort_values(sort_cols) if sort_cols else df
    print(shown[cols].to_string(index=False))


if __name__ == "__main__":
    main()
