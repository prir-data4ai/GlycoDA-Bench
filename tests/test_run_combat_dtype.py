"""Regression: run_combat must not fail on integer matrices (float output)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.tools.impl import run_combat


def test_run_combat_int_matrix_use_bio_covariate_false(tmp_path: Path) -> None:
    """Previously: subtracting batch means produced floats assigned into int64 columns -> TypeError."""
    n = 6
    # All-integer feature matrix (like some pipeline outputs coerced to int).
    x = pd.DataFrame(
        {
            "a": [10, 12, 11, 9, 10, 11],
            "b": [5, 6, 5, 4, 5, 6],
        }
    )
    meta = pd.DataFrame(
        {
            "batch_label": [0, 0, 0, 1, 1, 1],
            "group_label": [0, 1, 0, 1, 0, 1],
        }
    )
    inp = tmp_path / "x.csv"
    mpath = tmp_path / "meta.csv"
    out = tmp_path / "out.csv"
    x.to_csv(inp, index=False)
    meta.to_csv(mpath, index=False)

    path = run_combat(str(inp), str(out), str(mpath), use_bio_covariate=False)
    assert Path(path).exists()
    got = pd.read_csv(path)
    assert got.shape == x.shape
    assert got.dtypes.apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
