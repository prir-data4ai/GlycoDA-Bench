"""DE on GlycoForge-style wide matrix: glycans x samples with glycan_index column."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.evaluator import _normalize_de_feature_id, _normalize_truth, _result_metrics
from src.tools.impl import impute_missing_values, run_differential_expression


def test_run_de_wide_produces_g_prefixed_features(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "glycan_index": [1, 2, 3],
            "s1": [1.0, 2.0, 3.0],
            "s2": [1.1, 2.1, 3.1],
            "s3": [2.0, 1.0, 4.0],
            "s4": [2.1, 1.1, 4.1],
        }
    )
    meta = pd.DataFrame({"group_label": [0, 0, 1, 1], "batch_label": [0, 0, 0, 0]})
    inp = tmp_path / "x.csv"
    m = tmp_path / "m.csv"
    out = tmp_path / "de.csv"
    df.to_csv(inp, index=False)
    meta.to_csv(m, index=False)
    path = run_differential_expression(str(inp), str(out), str(m), alpha=0.5)
    de = pd.read_csv(path)
    feats = set(de["feature"].tolist())
    assert feats == {"G001", "G002", "G003"}


def test_evaluator_matches_g_truth_with_padding() -> None:
    assert _normalize_de_feature_id("G19") == "G019"
    assert _normalize_de_feature_id("g3") == "G003"


def test_result_metrics_de_f1_glycan_truth(tmp_path: Path) -> None:
    """Truth uses G002; DE from wide matrix names G002 — de_f1 > 0. Raw/proc are sample×feature for PVCA."""
    df = pd.DataFrame(
        {
            "glycan_index": [1, 2],
            "a": [10.0, 0.1],
            "b": [10.0, 0.1],
            "c": [0.1, 50.0],
            "d": [0.1, 50.0],
        }
    )
    meta = pd.DataFrame({"group_label": [0, 0, 1, 1], "batch_label": [0, 0, 0, 0]})
    mpath = tmp_path / "meta.csv"
    depath = tmp_path / "de.csv"
    truth_path = tmp_path / "truth.json"
    meta.to_csv(mpath, index=False)
    wide = tmp_path / "wide.csv"
    df.to_csv(wide, index=False)
    run_differential_expression(str(wide), str(depath), str(mpath), alpha=0.05)

    truth_path.write_text(
        '{"dataset_id":"t","true_differential":{"feature_ids":["G002"]}}',
        encoding="utf-8",
    )
    raw = pd.DataFrame(np.random.RandomState(0).rand(4, 3))
    proc = raw * 1.01
    rawp = tmp_path / "raw.csv"
    procp = tmp_path / "proc.csv"
    raw.to_csv(rawp, index=False)
    proc.to_csv(procp, index=False)
    truth = _normalize_truth(json.loads(truth_path.read_text(encoding="utf-8")))
    res = _result_metrics(str(rawp), str(procp), str(depath), str(mpath), truth)
    assert res.get("de_f1", 0.0) > 0.0


def test_left_censored_imputation_uses_featurewise_low_tail(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "G001": [1.0, 2.0, np.nan, np.nan],
            "G002": [10.0, 20.0, np.nan, 40.0],
        }
    )
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    df.to_csv(inp, index=False)

    path = impute_missing_values(str(inp), str(out), strategy="left_censored_small_value")
    got = pd.read_csv(path)

    # The old implementation filled every missing entry with the same tiny constant.
    # The improved left-censored path should respect per-feature low tails.
    assert got.loc[2, "G001"] > 0
    assert got.loc[2, "G002"] > got.loc[2, "G001"]
    assert got["G001"].isna().sum() == 0
    assert got["G002"].isna().sum() == 0
