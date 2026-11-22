from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.tools import analysis_full as af

IMPLEMENTATION_NOTE = "full_implementation_v1"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _glycan_feature_col(df: pd.DataFrame) -> str | None:
    if "glycan_id" in df.columns:
        return "glycan_id"
    if "glycan_index" in df.columns:
        return "glycan_index"
    return None



def _is_glycan_wide(df: pd.DataFrame) -> bool:
    """Detect GlycoForge wide layout: rows=glycans, first col is glycan_id or glycan_index."""
    return _glycan_feature_col(df) is not None


def _to_samples_x_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], bool]:
    """Normalise any matrix to samples x features.

    Returns (sample_x_feature_df, feature_names, was_transposed).
    For GlycoForge wide matrices the feature names are preserved as glycan IDs.
    """
    feature_col = _glycan_feature_col(df)
    if feature_col is not None:
        feature_ids = df[feature_col].astype(str).tolist()
        mat = df.drop(columns=[feature_col]).apply(pd.to_numeric, errors="coerce")
        t = mat.T.copy()
        t.columns = feature_ids
        t = t.reset_index(drop=True)
        return t, list(t.columns), True
    return df.apply(pd.to_numeric, errors="coerce"), list(df.columns), False


def _from_samples_x_features(
    sxf: pd.DataFrame, original_df: pd.DataFrame, was_transposed: bool
) -> pd.DataFrame:
    """Convert back to the original layout so downstream tools see the same shape."""
    if was_transposed:
        feature_col = _glycan_feature_col(original_df) or "glycan_id"
        gi = original_df[feature_col]
        mat = sxf.T.copy()
        mat.insert(0, feature_col, gi.values)
        mat.columns = original_df.columns
        return mat
    return sxf


def _write_sidecar_meta(output_csv: str, meta: dict) -> None:
    p = Path(output_csv)
    side = p.with_suffix(p.suffix + ".meta.json")
    meta = {**meta, "implementation_note": meta.get("implementation_note", IMPLEMENTATION_NOTE)}
    _ensure_parent(side)
    side.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def inspect_dataset(
    input_csv: str,
    output_csv: str,
    metadata_csv: str | None = None,
    **_: dict,
) -> str:
    raw_df = pd.read_csv(input_csv)
    sxf, feat_names, was_transposed = _to_samples_x_features(raw_df)
    arr = np.asarray(sxf, dtype=float)
    n_samples, n_features = sxf.shape
    miss = float(np.isnan(arr).sum() / max(arr.size, 1))
    zero_frac = float((arr == 0).sum() / max(arr.size, 1)) if arr.size else 0.0
    min_v = float(np.nanmin(arr)) if arr.size else 0.0
    max_v = float(np.nanmax(arr)) if arr.size else 0.0
    mean_v = float(np.nanmean(arr)) if arr.size else 0.0
    row_sums = np.nansum(arr, axis=1)
    const_like = float(np.std(row_sums) / max(np.mean(np.abs(row_sums)), 1e-12)) if len(row_sums) else 1.0
    composition_like = bool(const_like < 0.05 and n_features > 1)

    batch_ok = False
    group_ok = False
    if metadata_csv and Path(metadata_csv).exists():
        meta = pd.read_csv(metadata_csv)
        batch_ok = bool({"batch_label", "batch"} & set(meta.columns))
        group_ok = bool({"group_label", "label"} & set(meta.columns))

    out = pd.DataFrame(
        {
            "n_samples": [n_samples],
            "n_features": [n_features],
            "missing_fraction": [miss],
            "zero_fraction": [zero_frac],
            "min_value": [min_v],
            "max_value": [max_v],
            "mean_value": [mean_v],
            "rows_sum_near_constant_like_composition": [composition_like],
            "batch_label_available": [batch_ok],
            "group_label_available": [group_ok],
        }
    )
    p = Path(output_csv)
    _ensure_parent(p)
    out.to_csv(p, index=False)
    detail = {
        "tool": "inspect_dataset",
        "input_csv": input_csv,
        "n_samples": n_samples,
        "n_features": n_features,
        "missing_fraction": miss,
        "composition_proxy_std_relative_to_row_sum": float(const_like),
        "implementation_note": IMPLEMENTATION_NOTE,
    }
    _write_sidecar_meta(str(p), detail)
    return str(p)


def impute_missing_values(input_csv: str, output_csv: str, strategy: str = "median", **_: dict) -> str:
    raw_df = pd.read_csv(input_csv)
    x, _, was_transposed = _to_samples_x_features(raw_df)
    if strategy == "median":
        out = x.fillna(x.median(numeric_only=True))
    elif strategy == "left_censored_small_value":
        out = x.copy()
        for col in out.columns:
            s = out[col].astype(float)
            miss = s.isna()
            if not bool(miss.any()):
                continue
            obs = s.loc[~miss]
            if obs.empty:
                fill = 1e-12
            else:
                q10 = float(obs.quantile(0.10))
                low_tail = obs.loc[obs <= q10]
                fill = float(low_tail.median()) if not low_tail.empty else float(obs.min())
                fill = max(fill, 1e-12)
            out.loc[miss, col] = fill
    else:
        out = x.fillna(0.0)
    result = _from_samples_x_features(out, raw_df, was_transposed)
    p = Path(output_csv)
    _ensure_parent(p)
    result.to_csv(p, index=False)
    _write_sidecar_meta(str(p), {"tool": "impute_missing_values", "strategy": strategy})
    return str(p)


def clr_transform(
    input_csv: str,
    output_csv: str,
    pseudocount: float = 1e-6,
    samples_are_rows: bool = True,
    **_: dict,
) -> str:
    """
    CLR on compositional data.  Automatically detects GlycoForge wide layout
    (glycan_index present → rows=glycans, cols=samples) and normalises to
    samples×features before CLR, then writes back in the original layout.
    """
    raw_df = pd.read_csv(input_csv)
    sxf, _, was_transposed = _to_samples_x_features(raw_df)
    work = sxf
    if not samples_are_rows and not was_transposed:
        work = sxf.T
    clr = af.clr_closed(work, float(pseudocount))
    clr_df = pd.DataFrame(clr, columns=work.columns, index=work.index)
    if not samples_are_rows and not was_transposed:
        clr_df = clr_df.T
    result = _from_samples_x_features(clr_df, raw_df, was_transposed)
    p = Path(output_csv)
    _ensure_parent(p)
    result.to_csv(p, index=False)
    _write_sidecar_meta(
        str(p),
        {
            "tool": "clr_transform",
            "samples_are_rows": samples_are_rows,
            "pseudocount": pseudocount,
            "variant": "standard_clr_closure",
            "implementation_note": IMPLEMENTATION_NOTE,
        },
    )
    return str(p)


def check_batch_effect(input_csv: str, output_csv: str, metadata_csv: str, **_: dict) -> str:
    raw_df = pd.read_csv(input_csv)
    x, _, _ = _to_samples_x_features(raw_df)
    meta = pd.read_csv(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else "batch"
    batches = meta[batch_col].astype(str).values
    vals = []
    per_shift = []
    for c in x.columns:
        groups = [x.loc[batches == b, c].dropna().values for b in np.unique(batches)]
        if sum(len(g) > 1 for g in groups) < 2:
            continue
        f, _ = stats.f_oneway(*groups)
        vals.append(float(f))
        means = [np.nanmean(g) for g in groups]
        per_shift.append(float(np.nanstd(means)))
    batch_effect_score = float(np.tanh(np.mean(vals) / 10.0)) if vals else 0.0
    per_batch_mean_shift = float(np.nanmean(per_shift)) if per_shift else 0.0

    xm = np.asarray(x, dtype=float)
    xm = np.where(np.isnan(xm), np.nanmean(xm, axis=0), xm)
    if len(np.unique(batches)) >= 2 and xm.shape[1] >= 2:
        u = np.unique(batches)
        if len(u) == 2:
            m0 = np.nanmean(xm[batches == u[0]], axis=0)
            m1 = np.nanmean(xm[batches == u[1]], axis=0)
            pca_sep = float(np.linalg.norm(m0 - m1) / max(np.linalg.norm(np.nanstd(xm, axis=0)), 1e-12))
        else:
            grand = np.nanmean(xm, axis=0)
            pca_sep = float(np.nanmean([np.linalg.norm(np.nanmean(xm[batches == b], axis=0) - grand) for b in u]))
    else:
        pca_sep = 0.0

    out = pd.DataFrame(
        {
            "batch_effect_score": [batch_effect_score],
            "per_batch_mean_shift": [per_batch_mean_shift],
            "pca_separation_proxy": [pca_sep],
        }
    )
    p = Path(output_csv)
    _ensure_parent(p)
    out.to_csv(p, index=False)
    _write_sidecar_meta(str(p), {"tool": "check_batch_effect", "implementation_note": IMPLEMENTATION_NOTE})
    return str(p)


def run_combat(input_csv: str, output_csv: str, metadata_csv: str, use_bio_covariate: bool = True, **_: dict) -> str:
    """
    Empirical-Bayes ComBat (Johnson et al., 2007) via ``pycombat`` when possible;
    falls back to residual-based mean removal if ComBat cannot be fit.
    """
    raw_df = pd.read_csv(input_csv)
    x, _, was_transposed = _to_samples_x_features(raw_df)
    meta = pd.read_csv(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else "batch"
    label_col = "group_label" if "group_label" in meta.columns else ("label" if "label" in meta.columns else None)
    batches = meta[batch_col].astype(str).values
    xf = x.astype(float)
    variant = "pycombat_eb"
    out: pd.DataFrame

    if len(np.unique(batches)) < 2:
        out = xf.copy()
        variant = "skipped_single_batch"
    else:
        Y = af.ensure_finite_matrix(np.asarray(xf, dtype=float))
        X_bio = None
        if use_bio_covariate and label_col and label_col in meta.columns:
            yv = meta[label_col].astype(int).values
            u = np.unique(yv)
            if len(u) >= 2:
                X_bio = np.column_stack([(yv == g).astype(float) for g in u[:-1]])
        try:
            Y_out, variant = af.combat_pycombat(Y, batches, X_bio)
            Y_out = np.asarray(Y_out, dtype=float)
            if not np.isfinite(Y_out).all():
                raise ValueError("pycombat produced non-finite values")
            out = pd.DataFrame(af.ensure_finite_matrix(Y_out), columns=xf.columns, index=xf.index)
        except Exception:  # noqa: BLE001
            out = af.combat_residual_numpy(xf, meta, batch_col, label_col, use_bio_covariate)
            variant = "residual_fallback_after_pycombat_error"

    result = _from_samples_x_features(out, raw_df, was_transposed)
    p = Path(output_csv)
    _ensure_parent(p)
    result.to_csv(p, index=False)
    _write_sidecar_meta(
        str(p),
        {"tool": "run_combat", "variant": variant, "use_bio_covariate": use_bio_covariate},
    )
    return str(p)


def run_ratio_combat(input_csv: str, output_csv: str, metadata_csv: str, **_: dict) -> str:
    raw_df = pd.read_csv(input_csv)
    x, _, was_transposed = _to_samples_x_features(raw_df)
    meta = pd.read_csv(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else "batch"
    batches = meta[batch_col].astype(str).values
    gm = x.mean(axis=0).replace(0, np.nan)
    ratio = x.div(gm, axis=1)
    out = ratio.copy()
    for b in np.unique(batches):
        idx = batches == b
        out.loc[idx, :] = out.loc[idx, :] - out.loc[idx, :].median(axis=0)
    out = out + ratio.median(axis=0)
    result = _from_samples_x_features(out, raw_df, was_transposed)
    p = Path(output_csv)
    _ensure_parent(p)
    result.to_csv(p, index=False)
    _write_sidecar_meta(str(p), {"tool": "run_ratio_combat", "variant": "simplified_ratio_median"})
    return str(p)


def run_percentile_norm(input_csv: str, output_csv: str, metadata_csv: str, **_: dict) -> str:
    raw_df = pd.read_csv(input_csv)
    x, _, was_transposed = _to_samples_x_features(raw_df)
    meta = pd.read_csv(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else "batch"
    batches = meta[batch_col].astype(str).values
    out = x.astype(float).copy()
    for b in np.unique(batches):
        idx = batches == b
        out.loc[idx, :] = out.loc[idx, :].rank(pct=True)
    result = _from_samples_x_features(out, raw_df, was_transposed)
    p = Path(output_csv)
    _ensure_parent(p)
    result.to_csv(p, index=False)
    _write_sidecar_meta(str(p), {"tool": "run_percentile_norm", "variant": "per_batch_rank_pct"})
    return str(p)


def run_limma_batch_removal(input_csv: str, output_csv: str, metadata_csv: str, **_: dict) -> str:
    """OLS per feature: y ~ batch + group; subtract batch contribution (limma-style batch removal)."""
    raw_df = pd.read_csv(input_csv)
    x, _, was_transposed = _to_samples_x_features(raw_df)
    x = x.astype(float)
    meta = pd.read_csv(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else "batch"
    label_col = "group_label" if "group_label" in meta.columns else ("label" if "label" in meta.columns else None)
    if batch_col not in meta.columns:
        result = _from_samples_x_features(x, raw_df, was_transposed)
        p = Path(output_csv)
        _ensure_parent(p)
        result.to_csv(p, index=False)
        _write_sidecar_meta(str(p), {"tool": "run_limma_batch_removal", "note": "no_batch_column"})
        return str(p)
    out = af.limma_ols_batch_remove(x, meta, batch_col, label_col)
    result = _from_samples_x_features(out, raw_df, was_transposed)
    p = Path(output_csv)
    _ensure_parent(p)
    result.to_csv(p, index=False)
    _write_sidecar_meta(str(p), {"tool": "run_limma_batch_removal", "variant": "statsmodels_ols_batch_plus_group"})
    return str(p)


def run_harmony(input_csv: str, output_csv: str, metadata_csv: str, **_: dict) -> str:
    """Harmony on PCA embedding (harmonypy when installed) with inverse transform; else PCA-space ComBat."""
    raw_df = pd.read_csv(input_csv)
    xf, _, was_transposed = _to_samples_x_features(raw_df)
    meta = pd.read_csv(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else "batch"
    arr, variant = af.harmony_correct(xf, meta, batch_col)
    out = pd.DataFrame(arr, columns=xf.columns, index=xf.index)
    result = _from_samples_x_features(out, raw_df, was_transposed)
    p = Path(output_csv)
    _ensure_parent(p)
    result.to_csv(p, index=False)
    _write_sidecar_meta(str(p), {"tool": "run_harmony", "variant": variant})
    return str(p)


def run_differential_expression(
    input_csv: str,
    output_csv: str,
    metadata_csv: str,
    alpha: float = 0.05,
    group_col: str | None = None,
    input_space: str = "auto",
    multiple_testing: str = "bh",
    **_: dict,
) -> str:
    """
    Per-feature Welch t-test; optional Benjamini–Hochberg FDR (default ``multiple_testing='bh'``).
    """
    raw_df = pd.read_csv(input_csv)
    x, _, was_transposed = _to_samples_x_features(raw_df)
    meta = pd.read_csv(metadata_csv)
    label_col = group_col if group_col and group_col in meta.columns else (
        "group_label" if "group_label" in meta.columns else "label"
    )
    de_orientation = "glycan_features_from_wide_matrix" if was_transposed else "samples_x_features"
    y = meta[label_col].astype(int).values
    if len(x) != len(meta):
        raise ValueError(
            f"run_differential_expression: samples={len(x)} metadata={len(meta)} mismatch"
        )

    def canonical_feature_id(value: object) -> str:
        text = str(value).strip()
        if text.isdigit():
            return f"G{int(text):03d}"
        if len(text) > 1 and text[0].lower() == "g" and text[1:].isdigit():
            return f"G{int(text[1:]):03d}"
        return text

    rows = []
    for c in x.columns:
        a = x.loc[y == 0, c].astype(float).values
        b = x.loc[y == 1, c].astype(float).values
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        rows.append({"feature": canonical_feature_id(c), "pvalue": float(p), "effect": float(np.nanmean(b) - np.nanmean(a))})
    odf = pd.DataFrame(rows).sort_values("pvalue")
    pvals = odf["pvalue"].values.astype(float)
    odf["padj"] = af.benjamini_hochberg(pvals) if multiple_testing.lower() in {"bh", "fdr", "benjamini-hochberg"} else pvals
    if multiple_testing.lower() in {"bh", "fdr", "benjamini-hochberg"}:
        odf["significant"] = odf["padj"] < alpha
    else:
        odf["significant"] = odf["pvalue"] < alpha
    odf["rank"] = np.arange(1, len(odf) + 1)
    p = Path(output_csv)
    _ensure_parent(p)
    odf.to_csv(p, index=False)
    _write_sidecar_meta(
        str(p),
        {
            "tool": "run_differential_expression",
            "alpha": alpha,
            "group_col": label_col,
            "input_space": input_space,
            "de_orientation": de_orientation,
            "multiple_testing": multiple_testing,
        },
    )
    return str(p)


def run_motif_enrichment(
    input_csv: str,
    output_csv: str,
    feature_metadata_csv: str | None = None,
    tag_column: str = "motif_tag",
    **_: dict,
) -> str:
    """ORA (Fisher + BH) when ``feature_metadata_csv`` maps features to tags (e.g. motif_tag, effect_role)."""
    de = pd.read_csv(input_csv)
    if "feature" not in de.columns:
        out = pd.DataFrame({"status": ["error"], "detail": ["DE table must contain column 'feature'"]})
        p = Path(output_csv)
        _ensure_parent(p)
        out.to_csv(p, index=False)
        return str(p)

    fm_path = Path(feature_metadata_csv) if feature_metadata_csv else None
    if not fm_path or not fm_path.exists():
        cand = Path(input_csv).resolve().parent.parent / "feature_metadata.csv"
        fm_path = cand if cand.exists() else None

    if not fm_path or not fm_path.exists():
        total = max(len(de), 1)
        sig = int((de["significant"] == True).sum()) if "significant" in de.columns else 0  # noqa: E712
        out = pd.DataFrame(
            {
                "motif": ["global"],
                "enrichment_score": [sig / total],
                "status": ["no_feature_metadata"],
                "implementation_note": [IMPLEMENTATION_NOTE],
            }
        )
        p = Path(output_csv)
        _ensure_parent(p)
        out.to_csv(p, index=False)
        _write_sidecar_meta(str(p), {"tool": "run_motif_enrichment", "status": "no_metadata"})
        return str(p)

    fm = pd.read_csv(fm_path)
    tag_col = tag_column
    if tag_col not in fm.columns:
        for alt in ("effect_role", "glycan_class"):
            if alt in fm.columns:
                tag_col = alt
                break
    odf = af.motif_ora_hypergeom(de, fm, tag_col)
    if odf.empty:
        out = pd.DataFrame(
            {
                "motif": ["_empty_"],
                "enrichment_score": [0.0],
                "status": ["no_tags_or_no_overlap"],
            }
        )
    else:
        out = odf.rename(columns={"tag": "motif"})
        out["status"] = "ok"
    p = Path(output_csv)
    _ensure_parent(p)
    out.to_csv(p, index=False)
    _write_sidecar_meta(
        str(p),
        {"tool": "run_motif_enrichment", "feature_metadata": str(fm_path), "tag_column": tag_col},
    )
    return str(p)


def run_pca(input_csv: str, output_csv: str, n_components: int | None = None, **_: dict) -> str:
    raw_df = pd.read_csv(input_csv)
    x, _, _ = _to_samples_x_features(raw_df)
    out, pca_meta = af.pca_explained_variance(x, n_components=n_components)
    p = Path(output_csv)
    _ensure_parent(p)
    out.to_csv(p, index=False)
    _write_sidecar_meta(str(p), {"tool": "run_pca", **pca_meta})
    return str(p)
