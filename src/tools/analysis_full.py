"""Full statistical implementations for benchmark tools (ComBat, FDR, OLS batch, PCA, Harmony, ORA).

Fallbacks remain when optional deps (e.g. harmonypy+torch) are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values (FDR) via statsmodels."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(pvalues, dtype=float)
    if len(p) == 0:
        return p
    _, padj, _, _ = multipletests(p, method="fdr_bh")
    return np.asarray(padj, dtype=float)


def fillna_column_median(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float).copy()
    med = np.nanmedian(x, axis=0)
    global_med = np.nanmedian(x)
    if not np.isfinite(global_med):
        global_med = 0.0
    med = np.where(np.isfinite(med), med, global_med)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])
    x[~np.isfinite(x)] = global_med
    return x


def ensure_finite_matrix(arr: np.ndarray, fill_value: float | None = None) -> np.ndarray:
    """Replace NaN/inf using column medians, then a global fallback."""
    x = fillna_column_median(arr)
    if fill_value is None:
        fill_value = np.nanmedian(x)
    if not np.isfinite(fill_value):
        fill_value = 0.0
    x[~np.isfinite(x)] = float(fill_value)
    return x


def combat_pycombat(
    Y: np.ndarray,
    batch: np.ndarray,
    X_bio: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Empirical-Bayes ComBat (Johnson et al.) via pycombat."""
    from pycombat.pycombat import Combat

    combat = Combat(mode="p")
    if X_bio is not None and X_bio.size > 0:
        Y_out = combat.fit_transform(Y, batch, X=X_bio)
    else:
        Y_out = combat.fit_transform(Y, batch)
    return Y_out, "pycombat_eb"


def combat_residual_numpy(
    xf: pd.DataFrame,
    meta: pd.DataFrame,
    batch_col: str,
    label_col: str | None,
    use_bio_covariate: bool,
) -> pd.DataFrame:
    """Residual-based batch removal (preserves group means when use_bio_covariate)."""
    batches = meta[batch_col].astype(str).values
    work = pd.DataFrame(ensure_finite_matrix(np.asarray(xf, dtype=float)), columns=xf.columns, index=xf.index)
    out = work.copy()
    if use_bio_covariate and label_col and label_col in meta.columns:
        y = meta[label_col].astype(int).values
        group_means = {g: work.loc[y == g, :].mean(axis=0) for g in np.unique(y)}
        residuals = work.copy()
        for g, gm in group_means.items():
            residuals.loc[y == g, :] = work.loc[y == g, :] - gm
        for b in np.unique(batches):
            idx = batches == b
            batch_effect = residuals.loc[idx, :].mean(axis=0)
            out.loc[idx, :] = work.loc[idx, :] - batch_effect
    else:
        global_mean = work.mean(axis=0)
        for b in np.unique(batches):
            idx = batches == b
            batch_mean = work.loc[idx, :].mean(axis=0)
            out.loc[idx, :] = work.loc[idx, :] - batch_mean + global_mean
    out = pd.DataFrame(ensure_finite_matrix(np.asarray(out, dtype=float)), columns=out.columns, index=out.index)
    return out


def limma_ols_batch_remove(
    xf: pd.DataFrame,
    meta: pd.DataFrame,
    batch_col: str,
    label_col: str | None,
) -> pd.DataFrame:
    """Per-feature OLS: y ~ 1 + batch + group; subtract fitted batch contribution only."""
    import statsmodels.api as sm

    x = xf.astype(float).copy()
    n = len(x)
    batch_d = pd.get_dummies(meta[batch_col].astype(str), prefix="b", drop_first=True)
    parts: list[np.ndarray] = [np.ones((n, 1)), batch_d.values]
    if label_col and label_col in meta.columns:
        g = meta[label_col].astype(int).values
        g_d = pd.get_dummies(pd.Series(g), prefix="grp", drop_first=True)
        if g_d.shape[1] > 0:
            parts.append(g_d.values)
    Xmat = np.hstack(parts)
    n_batch_cols = batch_d.shape[1]
    b_slice = slice(1, 1 + n_batch_cols)
    out = x.copy()
    for col in x.columns:
        yv = x[col].values.astype(float)
        mask = ~np.isnan(yv)
        if mask.sum() < Xmat.shape[1] + 1:
            continue
        try:
            res = sm.OLS(yv[mask], Xmat[mask]).fit()
            beta = res.params
            batch_contrib = Xmat[:, b_slice] @ beta[b_slice]
            out[col] = yv - batch_contrib + np.nanmean(yv)
        except Exception:  # noqa: BLE001
            continue
    return out


def pca_explained_variance(x: pd.DataFrame, n_components: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """PCA via sklearn; returns summary table and meta."""
    arr = ensure_finite_matrix(np.asarray(x, dtype=float))
    n_samples, n_features = arr.shape
    k = n_components
    if k is None:
        k = min(n_samples, n_features, 50)
    k = max(1, min(k, n_samples, n_features))
    pca = PCA(n_components=k, svd_solver="full" if min(n_samples, n_features) < 5000 else "randomized", random_state=0)
    pca.fit(arr)
    evr = pca.explained_variance_ratio_
    rows = []
    for i in range(len(evr)):
        rows.append(
            {
                "pc": i + 1,
                "explained_variance_ratio": float(evr[i]),
                "explained_variance": float(pca.explained_variance_[i]),
            }
        )
    meta = {"n_components": k, "implementation": "sklearn_pca"}
    return pd.DataFrame(rows), meta


def harmony_correct(
    xf: pd.DataFrame,
    meta: pd.DataFrame,
    batch_col: str,
) -> tuple[np.ndarray, str]:
    """Harmony on PCA embedding; returns corrected data in original feature space (approximate)."""
    arr = ensure_finite_matrix(np.asarray(xf, dtype=float))
    n_samples, n_feat = arr.shape
    if len(np.unique(meta[batch_col].astype(str))) < 2:
        return arr, "skipped_single_batch"

    scaler = StandardScaler()
    Xs = scaler.fit_transform(arr)
    npc = max(2, min(20, n_samples - 1, n_feat))
    pca = PCA(n_components=npc, random_state=0)
    Z = pca.fit_transform(Xs)

    meta_df = meta[[batch_col]].copy()
    batch_key = batch_col

    try:
        import harmonypy as hm  # noqa: WPS433

        ho = hm.run_harmony(Z, meta_df, batch_key, verbose=False, max_iter_harmony=10)
        Z_corr = ho.Z_corr
        variant = "harmonypy_torch"
    except Exception:  # noqa: BLE001
        from pycombat.pycombat import Combat

        b = meta[batch_col].astype(str).values
        combat = Combat(mode="p")
        Z_corr = combat.fit_transform(Z, b)
        variant = "pc_space_pycombat_fallback"

    Xs_corr = pca.inverse_transform(Z_corr)
    out = scaler.inverse_transform(Xs_corr)
    return out, variant


def clr_closed(
    sxf: pd.DataFrame,
    pseudocount: float,
) -> np.ndarray:
    """Standard CLR: close rows to proportions, add pseudocount, center log abundances."""
    arr = np.asarray(sxf, dtype=float)
    row_med = np.nanmedian(arr, axis=1, keepdims=True)
    row_med = np.where(np.isfinite(row_med), row_med, 0.0)
    arr = np.where(np.isnan(arr), row_med, arr)
    arr[~np.isfinite(arr)] = 0.0
    row_sum = np.nansum(arr, axis=1, keepdims=True)
    comp = arr / np.maximum(row_sum, 1e-12)
    comp = np.maximum(comp, pseudocount)
    comp = comp / np.sum(comp, axis=1, keepdims=True)
    log_c = np.log(comp)
    return log_c - np.mean(log_c, axis=1, keepdims=True)


def motif_ora_hypergeom(
    de_df: pd.DataFrame,
    feature_meta: pd.DataFrame,
    tag_col: str,
) -> pd.DataFrame:
    """Fisher exact one-sided enrichment per tag (2x2: DE vs not, in-set vs out)."""
    if tag_col not in feature_meta.columns:
        return pd.DataFrame()

    sig = set(de_df.loc[de_df["significant"] == True, "feature"].astype(str))  # noqa: E712
    meta = feature_meta.copy()
    meta["feature"] = meta["glycan_id"].astype(str)
    meta[tag_col] = meta[tag_col].astype(str).replace("", np.nan)
    meta = meta.dropna(subset=[tag_col])
    universe = set(meta["feature"].astype(str))
    N = len(universe)
    K = len(sig & universe)
    rows = []
    for tag in meta[tag_col].unique():
        in_tag = set(meta.loc[meta[tag_col] == tag, "feature"].astype(str))
        n_path = len(in_tag & universe)
        if n_path == 0:
            continue
        k = len(sig & in_tag)
        a = k
        b = K - k
        c = n_path - k
        d = (N - n_path) - (K - k)
        if min(a, b, c, d) < 0:
            continue
        try:
            _, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        except ValueError:
            p = 1.0
        rows.append(
            {
                "tag": tag,
                "n_features": n_path,
                "n_sig_overlap": k,
                "pvalue": float(max(p, 1e-300)),
                "enrichment_score": k / max(n_path, 1),
            }
        )
    if not rows:
        return pd.DataFrame()
    odf = pd.DataFrame(rows).sort_values("pvalue")
    odf["padj"] = benjamini_hochberg(odf["pvalue"].values)
    odf["significant"] = odf["padj"] < 0.05
    return odf
