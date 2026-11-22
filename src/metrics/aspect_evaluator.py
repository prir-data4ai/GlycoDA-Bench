from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.planner.schema import WorkflowPlan
from src.tools.impl import _to_samples_x_features
from src.tools.registry import BATCH_TOOLS

ASPECT_NAMES = [
    "core_workflow_correctness",
    "case_awareness",
    "parameter_soundness",
    "workflow_economy",
    "diagnostic_reasoning",
    "action_selection",
    "audit_closure",
    "de_set_recovery",
    "de_ranking_quality",
    "batch_removal_quality",
    "biological_signal_preservation",
]


@lru_cache(maxsize=256)
def _read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@lru_cache(maxsize=256)
def _read_csv_head_cached(path: str, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


@lru_cache(maxsize=256)
def _load_truth_cached(hidden_truth_json: str) -> dict:
    truth_obj = json.loads(Path(hidden_truth_json).read_text(encoding="utf-8"))
    return _normalize_truth(truth_obj)


def _normalize_de_feature_id(name: str) -> str:
    s = str(name).strip()
    if re.match(r"^\d+$", s):
        return f"G{int(s):03d}"
    m = re.match(r"^G(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return f"G{int(m.group(1)):03d}"
    return s


def _normalize_feature_set(ids: set[str]) -> set[str]:
    return {_normalize_de_feature_id(x) for x in ids}


def _safe_unit_interval(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _safe_nonnegative(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, value))


def _batch_reduction_score(pvca_batch_before: float, pvca_batch_after: float) -> float:
    before = _safe_nonnegative(pvca_batch_before)
    if before <= 0.0:
        return 0.0
    after = _safe_nonnegative(pvca_batch_after)
    return _safe_unit_interval((before - after) / before)


def _manuscript_outcome_scores(result_metrics: dict) -> dict[str, float]:
    """Map continuous result metrics to the four manuscript outcome metrics."""
    return {
        "de_set_recovery": _safe_unit_interval(float(result_metrics.get("de_f1", 0.0))),
        "de_ranking_quality": _safe_unit_interval(float(result_metrics.get("de_ndcg_at_20", 0.0))),
        "batch_removal_quality": _safe_unit_interval(float(result_metrics.get("batch_reduction_score", 0.0))),
        "biological_signal_preservation": _safe_unit_interval(float(result_metrics.get("bio_preservation_score", 0.0))),
    }


def bucket_score(value: float) -> float:
    x = _safe_unit_interval(value)
    if x >= 0.80:
        return 1.0
    if x >= 0.45:
        return 0.5
    return 0.0


def _core_bucket_score(value: float) -> float:
    x = _safe_unit_interval(value)
    if x >= 0.99:
        return 1.0
    if x >= (1.0 / 3.0):
        return 0.5
    return 0.0


def _piecewise_score(value: float, anchors: list[tuple[float, float]]) -> float:
    if not anchors:
        return 0.0
    x = _safe_unit_interval(value)
    pts = sorted((float(a), _safe_unit_interval(b)) for a, b in anchors)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            ratio = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
            return float(y0 + ratio * (y1 - y0))
    return pts[-1][1]


def _pvca_proxy(x: pd.DataFrame, groups: np.ndarray) -> float:
    vals = []
    for c in x.columns:
        v = x[c].astype(float).values
        gm = [np.nanmean(v[groups == g]) for g in np.unique(groups)]
        gm = [z for z in gm if np.isfinite(z)]
        if gm:
            vals.append(np.nanvar(gm))
    return float(np.nanmean(vals)) if vals else 0.0


def _variance_ratio_summary(x: pd.DataFrame, bio: np.ndarray) -> float:
    uniq = np.unique(bio)
    if len(uniq) < 2:
        return 0.0
    vals = []
    for c in x.columns:
        num = np.nanvar(x.loc[bio == uniq[1], c])
        den = np.nanvar(x.loc[bio == uniq[0], c])
        if np.isfinite(num):
            vals.append(num / max(den, 1e-12))
    return float(np.nanmean(vals)) if vals else 0.0


@lru_cache(maxsize=256)
def _batch_signal_proxy_from_files(raw_csv: str, metadata_csv: str) -> float:
    raw_df = _read_csv_cached(raw_csv)
    raw, _, _ = _to_samples_x_features(raw_df)
    meta = _read_csv_cached(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else ("batch" if "batch" in meta.columns else None)
    label_col = "group_label" if "group_label" in meta.columns else ("label" if "label" in meta.columns else None)
    if batch_col is None or label_col is None:
        return 0.0
    batch = meta[batch_col].astype(str).values
    bio = meta[label_col].astype(int).values
    pvca_batch = _pvca_proxy(raw, batch)
    pvca_bio = _pvca_proxy(raw, bio)
    dominance = _safe_nonnegative(pvca_batch / max(pvca_bio, 1e-12))
    return float(max(_safe_unit_interval(pvca_batch), _safe_unit_interval(dominance / 2.0)))


def _batch_signal_proxy_from_matrix(raw_csv: str, metadata_csv: str) -> float:
    return _batch_signal_proxy_from_files(raw_csv, metadata_csv)


def _categorize_batch_signal(value: float) -> str:
    if value >= 0.45:
        return "high"
    if value >= 0.18:
        return "borderline"
    return "low"


def _latest_decision_for_tool(decision_log: list[dict], tool: str) -> dict | None:
    for item in reversed(decision_log):
        if item.get("chosen_tool") == tool:
            return item
    return None


def _audit_item(plan: WorkflowPlan, aspect: str) -> object | None:
    aliases = {
        "missing": {"missing", "missingness"},
        "batch": {"batch"},
        "compositionality": {"compositionality", "composition"},
        "followup": {"followup", "post_de_followup", "post_de", "follow_up"},
    }
    wanted = aliases.get(str(aspect).lower(), {str(aspect).lower()})
    for item in getattr(plan, "decision_audit", []) or []:
        if str(getattr(item, "aspect", "")).strip().lower() in wanted:
            return item
    return None


def _plan_text_blob(plan: WorkflowPlan) -> str:
    parts = [plan.workflow_name, plan.rationale, plan.final_analysis_target, *plan.stop_conditions]
    for step in plan.steps:
        parts.extend([step.tool, step.purpose])
    return " ".join(str(x or "") for x in parts).lower()


def _normalize_truth(truth_obj: dict) -> dict:
    gt = dict(truth_obj)
    inner = gt.pop("ground_truth", None)
    if isinstance(inner, dict):
        gt = {**inner, **gt}
    if "true_workflow_tags" not in gt:
        gt["true_workflow_tags"] = {
            "needs_imputation": bool(gt.get("severe_missingness") or gt.get("missingness_injected")),
            "needs_clr": True,
            "needs_batch_correction": bool(gt.get("batch_effect_significant") or gt.get("batch_injected")),
            "recommended_batch_family": gt.get("recommended_batch_family", "any_batch_tool"),
            "should_preserve_bio_covariate": bool(gt.get("batch_correction_protects_bio_covariate", True)),
        }
    tags = gt["true_workflow_tags"]
    tags.setdefault("batch_decision_profile", "required" if tags.get("needs_batch_correction") else "avoid")
    tags.setdefault("missing_decision_profile", "required" if tags.get("needs_imputation") else "avoid")
    tags.setdefault("motif_followup_worthwhile", False)
    tags.setdefault("paired_strategy_required", False)
    if "true_differential" not in gt:
        feats = gt.get("true_de_features") or gt.get("de_features") or []
        gt["true_differential"] = {"feature_ids": list(feats), "ranked": []}
    if "true_conditions" not in gt:
        gt["true_conditions"] = {
            "batch_injected": bool(gt.get("batch_injected", gt.get("batch_effect_significant"))),
            "missingness_injected": bool(gt.get("missingness_injected", gt.get("severe_missingness"))),
            "bio_strength": float(gt.get("biostrength", gt.get("bio_strength", 1.0))),
            "batch_strength": float(gt.get("kmu", gt.get("batch_strength", 1.0))),
            "missing_fraction_target": float(gt.get("missing_rate_target", gt.get("missing_fraction_target", 0.0))),
            "mnar_bias": float(gt.get("missing_bias_alpha", gt.get("mnar_bias", 1.0))),
            "confounded_signal": bool(gt.get("confounded_signal", False)),
        }
    if "scoring_targets" not in gt:
        gt["scoring_targets"] = {"expected_plan_requirements": [], "disallowed_plan_patterns": []}
    return gt


def compute_average_precision(ranked_features: list[str], positives: set[str]) -> float:
    if not ranked_features or not positives:
        return 0.0
    hits = 0
    precisions: list[float] = []
    seen: set[str] = set()
    for idx, feat in enumerate(ranked_features, start=1):
        if feat in seen:
            continue
        seen.add(feat)
        if feat in positives:
            hits += 1
            precisions.append(hits / idx)
    return float(sum(precisions) / max(len(positives), 1)) if precisions else 0.0


def _compute_ndcg_at_k(ranked_features: list[str], gt_features: set[str], k: int = 20) -> float:
    # Binary-relevance NDCG over the top-k ranked features.
    if not ranked_features or not gt_features or k <= 0:
        return 0.0
    rels = [1.0 if feat in gt_features else 0.0 for feat in ranked_features[:k]]
    dcg = float(sum(rel / (np.log2(idx + 2) ** 0.3) for idx, rel in enumerate(rels)))
    ideal_len = min(len(gt_features), k)
    if ideal_len <= 0:
        return 0.0
    idcg = float(sum(1.0 / (np.log2(idx + 2) ** 0.3) for idx in range(ideal_len)))
    if idcg <= 0:
        return 0.0
    return _safe_unit_interval(dcg / idcg)


def _bio_preservation_score(truth_retention: float, group_separation: float, structure_preservation: float) -> float:
    truth_retention = _safe_unit_interval(truth_retention)
    group_separation = _safe_unit_interval(group_separation)
    structure_preservation = _safe_unit_interval(structure_preservation)
    if truth_retention <= 0.0:
        return 0.0
    if truth_retention >= 1.0 and group_separation >= 1.0:
        return 1.0
    main_mean = (truth_retention + group_separation) / 2.0
    if main_mean >= 0.75 and structure_preservation >= 0.5:
        return 0.5
    return 0.0


def _text_specificity_score(text: str, expected_terms: tuple[str, ...] = ()) -> float:
    toks = [t for t in re.findall(r"[a-z_]+", str(text or "").lower()) if len(t) >= 4]
    if not toks:
        return 0.0
    uniq = set(toks)
    lexical = min(1.0, len(uniq) / 14.0)
    expected_hits = min(1.0, sum(1 for t in expected_terms if t in uniq) / max(len(expected_terms), 1)) if expected_terms else 0.0
    return _safe_unit_interval(0.7 * lexical + 0.3 * expected_hits)


def _extract_de_ranking(de: pd.DataFrame) -> list[str]:
    if de.empty or "feature" not in de.columns:
        return []
    tmp = de.copy()
    order_cols: list[str] = []
    ascending: list[bool] = []
    if "padj" in tmp.columns:
        tmp["__padj"] = pd.to_numeric(tmp["padj"], errors="coerce").fillna(1.0)
        order_cols.append("__padj")
        ascending.append(True)
    elif "pvalue" in tmp.columns:
        tmp["__pvalue"] = pd.to_numeric(tmp["pvalue"], errors="coerce").fillna(1.0)
        order_cols.append("__pvalue")
        ascending.append(True)
    elif "rank" in tmp.columns:
        tmp["__rank"] = pd.to_numeric(tmp["rank"], errors="coerce").fillna(len(tmp) + 1)
        order_cols.append("__rank")
        ascending.append(True)
    if "effect" in tmp.columns:
        tmp["__abs_effect"] = pd.to_numeric(tmp["effect"], errors="coerce").abs().fillna(0.0)
        order_cols.append("__abs_effect")
        ascending.append(False)
    ranked = tmp.sort_values(order_cols, ascending=ascending)["feature"].astype(str).tolist() if order_cols else tmp["feature"].astype(str).tolist()
    return [_normalize_de_feature_id(x) for x in ranked]


def _feature_effect_sizes(matrix: pd.DataFrame, bio: np.ndarray) -> pd.Series:
    uniq = np.unique(bio)
    if len(uniq) < 2:
        return pd.Series(dtype=float)
    mean_a = matrix.loc[bio == uniq[0]].mean(axis=0, numeric_only=True)
    mean_b = matrix.loc[bio == uniq[1]].mean(axis=0, numeric_only=True)
    return (mean_b - mean_a).astype(float)


def _bio_truth_retention_score(raw: pd.DataFrame, proc: pd.DataFrame, bio: np.ndarray, gt: set[str]) -> float:
    if not gt:
        return 0.5
    raw_eff = _feature_effect_sizes(raw, bio)
    proc_eff = _feature_effect_sizes(proc, bio)
    common = [g for g in gt if g in raw_eff.index and g in proc_eff.index]
    if not common:
        return 0.0
    raw_vals = raw_eff.loc[common].abs().replace(0, np.nan)
    proc_vals = proc_eff.loc[common].abs()
    ratios = (proc_vals / raw_vals).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean_ratio = float(ratios.mean()) if not ratios.empty else 0.0
    if mean_ratio >= 0.70:
        return 1.0
    if mean_ratio >= 0.40:
        return 0.5
    return 0.0


def detect_overcorrection(used_batch_tool: bool, pvca_bio_before: float, pvca_bio_after: float, f_stat_corr: float, bio_truth_retention: float) -> bool:
    if not used_batch_tool:
        return False
    bio_ratio = _safe_nonnegative(pvca_bio_after / max(pvca_bio_before, 1e-12))
    return bool(bio_ratio < 0.35 or (f_stat_corr < 0.35 and bio_truth_retention == 0.0))


def evaluate_audit_closure(plan: WorkflowPlan) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}
    scored: list[float] = []
    for aspect_name in ("missing", "batch", "compositionality", "followup"):
        item = _audit_item(plan, aspect_name)
        if item is None:
            val = 0.0
        else:
            observed = bool(getattr(item, "observed_evidence", []) or [])
            judgment = bool(str(getattr(item, "judgment", "") or "").strip())
            chosen = bool(str(getattr(item, "chosen_action", "") or "").strip())
            rejected = bool(getattr(item, "rejected_actions", []) or [])
            if not rejected:
                judgment_text = str(getattr(item, "judgment", "") or "").lower()
                chosen_text = str(getattr(item, "chosen_action", "") or "").lower()
                rejected = any(tok in judgment_text or tok in chosen_text for tok in ("risk", "avoid", "reject", "instead", "alternative"))
            hits = int(observed) + int(judgment) + int(chosen) + int(rejected)
            val = 1.0 if hits == 4 else (0.5 if hits >= 2 else 0.0)
        details[f"audit_{aspect_name}_closure"] = val
        scored.append(val)
    high_count = sum(1 for v in scored if v >= 0.75)
    mid_or_high_count = sum(1 for v in scored if v >= 0.5)
    if high_count >= 3:
        return 1.0, details
    if mid_or_high_count >= 2:
        return 0.5, details
    return 0.0, details


def _result_metrics(raw_csv: str, processed_csv: str, de_csv: str, metadata_csv: str, truth: dict) -> dict:
    raw_df = _read_csv_cached(raw_csv)
    proc_df = _read_csv_cached(processed_csv)
    raw, _, _ = _to_samples_x_features(raw_df)
    proc, _, _ = _to_samples_x_features(proc_df)
    raw.columns = [_normalize_de_feature_id(c) for c in raw.columns]
    proc.columns = [_normalize_de_feature_id(c) for c in proc.columns]
    meta = _read_csv_cached(metadata_csv)
    batch_col = "batch_label" if "batch_label" in meta.columns else "batch"
    label_col = "group_label" if "group_label" in meta.columns else "label"
    batch = meta[batch_col].astype(str).values
    bio = meta[label_col].astype(int).values
    pvca_batch_before = _pvca_proxy(raw, batch)
    pvca_batch_after = _pvca_proxy(proc, batch)
    pvca_bio_before = _pvca_proxy(raw, bio)
    pvca_bio_after = _pvca_proxy(proc, bio)
    batch_dominance = _safe_nonnegative(pvca_batch_after / max(pvca_bio_after, 1e-12))
    f_before = _variance_ratio_summary(raw, bio)
    f_after = _variance_ratio_summary(proc, bio)
    f_stat_corr = _safe_nonnegative(1.0 - abs(f_after - f_before) / max(abs(f_before), 1e-12))
    corr_raw = raw.corr(numeric_only=True).fillna(0).values
    corr_proc = proc.corr(numeric_only=True).fillna(0).values
    corr_struct_pres = float(np.corrcoef(corr_raw.flatten(), corr_proc.flatten())[0, 1]) if corr_raw.size and corr_proc.size else 0.0
    if not np.isfinite(corr_struct_pres):
        corr_struct_pres = 0.0
    de = _read_csv_cached(de_csv)
    pred_raw = set(de.loc[de["significant"] == True, "feature"].astype(str).tolist()) if "significant" in de.columns else set()  # noqa: E712
    td = truth.get("true_differential", {})
    gt_raw = {str(x) for x in (td.get("feature_ids", truth.get("true_de_features", truth.get("de_features", []))) or [])}
    pred = _normalize_feature_set(pred_raw)
    gt = _normalize_feature_set(gt_raw)
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / len(pred) if pred else 0.0
    ranked = _extract_de_ranking(de)
    ap = compute_average_precision(ranked, gt)
    top10_overlap = len(set(ranked[:10]) & gt) / max(min(len(gt), 10), 1) if gt else 0.0
    top20_overlap = len(set(ranked[:20]) & gt) / max(min(len(gt), 20), 1) if gt else 0.0
    ndcg_at_20 = _compute_ndcg_at_k(ranked, gt, k=20)
    return {
        "pvca_batch_variance_before": pvca_batch_before,
        "pvca_batch_variance_after": pvca_batch_after,
        "pvca_biological_variance_before": pvca_bio_before,
        "pvca_biological_variance_after": pvca_bio_after,
        "batch_dominance": batch_dominance,
        "de_precision": precision,
        "de_recall": recall,
        "de_f1": f1,
        "de_fpr": fpr,
        "de_average_precision": ap,
        "de_top10_overlap": top10_overlap,
        "de_top20_overlap": top20_overlap,
        "de_ndcg_at_20": ndcg_at_20,
        "f_stat_corr_preservation": f_stat_corr,
        "corr_structure_preservation": corr_struct_pres,
        "batch_reduction_score": _batch_reduction_score(pvca_batch_before, pvca_batch_after),
        "bio_preservation_score": _safe_unit_interval(0.5 * f_stat_corr + 0.5 * max(corr_struct_pres, 0.0)),
        "de_recovery_score": f1,
        "bio_truth_retention_score": _bio_truth_retention_score(raw, proc, bio, gt),
        "bio_group_separation_score": bucket_score(f_stat_corr),
    }


def _tool_sequence(plan: WorkflowPlan) -> list[str]:
    return [s.tool for s in plan.steps]


def _de_index(tools: list[str]) -> int | None:
    return tools.index("run_differential_expression") if "run_differential_expression" in tools else None


def _has_batch_tool(tools: list[str]) -> bool:
    return any(t in BATCH_TOOLS for t in tools)


def _missing_profile(tags: dict, missing_fraction: float) -> str:
    return str(tags.get("missing_decision_profile") or ("required" if missing_fraction > 0.03 else ("borderline" if missing_fraction > 0 else "avoid")))


def _batch_profile(tags: dict) -> str:
    return str(tags.get("batch_decision_profile", "required" if tags.get("needs_batch_correction") else "avoid"))


def _batch_case_match(profile: str, tools: list[str]) -> float:
    has_batch = _has_batch_tool(tools)
    has_check = "check_batch_effect" in tools
    if profile in {"required", "required_confounded"}:
        return 1.0 if has_batch else (0.5 if has_check else 0.0)
    if profile == "borderline":
        if has_check and not has_batch:
            return 1.0
        if has_check and has_batch:
            return 0.5
        return 0.0
    return 1.0 if not has_batch else 0.0


def _parameter_legality_and_reasonableness(plan: WorkflowPlan, matrix_layout_hint: str = "unknown") -> tuple[float, float, float, dict[str, float]]:
    legality = 1.0
    reason_scores: list[float] = []
    method_scores: list[float] = []
    details: dict[str, float] = {}
    for idx, step in enumerate(plan.steps, start=1):
        key = f"step_{idx:02d}_{step.tool}"
        if step.tool == "clr_transform":
            try:
                pc = float(step.params.get("pseudocount", 1e-6))
            except (TypeError, ValueError):
                pc = -1.0
            legal = 1.0 if pc > 0 else 0.0
            legality = min(legality, legal)
            details[f"{key}_legal"] = legal
            if legal:
                reason_scores.append(1.0 if 1e-12 <= pc <= 1e-2 else (0.75 if pc <= 1e-1 else 0.5))
                sar = step.params.get("samples_are_rows")
                if matrix_layout_hint == "glycan_x_sample":
                    method_scores.append(1.0 if sar is False else (0.5 if sar is None else 0.0))
                else:
                    method_scores.append(1.0 if sar in {True, None} else 0.5)
        elif step.tool == "run_differential_expression":
            try:
                alpha = float(step.params.get("alpha", 0.05))
            except (TypeError, ValueError):
                alpha = -1.0
            legal = 1.0 if 0 < alpha <= 1 else 0.0
            legality = min(legality, legal)
            details[f"{key}_legal"] = legal
            if legal:
                reason_scores.append(1.0 if alpha <= 0.1 else (0.75 if alpha <= 0.2 else 0.5))
                mt = str(step.params.get("multiple_testing", "bh") or "bh").lower()
                method_scores.append(1.0 if mt in {"bh", "fdr_bh", "bonferroni", "holm"} else 0.5)
        elif step.tool == "impute_missing_values":
            strategy = str(step.params.get("strategy", "median") or "median")
            legal = 1.0 if strategy in {"median", "left_censored_small_value"} else 0.0
            legality = min(legality, legal)
            details[f"{key}_legal"] = legal
            if legal:
                reason_scores.append(1.0)
                method_scores.append(1.0)
        elif step.tool == "run_combat":
            details[f"{key}_legal"] = 1.0
            use_cov = step.params.get("use_bio_covariate")
            reason_scores.append(1.0 if use_cov is not False else 0.5)
            method_scores.append(1.0 if use_cov is True else (0.75 if use_cov is None else 0.5))
        else:
            details[f"{key}_legal"] = 1.0
    return legality, float(np.mean(reason_scores)) if reason_scores else 1.0, float(np.mean(method_scores)) if method_scores else 1.0, details


def _parameter_soundness_score(legality: float, reasonableness: float, method_specific: float) -> float:
    if float(legality) <= 0.0:
        return 0.0

    def _bucket(v: float) -> str:
        x = _safe_unit_interval(v)
        if x >= 0.75:
            return "good"
        if x >= 0.5:
            return "okay"
        return "bad"

    reason_bucket = _bucket(reasonableness)
    method_bucket = _bucket(method_specific)
    if reason_bucket == "good" and method_bucket == "good":
        return 1.0
    if reason_bucket == "bad" and method_bucket == "bad":
        return 0.0
    return 0.5


def _workflow_economy_score(redundancy_level: float, unnecessary_tool_usage: float, template_stacking: float) -> float:
    values = {
        "redundancy_level": _safe_unit_interval(redundancy_level),
        "unnecessary_tool_usage": _safe_unit_interval(unnecessary_tool_usage),
        "template_stacking": _safe_unit_interval(template_stacking),
    }
    if values["redundancy_level"] <= 0.0 or values["template_stacking"] <= 0.0:
        return 0.0
    if all(v >= 0.75 for v in values.values()):
        return 1.0
    if sum(1 for v in values.values() if v >= 0.5) >= 2:
        return 0.5
    return 0.0


def _discrete_band(value: float) -> str:
    x = _safe_unit_interval(value)
    if x >= 0.75:
        return "high"
    if x >= 0.5:
        return "mid"
    return "low"


def _count_rule_score(values: list[float], high_needed: int, mid_needed: int, hard_gates: list[float] | None = None) -> float:
    if hard_gates and any(_safe_unit_interval(v) <= 0.0 for v in hard_gates):
        return 0.0
    bands = [_discrete_band(v) for v in values]
    high_count = sum(1 for b in bands if b == "high")
    mid_or_high_count = sum(1 for b in bands if b in {"high", "mid"})
    if high_count >= high_needed:
        return 1.0
    if mid_or_high_count >= mid_needed:
        return 0.5
    return 0.0


def _diagnostic_reasoning_score(missingness_diagnosis_quality: float, batch_diagnosis_quality: float, risk_stratification_quality: float, evidence_specificity: float) -> float:
    return _count_rule_score([
        _safe_unit_interval(missingness_diagnosis_quality),
        _safe_unit_interval(batch_diagnosis_quality),
        _safe_unit_interval(risk_stratification_quality),
        _safe_unit_interval(evidence_specificity),
    ], high_needed=3, mid_needed=2)


def _action_selection_score(protocol_compliance: float, missingness_action_quality: float, batch_action_quality: float, followup_action_quality: float) -> float:
    return _count_rule_score([
        _safe_unit_interval(missingness_action_quality),
        _safe_unit_interval(batch_action_quality),
        _safe_unit_interval(followup_action_quality),
    ], high_needed=3, mid_needed=2, hard_gates=[protocol_compliance])


def _core_workflow_score(missing_required_and_done: float, clr_required_and_before_de: float, batch_required_and_before_de: float) -> float:
    return _count_rule_score([
        _safe_unit_interval(missing_required_and_done),
        _safe_unit_interval(clr_required_and_before_de),
        _safe_unit_interval(batch_required_and_before_de),
    ], high_needed=3, mid_needed=1)


def _case_awareness_score(missing_case_judgment: float, batch_case_judgment: float, compositionality_awareness: float, confounding_awareness: float) -> float:
    return _count_rule_score([
        _safe_unit_interval(missing_case_judgment),
        _safe_unit_interval(batch_case_judgment),
        _safe_unit_interval(compositionality_awareness),
        _safe_unit_interval(confounding_awareness),
    ], high_needed=3, mid_needed=2)


def _batch_removal_quality_score(batch_reduction_magnitude: float, batch_case_match: float, overcorrection_flag: float) -> float:
    return _count_rule_score([
        _safe_unit_interval(batch_reduction_magnitude),
        _safe_unit_interval(batch_case_match),
    ], high_needed=2, mid_needed=1, hard_gates=[overcorrection_flag])


def compute_plan_score(plan: WorkflowPlan, truth: dict, matrix_layout_hint: str = "unknown") -> dict:
    tags = truth.get("true_workflow_tags", {})
    tools = _tool_sequence(plan)
    text = _plan_text_blob(plan)
    assessment = plan.dataset_assessment
    missing_profile = _missing_profile(tags, float(truth.get("true_conditions", {}).get("missing_fraction_target", 0.0) or 0.0))
    batch_profile = _batch_profile(tags)
    ide = _de_index(tools)
    core_detail = {
        "missing_required_and_done": 1.0 if (missing_profile != "required" or "impute_missing_values" in tools) else 0.0,
        "clr_required_and_before_de": 1.0 if (not tags.get("needs_clr", True) or ("clr_transform" in tools and (ide is None or tools.index("clr_transform") < ide))) else 0.0,
        "batch_required_and_before_de": 1.0 if (batch_profile not in {"required", "required_confounded"} or (_has_batch_tool(tools) and (ide is None or min(i for i, t in enumerate(tools) if t in BATCH_TOOLS) < ide))) else 0.0,
    }
    core_score = _core_workflow_score(
        core_detail["missing_required_and_done"],
        core_detail["clr_required_and_before_de"],
        core_detail["batch_required_and_before_de"],
    )
    missing_j = 1.0 if assessment is None else (1.0 if (assessment.detected_missingness == (missing_profile != "avoid")) else (0.5 if missing_profile != "required" else 0.0))
    batch_j = 0.5 if assessment is None else (1.0 if (assessment.detected_batch_risk or "batch" in text or batch_profile == "avoid") else (0.5 if batch_profile == "borderline" else 0.0))
    comp = 1.0 if ("clr" in text or "compositional" in text or "clr_transform" in tools or bool(getattr(assessment, "detected_compositionality", False))) else 0.0
    conf = 1.0 if (not (tags.get("should_preserve_bio_covariate") or truth.get("true_conditions", {}).get("confounded_signal")) or any(tok in text for tok in ("confound", "covariate", "preserve", "biology"))) else 0.0
    awareness_detail = {
        "missing_case_judgment": float(missing_j),
        "batch_case_judgment": float(batch_j),
        "compositionality_awareness": float(comp),
        "confounding_awareness": float(conf),
    }
    awareness_score = _case_awareness_score(missing_j, batch_j, comp, conf)
    legal, reasonable, method_specific, legality_detail = _parameter_legality_and_reasonableness(plan, matrix_layout_hint)
    param_score = _parameter_soundness_score(legal, reasonable, method_specific)
    param_detail = {
        "parameter_legality": float(legal),
        "parameter_reasonableness": float(reasonable),
        "method_specific_settings": float(method_specific),
    }
    param_detail.update(legality_detail)
    baseline = 2 + int(bool(tags.get("needs_clr", True))) + int(missing_profile == "required") + int(batch_profile in {"required", "required_confounded"})
    redundancy = max(0, len(tools) - baseline)
    redundant = 1.0 if redundancy == 0 else (0.75 if redundancy <= 2 else (0.5 if redundancy <= 4 else 0.0))
    unnecessary = 1.0
    if batch_profile == "avoid" and _has_batch_tool(tools):
        unnecessary = 0.0
    elif batch_profile == "borderline" and _has_batch_tool(tools) and "check_batch_effect" not in tools:
        unnecessary = 0.5
    elif missing_profile == "avoid" and "impute_missing_values" in tools:
        unnecessary = 0.5
    template_terms = sum(1 for tok in ("standard workflow", "generic workflow", "default pipeline", "always") if tok in text)
    template = 1.0 if template_terms == 0 else (0.75 if template_terms == 1 else (0.5 if template_terms == 2 else 0.0))
    economy_score = _workflow_economy_score(redundant, unnecessary, template)
    economy_detail = {
        "redundancy_level": float(redundant),
        "unnecessary_tool_usage": float(unnecessary),
        "template_stacking": float(template),
    }
    return {"core_workflow_correctness": core_score, "case_awareness": awareness_score, "parameter_soundness": param_score, "workflow_economy": economy_score, "aspect_details": {"core_workflow_correctness": core_detail, "case_awareness": awareness_detail, "parameter_soundness": param_detail, "workflow_economy": economy_detail}}


def compute_execution_score(plan: WorkflowPlan, trace: list[dict], truth: dict, had_hard_errors: bool, matrix_layout_hint: str = "unknown") -> dict:
    del matrix_layout_hint
    tags = truth.get("true_workflow_tags", {})
    tools = _tool_sequence(plan)
    text = _plan_text_blob(plan)
    missing_profile = _missing_profile(tags, float(truth.get("true_conditions", {}).get("missing_fraction_target", 0.0) or 0.0))
    batch_profile = _batch_profile(tags)
    has_check = "check_batch_effect" in tools
    batch_signal = _batch_signal_proxy_from_matrix(trace[0]["input"], trace[0]["params"].get("metadata_csv")) if trace and trace[0].get("params", {}).get("metadata_csv") else 0.0
    batch_level = _categorize_batch_signal(batch_signal)
    missing_diag = 1.0 if (missing_profile == "avoid" or "missing" in text or "impute" in text) else 0.0
    batch_diag = 1.0 if ((batch_level == "high" and has_check) or (batch_level == "borderline" and has_check) or (batch_level == "low" and not has_check)) else (0.5 if batch_level != "high" else 0.0)
    risk_terms = sum(1 for tok in ("risk", "substantial", "borderline", "weak", "strong", "uncertain", "diagnostic") if tok in text)
    risk = 1.0 if risk_terms >= 3 else (0.75 if risk_terms >= 1 else (0.5 if ("batch" in text or "missing" in text) else 0.0))
    spec = _text_specificity_score(text, expected_terms=("missing", "clr", "batch", "covariate", "motif"))
    spec = 1.0 if spec >= 0.75 else (0.75 if spec >= 0.5 else (0.5 if spec > 0 else 0.0))
    diag_score = _diagnostic_reasoning_score(missing_diag, batch_diag, risk, spec)
    diag_detail = {
        "missingness_diagnosis_quality": float(missing_diag),
        "batch_diagnosis_quality": float(batch_diag),
        "risk_stratification_quality": float(risk),
        "evidence_specificity": float(spec),
    }
    ide = _de_index(tools)
    protocol = 1.0 if (trace and not had_hard_errors and all(s.get("status") == "ok" for s in trace) and tools and tools[0] == "inspect_dataset") else 0.0
    if ide is not None and "clr_transform" in tools and tools.index("clr_transform") > ide:
        protocol = 0.0
    if missing_profile == "required" and "impute_missing_values" not in tools:
        protocol = 0.0
    miss_action = 1.0 if ((missing_profile == "required" and "impute_missing_values" in tools) or (missing_profile == "avoid" and "impute_missing_values" not in tools)) else (0.5 if missing_profile == "borderline" else 0.0)
    batch_action = _batch_case_match(batch_profile, tools)
    motif_needed = bool(tags.get("motif_followup_worthwhile", False))
    followup = 1.0 if (motif_needed == ("run_motif_enrichment" in tools)) else (0.5 if not motif_needed and "run_motif_enrichment" in tools else 0.0)
    action_score = _action_selection_score(protocol, miss_action, batch_action, followup)
    action_detail = {
        "protocol_compliance": float(protocol),
        "missingness_action_quality": float(miss_action),
        "batch_action_quality": float(batch_action),
        "followup_action_quality": float(followup),
    }
    audit_score, audit_detail = evaluate_audit_closure(plan)
    audit_detail["trace_warning_count"] = float(sum(len(s.get("warnings") or []) for s in trace))
    return {"diagnostic_reasoning": diag_score, "action_selection": action_score, "audit_closure": audit_score, "aspect_details": {"diagnostic_reasoning": diag_detail, "action_selection": action_detail, "audit_closure": audit_detail}}


def compute_decision_score(plan: WorkflowPlan, trace: list[dict], decision_log: list[dict], raw_csv: str, metadata_csv: str, truth: dict) -> dict:
    del trace, truth
    tools = _tool_sequence(plan)
    batch_signal = _batch_signal_proxy_from_matrix(raw_csv, metadata_csv)
    chosen_batch = next((tool for tool in tools if tool in BATCH_TOOLS), "")
    batch_decision = _latest_decision_for_tool(decision_log, chosen_batch) if chosen_batch else None
    rationale = str((batch_decision or {}).get("rationale", "") or "") + " " + _plan_text_blob(plan)
    return {"observed_batch_signal_score": batch_signal, "decision_log_available": 1.0 if decision_log else 0.0, "decision_rationale_specificity": _text_specificity_score(rationale, expected_terms=("batch", "clr", "covariate", "missing"))}


def evaluate_full(plan: WorkflowPlan, execution_trace: list[dict], raw_csv: str, processed_csv: str, de_csv: str, metadata_csv: str, hidden_truth_json: str, had_hard_errors: bool = False, plan_valid: bool = True, decision_log: list[dict] | None = None) -> dict:
    truth = _load_truth_cached(hidden_truth_json)
    raw_head = _read_csv_head_cached(raw_csv, 1)
    layout = "glycan_x_sample" if "glycan_index" in raw_head.columns else "sample_x_feature"
    trace_for_scoring = [dict(step) for step in execution_trace]
    if trace_for_scoring:
        first = dict(trace_for_scoring[0])
        first["input"] = raw_csv
        first["params"] = {**(first.get("params") or {}), "metadata_csv": metadata_csv}
        trace_for_scoring[0] = first
    res = _result_metrics(raw_csv, processed_csv, de_csv, metadata_csv, truth)
    plan_aspects = compute_plan_score(plan, truth, matrix_layout_hint=layout)
    exec_aspects = compute_execution_score(plan, trace_for_scoring, truth, had_hard_errors, matrix_layout_hint=layout)
    meta = compute_decision_score(plan, trace_for_scoring, decision_log or [], raw_csv, metadata_csv, truth)
    ap = float(res.get("de_average_precision", 0.0))
    top10 = float(res.get("de_top10_overlap", 0.0))
    top20 = float(res.get("de_top20_overlap", 0.0))
    ndcg_at_20 = float(res.get("de_ndcg_at_20", 0.0))
    used_batch_tool = _has_batch_tool(_tool_sequence(plan))
    overcorrected = detect_overcorrection(used_batch_tool, float(res.get("pvca_biological_variance_before", 0.0)), float(res.get("pvca_biological_variance_after", 0.0)), float(res.get("f_stat_corr_preservation", 0.0)), float(res.get("bio_truth_retention_score", 0.0)))
    batch_case_match = _batch_case_match(_batch_profile(truth.get("true_workflow_tags", {})), _tool_sequence(plan))
    batch_quality = _batch_removal_quality_score(
        float(res.get("batch_reduction_score", 0.0)),
        batch_case_match,
        0.0 if overcorrected else 1.0,
    )
    batch_detail = {
        "batch_reduction_magnitude": float(res.get("batch_reduction_score", 0.0)),
        "batch_case_match": batch_case_match,
        "overcorrection_flag": 0.0 if overcorrected else 1.0,
    }
    corr = float(res.get("corr_structure_preservation", 0.0))
    bio_struct = _safe_unit_interval(max(corr, 0.0))
    bio_truth = float(res.get("bio_truth_retention_score", 0.0))
    bio_group = float(res.get("bio_group_separation_score", 0.0))
    bio_quality = _bio_preservation_score(bio_truth, bio_group, bio_struct)
    bio_detail = {
        "bio_truth_retention": bio_truth,
        "bio_group_separation_preservation": bio_group,
        "bio_structure_preservation": bio_struct,
    }
    outcome_scores = _manuscript_outcome_scores(res)
    return {
        "core_workflow_correctness": float(plan_aspects["core_workflow_correctness"]),
        "case_awareness": float(plan_aspects["case_awareness"]),
        "parameter_soundness": float(plan_aspects["parameter_soundness"]),
        "workflow_economy": float(plan_aspects["workflow_economy"]),
        "diagnostic_reasoning": float(exec_aspects["diagnostic_reasoning"]),
        "action_selection": float(exec_aspects["action_selection"]),
        "audit_closure": float(exec_aspects["audit_closure"]),
        **outcome_scores,
        "aspect_details": {**plan_aspects.get("aspect_details", {}), **exec_aspects.get("aspect_details", {}), "de_set_recovery": {"de_f1_source": outcome_scores["de_set_recovery"]}, "de_ranking_quality": {"de_average_precision": ap, "de_top10_overlap": top10, "de_top20_overlap": top20, "de_ndcg_at_20": ndcg_at_20, "de_ranking_quality": outcome_scores["de_ranking_quality"]}, "batch_removal_quality": {**batch_detail, "manuscript_score": outcome_scores["batch_removal_quality"], "diagnostic_bucket": float(batch_quality), "overcorrection_detected": bool(overcorrected)}, "biological_signal_preservation": {**bio_detail, "manuscript_score": outcome_scores["biological_signal_preservation"], "diagnostic_bucket": float(bio_quality)}},
        **res,
        **meta,
        "plan_valid": bool(plan_valid),
    }


def evaluate_metrics(raw_csv: str, processed_csv: str, de_csv: str, metadata_csv: str, hidden_truth_json: str) -> dict:
    truth = _load_truth_cached(hidden_truth_json)
    res = _result_metrics(raw_csv, processed_csv, de_csv, metadata_csv, truth)
    corr = float(res.get("corr_structure_preservation", 0.0))
    bio_struct = _safe_unit_interval(max(corr, 0.0))
    bio_truth = float(res.get("bio_truth_retention_score", 0.0))
    bio_group = float(res.get("bio_group_separation_score", 0.0))
    bio_quality = _bio_preservation_score(bio_truth, bio_group, bio_struct)
    outcome_scores = _manuscript_outcome_scores(res)
    out = {name: 0.0 for name in ASPECT_NAMES}
    out.update({**outcome_scores, "aspect_details": {"de_set_recovery": {"de_f1_source": outcome_scores["de_set_recovery"]}, "de_ranking_quality": {"de_average_precision": float(res.get("de_average_precision", 0.0)), "de_top10_overlap": float(res.get("de_top10_overlap", 0.0)), "de_top20_overlap": float(res.get("de_top20_overlap", 0.0)), "de_ndcg_at_20": float(res.get("de_ndcg_at_20", 0.0)), "de_ranking_quality": outcome_scores["de_ranking_quality"]}, "batch_removal_quality": {"batch_reduction_magnitude": outcome_scores["batch_removal_quality"]}, "biological_signal_preservation": {"bio_truth_retention": bio_truth, "bio_group_separation_preservation": bio_group, "bio_structure_preservation": bio_struct, "manuscript_score": outcome_scores["biological_signal_preservation"], "diagnostic_bucket": bio_quality}}})
    out.update(res)
    return out


def top_failures(metrics: dict, threshold: float = 0.5) -> list[str]:
    failed = [aspect for aspect in ASPECT_NAMES if float(metrics.get(aspect, 0.0)) <= 0.0]
    return failed or [aspect for aspect in ASPECT_NAMES if float(metrics.get(aspect, 0.0)) <= threshold]
