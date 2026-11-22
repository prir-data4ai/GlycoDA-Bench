from __future__ import annotations

import json
from pathlib import Path

from src.data.dataset import resolve_hidden_truth_path
from src.metrics.evaluator import _normalize_truth
from src.tools.registry import BATCH_TOOLS
from src.planner.schema import WorkflowPlan

# Only these single minor partials map to "mostly_correct" (stricter than partially_reasonable).
# Optional motif / borderline-batch-without-check are no longer "mostly" alone — they need a clean bill otherwise.
MOSTLY_CORRECT_ALLOWED_PARTIALS: frozenset[str] = frozenset(
    {
        "Plan starts without inspect_dataset; analysis can still be partially reasonable",
    }
)


def evaluate_oracle(plan: WorkflowPlan, hidden_truth_path: str) -> dict:
    """Resolve truth file: legacy ``.../hidden_truth.json`` goes through ``resolve_hidden_truth_path``;
    an explicit path to ``_private_hidden_truth/<case>.json`` is read directly when present."""
    p = Path(hidden_truth_path)
    if p.name == "hidden_truth.json":
        truth_path = resolve_hidden_truth_path(p.parent)
    elif p.is_file():
        truth_path = p
    else:
        truth_path = resolve_hidden_truth_path(p)
    truth_obj = json.loads(truth_path.read_text(encoding="utf-8"))
    truth = _normalize_truth(truth_obj)
    tools = [s.tool for s in plan.steps]
    explanation: list[str] = []
    detail: dict[str, bool | str] = {}
    wrong_reasons: list[str] = []
    partial_reasons: list[str] = []

    tc = truth.get("true_conditions", {})
    batch_significant = bool(tc.get("batch_injected") or truth.get("batch_effect_significant"))
    severe_missing = bool(tc.get("missingness_injected") or truth.get("severe_missingness"))
    biological_covariate_protected = bool(truth.get("batch_correction_protects_bio_covariate", True))

    if not tools:
        return {
            "verdict": "wrong",
            "explanations": ["Empty plan: no tools selected."],
            "oracle_verdict_detail": {"empty_plan": True},
        }

    # Severity split:
    # - wrong_reasons: critical scientific/protocol violations
    # - partial_reasons: recoverable decision quality issues
    if tools[0] != "inspect_dataset":
        partial_reasons.append("Plan starts without inspect_dataset; analysis can still be partially reasonable")

    if batch_significant and not any(t in tools for t in BATCH_TOOLS):
        wrong_reasons.append("Detected batch injection but no batch correction tool in plan.")
    detail["batch_tool_required"] = batch_significant

    if not batch_significant and any(t in tools for t in BATCH_TOOLS):
        partial_reasons.append("Unnecessary batch correction: no meaningful batch effect present.")
    detail["unnecessary_batch_correction"] = not batch_significant and any(t in tools for t in BATCH_TOOLS)

    if "run_differential_expression" in tools and any(t in tools for t in BATCH_TOOLS):
        if tools.index("run_differential_expression") < min(
            tools.index(t) for t in BATCH_TOOLS if t in tools
        ):
            wrong_reasons.append("Batch correction appears after differential expression.")

    if any(t in tools for t in BATCH_TOOLS) and "clr_transform" not in tools:
        partial_reasons.append("Batch correction without explicit CLR/log-ratio transform may be incompatible.")

    if severe_missing and "impute_missing_values" not in tools:
        wrong_reasons.append("Severe missingness without imputation before downstream analysis.")

    if not severe_missing and "impute_missing_values" in tools:
        tags_needs = truth.get("true_workflow_tags", {}).get("needs_imputation", False)
        if not tags_needs:
            partial_reasons.append("Unnecessary imputation: data has no significant missing values.")

    tags = truth.get("true_workflow_tags", {})
    if tags.get("needs_imputation") and "impute_missing_values" not in tools:
        partial_reasons.append("Ground-truth tags expect imputation.")
    if (
        str(tags.get("batch_decision_profile", "")).strip().lower() == "borderline"
        and any(t in tools for t in BATCH_TOOLS)
        and "check_batch_effect" not in tools
    ):
        partial_reasons.append("Borderline batch scenario uses correction without explicit check_batch_effect")

    motif_worthwhile = bool(tags.get("motif_followup_worthwhile", False))
    if not motif_worthwhile and "run_motif_enrichment" in tools:
        partial_reasons.append(
            "Plan includes optional motif enrichment when not required by ground truth"
        )

    if biological_covariate_protected and any(t in tools for t in BATCH_TOOLS):
        explanation.append("Note: bio covariate preservation flagged in truth (qualitative).")

    explanation.extend(wrong_reasons)
    explanation.extend(partial_reasons)

    if wrong_reasons:
        verdict = "wrong"
    elif len(partial_reasons) >= 2:
        verdict = "partially_reasonable"
    elif len(partial_reasons) == 1:
        verdict = (
            "mostly_correct"
            if partial_reasons[0] in MOSTLY_CORRECT_ALLOWED_PARTIALS
            else "partially_reasonable"
        )
    else:
        verdict = "correct"

    return {
        "verdict": verdict,
        "explanations": explanation,
        "oracle_verdict_detail": detail,
    }
