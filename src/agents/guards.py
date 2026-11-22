from __future__ import annotations

from src.planner.schema import WorkflowPlan
from src.tools.registry import BATCH_TOOLS


def validate_plan(
    plan: WorkflowPlan,
    metadata_columns: list[str],
    available_tools: list[str] | None = None,
    *,
    dataset_has_missing: bool | None = None,
    dataset_has_batch: bool | None = None,
) -> tuple[bool, list[str]]:
    """Validate a workflow plan.

    Parameters
    ----------
    dataset_has_missing:
        If False, the "clr without impute" rule is relaxed to a warning instead
        of an issue.  When None (unknown) the stricter check applies.
    dataset_has_batch:
        If False, batch-column checks are skipped.
    """
    issues: list[str] = []
    warnings: list[str] = []
    tools = [s.tool for s in plan.steps]
    allowed = set(available_tools or [])

    if not tools:
        issues.append("Empty plan: no workflow steps provided.")
        return False, issues

    if allowed:
        unknown = sorted({t for t in tools if t not in allowed})
        if unknown:
            issues.append(f"Unknown tool(s) in plan: {', '.join(unknown)}.")

    if tools[0] != "inspect_dataset":
        issues.append("First step must be inspect_dataset for an executable workflow.")

    if "run_differential_expression" in tools:
        de_idx = tools.index("run_differential_expression")
        if any(t in BATCH_TOOLS for t in tools[de_idx + 1 :]):
            issues.append("Invalid order: batch correction appears after differential expression.")
        if "clr_transform" in tools and tools.index("clr_transform") > de_idx:
            issues.append("Invalid order: clr_transform appears after differential expression.")
        if "check_batch_effect" in tools and tools.index("check_batch_effect") > de_idx:
            issues.append("Invalid order: check_batch_effect appears after differential expression.")

    if "impute_missing_values" not in tools and "clr_transform" in tools:
        if dataset_has_missing is False:
            warnings.append("clr_transform without impute (acceptable: dataset has no missing values).")
        else:
            issues.append("Potential issue: clr_transform used without missing-value imputation.")

    if any(t in BATCH_TOOLS for t in tools):
        if dataset_has_batch is not False:
            if not ({"batch_label", "batch"} & set(metadata_columns)):
                issues.append("Invalid metadata: batch column missing for batch-correction tool.")

    for i, step in enumerate(plan.steps, start=1):
        if step.tool == "impute_missing_values":
            strategy = step.params.get("strategy", "median")
            if strategy not in {"median", "left_censored_small_value"}:
                issues.append(
                    f"Invalid params at step {i}: impute_missing_values.strategy={strategy!r} is unsupported."
                )
        if step.tool == "run_differential_expression":
            alpha = step.params.get("alpha", 0.05)
            if not isinstance(alpha, (int, float)) or not (0 < float(alpha) <= 1):
                issues.append(f"Invalid params at step {i}: run_differential_expression.alpha must be in (0, 1].")

    max_consec = 1
    run = 1
    for i in range(1, len(tools)):
        if tools[i] == tools[i - 1]:
            run += 1
            max_consec = max(max_consec, run)
        else:
            run = 1
    if max_consec >= 3:
        issues.append(f"Suspicious plan: {max_consec} consecutive duplicate steps detected.")

    return len(issues) == 0, issues
