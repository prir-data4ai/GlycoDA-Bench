from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.planner.schema import WorkflowPlan
from src.tools.registry import BATCH_TOOLS, TOOL_REGISTRY

MATRIX_UPDATE_TOOLS = {"impute_missing_values", "clr_transform"} | BATCH_TOOLS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_input_artifact(step_inputs: dict[str, str], artifacts: dict[str, str]) -> tuple[str, str]:
    """Return (resolved_path, source_key)."""
    key = step_inputs.get("input", "abundance")
    if key not in artifacts:
        key = "abundance"
    return artifacts.get(key, artifacts["abundance"]), key


def _soft_validate_step(
    tool: str,
    artifacts: dict[str, str],
    metadata_csv: str,
    step_inputs: dict[str, str],
    prior_tools: set[str],
) -> list[str]:
    warnings: list[str] = []
    input_key = step_inputs.get("input", "abundance")
    src = artifacts.get(input_key, artifacts["abundance"])
    meta_path = Path(metadata_csv)
    meta_ok = meta_path.exists()
    if tool in MATRIX_UPDATE_TOOLS | {"run_differential_expression", "check_batch_effect", "run_pca"}:
        _ = Path(src).name
        if tool == "run_differential_expression" and input_key == "abundance":
            if "impute_missing_values" not in prior_tools and "clr_transform" not in prior_tools:
                warnings.append("DE on raw abundance stream without prior impute/clr in trace (may be invalid).")
    if tool in {"run_combat", "run_ratio_combat", "run_percentile_norm", "run_limma_batch_removal", "run_harmony"}:
        if meta_ok:
            import pandas as pd

            m = pd.read_csv(metadata_csv)
            if not ({"batch_label", "batch"} & set(m.columns)):
                warnings.append("Batch correction requested but metadata lacks batch column.")
        else:
            warnings.append("Batch correction requested but metadata file missing.")
    return warnings


def _sanitize_step_params(tool: str, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce obviously harmful params toward safe execution defaults."""
    safe = dict(params or {})
    notes: list[str] = []

    if tool == "impute_missing_values":
        strategy = str(safe.get("strategy", "median") or "median")
        if strategy not in {"median", "left_censored_small_value"}:
            safe["strategy"] = "median"
            notes.append("Adjusted impute_missing_values.strategy to 'median' (unsupported value).")

    if tool == "clr_transform":
        raw_pc = safe.get("pseudocount", 1e-6)
        try:
            pc = float(raw_pc)
        except (TypeError, ValueError):
            pc = 1e-6
        if pc <= 0:
            safe["pseudocount"] = 1e-6
            notes.append("Adjusted clr_transform.pseudocount to 1e-6 (must be positive).")

    if tool == "run_combat" and safe.get("use_bio_covariate") is False:
        safe["use_bio_covariate"] = True
        notes.append("Adjusted run_combat.use_bio_covariate to true to preserve biological signal.")

    return safe, notes


@dataclass
class ExecutionResult:
    final_artifact: str
    trace_path: str
    artifacts: dict[str, str]
    trace: list[dict[str, Any]] = field(default_factory=list)
    first_step_is_inspect: bool = True
    had_hard_errors: bool = False


def execute_workflow(
    plan: WorkflowPlan,
    abundance_csv: str,
    metadata_csv: str,
    out_dir: str,
    clr_csv: str | None = None,
    feature_metadata_csv: str | None = None,
) -> ExecutionResult:
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    clr_path = clr_csv or abundance_csv
    artifacts: dict[str, str] = {
        "abundance": abundance_csv,
        "clr": clr_path,
        "metadata": metadata_csv,
    }
    if feature_metadata_csv and Path(feature_metadata_csv).exists():
        artifacts["feature_metadata"] = feature_metadata_csv
    trace: list[dict[str, Any]] = []
    prior_tools: set[str] = set()

    had_hard_errors = False
    for i, step in enumerate(plan.steps, start=1):
        if step.tool not in TOOL_REGISTRY:
            trace.append(
                {
                    "idx": i,
                    "tool": step.tool,
                    "status": "skipped",
                    "error": f"unknown tool: {step.tool}",
                    "started_at": _utc_now_iso(),
                    "finished_at": _utc_now_iso(),
                }
            )
            had_hard_errors = True
            continue

        fn = TOOL_REGISTRY[step.tool]
        output_csv = str(run_dir / f"step_{i:02d}_{step.tool}.csv")
        actual_input, input_key = _resolve_input_artifact(step.inputs, artifacts)
        kwargs, param_notes = _sanitize_step_params(step.tool, dict(step.params))
        warnings = _soft_validate_step(step.tool, artifacts, metadata_csv, step.inputs, prior_tools) + param_notes

        if "metadata_csv" in fn.__code__.co_varnames:
            kwargs["metadata_csv"] = artifacts["metadata"]
        if step.tool == "inspect_dataset" and "metadata_csv" in fn.__code__.co_varnames:
            kwargs.setdefault("metadata_csv", artifacts["metadata"])
        if "feature_metadata_csv" in fn.__code__.co_varnames:
            kwargs.setdefault("feature_metadata_csv", artifacts.get("feature_metadata"))

        started = _utc_now_iso()
        status = "ok"
        err_msg: str | None = None
        produced = actual_input
        try:
            produced = fn(actual_input, output_csv, **kwargs)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            err_msg = f"{exc}\n{traceback.format_exc()}"
            had_hard_errors = True
            trace.append(
                {
                    "idx": i,
                    "tool": step.tool,
                    "input": actual_input,
                    "input_key": input_key,
                    "output": output_csv,
                    "params": kwargs,
                    "status": status,
                    "error": err_msg,
                    "warnings": warnings,
                    "started_at": started,
                    "finished_at": _utc_now_iso(),
                }
            )
            continue

        artifacts[step.tool] = produced
        prior_tools.add(step.tool)
        if step.tool in MATRIX_UPDATE_TOOLS:
            artifacts["abundance"] = produced
            if step.tool == "clr_transform":
                artifacts["clr"] = produced

        trace.append(
            {
                "idx": i,
                "tool": step.tool,
                "input": actual_input,
                "input_key": input_key,
                "output": produced,
                "params": kwargs,
                "status": status,
                "warnings": warnings,
                "started_at": started,
                "finished_at": _utc_now_iso(),
            }
        )

    trace_path = str(run_dir / "execution_trace.json")
    Path(trace_path).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    ok_outputs = [t["output"] for t in trace if t.get("status") == "ok" and "output" in t]
    final_artifact = ok_outputs[-1] if ok_outputs else abundance_csv
    return ExecutionResult(
        final_artifact=final_artifact,
        trace_path=trace_path,
        artifacts=dict(artifacts),
        trace=trace,
        first_step_is_inspect=plan.steps[0].tool == "inspect_dataset" if plan.steps else False,
        had_hard_errors=had_hard_errors,
    )


def execute_step(
    step,
    artifacts: dict[str, str],
    metadata_csv: str,
    out_dir: str,
    step_index: int,
    prior_tools: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any], bool]:
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prior_tools = set(prior_tools or set())

    if step.tool not in TOOL_REGISTRY:
        trace_entry = {
            "idx": step_index,
            "tool": step.tool,
            "status": "skipped",
            "error": f"unknown tool: {step.tool}",
            "started_at": _utc_now_iso(),
            "finished_at": _utc_now_iso(),
        }
        return dict(artifacts), trace_entry, True

    fn = TOOL_REGISTRY[step.tool]
    output_csv = str(run_dir / f"step_{step_index:02d}_{step.tool}.csv")
    actual_input, input_key = _resolve_input_artifact(step.inputs, artifacts)
    kwargs, param_notes = _sanitize_step_params(step.tool, dict(step.params))
    warnings = _soft_validate_step(step.tool, artifacts, metadata_csv, step.inputs, prior_tools) + param_notes

    if "metadata_csv" in fn.__code__.co_varnames:
        kwargs["metadata_csv"] = artifacts["metadata"]
    if step.tool == "inspect_dataset" and "metadata_csv" in fn.__code__.co_varnames:
        kwargs.setdefault("metadata_csv", artifacts["metadata"])
    if "feature_metadata_csv" in fn.__code__.co_varnames:
        kwargs.setdefault("feature_metadata_csv", artifacts.get("feature_metadata"))

    started = _utc_now_iso()
    try:
        produced = fn(actual_input, output_csv, **kwargs)
    except Exception as exc:  # noqa: BLE001
        trace_entry = {
            "idx": step_index,
            "tool": step.tool,
            "input": actual_input,
            "input_key": input_key,
            "output": output_csv,
            "params": kwargs,
            "status": "error",
            "error": f"{exc}\n{traceback.format_exc()}",
            "warnings": warnings,
            "started_at": started,
            "finished_at": _utc_now_iso(),
        }
        return dict(artifacts), trace_entry, True

    next_artifacts = dict(artifacts)
    next_artifacts[step.tool] = produced
    if step.tool in MATRIX_UPDATE_TOOLS:
        next_artifacts["abundance"] = produced
        if step.tool == "clr_transform":
            next_artifacts["clr"] = produced

    trace_entry = {
        "idx": step_index,
        "tool": step.tool,
        "input": actual_input,
        "input_key": input_key,
        "output": produced,
        "params": kwargs,
        "status": "ok",
        "warnings": warnings,
        "started_at": started,
        "finished_at": _utc_now_iso(),
    }
    return next_artifacts, trace_entry, False
