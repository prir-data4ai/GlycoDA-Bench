from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from src.agents.guards import validate_plan
from src.config import apply_api_runtime_config, load_api_runtime_config, load_model_config, try_load_dotenv
from src.data.dataset import build_planner_context, load_dataset_paths, summarize_dataset
from src.executor.executor import ExecutionResult, MATRIX_UPDATE_TOOLS, execute_step
from src.llm.adapter import LLMAdapter
from src.metrics.evaluator import evaluate_full, top_failures
from src.oracle.rules import evaluate_oracle
from src.planner.planner import PlanGenerationResult, StepDecisionResult, plan_next_step_detailed, plan_workflow_detailed
from src.planner.schema import DatasetAssessment, WorkflowPlan, WorkflowStep
from src.reporting.reporter import write_run_report
from src.tools.impl import inspect_dataset
from src.tools.registry import BATCH_TOOLS, list_tool_names


def _validate_run_dir_name(name: str) -> str:
    """Single path segment under out-dir/runs/; avoids traversal and odd names."""
    if len(name) > 240:
        raise ValueError("run-dir-name is too long")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("run-dir-name must match [A-Za-z0-9_.-]+")
    return name


def _safe_run_dir_segment(s: str) -> str:
    """Filesystem-safe single segment (aligned with benchmark driver)."""
    t = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("._-")
    return t[:100] if t else "x"


def _validate_run_dir_relative_path(path: str) -> Path:
    """One or more segments under <out-dir>/runs/, e.g. model_name/seed42__full__case_01."""
    raw = path.replace("\\", "/").strip("/")
    if not raw or ".." in raw or raw.startswith("/"):
        raise ValueError("invalid run-dir path")
    parts = [p for p in raw.split("/") if p]
    if not parts or len(parts) > 8:
        raise ValueError("run-dir path must have 1–8 segments")
    out: list[str] = []
    for part in parts:
        if len(part) > 240:
            raise ValueError("run-dir segment too long")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", part):
            raise ValueError("run-dir segment must match [A-Za-z0-9_.-]+")
        out.append(part)
    return Path(*out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="YAML model config path")
    p.add_argument("--dataset", required=True, help="Dataset directory path")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api-config", default="configs/api.yaml")
    p.add_argument(
        "--planning-mode",
        choices=["full", "iterative"],
        default="full",
        help="Workflow generation mode: full plan once, or iterative step-by-step planning with execution feedback.",
    )
    p.add_argument(
        "--run-dir-name",
        default=None,
        help=(
            "Optional path under <out-dir>/runs/ (use / between segments). "
            "Benchmark uses <model_name>/<seed>__<planning_mode>__<dataset_id>. "
            "Default: <model_name>/<timestamp>__<dataset_id>."
        ),
    )
    return p.parse_args()


def _read_inspection_summary(path: Path) -> dict:
    row = pd.read_csv(path).iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def _build_observation_constraints(observation: dict) -> list[str]:
    hints: list[str] = []
    if float(observation.get("missing_fraction", 0.0) or 0.0) > 0:
        hints.append("Runtime inspection detected missing values; include impute_missing_values before clr/de.")
    if bool(observation.get("rows_sum_near_constant_like_composition", False)):
        hints.append("Runtime inspection suggests compositional data; apply clr_transform before differential expression.")
    if bool(observation.get("batch_label_available", False)):
        hints.append("Runtime inspection found batch labels; use check_batch_effect before deciding on batch correction.")
    return hints


def _write_validation_trace(path: Path, plan_steps: int, issues: list[str]) -> str:
    trace = [
        {
            "idx": 0,
            "tool": "plan_validation",
            "status": "error",
            "error": " | ".join(issues),
            "warnings": [],
            "plan_steps": plan_steps,
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return str(path)


def _invalid_execution_result(trace_path: str, abundance_csv: str, plan) -> ExecutionResult:
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    return ExecutionResult(
        final_artifact=abundance_csv,
        trace_path=trace_path,
        artifacts={"abundance": abundance_csv},
        trace=trace,
        first_step_is_inspect=plan.steps[0].tool == "inspect_dataset" if plan.steps else False,
        had_hard_errors=True,
    )


def _choose_replanned_result(
    original: PlanGenerationResult,
    candidate: PlanGenerationResult,
    original_valid: bool,
    candidate_valid: bool,
) -> PlanGenerationResult:
    if candidate_valid:
        return candidate
    if not original_valid:
        return candidate
    return original


def _summarize_csv_for_llm(path: str) -> dict:
    csv_path = Path(path)
    if not csv_path.exists():
        return {"path": path, "exists": False}
    df = pd.read_csv(csv_path)
    summary: dict = {
        "path": path,
        "exists": True,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns[:12]],
    }
    if df.shape[0] == 1:
        row = df.iloc[0].to_dict()
        summary["row_values"] = {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in list(row.items())[:16]
        }
    if {"significant", "feature"} <= set(df.columns):
        sig = pd.to_numeric(df["significant"], errors="coerce").fillna(0)
        summary["significant_count"] = int(sig.astype(bool).sum())
        if "padj" in df.columns:
            top = df.nsmallest(min(5, len(df)), "padj")[["feature", "padj"]].to_dict(orient="records")
            summary["top_features"] = top
    return summary


def _build_iterative_artifacts(dataset) -> dict[str, str]:
    artifacts = {
        "abundance": str(dataset.abundance_csv),
        "clr": str(dataset.clr_csv if dataset.clr_csv.exists() else dataset.abundance_csv),
        "metadata": str(dataset.sample_metadata_csv),
    }
    if dataset.feature_metadata_csv.exists():
        artifacts["feature_metadata"] = str(dataset.feature_metadata_csv)
    return artifacts


def _merge_usage(total: dict, addition: dict) -> dict:
    for key in ("input_tokens", "output_tokens", "total_tokens", "call_count"):
        total[key] = int(total.get(key, 0) or 0) + int(addition.get(key, 0) or 0)
    return total


def _state_path(run_dir: Path) -> Path:
    return run_dir / "run_state.json"


def _trace_path(run_dir: Path) -> Path:
    return run_dir / "tool_outputs" / "execution_trace.json"


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_run_logger(run_dir: Path) -> Callable[[str], None]:
    from src.app.log_utils import colorize_tag
    log_path = run_dir / "run.log"

    def _log(message: str) -> None:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plain_line = f"{ts} {message}"
        colored_line = f"{ts} {colorize_tag(message)}"
        print(colored_line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(plain_line + "\n")

    return _log


def _build_base_artifacts(dataset) -> dict[str, str]:
    artifacts = {
        "abundance": str(dataset.abundance_csv),
        "clr": str(dataset.clr_csv if dataset.clr_csv.exists() else dataset.abundance_csv),
        "metadata": str(dataset.sample_metadata_csv),
    }
    if dataset.feature_metadata_csv.exists():
        artifacts["feature_metadata"] = str(dataset.feature_metadata_csv)
    return artifacts


def _iterative_allowed_next_tools(
    runtime_observation: dict,
    planner_ctx: dict,
    steps: list[WorkflowStep],
    tool_names: list[str],
) -> tuple[list[str], str]:
    executed = [s.tool for s in steps]
    executed_set = set(executed)
    if not executed:
        return ["inspect_dataset"], "must_start_with_inspect"

    has_missing = float(runtime_observation.get("missing_fraction", 0.0) or 0.0) > 0
    missing_frac = float(runtime_observation.get("missing_fraction", 0.0) or 0.0)
    dataset_summary = dict(planner_ctx.get("dataset_summary", {}) or {})
    is_comp = bool(
        runtime_observation.get("rows_sum_near_constant_like_composition", False)
        or dataset_summary.get("is_compositional", False)
    )
    has_batch_labels = bool(runtime_observation.get("batch_label_available", False))
    de_done = "run_differential_expression" in executed_set
    batch_checked = "check_batch_effect" in executed_set
    batch_done = any(t in executed_set for t in BATCH_TOOLS)

    allowed: list[str] = []
    reason = "default_progression"

    if de_done:
        reason = "post_de_followup"
        for tool in ("run_motif_enrichment", "run_pca"):
            if tool in tool_names and tool not in executed_set:
                allowed.append(tool)
        return allowed, reason

    if has_missing and "impute_missing_values" not in executed_set:
        allowed.append("impute_missing_values")
        if missing_frac <= 0.03 and is_comp and "clr_transform" not in executed_set:
            allowed.append("clr_transform")
        reason = "missingness_gate"
        return [t for t in allowed if t in tool_names], reason

    if is_comp and "clr_transform" not in executed_set:
        reason = "clr_gate"
        return [t for t in ["clr_transform"] if t in tool_names], reason

    if has_batch_labels and not batch_checked:
        reason = "batch_diagnostic_gate"
        return [t for t in ["check_batch_effect"] if t in tool_names], reason

    if has_batch_labels and batch_checked and not batch_done:
        reason = "batch_action_gate"
        allowed = [t for t in tool_names if t in BATCH_TOOLS]
        if "run_differential_expression" in tool_names:
            allowed.append("run_differential_expression")
        return allowed, reason

    reason = "terminal_analysis_gate"
    return [t for t in ["run_differential_expression"] if t in tool_names], reason


def _run_iterative_workflow(
    adapter: LLMAdapter,
    planner_ctx: dict,
    tool_names: list[str],
    constraints: list[str],
    runtime_observation: dict,
    dataset,
    run_dir: Path,
    log: Callable[[str], None],
) -> tuple[WorkflowPlan, StepDecisionResult | None, ExecutionResult, float, float, bool, list[str], dict, list[dict], str, list[dict]]:
    state_payload: dict = {}
    state_file = _state_path(run_dir)
    if state_file.exists():
        try:
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            state_payload = {}
    saved_plan = state_payload.get("plan") or {}
    saved_steps = saved_plan.get("steps") or []
    trace = list(state_payload.get("execution_trace") or [])
    resume_ok_prefix = 0
    for entry in trace:
        if entry.get("status") != "ok":
            break
        resume_ok_prefix += 1
    steps = [WorkflowStep.model_validate(s) for s in saved_steps[:resume_ok_prefix]]
    llm_calls = list(state_payload.get("llm_calls") or [])
    llm_usage = dict(state_payload.get("llm_usage") or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0})
    planning_latency_sec = float(state_payload.get("planning_latency_sec", 0.0) or 0.0)
    execution_latency_sec = float(state_payload.get("execution_latency_sec", 0.0) or 0.0)
    final_target = str(saved_plan.get("final_analysis_target") or "")
    stop_conditions: list[str] = list(saved_plan.get("stop_conditions") or [])
    step_limit = 12
    planner_failure_reason = str(state_payload.get("planner_failure_reason") or "")
    last_decision_result: StepDecisionResult | None = None
    had_hard_errors = any(t.get("status") != "ok" for t in trace)
    artifacts = _rebuild_artifacts_from_trace(dataset, trace)
    decision_log = list(state_payload.get("decision_log") or [])

    if trace:
        log(f"[resume] iterative run state found: {len(trace)} traced step(s), {sum(t.get('status') == 'ok' for t in trace)} successful")

    for step_index in range(1, step_limit + 1):
        if step_index <= len(trace):
            existing = trace[step_index - 1]
            if existing.get("status") == "ok":
                continue
        iterative_ctx = dict(planner_ctx)
        iterative_ctx["runtime_observation"] = runtime_observation
        iterative_ctx["executed_tools"] = [t.get("tool") for t in trace if t.get("status") == "ok" and t.get("tool")]
        iterative_ctx["current_artifacts"] = {k: Path(v).name for k, v in artifacts.items()}
        iterative_ctx["execution_trace_summary"] = [
            {
                "idx": t.get("idx"),
                "tool": t.get("tool"),
                "status": t.get("status"),
                "warnings": t.get("warnings", []),
                "output_summary": _summarize_csv_for_llm(t["output"]) if t.get("status") == "ok" and t.get("output") else {},
            }
            for t in trace
        ]
        allowed_next_tools, protocol_reason = _iterative_allowed_next_tools(runtime_observation, planner_ctx, steps, tool_names)
        iterative_ctx["allowed_next_tools"] = allowed_next_tools
        iterative_ctx["decision_protocol_reason"] = protocol_reason
        iterative_constraints = constraints + [
            "Decide exactly one next step or stop.",
            "First step must be inspect_dataset.",
            "Do not repeat a completed tool unless the new result clearly justifies it.",
            "After run_differential_expression, stop unless a downstream interpretation/QC step is justified.",
            f"At this stage, allowed_next_tools={allowed_next_tools}; if you continue, choose one of them.",
        ]

        t0 = time.perf_counter()
        log(f"[planning] iterative step {step_index}: requesting next step")
        decision_result = plan_next_step_detailed(adapter, iterative_ctx, tool_names, iterative_constraints)
        planning_latency_sec += time.perf_counter() - t0
        last_decision_result = decision_result
        planner_failure_reason = decision_result.planner_failure_reason
        llm_calls.extend(decision_result.llm_calls)
        _merge_usage(llm_usage, decision_result.llm_usage)
        decision = decision_result.decision

        if decision.updated_final_analysis_target:
            final_target = decision.updated_final_analysis_target
        if decision.done:
            decision_log.append(
                {
                    "step_index": step_index,
                    "decision_source": decision_result.decision_source,
                    "allowed_next_tools": allowed_next_tools,
                    "protocol_reason": protocol_reason,
                    "chosen_tool": None,
                    "done": True,
                    "rationale": decision.rationale,
                    "stop_reason": decision.stop_reason,
                }
            )
            stop_conditions.append(decision.stop_reason or "llm_requested_stop")
            break

        step = decision.next_step
        if step is None:
            stop_conditions.append("llm_returned_no_step")
            break

        if allowed_next_tools and step.tool not in allowed_next_tools:
            decision_log.append(
                {
                    "step_index": step_index,
                    "decision_source": decision_result.decision_source,
                    "allowed_next_tools": allowed_next_tools,
                    "protocol_reason": protocol_reason,
                    "chosen_tool": step.tool,
                    "done": False,
                    "rationale": decision.rationale,
                    "stop_reason": "invalid_next_tool_for_stage",
                }
            )
            stop_conditions.append(f"invalid_next_tool:{step.tool}")
            had_hard_errors = True
            break

        if steps and step.tool == "inspect_dataset":
            stop_conditions.append("planner_repeated_inspect_dataset")
            break

        if step.tool in BATCH_TOOLS and "check_batch_effect" not in [s.tool for s in steps]:
            step.purpose = step.purpose or "batch correction"

        exec_t0 = time.perf_counter()
        log(f"[step {step_index}/{step_limit}] starting {step.tool}")
        artifacts, trace_entry, step_failed = execute_step(
            step,
            artifacts,
            str(dataset.sample_metadata_csv),
            str(run_dir / "tool_outputs"),
            step_index,
            prior_tools={t.get('tool') for t in trace if t.get('status') == 'ok' and t.get('tool')},
        )
        execution_latency_sec += time.perf_counter() - exec_t0
        if len(trace) >= step_index:
            trace[step_index - 1] = trace_entry
        else:
            trace.append(trace_entry)
        steps.append(step)
        decision_log.append(
            {
                "step_index": step_index,
                "decision_source": decision_result.decision_source,
                "allowed_next_tools": allowed_next_tools,
                "protocol_reason": protocol_reason,
                "chosen_tool": step.tool,
                "done": False,
                "rationale": decision.rationale,
                "stop_reason": decision.stop_reason,
            }
        )
        had_hard_errors = had_hard_errors or step_failed
        _log_step_result(log, trace_entry)
        _persist_run_checkpoint(
            run_dir,
            status="executing",
            plan=WorkflowPlan(
                workflow_name="iterative_workflow",
                rationale="Iteratively decided by the LLM with execution feedback after each step.",
                steps=steps,
                dataset_assessment=DatasetAssessment(
                    detected_missingness=float(runtime_observation.get("missing_fraction", 0.0) or 0.0) > 0,
                    detected_batch_risk=bool(runtime_observation.get("batch_label_available", False)),
                    detected_compositionality=bool(runtime_observation.get("rows_sum_near_constant_like_composition", False)),
                    recommended_input_matrix="abundance",
                ),
                final_analysis_target=final_target or "differential_expression",
                stop_conditions=stop_conditions,
            ),
            trace=trace,
            artifacts=artifacts,
            llm_usage=llm_usage,
            llm_calls=llm_calls,
            decision_log=decision_log,
            planning_latency_sec=planning_latency_sec,
            execution_latency_sec=execution_latency_sec,
            planner_failure_reason=planner_failure_reason,
            plan_source="iterative",
        )

        if step_failed:
            stop_conditions.append(f"execution_error_at_step_{step_index}")
            break

        if step.tool == "run_differential_expression":
            stop_conditions.append("primary_analysis_completed")

    else:
        stop_conditions.append("step_limit_reached")

    trace_path = str(run_dir / "tool_outputs" / "execution_trace.json")
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    Path(trace_path).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    final_artifact = trace[-1]["output"] if trace and trace[-1].get("status") == "ok" else str(dataset.abundance_csv)
    exec_result = ExecutionResult(
        final_artifact=final_artifact,
        trace_path=trace_path,
        artifacts=artifacts,
        trace=trace,
        first_step_is_inspect=bool(steps and steps[0].tool == "inspect_dataset"),
        had_hard_errors=had_hard_errors,
    )
    plan = WorkflowPlan(
        workflow_name="iterative_workflow",
        rationale="Iteratively decided by the LLM with execution feedback after each step.",
        steps=steps,
        dataset_assessment=DatasetAssessment(
            detected_missingness=float(runtime_observation.get("missing_fraction", 0.0) or 0.0) > 0,
            detected_batch_risk=bool(runtime_observation.get("batch_label_available", False)),
            detected_compositionality=bool(runtime_observation.get("rows_sum_near_constant_like_composition", False)),
            recommended_input_matrix="abundance",
        ),
        final_analysis_target=final_target or "differential_expression",
        stop_conditions=stop_conditions,
    )

    meta_cols = pd.read_csv(dataset.sample_metadata_csv).columns.tolist()
    has_missing = float(runtime_observation.get("missing_fraction", 0.0) or 0.0) > 0
    has_batch = bool(runtime_observation.get("batch_label_available", False))
    plan_valid, guard_issues = validate_plan(
        plan,
        meta_cols,
        tool_names,
        dataset_has_missing=has_missing,
        dataset_has_batch=has_batch,
    )
    return (
        plan,
        last_decision_result,
        exec_result,
        planning_latency_sec,
        execution_latency_sec,
        plan_valid,
        guard_issues,
        llm_usage,
        llm_calls,
        planner_failure_reason,
        decision_log,
    )


def _rebuild_artifacts_from_trace(dataset, trace: list[dict]) -> dict[str, str]:
    artifacts = _build_base_artifacts(dataset)
    for entry in trace:
        if entry.get("status") != "ok":
            break
        tool = entry.get("tool")
        output = entry.get("output")
        if not tool or not output or not Path(output).exists():
            break
        artifacts[str(tool)] = str(output)
        if tool in MATRIX_UPDATE_TOOLS:
            artifacts["abundance"] = str(output)
            if tool == "clr_transform":
                artifacts["clr"] = str(output)
    return artifacts


def _matching_resume_prefix(plan: WorkflowPlan, trace: list[dict]) -> int:
    prefix = 0
    for idx, step in enumerate(plan.steps, start=1):
        if idx > len(trace):
            break
        entry = trace[idx - 1]
        if entry.get("status") != "ok":
            break
        if int(entry.get("idx", idx)) != idx:
            break
        if entry.get("tool") != step.tool:
            break
        if dict(entry.get("params") or {}) != dict(step.params or {}):
            break
        output = entry.get("output")
        if not output or not Path(output).exists():
            break
        prefix = idx
    return prefix


def _persist_run_checkpoint(
    run_dir: Path,
    *,
    status: str,
    plan: WorkflowPlan | None,
    trace: list[dict],
    artifacts: dict[str, str] | None,
    llm_usage: dict,
    llm_calls: list[dict],
    decision_log: list[dict] | None = None,
    planning_latency_sec: float,
    execution_latency_sec: float,
    planner_failure_reason: str,
    plan_source: str,
) -> None:
    if plan is not None:
        _json_dump(run_dir / "plan.json", plan.model_dump())
    _json_dump(_trace_path(run_dir), trace)
    payload = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "plan": plan.model_dump() if plan is not None else None,
        "execution_trace": trace,
        "artifacts": artifacts or {},
        "llm_usage": llm_usage,
        "llm_calls": llm_calls,
        "decision_log": decision_log or [],
        "planning_latency_sec": planning_latency_sec,
        "execution_latency_sec": execution_latency_sec,
        "planner_failure_reason": planner_failure_reason,
        "plan_source": plan_source,
    }
    _json_dump(_state_path(run_dir), payload)


def _log_step_result(log: Callable[[str], None], trace_entry: dict) -> None:
    idx = trace_entry.get("idx", "?")
    tool = trace_entry.get("tool", "")
    status = trace_entry.get("status", "")
    started = trace_entry.get("started_at")
    finished = trace_entry.get("finished_at")
    elapsed = ""
    if started and finished:
        try:
            elapsed_sec = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
            elapsed = f" in {elapsed_sec.total_seconds():.2f}s"
        except ValueError:
            elapsed = ""
    if status == "ok":
        log(f"[step {idx}] completed {tool}{elapsed}")
    else:
        err = str(trace_entry.get("error") or "").splitlines()[0]
        log(f"[step {idx}] {status} {tool}{elapsed}: {err}")


def _run_full_workflow(
    plan: WorkflowPlan,
    dataset,
    run_dir: Path,
    *,
    llm_usage: dict,
    llm_calls: list[dict],
    planning_latency_sec: float,
    planner_failure_reason: str,
    plan_source: str,
    log: Callable[[str], None],
) -> tuple[ExecutionResult, float]:
    trace_file = _trace_path(run_dir)
    existing_trace: list[dict] = []
    if trace_file.exists():
        try:
            existing_trace = json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception:
            existing_trace = []
    resume_prefix = _matching_resume_prefix(plan, existing_trace)
    trace = list(existing_trace[:resume_prefix])
    artifacts = _rebuild_artifacts_from_trace(dataset, trace)
    if resume_prefix:
        log(f"[resume] full run state found: resuming after step {resume_prefix}")

    exec_t0 = time.perf_counter()
    had_hard_errors = any(t.get("status") != "ok" for t in trace)
    for step_index, step in enumerate(plan.steps, start=1):
        if step_index <= resume_prefix:
            continue
        log(f"[step {step_index}/{len(plan.steps)}] starting {step.tool}")
        artifacts, trace_entry, step_failed = execute_step(
            step,
            artifacts,
            str(dataset.sample_metadata_csv),
            str(run_dir / "tool_outputs"),
            step_index,
            prior_tools={t.get("tool") for t in trace if t.get("status") == "ok" and t.get("tool")},
        )
        trace.append(trace_entry)
        had_hard_errors = had_hard_errors or step_failed
        _log_step_result(log, trace_entry)
        _persist_run_checkpoint(
            run_dir,
            status="executing",
            plan=plan,
            trace=trace,
            artifacts=artifacts,
            llm_usage=llm_usage,
            llm_calls=llm_calls,
            planning_latency_sec=planning_latency_sec,
            execution_latency_sec=time.perf_counter() - exec_t0,
            planner_failure_reason=planner_failure_reason,
            plan_source=plan_source,
        )
    execution_latency_sec = time.perf_counter() - exec_t0
    final_artifact = trace[-1]["output"] if trace and trace[-1].get("status") == "ok" else str(dataset.abundance_csv)
    exec_result = ExecutionResult(
        final_artifact=final_artifact,
        trace_path=str(trace_file),
        artifacts=artifacts,
        trace=trace,
        first_step_is_inspect=plan.steps[0].tool == "inspect_dataset" if plan.steps else False,
        had_hard_errors=had_hard_errors,
    )
    return exec_result, execution_latency_sec


def run_single_job(
    model_config_path: str,
    dataset_path: str,
    out_dir: str = "outputs",
    api_config: str = "configs/api.yaml",
    seed: int = 42,
    planning_mode: str = "full",
    run_dir_name: str | None = None,
) -> dict:
    """Run a single benchmark job in-process (no subprocess).

    This is the core logic extracted from ``main()`` so that
    ``run_benchmark.py`` can call it directly via asyncio / thread-pool
    instead of spawning a heavy subprocess per job.

    Returns the report payload dict (same content as report.json).
    """
    try_load_dotenv()
    from src.app.log_utils import configure_colored_logging
    configure_colored_logging()
    apply_api_runtime_config(load_api_runtime_config(api_config))
    model_cfg = load_model_config(model_config_path)
    if model_cfg.model == "qwen3.5-plus" and os.getenv("GLYCOAGENT_MODEL"):
        model_cfg.model = os.environ["GLYCOAGENT_MODEL"].replace("openai:", "")
    dataset = load_dataset_paths(dataset_path)
    summary = summarize_dataset(dataset)
    planner_ctx = build_planner_context(summary, dataset_root=dataset.root)
    adapter = LLMAdapter(model_cfg)
    tool_names = list_tool_names()

    if run_dir_name:
        run_dir = Path(out_dir) / "runs" / _validate_run_dir_relative_path(run_dir_name)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_seg = _safe_run_dir_segment(model_cfg.name)
        leaf = f"{ts}__{_safe_run_dir_segment(str(summary['dataset_id']))}"
        run_dir = Path(out_dir) / "runs" / _validate_run_dir_relative_path(f"{model_seg}/{leaf}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _make_run_logger(run_dir)
    log(f"[run] model={model_cfg.name} dataset={summary['dataset_id']} planning_mode={planning_mode} run_dir={run_dir}")

    return _run_job_core(
        model_cfg=model_cfg,
        dataset=dataset,
        summary=summary,
        planner_ctx=planner_ctx,
        adapter=adapter,
        tool_names=tool_names,
        run_dir=run_dir,
        seed=seed,
        planning_mode=planning_mode,
        log=log,
    )


def _run_job_core(
    *,
    model_cfg,
    dataset,
    summary: dict,
    planner_ctx: dict,
    adapter: LLMAdapter,
    tool_names: list[str],
    run_dir: Path,
    seed: int,
    planning_mode: str,
    log: Callable[[str], None],
) -> dict:
    """Shared core logic for both run_single_job() and main()."""

    constraints = [
        "agent cannot read hidden_truth.json",
        "workflow output must be structured JSON (WorkflowPlan)",
        "impute before clr/de when missingness high",
        "batch correction must be before de",
    ]

    preflight_inspect_path = run_dir / "preflight_inspect.csv"
    log("[preflight] inspecting dataset")
    inspect_dataset(
        str(dataset.abundance_csv),
        str(preflight_inspect_path),
        metadata_csv=str(dataset.sample_metadata_csv),
    )
    runtime_observation = _read_inspection_summary(preflight_inspect_path)
    observation_constraints = _build_observation_constraints(runtime_observation)
    planner_ctx["runtime_observation"] = runtime_observation

    meta_cols = pd.read_csv(dataset.sample_metadata_csv).columns.tolist()
    has_missing = float(runtime_observation.get("missing_fraction", 0.0) or 0.0) > 0
    has_batch = bool(runtime_observation.get("batch_label_available", False))
    replan_applied = False
    replan_reason = ""
    if planning_mode == "iterative":
        (
            plan,
            _last_decision_result,
            exec_result,
            planning_latency_sec,
            execution_latency_sec,
            plan_valid,
            guard_issues,
            llm_usage,
            llm_calls,
            planner_failure_reason,
            decision_log,
        ) = _run_iterative_workflow(
            adapter,
            planner_ctx,
            tool_names,
            constraints + observation_constraints,
            runtime_observation,
            dataset,
            run_dir,
            log,
        )
        plan_source = "iterative"
    else:
        decision_log = []
        state_payload: dict = {}
        if _state_path(run_dir).exists():
            try:
                state_payload = json.loads(_state_path(run_dir).read_text(encoding="utf-8"))
            except Exception:
                state_payload = {}
        saved_plan = state_payload.get("plan")
        if saved_plan:
            log("[resume] loading saved plan from checkpoint")
            plan = WorkflowPlan.model_validate(saved_plan)
            planning_latency_sec = float(state_payload.get("planning_latency_sec", 0.0) or 0.0)
            llm_usage = dict(state_payload.get("llm_usage") or {})
            llm_calls = list(state_payload.get("llm_calls") or [])
            planner_failure_reason = str(state_payload.get("planner_failure_reason") or "")
            plan_source = str(state_payload.get("plan_source") or "resume_checkpoint")
        else:
            log("[planning] generating workflow plan")
            t0 = time.perf_counter()
            plan_result = plan_workflow_detailed(adapter, planner_ctx, tool_names, constraints)
            planning_latency_sec = time.perf_counter() - t0
            plan = plan_result.plan
            llm_usage = plan_result.llm_usage or {}
            llm_calls = plan_result.llm_calls
            planner_failure_reason = plan_result.planner_failure_reason
            plan_source = plan_result.plan_source
            plan_valid, guard_issues = validate_plan(
                plan, meta_cols, tool_names,
                dataset_has_missing=has_missing,
                dataset_has_batch=has_batch,
            )

            if (not plan_valid) or plan_result.plan_source != "structured":
                replan_ctx = dict(planner_ctx)
                replan_ctx["runtime_observation"] = runtime_observation
                replan_ctx["previous_plan"] = plan.model_dump()
                replan_constraints = constraints + observation_constraints + [f"Fix prior validation issues: {guard_issues}"]
                log("[planning] initial plan needs repair, requesting replanning")
                t1 = time.perf_counter()
                replanned = plan_workflow_detailed(adapter, replan_ctx, tool_names, replan_constraints)
                planning_latency_sec += time.perf_counter() - t1
                replanned_valid, replanned_issues = validate_plan(
                    replanned.plan, meta_cols, tool_names,
                    dataset_has_missing=has_missing,
                    dataset_has_batch=has_batch,
                )
                replan_applied = True
                replan_reason = "invalid_initial_plan" if not plan_valid else f"planner_source={plan_result.plan_source}"
                chosen = _choose_replanned_result(plan_result, replanned, plan_valid, replanned_valid)
                if chosen is replanned:
                    plan_result = replanned
                    plan = replanned.plan
                    plan_valid = replanned_valid
                    guard_issues = replanned_issues
                    llm_usage = plan_result.llm_usage or {}
                    llm_calls = plan_result.llm_calls
                    planner_failure_reason = plan_result.planner_failure_reason
                    plan_source = plan_result.plan_source
            _persist_run_checkpoint(
                run_dir,
                status="planned",
                plan=plan,
                trace=[],
                artifacts=_build_base_artifacts(dataset),
                llm_usage=llm_usage,
                llm_calls=llm_calls,
                decision_log=decision_log,
                planning_latency_sec=planning_latency_sec,
                execution_latency_sec=0.0,
                planner_failure_reason=planner_failure_reason,
                plan_source=plan_source,
            )

        plan_valid, guard_issues = validate_plan(
            plan, meta_cols, tool_names,
            dataset_has_missing=has_missing,
            dataset_has_batch=has_batch,
        )
        if plan_valid:
            exec_result, execution_latency_sec = _run_full_workflow(
                plan,
                dataset,
                run_dir,
                llm_usage=llm_usage,
                llm_calls=llm_calls,
                planning_latency_sec=planning_latency_sec,
                planner_failure_reason=planner_failure_reason,
                plan_source=plan_source,
                log=log,
            )
        else:
            validation_trace_path = _write_validation_trace(run_dir / "execution_trace.json", len(plan.steps), guard_issues)
            exec_result = _invalid_execution_result(validation_trace_path, str(dataset.abundance_csv), plan)
            execution_latency_sec = 0.0
            log(f"[validation] plan invalid: {' | '.join(guard_issues)}")

    de_csv = exec_result.artifacts.get("run_differential_expression", exec_result.final_artifact)
    processed = exec_result.artifacts.get("abundance", str(dataset.abundance_csv))
    metric = evaluate_full(
        plan,
        exec_result.trace,
        raw_csv=str(dataset.abundance_csv),
        processed_csv=processed,
        de_csv=de_csv,
        metadata_csv=str(dataset.sample_metadata_csv),
        hidden_truth_json=str(dataset.hidden_truth_json),
        had_hard_errors=exec_result.had_hard_errors,
        plan_valid=plan_valid,
        decision_log=decision_log,
    )
    oracle = evaluate_oracle(plan, str(dataset.hidden_truth_json))

    tf = top_failures(metric, threshold=0.5)
    metric["llm_input_tokens"] = int(llm_usage.get("input_tokens", 0) or 0)
    metric["llm_output_tokens"] = int(llm_usage.get("output_tokens", 0) or 0)
    metric["llm_total_tokens"] = int(llm_usage.get("total_tokens", 0) or 0)
    metric["llm_call_count"] = int(llm_usage.get("call_count", 0) or 0)
    payload = {
        "model": model_cfg.model,
        "model_name": model_cfg.name,
        "dataset_id": summary["dataset_id"],
        "scenario_type": planner_ctx.get("scenario_type"),
        "task_type": planner_ctx.get("task_type"),
        "seed": seed,
        "planning_mode": planning_mode,
        "plan": plan.model_dump(),
        "plan_source": plan_source,
        "planner_failure_reason": planner_failure_reason,
        "plan_valid": plan_valid,
        "guard_issues": guard_issues,
        "replan_applied": replan_applied,
        "replan_reason": replan_reason,
        "llm_usage": llm_usage,
        "llm_calls": llm_calls,
        "decision_log": decision_log,
        "runtime_observation": runtime_observation,
        "preflight_inspect_path": str(preflight_inspect_path),
        "first_step_is_inspect": exec_result.first_step_is_inspect,
        "execution_trace_path": exec_result.trace_path,
        "execution_trace": exec_result.trace,
        "planning_latency_sec": planning_latency_sec,
        "execution_latency_sec": execution_latency_sec,
        "oracle": oracle,
        "metrics": metric,
        "top_failures": tf,
    }
    _json_dump(run_dir / "plan.json", plan.model_dump())
    _json_dump(run_dir / "oracle.json", oracle)
    _json_dump(run_dir / "metrics.json", metric)
    json_path, html_path = write_run_report(str(run_dir), payload)
    _persist_run_checkpoint(
        run_dir,
        status="completed",
        plan=plan,
        trace=exec_result.trace,
        artifacts=exec_result.artifacts,
        llm_usage=llm_usage,
        llm_calls=llm_calls,
        decision_log=decision_log,
        planning_latency_sec=planning_latency_sec,
        execution_latency_sec=execution_latency_sec,
        planner_failure_reason=planner_failure_reason,
        plan_source=plan_source,
    )
    log(f"[done] report={json_path}")
    return payload


def main() -> None:
    args = parse_args()
    try_load_dotenv()

    # Configure colored logging for adapter / planner loggers in this subprocess
    from src.app.log_utils import configure_colored_logging
    configure_colored_logging()
    apply_api_runtime_config(load_api_runtime_config(args.api_config))
    model_cfg = load_model_config(args.model)
    if model_cfg.model == "qwen3.5-plus" and os.getenv("GLYCOAGENT_MODEL"):
        model_cfg.model = os.environ["GLYCOAGENT_MODEL"].replace("openai:", "")
    dataset = load_dataset_paths(args.dataset)
    summary = summarize_dataset(dataset)
    planner_ctx = build_planner_context(summary, dataset_root=dataset.root)
    adapter = LLMAdapter(model_cfg)
    tool_names = list_tool_names()

    if args.run_dir_name:
        run_dir = Path(args.out_dir) / "runs" / _validate_run_dir_relative_path(args.run_dir_name)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_seg = _safe_run_dir_segment(model_cfg.name)
        leaf = f"{ts}__{_safe_run_dir_segment(str(summary['dataset_id']))}"
        run_dir = Path(args.out_dir) / "runs" / _validate_run_dir_relative_path(f"{model_seg}/{leaf}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _make_run_logger(run_dir)
    log(f"[run] model={model_cfg.name} dataset={summary['dataset_id']} planning_mode={args.planning_mode} run_dir={run_dir}")

    payload = _run_job_core(
        model_cfg=model_cfg,
        dataset=dataset,
        summary=summary,
        planner_ctx=planner_ctx,
        adapter=adapter,
        tool_names=tool_names,
        run_dir=run_dir,
        seed=args.seed,
        planning_mode=args.planning_mode,
        log=log,
    )
    json_path = str(run_dir / "report.json")
    html_path = str(run_dir / "report.html")
    print(f"Run complete.\nJSON: {json_path}\nHTML: {html_path}")


if __name__ == "__main__":
    main()
