from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.llm.adapter import LLMAdapter
from src.planner.schema import DatasetAssessment, StepDecision, WorkflowPlan, WorkflowStep

logger = logging.getLogger(__name__)


@dataclass
class PlanGenerationResult:
    plan: WorkflowPlan
    plan_source: str
    planner_failure_reason: str = ""
    llm_usage: dict = field(default_factory=dict)
    llm_calls: list[dict] = field(default_factory=list)


@dataclass
class StepDecisionResult:
    decision: StepDecision
    decision_source: str
    planner_failure_reason: str = ""
    llm_usage: dict = field(default_factory=dict)
    llm_calls: list[dict] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _normalize_tool_name(raw: str, tool_names: list[str]) -> str | None:
    s = raw.strip().lower()
    if s in tool_names:
        return s
    aliases = {
        "impute": "impute_missing_values",
        "missing": "impute_missing_values",
        "clr": "clr_transform",
        "combat": "run_combat",
        "de": "run_differential_expression",
        "differential": "run_differential_expression",
        "pca": "run_pca",
        "inspect": "inspect_dataset",
        "batch_check": "check_batch_effect",
        "batch": "check_batch_effect",
        "harmony": "run_harmony",
        "percentile": "run_percentile_norm",
        "limma": "run_limma_batch_removal",
        "motif": "run_motif_enrichment",
        "ratio": "run_ratio_combat",
    }
    for k, v in aliases.items():
        if k in s and v in tool_names:
            return v
    return None


def _dedup_consecutive(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    """Remove runs of >=3 identical consecutive tools (likely parse artefacts)."""
    if len(steps) <= 2:
        return steps
    out: list[WorkflowStep] = [steps[0]]
    run = 1
    for i in range(1, len(steps)):
        if steps[i].tool == steps[i - 1].tool:
            run += 1
        else:
            run = 1
        if run < 3:
            out.append(steps[i])
    return out


def _strip_thinking_tags(text: str) -> str:
    """Remove non-answer reasoning blocks that can contain JSON-like braces."""
    return re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", str(text), flags=re.IGNORECASE).strip()


def _fallback_parse_plan(text: str, tool_names: list[str]) -> tuple[WorkflowPlan, str]:
    """Parse a plan from free-form text. Returns (plan, failure_reason)."""
    text = _strip_thinking_tags(text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            steps = obj.get("steps") or obj.get("workflow_steps") or []
            parsed_steps = []
            for s in steps:
                if not isinstance(s, dict):
                    continue
                t = s.get("tool") or s.get("name") or s.get("step") or s.get("method") or ""
                norm = _normalize_tool_name(str(t), tool_names)
                if norm:
                    raw_inputs = s.get("inputs", {})
                    raw_params = s.get("params", {})
                    parsed_steps.append(
                        WorkflowStep(
                            tool=norm,
                            inputs=raw_inputs if isinstance(raw_inputs, dict) else {},
                            params=raw_params if isinstance(raw_params, dict) else {},
                            purpose=str(s.get("purpose", "")),
                        )
                    )
            parsed_steps = _dedup_consecutive(parsed_steps)
            if parsed_steps:
                da_raw = obj.get("dataset_assessment")
                da = None
                if isinstance(da_raw, dict):
                    try:
                        da = DatasetAssessment.model_validate(da_raw)
                    except Exception:
                        da = DatasetAssessment()
                return (
                    WorkflowPlan(
                        workflow_name=obj.get("workflow_name", "llm_workflow"),
                        rationale=obj.get("rationale", obj.get("reasoning", "fallback parsed")),
                        steps=parsed_steps,
                        dataset_assessment=da,
                        final_analysis_target=str(obj.get("final_analysis_target", "")),
                        stop_conditions=list(obj.get("stop_conditions", []) or []),
                    ),
                    "fallback_json_parsed",
                )
        except json.JSONDecodeError as exc:
            logger.warning("Fallback JSON parse failed: %s", exc)

    steps = []
    for line in text.splitlines():
        norm = _normalize_tool_name(line, tool_names)
        if norm:
            steps.append(WorkflowStep(tool=norm))
    steps = _dedup_consecutive(steps)
    if steps:
        return (
            WorkflowPlan(
                workflow_name="fallback_workflow",
                rationale="Parsed from free-form LLM output (line scan)",
                steps=steps,
                dataset_assessment=DatasetAssessment(),
                final_analysis_target="differential_expression",
                stop_conditions=[],
            ),
            "fallback_line_parsed",
        )

    return (
        WorkflowPlan(
            workflow_name="empty_fallback",
            rationale="LLM produced no parseable plan",
            steps=[],
            dataset_assessment=None,
            final_analysis_target="",
            stop_conditions=[],
        ),
        "empty_plan_unparseable",
    )


def plan_workflow(
    adapter: LLMAdapter,
    planner_context: dict,
    tool_names: list[str],
    extra_constraints: list[str] | None = None,
) -> WorkflowPlan:
    return plan_workflow_detailed(adapter, planner_context, tool_names, extra_constraints).plan


def _usage_totals(llm_calls: list[dict]) -> dict:
    return {
        "input_tokens": sum(int(c.get("input_tokens", 0) or 0) for c in llm_calls),
        "output_tokens": sum(int(c.get("output_tokens", 0) or 0) for c in llm_calls),
        "total_tokens": sum(int(c.get("total_tokens", 0) or 0) for c in llm_calls),
        "call_count": len(llm_calls),
    }


def plan_workflow_detailed(
    adapter: LLMAdapter,
    planner_context: dict,
    tool_names: list[str],
    extra_constraints: list[str] | None = None,
) -> PlanGenerationResult:
    root = _repo_root()
    sys_prompt_path = root / "configs/prompts/planner_system.txt"
    user_prompt_path = root / "configs/prompts/planner_user.txt"
    system = (
        sys_prompt_path.read_text(encoding="utf-8")
        if sys_prompt_path.exists()
        else "You are a glycomics workflow planner. Output structured JSON only."
    )
    base_user = (
        user_prompt_path.read_text(encoding="utf-8")
        if user_prompt_path.exists()
        else "Create a valid workflow plan from context."
    )
    cons = list(extra_constraints or [])
    constraint_block = json.dumps(planner_context.get("constraints", {}), ensure_ascii=False)
    user = (
        f"{base_user}\n\n"
        f"Planner context (JSON; do not fabricate tool outputs): {json.dumps(planner_context, ensure_ascii=False)}\n"
        f"Allowed tool names: {tool_names}\n"
        f"Additional runtime constraints: {json.dumps(cons, ensure_ascii=False)}\n"
        f"Structured constraints object: {constraint_block}\n"
        "Output must match WorkflowPlan: workflow_name, rationale, dataset_assessment, "
        "decision_audit[{aspect, observed_evidence, judgment, chosen_action, rejected_actions, confidence}], "
        "steps[{tool, inputs, params, purpose}], final_analysis_target, stop_conditions."
    )

    structured_error: str | None = None
    llm_calls: list[dict] = []
    try:
        structured = adapter.invoke_structured_with_usage(system, user, WorkflowPlan)
        plan = structured.content
        llm_calls.append(
            {
                "stage": "structured_plan",
                "mode": "structured",
                **asdict(structured.usage),
            }
        )
        if plan.steps:
            plan.steps = _dedup_consecutive(plan.steps)
            usage_totals = _usage_totals(llm_calls)
            return PlanGenerationResult(
                plan=plan,
                plan_source="structured",
                llm_usage=usage_totals,
                llm_calls=llm_calls,
            )
        structured_error = "structured_output_returned_empty_steps"
    except Exception as exc:
        structured_error = f"structured_output_failed: {exc}"

    logger.info("Structured plan failed (%s), falling back to text parse", structured_error)
    text_result = adapter.invoke_text_with_usage(
        system,
        user + "\nReturn JSON with workflow_name, rationale, dataset_assessment, decision_audit, final_analysis_target, stop_conditions, steps[].",
    )
    text = str(text_result.content)
    llm_calls.append(
        {
            "stage": "fallback_text_plan",
            "mode": "text",
            **asdict(text_result.usage),
        }
    )
    plan, failure_reason = _fallback_parse_plan(text, tool_names)
    if failure_reason:
        logger.info("Fallback parse result: %s (steps=%d)", failure_reason, len(plan.steps))
        plan.rationale = f"[{failure_reason}] {plan.rationale}"
    usage_totals = _usage_totals(llm_calls)
    return PlanGenerationResult(
        plan=plan,
        plan_source="fallback_text",
        planner_failure_reason=f"{structured_error or 'structured_output_failed'} | {failure_reason}",
        llm_usage=usage_totals,
        llm_calls=llm_calls,
    )


def _fallback_parse_step_decision(text: str, tool_names: list[str]) -> tuple[StepDecision, str]:
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            done = bool(obj.get("done", False))
            raw_step = obj.get("next_step")
            if done:
                return (
                    StepDecision(
                        done=True,
                        rationale=str(obj.get("rationale", "")),
                        updated_final_analysis_target=str(obj.get("updated_final_analysis_target", "")),
                        stop_reason=str(obj.get("stop_reason", "llm_done")),
                    ),
                    "fallback_json_done",
                )
            if isinstance(raw_step, dict):
                t = raw_step.get("tool") or raw_step.get("name") or raw_step.get("step") or raw_step.get("method") or ""
                norm = _normalize_tool_name(str(t), tool_names)
                if norm:
                    return (
                        StepDecision(
                            done=False,
                            next_step=WorkflowStep(
                                tool=norm,
                                inputs=raw_step.get("inputs", {}) if isinstance(raw_step.get("inputs"), dict) else {},
                                params=raw_step.get("params", {}) if isinstance(raw_step.get("params"), dict) else {},
                                purpose=str(raw_step.get("purpose", "")),
                            ),
                            rationale=str(obj.get("rationale", "")),
                            updated_final_analysis_target=str(obj.get("updated_final_analysis_target", "")),
                            stop_reason=str(obj.get("stop_reason", "")),
                        ),
                        "fallback_json_step",
                    )
        except json.JSONDecodeError as exc:
            logger.warning("Fallback step-decision JSON parse failed: %s", exc)

    for line in text.splitlines():
        norm = _normalize_tool_name(line, tool_names)
        if norm:
            return (
                StepDecision(done=False, next_step=WorkflowStep(tool=norm), rationale="Parsed from free-form LLM output."),
                "fallback_line_step",
            )

    return (
        StepDecision(done=True, rationale="LLM produced no parseable next step.", stop_reason="unparseable_decision"),
        "fallback_unparseable_done",
    )


def plan_next_step_detailed(
    adapter: LLMAdapter,
    planner_context: dict,
    tool_names: list[str],
    extra_constraints: list[str] | None = None,
) -> StepDecisionResult:
    system = (
        "You are a glycomics workflow controller. "
        "Choose exactly one next analysis step at a time based on the dataset summary and accumulated execution observations. "
        "Do not invent tool outputs. Prefer minimal, valid, executable next steps."
    )
    cons = list(extra_constraints or [])
    user = (
        "Iterative workflow planning context (JSON): "
        f"{json.dumps(planner_context, ensure_ascii=False)}\n"
        f"Allowed tool names: {tool_names}\n"
        f"Additional runtime constraints: {json.dumps(cons, ensure_ascii=False)}\n"
        "Return structured JSON matching StepDecision: "
        "done, next_step{tool, inputs, params, purpose}, rationale, updated_final_analysis_target, stop_reason. "
        "Set done=true only when the workflow should stop."
    )

    structured_error: str | None = None
    llm_calls: list[dict] = []
    try:
        structured = adapter.invoke_structured_with_usage(system, user, StepDecision)
        decision = structured.content
        llm_calls.append(
            {
                "stage": "structured_next_step",
                "mode": "structured",
                **asdict(structured.usage),
            }
        )
        return StepDecisionResult(
            decision=decision,
            decision_source="structured",
            llm_usage=_usage_totals(llm_calls),
            llm_calls=llm_calls,
        )
    except Exception as exc:
        structured_error = f"structured_output_failed: {exc}"

    logger.info("Structured step decision failed (%s), falling back to text parse", structured_error)
    text_result = adapter.invoke_text_with_usage(
        system,
        user + "\nReturn JSON with done, next_step, rationale, updated_final_analysis_target, stop_reason.",
    )
    text = str(text_result.content)
    llm_calls.append(
        {
            "stage": "fallback_text_next_step",
            "mode": "text",
            **asdict(text_result.usage),
        }
    )
    decision, failure_reason = _fallback_parse_step_decision(text, tool_names)
    if failure_reason and not decision.rationale:
        decision.rationale = f"[{failure_reason}]"
    return StepDecisionResult(
        decision=decision,
        decision_source="fallback_text",
        planner_failure_reason=f"{structured_error or 'structured_output_failed'} | {failure_reason}",
        llm_usage=_usage_totals(llm_calls),
        llm_calls=llm_calls,
    )


# ------------------------------------------------------------------
# Async versions of planning functions for concurrent benchmark runs
# ------------------------------------------------------------------

async def aplan_workflow_detailed(
    adapter: LLMAdapter,
    planner_context: dict,
    tool_names: list[str],
    extra_constraints: list[str] | None = None,
) -> PlanGenerationResult:
    """Async version of plan_workflow_detailed using adapter's async methods."""
    root = _repo_root()
    sys_prompt_path = root / "configs/prompts/planner_system.txt"
    user_prompt_path = root / "configs/prompts/planner_user.txt"
    system = (
        sys_prompt_path.read_text(encoding="utf-8")
        if sys_prompt_path.exists()
        else "You are a glycomics workflow planner. Output structured JSON only."
    )
    base_user = (
        user_prompt_path.read_text(encoding="utf-8")
        if user_prompt_path.exists()
        else "Create a valid workflow plan from context."
    )
    cons = list(extra_constraints or [])
    constraint_block = json.dumps(planner_context.get("constraints", {}), ensure_ascii=False)
    user = (
        f"{base_user}\n\n"
        f"Planner context (JSON; do not fabricate tool outputs): {json.dumps(planner_context, ensure_ascii=False)}\n"
        f"Allowed tool names: {tool_names}\n"
        f"Additional runtime constraints: {json.dumps(cons, ensure_ascii=False)}\n"
        f"Structured constraints object: {constraint_block}\n"
        "Output must match WorkflowPlan: workflow_name, rationale, dataset_assessment, "
        "decision_audit[{aspect, observed_evidence, judgment, chosen_action, rejected_actions, confidence}], "
        "steps[{tool, inputs, params, purpose}], final_analysis_target, stop_conditions."
    )

    structured_error: str | None = None
    llm_calls: list[dict] = []
    try:
        structured = await adapter.ainvoke_structured_with_usage(system, user, WorkflowPlan)
        plan = structured.content
        llm_calls.append(
            {
                "stage": "structured_plan",
                "mode": "structured",
                **asdict(structured.usage),
            }
        )
        if plan.steps:
            plan.steps = _dedup_consecutive(plan.steps)
            usage_totals = _usage_totals(llm_calls)
            return PlanGenerationResult(
                plan=plan,
                plan_source="structured",
                llm_usage=usage_totals,
                llm_calls=llm_calls,
            )
        structured_error = "structured_output_returned_empty_steps"
    except Exception as exc:
        structured_error = f"structured_output_failed: {exc}"

    logger.info("Structured plan failed (%s), falling back to text parse", structured_error)
    text_result = await adapter.ainvoke_text_with_usage(
        system,
        user + "\nReturn JSON with workflow_name, rationale, dataset_assessment, decision_audit, final_analysis_target, stop_conditions, steps[].",
    )
    text = str(text_result.content)
    llm_calls.append(
        {
            "stage": "fallback_text_plan",
            "mode": "text",
            **asdict(text_result.usage),
        }
    )
    plan, failure_reason = _fallback_parse_plan(text, tool_names)
    if failure_reason:
        logger.info("Fallback parse result: %s (steps=%d)", failure_reason, len(plan.steps))
        plan.rationale = f"[{failure_reason}] {plan.rationale}"
    usage_totals = _usage_totals(llm_calls)
    return PlanGenerationResult(
        plan=plan,
        plan_source="fallback_text",
        planner_failure_reason=f"{structured_error or 'structured_output_failed'} | {failure_reason}",
        llm_usage=usage_totals,
        llm_calls=llm_calls,
    )


async def aplan_next_step_detailed(
    adapter: LLMAdapter,
    planner_context: dict,
    tool_names: list[str],
    extra_constraints: list[str] | None = None,
) -> StepDecisionResult:
    """Async version of plan_next_step_detailed using adapter's async methods."""
    system = (
        "You are a glycomics workflow controller. "
        "Choose exactly one next analysis step at a time based on the dataset summary and accumulated execution observations. "
        "Do not invent tool outputs. Prefer minimal, valid, executable next steps."
    )
    cons = list(extra_constraints or [])
    user = (
        "Iterative workflow planning context (JSON): "
        f"{json.dumps(planner_context, ensure_ascii=False)}\n"
        f"Allowed tool names: {tool_names}\n"
        f"Additional runtime constraints: {json.dumps(cons, ensure_ascii=False)}\n"
        "Return structured JSON matching StepDecision: "
        "done, next_step{tool, inputs, params, purpose}, rationale, updated_final_analysis_target, stop_reason. "
        "Set done=true only when the workflow should stop."
    )

    structured_error: str | None = None
    llm_calls: list[dict] = []
    try:
        structured = await adapter.ainvoke_structured_with_usage(system, user, StepDecision)
        decision = structured.content
        llm_calls.append(
            {
                "stage": "structured_next_step",
                "mode": "structured",
                **asdict(structured.usage),
            }
        )
        return StepDecisionResult(
            decision=decision,
            decision_source="structured",
            llm_usage=_usage_totals(llm_calls),
            llm_calls=llm_calls,
        )
    except Exception as exc:
        structured_error = f"structured_output_failed: {exc}"

    logger.info("Structured step decision failed (%s), falling back to text parse", structured_error)
    text_result = await adapter.ainvoke_text_with_usage(
        system,
        user + "\nReturn JSON with done, next_step, rationale, updated_final_analysis_target, stop_reason.",
    )
    text = str(text_result.content)
    llm_calls.append(
        {
            "stage": "fallback_text_next_step",
            "mode": "text",
            **asdict(text_result.usage),
        }
    )
    decision, failure_reason = _fallback_parse_step_decision(text, tool_names)
    if failure_reason and not decision.rationale:
        decision.rationale = f"[{failure_reason}]"
    return StepDecisionResult(
        decision=decision,
        decision_source="fallback_text",
        planner_failure_reason=f"{structured_error or 'structured_output_failed'} | {failure_reason}",
        llm_usage=_usage_totals(llm_calls),
        llm_calls=llm_calls,
    )
