from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.skip("Legacy score-architecture tests replaced by aspect-based evaluator tests.", allow_module_level=True)

from src.agents.guards import validate_plan
from src.data.dataset import build_planner_context, load_dataset_paths, summarize_dataset
from src.executor.executor import execute_step, execute_workflow
from src.metrics.evaluator import compute_plan_score, evaluate_full
from src.planner.planner import plan_next_step_detailed
from src.planner.schema import WorkflowPlan, WorkflowStep


def test_dataset_contract_mock_demo():
    paths = load_dataset_paths("data/mock_demo")
    assert paths.abundance_csv.exists()
    assert paths.manifest_json.exists()
    man = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert "dataset_summary" in man
    assert man["constraints"].get("hidden_truth_forbidden") is True
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    assert "true_workflow_tags" in ht or "ground_truth" in ht


def test_planner_context_strips_oracle_like_task_hints():
    paths = load_dataset_paths("data/mock_demo")
    summary = summarize_dataset(paths)
    ctx = build_planner_context(summary, dataset_root=paths.root)
    known = ctx["task"]["known_constraints"]
    assert "missing_values_present" not in known
    assert "batch_structure_expected" not in known
    assert "compositionally_closed" not in known
    assert "batch_labels_available" in known


def test_planner_context_uses_task_card_scenario_fallback(tmp_path: Path):
    task_payload = {
        "known_constraints": {
            "scenario_type": "batch_only",
            "batch_labels_available": True,
        }
    }
    (tmp_path / "task.json").write_text(json.dumps(task_payload), encoding="utf-8")
    summary = {
        "dataset_id": "tmp_case",
        "manifest": {"task_type": "workflow_recommendation", "scenario_type": "unknown"},
        "dataset_summary": {},
    }
    ctx = build_planner_context(summary, dataset_root=tmp_path)
    assert ctx["scenario_type"] == "batch_only"


def test_workflow_plan_coerces_object_stop_conditions():
    """Models sometimes emit stop_conditions as [{\"condition\": \"...\"}, ...]; schema must accept."""
    raw = {
        "workflow_name": "t",
        "rationale": "r",
        "steps": [{"tool": "inspect_dataset", "inputs": {}, "params": {}, "purpose": ""}],
        "stop_conditions": [
            {"condition": "No significant batch effect with QC diagnostics"},
            {"condition": "Severe batch confounding; apply batch correction method"},
        ],
    }
    plan = WorkflowPlan.model_validate(raw)
    assert len(plan.stop_conditions) == 2
    assert all(isinstance(x, str) for x in plan.stop_conditions)
    assert "batch effect" in plan.stop_conditions[0]


def test_workflow_plan_accepts_decision_audit():
    raw = {
        "workflow_name": "t",
        "rationale": "r",
        "decision_audit": [
            {
                "aspect": "batch",
                "observed_evidence": "batch labels available",
                "judgment": "diagnose first",
                "chosen_action": "check_batch_effect",
                "rejected_actions": "run_combat immediately",
                "confidence": "0.8",
            }
        ],
        "steps": [{"tool": "inspect_dataset", "inputs": {}, "params": {}, "purpose": ""}],
    }
    plan = WorkflowPlan.model_validate(raw)
    assert len(plan.decision_audit) == 1
    assert plan.decision_audit[0].aspect == "batch"
    assert plan.decision_audit[0].rejected_actions == ["run_combat immediately"]


def test_fallback_parse_plan_accepts_object_stop_conditions():
    from src.planner.planner import _fallback_parse_plan
    from src.tools.registry import list_tool_names

    text = (
        '{"workflow_name":"w","rationale":"r","steps":[{"tool":"clr_transform"}],'
        '"stop_conditions":[{"condition":"Stop A"},{"condition":"Stop B"}]}'
    )
    plan, reason = _fallback_parse_plan(text, list_tool_names())
    assert plan.steps
    assert reason == "fallback_json_parsed"
    assert plan.stop_conditions == ["Stop A", "Stop B"]


def test_planner_schema_roundtrip_mock():
    from src.config import load_model_config
    from src.llm.adapter import LLMAdapter
    from src.planner.planner import plan_workflow_detailed
    from src.tools.registry import list_tool_names

    cfg = load_model_config("configs/models/mock.yaml")
    adapter = LLMAdapter(cfg)
    paths = load_dataset_paths("data/mock_demo")
    summary = summarize_dataset(paths)
    ctx = build_planner_context(summary)
    result = plan_workflow_detailed(adapter, ctx, list_tool_names(), ["test"])
    plan = result.plan
    assert plan.workflow_name
    assert plan.dataset_assessment is not None or len(plan.steps) > 0
    assert result.plan_source == "structured"
    assert result.planner_failure_reason == ""
    dumped = plan.model_dump()
    assert "steps" in dumped


def test_executor_runs_mock_plan(tmp_path: Path):
    from src.config import load_model_config
    from src.llm.adapter import LLMAdapter
    from src.planner.planner import plan_workflow
    from src.tools.registry import list_tool_names

    cfg = load_model_config("configs/models/mock.yaml")
    adapter = LLMAdapter(cfg)
    paths = load_dataset_paths("data/mock_demo")
    summary = summarize_dataset(paths)
    ctx = build_planner_context(summary)
    plan = plan_workflow(adapter, ctx, list_tool_names(), [])
    out = execute_workflow(
        plan,
        str(paths.abundance_csv),
        str(paths.sample_metadata_csv),
        str(tmp_path / "tool_outputs"),
        clr_csv=str(paths.clr_csv),
    )
    assert out.trace_path
    assert not out.had_hard_errors
    trace = json.loads(Path(out.trace_path).read_text(encoding="utf-8"))
    assert isinstance(trace, list)
    assert trace[0].get("status") == "ok"


def test_iterative_next_step_mock():
    from src.config import load_model_config
    from src.llm.adapter import LLMAdapter
    from src.tools.registry import list_tool_names

    cfg = load_model_config("configs/models/mock.yaml")
    adapter = LLMAdapter(cfg)
    paths = load_dataset_paths("data/mock_demo")
    summary = summarize_dataset(paths)
    ctx = build_planner_context(summary)
    ctx["runtime_observation"] = {"missing_fraction": 0.1, "batch_label_available": True}
    ctx["executed_tools"] = []
    ctx["execution_trace_summary"] = []
    result = plan_next_step_detailed(adapter, ctx, list_tool_names(), ["First step must be inspect_dataset."])
    assert result.decision.done is False
    assert result.decision.next_step is not None
    assert result.decision.next_step.tool == "inspect_dataset"


def test_execute_step_updates_artifacts(tmp_path: Path):
    paths = load_dataset_paths("data/mock_demo")
    artifacts = {
        "abundance": str(paths.abundance_csv),
        "clr": str(paths.clr_csv),
        "metadata": str(paths.sample_metadata_csv),
    }
    next_artifacts, trace_entry, had_error = execute_step(
        WorkflowStep(tool="inspect_dataset"),
        artifacts,
        str(paths.sample_metadata_csv),
        str(tmp_path / "tool_outputs"),
        1,
        prior_tools=set(),
    )
    assert had_error is False
    assert trace_entry["status"] == "ok"
    assert "inspect_dataset" in next_artifacts


def test_execute_step_sanitizes_harmful_params(tmp_path: Path):
    paths = load_dataset_paths("data/mock_demo")
    artifacts = {
        "abundance": str(paths.abundance_csv),
        "clr": str(paths.clr_csv),
        "metadata": str(paths.sample_metadata_csv),
    }

    _, trace_entry, had_error = execute_step(
        WorkflowStep(tool="clr_transform", params={"pseudocount": 0}),
        artifacts,
        str(paths.sample_metadata_csv),
        str(tmp_path / "tool_outputs"),
        1,
        prior_tools=set(),
    )
    assert had_error is False
    assert trace_entry["status"] == "ok"
    assert trace_entry["params"]["pseudocount"] == 1e-6
    assert any("pseudocount" in w for w in trace_entry["warnings"])


def test_full_workflow_resume_reuses_completed_prefix(tmp_path: Path):
    from src.app.run_single_model import _run_full_workflow, _trace_path

    paths = load_dataset_paths("data/mock_demo")
    plan = WorkflowPlan(
        workflow_name="resume_case",
        rationale="resume",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
        ],
    )
    run_dir = tmp_path / "resume_case"
    artifacts = {
        "abundance": str(paths.abundance_csv),
        "clr": str(paths.clr_csv),
        "metadata": str(paths.sample_metadata_csv),
    }
    _, first_trace, had_error = execute_step(
        WorkflowStep(tool="inspect_dataset"),
        artifacts,
        str(paths.sample_metadata_csv),
        str(run_dir / "tool_outputs"),
        1,
        prior_tools=set(),
    )
    assert had_error is False
    _trace_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    _trace_path(run_dir).write_text(json.dumps([first_trace], indent=2), encoding="utf-8")

    exec_result, _ = _run_full_workflow(
        plan,
        paths,
        run_dir,
        llm_usage={},
        llm_calls=[],
        planning_latency_sec=0.0,
        planner_failure_reason="",
        plan_source="structured",
        log=lambda _msg: None,
    )
    assert len(exec_result.trace) == 3
    assert exec_result.trace[0]["started_at"] == first_trace["started_at"]
    assert exec_result.trace[1]["tool"] == "impute_missing_values"
    assert exec_result.trace[2]["tool"] == "clr_transform"


def test_evaluator_three_layer_scores():
    paths = load_dataset_paths("data/mock_demo")
    plan_good = WorkflowPlan(
        workflow_name="good",
        rationale="x",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    trace = [{"status": "ok", "tool": "inspect_dataset", "warnings": []}]
    m = evaluate_full(
        plan_good,
        trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )
    assert "plan_score" in m and "execution_score" in m and "decision_score" in m and "result_score" in m
    assert "composite_score" not in m


def test_full_mode_decision_audit_affects_decision_score():
    paths = load_dataset_paths("data/mock_demo")
    trace = [{"status": "ok", "tool": "inspect_dataset", "warnings": []}]
    plan_rich = WorkflowPlan(
        workflow_name="rich",
        rationale="x",
        decision_audit=[
            {
                "aspect": "missingness",
                "observed_evidence": ["missing fraction observed"],
                "judgment": "impute required",
                "chosen_action": "impute_missing_values",
                "rejected_actions": ["skip_imputation"],
                "confidence": 0.9,
            },
            {
                "aspect": "compositionality",
                "observed_evidence": ["compositional row sums"],
                "judgment": "clr before de",
                "chosen_action": "clr_transform",
                "rejected_actions": ["de_on_raw_abundance"],
                "confidence": 0.9,
            },
            {
                "aspect": "batch",
                "observed_evidence": ["batch labels", "batch shift"],
                "judgment": "diagnose then correct",
                "chosen_action": "check_batch_effect then run_combat",
                "rejected_actions": ["ignore_batch"],
                "confidence": 0.8,
            },
        ],
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_thin = WorkflowPlan(
        workflow_name="thin",
        rationale="x",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    m_rich = evaluate_full(
        plan_rich,
        trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )
    m_thin = evaluate_full(
        plan_thin,
        trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )
    assert m_rich["decision_score"] > m_thin["decision_score"]
    assert m_thin["decision_score"] > 0.05


def test_bad_plan_lower_plan_score():
    paths = load_dataset_paths("data/mock_demo")
    plan_bad = WorkflowPlan(
        workflow_name="bad",
        rationale="skip impute and clr",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth

    truth = _normalize_truth(ht)
    s_bad = compute_plan_score(plan_bad, truth)["plan_score"]
    plan_good = WorkflowPlan(
        workflow_name="good",
        rationale="ok",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    s_good = compute_plan_score(plan_good, truth)["plan_score"]
    assert s_bad < s_good


def test_validate_plan_blocks_bad_order_and_params():
    plan = WorkflowPlan(
        workflow_name="bad",
        rationale="bad order and params",
        steps=[
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_differential_expression", params={"alpha": 2}),
            WorkflowStep(tool="run_combat"),
        ],
    )
    ok, issues = validate_plan(
        plan,
        metadata_columns=["sample_id", "batch", "label"],
        available_tools=["inspect_dataset", "clr_transform", "run_differential_expression", "run_combat"],
    )
    assert not ok
    assert any("First step must be inspect_dataset" in x for x in issues)
    assert any("batch correction appears after differential expression" in x for x in issues)
    assert any("alpha must be in (0, 1]" in x for x in issues)


def test_guard_relaxes_impute_when_no_missing():
    """clr without impute is accepted when dataset has no missing values."""
    plan = WorkflowPlan(
        workflow_name="no_missing",
        rationale="clean data",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    ok_strict, issues_strict = validate_plan(
        plan,
        metadata_columns=["sample_id", "label"],
        available_tools=["inspect_dataset", "clr_transform", "run_differential_expression"],
    )
    assert not ok_strict
    assert any("imputation" in x.lower() for x in issues_strict)

    ok_relaxed, issues_relaxed = validate_plan(
        plan,
        metadata_columns=["sample_id", "label"],
        available_tools=["inspect_dataset", "clr_transform", "run_differential_expression"],
        dataset_has_missing=False,
    )
    assert ok_relaxed
    assert not any("imputation" in x.lower() for x in issues_relaxed)


def test_plan_invalid_penalty_applied():
    """Invalid plans should have scores penalized (halved)."""
    paths = load_dataset_paths("data/mock_demo")
    plan = WorkflowPlan(
        workflow_name="test",
        rationale="x",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    trace = [{"status": "ok", "tool": "inspect_dataset", "warnings": []}]
    m_valid = evaluate_full(
        plan, trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
        plan_valid=True,
    )
    m_invalid = evaluate_full(
        plan, trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
        plan_valid=False,
    )
    assert m_invalid["plan_score"] < m_valid["plan_score"]
    assert m_invalid["execution_score"] < m_valid["execution_score"]
    assert m_invalid["plan_valid_penalty_applied"] is True
    assert m_valid["plan_valid_penalty_applied"] is False


def test_oracle_mostly_correct_verdict():
    """Oracle returns 'mostly_correct' only for a single whitelisted minor partial.

    Only 'starts without inspect_dataset' maps to mostly_correct when it is the sole partial.
    """
    from src.oracle.rules import evaluate_oracle

    paths = load_dataset_paths("data/mock_demo")
    plan = WorkflowPlan(
        workflow_name="starts_with_impute",
        rationale="skip inspect; otherwise protocol-consistent for mock_demo truth",
        steps=[
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    result = evaluate_oracle(plan, str(paths.hidden_truth_json))
    assert result["verdict"] == "mostly_correct", f"Got {result['verdict']}: {result['explanations']}"


def test_oracle_optional_motif_when_not_worthwhile_is_partially_reasonable():
    """Optional motif when GT says not worthwhile is no longer 'mostly_correct' alone (stricter oracle)."""
    import tempfile

    from src.oracle.rules import evaluate_oracle

    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    ht.setdefault("true_workflow_tags", {})["motif_followup_worthwhile"] = False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ht, f)
        tmp_truth = f.name

    plan = WorkflowPlan(
        workflow_name="near_perfect",
        rationale="adds optional motif when not required",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
            WorkflowStep(tool="run_motif_enrichment"),
        ],
    )
    result = evaluate_oracle(plan, tmp_truth)
    Path(tmp_truth).unlink()
    assert result["verdict"] == "partially_reasonable", f"Got {result['verdict']}: {result['explanations']}"


def test_oracle_empty_plan_is_wrong():
    from src.oracle.rules import evaluate_oracle

    paths = load_dataset_paths("data/mock_demo")
    plan = WorkflowPlan(workflow_name="empty", rationale="nothing", steps=[])
    result = evaluate_oracle(plan, str(paths.hidden_truth_json))
    assert result["verdict"] == "wrong"


def test_executor_skips_unknown_tool(tmp_path: Path):
    """Unknown tools should be skipped, not abort the entire execution."""
    paths = load_dataset_paths("data/mock_demo")
    plan = WorkflowPlan(
        workflow_name="with_unknown",
        rationale="test",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="nonexistent_tool_xyz"),
            WorkflowStep(tool="clr_transform"),
        ],
    )
    out = execute_workflow(
        plan,
        str(paths.abundance_csv),
        str(paths.sample_metadata_csv),
        str(tmp_path / "tool_outputs"),
        clr_csv=str(paths.clr_csv),
    )
    assert out.had_hard_errors
    statuses = [t["status"] for t in out.trace]
    assert "skipped" in statuses
    assert "ok" in statuses


def test_batch_tools_single_source():
    """BATCH_TOOLS should be importable from registry and used everywhere."""
    from src.tools.registry import BATCH_TOOLS as reg_bt
    from src.metrics.evaluator import BATCH_TOOLS as eval_bt  # noqa: F811 — re-export

    assert reg_bt is eval_bt
    assert "run_combat" in reg_bt
    assert isinstance(reg_bt, frozenset)


def test_stable_run_dir_name_uses_dataset_id():
    from src.app.run_benchmark import _stable_run_dir_name

    name = _stable_run_dir_name(
        Path("configs/models/mock.yaml"),
        Path("data/mock_demo"),
        seed=42,
        planning_mode="iterative",
    )
    assert name == "seed42__iterative__mock_demo"


def test_combat_preserves_biological_signal(tmp_path: Path):
    """ComBat should remove batch effect but preserve group differences."""
    import numpy as np
    import pandas as pd
    from src.tools.impl import run_combat

    rng = np.random.default_rng(42)
    n_per_group = 20
    n_features = 10
    n = n_per_group * 2

    bio_effect = np.zeros(n_features)
    bio_effect[:3] = 3.0

    data = rng.normal(0, 1, (n, n_features))
    groups = np.array([0] * n_per_group + [1] * n_per_group)
    batches = np.array([0, 1] * n)[:n]
    data[groups == 1] += bio_effect
    data[batches == 1] += 2.0

    df = pd.DataFrame(data, columns=[f"F{i}" for i in range(n_features)])
    df.to_csv(tmp_path / "input.csv", index=False)
    meta = pd.DataFrame({"sample_id": [f"S{i}" for i in range(n)], "group_label": groups, "batch_label": batches})
    meta.to_csv(tmp_path / "meta.csv", index=False)

    out_path = str(tmp_path / "combat_out.csv")
    run_combat(str(tmp_path / "input.csv"), out_path, str(tmp_path / "meta.csv"), use_bio_covariate=True)

    result = pd.read_csv(out_path)
    from scipy.stats import ttest_ind

    for i in range(3):
        a = result[f"F{i}"].values[groups == 0]
        b = result[f"F{i}"].values[groups == 1]
        _, p = ttest_ind(a, b, equal_var=False)
        assert p < 0.05, f"Feature F{i} lost biological signal after ComBat (p={p:.4f})"

    for i in range(3, n_features):
        a = result[f"F{i}"].values[groups == 0]
        b = result[f"F{i}"].values[groups == 1]
        _, p = ttest_ind(a, b, equal_var=False)
        assert p > 0.01, f"Feature F{i} has false positive after ComBat (p={p:.6f})"


def test_plan_score_penalizes_unnecessary_batch():
    """Plan with unnecessary batch correction should score lower."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth

    truth = _normalize_truth(ht)
    truth["true_workflow_tags"]["needs_batch_correction"] = False
    truth["true_workflow_tags"]["batch_decision_profile"] = "avoid"
    truth["true_workflow_tags"]["recommended_batch_family"] = "none"
    truth["true_workflow_tags"]["motif_followup_worthwhile"] = False
    truth["true_conditions"]["batch_injected"] = False

    plan_with_batch = WorkflowPlan(
        workflow_name="over_engineered",
        rationale="adds batch when not needed",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_correct = WorkflowPlan(
        workflow_name="minimal",
        rationale="no batch needed",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    s_over = compute_plan_score(plan_with_batch, truth)["plan_score"]
    s_correct = compute_plan_score(plan_correct, truth)["plan_score"]
    assert s_over < s_correct, f"Unnecessary batch not penalized: {s_over} >= {s_correct}"


def test_plan_template_penalty_is_reported():
    """Template penalty should be surfaced when optional heavy steps are blindly applied."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth

    truth = _normalize_truth(ht)
    truth["true_workflow_tags"]["needs_batch_correction"] = False
    truth["true_workflow_tags"]["batch_decision_profile"] = "avoid"
    truth["true_conditions"]["batch_injected"] = False

    plan_with_batch = WorkflowPlan(
        workflow_name="template_like",
        rationale="always run same pipeline",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    out = compute_plan_score(plan_with_batch, truth)
    assert out.get("template_penalty", 0.0) > 0.0


def test_plan_score_prefers_batch_check_on_borderline_cases():
    """Borderline batch cases should reward checking before default correction."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth

    truth = _normalize_truth(ht)
    truth["true_workflow_tags"]["needs_batch_correction"] = True
    truth["true_workflow_tags"]["batch_decision_profile"] = "borderline"
    truth["true_workflow_tags"]["recommended_batch_family"] = "any_batch_tool"

    plan_checked = WorkflowPlan(
        workflow_name="checked",
        rationale="batch looks borderline so verify first",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_direct = WorkflowPlan(
        workflow_name="direct",
        rationale="always correct batch",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    s_checked = compute_plan_score(plan_checked, truth)["plan_score"]
    s_direct = compute_plan_score(plan_direct, truth)["plan_score"]
    assert s_checked > s_direct, (
        f"Borderline plan with explicit checking ({s_checked:.3f}) should outscore direct correction ({s_direct:.3f})"
    )


def test_execution_score_gate_caps_critical_failures():
    """Execution score should be capped on critical requirement failures."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth, compute_execution_score

    truth = _normalize_truth(ht)
    truth["true_workflow_tags"]["needs_batch_correction"] = True
    truth["true_workflow_tags"]["needs_imputation"] = True

    plan_missing_required = WorkflowPlan(
        workflow_name="missing_required",
        rationale="skips required steps",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    trace_ok = [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_missing_required.steps]
    out = compute_execution_score(plan_missing_required, trace_ok, truth, had_hard_errors=False)
    assert out["execution_gate_cap_applied"] is True
    assert out["execution_score"] < 0.6


def test_oracle_penalizes_unnecessary_batch():
    """Oracle should flag unnecessary batch correction."""
    from src.oracle.rules import evaluate_oracle

    paths = load_dataset_paths("data/mock_demo")
    truth = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    truth["true_conditions"]["batch_injected"] = False
    truth["batch_effect_significant"] = False
    truth["ground_truth"]["batch_effect_significant"] = False
    truth["true_workflow_tags"]["batch_decision_profile"] = "avoid"
    truth["true_workflow_tags"]["recommended_batch_family"] = "none"

    truth_path = Path("data/mock_demo/hidden_truth.json")
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(truth, f)
        tmp_truth = f.name

    plan = WorkflowPlan(
        workflow_name="over_batch",
        rationale="adds batch when not needed",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    result = evaluate_oracle(plan, tmp_truth)
    Path(tmp_truth).unlink()
    assert result["verdict"] in ("mostly_correct", "partially_reasonable"), (
        f"Expected partial penalty for unnecessary batch, got: {result['verdict']}"
    )


def test_parsimony_differentiates_minimal_vs_bloated():
    """Bloated plans with redundant QC steps should get lower plan_score and execution_score."""
    from src.metrics.evaluator import _normalize_truth, compute_execution_score

    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    truth = _normalize_truth(ht)

    plan_minimal = WorkflowPlan(
        workflow_name="minimal",
        rationale="exactly what is needed",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_bloated = WorkflowPlan(
        workflow_name="bloated",
        rationale="lots of extra steps",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_pca"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_pca"),
            WorkflowStep(tool="run_differential_expression"),
            WorkflowStep(tool="run_motif_enrichment"),
        ],
    )

    ps_min = compute_plan_score(plan_minimal, truth)
    ps_bloat = compute_plan_score(plan_bloated, truth)
    assert ps_min["plan_score"] > ps_bloat["plan_score"], (
        f"Minimal plan ({ps_min['plan_score']:.3f}) should outscore bloated ({ps_bloat['plan_score']:.3f})"
    )

    trace_ok = [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_minimal.steps]
    trace_bloat = [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_bloated.steps]
    es_min = compute_execution_score(plan_minimal, trace_ok, truth, had_hard_errors=False)
    es_bloat = compute_execution_score(plan_bloated, trace_bloat, truth, had_hard_errors=False)
    assert es_min["execution_score"] > es_bloat["execution_score"], (
        f"Minimal exec ({es_min['execution_score']:.3f}) should outscore bloated ({es_bloat['execution_score']:.3f})"
    )


def test_plan_score_rewards_paired_acknowledgement_and_motif_judgment():
    """Plans should acknowledge paired structure and avoid skipping worthwhile motif follow-up."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth

    truth = _normalize_truth(ht)
    truth["true_workflow_tags"]["paired_strategy_required"] = True
    truth["true_workflow_tags"]["motif_followup_worthwhile"] = True
    truth["true_conditions"]["confounded_signal"] = True

    plan_aware = WorkflowPlan(
        workflow_name="aware",
        rationale="Paired samples are partially confounded with batch, so preserve biology and state a paired fallback.",
        final_analysis_target="paired differential analysis with motif follow-up",
        stop_conditions=["Stop if pair matching is broken."],
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": True}),
            WorkflowStep(tool="run_differential_expression"),
            WorkflowStep(tool="run_motif_enrichment"),
        ],
    )
    plan_blind = WorkflowPlan(
        workflow_name="blind",
        rationale="standard workflow",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": True}),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )

    s_aware = compute_plan_score(plan_aware, truth)["plan_score"]
    s_blind = compute_plan_score(plan_blind, truth)["plan_score"]
    assert s_aware > s_blind, (
        f"Plan that acknowledges pairing and motif follow-up ({s_aware:.3f}) should outscore blind plan ({s_blind:.3f})"
    )


def test_param_quality_differentiates_good_vs_bad():
    """Plans with poor parameter choices should get lower scores."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth

    truth = _normalize_truth(ht)

    plan_good_params = WorkflowPlan(
        workflow_name="good_params",
        rationale="reasonable defaults",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values", params={"strategy": "median"}),
            WorkflowStep(tool="clr_transform", params={"pseudocount": 1e-6}),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": True}),
            WorkflowStep(tool="run_differential_expression", params={"alpha": 0.05}),
        ],
    )
    plan_bad_params = WorkflowPlan(
        workflow_name="bad_params",
        rationale="questionable parameter choices",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values", params={"strategy": "zero_fill"}),
            WorkflowStep(tool="clr_transform", params={"pseudocount": 5.0}),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": False}),
            WorkflowStep(tool="run_differential_expression", params={"alpha": 0.5}),
        ],
    )

    s_good = compute_plan_score(plan_good_params, truth)["plan_score"]
    s_bad = compute_plan_score(plan_bad_params, truth)["plan_score"]
    assert s_good > s_bad, (
        f"Good params ({s_good:.3f}) should outscore bad params ({s_bad:.3f})"
    )


def test_param_quality_has_finer_separation_across_gradations():
    """Parameter quality should separate mild, medium, and severe parameter issues."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth, compute_execution_score

    truth = _normalize_truth(ht)

    plan_mild = WorkflowPlan(
        workflow_name="mild_params",
        rationale="slightly loose alpha",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values", params={"strategy": "median"}),
            WorkflowStep(tool="clr_transform", params={"pseudocount": 1e-6}),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": True}),
            WorkflowStep(tool="run_differential_expression", params={"alpha": 0.12}),
        ],
    )
    plan_medium = WorkflowPlan(
        workflow_name="medium_params",
        rationale="worse alpha and pseudocount",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values", params={"strategy": "median"}),
            WorkflowStep(tool="clr_transform", params={"pseudocount": 0.2}),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": False}),
            WorkflowStep(tool="run_differential_expression", params={"alpha": 0.25}),
        ],
    )
    plan_severe = WorkflowPlan(
        workflow_name="severe_params",
        rationale="multiple poor parameter choices",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values", params={"strategy": "zero_fill"}),
            WorkflowStep(tool="clr_transform", params={"pseudocount": 5.0, "samples_are_rows": True}),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": False}),
            WorkflowStep(tool="run_differential_expression", params={"alpha": 0.5, "input_space": "weird_space"}),
            WorkflowStep(tool="run_pca", params={"n_components": 20}),
        ],
    )

    mild_plan = compute_plan_score(plan_mild, truth)["plan_checks"]["param_quality"]
    medium_plan = compute_plan_score(plan_medium, truth)["plan_checks"]["param_quality"]
    severe_plan = compute_plan_score(plan_severe, truth)["plan_checks"]["param_quality"]

    trace_mild = [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_mild.steps]
    trace_medium = [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_medium.steps]
    trace_severe = [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_severe.steps]

    mild_exec = compute_execution_score(plan_mild, trace_mild, truth, had_hard_errors=False)["params_reasonable"]
    medium_exec = compute_execution_score(plan_medium, trace_medium, truth, had_hard_errors=False)["params_reasonable"]
    severe_exec = compute_execution_score(plan_severe, trace_severe, truth, had_hard_errors=False)["params_reasonable"]

    assert mild_plan > medium_plan > severe_plan, (mild_plan, medium_plan, severe_plan)
    assert mild_exec > medium_exec > severe_exec, (mild_exec, medium_exec, severe_exec)


def test_decision_audit_and_protocol_have_finer_separation():
    """Decision sub-metrics should no longer collapse to the same few constants."""
    paths = load_dataset_paths("data/mock_demo")
    trace = [{"status": "ok", "tool": "inspect_dataset", "warnings": []}]

    def _run(plan: WorkflowPlan, decision_log: list[dict]) -> dict:
        return evaluate_full(
            plan,
            trace,
            raw_csv=str(paths.abundance_csv),
            processed_csv=str(paths.abundance_csv),
            de_csv=str(paths.abundance_csv),
            metadata_csv=str(paths.sample_metadata_csv),
            hidden_truth_json=str(paths.hidden_truth_json),
            had_hard_errors=False,
            decision_log=decision_log,
        )

    plan_strong = WorkflowPlan(
        workflow_name="strong_decision",
        rationale="x",
        decision_audit=[
            {
                "aspect": "missingness",
                "observed_evidence": ["missing fraction observed", "na fraction elevated"],
                "judgment": "impute required",
                "chosen_action": "impute_missing_values",
                "rejected_actions": ["skip_imputation", "zero_fill"],
                "confidence": 0.95,
            },
            {
                "aspect": "compositionality",
                "observed_evidence": ["compositional row sums", "clr required"],
                "judgment": "clr before de",
                "chosen_action": "clr_transform",
                "rejected_actions": ["de_on_raw_abundance"],
                "confidence": 0.9,
            },
            {
                "aspect": "batch",
                "observed_evidence": ["batch labels", "batch shift", "batch risk"],
                "judgment": "diagnose then correct",
                "chosen_action": "check_batch_effect then run_combat",
                "rejected_actions": ["ignore_batch", "correct_without_check"],
                "confidence": 0.9,
            },
            {
                "aspect": "post_de_followup",
                "observed_evidence": ["de significant features", "post de followup"],
                "judgment": "motif followup not needed",
                "chosen_action": "no_followup",
                "rejected_actions": ["run_motif_enrichment"],
                "confidence": 0.85,
            },
        ],
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_medium = WorkflowPlan(
        workflow_name="medium_decision",
        rationale="x",
        decision_audit=[
            {
                "aspect": "missingness",
                "observed_evidence": ["missing"],
                "judgment": "impute maybe",
                "chosen_action": "impute_missing_values",
                "rejected_actions": [],
                "confidence": 0.75,
            },
            {
                "aspect": "compositionality",
                "observed_evidence": ["row"],
                "judgment": "clr before de",
                "chosen_action": "clr_transform",
                "rejected_actions": [],
                "confidence": 0.7,
            },
            {
                "aspect": "batch",
                "observed_evidence": ["batch labels"],
                "judgment": "check batch",
                "chosen_action": "check_batch_effect",
                "rejected_actions": [],
                "confidence": 0.7,
            },
        ],
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_thin = WorkflowPlan(
        workflow_name="thin_decision",
        rationale="x",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )

    strong = _run(plan_strong, [])
    medium = _run(plan_medium, [{"chosen_tool": "run_combat", "stop_reason": "invalid_next_tool_for_stage"}])
    thin = _run(plan_thin, [{"chosen_tool": "run_combat", "stop_reason": "invalid_next_tool_for_stage"}, {"chosen_tool": "run_combat", "stop_reason": "invalid_next_tool_for_stage"}])

    assert strong["decision_audit_batch_quality"] > medium["decision_audit_batch_quality"] > thin["decision_audit_batch_quality"]
    assert strong["decision_protocol_compliance"] > medium["decision_protocol_compliance"] > thin["decision_protocol_compliance"]
    assert strong["decision_score"] > medium["decision_score"] > thin["decision_score"]


def test_aggregate_scores_are_stricter_but_still_separate_quality_levels():
    """Aggregate process scores should be harder to max out while preserving ordering."""
    paths = load_dataset_paths("data/mock_demo")
    ht = json.loads(paths.hidden_truth_json.read_text(encoding="utf-8"))
    from src.metrics.evaluator import _normalize_truth, compute_execution_score

    truth = _normalize_truth(ht)
    trace = [{"status": "ok", "tool": "inspect_dataset", "warnings": []}]

    plan_good = WorkflowPlan(
        workflow_name="good",
        rationale="paired-aware and biologically careful",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat", params={"use_bio_covariate": True}),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_mid = WorkflowPlan(
        workflow_name="mid",
        rationale="reasonable but skips some needed rigor",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="check_batch_effect"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_bad = WorkflowPlan(
        workflow_name="bad",
        rationale="too thin",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )

    good_plan = compute_plan_score(plan_good, truth)["plan_score"]
    mid_plan = compute_plan_score(plan_mid, truth)["plan_score"]
    bad_plan = compute_plan_score(plan_bad, truth)["plan_score"]

    good_exec = compute_execution_score(plan_good, [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_good.steps], truth, had_hard_errors=False)["execution_score"]
    mid_exec = compute_execution_score(plan_mid, [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_mid.steps], truth, had_hard_errors=False)["execution_score"]
    bad_exec = compute_execution_score(plan_bad, [{"status": "ok", "tool": s.tool, "warnings": []} for s in plan_bad.steps], truth, had_hard_errors=False)["execution_score"]

    good_dec = evaluate_full(
        plan_good, trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )["decision_score"]
    mid_dec = evaluate_full(
        plan_mid, trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )["decision_score"]
    bad_dec = evaluate_full(
        plan_bad, trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )["decision_score"]

    assert good_plan > mid_plan > bad_plan
    assert good_exec > mid_exec >= bad_exec
    assert good_dec > mid_dec > bad_dec
    assert good_plan < 0.7
    assert good_exec < 0.8


def test_decision_scores_vary_without_decision_log_from_rationale_and_audit_quality():
    """Decision defaults should not collapse to near-constants when decision_log is absent."""
    paths = load_dataset_paths("data/mock_demo")
    trace = [{"status": "ok", "tool": "inspect_dataset", "warnings": []}]

    plan_generic = WorkflowPlan(
        workflow_name="thin",
        rationale="x",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    plan_specific = WorkflowPlan(
        workflow_name="specific",
        rationale="Observed missingness motivates imputation, CLR should precede DE, and batch structure should be corrected before DE while preserving biology.",
        final_analysis_target="DE after imputation, CLR, and batch-aware correction",
        decision_audit=[
            {
                "aspect": "missingness",
                "observed_evidence": ["missing fraction observed"],
                "judgment": "impute before downstream steps",
                "chosen_action": "impute_missing_values",
                "rejected_actions": ["skip_imputation"],
                "confidence": 0.9,
            },
            {
                "aspect": "compositionality",
                "observed_evidence": ["row sums and composition-like structure"],
                "judgment": "clr needed before de",
                "chosen_action": "clr_transform",
                "rejected_actions": ["de_on_raw_abundance"],
                "confidence": 0.9,
            },
            {
                "aspect": "batch",
                "observed_evidence": ["batch labels available", "batch risk"],
                "judgment": "correct batch before de",
                "chosen_action": "run_combat",
                "rejected_actions": ["ignore_batch"],
                "confidence": 0.8,
            },
        ],
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )

    generic = evaluate_full(
        plan_generic,
        trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )
    specific = evaluate_full(
        plan_specific,
        trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )

    assert specific["decision_protocol_compliance"] > generic["decision_protocol_compliance"]
    assert specific["decision_rationale_quality"] > generic["decision_rationale_quality"]
    assert specific["decision_score"] > generic["decision_score"]


def test_outcome_alignment_gate_is_mild_for_non_catastrophic_results():
    """Result alignment should not dominate process scores once outputs are at least minimally plausible."""
    paths = load_dataset_paths("data/mock_demo")
    plan = WorkflowPlan(
        workflow_name="good",
        rationale="impute, clr, then batch-aware de",
        steps=[
            WorkflowStep(tool="inspect_dataset"),
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    trace = [{"status": "ok", "tool": "inspect_dataset", "warnings": []}]
    m = evaluate_full(
        plan,
        trace,
        raw_csv=str(paths.abundance_csv),
        processed_csv=str(paths.abundance_csv),
        de_csv=str(paths.abundance_csv),
        metadata_csv=str(paths.sample_metadata_csv),
        hidden_truth_json=str(paths.hidden_truth_json),
        had_hard_errors=False,
    )
    assert 0.84 <= m["outcome_alignment_gate"] <= 1.0
