from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.metrics.aspect_evaluator import (
    ASPECT_NAMES,
    _batch_reduction_score,
    _bio_preservation_score,
    _compute_ndcg_at_k,
    _manuscript_outcome_scores,
    compute_plan_score,
    evaluate_audit_closure,
)
from src.metrics.evaluator import evaluate_full
from src.planner.schema import WorkflowPlan


def _write_basic_bundle(tmp_path: Path) -> tuple[str, str, str, str, str]:
    raw = pd.DataFrame(
        {
            "glycan_index": [1, 2],
            "s1": [10.0, 0.1],
            "s2": [10.0, 0.2],
            "s3": [0.1, 10.0],
            "s4": [0.2, 10.0],
        }
    )
    meta = pd.DataFrame({"group_label": [0, 0, 1, 1], "batch_label": [0, 0, 0, 0]})
    de = pd.DataFrame(
        {
            "feature": ["G002", "G001"],
            "pvalue": [1e-6, 0.2],
            "padj": [1e-5, 0.4],
            "effect": [1.2, -0.2],
            "significant": [True, False],
            "rank": [1, 2],
        }
    )
    truth = {"dataset_id": "t", "true_differential": {"feature_ids": ["G002"]}, "true_workflow_tags": {"needs_imputation": False, "needs_clr": True, "needs_batch_correction": False}}
    raw_p = tmp_path / "raw.csv"
    proc_p = tmp_path / "proc.csv"
    meta_p = tmp_path / "meta.csv"
    de_p = tmp_path / "de.csv"
    truth_p = tmp_path / "truth.json"
    raw.to_csv(raw_p, index=False)
    raw.to_csv(proc_p, index=False)
    meta.to_csv(meta_p, index=False)
    de.to_csv(de_p, index=False)
    truth_p.write_text(json.dumps(truth), encoding="utf-8")
    return str(raw_p), str(proc_p), str(de_p), str(meta_p), str(truth_p)


def test_evaluate_full_returns_new_aspect_schema(tmp_path: Path) -> None:
    raw_csv, processed_csv, de_csv, metadata_csv, truth_json = _write_basic_bundle(tmp_path)
    plan = WorkflowPlan.model_validate(
        {
            "workflow_name": "clr_then_de",
            "rationale": "Use CLR for compositional data before differential expression.",
            "steps": [
                {"tool": "inspect_dataset", "inputs": {}, "params": {}, "purpose": "inspect"},
                {"tool": "clr_transform", "inputs": {}, "params": {"pseudocount": 1e-6, "samples_are_rows": False}, "purpose": "clr"},
                {"tool": "run_differential_expression", "inputs": {}, "params": {"alpha": 0.05, "multiple_testing": "bh"}, "purpose": "de"},
            ],
            "decision_audit": [
                {"aspect": "compositionality", "observed_evidence": ["rows are compositional"], "judgment": "clr required", "chosen_action": "apply clr", "rejected_actions": ["skip clr"], "confidence": 0.9}
            ],
        }
    )
    metrics = evaluate_full(
        plan,
        execution_trace=[{"tool": "inspect_dataset", "input": raw_csv, "params": {"metadata_csv": metadata_csv}, "status": "ok", "warnings": []}],
        raw_csv=raw_csv,
        processed_csv=processed_csv,
        de_csv=de_csv,
        metadata_csv=metadata_csv,
        hidden_truth_json=truth_json,
    )
    process_aspects = ASPECT_NAMES[:7]
    outcome_aspects = ASPECT_NAMES[7:]
    for aspect in process_aspects:
        assert aspect in metrics
        assert metrics[aspect] in {0.0, 0.5, 1.0}
    for aspect in outcome_aspects:
        assert aspect in metrics
        assert 0.0 <= metrics[aspect] <= 1.0
    assert "plan_score" not in metrics
    assert metrics["de_set_recovery"] == 1.0
    assert metrics["de_ranking_quality"] == 1.0


def test_audit_closure_uses_new_four_part_rule() -> None:
    rich = WorkflowPlan.model_validate(
        {
            "workflow_name": "x",
            "rationale": "x",
            "steps": [],
            "decision_audit": [
                {"aspect": "missingness", "observed_evidence": ["0 missing"], "judgment": "skip impute", "chosen_action": "skip", "rejected_actions": ["impute"], "confidence": 0.9},
                {"aspect": "batch", "observed_evidence": ["low signal"], "judgment": "skip batch", "chosen_action": "skip", "rejected_actions": ["combat"], "confidence": 0.9},
                {"aspect": "compositionality", "observed_evidence": ["compositional"], "judgment": "use clr", "chosen_action": "clr", "rejected_actions": ["raw"], "confidence": 0.9},
                {"aspect": "post_de_followup", "observed_evidence": ["motif requested"], "judgment": "run motif", "chosen_action": "motif", "rejected_actions": ["stop"], "confidence": 0.9},
            ],
        }
    )
    sparse = WorkflowPlan.model_validate({"workflow_name": "x", "rationale": "x", "steps": [], "decision_audit": [{"aspect": "batch", "observed_evidence": ["signal"], "judgment": "check"}]})
    assert evaluate_audit_closure(rich)[0] == 1.0
    assert evaluate_audit_closure(sparse)[0] in {0.0, 0.5}


def test_core_workflow_correctness_gives_full_credit_when_all_core_steps_match() -> None:
    plan = WorkflowPlan.model_validate(
        {
            "workflow_name": "full_core",
            "rationale": "Impute, CLR, handle batch before DE.",
            "steps": [
                {"tool": "inspect_dataset", "inputs": {}, "params": {}, "purpose": "inspect"},
                {"tool": "impute_missing_values", "inputs": {}, "params": {}, "purpose": "impute"},
                {"tool": "clr_transform", "inputs": {}, "params": {"pseudocount": 1e-6}, "purpose": "clr"},
                {"tool": "run_combat", "inputs": {}, "params": {}, "purpose": "batch"},
                {"tool": "run_differential_expression", "inputs": {}, "params": {}, "purpose": "de"},
            ],
        }
    )
    truth = {
        "true_conditions": {"missing_fraction_target": 0.2},
        "true_workflow_tags": {
            "needs_clr": True,
            "missing_decision_profile": "required",
            "batch_decision_profile": "required",
        },
    }

    metrics = compute_plan_score(plan, truth)

    assert metrics["core_workflow_correctness"] == 1.0
    assert metrics["aspect_details"]["core_workflow_correctness"] == {
        "missing_required_and_done": 1.0,
        "clr_required_and_before_de": 1.0,
        "batch_required_and_before_de": 1.0,
    }


def test_core_workflow_correctness_allows_partial_credit() -> None:
    plan = WorkflowPlan.model_validate(
        {
            "workflow_name": "partial_core",
            "rationale": "Use CLR before DE but skip imputation and batch correction.",
            "steps": [
                {"tool": "inspect_dataset", "inputs": {}, "params": {}, "purpose": "inspect"},
                {"tool": "clr_transform", "inputs": {}, "params": {"pseudocount": 1e-6}, "purpose": "clr"},
                {"tool": "run_differential_expression", "inputs": {}, "params": {}, "purpose": "de"},
            ],
        }
    )
    truth = {
        "true_conditions": {"missing_fraction_target": 0.2},
        "true_workflow_tags": {
            "needs_clr": True,
            "missing_decision_profile": "required",
            "batch_decision_profile": "required",
        },
    }

    metrics = compute_plan_score(plan, truth)

    assert metrics["aspect_details"]["core_workflow_correctness"] == {
        "missing_required_and_done": 0.0,
        "clr_required_and_before_de": 1.0,
        "batch_required_and_before_de": 0.0,
    }
    assert metrics["core_workflow_correctness"] == 0.5


def test_de_ranking_quality_uses_ndcg_at_20(tmp_path: Path) -> None:
    raw_csv, processed_csv, _, metadata_csv, truth_json = _write_basic_bundle(tmp_path)
    de = pd.DataFrame(
        {
            "feature": ["G999", "G002", "G001"],
            "padj": [0.001, 0.002, 0.003],
            "pvalue": [0.001, 0.002, 0.003],
            "effect": [5.0, 1.2, 0.1],
            "significant": [False, True, False],
            "rank": [1, 2, 3],
        }
    )
    de_csv = tmp_path / "de_ranked.csv"
    de.to_csv(de_csv, index=False)
    plan = WorkflowPlan.model_validate(
        {
            "workflow_name": "clr_then_de",
            "rationale": "Use CLR for compositional data before differential expression.",
            "steps": [
                {"tool": "inspect_dataset", "inputs": {}, "params": {}, "purpose": "inspect"},
                {"tool": "clr_transform", "inputs": {}, "params": {"pseudocount": 1e-6, "samples_are_rows": False}, "purpose": "clr"},
                {"tool": "run_differential_expression", "inputs": {}, "params": {"alpha": 0.05, "multiple_testing": "bh"}, "purpose": "de"},
            ],
        }
    )

    metrics = evaluate_full(
        plan,
        execution_trace=[{"tool": "inspect_dataset", "input": raw_csv, "params": {"metadata_csv": metadata_csv}, "status": "ok", "warnings": []}],
        raw_csv=raw_csv,
        processed_csv=processed_csv,
        de_csv=str(de_csv),
        metadata_csv=metadata_csv,
        hidden_truth_json=truth_json,
    )

    expected_ndcg = _compute_ndcg_at_k(["G999", "G002", "G001"], {"G002"}, k=20)
    assert metrics["de_average_precision"] == 0.5
    assert metrics["de_top10_overlap"] == 1.0
    assert metrics["de_top20_overlap"] == 1.0
    assert metrics["de_ndcg_at_20"] == expected_ndcg
    assert metrics["de_ranking_quality"] == expected_ndcg
    assert metrics["aspect_details"]["de_ranking_quality"]["de_ndcg_at_20"] == expected_ndcg
    assert metrics["aspect_details"]["de_ranking_quality"]["de_ranking_quality"] == expected_ndcg


def test_biological_signal_preservation_uses_structure_as_auxiliary_signal() -> None:
    assert _bio_preservation_score(1.0, 1.0, 0.0) == 1.0
    assert _bio_preservation_score(1.0, 0.5, 0.5) == 0.5
    assert _bio_preservation_score(0.5, 1.0, 0.5) == 0.5
    assert _bio_preservation_score(1.0, 0.5, 0.0) == 0.0
    assert _bio_preservation_score(0.0, 1.0, 1.0) == 0.0


def test_batch_reduction_score_uses_fractional_pvca_drop() -> None:
    assert _batch_reduction_score(0.25, 0.05) == pytest.approx(0.8)
    assert _batch_reduction_score(0.25, 0.30) == 0.0
    assert _batch_reduction_score(0.0, 0.0) == 0.0


def test_manuscript_outcomes_use_continuous_result_metrics() -> None:
    scores = _manuscript_outcome_scores(
        {
            "de_f1": 0.25,
            "de_ndcg_at_20": 0.60,
            "batch_reduction_score": 0.375,
            "bio_preservation_score": 0.625,
        }
    )

    assert scores == {
        "de_set_recovery": 0.25,
        "de_ranking_quality": 0.60,
        "batch_removal_quality": 0.375,
        "biological_signal_preservation": 0.625,
    }
