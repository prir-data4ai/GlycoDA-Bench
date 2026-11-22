from __future__ import annotations

from pathlib import Path

import pytest

from src.data.dataset import resolve_hidden_truth_path
from src.oracle.rules import evaluate_oracle
from src.planner.schema import WorkflowPlan, WorkflowStep


def test_oracle_marks_missing_imputation_as_partial():
    plan = WorkflowPlan(
        workflow_name="bad_plan",
        rationale="missing imputation",
        steps=[
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    ht = resolve_hidden_truth_path(Path("data/mock_demo"))
    if not ht.is_file():
        pytest.skip("mock_demo hidden truth not available")
    out = evaluate_oracle(plan, str(ht))
    assert out["verdict"] == "wrong"


def test_oracle_missing_inspect_is_not_auto_wrong():
    plan = WorkflowPlan(
        workflow_name="no_inspect_but_structured",
        rationale="starts directly with preprocessing",
        steps=[
            WorkflowStep(tool="impute_missing_values"),
            WorkflowStep(tool="clr_transform"),
            WorkflowStep(tool="run_combat"),
            WorkflowStep(tool="run_differential_expression"),
        ],
    )
    ht = resolve_hidden_truth_path(Path("data/mock_demo"))
    if not ht.is_file():
        pytest.skip("mock_demo hidden truth not available")
    out = evaluate_oracle(plan, str(ht))
    assert out["verdict"] in {"mostly_correct", "partially_reasonable"}
