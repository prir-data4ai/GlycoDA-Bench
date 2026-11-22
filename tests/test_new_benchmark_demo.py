from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.data.dataset import build_planner_context, load_dataset_paths, summarize_dataset
from src.llm.adapter import LLMAdapter
from src.planner.planner import plan_workflow
from src.tools.registry import list_tool_names


def test_mock_planner_outputs_structured_plan(tmp_path: Path):
    dataset = tmp_path / "mock_demo"
    dataset.mkdir()
    pd.DataFrame({"glycan_index": [1, 2], "s1": [0.6, 0.4], "s2": [0.5, 0.5]}).to_csv(
        dataset / "abundance.csv", index=False
    )
    pd.DataFrame({"group_label": [0, 1], "batch_label": [0, 0]}).to_csv(
        dataset / "sample_metadata.csv", index=False
    )
    (dataset / "manifest.json").write_text(
        '{"dataset_id":"mock_demo","dataset_summary":{"n_samples":2,"n_features":2,"n_batches":1}}',
        encoding="utf-8",
    )
    cfg = ModelConfig(name="mock", provider="mock", model="mock", allow_missing_api_key=True)
    adapter = LLMAdapter(cfg)
    paths = load_dataset_paths(dataset)
    summary = summarize_dataset(paths)
    ctx = build_planner_context(summary)
    plan = plan_workflow(adapter, ctx, list_tool_names(), ["test"])
    assert plan.workflow_name
    assert len(plan.steps) > 0
    assert all(step.tool for step in plan.steps)
