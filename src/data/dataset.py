from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.registry import export_tool_specs_for_manifest


def resolve_hidden_truth_path(dataset_dir: str | Path) -> Path:
    """Resolve evaluator ground truth: optional env dir, then pack-private file, then legacy in-case file.

    Order avoids leaking truth into agent-visible directories: prefer
    ``GLYCOAGENT_HIDDEN_TRUTH_ROOT/<case_name>.json`` or
    ``<pack_root>/_private_hidden_truth/<case_name>.json`` over ``<case>/hidden_truth.json``.
    """
    root = Path(dataset_dir).resolve()
    env_root = (os.environ.get("GLYCOAGENT_HIDDEN_TRUTH_ROOT") or "").strip()
    if env_root:
        ep = Path(env_root) / f"{root.name}.json"
        if ep.is_file():
            return ep
    private = root.parent / "_private_hidden_truth" / f"{root.name}.json"
    if private.is_file():
        return private
    return root / "hidden_truth.json"


@dataclass
class DatasetPaths:
    root: Path
    abundance_csv: Path
    clr_csv: Path
    sample_metadata_csv: Path
    manifest_json: Path
    hidden_truth_json: Path
    task_json: Path
    feature_metadata_csv: Path


def load_dataset_paths(dataset_dir: str | Path) -> DatasetPaths:
    root = Path(dataset_dir)
    return DatasetPaths(
        root=root,
        abundance_csv=root / "abundance.csv",
        clr_csv=root / "clr.csv",
        sample_metadata_csv=root / "sample_metadata.csv",
        manifest_json=root / "manifest.json",
        hidden_truth_json=resolve_hidden_truth_path(root),
        task_json=root / "task.json",
        feature_metadata_csv=root / "feature_metadata.csv",
    )


def _compute_dataset_summary(abundance: pd.DataFrame, meta: pd.DataFrame, clr_path: Path) -> dict:
    # GlycoForge wide: rows = glycans, columns = sample intensities (+ glycan ID column).
    feature_col = "glycan_id" if "glycan_id" in abundance.columns else ("glycan_index" if "glycan_index" in abundance.columns else None)
    if feature_col is not None:
        numeric = abundance.drop(columns=[feature_col]).apply(pd.to_numeric, errors="coerce")
        arr = np.asarray(numeric, dtype=float)
        n_samples = int(numeric.shape[1])
        n_features = int(numeric.shape[0])
        row_sums = np.nansum(arr, axis=1)
    else:
        arr = np.asarray(abundance, dtype=float)
        n_samples = int(abundance.shape[0])
        n_features = int(abundance.shape[1])
        row_sums = np.nansum(arr, axis=1)
    miss = float(np.isnan(arr).sum() / max(arr.size, 1))
    const_like = float(np.std(row_sums) / max(np.mean(np.abs(row_sums)), 1e-12)) if len(row_sums) else 1.0
    n_batches = 0
    if "batch_label" in meta.columns:
        n_batches = int(meta["batch_label"].nunique())
    elif "batch" in meta.columns:
        n_batches = int(meta["batch"].nunique())
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "matrix_layout": "glycan_x_sample" if feature_col is not None else "sample_x_feature",
        "has_missing_values": bool(miss > 1e-9),
        "missing_fraction_observed": miss,
        "has_batch_labels": bool({"batch_label", "batch"} & set(meta.columns)),
        "n_batches": n_batches,
        "is_compositional": bool(const_like < 0.08 and max(n_features, n_samples) > 1),
        "has_clr_matrix": clr_path.exists(),
        "source": "glycoforge" if "glycoforge" in str(clr_path).lower() else "unknown",
    }


def summarize_dataset(paths: DatasetPaths) -> dict:
    abundance = pd.read_csv(paths.abundance_csv)
    meta = pd.read_csv(paths.sample_metadata_csv)
    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    ds = manifest.get("dataset_summary")
    if not ds:
        ds = _compute_dataset_summary(abundance, meta, paths.clr_csv)
        ds["source"] = manifest.get("source", ds.get("source", "unknown"))
    dataset_id = manifest.get("dataset_id", paths.root.name)
    return {
        "dataset_id": dataset_id,
        "n_rows": int(ds.get("n_samples", abundance.shape[0])),
        "n_cols": int(ds.get("n_features", abundance.shape[1])),
        "missing_fraction": float(ds.get("missing_fraction_observed", abundance.isna().sum().sum() / max(abundance.size, 1))),
        "n_batches": int(ds.get("n_batches", meta["batch_label"].nunique() if "batch_label" in meta.columns else 1)),
        "manifest": manifest,
        "dataset_summary": ds,
    }


def _infer_scenario_type(manifest: dict) -> str:
    """When manifest omits scenario_type, infer from generation.params (legacy GlycoForge packs)."""
    st = manifest.get("scenario_type")
    if st and str(st).lower() not in {"", "unknown"}:
        return str(st)
    gen = manifest.get("generation") or {}
    params = gen.get("params") or {}
    if not params:
        return str(manifest.get("scenario", "unknown"))
    nb = int(params.get("n_batches", 1) or 1)
    mf = float(params.get("missing_fraction", 0.0) or 0.0)
    if nb <= 1 and mf > 0.05:
        return "missing_only"
    if nb > 1 and mf <= 0.02:
        return "batch_only"
    if nb > 1 and mf > 0.05:
        return "batch_and_missing"
    return "unknown"


def _load_task_card(dataset_root: Path) -> dict | None:
    task_path = dataset_root / "task.json"
    if task_path.exists():
        return json.loads(task_path.read_text(encoding="utf-8"))
    return None


def _planner_safe_task_view(task_card: dict | None) -> dict:
    """Strip oracle-like hints that collapse the planning problem into template matching."""
    if not isinstance(task_card, dict):
        return {}
    known = dict(task_card.get("known_constraints", {}) or {})
    safe_constraints = {
        "data_type": known.get("data_type", ""),
        "batch_labels_available": bool(known.get("batch_labels_available", False)),
        "paired_or_coupled_requested": bool(known.get("paired_or_coupled_requested", False)),
        "motif_driven_signal_requested": bool(known.get("motif_driven_signal_requested", False)),
        "scenario_type": known.get("scenario_type", ""),
    }
    schema = dict(task_card.get("schema", {}) or {})
    return {
        "analysis_goal": task_card.get("analysis_goal", ""),
        "task_description": task_card.get("task_description", ""),
        "known_constraints": safe_constraints,
        "schema": schema,
        "expected_output": task_card.get("expected_output", {}),
        "decision_focus": list(task_card.get("decision_focus", []) or []),
        "stop_conditions": list(task_card.get("stop_conditions", []) or []),
    }


def build_planner_context(summary: dict, dataset_root: Path | None = None) -> dict:
    """Structured context for planner: no raw matrices, manifest-driven.

    When a task.json exists, its fields are promoted to top-level planner
    context so the LLM sees an explicit task card with analysis goal,
    data type, known constraints, and expected output.
    """
    manifest = summary.get("manifest", {})
    tools = manifest.get("available_tools") or export_tool_specs_for_manifest()
    constraints = manifest.get("constraints") or {
        "hidden_truth_forbidden": True,
        "max_steps": 20,
        "must_output_json_plan": True,
    }
    task_card = None
    if dataset_root is not None:
        task_card = _load_task_card(dataset_root)
    scenario = manifest.get("scenario_type") or manifest.get("scenario")
    if not scenario or str(scenario).lower() == "unknown":
        scenario = _infer_scenario_type(manifest)
    if (not scenario or str(scenario).lower() == "unknown") and isinstance(task_card, dict):
        kc = dict(task_card.get("known_constraints", {}) or {})
        task_scenario = str(kc.get("scenario_type", "") or "").strip()
        if task_scenario and task_scenario.lower() != "unknown":
            scenario = task_scenario

    ctx: dict = {
        "dataset_id": summary["dataset_id"],
        "task_type": manifest.get("task_type", "workflow_recommendation"),
        "scenario_type": scenario,
        "dataset_summary": summary.get("dataset_summary", {}),
        "constraints": constraints,
        "available_tools": tools,
        "schema": manifest.get("schema", {}),
    }

    if task_card:
        ctx["task"] = _planner_safe_task_view(task_card)
    else:
        ctx["task"] = manifest.get("task", {})

    return ctx
