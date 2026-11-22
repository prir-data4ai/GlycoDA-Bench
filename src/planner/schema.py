from __future__ import annotations

import json

from pydantic import BaseModel, Field, model_validator


def _normalize_stop_condition_items(raw: object) -> list[str]:
    """LLMs often emit stop_conditions as objects like {\"condition\": \"...\"}; coerce to strings."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            if item.strip():
                out.append(item)
        elif isinstance(item, dict):
            text = (
                item.get("condition")
                or item.get("text")
                or item.get("description")
                or item.get("message")
                or item.get("rationale")
            )
            if text:
                out.append(str(text))
            else:
                out.append(json.dumps(item, ensure_ascii=False))
        elif item is not None:
            out.append(str(item))
    return out


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class DatasetAssessment(BaseModel):
    """Planner's assessment from manifest/summary only (no hidden truth)."""

    detected_missingness: bool = False
    detected_batch_risk: bool = False
    detected_compositionality: bool = True
    recommended_input_matrix: str = "abundance"

    @model_validator(mode="before")
    @classmethod
    def _coerce_bools(cls, data):
        if not isinstance(data, dict):
            return data
        for key in ("detected_missingness", "detected_batch_risk", "detected_compositionality"):
            v = data.get(key)
            if isinstance(v, dict):
                data[key] = any(v.values()) if v else False
            elif v is not None and not isinstance(v, bool):
                data[key] = bool(v)
        return data


class WorkflowStep(BaseModel):
    tool: str
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)
    purpose: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data):
        if not isinstance(data, dict):
            return data
        for key in ("inputs", "params"):
            v = data.get(key)
            if isinstance(v, list):
                data[key] = {}
            elif v is not None and not isinstance(v, dict):
                data[key] = {}
        return data


class DecisionAuditItem(BaseModel):
    aspect: str
    observed_evidence: list[str] = Field(default_factory=list)
    judgment: str = ""
    chosen_action: str = ""
    rejected_actions: list[str] = Field(default_factory=list)
    confidence: float = 0.5

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data):
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for key in ("observed_evidence", "rejected_actions"):
            v = out.get(key)
            if v is None:
                out[key] = []
            elif isinstance(v, str):
                out[key] = [v] if v.strip() else []
            elif not isinstance(v, list):
                out[key] = [str(v)]
        conf = out.get("confidence", 0.5)
        try:
            out["confidence"] = float(conf)
        except (TypeError, ValueError):
            out["confidence"] = 0.5
        return out


class WorkflowPlan(BaseModel):
    workflow_name: str
    rationale: str
    steps: list[WorkflowStep]
    dataset_assessment: DatasetAssessment | None = None
    decision_audit: list[DecisionAuditItem] = Field(default_factory=list)
    final_analysis_target: str = ""
    stop_conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _compat_keys(cls, data):
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "steps" not in out and "workflow_steps" in out:
            mapped = []
            for s in out.get("workflow_steps", []):
                if isinstance(s, dict):
                    mapped.append(
                        {
                            "tool": s.get("tool") or s.get("name"),
                            "inputs": s.get("inputs", {}),
                            "params": s.get("params", {}),
                            "purpose": s.get("purpose", ""),
                        }
                    )
            out["steps"] = mapped
        if "workflow_name" not in out:
            out["workflow_name"] = out.get("name", "llm_workflow")
        if "rationale" not in out:
            out["rationale"] = out.get("reasoning", "LLM generated plan")
        if "dataset_assessment" not in out:
            out["dataset_assessment"] = None
        if "decision_audit" not in out:
            audit = out.get("decision_table") or out.get("judgment_audit") or []
            out["decision_audit"] = audit if isinstance(audit, list) else []
        if "final_analysis_target" not in out:
            out["final_analysis_target"] = out.get("final_target", "")
        raw_sc = out.get("stop_conditions")
        out["stop_conditions"] = _normalize_stop_condition_items(raw_sc if raw_sc is not None else [])
        return out


class StepDecision(BaseModel):
    done: bool = False
    next_step: WorkflowStep | None = None
    rationale: str = ""
    updated_final_analysis_target: str = ""
    stop_reason: str = ""

    @model_validator(mode="after")
    def _validate_done_state(self):
        if self.done and self.next_step is not None:
            return self
        if (not self.done) and self.next_step is None:
            raise ValueError("next_step is required when done=false")
        return self
