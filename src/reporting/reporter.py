from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.metrics.aspect_evaluator import ASPECT_NAMES

RAW_METRIC_COLUMNS = [
    "batch_reduction_score",
    "bio_preservation_score",
    "de_recovery_score",
    "de_precision",
    "de_recall",
    "de_f1",
    "de_fpr",
    "de_average_precision",
    "de_top10_overlap",
    "de_top20_overlap",
    "f_stat_corr_preservation",
    "corr_structure_preservation",
    "observed_batch_signal_score",
]


def leaderboard_aspect_details_from_metrics(m: dict) -> dict[str, object]:
    details = m.get("aspect_details") or {}
    out: dict[str, object] = {}
    for aspect, items in details.items():
        if not isinstance(items, dict):
            continue
        for key, value in items.items():
            out[f"{aspect}__{key}"] = value
    return out


def leaderboard_aspect_details_skipped() -> dict[str, object]:
    return {}


def leaderboard_classify_failure(payload: dict) -> str:
    if not payload.get("plan", {}).get("steps"):
        return "empty_plan"
    if not payload.get("plan_valid", True):
        return "invalid_plan"
    oracle = payload.get("oracle", {}).get("verdict", "")
    if oracle == "wrong":
        return "wrong_oracle"
    if float(payload.get("metrics", {}).get("core_workflow_correctness", 0.0)) <= 0.0:
        return "core_workflow_failure"
    return "none"


def leaderboard_row_from_report_payload(payload: dict) -> dict:
    m = payload.get("metrics", {})
    oracle = payload.get("oracle") or {}
    row = {
        "model_name": payload["model_name"],
        "dataset_id": payload["dataset_id"],
        "scenario_type": payload.get("scenario_type", ""),
        "task_type": payload.get("task_type", ""),
        "oracle_verdict": oracle.get("verdict", ""),
        **{name: m.get(name, 0.0) for name in ASPECT_NAMES},
        **{name: m.get(name, "") for name in RAW_METRIC_COLUMNS},
        "llm_input_tokens": m.get("llm_input_tokens", (payload.get("llm_usage") or {}).get("input_tokens", 0)),
        "llm_output_tokens": m.get("llm_output_tokens", (payload.get("llm_usage") or {}).get("output_tokens", 0)),
        "llm_total_tokens": m.get("llm_total_tokens", (payload.get("llm_usage") or {}).get("total_tokens", 0)),
        "llm_call_count": m.get("llm_call_count", (payload.get("llm_usage") or {}).get("call_count", 0)),
        "plan_valid": payload.get("plan_valid", True),
        "plan_source": payload.get("plan_source", ""),
        "failure_type": leaderboard_classify_failure(payload),
        "run_status": "completed",
        "skip_reason": "",
        "skip_detail": "",
        "planning_latency_sec": payload.get("planning_latency_sec", 0.0),
        "execution_latency_sec": payload.get("execution_latency_sec", 0.0),
        "tool_count": len(payload.get("plan", {}).get("steps", [])),
    }
    row.update(leaderboard_aspect_details_from_metrics(m))
    return row


def write_run_report(out_dir: str, payload: dict) -> tuple[str, str]:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    json_path = p / "report.json"
    html_path = p / "report.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    m = payload.get("metrics", {})
    llm_usage = payload.get("llm_usage", {})

    def _fmt(v, digits: int = 4) -> str:
        if isinstance(v, bool):
            return str(v)
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)

    html = [
        "<html><head><style>",
        "body{font-family:sans-serif;margin:24px}",
        "table{border-collapse:collapse;margin-bottom:16px}",
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}",
        "th{background:#f2f2f2;text-align:left}",
        ".section{margin-top:20px}",
        "</style></head><body>",
        "<h1>GlycoAgentBench Run Report</h1>",
        f"<p><b>Model:</b> {payload.get('model_name','')} &nbsp; <b>Dataset:</b> {payload.get('dataset_id','')} &nbsp; <b>Oracle:</b> {(payload.get('oracle') or {}).get('verdict','')}</p>",
        '<div class="section"><h2>Aspect Scores</h2><table>',
    ]
    for aspect in ASPECT_NAMES:
        html.append(f"<tr><th>{aspect}</th><td>{_fmt(m.get(aspect))}</td></tr>")
    html.append("</table></div>")
    html.append('<div class="section"><h2>LLM Usage</h2><table>')
    for key in ("input_tokens", "output_tokens", "total_tokens", "call_count"):
        html.append(f"<tr><th>{key}</th><td>{_fmt(llm_usage.get(key, m.get(f'llm_{key}', 0)), 0)}</td></tr>")
    html.append("</table></div>")
    html.append('<div class="section"><h2>Aspect Details</h2><table>')
    for aspect, items in (m.get("aspect_details") or {}).items():
        if not isinstance(items, dict):
            continue
        for key, value in items.items():
            html.append(f"<tr><th>{aspect}.{key}</th><td>{_fmt(value)}</td></tr>")
    html.append("</table></div>")
    html.append('<div class="section"><h2>Raw Result Signals</h2><table>')
    for key in RAW_METRIC_COLUMNS:
        html.append(f"<tr><th>{key}</th><td>{_fmt(m.get(key,''))}</td></tr>")
    html.append("</table></div>")
    html.append('<div class="section"><h2>Full JSON</h2><pre>')
    html.append(json.dumps(payload, indent=2))
    html.append("</pre></div></body></html>")
    html_path.write_text("".join(html), encoding="utf-8")
    return str(json_path), str(html_path)


_LEADERBOARD_COLUMNS = [
    "model_name",
    "dataset_id",
    "scenario_type",
    "task_type",
    "oracle_verdict",
    *ASPECT_NAMES,
    *RAW_METRIC_COLUMNS,
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_total_tokens",
    "llm_call_count",
    "plan_valid",
    "plan_source",
    "failure_type",
    "run_status",
    "skip_reason",
    "skip_detail",
    "planning_latency_sec",
    "execution_latency_sec",
    "tool_count",
]


def write_leaderboard(out_csv: str, rows: list[dict]) -> str:
    p = Path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=_LEADERBOARD_COLUMNS)
    else:
        for c in _LEADERBOARD_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        lead = [c for c in _LEADERBOARD_COLUMNS if c in df.columns]
        extras = [c for c in df.columns if c not in lead]
        df = df[lead + extras].sort_values([c for c in ["model_name", "dataset_id"] if c in df.columns])
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return str(p)


def plot_scenario_comparison(out_png: str, leaderboard_csv: str) -> str:
    df = pd.read_csv(leaderboard_csv)
    if "run_status" in df.columns:
        done = df[df["run_status"].astype(str).str.lower() == "completed"]
        if not done.empty:
            df = done
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        plt.figure(figsize=(6, 3))
        plt.title("No benchmark results")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        return str(p)
    top_cols = ASPECT_NAMES[:]
    heat_cols = [c for c in ["de_f1", "de_average_precision", "batch_reduction_score", "bio_preservation_score"] if c in df.columns]
    for c in top_cols + heat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby("model_name", as_index=False)[top_cols + heat_cols].mean()
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(agg) * 3), 6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax1 = axes[0]
    x = range(len(agg))
    width = max(0.05, 0.75 / max(len(top_cols), 1))
    for idx, col in enumerate(top_cols):
        ax1.bar([i + (idx - (len(top_cols) - 1) / 2) * width for i in x], agg[col], width=width, label=col)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(agg["model_name"].astype(str), rotation=30, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Aspect Scores")
    ax1.legend(fontsize=7, ncol=2)
    ax2 = axes[1]
    if heat_cols:
        mat = agg.set_index("model_name")[heat_cols].T
        im = ax2.imshow(mat.values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        ax2.set_xticks(range(len(mat.columns)))
        ax2.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=8)
        ax2.set_yticks(range(len(heat_cols)))
        ax2.set_yticklabels(heat_cols, fontsize=8)
        fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.04)
    else:
        ax2.text(0.5, 0.5, "No raw metrics", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    return str(p)


def write_benchmark_breakdown(out_json: str, leaderboard_csv: str) -> str:
    df = pd.read_csv(leaderboard_csv)
    if "run_status" in df.columns:
        done = df[df["run_status"].astype(str).str.lower() == "completed"]
        if not done.empty:
            df = done
    agg_cols = [c for c in ASPECT_NAMES + RAW_METRIC_COLUMNS if c in df.columns]
    for c in agg_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out: dict = {"by_scenario_type": {}, "by_task_type": {}, "per_model_scenario_means": {}, "per_model_failure_breakdown": {}}
    if df.empty:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out_json
    if "scenario_type" in df.columns:
        out["by_scenario_type"] = df.groupby(["model_name", "scenario_type"], as_index=False)[agg_cols].mean().to_dict(orient="records")
    if "task_type" in df.columns:
        out["by_task_type"] = df.groupby(["model_name", "task_type"], as_index=False)[agg_cols].mean().to_dict(orient="records")
    if "scenario_type" in df.columns:
        out["per_model_scenario_means"] = {col: df.pivot_table(index="model_name", columns="scenario_type", values=col, aggfunc="mean").fillna(0).to_dict() for col in agg_cols}
    if "failure_type" in df.columns:
        out["per_model_failure_breakdown"] = {str(m): grp["failure_type"].value_counts().to_dict() for m, grp in df.groupby("model_name")}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_json
