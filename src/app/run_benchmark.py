from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from src.config import (
    apply_api_runtime_config,
    load_api_runtime_config,
    load_benchmark_config,
    load_model_config,
    repo_root,
    try_load_dotenv,
)
from src.reporting.reporter import (
    leaderboard_aspect_details_skipped,
    leaderboard_row_from_report_payload,
    plot_scenario_comparison,
    write_benchmark_breakdown,
    write_leaderboard,
)
from src.metrics.aspect_evaluator import ASPECT_NAMES
from src.app.run_single_model import _validate_run_dir_relative_path


def _discover_model_files(models_dir: str | Path) -> list[Path]:
    """Return runnable model YAML files, excluding the release inventory manifest."""
    return sorted(
        path
        for path in Path(models_dir).glob("*.yaml")
        if path.name != "reported_models.yaml"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", required=True)
    p.add_argument("--benchmark-config", required=True)
    p.add_argument("--api-config", default="configs/api.yaml")
    p.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="Basenames to skip, e.g. mock.yaml",
    )
    p.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Override config: max dataset dirs when using datasets_root (None = use YAML or all).",
    )
    p.add_argument(
        "--stratify-scenarios",
        action="store_true",
        help=(
            "When limiting datasets, select a round-robin mix across scenario_type values "
            "instead of taking only the first sorted dataset dirs."
        ),
    )
    p.add_argument(
        "--dataset-selection-seed",
        type=int,
        default=42,
        help="Seed for stratified dataset selection order (default: 42).",
    )
    p.add_argument(
        "--only-models",
        nargs="*",
        default=None,
        help="If set, only run these model YAML files (basenames), e.g. qwen3.5-plus.yaml gui-plus-2026-02-26.yaml",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run N single-model jobs in parallel (async in-process via thread pool). Default 1 (serial). "
            "Use 2–8 for throughput; much lighter than the old subprocess mode."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed for all runs (default: use first seed from benchmark config, or 42).",
    )
    p.add_argument(
        "--planning-mode",
        choices=["full", "iterative"],
        default="full",
        help="Pass through workflow generation mode to run_single_model.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip (model, dataset, seed, planning_mode) jobs that already have a completed "
            "report.json under runs/ (resume after failures without redoing finished cases)."
        ),
    )
    p.add_argument(
        "--checkpoint-file",
        default=None,
        metavar="PATH",
        help=(
            "Append one JSON line per finished job (ok, skipped, or resumed). "
            "Default: <reports>/benchmark_<ts>/benchmark_checkpoint.jsonl"
        ),
    )
    p.add_argument(
        "--no-job-log-stream",
        action="store_true",
        help="Do not stream subprocess stdout to the console (logs are still written to per-job files).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging for the benchmark driver.",
    )
    p.add_argument(
        "--job-retries",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If a job exits with skip_reason=network_transport, re-run the subprocess up to N "
            "additional times with exponential backoff (default 0)."
        ),
    )
    p.add_argument(
        "--job-retry-delay",
        type=float,
        default=8.0,
        metavar="SEC",
        help="Base seconds before each transport retry; doubled each attempt, capped at 120s (default 8).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Also write benchmark-driver logs to this UTF-8 plain-text file.",
    )
    return p.parse_args()


def _only_model_basenames(tokens: list[str]) -> set[str]:
    out: set[str] = set()
    for t in tokens:
        s = t.strip()
        if not s:
            continue
        out.add(s if s.endswith(".yaml") else f"{s}.yaml")
    return out


def _resolve_dataset_paths(
    bcfg,
    max_datasets_override: int | None = None,
    *,
    apply_cap: bool = True,
) -> list[Path]:
    paths: list[Path] = [Path(x) for x in bcfg.datasets]
    raw = bcfg.model_dump()
    root = raw.get("datasets_root")
    if root:
        r = Path(root)
        pat = raw.get("dataset_glob") or "gf_case_*"
        extra = sorted(p for p in r.glob(pat) if p.is_dir())
        cap = max_datasets_override
        if cap is None:
            cap = raw.get("max_datasets")
        if apply_cap and cap is not None:
            extra = extra[: int(cap)]
        paths.extend(extra)
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _dataset_task_type(dataset_dir: Path) -> str:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return "workflow_recommendation"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return "workflow_recommendation"
    return str(payload.get("task_type", "workflow_recommendation"))


def _filter_datasets_by_task_type(datasets: list[Path], allowed_task_types: list[str] | None) -> list[Path]:
    if not allowed_task_types:
        return datasets
    allowed = {str(x).strip() for x in allowed_task_types if str(x).strip()}
    if not allowed:
        return datasets
    return [d for d in datasets if _dataset_task_type(d) in allowed]


def _stratified_by_scenario(datasets: list[Path], limit: int, seed: int) -> list[Path]:
    """Round-robin dataset selection across scenario_type to avoid low-diversity heads."""
    if limit <= 0 or not datasets:
        return []
    buckets: dict[str, list[Path]] = {}
    for d in datasets:
        _, scenario, _ = _read_dataset_manifest_meta(d)
        key = scenario.strip() or "unknown"
        buckets.setdefault(key, []).append(d)
    rng = random.Random(int(seed))
    for vals in buckets.values():
        rng.shuffle(vals)
    keys = sorted(buckets.keys())
    out: list[Path] = []
    idx_by_key = {k: 0 for k in keys}
    while len(out) < limit:
        made_progress = False
        for k in keys:
            idx = idx_by_key[k]
            vals = buckets[k]
            if idx >= len(vals):
                continue
            out.append(vals[idx])
            idx_by_key[k] = idx + 1
            made_progress = True
            if len(out) >= limit:
                break
        if not made_progress:
            break
    return out


def _safe_run_dir_segment(s: str) -> str:
    """Alphanumeric/._- only; run_single_model --run-dir-name validation."""
    t = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._-")
    return (t[:100] if t else "x")


def _stable_run_dir_name(model: Path, dataset: Path, seed: int, planning_mode: str) -> str:
    """Leaf directory under runs/<model_name>/ (no model in name; parent holds model)."""
    dataset_id, _, _ = _read_dataset_manifest_meta(dataset)
    return (
        f"seed{int(seed)}__{_safe_run_dir_segment(planning_mode)}__"
        f"{_safe_run_dir_segment(dataset_id)}"
    )


def _model_runs_dir_name(model: Path) -> str:
    """Top-level folder under runs/, from YAML model ``name`` (e.g. qwen3-1.7b-base)."""
    mcfg = load_model_config(model)
    return _safe_run_dir_segment(mcfg.name)


def _benchmark_run_rel_path(model: Path, dataset: Path, seed: int, planning_mode: str) -> str:
    """Relative path under runs/: <model_name>/<seed>__<mode>__<dataset_id>."""
    return f"{_model_runs_dir_name(model)}/{_stable_run_dir_name(model, dataset, seed, planning_mode)}"


def _leaderboard_csv_name(ts: str, model_files: list[Path]) -> str:
    """leaderboard_<timestamp>_<model_slug>.csv; multi-model joins slugs with underscores."""
    if len(model_files) == 1:
        slug = _safe_run_dir_segment(load_model_config(model_files[0]).name)
        return f"leaderboard_{ts}_{slug}.csv"
    slugs = [_safe_run_dir_segment(load_model_config(m).name) for m in sorted(model_files, key=lambda p: p.name)]
    joined = "_".join(slugs)
    if len(joined) > 180:
        joined = joined[:180] + "_etc"
    return f"leaderboard_{ts}_{joined}.csv"


def _read_dataset_manifest_meta(dataset: Path) -> tuple[str, str, str]:
    """Return (dataset_id, scenario_type, task_type) for leaderboard rows."""
    manifest_path = dataset / "manifest.json"
    task_path = dataset / "task.json"
    dataset_id = dataset.name
    scenario_type, task_type = "unknown", ""
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_id = str(payload.get("dataset_id", dataset_id))
            scenario_type = str(payload.get("scenario_type") or payload.get("scenario") or "unknown")
            if not scenario_type or scenario_type.lower() == "unknown":
                gen = payload.get("generation") or {}
                params = gen.get("params") or {}
                if params:
                    nb = int(params.get("n_batches", 1) or 1)
                    mf = float(params.get("missing_fraction", 0.0) or 0.0)
                    if nb <= 1 and mf > 0.05:
                        scenario_type = "missing_only"
                    elif nb > 1 and mf <= 0.02:
                        scenario_type = "batch_only"
                    elif nb > 1 and mf > 0.05:
                        scenario_type = "batch_and_missing"
            task_type = str(payload.get("task_type", "workflow_recommendation"))
        except Exception:
            task_type = "workflow_recommendation"
    if (not scenario_type or scenario_type.lower() == "unknown") and task_path.exists():
        try:
            task_payload = json.loads(task_path.read_text(encoding="utf-8"))
            known = dict(task_payload.get("known_constraints", {}) or {})
            task_scenario = str(known.get("scenario_type", "") or "").strip()
            if task_scenario and task_scenario.lower() != "unknown":
                scenario_type = task_scenario
        except Exception:
            pass
    if not scenario_type or scenario_type.lower() == "unknown":
        # Last-resort fallback from observable files.
        abundance_csv = dataset / "abundance.csv"
        metadata_csv = dataset / "sample_metadata.csv"
        try:
            nb = 1
            mf = 0.0
            if metadata_csv.exists():
                mdf = pd.read_csv(metadata_csv)
                if "batch_label" in mdf.columns:
                    nb = int(mdf["batch_label"].nunique())
                elif "batch" in mdf.columns:
                    nb = int(mdf["batch"].nunique())
            if abundance_csv.exists():
                adf = pd.read_csv(abundance_csv)
                mf = float(adf.isna().sum().sum() / max(adf.size, 1))
            if nb <= 1 and mf > 0.05:
                scenario_type = "missing_only"
            elif nb > 1 and mf <= 0.02:
                scenario_type = "batch_only"
            elif nb > 1 and mf > 0.05:
                scenario_type = "batch_and_missing"
            else:
                scenario_type = "unknown"
        except Exception:
            scenario_type = "unknown"
    return dataset_id, scenario_type, task_type


def _backfill_row_scenario_types(rows: list[dict], datasets: list[Path]) -> None:
    """Fill missing/unknown scenario_type/task_type in leaderboard rows from dataset manifests."""
    if not rows or not datasets:
        return
    lookup: dict[str, tuple[str, str]] = {}
    for d in datasets:
        dataset_id, scenario_type, task_type = _read_dataset_manifest_meta(d)
        lookup[str(dataset_id)] = (str(scenario_type or "unknown"), str(task_type or "workflow_recommendation"))
    for row in rows:
        did = str(row.get("dataset_id", "") or "")
        if not did:
            continue
        st = str(row.get("scenario_type", "") or "")
        tt = str(row.get("task_type", "") or "")
        if did in lookup:
            l_st, l_tt = lookup[did]
            if not st or st.lower() == "unknown":
                row["scenario_type"] = l_st
            if not tt:
                row["task_type"] = l_tt


def _truncate_detail(s: str, max_len: int = 480) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _infer_skip_reason(stderr: str, stdout: str) -> tuple[str, str]:
    """Classify subprocess failure: network/TLS vs other. Returns (reason_code, short_detail)."""
    text = f"{stderr}\n{stdout}".strip()
    low = text.lower()
    transport_markers = (
        "ssl",
        "unexpected_eof",
        "eof occurred",
        "connection",
        "timeout",
        "timed out",
        "reset",
        "proxy",
        "502",
        "503",
        "504",
        "connecterror",
        "connection error",
        "network is unreachable",
        "broken pipe",
        "tls",
        "unexpected response",
        "no 'choices'",
    )
    is_transport = any(m in low for m in transport_markers)
    reason = "network_transport" if is_transport else "subprocess_error"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    detail = _truncate_detail(lines[-1] if lines else text[:240])
    if not detail:
        detail = "non-zero subprocess exit"
    return reason, detail


def _configure_logging(verbose: bool, log_file: str | None = None) -> logging.Logger:
    from src.app.log_utils import configure_colored_logging
    level = logging.DEBUG if verbose else logging.INFO
    configure_colored_logging(level)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_glycoda_file_handler", False):
            root.removeHandler(handler)
            handler.close()
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler._glycoda_file_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)
    return logging.getLogger("glycoagentbench.benchmark")


def _resume_key(model_name: str, dataset_id: str, seed: int, planning_mode: str) -> tuple[str, str, int, str]:
    return (model_name, dataset_id, int(seed), str(planning_mode))


def _build_existing_runs_index(runs_root: Path) -> dict[tuple[str, str, int, str], Path]:
    """Map (model_name, dataset_id, seed, planning_mode) -> report.json path (newest if duplicates)."""
    index: dict[tuple[str, str, int, str], Path] = {}
    if not runs_root.is_dir():
        return index
    for report_path in runs_root.glob("**/report.json"):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mn = payload.get("model_name")
        did = payload.get("dataset_id")
        if not mn or not did:
            continue
        sd = int(payload.get("seed", 42))
        pm = str(payload.get("planning_mode", "full"))
        key = _resume_key(str(mn), str(did), sd, pm)
        prev = index.get(key)
        if prev is None or report_path.stat().st_mtime > prev.stat().st_mtime:
            index[key] = report_path
    return index


def _guess_report_path_for_row(runs_root: Path, row: dict) -> str | None:
    """Find report.json for a row (nested runs/<model_name>/... or legacy flat folder)."""
    mn = row.get("model_name")
    did = row.get("dataset_id")
    if not mn or not did:
        return None
    safe_mn = _safe_run_dir_segment(str(mn))
    safe_did = _safe_run_dir_segment(str(did))
    nested = runs_root / safe_mn
    if nested.is_dir():
        cand = sorted(
            nested.glob(f"*__*__{safe_did}/report.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if cand:
            return str(cand[-1])
    cand_old = sorted(
        runs_root.glob(f"*__{mn}__{did}/report.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(cand_old[-1]) if cand_old else None


def _try_resume_row(
    model: Path,
    dataset: Path,
    seed: int,
    planning_mode: str,
    index: dict[tuple[str, str, int, str], Path],
    logger: logging.Logger,
) -> dict | None:
    """If skip-existing finds a matching report.json, return leaderboard row; else None."""
    mcfg = load_model_config(model)
    dataset_id, _, _ = _read_dataset_manifest_meta(dataset)
    key = _resume_key(mcfg.name, dataset_id, seed, planning_mode)
    rp = index.get(key)
    if rp is None:
        return None
    try:
        payload = json.loads(rp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Resume skipped: unreadable report %s (%s)", rp, exc)
        return None
    logger.info("Resume: using existing report model=%s dataset_id=%s file=%s", mcfg.name, dataset_id, rp)
    return leaderboard_row_from_report_payload(payload)


def _append_checkpoint_line(path: Path, record: dict[str, object]) -> None:
    """Append one JSON line; create file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tail_text(s: str, max_lines: int = 24) -> str:
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _run_subprocess_with_logging(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    *,
    stream_to_console: bool,
    logger: logging.Logger,
    append: bool = False,
    preamble: str | None = None,
) -> tuple[int, str]:
    """Run subprocess; merge stdout/stderr; write full log to log_path. Returns (returncode, combined_output)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8", errors="replace") as log_f:
        if preamble:
            log_f.write(preamble if preamble.endswith("\n") else preamble + "\n")
        for line in proc.stdout:
            log_f.write(line)
            chunks.append(line)
            if stream_to_console:
                logger.info("[job] %s", line.rstrip())
    rc = int(proc.wait())
    return rc, "".join(chunks)


def _git_revision() -> tuple[str | None, bool | None]:
    """(git commit hash, repo_dirty). None if not a git checkout or git missing."""
    try:
        root = repo_root()
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        commit = cp.stdout.strip() if cp.returncode == 0 else None
        st = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
        return commit, dirty
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def _skip_leaderboard_row(
    model_stem: str,
    dataset: Path,
    *,
    skip_reason: str,
    skip_detail: str,
) -> dict:
    dataset_id, scenario_type, task_type = _read_dataset_manifest_meta(dataset)
    nan = float("nan")
    return {
        "model_name": model_stem,
        "dataset_id": dataset_id,
        "scenario_type": scenario_type,
        "task_type": task_type,
        "oracle_verdict": "skipped",
        **{name: nan for name in ASPECT_NAMES},
        **leaderboard_aspect_details_skipped(),
        "batch_reduction_score": nan,
        "bio_preservation_score": nan,
        "de_recovery_score": nan,
        "de_average_precision": nan,
        "de_top10_overlap": nan,
        "de_top20_overlap": nan,
        "de_precision": nan,
        "de_recall": nan,
        "de_f1": nan,
        "de_fpr": nan,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_total_tokens": 0,
        "llm_call_count": 0,
        "plan_valid": False,
        "plan_source": "",
        "failure_type": "skipped",
        "run_status": "skipped",
        "skip_reason": skip_reason,
        "skip_detail": skip_detail,
        "planning_latency_sec": nan,
        "execution_latency_sec": nan,
        "tool_count": 0,
    }


def _run_benchmark_job_serial(
    model: Path,
    dataset: Path,
    out_root: Path,
    api_config: Path,
    runs_root: Path,
    seed: int = 42,
    planning_mode: str = "full",
    *,
    job_log_path: Path,
    stream_to_console: bool,
    logger: logging.Logger,
    job_retries: int = 0,
    job_retry_delay_sec: float = 8.0,
) -> tuple[dict | None, Path | None]:
    run_rel = _benchmark_run_rel_path(model, dataset, seed, planning_mode)
    cmd = [
        sys.executable,
        "-m",
        "src.app.run_single_model",
        "--model",
        str(model),
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_root),
        "--api-config",
        str(api_config),
        "--seed",
        str(seed),
        "--planning-mode",
        str(planning_mode),
        "--run-dir-name",
        run_rel,
    ]
    max_attempts = max(1, int(job_retries) + 1)
    last_row: dict | None = None
    last_rp: Path | None = None
    for attempt in range(max_attempts):
        preamble = (
            f"\n--- benchmark driver retry {attempt + 1}/{max_attempts} (after network_transport) ---\n"
            if attempt > 0
            else None
        )
        logger.info(
            "Starting job (serial): model=%s dataset=%s run_dir=%s log=%s attempt=%s/%s",
            model.name,
            dataset.name,
            run_rel,
            job_log_path,
            attempt + 1,
            max_attempts,
        )
        rc, combined = _run_subprocess_with_logging(
            cmd,
            repo_root(),
            job_log_path,
            stream_to_console=stream_to_console,
            logger=logger,
            append=attempt > 0,
            preamble=preamble,
        )
        if rc != 0:
            reason, detail = _infer_skip_reason(combined, combined)
            logger.warning(
                "Job failed model=%s dataset=%s rc=%s reason=%s detail=%s",
                model.stem,
                dataset.name,
                rc,
                reason,
                detail,
            )
            if combined and not stream_to_console:
                logger.warning("Last lines from log:\n%s", _tail_text(combined))
            row = _skip_leaderboard_row(model.stem, dataset, skip_reason=reason, skip_detail=detail)
            last_row, last_rp = row, None
            if reason == "network_transport" and attempt < max_attempts - 1:
                delay = min(120.0, float(job_retry_delay_sec) * (2**attempt))
                logger.warning(
                    "Transport failure; sleeping %.1fs then retrying model=%s dataset=%s",
                    delay,
                    model.stem,
                    dataset.name,
                )
                time.sleep(delay)
                continue
            return row, None
        report_path = out_root / "runs" / _validate_run_dir_relative_path(run_rel) / "report.json"
        if not report_path.exists():
            logger.warning("Missing report at %s (see %s)", report_path, job_log_path)
            return (
                _skip_leaderboard_row(
                    model.stem,
                    dataset,
                    skip_reason="missing_report",
                    skip_detail=f"expected report missing: {report_path}",
                ),
                None,
            )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        logger.info("Job finished model=%s dataset=%s report=%s", model.stem, dataset.name, report_path)
        return leaderboard_row_from_report_payload(payload), report_path
    return last_row, last_rp


def _run_benchmark_job_parallel(
    model: Path,
    dataset: Path,
    out_root: Path,
    api_config: Path,
    seed: int = 42,
    planning_mode: str = "full",
    *,
    job_log_path: Path,
    logger: logging.Logger,
    job_retries: int = 0,
    job_retry_delay_sec: float = 8.0,
) -> tuple[dict | None, Path | None]:
    run_rel = _benchmark_run_rel_path(model, dataset, seed, planning_mode)
    cmd = [
        sys.executable,
        "-m",
        "src.app.run_single_model",
        "--model",
        str(model),
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_root),
        "--api-config",
        str(api_config),
        "--seed",
        str(seed),
        "--planning-mode",
        str(planning_mode),
        "--run-dir-name",
        run_rel,
    ]
    max_attempts = max(1, int(job_retries) + 1)
    last_row: dict | None = None
    last_rp: Path | None = None
    for attempt in range(max_attempts):
        preamble = (
            f"\n--- benchmark driver retry {attempt + 1}/{max_attempts} (after network_transport) ---\n"
            if attempt > 0
            else None
        )
        logger.info(
            "Starting job (parallel): model=%s dataset=%s run_dir=%s log=%s attempt=%s/%s",
            model.name,
            dataset.name,
            run_rel,
            job_log_path,
            attempt + 1,
            max_attempts,
        )
        rc, combined = _run_subprocess_with_logging(
            cmd,
            repo_root(),
            job_log_path,
            stream_to_console=False,
            logger=logger,
            append=attempt > 0,
            preamble=preamble,
        )
        if rc != 0:
            reason, detail = _infer_skip_reason(combined, combined)
            logger.warning(
                "Job failed model=%s dataset=%s rc=%s reason=%s (log=%s)",
                model.stem,
                dataset.name,
                rc,
                reason,
                job_log_path,
            )
            logger.debug("Tail:\n%s", _tail_text(combined))
            row = _skip_leaderboard_row(model.stem, dataset, skip_reason=reason, skip_detail=detail)
            last_row, last_rp = row, None
            if reason == "network_transport" and attempt < max_attempts - 1:
                delay = min(120.0, float(job_retry_delay_sec) * (2**attempt))
                logger.warning(
                    "Transport failure; sleeping %.1fs then retrying model=%s dataset=%s",
                    delay,
                    model.stem,
                    dataset.name,
                )
                time.sleep(delay)
                continue
            return row, None
        report_path = out_root / "runs" / _validate_run_dir_relative_path(run_rel) / "report.json"
        if not report_path.exists():
            logger.warning("Missing report at %s (log=%s)", report_path, job_log_path)
            return (
                _skip_leaderboard_row(
                    model.stem,
                    dataset,
                    skip_reason="missing_report",
                    skip_detail=f"expected report missing: {report_path}",
                ),
                None,
            )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        logger.info("Job finished model=%s dataset=%s report=%s", model.stem, dataset.name, report_path)
        return leaderboard_row_from_report_payload(payload), report_path
    return last_row, last_rp


async def _run_benchmark_job_inprocess(
    model: Path,
    dataset: Path,
    out_root: Path,
    api_config: Path,
    seed: int = 42,
    planning_mode: str = "full",
    *,
    logger: logging.Logger,
    job_retries: int = 0,
    job_retry_delay_sec: float = 8.0,
) -> tuple[dict | None, Path | None]:
    """Run a benchmark job in-process using asyncio + thread pool.

    Instead of spawning a heavy subprocess, this calls
    ``run_single_model.run_single_job()`` directly in a thread,
    dramatically reducing memory overhead and enabling shared-state
    communication.
    """
    from src.app.run_single_model import run_single_job

    run_rel = _benchmark_run_rel_path(model, dataset, seed, planning_mode)
    max_attempts = max(1, int(job_retries) + 1)
    last_row: dict | None = None
    last_rp: Path | None = None
    loop = asyncio.get_event_loop()

    for attempt in range(max_attempts):
        logger.info(
            "Starting job (in-process async): model=%s dataset=%s run_dir=%s attempt=%s/%s",
            model.name,
            dataset.name,
            run_rel,
            attempt + 1,
            max_attempts,
        )
        try:
            # Run the CPU/IO-bound job in a thread pool to avoid blocking the event loop
            payload = await loop.run_in_executor(
                None,
                lambda: run_single_job(
                    model_config_path=str(model),
                    dataset_path=str(dataset),
                    out_dir=str(out_root),
                    api_config=str(api_config),
                    seed=seed,
                    planning_mode=planning_mode,
                    run_dir_name=run_rel,
                ),
            )
            report_path = out_root / "runs" / _validate_run_dir_relative_path(run_rel) / "report.json"
            logger.info("Job finished (in-process) model=%s dataset=%s", model.stem, dataset.name)
            return leaderboard_row_from_report_payload(payload), report_path
        except Exception as exc:
            detail = _truncate_detail(str(exc))
            reason, short_detail = _infer_skip_reason(str(exc), str(exc))
            logger.warning(
                "Job failed (in-process) model=%s dataset=%s attempt=%s/%s reason=%s: %s",
                model.stem,
                dataset.name,
                attempt + 1,
                max_attempts,
                reason,
                detail,
            )
            row = _skip_leaderboard_row(model.stem, dataset, skip_reason=reason, skip_detail=short_detail)
            last_row, last_rp = row, None
            if reason == "network_transport" and attempt < max_attempts - 1:
                delay = min(120.0, float(job_retry_delay_sec) * (2**attempt))
                logger.warning(
                    "Transport failure; sleeping %.1fs then retrying model=%s dataset=%s",
                    delay,
                    model.stem,
                    dataset.name,
                )
                await asyncio.sleep(delay)
                continue
            return row, None
    return last_row, last_rp


def main() -> None:
    args = parse_args()
    logger = _configure_logging(args.verbose, log_file=args.log_file)
    try_load_dotenv()
    apply_api_runtime_config(load_api_runtime_config(args.api_config))
    bcfg = load_benchmark_config(args.benchmark_config)
    model_files = _discover_model_files(args.models_dir)
    for ex in args.exclude_models:
        model_files = [m for m in model_files if m.name != ex and str(m) != ex]
    if args.only_models:
        allow = _only_model_basenames(args.only_models)
        model_files = [m for m in model_files if m.name in allow]
        if not model_files:
            logger.error("--only-models %r matched no files under %s", allow, args.models_dir)
            sys.exit(2)
    datasets = _resolve_dataset_paths(
        bcfg,
        max_datasets_override=args.max_datasets,
        apply_cap=not bool(args.stratify_scenarios),
    )
    datasets = _filter_datasets_by_task_type(datasets, bcfg.allowed_task_types)
    if args.stratify_scenarios:
        cap = int(args.max_datasets) if args.max_datasets is not None else None
        if cap is None:
            raw = bcfg.model_dump()
            cfg_cap = raw.get("max_datasets")
            cap = int(cfg_cap) if cfg_cap is not None else None
        if cap is not None:
            datasets = _stratified_by_scenario(datasets, limit=cap, seed=int(args.dataset_selection_seed))
            logger.info(
                "Dataset selection: stratified by scenario_type (limit=%s, seed=%s).",
                cap,
                args.dataset_selection_seed,
            )
        else:
            logger.info("--stratify-scenarios ignored because no dataset cap is set.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(bcfg.output_dir)
    runs_root = out_root / "runs"
    reports_root = out_root / "reports" / f"benchmark_{ts}"
    job_logs_root = reports_root / "job_logs"
    boards_root = out_root / "leaderboards"
    runs_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    job_logs_root.mkdir(parents=True, exist_ok=True)
    boards_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else reports_root / "benchmark_checkpoint.jsonl"
    seed = args.seed if args.seed is not None else (bcfg.seeds[0] if bcfg.seeds else 42)
    resume_index = _build_existing_runs_index(runs_root) if args.skip_existing else {}
    if args.skip_existing:
        logger.info("Skip-existing enabled: loaded %s prior report(s) from %s", len(resume_index), runs_root)

    jobs = [(m, d) for m in model_files for d in datasets]
    workers = max(1, int(args.workers))
    stream_sub = (not args.no_job_log_stream) and workers == 1
    rows: list[dict] = []
    n_resumed = 0

    logger.info(
        "Benchmark grid: %s model(s) x %s dataset(s) = %s job(s); workers=%s; planning_mode=%s; seed=%s; job_logs=%s",
        len(model_files),
        len(datasets),
        len(jobs),
        workers,
        args.planning_mode,
        seed,
        job_logs_root,
    )
    if workers > 1:
        logger.info(
            "Async in-process mode: jobs run in thread pool (no subprocess overhead); logs written per-job under run dirs.",
        )

    def _checkpoint(
        status: str,
        model: Path,
        dataset: Path,
        *,
        report_path: str | None = None,
        log_path: str | None = None,
        skip_reason: str | None = None,
        skip_detail: str | None = None,
    ) -> None:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "model_yaml": str(model),
            "dataset_dir": str(dataset.resolve()),
            "seed": seed,
            "planning_mode": args.planning_mode,
            "report_path": report_path,
            "log_path": log_path,
            "skip_reason": skip_reason,
            "skip_detail": (skip_detail[:500] + "…") if skip_detail and len(skip_detail) > 500 else skip_detail,
        }
        _append_checkpoint_line(checkpoint_path, rec)

    if workers == 1:
        for model, dataset in jobs:
            job_log = job_logs_root / f"{model.stem}__{dataset.name}.log"
            if args.skip_existing:
                resumed = _try_resume_row(model, dataset, seed, args.planning_mode, resume_index, logger)
                if resumed is not None:
                    rows.append(resumed)
                    n_resumed += 1
                    mcfg = load_model_config(model)
                    did_m, _, _ = _read_dataset_manifest_meta(dataset)
                    rp = resume_index.get(_resume_key(mcfg.name, did_m, seed, args.planning_mode))
                    _checkpoint("resumed", model, dataset, report_path=str(rp) if rp else None)
                    continue
            row, report_path_done = _run_benchmark_job_serial(
                model,
                dataset,
                out_root,
                Path(args.api_config),
                runs_root,
                seed=seed,
                planning_mode=args.planning_mode,
                job_log_path=job_log,
                stream_to_console=stream_sub,
                logger=logger,
                job_retries=int(args.job_retries),
                job_retry_delay_sec=float(args.job_retry_delay),
            )
            if row is not None:
                rows.append(row)
                st = str(row.get("run_status", "completed")).lower()
                if st == "skipped":
                    _checkpoint(
                        "skipped",
                        model,
                        dataset,
                        log_path=str(job_log),
                        skip_reason=str(row.get("skip_reason", "")),
                        skip_detail=str(row.get("skip_detail", "")),
                    )
                else:
                    _checkpoint(
                        "completed",
                        model,
                        dataset,
                        report_path=str(report_path_done) if report_path_done else None,
                        log_path=str(job_log),
                    )
    else:
        # ---- Async in-process parallel mode ----
        # Uses asyncio + thread pool instead of subprocess for each job.
        # This dramatically reduces memory overhead (no separate Python process per job)
        # and simplifies inter-job communication.

        async def _async_parallel_job(m: Path, d: Path, sem: asyncio.Semaphore) -> dict:
            async with sem:
                if args.skip_existing:
                    # Resume check is fast (file I/O only), run in executor
                    loop = asyncio.get_event_loop()
                    resumed = await loop.run_in_executor(
                        None,
                        lambda: _try_resume_row(m, d, seed, args.planning_mode, resume_index, logger),
                    )
                    if resumed is not None:
                        return {"kind": "resumed", "row": resumed, "model": m, "dataset": d}
                try:
                    prow, prpath = await _run_benchmark_job_inprocess(
                        m,
                        d,
                        out_root,
                        Path(args.api_config),
                        seed=seed,
                        planning_mode=args.planning_mode,
                        logger=logger,
                        job_retries=int(args.job_retries),
                        job_retry_delay_sec=float(args.job_retry_delay),
                    )
                    return {"kind": "ran", "row": prow, "model": m, "dataset": d, "report_path": prpath}
                except Exception as exc:  # noqa: BLE001
                    detail = _truncate_detail(str(exc))
                    logger.exception("Async job error model=%s dataset=%s: %s", m.stem, d.name, exc)
                    return {
                        "kind": "error",
                        "row": _skip_leaderboard_row(m.stem, d, skip_reason="async_error", skip_detail=detail),
                        "model": m,
                        "dataset": d,
                    }

        async def _run_all_parallel():
            sem = asyncio.Semaphore(workers)
            tasks = [_async_parallel_job(m, d, sem) for m, d in jobs]
            return await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            "Async in-process parallel mode: %s concurrent workers (no subprocess overhead).",
            workers,
        )
        results = asyncio.run(_run_all_parallel())

        for payload in results:
            m, d = payload["model"], payload["dataset"]
            if payload.get("kind") == "resumed":
                rows.append(payload["row"])
                n_resumed += 1
                mcfg = load_model_config(m)
                did_m, _, _ = _read_dataset_manifest_meta(d)
                rp = resume_index.get(_resume_key(mcfg.name, did_m, seed, args.planning_mode))
                _checkpoint("resumed", m, d, report_path=str(rp) if rp else None)
                continue
            if payload.get("kind") == "error":
                row = payload.get("row")
                if row is not None:
                    rows.append(row)
                _checkpoint(
                    "skipped", m, d,
                    skip_reason=str(row.get("skip_reason", "") if row else "async_error"),
                    skip_detail=str(row.get("skip_detail", "") if row else ""),
                )
                continue
            row = payload.get("row")
            if row is not None:
                rows.append(row)
                st = str(row.get("run_status", "completed")).lower()
                if st == "skipped":
                    _checkpoint(
                        "skipped", m, d,
                        skip_reason=str(row.get("skip_reason", "")),
                        skip_detail=str(row.get("skip_detail", "")),
                    )
                else:
                    rp_done = payload.get("report_path")
                    if isinstance(rp_done, Path):
                        rp_done = str(rp_done)
                    elif not rp_done and row:
                        rp_done = _guess_report_path_for_row(runs_root, row)
                    _checkpoint(
                        "completed", m, d,
                        report_path=rp_done,
                    )

    _backfill_row_scenario_types(rows, datasets)

    leaderboard_path = boards_root / _leaderboard_csv_name(ts, model_files)
    leaderboard_csv = write_leaderboard(str(leaderboard_path), rows)

    latest_link = boards_root / "leaderboard.csv"
    shutil.copy2(leaderboard_path, latest_link)
    plot_scenario_comparison(str(reports_root / "scenario_comparison.png"), leaderboard_csv)
    write_benchmark_breakdown(str(reports_root / "benchmark_breakdown.json"), leaderboard_csv)
    df = pd.read_csv(leaderboard_csv)
    git_commit, git_dirty = _git_revision()
    reproducibility: dict = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "benchmark_config": str(Path(args.benchmark_config).resolve()),
        "api_config": str(Path(args.api_config).resolve()),
        "models_dir": str(Path(args.models_dir).resolve()),
        "seed": int(seed),
        "workers": int(workers),
        "job_retries": int(args.job_retries),
        "job_retry_delay_sec": float(args.job_retry_delay),
        "stratify_scenarios": bool(args.stratify_scenarios),
        "dataset_selection_seed": int(args.dataset_selection_seed),
        "planning_mode": str(args.planning_mode),
        "skip_existing": bool(args.skip_existing),
        "checkpoint_file": str(checkpoint_path.resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    if "run_status" in df.columns:
        _rs = df["run_status"].astype(str).str.lower()
        df_done = df[_rs == "completed"].copy()
        n_skipped = int((_rs == "skipped").sum())
    else:
        df_done = df
        n_skipped = 0
    n_completed = int(len(df_done))
    if n_skipped > 0 and "skip_reason" in df.columns:
        sk = df[df["run_status"].astype(str).str.lower() == "skipped"]
        (reports_root / "skipped_runs.json").write_text(
            json.dumps(
                sk[["model_name", "dataset_id", "scenario_type", "skip_reason", "skip_detail"]].to_dict(
                    orient="records"
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    score_cols = ASPECT_NAMES[:]
    sub_metric_cols = [
        "batch_reduction_score", "bio_preservation_score", "de_recovery_score",
        "de_average_precision", "de_top10_overlap", "de_top20_overlap",
        "de_precision", "de_recall", "de_f1", "de_fpr",
        "f_stat_corr_preservation", "corr_structure_preservation",
        "observed_batch_signal_score",
        "llm_input_tokens", "llm_output_tokens", "llm_total_tokens", "llm_call_count",
    ]
    scenario_summary_cols = score_cols + ["batch_reduction_score", "bio_preservation_score", "de_f1"]
    all_numeric_cols = score_cols + sub_metric_cols

    def _safe_mean(frame: pd.DataFrame, col: str) -> float:
        if col not in frame.columns:
            return 0.0
        s = pd.to_numeric(frame[col], errors="coerce")
        return float(s.mean()) if not s.isna().all() else 0.0

    summary: dict = {
        "reproducibility": reproducibility,
        "metric_notes": {
            "batch_reduction_score": (
                "Statistical proxy: reduction in batch-stratified variance between raw and processed matrices; "
                "not a clinical or wet-lab batch-effect gold standard."
            ),
            "bio_preservation_score": (
                "Statistical proxy: blend of group-variance ratio stability and correlation-structure similarity; "
                "not pathway- or biology-level validation."
            ),
            "aspect_scores": (
                "11 discrete aspect scores in {1, 0.5, 0}. Each aspect uses A-level gate items and a bucketed "
                "weighted mean of B/C items."
            ),
        },
        "n_runs_total": int(len(df)),
        "n_runs_completed": n_completed,
        "n_runs_skipped": n_skipped,
        "n_runs_resumed_from_disk": int(n_resumed),
        "oracle_pass_rate": float((df_done["oracle_verdict"] == "correct").mean()) if not df_done.empty else 0.0,
        "invalid_workflow_rate": float((~df_done["plan_valid"].astype(bool)).mean()) if not df_done.empty else 0.0,
        "tool_diversity": int(df_done["tool_count"].nunique()) if not df_done.empty else 0,
        "avg_planning_latency_sec": float(df_done["planning_latency_sec"].mean()) if not df_done.empty else 0.0,
        "avg_execution_latency_sec": float(df_done["execution_latency_sec"].mean()) if not df_done.empty else 0.0,
    }
    if n_skipped > 0 and "skip_reason" in df.columns:
        sk_only = df[df["run_status"].astype(str).str.lower() == "skipped"]
        summary["skip_reason_breakdown"] = {
            str(k): int(v) for k, v in sk_only["skip_reason"].value_counts().to_dict().items()
        }
    if not df_done.empty:
        for col in all_numeric_cols:
            summary[f"mean_{col}"] = _safe_mean(df_done, col)

    if not df.empty and "model_name" in df.columns:
        per_model: dict = {}
        for model_name, grp in df.groupby("model_name"):
            gdone = (
                grp[grp["run_status"].astype(str).str.lower() == "completed"]
                if "run_status" in grp.columns
                else grp
            )
            n_tot = int(len(grp))
            n_done = int(len(gdone))
            model_stats: dict = {
                "n_runs_total": n_tot,
                "n_runs_completed": n_done,
                "n_runs_skipped": n_tot - n_done,
                "oracle_correct_rate": float((gdone["oracle_verdict"] == "correct").mean()) if not gdone.empty else 0.0,
                "oracle_mostly_correct_rate": float((gdone["oracle_verdict"] == "mostly_correct").mean())
                if not gdone.empty
                else 0.0,
                "invalid_plan_rate": float((~gdone["plan_valid"].astype(bool)).mean()) if not gdone.empty else 0.0,
                "avg_planning_latency_sec": float(gdone["planning_latency_sec"].mean()) if not gdone.empty else 0.0,
            }
            for col in all_numeric_cols:
                model_stats[f"mean_{col}"] = _safe_mean(gdone, col)
            if "failure_type" in grp.columns:
                fc = grp["failure_type"].value_counts().to_dict()
                model_stats["failure_breakdown"] = {str(k): int(v) for k, v in fc.items()}
            if "scenario_type" in gdone.columns and not gdone.empty:
                scen_cols = [c for c in scenario_summary_cols if c in gdone.columns]
                model_stats["by_scenario"] = (
                    gdone.groupby("scenario_type", as_index=False)[scen_cols].mean().to_dict(orient="records")
                )
            per_model[str(model_name)] = model_stats
        summary["per_model"] = per_model

    if not df.empty and "failure_type" in df.columns:
        fc_all = df["failure_type"].value_counts().to_dict()
        summary["failure_breakdown"] = {str(k): int(v) for k, v in fc_all.items()}

    if not df_done.empty and "scenario_type" in df_done.columns:
        st_cols = [c for c in scenario_summary_cols if c in df_done.columns]
        summary["by_scenario_type"] = (
            df_done.groupby("scenario_type", as_index=False)[st_cols].mean().to_dict(orient="records")
        )
    if not df_done.empty and "task_type" in df_done.columns:
        tt_cols = [c for c in scenario_summary_cols if c in df_done.columns]
        summary["by_task_type"] = (
            df_done.groupby("task_type", as_index=False)[tt_cols].mean().to_dict(orient="records")
        )
    (reports_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Benchmark done: leaderboard=%s checkpoint=%s resumed=%s completed_rows=%s skipped=%s",
        leaderboard_csv,
        checkpoint_path,
        n_resumed,
        n_completed,
        n_skipped,
    )


if __name__ == "__main__":
    main()
