from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any


PUBLIC_FIELDS = (
    "dataset_id",
    "scenario_type",
    "task_type",
    "model_name",
    "seed",
    "planning_mode",
    "plan_valid",
    "plan_source",
    "planner_failure_reason",
    "plan",
    "execution_trace",
    "decision_log",
    "metrics",
    "oracle",
    "guard_issues",
    "top_failures",
)
SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "headers",
    "base_url",
    "endpoint",
    "credential",
    "credentials",
    "secret",
    "access_token",
    "refresh_token",
    "raw_provider_message",
    "raw_response",
    "provider_response",
}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
UNIX_ABSOLUTE = re.compile(r"^/(?:home|Users|private|tmp|var|mnt|opt)/")


def _artifact_uri(value: str) -> str:
    if WINDOWS_ABSOLUTE.match(value):
        name = PureWindowsPath(value).name
    else:
        name = Path(value).name
    return f"artifact://{name or 'redacted'}"


def sanitize_for_publication(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize_for_publication(item)
            for key, item in value.items()
            if str(key).lower() not in SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [sanitize_for_publication(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_publication(item) for item in value]
    if isinstance(value, str) and (WINDOWS_ABSOLUTE.match(value) or UNIX_ABSOLUTE.match(value)):
        return _artifact_uri(value)
    return value


def export_public_trace(source: Path, output: Path) -> Path:
    source = Path(source)
    output = Path(output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    public = {key: payload[key] for key in PUBLIC_FIELDS if key in payload}
    public["publication_note"] = (
        "Sanitized representative trace. Raw provider messages, credentials, endpoints, "
        "and local absolute paths are intentionally excluded."
    )
    sanitized = sanitize_for_publication(public)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "trace"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export sanitized representative benchmark traces.")
    parser.add_argument("--source", action="append", type=Path, required=True, help="Path to a report.json; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/traces"))
    args = parser.parse_args()
    for source in args.source:
        payload = json.loads(source.read_text(encoding="utf-8"))
        name = _slug(f"{payload.get('dataset_id', source.parent.name)}__{payload.get('model_name', 'model')}")
        output = args.output_dir / f"{name}.json"
        export_public_trace(source, output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
