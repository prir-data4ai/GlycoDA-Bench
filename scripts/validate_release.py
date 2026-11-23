from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


FORBIDDEN_ANY_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
}
FORBIDDEN_ROOTS = {
    "data",
    "outputs",
    "notebooks",
    "glycoforge_benchmark_648",
    "_private_hidden_truth",
}
CONTENT_SCAN_EXCLUSIONS = {
    "tests/test_release_hygiene.py",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PLACEHOLDER_MARKERS = (
    "${",
    "getenv(",
    "os.environ",
    "<redacted>",
    "<placeholder>",
    "changeme",
    "your_",
)

WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\")
UNIX_HOME_RE = re.compile(r"/(?:home|Users)/[^/\s]+/")
INTERNAL_HOST_RE = re.compile(
    r"(?:https?://)?(?:"
    r"10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
    r"[^\s/'\"]+\.(?:polaris|woa\.com)(?=[:/\s'\"]|$)"
    r")",
    flags=re.IGNORECASE,
)
CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?im)^\s*(?:api[_-]?key|auth[_-]?token|access[_-]?token|"
    r"ss_auth_token|authorization)\s*[:=]\s*['\"]?([^\s#'\"]+)"
)
PYTHON_CREDENTIAL_RE = re.compile(
    r"(?im)^\s*(?:api[_-]?key|auth[_-]?token|access[_-]?token|"
    r"ss_auth_token|authorization)\s*=\s*['\"]([^'\"]+)['\"]"
)
BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._:-]{12,}")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_placeholder(value: str, text: str) -> bool:
    lower = f"{value}\n{text}".lower()
    return value.lower() in {"", "empty", "none", "null"} or any(marker.lower() in lower for marker in PLACEHOLDER_MARKERS)


def _candidate_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=False,
            capture_output=True,
        )
        if proc.returncode == 0:
            return [root / item.decode("utf-8") for item in proc.stdout.split(b"\0") if item]
    return [path for path in root.rglob("*") if path.is_file()]


def validate_repository(root: Path) -> list[str]:
    root = Path(root).resolve()
    findings: list[str] = []
    for path in sorted(_candidate_files(root)):
        # ``git ls-files`` includes paths staged or deleted only in the working tree.
        # A release candidate should validate the files that will actually be present.
        if not path.is_file():
            continue
        rel = _relative(path, root)
        rel_parts = path.relative_to(root).parts
        parts = set(rel_parts)
        if ".git" in parts or ".superpowers" in parts:
            continue
        if rel.startswith("docs/superpowers/"):
            continue
        if (
            (rel_parts and rel_parts[0] in FORBIDDEN_ROOTS)
            or any(part in FORBIDDEN_ANY_PARTS or part.startswith("tmp_") for part in parts)
        ):
            findings.append(f"forbidden path: {rel}")
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".log"}:
            findings.append(f"forbidden file type: {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if rel in CONTENT_SCAN_EXCLUSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 text file: {rel}")
            continue
        if WINDOWS_ABSOLUTE_RE.search(text):
            findings.append(f"absolute Windows path: {rel}")
        if UNIX_HOME_RE.search(text):
            findings.append(f"absolute user path: {rel}")
        if INTERNAL_HOST_RE.search(text):
            findings.append(f"internal host: {rel}")
        credential_re = PYTHON_CREDENTIAL_RE if path.suffix.lower() == ".py" else CREDENTIAL_ASSIGNMENT_RE
        for match in credential_re.finditer(text):
            value = match.group(1).strip()
            if not _is_placeholder(value, match.group(0)):
                findings.append(f"credential-like value: {rel}")
                break
        if BEARER_RE.search(text) and "${" not in text and "getenv(" not in text:
            findings.append(f"credential-like bearer value: {rel}")
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a public GlycoAgentBench release tree.")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    findings = validate_repository(Path(args.root))
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}")
        print(f"Release validation failed with {len(findings)} finding(s).")
        return 1
    print("Release validation passed: no forbidden files or secret-like values found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

