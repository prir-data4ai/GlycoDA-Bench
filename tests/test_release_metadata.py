from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import yaml


REQUIRED_FILES = {
    "README.md",
    "SECURITY.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "VERSION",
    "docs/METRICS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/DATA_AND_CODE_AVAILABILITY.md",
    "docs/WORKFLOW_GROUND_TRUTH.md",
    "docs/V1_TO_V3.md",
    ".github/workflows/tests.yml",
}
METRICS = {
    "Core Workflow Correctness",
    "Case Awareness",
    "Parameter Soundness",
    "Workflow Economy",
    "Diagnostic Reasoning",
    "Action Selection",
    "Audit Closure",
    "Differential Set Recovery",
    "Differential Ranking Quality",
    "Batch Removal Quality",
    "Biological Signal Preservation",
}


def test_required_release_metadata_exists() -> None:
    missing = sorted(path for path in REQUIRED_FILES if not Path(path).is_file())
    assert missing == []


def test_readme_and_availability_name_public_artifacts() -> None:
    text = (Path("README.md").read_text(encoding="utf-8") + "\n" + Path("docs/DATA_AND_CODE_AVAILABILITY.md").read_text(encoding="utf-8"))

    assert "https://huggingface.co/datasets/xiaokui17/GlycoDA-Bench" in text
    assert "https://huggingface.co/datasets/xiaokui17/glycoforge_benchmark_648" in text
    assert "https://github.com/evolan17/GlycoDA-Bench" in text
    assert "participant-facing" in text
    assert "evaluator" in text.lower()
    assert "hidden truth" in text.lower()


def test_release_metadata_separates_manuscript_v1_results_from_v2_extension() -> None:
    text = (Path("README.md").read_text(encoding="utf-8") + "\n" + Path("docs/DATA_AND_CODE_AVAILABILITY.md").read_text(encoding="utf-8")).lower()

    assert "manuscript reproduction release" in text
    assert "v2 paired/coupled extension" in text
    assert "no manuscript model scores" in text


def test_release_uses_apache_license_and_attributes_glycoforge() -> None:
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert citation["license"] == "Apache-2.0"
    assert citation["repository-code"] == "https://github.com/evolan17/GlycoDA-Bench"
    assert "https://github.com/BojarLab/GlycoForge" in notices
    assert "v0.2.0" in notices
    assert "3e3ed302b2e3aaef1bc47c407c3528939c482a54" in notices


def test_availability_assigns_every_reviewer_artifact_to_a_release_destination() -> None:
    text = Path("docs/DATA_AND_CODE_AVAILABILITY.md").read_text(encoding="utf-8")

    for phrase in (
        "`participant`",
        "`evaluator_truth`",
        "`results`",
        "all 648",
        "representative execution traces",
        "GlycoForge 0.2.0",
        "3e3ed302b2e3aaef1bc47c407c3528939c482a54",
        "evaluation timestamps",
    ):
        assert phrase in text


def test_metrics_document_covers_all_eleven_metrics_and_edge_cases() -> None:
    text = Path("docs/METRICS.md").read_text(encoding="utf-8")

    assert METRICS <= {line.lstrip("# ").strip() for line in text.splitlines()}
    assert "NDCG@20" in text
    assert "0.3" in text
    assert "zero denominator" in text.lower()
    assert "{0, 0.5, 1}" in text


def test_reproducibility_documents_agent_protocol() -> None:
    text = Path("docs/REPRODUCIBILITY.md").read_text(encoding="utf-8").lower()

    for phrase in ("full", "iterative", "12", "stopping", "failed tool", "plan revision", "temperature"):
        assert phrase in text


def test_security_and_citation_metadata_are_parseable() -> None:
    security = Path("SECURITY.md").read_text(encoding="utf-8").lower()
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))

    assert "rotate" in security
    assert "environment variable" in security
    assert citation["title"].startswith("GlycoDA-Bench")
    assert len(citation["authors"]) == 11


def test_package_metadata_matches_the_release_and_exposes_the_benchmark_cli() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["version"] == Path("VERSION").read_text(encoding="utf-8").strip()
    assert project["requires-python"] == ">=3.10"
    assert project["scripts"]["glyco-bench"] == "src.app.run_benchmark:main"

