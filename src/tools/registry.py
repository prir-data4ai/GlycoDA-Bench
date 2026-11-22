from __future__ import annotations

from typing import Any

from src.tools import impl

BATCH_TOOLS: frozenset[str] = frozenset({
    "run_combat",
    "run_ratio_combat",
    "run_percentile_norm",
    "run_limma_batch_removal",
    "run_harmony",
})

TOOL_REGISTRY = {
    "inspect_dataset": impl.inspect_dataset,
    "clr_transform": impl.clr_transform,
    "impute_missing_values": impl.impute_missing_values,
    "check_batch_effect": impl.check_batch_effect,
    "run_combat": impl.run_combat,
    "run_ratio_combat": impl.run_ratio_combat,
    "run_percentile_norm": impl.run_percentile_norm,
    "run_limma_batch_removal": impl.run_limma_batch_removal,
    "run_harmony": impl.run_harmony,
    "run_differential_expression": impl.run_differential_expression,
    "run_motif_enrichment": impl.run_motif_enrichment,
    "run_pca": impl.run_pca,
}

# OpenAI-style tool schema snippets for manifest.available_tools and planner injection.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "inspect_dataset",
        "description": "Summarize matrix shape, missingness, compositionality hints, and metadata availability.",
        "required_inputs": ["input_csv (abundance or clr matrix)", "metadata_csv"],
        "key_params": [],
    },
    {
        "name": "impute_missing_values",
        "description": "Impute missing values before CLR/DE when missingness is present.",
        "required_inputs": ["input_csv"],
        "key_params": ["strategy: median | left_censored_small_value"],
    },
    {
        "name": "clr_transform",
        "description": (
            "CLR for compositional glycomics. GlycoForge/benchmark matrices use glycans as rows and samples as "
            "columns (first column glycan_index); set samples_are_rows=false unless your CSV is transposed."
        ),
        "required_inputs": ["input_csv"],
        "key_params": ["pseudocount", "samples_are_rows"],
    },
    {
        "name": "check_batch_effect",
        "description": "Proxy metrics for batch structure vs biological groups.",
        "required_inputs": ["input_csv", "metadata_csv"],
        "key_params": [],
    },
    {
        "name": "run_combat",
        "description": "Empirical-Bayes ComBat (pycombat); preserves biological covariate when use_bio_covariate=true.",
        "required_inputs": ["input_csv", "metadata_csv"],
        "key_params": ["use_bio_covariate"],
    },
    {
        "name": "run_ratio_combat",
        "description": "Ratio-based batch adjustment (simplified proxy).",
        "required_inputs": ["input_csv", "metadata_csv"],
        "key_params": [],
    },
    {
        "name": "run_percentile_norm",
        "description": "Per-batch percentile normalization (simplified proxy).",
        "required_inputs": ["input_csv", "metadata_csv"],
        "key_params": [],
    },
    {
        "name": "run_limma_batch_removal",
        "description": "Per-feature OLS y~batch+group; remove batch fitted values (statsmodels).",
        "required_inputs": ["input_csv", "metadata_csv"],
        "key_params": [],
    },
    {
        "name": "run_harmony",
        "description": "Harmony on PCA embedding (harmonypy if installed); else ComBat in PC space + inverse PCA.",
        "required_inputs": ["input_csv", "metadata_csv"],
        "key_params": [],
    },
    {
        "name": "run_differential_expression",
        "description": "Welch t-test per feature; BH-FDR (padj) by default; significant uses padj<alpha.",
        "required_inputs": ["input_csv", "metadata_csv"],
        "key_params": ["alpha", "group_col", "input_space", "multiple_testing: bh | none"],
    },
    {
        "name": "run_motif_enrichment",
        "description": "Fisher ORA + BH-FDR over feature tags (feature_metadata.csv; motif_tag or effect_role).",
        "required_inputs": ["input_csv (DE table)", "feature_metadata_csv optional"],
        "key_params": ["tag_column"],
    },
    {
        "name": "run_pca",
        "description": "sklearn PCA explained variance for top components.",
        "required_inputs": ["input_csv"],
        "key_params": ["n_components"],
    },
]


def list_tool_names() -> list[str]:
    return list(TOOL_REGISTRY.keys())


def export_tool_specs_for_manifest() -> list[dict[str, Any]]:
    """Return serializable tool specs for manifest.available_tools."""
    return [dict(s) for s in TOOL_SPECS]
