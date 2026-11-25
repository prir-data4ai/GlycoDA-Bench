# Data and Code Availability

## Reviewer-required release map

The public release uses GitHub for executable, reviewable source material and Hugging
Face for versioned data and large tabular artifacts. The destinations below are part of
the release contract rather than optional examples, and the manuscript release covers
all 648 reported benchmark cases.

| Artifact requested during review | Public destination | Release rule |
| --- | --- | --- |
| All 648 agent-visible manuscript cases, including abundance matrices, sample/feature metadata, task cards, case manifests, and `scenario_index.csv` | Hugging Face manuscript reproduction release `prirdata4ai/GlycoDA-Bench`, component `participant` | Public and versioned; exactly 648 unique case IDs and exact hashes associated with the reported results. |
| Differential-feature truth, importance ordering, workflow requirements, motif truth, and clean/oracle references used only by scoring | Manuscript release component `evaluator_truth` | Public at publication for reproducibility, but never mounted into or named in an evaluated agent's context. |
| Per-case scores for all 22 systems, failure/status fields, and exact aggregation inputs | Manuscript release component `results/manuscript_v1` | Public under the immutable `manuscript-v1.0.0` tag and explicitly labeled `legacy-v1-report`; raw provider text may be redacted if redistribution terms require it. |
| Realized two-block paired/coupled cases | Hugging Face v2 paired/coupled extension `xiaokui17/GlycoDA-Bench` | Public supplementary release; no manuscript model scores are attributed to v2. |
| Agent implementation, system/user prompts, structured schemas, tool descriptions and implementations, stopping/failure policy, metric implementations, rescoring scripts, tests, and model configuration templates | GitHub source repository | Public under the code license and identified by a Git tag/commit. |
| Complete 22-system inventory, model/API identifiers, decoding settings, evaluation timestamps, source result filenames, run count, and failure policy | GitHub (`configs/models/reported_models.yaml` and `docs/REPRODUCIBILITY.md`) | Public; values must match manuscript tables and captions. |
| Workflow-ground-truth derivation, required/optional/unnecessary action criteria, alternative valid workflows, and automatic scoring provenance | GitHub (`docs/WORKFLOW_GROUND_TRUTH.md`, `docs/METRICS.md`, and evaluator source) plus the Hugging Face `evaluator_truth` schema | Public and versioned; evaluated agents receive none of the truth values. |
| Difficulty definitions, empirical performance by difficulty/task family/signal condition, and synthetic-data representativeness/limitations | Hugging Face dataset card and `results` configuration, with the same summaries in the manuscript or supplement | Public, stratified, and explicit about requested versus actually realized scenarios. |
| Exact process-metric table and representative execution traces illustrating successful and failed evidence-judgment-action chains | GitHub release assets or manuscript supplement | Publish compact tables and sanitized representative execution traces; do not expose credentials or provider-restricted reasoning text. |
| Benchmark construction script, scenario grid, generator pin, random seeds, validation command, and checksums | Hugging Face dataset repository plus this reproducibility documentation | Public and versioned with the data release. |

## Participant-facing data and version mapping

The exact 648 cases associated with the manuscript results are the **manuscript
reproduction release** at:

<https://huggingface.co/datasets/prirdata4ai/GlycoDA-Bench/tree/manuscript-v1.0.0>

The supplementary **v2 paired/coupled extension** is available at:

<https://huggingface.co/datasets/xiaokui17/GlycoDA-Bench/tree/v2.0.0>

The tags resolve to commits `987e7b792c36eefbb7b6cff11e04f685e720b53d` and
`a582af86c021a39a65fa6fedb39437803449dd68`, respectively.

Each dataset card and release manifest identifies the `participant` files: observed
matrices, sample and feature metadata, task descriptions, and the structural manifest
required by the runner. Completeness means no benchmark case is omitted; it does not
authorize use of evaluator truth or controlled generation states. The manuscript
results must not be relabeled as v2 results because the two releases have different
case hashes.

## Evaluator truth

Ground-truth differential features, importance ordering, workflow requirements, and
motif follow-up truth are released as the documented `evaluator_truth` component.
During an evaluation they are made available only to the scorer through
`GLYCOAGENT_HIDDEN_TRUTH_ROOT`. Public archival does not authorize agent access: the
truth files remain separate from participant inputs and carry a no-agent-access warning.

## Code

The public experiment implementation is intended for:

<https://github.com/prir-data4ai/GlycoDA-Bench>

It contains the prompts, agent scaffold, tools, failure policy, eleven metric
implementations, model metadata, rescore scripts, and tests. Large per-case tables are
released under `results/manuscript_v1` in the manuscript reproduction dataset. The code
repository additionally provides compact per-system and stratified metric tables and
representative execution traces requested by the reviewers.
Provider-restricted raw text can be replaced by a sanitized trace that preserves the
observed tool sequence, intermediate observations, judgments, actions, and failure
labels.

The eleven-metric evaluator is the `aspect-v3-eleven-metric` contract for new or
explicitly rescored runs. The unchanged 22-system archive remains
`legacy-v1-report`; `docs/RESULT_SCHEMAS.md` defines the boundary.

## Generator provenance

GlycoForge is available under the MIT License at
<https://github.com/BojarLab/GlycoForge>. It is another authors' independently
maintained project and is therefore referenced rather than republished here. The
benchmark manifests report **GlycoForge 0.2.0** and pin exact generation commit
`3e3ed302b2e3aaef1bc47c407c3528939c482a54`. Upstream's `v0.2.0` tag points to
`a75b7b7b23d30024a10e87a47da40abc5e04c344`; the manifest commit is the exact build
provenance. The benchmark-specific construction script accompanies the Hugging Face
data release; the manuscript-v1 archived source has
SHA-256 `da4399bc79b52f5520edcd8a0f7355662980008d0b10dd7f4e7273283b1445f4`.
The released `scenario_index.csv` has SHA-256
`144af14a8f67a39dfc72277f9bdfeb97046013912c8ac6bd710ec3fc7c55799e`.
These pins must be accompanied by the scenario grid, per-case seeds, validation command,
and file checksums; the core GlycoForge repository alone is not a substitute for the
benchmark-specific build recipe.

The 648 case directories were materialized specifically for GlycoDA-Bench rather than
selected as a ready-made GlycoForge benchmark. The public index must preserve both the
requested and actual construction fields (`requested_mode`, `actual_mode`,
`paired_requested`, and `paired_realized`) so that proxy cases are not described as
realized paired datasets.

## Suggested manuscript statement

> The complete GlycoDA-Bench manuscript reproduction release, comprising 648 benchmark
> cases, evaluator truth, and 22-system per-case results, is available from the
> versioned Hugging Face repository. Evaluator files were not accessible to evaluated
> agents. A separate v2 paired/coupled extension supplies realized two-block cases; no
> manuscript model scores are attributed to v2. The agent scaffold, prompts, tool
> implementations, evaluation metrics, model configuration templates, and scripts used
> to reproduce the reported analyses are available from the versioned code repository.
> GlycoForge is the underlying data-generation framework, whereas GlycoDA-Bench is the
> benchmark suite constructed from a pinned GlycoForge version and benchmark-specific
> scenario definitions.

