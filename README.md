# GlycoDA-Bench Agent Evaluation Code

*A ground-truth-based benchmark for evaluating LLM agents in compositional glycomics
analysis and workflow recommendation, powered by the separately maintained GlycoForge
generator.*

This repository contains the public plan-and-execute agent scaffold, prompts,
glycomics analysis tools, and evaluation code used by **GlycoDA-Bench**. The
benchmark evaluates whether language-model agents can diagnose data conditions,
construct statistically admissible workflows, execute those workflows, and recover
known biological signals.

GlycoForge and GlycoDA-Bench serve different roles: **GlycoForge** is the controlled
glycomics data-generation framework, while **GlycoDA-Bench** is the 648-case benchmark
constructed on top of it. This repository is the experiment and scoring code; it does
not contain the benchmark matrices or evaluator hidden truth.

## Public artifacts

- Manuscript reproduction release (648 cases, evaluator truth, and 22-system
  per-case results):
  <https://huggingface.co/datasets/xiaokui17/glycoforge_benchmark_648/tree/manuscript-v1.0.0>
- V2 paired/coupled extension (648 cases with realized two-block scenarios):
  <https://huggingface.co/datasets/xiaokui17/GlycoDA-Bench/tree/v2.0.0>
- Experiment code: <https://github.com/evolan17/GlycoDA-Bench/tree/v3.0.1>
- Data generator framework: <https://github.com/BojarLab/GlycoForge>

The manuscript reproduction release is the data release associated with the reported
22-system results. The v2 paired/coupled extension is a separate supplementary release;
no manuscript model scores are claimed for v2. Each dataset card and release manifest
distinguishes the `participant`, `evaluator_truth`, and `results` components. Evaluator
truth must never be mounted into an evaluated agent's runtime, even when it is publicly
archived for post-publication reproducibility. See
[`docs/DATA_AND_CODE_AVAILABILITY.md`](docs/DATA_AND_CODE_AVAILABILITY.md).

The archived 22-system CSVs retain the historical `legacy-v1-report` scoring schema.
The code also provides the `aspect-v3-eleven-metric` evaluator for new or explicitly
rescored runs. These schemas are versioned separately and are never silently mixed;
see [`docs/RESULT_SCHEMAS.md`](docs/RESULT_SCHEMAS.md).

## Repository contents

```text
configs/
  api.example.yaml            # credential-free runtime template
  benchmark/                  # benchmark selection settings
  models/                     # 22 sanitized reported-system configurations
  prompts/                    # planner system and user prompts
docs/
  DATA_AND_CODE_AVAILABILITY.md
  METRICS.md
  REPRODUCIBILITY.md
  RESULT_SCHEMAS.md
  WORKFLOW_GROUND_TRUTH.md
  V1_TO_V3.md
scripts/
  validate_release.py         # public-release hygiene gate
  rescore_runs_to_*.py        # metric aggregation utilities
src/
  app/                        # benchmark and single-model entry points
  planner/                    # full and iterative planning
  executor/                   # tool execution
  tools/                      # glycomics analysis tools
  metrics/                    # current eleven-metric evaluator
tests/
```

Intentionally absent: benchmark data, hidden truth, clean/oracle matrices, raw model
outputs, local API files, internal service endpoints, notebooks, caches, and temporary
files.

## Installation

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis_full]"
```

Copy `.env.example` to `.env` or set the variables in your shell:

```text
GLYCODA_API_KEY=<provider key>
GLYCODA_BASE_URL=<OpenAI-compatible API base URL>
```

Never commit `.env` or a populated API configuration.

## Dataset placement

Download or clone only the participant-facing case inputs. A case directory is expected
to provide at least `abundance.csv`, `sample_metadata.csv`, `feature_metadata.csv`,
`manifest.json`, and `task.json`. Point the benchmark YAML to the local case root.
Evaluator truth is resolved separately through `GLYCOAGENT_HIDDEN_TRUTH_ROOT`.

## Running the benchmark

Full-plan mode asks the model for one complete workflow before execution:

```bash
python -m src.app.run_benchmark \
  --models-dir configs/models \
  --benchmark-config configs/benchmark/glycoforge_648.yaml \
  --only-models qwen3-8b-instruct.yaml \
  --planning-mode full \
  --api-config configs/api.example.yaml \
  --workers 1
```

Iterative mode observes each intermediate result before selecting the next tool:

```bash
python -m src.app.run_benchmark \
  --models-dir configs/models \
  --benchmark-config configs/benchmark/glycoforge_648.yaml \
  --only-models qwen3-8b-thinking.yaml \
  --planning-mode iterative \
  --api-config configs/api.example.yaml \
  --workers 1 \
  --job-retries 2
```

Use `--stratify-scenarios` when running a limited subset and `--log-file PATH` for a
plain-text driver log. Generated outputs are written below `outputs/` and are ignored by
Git.

## Evaluation

Seven process metrics use the discrete rubric `{0, 0.5, 1}` and four analytical outcome
metrics remain continuous in `[0, 1]`. Their definitions, formulas, and edge-case rules
are documented in [`docs/METRICS.md`](docs/METRICS.md). The construction facts,
required/optional/unnecessary workflow actions, and treatment of alternative valid
workflows are documented in
[`docs/WORKFLOW_GROUND_TRUTH.md`](docs/WORKFLOW_GROUND_TRUTH.md).

## Verification

```bash
python -m pytest -q
python scripts/validate_release.py .
```

## Reproducibility notes

The exact prompts are versioned under `configs/prompts/`. The 22-system reported model
inventory, experimental model/API identifiers, and per-system evaluation timestamps are
in `configs/models/reported_models.yaml`. Timestamps are transcribed from the detailed
per-model result CSV filenames; those filenames do not encode a timezone, so none is
inferred in the manifest.

## License and citation

Code is released under the Apache License 2.0. Cite the accompanying manuscript using
[`CITATION.cff`](CITATION.cff). GlycoForge is not vendored here: the benchmark build
reports GlycoForge 0.2.0 and pins the exact upstream commit
`3e3ed302b2e3aaef1bc47c407c3528939c482a54`. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.

