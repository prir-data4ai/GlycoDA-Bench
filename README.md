# GlycoDA-Bench Agent Evaluation Code

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

- Participant-facing dataset: <https://huggingface.co/datasets/xiaokui17/GlycoDA-Bench>
- Experiment code: <https://github.com/evolan17/GlycoAgentBench-v3>
- Data generator framework: <https://github.com/BojarLab/GlycoForge>

Evaluator-only hidden truth must be kept separate from participant-facing inputs and
must not be exposed to evaluated agents. See
[`docs/DATA_AND_CODE_AVAILABILITY.md`](docs/DATA_AND_CODE_AVAILABILITY.md).

## Repository contents

```text
configs/
  api.example.yaml            # credential-free runtime template
  benchmark/                  # benchmark selection settings
  models/                     # 23 sanitized reported-system configurations
  prompts/                    # planner system and user prompts
docs/
  DATA_AND_CODE_AVAILABILITY.md
  METRICS.md
  REPRODUCIBILITY.md
  V1_TO_V3.md
scripts/
  validate_release.py         # public-release hygiene gate
  rescore_runs_to_*.py        # metric aggregation utilities
src/
  app/                        # benchmark and single-model entry points
  planner/                    # full and iterative planning
  executor/                   # tool execution
  tools/                      # glycomics analysis tools
  metrics/                    # eleven manuscript metrics
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
are documented in [`docs/METRICS.md`](docs/METRICS.md).

## Verification

```bash
python -m pytest -q
python scripts/validate_release.py .
```

## Reproducibility notes

The exact prompts are versioned under `configs/prompts/`. The reported model inventory
and experimental model IDs are in `configs/models/reported_models.yaml`. The available
source archive did not record exact access dates for every model; this unresolved item is
marked explicitly in the manifest and must be confirmed before the camera-ready release.

## License and citation

Code is released under the MIT License. Cite the accompanying manuscript using
[`CITATION.cff`](CITATION.cff).

