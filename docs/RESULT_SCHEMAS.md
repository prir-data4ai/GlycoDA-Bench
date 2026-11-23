# Result Schema Versions

GlycoDA-Bench publishes two explicit result contracts. They describe different
evaluation records and must not be merged by matching similarly named columns.

## `legacy-v1-report`

This is the immutable schema of the 22 historical manuscript result CSVs in the
Hugging Face `manuscript-v1.0.0` release. It contains the original top-level
`plan_score`, `execution_score`, `decision_score`, and `result_score` fields together
with the original diagnostic and outcome columns, status fields, latency, token, and
tool counts. The public `model_summary.csv` and `stratified_summary.csv` summarize only
columns that already exist in those archived CSVs; no experimental value is rewritten.

The schema ID is recorded in `results/release_manifest.json`. The source CSV SHA-256
values, row counts, model aliases, and evaluation timestamps are recorded per model.

## `aspect-v3-eleven-metric`

This is the output contract of `src/metrics/aspect_evaluator.py` for new runs or runs
that are deliberately rescored with the current evaluator. It exposes seven process
metrics and four continuous outcome metrics documented in `docs/METRICS.md`.

An archived `legacy-v1-report` row is not automatically an
`aspect-v3-eleven-metric` row. Publishing or comparing a table requires its schema ID,
code commit, dataset revision, and whether any explicit rescoring was performed.

## Reproducibility rule

- Reproduce the historical 22-system archive from the tagged manuscript dataset using
  the legacy schema and its pinned source hashes.
- Use the eleven-metric evaluator for prospective evaluations or a separately labeled
  rescoring analysis.
- Never replace archived experimental values in place. A rescore is a new artifact
  with a new manifest and must not be presented as an unchanged historical table.

