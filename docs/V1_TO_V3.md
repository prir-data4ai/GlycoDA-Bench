# V1 to V3 Release Notes

## Source relationship

V3 is a clean public release assembled from three sources:

- the paper-oriented V1 scoring release at commit
  `b881576260abec5322c994c5504dfaeaa0054960`;
- compatible runner and model-inventory improvements from the later experiment working
  copy; and
- historical execution context from `GlycoAgentBench` commit
  `f48f7179240c6659c6d36a57213fb26b35d6d10a`.

The historical `GlycoAgentBench` checkout is not identical to the public
`GlycoDA-Bench-v1` repository and does not contain the paper aspect evaluator. V3 uses
the paper evaluator as the scoring authority.

## Metric changes

- Restores the eleven manuscript metric names.
- Keeps seven process metrics discrete at `{0, 0.5, 1}`.
- Uses continuous F1 for DSR and modified NDCG@20 for DRQ.
- Makes top-level BRQ the relative batch-variance reduction defined in the manuscript.
- Makes top-level BSP the mean of F-statistic and correlation-structure preservation.
- Retains older bucketed batch/bio diagnostics only inside `aspect_details`.

Because historical V1 used diagnostic buckets as top-level BRQ/BSP in some paths,
rescored V3 values can differ. Published tables must be generated consistently from one
version and label the version/commit.

## Execution changes

- Preserves full and iterative planning, resumable jobs, stratified partial selection,
  transport retries, checkpointing, and nested run directories.
- Adds plain-text driver logging.
- Removes `<think>...</think>` blocks before fallback JSON parsing.
- Preserves the V1 evaluator/report schema instead of adopting the incompatible local
  `plan_score/execution_score/decision_score/result_score` schema.

## Security and packaging changes

- Removes benchmark data, hidden truth, outputs, notebooks, caches, and local scripts.
- Removes populated API files, internal service defaults, workspace IDs, and tokens.
- Replaces provider credentials and endpoints with environment-variable templates.
- Adds 22 sanitized reported-system configurations and a release validator.

## Model metadata correction

- The release inventory contains the 22 systems present in the detailed experimental
  CSVs. Qwen3 32B Base is not included because it was not evaluated.
- Per-system evaluation timestamps are transcribed from the source result filenames.
  Their timezone was not encoded and is therefore left unspecified.

