# Reproducibility Protocol

## Agent interface

Every system receives the same planner system prompt, user prompt template, case task
context, and registered tool inventory. Prompts are stored in `configs/prompts/`.
Participant-visible context excludes hidden truth and controlled generation states.

The agent supports two execution modes:

- **full**: the model produces a complete workflow plan before any tool is run;
- **iterative**: the model chooses exactly one next tool after observing the previous
  tool output and may perform plan revision after each successful step.

The manuscript results must state which mode was used. Mixing modes within one reported
comparison is not permitted unless explicitly analyzed.

## Iterative control and stopping rules

Iterative execution has a maximum of **12** decisions. The first action must inspect the
dataset. A completed tool cannot be repeated without new evidence. The run stops when:

- the model declares completion;
- differential expression completes and no justified downstream step remains;
- the 12-step limit is reached;
- the model returns no next step;
- the model selects a tool outside the allowed stage;
- dataset inspection is repeated without justification; or
- a failed tool call produces a hard execution error.

An invalid next tool is logged as a protocol failure. A failed tool stops execution; it
is not silently replaced by an oracle action. Full-plan mode may request one plan repair
when structural validation fails. Iterative plan revision is driven only by public
runtime observations and tool outputs.

## Tools

The shared library provides dataset inspection, missing-value imputation, CLR
transformation, batch diagnosis, several batch-correction variants, differential
analysis, PCA, and optional motif enrichment. Tool implementations are in `src/tools/`.
All systems receive the same tool schemas and implementation.

## Models and decoding

The reported inventory contains 22 systems in `configs/models/reported_models.yaml`.
All archived model configurations use `temperature: 0.7`. API keys and endpoints are
provided through `GLYCODA_API_KEY` and `GLYCODA_BASE_URL`; no provider credential is
stored in the repository. Thinking variants pass `enable_thinking` and a thinking
budget through the OpenAI-compatible request body.

Each model entry records the experimental model/API identifier, the detailed source
result CSV, and the evaluation timestamp transcribed from that CSV filename. The source
filenames do not encode a timezone, so the manifest records the local timestamp without
inventing a UTC offset.

For public result tables, `reported_models.yaml` is the authoritative normalization
map. The original `model_name` field is retained as `raw_model_name` because two source
CSVs contain historical aliases: 4 of 648 Qwen3-1.7B-Base rows contain malformed
size-suffixed aliases, and 12 of 648 GLM-5.1 rows use `stream-server-glm5.1`. These are
metadata aliases only; all 22 source CSVs contain 648 rows and 648 unique case IDs.

The reported table contains one benchmark replicate per model-case combination with
benchmark seed 42; transport retries are not counted as independent replicates.
Completed, failed, skipped, and resumed jobs remain distinguishable in the per-case
result status fields, and failed or incomplete runs are not silently replaced by
successful reruns in aggregate tables.

## Seeds, retries, and failures

The default benchmark seed is 42 unless overridden. `--job-retries` retries only
transport-class failures with exponential backoff. Statistical or protocol failures are
not retried. Every completed, skipped, resumed, or failed job is recorded in the
checkpoint and report metadata.

## Data and evaluator truth

The reported manuscript results map to the Hugging Face **manuscript reproduction
release** at `xiaokui17/glycoforge_benchmark_648`. Its release manifests distinguish the
`participant`, `evaluator_truth`, and `results` components. Evaluator truth is resolved
with `GLYCOAGENT_HIDDEN_TRUTH_ROOT`; agents must not receive this path, its contents,
clean matrices, or generation diagnostics. The 22 per-case score tables and aggregation
inputs belong only to this release.

`xiaokui17/GlycoDA-Bench` is the **v2 paired/coupled extension**, whose 216 paired cases
contain two realized compositional blocks. It is released as a supplementary benchmark;
no manuscript model scores are associated with v2.

## Benchmark construction record

Both 648-case releases were materialized specifically for GlycoDA-Bench from
benchmark-specific scenario grids; they were not selected as pre-existing GlycoForge
benchmarks. The manifests report GlycoForge 0.2.0 and exact generation commit
`3e3ed302b2e3aaef1bc47c407c3528939c482a54`. The upstream source remains at
`BojarLab/GlycoForge` under MIT and is not copied into this repository. Dataset releases
preserve their construction scripts, scenario indexes, checksums, and per-case seeds.
Requested and actual modes remain separate fields so the manuscript reproduction
release's proxy cases and v2's realized paired cases are described literally.

## Reproduction checklist

1. Record this repository commit and dataset revision.
2. Record the GlycoForge generator version/commit and benchmark build manifest.
3. Record model/API identifiers and access dates.
4. Fix the planning mode, seed list, temperature, worker count, and retry policy.
5. Run all 648 participant-facing cases against the sealed evaluator truth.
6. Preserve per-case metrics, execution traces, failures, and aggregation inputs.
7. Generate manuscript tables only from versioned per-case result files.

