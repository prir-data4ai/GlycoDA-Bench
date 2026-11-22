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

The reported inventory contains 23 systems in `configs/models/reported_models.yaml`.
All archived model configurations use `temperature: 0.7`. API keys and endpoints are
provided through `GLYCODA_API_KEY` and `GLYCODA_BASE_URL`; no provider credential is
stored in the repository. Thinking variants pass `enable_thinking` and a thinking
budget through the OpenAI-compatible request body.

Exact access dates were not present in the available source archive and remain marked
as unresolved in the manifest. Authors must confirm them before final submission rather
than infer dates from file timestamps.

## Seeds, retries, and failures

The default benchmark seed is 42 unless overridden. `--job-retries` retries only
transport-class failures with exponential backoff. Statistical or protocol failures are
not retried. Every completed, skipped, resumed, or failed job is recorded in the
checkpoint and report metadata.

## Data and evaluator truth

Participant matrices come from the versioned Hugging Face release. Evaluator truth is
resolved separately with `GLYCOAGENT_HIDDEN_TRUTH_ROOT`. Agents must not receive this
path, its contents, clean matrices, or generation diagnostics.

## Reproduction checklist

1. Record this repository commit and dataset revision.
2. Record the GlycoForge generator version/commit and benchmark build manifest.
3. Record model/API identifiers and access dates.
4. Fix the planning mode, seed list, temperature, worker count, and retry policy.
5. Run all 648 participant-facing cases against the sealed evaluator truth.
6. Preserve per-case metrics, execution traces, failures, and aggregation inputs.
7. Generate manuscript tables only from versioned per-case result files.

