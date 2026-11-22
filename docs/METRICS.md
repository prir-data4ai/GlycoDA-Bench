# Evaluation Metrics

GlycoDA-Bench reports seven process metrics and four outcome metrics. Process metrics
use `{0, 0.5, 1}` for failure, partial validity, and reliable validity. Outcome metrics
are continuous on `[0, 1]`.

## Process metrics

### Core Workflow Correctness

Checks whether required missing-value handling, CLR transformation, and batch-sensitive
intervention are present and occur before differential analysis. All required core
conditions satisfied gives `1`; at least one valid required condition gives `0.5`; no
required condition satisfied gives `0`.

### Case Awareness

Measures whether the plan recognizes observed missingness, batch risk,
compositionality, and biological confounding. At least three reliable judgments give
`1`; two partial/reliable judgments give `0.5`; otherwise `0`.

### Parameter Soundness

Combines parameter legality, statistical reasonableness, and method-specific settings.
Illegal parameters fail the metric. Reliable settings across the three components give
`1`, partial validity gives `0.5`, and invalid settings give `0`.

### Workflow Economy

Penalizes redundant tools, unjustified intervention, and template-stacked workflows.
Economical, case-specific workflows give `1`; limited redundancy gives `0.5`; severe
redundancy or unnecessary intervention gives `0`.

### Diagnostic Reasoning

Evaluates missingness diagnosis, batch diagnosis, risk stratification, and evidence
specificity. Three reliable components give `1`; two partial/reliable components give
`0.5`; otherwise `0`.

### Action Selection

Evaluates protocol compliance and whether actions match missingness, batch, and
follow-up requirements. A protocol violation is a hard failure. Otherwise, all action
components reliable gives `1`, partial agreement gives `0.5`, and invalid actions give
`0`.

### Audit Closure

Checks whether observations, judgments, chosen actions, and rejected alternatives form
a coherent audit record for missingness, batch, compositionality, and follow-up. Three
closed aspects give `1`; two partially or fully closed aspects give `0.5`; otherwise
`0`.

## Outcome metrics

### Differential Set Recovery

Differential Set Recovery (DSR) is F1 between the predicted significant feature set and
the ground-truth differential set:

```text
DSR = 2 * precision * recall / (precision + recall)
```

DSR is `0` when both precision and recall contribute no positive denominator.

### Differential Ranking Quality

Differential Ranking Quality (DRQ) is a modified binary-relevance NDCG@20. For rank
`i` starting at one, the discount is `(log2(i + 1))^0.3`. The exponent `0.3` slows the
discount so recovery across the full top-20 set matters rather than only the first few
positions. DRQ is DCG@20 divided by the ideal DCG@20 and is `0` when no ranked output or
ground-truth positives are available.

### Batch Removal Quality

Batch Removal Quality (BRQ) is the relative reduction in the PVCA-style batch variance
proxy:

```text
BRQ = clip((batch_before - batch_after) / batch_before, 0, 1)
```

The zero denominator rule is: if `batch_before <= 0`, BRQ is `0`. Over-correction and
case-matching diagnostics are retained in `aspect_details` but do not replace the
continuous manuscript BRQ.

### Biological Signal Preservation

Biological Signal Preservation (BSP) averages preservation of the group-separation
F-statistic proxy and preservation of the feature-correlation structure:

```text
BSP = 0.5 * f_statistic_preservation + 0.5 * correlation_structure_preservation
```

Both components are clipped to `[0, 1]`. Additional truth-retention diagnostics are
reported in `aspect_details` but do not replace the continuous manuscript BSP.

## Implementation

The authoritative implementation is `src/metrics/aspect_evaluator.py`. The public
top-level keys are:

```text
core_workflow_correctness, case_awareness, parameter_soundness,
workflow_economy, diagnostic_reasoning, action_selection, audit_closure,
de_set_recovery, de_ranking_quality, batch_removal_quality,
biological_signal_preservation
```

