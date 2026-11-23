# Workflow Ground Truth and Admissible Alternatives

## What is ground truth

GlycoDA-Bench separates two layers that should not be conflated.

1. **Case facts known by construction.** GlycoForge records the biological labels,
   injected batch and missingness conditions, differential-feature set and ranking,
   motif flag, dependence request, simulation parameters, and clean/oracle references.
   These values are deterministic outputs of the versioned generator and case seed; they
   are not inferred by a human annotator or an LLM.
2. **Admissible workflow requirements.** Statistical requirements are derived from the
   case facts, participant-visible dataset properties, and expert-defined rules encoded
   in `src/oracle/rules.py` and `src/metrics/aspect_evaluator.py`. These requirements are
   a transparent benchmark policy, not a claim that one unique workflow is universally
   optimal.

The scorer is fully automatic and rule-based. No LLM judge and no per-case human rating
is used. Expert review is still relevant for validating the scientific reasonableness
and scope of the rule set; deterministic generation alone does not make a normative
workflow rule beyond criticism.

## Required, optional, and unnecessary actions

| Case condition | Required or reliable behavior | Optional/admissible alternatives | Unnecessary or invalid behavior |
| --- | --- | --- | --- |
| Every case | Inspect the dataset before committing to interventions; preserve the declared matrix orientation. | Equivalent inspection evidence represented in the structured plan can receive partial credit where the rubric permits it. | Empty plans or tools outside the registered protocol. |
| Compositional abundance | Apply CLR/log-ratio handling before downstream differential analysis and before compatible batch intervention. | Legal pseudocount choices and other parameterizations accepted by the public schema. | Differential analysis or batch correction in an incompatible untransformed space. |
| Injected/material missingness | Impute before CLR and downstream analysis using a legal strategy. | `median` or `left_censored_small_value`, selected to match the diagnosed pattern. | Omitting required imputation, or imputing data with no meaningful missingness without justification. |
| Meaningful batch structure | Diagnose the batch structure and apply a compatible correction before differential analysis while preserving the biological covariate. | Any registered batch family may be valid when its assumptions and parameters fit: ComBat, ratio adjustment, percentile normalization, limma-style removal, or Harmony. | No correction when required, correction after differential analysis, or correction when no meaningful batch structure exists. |
| Borderline batch evidence | Run an explicit batch check and make a case-grounded decision. | Either correction or no correction can be accepted when supported by the observed diagnostic and recorded rationale. | Applying correction without checking the borderline evidence. |
| Motif follow-up worthwhile | Perform motif-level follow-up after a valid differential result. | Omission can receive partial rather than hard-failure treatment when the core workflow is otherwise valid, according to the public rubric. | Treating motif enrichment as a substitute for feature-level differential analysis. |
| Motif follow-up not required | Stop after the justified core analysis. | Motif exploration may be reported as exploratory when clearly labeled. | Presenting unnecessary motif analysis as a required benchmark step. |
| Paired/coupled structure realized | Acknowledge dependence and use a dependence-aware analysis supported by the registered tool protocol. | Equivalent dependence-aware methods are admissible when represented by an accepted tool/action category. | Treating dependent observations as independent without qualification. |

## How alternative valid workflows are recognized

Scoring operates on action categories, order constraints, parameters, and recorded
evidence rather than exact string matching against one canonical sequence. For example,
the batch requirement accepts any member of the public `BATCH_TOOLS` registry when the
method is justified and ordered before differential analysis. Optional steps are not
required for full core-workflow credit, while unnecessary interventions affect economy
and case-awareness scores rather than automatically invalidating every other component.

The seven process metrics separately score core ordering, case awareness, parameter
soundness, economy, diagnostic reasoning, action selection, and audit closure. Their
`0`, `0.5`, and `1` rules are documented in `docs/METRICS.md`; the executable authority
is `src/metrics/aspect_evaluator.py`.

## Validation and leakage control

- Every truth record is linked by `dataset_id` to one of the 648 participant cases.
- The release preserves the generator version, case seed, scenario fields, and checksums
  so construction facts can be audited.
- Rule behavior is covered by `tests/test_oracle_rules.py` and
  `tests/test_aspect_scoring.py`.
- Evaluator truth is stored separately from participant inputs and resolved only through
  `GLYCOAGENT_HIDDEN_TRUTH_ROOT`; evaluated agents must not receive that path or network
  access to the truth artifact.
- Results should be interpreted within the implemented scenario and tool library. A
  statistically defensible workflow outside the registered alternatives may be
  under-credited and should be added through a versioned rubric revision rather than an
  ad hoc per-case judgment.
