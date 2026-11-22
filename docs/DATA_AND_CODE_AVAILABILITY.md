# Data and Code Availability

## Participant-facing data

The 648 GlycoDA-Bench cases are distributed through:

<https://huggingface.co/datasets/xiaokui17/GlycoDA-Bench>

The public participant-facing release should contain only the matrices, sample and
feature metadata, task descriptions, and non-oracle structural metadata required by an
agent. It must not expose evaluator hidden truth, clean/oracle matrices, controlled
generation intermediates, or answer-bearing diagnostics.

## Evaluator truth

Ground-truth differential features, importance ordering, workflow requirements, and
motif follow-up truth belong in a separately versioned evaluator artifact. During an
evaluation they are made available only to the scorer through
`GLYCOAGENT_HIDDEN_TRUTH_ROOT`. If full truth is released for reproducibility, it should
remain separate from participant inputs and carry a clear no-agent-access warning.

## Code

The public experiment implementation is intended for:

<https://github.com/evolan17/GlycoAgentBench-v3>

It contains the prompts, agent scaffold, tools, failure policy, eleven metric
implementations, model metadata, rescore scripts, and tests. Generated model outputs are
not committed; a versioned result artifact should accompany the paper if licensing and
provider terms permit redistribution.

## Generator provenance

GlycoForge is available at <https://github.com/BojarLab/GlycoForge>. The benchmark
release must additionally record the exact generator commit, scenario grid, random
seeds, validation command, and checksums used to build the 648 cases. The core
GlycoForge repository alone is not a substitute for the benchmark-specific build
recipe.

## Suggested manuscript statement

> The complete participant-facing GlycoDA-Bench release, comprising 648 benchmark
> cases, is available from Hugging Face. The public inputs exclude evaluator-only hidden
> truth and controlled generation states. The agent scaffold, prompts, tool
> implementations, evaluation metrics, model configuration templates, and scripts used
> to reproduce the reported analyses are available from the versioned code repository.
> Evaluator truth is distributed separately and was not accessible to evaluated agents.
> GlycoForge is the underlying data-generation framework, whereas GlycoDA-Bench is the
> benchmark suite constructed from a pinned GlycoForge version and benchmark-specific
> scenario definitions.

