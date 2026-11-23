# Representative sanitized traces

These three examples come from the manuscript reproduction release and cover the three
task families:

- `gf_case_0001`: single-workflow case;
- `gf_case_0217`: batch-correction-centered case;
- `gf_case_0433`: paired/coupled-requested proxy case in the manuscript release.

Each JSON file preserves the structured plan, observed tool sequence, decision log,
metrics, and evaluator verdict needed to understand the evidence–judgment–action chain.
Raw provider messages, service endpoints, credentials, and local absolute paths are not
included. These examples are explanatory artifacts, not additional model runs.

Regenerate them with `scripts/export_public_traces.py` from the corresponding archived
`report.json` files. The manuscript release's paired/coupled-requested cases are proxy
scenarios; realized two-block examples belong to the separate v2 extension.
