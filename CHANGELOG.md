# Changelog

## 3.0.1 - 2026-11-23

- Versioned the unchanged manuscript result archive as `legacy-v1-report` and the
  current evaluator as `aspect-v3-eleven-metric`.
- Pinned immutable Hugging Face revisions `manuscript-v1.0.0` and `v2.0.0`.
- Made public-result construction fail closed on missing, invalid, non-finite, or
  out-of-range source fields.
- Hardened trace sanitization and release validation for embedded paths, URLs,
  credentials, and deleted tracked files.
- Added Apache-2.0 package metadata.

## 3.0.0 - 2026-11-23

- Prepared a clean public release without data, hidden truth, outputs, or credentials.
- Restored and tested the eleven manuscript metrics.
- Aligned DSR, DRQ, BRQ, and BSP with the manuscript formulas.
- Preserved compatible full/iterative execution, stratification, retries, and resume
  behavior.
- Added plain-text logging and robust thinking-block parsing.
- Added 22 sanitized reported-system configurations with source-result timestamps.
- Added release hygiene validation, publication documentation, citation metadata, and
  continuous integration.

