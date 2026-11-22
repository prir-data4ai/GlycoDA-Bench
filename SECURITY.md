# Security Policy

## Credentials

Do not commit API keys, bearer tokens, service credentials, internal endpoints, populated
`.env` files, or local API YAML. Configure providers through environment variables such
as `GLYCODA_API_KEY` and `GLYCODA_BASE_URL`.

If a credential has ever appeared in a public commit, deleting the file is insufficient:
rotate or revoke the credential at the provider, then purge it from Git history if the
repository must be retained.

Before every public release run:

```bash
python scripts/validate_release.py .
```

The validator checks tracked files for credential-like values, internal hosts, absolute
machine paths, hidden truth, data, outputs, caches, and temporary directories.

## Reporting

Report security issues privately to the repository maintainers. Do not include a live
credential or private endpoint in a public GitHub issue.

