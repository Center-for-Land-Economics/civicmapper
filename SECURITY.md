# Security Policy

## Reporting a vulnerability

Please report security issues **privately** to
**greg@landeconomics.org** — do not open a public GitHub issue for
vulnerabilities. We'll acknowledge your report as soon as we can and keep you
informed as we work on a fix.

## Scope

Civic Mapper is a fully public application: there is no login, no user
accounts, and no user-submitted data stored server-side. The attack surface is
small — a static frontend plus a read-only data proxy (`api/`) that serves
parquet/PMTiles files with CORS and per-IP rate limiting. Reports about that
proxy (e.g. path traversal, CORS bypass, denial of service) are especially
welcome.

## Supported versions

Only the latest `main` branch is supported. There are no maintained release
branches; fixes land on `main` and deploy from there.
