# Contributing to Civic Mapper

Thanks for your interest! Contributions of all kinds are welcome — bug fixes,
frontend features, documentation, and especially new cities.

## Dev environment

Prerequisites: **Node 20+**. Python 3.11+ is only needed if you work on the
ETL side.

```bash
./start-local.sh        # macOS / Linux — API on :8080, frontend on :5173
start-local.bat         # Windows
```

Or run the pieces individually — see [viz/README.md](viz/README.md) and
[api/README.md](api/README.md). No environment variables are needed for a
basic local run.

For the Python ETL environment (venv, Jupyter kernel, tippecanoe/pmtiles),
follow [data/README.md](data/README.md).

## Tests and checks

Run these before opening a PR:

```bash
# API — unit tests (Node's built-in test runner)
cd api && npm test

# Frontend — type check and production build (there is no frontend unit-test
# suite yet; typecheck + build are the required checks)
cd viz && npm run typecheck
cd viz && npm run build
```

There is also a headless smoke test that loads every deployed city
(`scripts/smoke_dev_cities.py`); it targets the dev site and requires
Playwright, so it's optional for contributors.

## Pull requests

- Open PRs against `main`.
- Keep PRs focused — one fix or feature per PR.
- Make sure `npm test` (api) and `npm run typecheck` + `npm run build` (viz)
  pass.
- Describe what changed and how you verified it (a screenshot helps for
  visual changes).
- Don't commit generated artifacts (`viz/dist/`, `node_modules/`) or large
  datasets.

## Contributing a new city

This is the highest-impact contribution. The full procedure — finding parcel
and assessment data, ETL, the canonical GeoParquet export, optional PMTiles
and parking data, and frontend registration — is documented in:

- [docs/add-city-playbook.md](docs/add-city-playbook.md) — the step-by-step playbook
- [docs/parcel-parquet-format.md](docs/parcel-parquet-format.md) — the required output format

The short version: a city needs open parcel **geometry** plus per-parcel
**assessed values** (land and improvement, or at least land). If you know of a
city with good open data but don't want to build it yourself, open a
[city request issue](.github/ISSUE_TEMPLATE/city_request.md) with links to the
data portals — scouting the data sources is half the work.

## Questions

Open an issue, or reach the maintainers at the
[Center for Land Economics](https://landeconomics.org).
