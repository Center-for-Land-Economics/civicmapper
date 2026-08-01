# Civic Mapper

Interactive 3D maps of parcel-level property data — land value, improvement
value, value per acre, improvement-to-land ratio, surface parking, and more —
for cities across the US and beyond.

**Live site:** [civicmapper.org](https://civicmapper.org)
(dev: [dev.civicmapper.org](https://dev.civicmapper.org))

Maintained by the [Center for Land Economics](https://landeconomics.org).

## How it works

Each city's parcel data is cleaned and exported to
[GeoParquet](docs/parcel-parquet-format.md) by a Python ETL notebook or script.
Larger cities also get PMTiles vector tiles (with an H3 hexagon overlay) so
they render at any scale. The frontend loads these files over HTTP and renders
them with MapLibre GL; a small Express proxy serves the files with CORS and
rate limiting in deployed environments.

## Tech stack

- **Frontend** — TypeScript, Vite, MapLibre GL JS, PMTiles, hyparquet/geoparquet, three.js
- **API** — Node.js, Express (a thin data proxy: CORS, rate limiting, blob pass-through)
- **ETL** — Python 3.11+, GeoPandas, Jupyter notebooks, tippecanoe/pmtiles

## Repository layout

| Path | What it is |
|---|---|
| `viz/` | The map frontend (Vite + MapLibre). See [viz/README.md](viz/README.md) |
| `api/` | Express data proxy. See [api/README.md](api/README.md) |
| `data/` | Per-city ETL notebooks, shared Python utilities, post-processing scripts. See [data/README.md](data/README.md) |
| `docs/` | Design docs and playbooks ([add a city](docs/add-city-playbook.md), [parquet format](docs/parcel-parquet-format.md), …) |
| `scripts/` | Repo-level utilities (e.g. headless smoke test of every deployed city) |
| `site/` | Static landing page for civicmapper.org |
| `.github/` | CI/CD workflows (deploy, ETL runs, smoke tests) |

## Quickstart

Prerequisites: **Node 20+** (frontend/API), **Python 3.11+** (only needed for
the ETL side).

The one-shot way — starts the API on port 8080 and the Vite dev server on
port 5173, installing dependencies on first run:

```bash
./start-local.sh        # macOS / Linux
start-local.bat         # Windows (opens two terminal windows)
```

Or manually:

```bash
# Terminal 1 — API (optional for local dev; the frontend has its own dev proxy)
cd api
npm install
npm start               # http://localhost:8080

# Terminal 2 — frontend
cd viz
npm install
npm run dev             # http://localhost:5173
```

Open <http://localhost:5173> — it takes you to the city picker
(`cities.html`); each city opens as `app.html?city=<key>`.

No environment variables are required for a basic local run: in dev mode the
Vite server proxies dataset requests to a hosted data endpoint, so cities load
out of the box. To run against **your own data**, either:

- drop a `<city>-<state>-parcels.parquet` file into `viz/public/` — in dev the
  app prefers a local copy of the current city's file when one exists, or
- point `VITE_PARQUET_BASE_URL` (and friends — see
  [viz/README.md](viz/README.md)) at your own parquet/PMTiles hosting.

For the ETL side (producing those parquet/PMTiles files yourself), follow
[data/README.md](data/README.md).

## Adding a city

The end-to-end procedure — sourcing parcel data, ETL, PMTiles, frontend
registration, deployment — is documented in
[docs/add-city-playbook.md](docs/add-city-playbook.md). The required output
format is specified in
[docs/parcel-parquet-format.md](docs/parcel-parquet-format.md).

Want a city but don't want to build it? Open a
[city request](.github/ISSUE_TEMPLATE/city_request.md) issue.

## Deploying your own instance

Civic Mapper works like a library: the engine is generic, and a deployment is
the engine plus configuration pointing at *your* data hosting. To stand up
your own instance:

1. **Host the data** — parquet/PMTiles/metadata files on any static host that
   supports HTTP range requests (object storage, a CDN, or the bundled
   [api/](api/README.md) proxy in front of a private bucket). Produce the
   files with the [ETL toolkit](data/README.md), following the
   [parquet format spec](docs/parcel-parquet-format.md).
2. **Point the frontend at it** — set `VITE_PARQUET_BASE_URL` (and friends;
   the full list with docs is in
   [viz/env/.env.example](viz/env/.env.example)). ETL-side variables are in
   [data/.env.example](data/.env.example). Secrets never live in this repo —
   deployments supply them via environment / CI secrets.
3. **Register your cities** — each city is a drop-in
   `viz/src/cities/<key>.json` + `viz/src/dictionaries/<key>.json`; no shared
   files to edit.
4. **CI/CD** — the workflows in `.github/workflows/` are the maintainers'
   deploy pipelines and are guarded by repository owner, so they skip
   automatically on forks (no failing runs). Copy one as a starting point for
   your own target and supply your own repository secrets.

## Data attribution

Parcel and assessment data come from each jurisdiction's open-data programs.
Per-source licenses and credits are listed in [ATTRIBUTION.md](ATTRIBUTION.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: see
[SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE).
