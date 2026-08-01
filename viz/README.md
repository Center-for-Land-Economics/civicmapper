# viz — Civic Mapper frontend

The map application: a Vite + TypeScript app that renders parcel GeoParquet
and PMTiles with MapLibre GL. Entry pages: `cities.html` (city picker),
`app.html?city=<key>` (map workspace), `parking.html`, `contribute.html`.

## Run

```bash
npm install
npm run dev        # dev server at http://localhost:5173
npm run build      # production bundle in dist/
npm run preview    # serve the built bundle at http://localhost:4173
npm run typecheck  # tsc --noEmit
```

There is no unit-test suite; `typecheck` + `build` are the checks.

## Data loading and env vars

No env vars are required for local dev: the Vite server proxies `/data/*` to a
parquet/PMTiles host (see `vite.config.ts`), and `/api/*` to a local API on
port 8080. In dev, a parquet dropped into `public/` (e.g.
`<city>-<state>-parcels.parquet`) is preferred over the remote copy for that
city.

Env files are loaded from `viz/env/` (see `envDir` in `vite.config.ts`), not
the package root. Useful variables (all optional):

| Variable | Purpose |
|---|---|
| `VITE_PARQUET_BASE_URL` | Base URL for parquet/PMTiles hosting (also the dev proxy's upstream — don't set it to `/data`) |
| `VITE_PMTILES_BASE_URL` | Override base URL for PMTiles only (e.g. a local tile server) |
| `VITE_PMTILES_LOCAL_PREFIX` | Scope the PMTiles/dataset override to files matching this prefix (e.g. `houston-tx`) |
| `VITE_DEFAULT_DATASET_URL` | Hard override: load this one parquet (scoped by the prefix above) |
| `VITE_API_BASE` | API base path/URL (default `/api`) |

Registering a new city = two JSON files, `src/cities/<city>.json` (config + coords)
and `src/dictionaries/<city>.json` (field labels) — both auto-discovered at build
time; see [docs/add-city-playbook.md](../docs/add-city-playbook.md).
