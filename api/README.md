# api — Civic Mapper data proxy

A small Express service that proxies parcel datasets (GeoParquet, PMTiles,
metadata JSON) from blob storage to the browser, adding CORS headers, per-IP
rate limiting, and range-request pass-through. Deployed builds of the frontend
fetch all data through it; it can also append a SAS token server-side so the
backing storage container can stay private.

## Run

```bash
npm install
npm start          # listens on port 8080 (or $PORT)
```

## Test

```bash
npm test           # Node's built-in test runner (test/server.test.js)
```

## Endpoints

- `GET /healthz` — health check
- `GET|HEAD /data/:filename` (also `/api/data/...`) — proxy a dataset blob
- `GET|HEAD /data/parking/:filename` — parking datasets
- `GET|HEAD /data/staging/:stagingId/:filename` — staged datasets for review
- `POST /telemetry` — browser RUM events (logged to stdout as JSON)

## Env vars

| Variable | Purpose |
|---|---|
| `PORT` | Listen port (default `8080`) |
| `DATA_PROXY_BASE_URL` | Upstream blob container base URL (alias: `BLOB_STORAGE_BASE`) |
| `BLOB_SAS_TOKEN` | Optional SAS token appended to upstream requests (no leading `?`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS allowlist (`*` allows any; defaults cover civicmapper.org + localhost) |
