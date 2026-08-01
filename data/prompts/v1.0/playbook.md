# Add A New City

This document is the end-to-end playbook for adding a new city to CivicMapper.
It is written for engineers and coding agents. The goal is that a new city can be
added without tribal knowledge.

This repo has three separate concerns:

1. Parcel ETL: find source data, clean it, classify it, exclude exempt parcels, export canonical GeoParquet.
2. App integration: register the city, add the data dictionary, optionally generate PMTiles and parking datasets.
3. Deployment: verify locally, upload to dev, validate, then promote to prod.

Use this document as the checklist.

## Outcome

For a fully integrated city, you should end with:

- A city ETL notebook or script under `data/jurisidictions/`
- A canonical parcel parquet named `<city>-<state>-parcels.parquet`
- Optionally PMTiles + metadata for larger cities
- Optionally parking data under `parking/<city>-<state>-parking-lots.*`
- A city registry file `viz/src/cities/<city>.json`
- A city dictionary in `viz/src/dictionaries/<city>.json`
- A city dictionary `viz/src/dictionaries/<city>.json` (the picker card is generated automatically)
- Dev deployment validated
- Prod deployment validated

## 1. Choose The City Key And Filenames

Pick these first and keep them stable:

- `city key`: lowercase slug used in URLs and code, for example `baltimore`, `nyc`, `stpaul`
- `state`: two-letter lowercase state code
- canonical parcel filename: `<city>-<state>-parcels.parquet`
- PMTiles filename: `<city>-<state>-parcels.pmtiles`
- PMTiles metadata filename: `<city>-<state>-parcels-metadata.json`
- parking filename: `<city>-<state>-parking-lots.parquet`
- parking metadata filename: `<city>-<state>-parking-lots-metadata.json`

Register the canonical naming in [parquet_registry.py](data/parquet_registry.py) if the city is new.

## 2. Find The Source Data

For each city, find:

- parcel geometry source
- assessed land value source
- assessed improvement/building value source
- parcel identifier source
- parcel detail link source, if available
- exemption fields or ownership fields
- land use fields

Typical sources:

- ArcGIS FeatureServer / MapServer
- city open data portals
- county assessor portals
- state parcel datasets

Important:

- Many “city” projects actually start from a countywide or regional dataset. Restrict to the city boundary before export.
- Some cities expose city-only data already. Baltimore is an example of a city-only source.

When the source is broader than the city:

- obtain a reliable city boundary
- filter to parcels intersecting or contained by that boundary
- document the filter logic in the notebook/script

Do not assume the dataset is already city-only.

## 3. Create The ETL Notebook Or Script

Use existing notebooks as patterns:

- Denver: PMTiles large-city pattern
- NYC: PMTiles large-city pattern with explicit metadata generation
- Cincinnati / Spokane / St. Paul: parcel export plus optional parking
- Baltimore: city-specific run script plus PMTiles and parking

Recommended location:

- notebook: `data/jurisidictions/<city>.ipynb`
- optional script: `data/jurisidictions/run_<city>.py`

The ETL should be explicit and reproducible. Do not bury critical logic in ad hoc notebook cells with no comments.

## 4. Restrict To The City

If the source is countywide or larger:

- filter on jurisdiction field if the source has one
- otherwise spatially clip/filter using a city boundary

Validate before moving on:

- total row count looks plausible
- bounds look correct
- obvious neighboring municipalities are excluded

This is one of the easiest places to silently ship bad data.

## 5. Handle Condos, Duplicates, And Multi-Parcel Cases

Before classification/export, inspect duplicate parcel identifiers.

Common patterns:

- condo units duplicated on the same footprint
- multiple tax records sharing one geometry
- one assessor parcel split into multiple rows
- condo buildings or units sitting on a separate shared land parcel that has no normal assessor schedule row

Typical fix:

- detect duplicates on the parcel key
- aggregate numeric fields by sum where appropriate
- keep categorical fields with a deterministic rule such as `first`
- union geometries when duplicate rows represent the same parcel footprint

Baltimore uses this pattern in [run_baltimore.py](data/jurisidictions/run_baltimore.py).

Fort Collins lesson for future cities:

- some condo complexes did not fail as duplicate keys at all; the visible “tower” rows were valid child parcels sitting on top of a different shared ground parcel
- the real ground parcel sometimes had a blank/null schedule field, but in other cases it was an association/common-area parcel that looked exempt or non-taxable and would be removed by a normal exempt filter
- this means condo cleanup has to happen before you permanently disregard those parent candidates, or you will leave unit parcels floating above missing land in 3D

Short diagnostic for future cities:

- if condo rows still look wrong after duplicate-key or same-footprint collapse, search the raw geometry feed for larger polygons that spatially intersect many condo rows, even if they have blank schedules, association owners, common-area names, or non-taxable account types
- pay extra attention when the condo child rows all have zero or near-zero land square footage, or when there is an obvious missing parcel underneath the buildings in the map
- subdivision name can help confirm the pattern, but it should be a sanity check, not the only merge key; spatial overlap and shared site geometry are more reliable
- prefer a hierarchy like: shared raw ground parcel first, common-element plat second, and only then narrower fallback heuristics
- when you do exempt filtering, make sure condo parent candidates are preserved long enough to participate in the merge, then drop the leftover exempt parcels after the condo ground parcel is formed

Do not skip this step for cities with condo-heavy downtowns.

## 6. Exclude Exempt Parcels Correctly

This is mandatory. Exempt land often has distorted or unusable assessed values and will skew the map badly.

You need an `exemption_flag` field in the export.

Examples already in the repo:

- direct exemption flag from source, for example `full_exmp`, `TaxExemptYN`
- derived from category such as `E/EC`
- derived from exempt value fields such as `LANDEXMP` and `IMPREXMP`
- derived from government ownership when source coding is inconsistent

Rule:

- If a parcel is truly exempt, set `exemption_flag = 1`
- Exclude those parcels from the shipped parcel dataset unless there is a strong reason not to

Baltimore-specific lesson:

- government-owned parcels can be miscoded as commercial and have zero exempt value fields
- ownership heuristics may be necessary

At minimum, inspect:

- government-owned parcels
- schools
- parks
- utility land
- public housing
- state and federal property

Sanity checks:

- `exemption_flag` should not be all zeros unless the city truly lacks exempt data
- sample obviously public parcels manually
- compare high-value outliers; many “weird giant bars” are exempt/government records that leaked in

## 7. Build Original And Refined Property Categories

Every city needs:

- original/raw category field
- refined underutilization field

Usually these become:

- `property_land_use_category` or `PROPERTY_CATEGORY`
- `property_land_use_refined` or `property_category_refined`

The refined field is used for the main “Vacant / Parking Lot / Underdeveloped” logic.

Typical refined logic:

```python
def categorize_property_refined(row):
    cat = str(row["PROPERTY_CATEGORY"])
    if "Vacant" in cat:
        return "Vacant"
    elif "Parking" in cat:
        return "Parking Lot"
    elif row["improvement_value"] < 0.5 * (row["land_value"] + row["improvement_value"]):
        return "Underdeveloped"
    else:
        return None
```

You should adapt the original categorization to the local schema first:

- use land use code fields
- use use-group fields
- use building class codes
- use vacancy flags

The classification should be city-specific, not generic.

Always print value counts for:

- original category
- refined category

Review those counts before exporting.

## 8. Compute Canonical Fields

The app expects the canonical parcel schema documented in
[parcel-parquet-format.md](docs/parcel-parquet-format.md).

Common fields to populate:

- `geometry`
- `exemption_flag`
- `property_land_use_category` or `PROPERTY_CATEGORY`
- `property_land_use_refined` or `property_category_refined`
- `current_full_land_value`
- `improvement_value`
- `full_market_value`
- per-sqft fields
- `TLLDIMPROV`
- `IMPR_LAND_RATIO`
- `IMPR_LAND_PCT`
- `IMPR_PCT_TOTAL`
- `link`

Notes:

- geometry must be polygonal and valid
- CRS must be `EPSG:4326`
- parcel area should be computed from geometry if the source area is unreliable
- links should point to the assessor/source record when possible

The helper [add_improvement_ratio_fields](data/parcel_calculations.py)
should be used for the improvement/land ratio metrics.

## 9. Save The Canonical Parcel Output

Preferred local output path:

- `data/jurisidictions/data/<city>/<city>-<state>-parcels.parquet`

Also save:

- a dated parquet snapshot when useful
- raw/cache parquet if the source extraction is expensive

Use canonical names that match `data/parquet_registry.py`.

## 10. Decide Whether The City Needs PMTiles

Small and moderate cities can use the raw GeoParquet browser path.
Larger cities should use PMTiles.

Use PMTiles when:

- parcel count is large
- browser parsing is slow
- the full dataset makes the map unstable
- low-zoom rendering needs a pre-aggregated layer

Examples already using PMTiles:

- Denver
- NYC
- Baltimore

Generate PMTiles with:

```bash
python data/scripts/parquet_to_pmtiles.py \
  --city <city> \
  --file data/jurisidictions/data/<city>/<city>-<state>-parcels.parquet \
  --upload --overwrite
```

This script:

- computes PMTiles metadata
- builds a low-zoom aggregate layer
- creates `.pmtiles`
- uploads PMTiles + metadata to blob storage

If a city looks bad fully zoomed out, tune the low-zoom layer in:

- [parquet_to_pmtiles.py](data/scripts/parquet_to_pmtiles.py)
- city dictionary `_config`

## 11. Optional: Run The Surface Parking Pipeline

Parking is a separate dataset and page. It is not derived automatically from parcel ETL.

Output location:

- `data/parking/<city>/`

Primary script:

- [parking_lot_extraction.py](data/scripts/parking_lot_extraction.py)

Two modes:

- full NAIP + ML segmentation pipeline
- `--osm-only` fast fallback

Examples:

```bash
python data/scripts/parking_lot_extraction.py --city <city> --upload --overwrite
python data/scripts/parking_lot_extraction.py --city <city> --osm-only --upload --overwrite
```

Requirements for the parking UI to show:

- city has `parkingFilename` in its `viz/src/cities/<city>.json`
- city dictionary `_config.hasParkingData` is `true`
- `VITE_PARKING_ENABLED` is enabled in the build
- the parking parquet and metadata exist in blob storage under `parking/`

## 12. Register The City In The Frontend

Create `viz/src/cities/<city>.json` — the registry is data-driven and discovers
every JSON in that folder via `import.meta.glob`, so this file is the ONLY
registration step (no shared source files change; the city picker card is
generated automatically). The filename minus `.json` is the city key used in
`?city=<key>` URLs. Include:

- `displayName`
- `state`
- `coords` — `[lng, lat]` map center (REQUIRED; the loader rejects the file without it)
- `filename`
- `pmtilesFilename` if using PMTiles
- `parkingFilename` if parking exists
- `devCategoryField`
- `origCategoryField`
- `aliases` if needed

See any existing file in `viz/src/cities/` for the shape; the full field
reference is the `CityDef` interface in [cities.ts](viz/src/cities.ts).

## 13. Create The Data Dictionary

Add `viz/src/dictionaries/<city>.json`.

This file does two things:

1. labels fields for UI display
2. stores city `_config`

Common `_config` keys:

- `usePmtiles`
- `pmtilesUrl`
- `pmtilesMetadataUrl`
- `hasParkingData`
- `lowZoomOpacityMultiplier`
- `lowZoomFadeStart`
- `lowZoomFadeEnd`
- city-specific smoothing or field overrides if needed

Only fields in the dictionary survive into popup rendering, so include any field you want visible in the app.

## 14. Local Verification

Start local app + API:

```bash
./start-local.sh
```

Local URLs:

- app: `http://localhost:5173/app.html?city=<city>`
- parking: `http://localhost:5173/parking.html?city=<city>`

Before deploying, verify:

- city appears in selector / city card list
- map loads without console errors
- PMTiles city loads via proxied `/api/data/...` path
- exempt parcels are absent
- category filters work
- popup fields are correct
- parking link appears only when expected
- parking page loads when expected

Build the frontend locally:

```bash
cd viz
npm run build
```

Run API tests when backend/CORS behavior changes:

```bash
cd api
npm test
```

## 15. Dev Deployment

### Frontend

The dev frontend deploy is automatic on push to `develop` when files under `viz/**` change.

Workflow:

- [deploy-frontend-dev.yml](.github/workflows/deploy-frontend-dev.yml)

Important build env:

- `VITE_API_BASE=https://api.dev.civicmapper.org`
- `VITE_TILE_API_URL=https://tiles.dev.civicmapper.org`
- `VITE_PARKING_ENABLED=true`

### Backend

The dev backend deploy is automatic on push to `develop` when files under `api/**` change.

Workflow:

- [deploy-backend-dev.yml](.github/workflows/deploy-backend-dev.yml)

### Data

Frontend/backend deploys do not upload city data automatically.
Parcel, PMTiles, and parking datasets must be uploaded explicitly to blob storage.

Dev blob targets:

- top-level parcel files: `parquets-dev/<file>`
- PMTiles files: `parquets-dev/<file>`
- parking files: `parquets-dev/parking/<file>`

After upload, validate live dev URLs:

- `https://dev.civicmapper.org/app.html?city=<city>`
- `https://dev.civicmapper.org/parking.html?city=<city>` if parking exists

## 16. Prod Deployment

### Code

Merge `develop` into `main`, then push `main`.

That triggers:

- [deploy-frontend-prod.yml](.github/workflows/deploy-frontend-prod.yml)
- [deploy-backend-prod.yml](.github/workflows/deploy-backend-prod.yml)

The prod frontend workflow already builds with:

- `VITE_API_BASE=https://api.civicmapper.org`
- `VITE_TILE_API_URL=https://tiles.civicmapper.org`
- `VITE_PARKING_ENABLED=true`

### Data

Promote or upload the validated dev artifacts to `parquets-prod`.

Required prod artifacts, depending on city:

- parcel parquet
- PMTiles
- PMTiles metadata
- parking parquet
- parking metadata

Validate live prod URLs:

- `https://civicmapper.org/app.html?city=<city>`
- `https://civicmapper.org/parking.html?city=<city>`

## 17. CORS And Data Access

Do not solve city-specific deployment issues by changing Azure Blob CORS for every new file.

Current model:

- local dev can use direct blob access or Vite proxy
- deployed apps should use the API proxy for data access
- parking datasets also go through the API proxy
- PMTiles metadata and files should also go through the API proxy in deployed environments

Relevant files:

- [viz/src/config.ts](viz/src/config.ts)
- [api/src/server.js](api/src/server.js)
- [viz/vite.config.ts](viz/vite.config.ts)
- [staticwebapp.config.json](viz/public/staticwebapp.config.json)

Rules:

- new city files should be referenced through existing `/data/...` and `/data/parking/...` API proxy routes in deployed environments
- if a deployment shows CORS errors, first confirm the app is using the proxy path instead of direct blob URLs
- if you change frontend/API origins, update API allowlists and CSP/connect-src as needed

For ordinary new-city additions, no new per-city CORS configuration should be required.

## 18. Final Release Checklist

- Source data identified and documented
- City restriction applied if source is larger than the city
- Duplicate/condo logic handled
- Exempt parcels excluded correctly
- Refined categories validated
- Canonical parquet exported in `EPSG:4326`
- City registered in `parquet_registry.py`
- City registry file `viz/src/cities/<city>.json` created (with `coords`)
- City dictionary `viz/src/dictionaries/<city>.json` added (auto-discovered)
- PMTiles generated for large cities
- Parking dataset generated if needed
- Dev blob artifacts uploaded
- Dev app validated
- Prod blob artifacts promoted/uploaded
- `develop` merged to `main`
- Prod app validated

## 19. Practical Warnings

- Never assume exempt logic from one city applies to another.
- Never assume ownership fields are clean.
- Never assume a city dataset is actually city-only.
- Large cities should bias toward PMTiles.
- Parking data is a separate deployment path from parcel data.
- A city can appear in the app but still fail if:
  - dictionary import is missing
  - PMTiles metadata was not uploaded
  - parking files were uploaded but `hasParkingData` is false
  - build-time env flags were not present in the deployed bundle

When in doubt, inspect the live deployed JS bundle and the live API responses, not just local code.
