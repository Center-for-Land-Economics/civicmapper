---
name: add-city
description: >-
  Use when adding a new city/jurisdiction to CivicMapper (the geovizwiz-sb repo) —
  parcel ETL, PMTiles bake, surface parking, and frontend registration. This captures
  the hard-won gotchas that are NOT in docs/add-city-playbook.md (the step-by-step
  procedure). Read both. Especially relevant for US county/appraisal-district parcel
  data, Texas SPTB cities, and anything involving dollar-value joins, H3 hex tiles,
  or the parquets-dev blob.
---

# Adding a City to CivicMapper — Lessons & Gotchas

`docs/add-city-playbook.md` is the canonical procedure. This file is the field notes:
the non-obvious things that cost real time. Pair them. Recent worked examples to copy:
`data/jurisidictions/run_dallas.py`, `run_sanantonio.py`, `run_bcs.py`.

## 1. Data sourcing

- **Appraisal-district / county GIS hosts are frequently firewalled from this
  environment** (seen: `maps.bexar.org`, `maps.bcad.org`, `gis.lojic.org` — DNS resolves
  but TCP times out, from both Python `requests` AND WebFetch). Don't fight it. **Find a
  hosted ArcGIS Online mirror** on `services*.arcgis.com` via the AGOL search API:
  `https://www.arcgis.com/sharing/rest/search?q=<county> parcels&f=json`. One publisher
  org often hosts many counties (e.g. `82iS1Pc7dgs3LFZv` hosts both Bexar and Brazos
  "<county> parcels" layers with the same schema).
- **Always probe endpoints live before writing the ETL.** Web-search/WebFetch summaries of
  layer fields are routinely wrong or incomplete — the WebFetch summary of DallasTaxParcels
  missed `SPTBCODE`, `TOTEXEMPT`, `Website`, etc. A 10-line `requests` probe (layer `?f=json`
  for fields, `returnCountOnly`, a 3-row sample, distinct values of the class field) saves
  an entire wrong ETL.
- **Geometry and dollar values are usually in different places.** Three patterns seen:
  - *One-stop* (best): hosted layer has geometry + `LAND_VALUE/IMP_VALUE/MKT_VALUE` + state
    class + area (San Antonio `Bexar_parcels_all`, Bryan/CS `Brazos_County_Parcels`). Fully
    automated, no download.
  - *Geometry + separate appraisal export*: GIS layer has class/owner/area but no values;
    values come from a manually-downloaded appraisal roll joined on account id (Dallas DCAD
    zip, Austin TCAD `PROP.TXT`). Mirror the manual-download header comment in run_austin.py.
  - *Values not in open data at all*: Kentucky PVAs (Louisville) — the public parcel layer
    has zero value fields and values need a PVA records request. **Surface this to the user
    as a blocker; don't scrape.** (See memory: louisville-blocked-on-pva-values.)
- **Pull geometry with `f=geojson`** (clean CRS + multipolygon/holes) paginated at 2000/page,
  retry per page, and **cache to `<city>-<state>-geometry.parquet`** so re-runs skip the
  multi-minute download.

## 2. The dedup N× value-inflation bug (the expensive one)

A single appraisal account is often split into **many GIS polygons** (corridors, multi-part
parcels — the UP railroad in Dallas was 52). A left join broadcasts the account's single
value onto every polygon. If dedup then **sums** the value across an account's rows, you get
**polygon-count × the real value** (Dallas railroad showed $4.30B = 52 × $82.7M). It hit 249
Dallas accounts. Symptom: a few parcels with absurd values/$-per-sqft.

**Rule for the group-by-account dedup:**
- value columns (`tot_appr_val`, `land_val`, `bld_val`) → **`first`** (account-level, broadcast
  identically — never sum).
- per-polygon GIS area → **`sum`**; reported account-level area → **`first`**.
- geometry → `unary_union`.

## 3. Sliver remnants → meaningless $/sqft spikes

Tiny fragment polygons (<500 sqft) carry a real account value → astronomical $/sqft. Flag
them in the ETL: `likely_remnant = (land_area_sqft < 500)`. Two-layer fix:
- **Detail layer**: frontend `hideRemnants: true` in `cities.ts` filters `likely_remnant != 1`
  (centralized in `setParcelLayerFilter`, no re-bake needed).
- **Hex layer**: bake with `--drop-remnants` so the H3 aggregate drops them too. **This needs
  a re-bake.** The detail-layer filter alone does NOT clean the hexes.
- **A citywide average will hide local hex spikes.** When a user says they see distortion in
  the hexes, believe them — a single high-$/sqft sliver in a small/sparse hex dominates that
  hex's area-weighted value even though it's invisible in the citywide mean.

> **DEFERRED / future UX idea (flagged 2026-07-20, not yet scheduled — do not build without
> asking):** Right now the only tools for implausible assessor records are *hide* (remnant
> filter / `hideRemnants`) and *flag as erroneous* (the `gp-error` layer, driven by
> `REALLANDVA <= 0`). Both are blunt: they suppress or red-outline, but never *explain*. We
> should add a UX affordance that surfaces "anomalous" assessments to the user rather than
> silently dropping them — e.g. an "anomalies" toggle/panel, a popup annotation ("tiny parcel
> carrying a large value — $/sqft unreliable"), or a per-parcel confidence badge. Motivating
> case: Hartford's `0 OLIVE ST` stub — a genuine assessor record of a ~305 sqft fragment
> carrying $360,400 land value ($1,180/sqft vs a ~$2/sqft industrial block), i.e. value that
> economically belongs to a ~3–4 acre holding but is attached to a sliver polygon. Not a bug on
> our side (both the city GIS `TotAcreage` and CT CAMA `Land_Acres` agree it's ~305 sqft), so
> "fix the area" isn't an option — the honest move is to *communicate* the anomaly. Scope this
> as an app-wide feature (condos with `land_value=0` are the same class of problem), not per-city.

## 4. City-limits clipping

- **Clip on the authoritative boundary** (centroid-within), not on a situs-city or owner field.
  `addr_city`/mailing city is the OWNER's address and is unreliable for jurisdiction.
- **Multi-city units**: union the boundaries (Bryan/College Station = `ID IN (...)` → `unary_union`,
  one city entry covering both, parcels in either included).
- **Multi-county cities**: the city may span counties but appraisal data is per-county
  (City of Dallas spans 5 counties; DCAD only covers Dallas County → keep the Dallas-County
  portion, document the dropped slivers).

## 5. Classification

- **Texas SPTB state codes are statewide**, so one `categorize()` (prefix-based: A=SF, B=MF,
  C=vacant, D/E=ag, F1=commercial/F2=industrial, G=mineral, J=utility, L/M/N/O/S=personal,
  X=exempt) works for every TX city — copy it from run_dallas/run_sanantonio/run_bcs.
- **Non-Texas needs its own scheme.** Don't assume. Louisville/Jefferson KY uses
  `ClassGroup`/`PropClass` codes, not SPTB.
- **Exemptions**: prefer an authoritative flag (DCAD `TOTAL_EXEMPTION`, BCAD `Exemptions`
  starting `EX-`). Where there's no flag, fall back to state-class `X*` + an owner-keyword
  heuristic (city/county/state/ISD/university). Exclude `Utility`, `Mineral`, and
  `Personal Property/Inventory` categories from the shipped set.

## 6. Condos & multi-record parcels (a documented trap)

Condos are the classic parcel-data landmine — see `docs/add-city-playbook.md` §5 and the
"Fort Collins lesson". Failure modes:
- Per-unit tax records **stacked on one footprint** (N overlapping parcels at one spot).
- Condo units sitting on a **shared ground / common-area parcel** that a naive exempt filter
  strips out → unit parcels "floating above missing land in 3D."

Handling already in the repo:
- `run_fort_collins.py` — the sophisticated version: common-area/association detection,
  `CONDO_PARENT_MIN_RATIO`, condo-category merge. Copy from here for condo-heavy assessor feeds.
- `run_baltimore.py` — the reference sum-values / first-categoricals / union-geometry dedup.

**Account-level dedup alone does NOT handle condos** — condo units have *separate* accounts,
so `groupby(account)` won't merge them. Whether you need explicit condo logic depends on the
source: **diagnose it.** Count parcels sharing a footprint/centroid in the output:
```python
c = g.geometry.representative_point()
vc = (c.x.round(5).astype(str)+","+c.y.round(5).astype(str)).value_counts()
print("stacked clusters:", int((vc>1).sum()), "max stack:", int(vc.max()))
```
The hosted StratMap/PACS layers used for Dallas/San Antonio/Bryan-CS return **one footprint per
condo building** (verified: ~0 stacked clusters), so no condo collapse was needed. County
assessor feeds with true per-unit polygons (Fort Collins-style) WILL stack and need collapsing.
When you do collapse, **preserve common-area/parent parcels through the exempt filter first**,
then merge — drop leftover exempt parcels only after the ground parcel is formed.

### 6a. Smoke alarms: signals parcels are NOT properly treated (learned on Olympia, 2026-06)

These are the tells that a feed maps units as placeholder stubs and you have NOT merged them.
**Run these on the FINAL parquet before baking/uploading.** Any trip ⇒ stop and treat condos.
The reference implementation that handles every item below is
[run_olympia.py](data/jurisidictions/run_olympia.py) (Thurston County) — copy its
common-area capture + stub-merge block for any condo/suite/mobile-home-heavy assessor feed.

1. **Thin tall "pencils/slivers" in the 3D view** — the #1 visual tell. A tiny footprint with a
   real value extrudes into a needle. = unit placeholders (condos, office/retail/medical suites,
   mobile homes) mapped as ~100–400 sqft stub squares.
2. **`land_value_per_sqft` max/p99 wildly above neighbors** — compare to nearby *non-stub* parcels.
   Real Olympia downtown land was $40–164/sqft; stubs computed to $900–14,750/sqft. A 5–50× gap ⇒
   the area denominator is a stub footprint, not real land. ALWAYS calibrate against neighbors.
3. **Many sub-1000 sqft footprints carrying value** — `(geom_sqft < 1000) & (value > 0)`. Real lots
   jump to >2000 sqft; a spike at a fixed tiny size (100/168/224/400 sqft) is generated placeholders.
   A *fixed* size threshold misses the larger placeholder sizes — key off **effective land area**.
4. **Stated area (`TOTAL_ACRES`) == 0, or a tiny share, on valued parcels** — units carry value but
   little/no land; real parcels always carry a real stated acreage.
5. **Interior rings (holes) in a parcel polygon** — the assessor's common-area polygon is the
   development land with each unit footprint *punched out*. Merging onto it inherits the holes.
6. **A development rendering as several adjacent pencils** — perimeter-arranged units fragmented by
   a too-small clustering buffer (Black Lake's 13 units span 67×71 m; a 10 m buffer split them 3 ways).
7. **A short numeric `PARCEL_NO` parcel with NULL attributes** under/among a stub cluster — that IS
   the development's common-area LAND parcel (its number is the plat root the units share).

Diagnostic to drop into the ETL (projected CRS, e.g. the city's UTM zone):
```python
gp = final.to_crs(utm_epsg); a = gp.geometry.area * 10.7639          # footprint sqft
lv = pd.to_numeric(final["land_value"], errors="coerce") / final["land_area_sqft"]
print("footprint sqft p1/p5/p10:", [round(a.quantile(q)) for q in (.01,.05,.10)])
print("sub-500 / sub-1000 sqft:", int((a<500).sum()), int((a<1000).sum()))
print("land $/sqft p50/p99/max:", round(lv.median()), round(lv.quantile(.99)), round(lv.max()))
holes = final.geometry.apply(lambda g: 0 if g is None else
    sum(len(p.interiors) for p in (g.geoms if g.geom_type=="MultiPolygon" else [g])))
print("parcels with holes:", int((holes>0).sum()))
```

### 6b. The fix — merge units DOWN onto their real land parcel (gets condos right first try)

The structure: a feed maps each unit as a tiny stub carrying value; the development's **real land is
a separate parcel** — usually a **common-area parcel** (true footprint, NULL assessor attributes,
`PARCEL_NO` = the plat root that *prefixes* its units, e.g. units `35340000800` → common `3534`),
sometimes a **valued parent** with units carved out as holes (those holes are legit — real fee-simple
lots like townhomes — do NOT fill them).

1. **Capture common-area land parcels FIRST** (before any active/STATUS or exempt filter drops them):
   short numeric `PARCEL_NO` + null PROP_TYPE + null value; dissolve multi-polygon plats by number.
2. **Detect unit stubs**: `geom_sqft < 1000` AND (the account **prefix-matches a common plat** OR
   (it has a building AND **effective land area** < ~3000 sqft)). Effective land = stated acreage if
   present else geom — this is what inflates $/sqft, so it catches every placeholder size, zero-share
   AND small-share units; the plat-membership test additionally catches vacant units (no building) and
   units whose stated share exceeds the size cap.
3. **Split**: `land_value > 0` → MERGE; `land_value == 0` → DROP (mobile homes / personal property on
   leased land — the park land, e.g. Thurston `PROP_TYPE='PRK'`, is its own real parcel; don't confuse
   `PRK`=park with parking).
4. **Merge each development DOWN onto its common-area parcel** (longest `PARCEL_NO`-prefix match): sum
   land+building value onto the real footprint, and **fill the common parcel's holes** (keep the solid
   exterior — the holes are punched-out unit footprints that belong to the development). For the ~1%
   with no common parcel, fall back to a building-snapped convex hull (stubs ∪ overlapping Overture
   buildings) with a min-width floor so a collinear stub row isn't a thin wall.
5. **Re-calibrate**: merged developments' land $/sqft must land in the neighbors' range; if not, the
   footprint or the value-summing is wrong. Then **recompute any derived datasets** (surface parking
   spatial-joins to parcels; the underutilized/`parcels_under` layer) so they pick up the merged land.

Olympia result: 806 unit stubs → 94 development parcels (99% onto real common-area land); rendered
land $/sqft max 14,750 → 155; condos render as proper blocks instead of a forest of pencils.

## 7. Frontend + deploy

- **`upload_city_dev.py <city>` is the consolidated, registry-driven uploader** — it pushes
  parcel parquet + PMTiles + metadata + parking (whatever exists locally). Do NOT write
  bespoke `upload_<city>_dev.py` scripts (the old ones were deleted).
- **`utils.dictionary.ts` auto-loads `dictionaries/<key>.json` via `import.meta.glob`** — no
  manual switch to edit (the playbook is stale on this). Just create the JSON.
- **Local dev serves tiles straight from the `parquets-dev` blob** (no `VITE_PMTILES_BASE_URL`).
  So: nothing shows on localhost until artifacts are uploaded, AND `config.ts` derives
  `PARKING_DATASET_URL`/`PMTILES` URLs at module load from the city registry → **restart
  `npm run dev` (not just HMR) after adding/editing `src/cities/<key>.json`** (the registry
  glob map is built at server start, same as dictionaries), or e.g. a newly-added
  `parkingFilename` stays null.
- **A brand-new `dictionaries/<key>.json` also needs a full `npm run dev` restart.**
  `utils.dictionary.ts` builds its `import.meta.glob('./dictionaries/*.json')` map at server
  start; a dictionary file created *after* the server is running is absent from that map, so
  `loadDataDictionary()` early-returns, `CITY_CONFIG` stays null, and **`cityUsesPmtiles()` is
  false** → the app silently falls back to the raw-parquet detail path and **never adds the H3
  `parcels_low` hex layer**. Symptom: detail parcels render when zoomed in (loaded from the
  `.parquet`) but **the low-zoom hexes are missing** — looks like a bake/data bug but is just a
  stale glob. Restart the dev server.
- **Bump `pmtilesVersion` (and `parkingVersion`) on every re-bake/re-upload.** The blob/CDN
  serves stale byte-ranges otherwise → corrupt/missing tiles even when the file is correct.
  The version is a cache-bust query param.
- Registration touchpoints for a new city: `data/parquet_registry.py`,
  `viz/src/cities/<key>.json` (CityDef fields + REQUIRED coords [lng, lat]; auto-discovered
  via import.meta.glob — cities.ts is now just the loader), `viz/src/dictionaries/<key>.json`
  (picker card is generated automatically — cities.html needs no edit),
  `scripts/smoke_dev_cities.py` list, and `CITY_OSM_QUERIES` in
  `data/scripts/parking_lot_extraction.py`.

## 8. Environment quirks

- **Windows console is cp1252** — scripts that `print` emoji (✅) crash with
  `UnicodeEncodeError`. Run Python with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.
- **Parking has two modes — OSM-only (fast) and the real NAIP+ML satellite pipeline.** All
  *shipped* cities were `--osm-only`, which badly undercounts parking outside well-mapped
  downtowns. The ML path now works (NIR veg-strip + road removal + a parcel-context FP flag,
  all default-on) and finds the missing suburban/commercial parking. To run it set up the
  `data/.venv` (Python 3.12, CUDA torch, `transformers==4.46.3`, GDAL via cgohlke `osgeo`
  wheel) per `data/requirements-parking.txt` (the satellite pipeline's full setup notes are
  maintained in the maintainers' internal docs)
  — it has the Windows gotchas (the osgeo c-ares DNS blocker + rasterio fetch workaround), the
  area-CRS fix, and the tile/cost plan for citywide/all-cities runs. Classification + the
  parcel-context flag also work on OSM-only data (Overture buildings via DuckDB).
- **OSM Overpass fetches hang intermittently** (the fuel-station step especially: 180s
  timeouts × retries × cells → can spin 70+ min). It's transient — kill and re-run usually
  clears it first try. Set a Monitor on the log for `fuel features|Done!|Bailing|Traceback`
  to catch the outcome fast instead of waiting.
- **PMTiles bake on Windows uses WSL**: `--wsl` (tippecanoe + pmtiles live in WSL). Verify a
  city's PMTiles contains all three expected layers with
  `wsl bash -c "pmtiles show --metadata <file>"` (expect `parcels` z13-14, `parcels_low` z0-12,
  and `parcels_under` z2-14). Note `wsl pmtiles show /mnt/c/...` from Git-bash mangles the path —
  wrap it in `wsl bash -c "..."`.
- **The "Vacant & Underdeveloped" tab renders a dedicated `parcels_under` layer, NOT the hexes.**
  That tab only ever shows the underutilized subset (Vacant / Parking Lot / Underdeveloped), which
  is a small fraction of parcels (Detroit: ~80k of ~300k). The full parcel layer can't tile below
  z13 (drop-densest deletes ~98%, no-drop coalescing stalls tippecanoe) — but the *subset* tiles
  cleanly across ALL zooms, so the bake emits it as `parcels_under` (z2-14, `--drop-densest-as-needed`)
  and writes `"underutilizedSourceLayer":"parcels_under"` into the metadata. The frontend reads that
  key to point the under tab at it; cities baked before this fall back to the z13+ `parcels` layer
  (under tab empty until you zoom in) → **re-bake to fix**. An H3 hex summary for this tab was tried
  and removed — the aggregate can't honour the category filter and a combined-% hex was unhelpful.
- **H3 is the standard low-zoom layer and is ON by default.** Any city large enough for PMTiles
  (rule of thumb **~100k+ parcels**, or when the browser GeoParquet path is sluggish/unstable)
  bakes its `parcels_low` aggregate as **H3 hexes** — `parquet_to_pmtiles.py` defaults to H3, so
  just run it; `--no-h3` is a legacy escape hatch for reproducing old square-grid bakes, not for
  new cities. (Historically `--h3` was opt-in; the default was flipped on 2026-06-10.) Pair with
  `--drop-remnants` whenever the city sets `hideRemnants:true` so the hexes match the filtered
  detail layer. Tune zoom bands via `H3_RESOLUTIONS`/`H3_ZOOM_BANDS` in the script.

## 9. Popups: derive, don't bake

Land size / building size in the popup are **derived on demand** as `value ÷ value-per-sqft`
(see `buildPopupHTML` in `main.ts`) — both are already in the tile props for every city. Don't
add data columns or re-bake for values you can compute client-side.
