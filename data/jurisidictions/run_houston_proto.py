#!/usr/bin/env python3
"""
Canonical Houston bake: full Harris County parcels tagged by municipal jurisdiction.

Instead of dropping everything outside the City of Houston, we keep parcels for ALL
jurisdictions and tag each one with the city it falls in. The frontend dims / hides the
interior enclave cities (West University Place, Bellaire, Southside Place, the Memorial
Villages) that are carved out of Houston's tax base. This replaced the old city-clipped
Houston bake; it is the dataset served as ?city=houston (houston-tx-parcels.*).

Scope: the FULL county (BBOX = None) → ~1.47M parcels, baked to PMTiles. Set BBOX to a
bounding box to iterate quickly on a subset via the browser GeoParquet path (no PMTiles bake).

Sources (both reachable from this env, both HCAD, same vintage -> guaranteed-aligned):
- Parcels (one-stop: geometry + land/bld/market values + state_class + area):
    https://www.gis.hctx.net/arcgis/rest/services/HCAD/Parcels/MapServer/0
- Municipal boundaries within Harris County (34 cities incl. all the enclaves):
    https://www.gis.hctx.net/arcgis/rest/services/HCAD/HCAD_Cities/MapServer/0

Outputs:
- data/jurisidictions/data/houston_proto/houston-tx-parcels.parquet
- data/jurisidictions/data/houston_proto/houston-harris-cities.geojson   (jurisdiction boundaries)

Notes:
- jurisdiction column: cleaned city name ("West University Place", "Houston", ...) by
  centroid-within HCAD_Cities; parcels in no city -> "Unincorporated Harris County".
- HCAD land_sqft is frequently 0 -> denominator falls back to StatedArea, then Acreage*43560,
  then geodesic polygon area. Emits area_source + likely_remnant like the other runners.
- state_class is Texas SPTB-style (A1=SF dominant) -> reuse the standard categorize().
- MapServer (not FeatureServer) but supports /query with f=geojson, paginated.
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from datetime import datetime
from pathlib import Path
from shapely.ops import unary_union
from pyproj import Geod
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "houston"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "houston-harris-proto-geometry.parquet"
CITIES_CACHE = DATA_DIR / "houston-harris-cities.geojson"
SHAPES_DIR = DATA_DIR / "Houston Shape Files"

# Drop Unincorporated Harris County (~48% of parcels) for performance — it was only ever shown
# de-emphasized, so the tax-base story is unchanged. See backup_full_harris_2026-06-16/.
DROP_UNINCORPORATED = True

# Extra region-grouping layers joined onto each parcel (centroid-within), in addition to the
# municipal `jurisdiction`. Each becomes a categorical column the bake tags per hex and the
# frontend offers as a switchable group. (geojson filename, source col, new col, value xform)
REGION_LAYERS = [
    ("City Council Districts.geojson", "DISTRICT", "council_district", lambda v: f"District {v}"),
    ("Super Neighborhoods.geojson", "SNBNAME", "super_neighborhood", lambda v: str(v)),
    ("Civic Clubs.geojson", "CivicName", "civic_club", lambda v: str(v)),
]
REGION_NONE = "(None)"

PARCELS_URL = "https://www.gis.hctx.net/arcgis/rest/services/HCAD/Parcels/MapServer/0/query"
CITIES_URL = "https://www.gis.hctx.net/arcgis/rest/services/HCAD/HCAD_Cities/MapServer/0/query"

# Full-county bulk sources. The paginated REST layer above is the one-stop (geometry+values)
# source and is fine for a bbox, but pulling all ~1.55M parcels means ~1,500 requests that
# the server throttles. For the whole county, prefer HCAD's bulk downloads — geometry from
# the GIS shapefile, joined on the account number to the appraisal-roll export:
#   GIS parcels (geometry + HCAD account `HCAD_NUM`):
#     https://download.hcad.org/data/GIS/Parcels.zip                       (~201 MB)
#   Real-property roll (state class + land/bld/market values + owner), current year — the zip
#   contains `real_acct.txt` (tab-delimited):
#     https://download.hcad.org/data/CAMA/2026/Real_acct_owner.zip         (~210 MB)
# Both are public, direct-download URLs (no form). If the automated fetch is blocked, drop the
# two zips into DATA_DIR under the BULK_*_ZIP names below and re-run.
BULK_GIS_URL = "https://download.hcad.org/data/GIS/Parcels.zip"
BULK_ROLL_URL = "https://download.hcad.org/data/CAMA/2026/Real_acct_owner.zip"
BULK_GIS_ZIP = DATA_DIR / "hcad-parcels-gis.zip"
BULK_ROLL_ZIP = DATA_DIR / "hcad-real-acct-owner.zip"

# Full Harris County (~1.55M parcels). Set to a "minx,miny,maxx,maxy" string to scope to a
# bbox (e.g. the west-central enclave cluster "-95.50,29.69,-95.40,29.76") for fast iteration.
BBOX = None

PARCEL_FIELDS = ("acct_num,state_class,land_use,land_value,bld_value,impr_value,"
                 "total_appraised_val,total_market_val,land_sqft,StatedArea,Acreage,"
                 "owner_name_1,site_city,CONDO_FLAG")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 1000  # HCAD MapServer maxRecordCount is 1000; requesting more silently returns 1000
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bbox_params():
    if not BBOX:
        return {"where": "1=1"}
    return {"where": "1=1", "geometry": BBOX, "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects"}


def fetch_page_bytes(url, base, off, out_fields):
    """Fetch ONE paginated GeoJSON page (resultOffset=off) as raw bytes, with retry/backoff.
    Only the HTTP call runs here — GDAL/pyogrio parsing is NOT thread-safe (concurrent
    /vsimem reads collide), so parsing happens back in the main thread."""
    last = None
    for attempt in range(5):
        try:
            r = requests.get(url, params={**base, "outFields": out_fields, "returnGeometry": "true",
                             "resultOffset": off, "resultRecordCount": PAGE,
                             "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=180)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"page @{off} failed after retries: {last}")


def fetch_paginated(url, base, out_fields="*", workers=12):
    """Parallel paginated GeoJSON pull (HCAD supports pagination; maxRecordCount=1000). The
    sequential pull is ~3.5h for the full county, so fan out `workers` concurrent HTTP
    requests (~15-20 min). Parsing is done in THIS (main) thread — GDAL isn't thread-safe."""
    cnt = requests.get(url, params={**base, "returnCountOnly": "true", "f": "json"},
                       headers=HEADERS, timeout=120).json().get("count", 0)
    offsets = list(range(0, cnt, PAGE))
    log(f"  pulling {cnt:,} features in {len(offsets)} pages ({workers}-way parallel HTTP)...")
    pages = [None] * len(offsets)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_page_bytes, url, base, off, out_fields): i for i, off in enumerate(offsets)}
        for fut in as_completed(futs):
            content = fut.result()                      # bytes from a worker thread
            g = gpd.read_file(io.BytesIO(content))      # parse in the main thread (GDAL-safe)
            pages[futs[fut]] = g if len(g) else None
            done += 1
            if done % 100 == 0 or done == len(offsets):
                log(f"    fetched+parsed {done}/{len(offsets)} pages")
    pages = [p for p in pages if p is not None]
    return gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")


# -- 1. Municipal boundaries (cities within Harris County) --------------------
def fetch_cities():
    if CITIES_CACHE.exists():
        log(f"Using cached city boundaries: {CITIES_CACHE.name}")
        return gpd.read_file(CITIES_CACHE)
    log("Pulling HCAD_Cities boundaries...")
    r = requests.get(CITIES_URL, params={"where": "1=1", "outFields": "name,code",
                     "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    cities = gpd.read_file(io.BytesIO(r.content)).to_crs("EPSG:4326")
    # "CITY OF WEST UNIVERSITY PLACE" -> "West University Place"
    cities["jurisdiction"] = (cities["name"].astype(str).str.replace(r"^CITY OF ", "", regex=True)
                              .str.title().str.strip())
    cities = cities[["jurisdiction", "code", "geometry"]]
    cities.to_file(CITIES_CACHE, driver="GeoJSON")
    log(f"  cached {len(cities)} city boundaries -> {CITIES_CACHE.name}")
    return cities


# -- 2. Parcels (cached) ------------------------------------------------------
def norm_acct(s):
    return s.astype(str).str.strip().str.upper()


def ensure_bulk_files():
    """Make sure the two HCAD bulk zips are present, downloading if missing."""
    for z, url in ((BULK_GIS_ZIP, BULK_GIS_URL), (BULK_ROLL_ZIP, BULK_ROLL_URL)):
        if z.exists() and z.stat().st_size > 1_000_000:
            continue
        log(f"Downloading {url} -> {z.name} (large; one-time)...")
        try:
            with requests.get(url, headers=HEADERS, timeout=900, stream=True) as r:
                r.raise_for_status()
                with open(z, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as e:
            raise SystemExit(
                f"\nCould not download {url} ({e}).\nManually download it and place it at "
                f"{z}, then re-run.\n")


def read_bulk_parcels():
    """Full-county parcels from HCAD's bulk downloads: geometry from the GIS shapefile joined
    on the account (HCAD_NUM == real_acct.acct) to the appraisal-roll export. Produces the
    same canonical columns the REST path does so the rest of the ETL is unchanged."""
    ensure_bulk_files()
    log("Reading parcel geometry from GIS shapefile (EPSG:2278 -> 4326)...")
    shp = f"/vsizip/{BULK_GIS_ZIP.as_posix()}/Parcels/Parcels.shp"
    geom = gpd.read_file(shp, columns=["HCAD_NUM", "StatedArea", "Acreage"])
    geom = geom.to_crs("EPSG:4326")
    geom["acct"] = norm_acct(geom["HCAD_NUM"])
    geom = geom[geom["acct"].str.len() > 0]
    log(f"  {len(geom):,} parcel geometries")

    log("Reading appraisal roll real_acct.txt (state class + values + owner)...")
    NEED = ["acct", "state_class", "land_val", "bld_val", "x_features_val",
            "tot_appr_val", "tot_mkt_val", "land_ar", "mailto"]
    with zipfile.ZipFile(BULK_ROLL_ZIP) as z, z.open("real_acct.txt") as fh:
        roll = pd.read_csv(fh, sep="\t", usecols=NEED, dtype=str,
                           encoding="latin-1", on_bad_lines="skip")
    roll["acct"] = norm_acct(roll["acct"])
    for c in ["land_val", "bld_val", "x_features_val", "tot_appr_val", "tot_mkt_val", "land_ar"]:
        roll[c] = pd.to_numeric(roll[c], errors="coerce")
    roll = roll.drop_duplicates("acct")
    log(f"  {len(roll):,} appraisal rows")

    parcel = geom.merge(roll, on="acct", how="left")
    matched = int(parcel["tot_mkt_val"].notna().sum())
    log(f"  joined: {matched:,}/{len(parcel):,} matched values "
        f"({100*matched/max(len(parcel),1):.1f}%)")
    # Map to the canonical names the rest of the ETL (sections 3-7) expects.
    parcel["acct_num"] = parcel["acct"]
    parcel["land_value"] = parcel["land_val"]
    parcel["bld_value"] = parcel["bld_val"]
    parcel["impr_value"] = parcel["bld_val"].fillna(0) + parcel["x_features_val"].fillna(0)
    parcel["total_appraised_val"] = parcel["tot_appr_val"]
    parcel["total_market_val"] = parcel["tot_mkt_val"]
    parcel["land_sqft"] = parcel["land_ar"]
    parcel["owner_name_1"] = parcel["mailto"]
    return parcel


def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached parcels: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    if BBOX:
        log("Pulling HCAD parcels via REST (bbox)...")
        gdf = fetch_paginated(PARCELS_URL, bbox_params(), out_fields=PARCEL_FIELDS)
    else:
        gdf = read_bulk_parcels()
    gdf.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached parcels -> {GEOM_CACHE.name} ({len(gdf):,} rows)")
    return gdf


cities = fetch_cities()
parcel = fetch_parcels()
log(f"Raw parcels: {len(parcel):,}")

# normalize CRS
if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
# Vectorized validity fix (per-row .apply is far too slow at 1.5M parcels).
parcel = parcel[parcel.geometry.notnull()].copy()
invalid = ~parcel.geometry.is_valid
if invalid.any():
    parcel.loc[invalid, "geometry"] = parcel.loc[invalid, "geometry"].buffer(0)
parcel = parcel[parcel.geometry.notnull() & parcel.geometry.is_valid].copy()

# -- 3. Tag jurisdiction by centroid-within city ------------------------------
log("Tagging jurisdiction by centroid-within-city...")
cent = parcel.geometry.to_crs(3857).centroid.to_crs(4326)
pts = gpd.GeoDataFrame(parcel.drop(columns="geometry").copy(), geometry=cent, crs="EPSG:4326")
tagged = gpd.sjoin(pts, cities[["jurisdiction", "geometry"]], how="left", predicate="within")
# a centroid can theoretically match >1 (overlapping boundary slivers); keep first
tagged = tagged[~tagged.index.duplicated(keep="first")]
parcel["jurisdiction"] = tagged["jurisdiction"].fillna("Unincorporated Harris County").values
log(f"  jurisdiction counts:\n{parcel['jurisdiction'].value_counts().to_string()}")

# -- 3b. Tag extra region groups (council district / super neighborhood / civic club) ---------
# Reuse the same centroid points; mirrors augment_houston_regions.py so re-running the ETL and
# the one-off augmentation produce the same columns.
for fname, src_col, dst_col, transform in REGION_LAYERS:
    path = SHAPES_DIR / fname
    if not path.exists():
        log(f"  WARN region layer missing, skipping {dst_col}: {path}")
        parcel[dst_col] = REGION_NONE
        continue
    layer = gpd.read_file(path)
    # Some files declare EPSG:4326 but actually carry Web-Mercator coords (e.g. Civic Clubs) —
    # detect by bounds so to_crs(4326) isn't a silent no-op that matches nothing.
    minx, miny, maxx, maxy = layer.total_bounds
    if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
        layer = layer.set_crs("EPSG:3857", allow_override=True)
    elif layer.crs is None:
        layer = layer.set_crs("EPSG:4326")
    layer = layer.to_crs("EPSG:4326")
    layer = layer[layer.geometry.notnull()].copy()
    inv = ~layer.geometry.is_valid
    if inv.any():
        layer.loc[inv, "geometry"] = layer.loc[inv, "geometry"].buffer(0)  # repair self-intersections
    layer = layer[layer.geometry.notnull() & layer.geometry.is_valid][[src_col, "geometry"]]
    layer = layer.rename(columns={src_col: dst_col})
    rt = gpd.sjoin(pts, layer, how="left", predicate="within")
    rt = rt[~rt.index.duplicated(keep="first")]
    parcel[dst_col] = rt[dst_col].map(lambda v: transform(v) if pd.notna(v) else REGION_NONE).values
    log(f"  {dst_col}: {(parcel[dst_col] != REGION_NONE).sum():,} matched, "
        f"{(parcel[dst_col] == REGION_NONE).sum():,} -> {REGION_NONE}")

if DROP_UNINCORPORATED:
    before = len(parcel)
    parcel = parcel[parcel["jurisdiction"] != "Unincorporated Harris County"].copy()
    log(f"  dropped Unincorporated Harris County: {before:,} -> {len(parcel):,}")

# -- 4. Canonical value / class fields ----------------------------------------
for c in ["land_value", "bld_value", "impr_value", "total_appraised_val", "total_market_val",
          "land_sqft", "StatedArea", "Acreage"]:
    parcel[c] = pd.to_numeric(parcel.get(c), errors="coerce")
parcel["land_value"] = parcel["land_value"].fillna(0)
# improvement: prefer impr_value (total improvements), fall back to bld_value (building only)
parcel["improvement_value"] = parcel["impr_value"].where(parcel["impr_value"].fillna(0) > 0,
                                                         parcel["bld_value"]).fillna(0)
parcel["full_market_value"] = (parcel["total_market_val"]
                               .where(parcel["total_market_val"].fillna(0) > 0,
                                      parcel["total_appraised_val"])
                               .where(lambda s: s.fillna(0) > 0,
                                      parcel["land_value"] + parcel["improvement_value"]))
parcel["state_class"] = parcel["state_class"].astype(str).str.strip().str.upper()
parcel["mailto"] = parcel.get("owner_name_1", "")


def categorize(v):
    """Texas SPTB state class -> coarse category (same crosswalk as the TX runners)."""
    raw = str(v or "").strip().upper()
    if not raw or raw == "NAN":
        return "Other"
    if raw.startswith("X"):
        return "Exempt"
    if raw.startswith("A"):
        return "Single Family"
    if raw.startswith("B"):
        return "Multifamily"
    if raw.startswith("C"):
        return "Vacant Residential"
    if raw.startswith("D") or raw.startswith("E"):
        return "Agricultural / Rural"
    if raw.startswith("F"):
        return "Industrial" if raw.startswith("F2") else "Commercial"
    if raw.startswith("G"):
        return "Mineral / Oil & Gas"
    if raw.startswith("J") or raw.startswith("U"):
        return "Utility"
    if raw[0] in ("L", "M", "N", "O", "S"):
        return "Personal Property / Inventory"
    return "Other"


parcel["property_land_use_category"] = parcel["state_class"].apply(categorize)

# -- 5. Exempt filter ---------------------------------------------------------
ex = parcel.copy()
ebs = ex["property_land_use_category"].isin(["Exempt"])
KW = ["CITY OF HOUSTON", "HARRIS COUNTY", "STATE OF TEXAS", "HOUSTON ISD", "HOUSTON IND SCH",
      "UNITED STATES", "US GOVT", "HOUSING AUTHORITY", "METRO", "UNIVERSITY OF",
      "TEXAS DEPT", "HARRIS CO"]
eown = ex["mailto"].astype(str).str.upper().str.contains("|".join(KW), na=False)
ex["exemption_flag"] = (ebs | eown).astype(int)
ex = ex[ex["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["property_land_use_category"]
ex = ex[~ex["property_land_use_category"].isin(
    {"Mineral / Oil & Gas", "Personal Property / Inventory"})].copy()
ex["land_value"] = pd.to_numeric(ex.get("land_value", np.nan), errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(ex, fetch_footprints=False)
log(f"After exempt/refine -> {len(ex):,}")


# -- 6. Area denominator (land_sqft -> StatedArea -> Acreage -> projected GIS area) ------
# Vectorized geometry area via the source CRS EPSG:2278 (NAD83 / Texas South Central, US
# survey feet) — projected area is already square feet, and this is ~1000x faster than a
# per-polygon geodesic .apply at full-county scale (and the fallback only matters for the
# few parcels lacking a reported land_ar).
log("Computing GIS areas (vectorized, EPSG:2278 ft)...")
ex["gis_area_sqft"] = ex.geometry.to_crs(2278).area
ex.loc[ex["gis_area_sqft"] < 1, "gis_area_sqft"] = np.nan

reported = ex["land_sqft"].where(ex["land_sqft"] > 0)
reported = reported.fillna(ex["StatedArea"].where(ex["StatedArea"] > 0))
reported = reported.fillna((ex["Acreage"] * SQFT_PER_ACRE).where(ex["Acreage"] > 0))
ex["reported_sqft"] = reported
use_reported = ex["reported_sqft"] > 0
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["gis_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value"] = pd.to_numeric(ex["full_market_value"], errors="coerce")
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
ex["link"] = "https://public.hcad.org/records/details.asp?crypt=" + ex.get("acct_num", "").astype(str)

# -- 7. Export ----------------------------------------------------------------
COLUMNS = ["geometry", "jurisdiction", "council_district", "super_neighborhood", "civic_club",
           "exemption_flag", "property_land_use_category",
           "property_land_use_refined", "full_market_value", "full_market_value_per_sqft",
           "land_value", "land_value_per_sqft", "improvement_value", "improvement_value_per_sqft",
           "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link",
           "land_area_acres", "area_source", "likely_remnant"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
out = DATA_DIR / "houston-tx-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"houston-tx-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"jurisdiction: {final['jurisdiction'].value_counts().to_dict()}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
