#!/usr/bin/env python3
"""
Build Detroit's canonical parcel parquet from two City of Detroit open-data layers.

Sources (both published by the City of Detroit assessor org on ArcGIS Online,
services2.arcgis.com/qvkbeam7Wirps6zC — the data.detroitmi.gov hub):

- Geometry + class + use + lot area + tax status + address:
  "Parcels (Current)"  parcel_file_current/FeatureServer/0   (~378k polygons)
  This file is ALREADY the City-of-Detroit parcel file (enclaves Highland Park /
  Hamtramck have their own assessors and are not in it), so NO boundary clip is
  needed — it is inherently city-only (cf. Baltimore / Cleveland city-only sources).

- Assessed / true-cash / land values:
  "Tentative Assessment Roll 2026"  tentative_assessment_roll_2026/FeatureServer/0
  This table has NO geometry; we join it to the parcel geometry on `parcel_id`.
  It carries the three value fields the Parcels layer is missing:
    amt_estimated_true_cash_value  -> full market value (TCV)
    amt_land_value                 -> land value, *on the same TCV basis*
                                      (verified: for VACANT parcels land == TCV)
    amt_assessed_value             -> State Equalized / assessed value (50%-basis)
  Improvement (building) value is therefore TCV - land_value.

Michigan property_class crosswalk (statewide):
  2xx = Commercial, 3xx = Industrial, 4xx = Residential;
  x01 improved, x02 vacant, x03 common-element/assessed-with-others, x07 condominium.
  Residential improved/condo is split Single Family vs Multifamily via
  use_code_description (DUPLEX / *FAMILY / APT* / FLAT / ROW HOUSE / COOPERATIVE).

Detroit is large (~378k parcels) -> ship PMTiles. Bake + upload separately:
    python data/scripts/parquet_to_pmtiles.py --city detroit --h3 --wsl --upload --overwrite

Outputs:
- data/jurisidictions/data/detroit/detroit-mi-parcels.parquet
- data/jurisidictions/data/detroit/detroit-mi-parcels_YYYY_MM_DD.parquet
"""
from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "detroit"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "detroit-mi-geometry.parquet"
VALS_CACHE = DATA_DIR / "detroit-mi-values.parquet"

BASE = "https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services"
PARCELS_URL = f"{BASE}/parcel_file_current/FeatureServer/0/query"
ROLL_URL = f"{BASE}/tentative_assessment_roll_2026/FeatureServer/0/query"

GEOM_FIELDS = ("parcel_id,address,property_class,property_class_description,"
               "use_code_description,tax_status,total_square_footage,total_acreage,"
               "is_improved,total_floor_area,year_built")
ROLL_FIELDS = ("parcel_id,tax_status,amt_estimated_true_cash_value,amt_land_value,"
               "amt_assessed_value")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 1000
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm_pid(s):
    return s.astype(str).str.strip().str.upper()


def _count(url, where="1=1"):
    return requests.get(url, params={"where": where, "returnCountOnly": "true", "f": "json"},
                        headers=HEADERS, timeout=120).json().get("count", 0)


def fetch_geojson(url, fields, want_geom):
    """Paginated pull of an ArcGIS layer -> GeoDataFrame (geom) or DataFrame (attrs)."""
    total = _count(url)
    log(f"  pulling {total:,} rows from {url.split('/services/')[1].split('/FeatureServer')[0]} "
        f"({'geojson' if want_geom else 'json'})...")
    pages, off = [], 0
    while off < total:
        params = {"where": "1=1", "outFields": fields,
                  "resultOffset": off, "resultRecordCount": PAGE,
                  "returnGeometry": "true" if want_geom else "false"}
        if want_geom:
            params.update({"outSR": 4326, "f": "geojson"})
        else:
            params["f"] = "json"
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, headers=HEADERS, timeout=180)
                r.raise_for_status()
                break
            except Exception as e:  # transient AGOL hiccup
                if attempt == 3:
                    raise
                log(f"    retry page@{off} ({e})")
                time.sleep(2 * (attempt + 1))
        if want_geom:
            gdf = gpd.read_file(io.BytesIO(r.content))
            n = len(gdf)
        else:
            feats = r.json().get("features", [])
            gdf = pd.DataFrame([f["attributes"] for f in feats])
            n = len(gdf)
        if n == 0:
            break
        pages.append(gdf)
        off += n
        if off % 50000 < PAGE:
            log(f"    {off:,}/{total:,}")
        if n < PAGE:
            break
    if want_geom:
        return gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    return pd.concat(pages, ignore_index=True)


# ── 1. Parcel geometry + attributes (cached) ─────────────────────────────────
if GEOM_CACHE.exists():
    log(f"Using cached geometry: {GEOM_CACHE.name}")
    geom = gpd.read_parquet(GEOM_CACHE)
else:
    log("Fetching Detroit parcel geometry (parcel_file_current)...")
    geom = fetch_geojson(PARCELS_URL, GEOM_FIELDS, want_geom=True)
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached -> {GEOM_CACHE.name} ({len(geom):,} rows)")

geom["pid"] = norm_pid(geom["parcel_id"])
geom = geom[geom["pid"].str.len() > 0].copy()

# ── 2. Assessment-roll values (cached) ───────────────────────────────────────
if VALS_CACHE.exists():
    log(f"Using cached values: {VALS_CACHE.name}")
    vals = pd.read_parquet(VALS_CACHE)
else:
    log("Fetching Detroit assessment roll values (tentative_assessment_roll_2026)...")
    vals = fetch_geojson(ROLL_URL, ROLL_FIELDS, want_geom=False)
    vals.to_parquet(VALS_CACHE, index=False)
    log(f"  cached -> {VALS_CACHE.name} ({len(vals):,} rows)")

vals["pid"] = norm_pid(vals["parcel_id"])
for c in ["amt_estimated_true_cash_value", "amt_land_value", "amt_assessed_value"]:
    vals[c] = pd.to_numeric(vals.get(c), errors="coerce")
vals = vals.rename(columns={"tax_status": "roll_tax_status"})
vals = vals.drop_duplicates("pid")[
    ["pid", "roll_tax_status", "amt_estimated_true_cash_value", "amt_land_value",
     "amt_assessed_value"]]

# ── 3. Join values onto geometry ─────────────────────────────────────────────
parcel = geom.merge(vals, on="pid", how="left")
matched = int(parcel["amt_estimated_true_cash_value"].notna().sum())
log(f"Joined {len(parcel):,} parcels | matched roll values {matched:,} "
    f"({100 * matched / max(len(parcel), 1):.1f}%)")

# Tax status: prefer the parcel file's current status, fall back to the roll's.
ts = parcel["tax_status"].astype(str).str.strip().str.upper()
ts = ts.where(ts.ne("") & ts.ne("NONE") & ts.ne("NAN"),
              parcel["roll_tax_status"].astype(str).str.strip().str.upper())
parcel["tax_status_norm"] = ts

# Values: TCV = full market; land on TCV basis; improvement = TCV - land.
# Where the roll didn't match, fall back to 2x assessed (MI SEV->TCV ~2x) for the
# market value and leave the land/improvement split null (can't derive it).
av2 = parcel["amt_assessed_value"] * 2.0
parcel["full_market_value"] = parcel["amt_estimated_true_cash_value"].where(
    parcel["amt_estimated_true_cash_value"].notna(), av2)
parcel["land_value"] = parcel["amt_land_value"]
impr = parcel["full_market_value"] - parcel["land_value"]
parcel["improvement_value"] = impr.clip(lower=0)  # guard rounding -> tiny negatives

# ── 4. Dedup on parcel_id (rare) — values are per-parcel (first), area summed,
#       geometry unioned. Detroit's parcel file is ~1 polygon per parcel_id. ──
ndup = parcel.duplicated(subset=["pid"], keep=False).sum()
if ndup:
    log(f"Collapsing {ndup:,} duplicate parcel_id rows...")
    value_cols = ["full_market_value", "land_value", "improvement_value",
                  "amt_estimated_true_cash_value", "amt_land_value", "amt_assessed_value"]
    area_cols = [c for c in ["total_square_footage"] if c in parcel.columns]
    first_cols = [c for c in parcel.columns
                  if c not in set(value_cols + area_cols + ["geometry", "pid"])]
    agg = {c: "first" for c in value_cols if c in parcel.columns}
    agg.update({c: "sum" for c in area_cols})
    agg.update({c: "first" for c in first_cols})
    coll = parcel.groupby("pid", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("pid", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None])
        if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs=geom.crs)
log(f"After dedup -> {len(parcel):,}")

# ── 5. Categorize from Michigan property_class + use_code_description ─────────
MF_HINTS = ["DUPLEX", "TWO FAMILY", "THREE FAMILY", "FOUR FAMILY", "FIVE FAMILY",
            "SIX FAMILY", "APT", "APARTMENT", "FLAT", "ROW HOUSE", "COOPERATIVE",
            "DORMITORY", "PUBLIC LODGING"]
PARKING_HINTS = ["PARKING LOT", "GARAGE-PARKING"]


def categorize(row):
    pc = pd.to_numeric(row.get("property_class"), errors="coerce")
    pc = int(pc) if pd.notna(pc) else 0
    u = str(row.get("use_code_description") or "").upper()
    if any(k in u for k in PARKING_HINTS):
        return "Parking"
    if pc == 402:
        return "Vacant Residential"
    if pc == 202:
        return "Vacant Commercial"
    if pc == 302:
        return "Vacant Industrial"
    if pc in (401, 407):
        if pc == 401 and any(k in u for k in MF_HINTS):
            return "Multifamily"
        return "Residential Condo" if pc == 407 else "Single Family"
    if pc in (201, 207):
        return "Commercial"
    if pc in (301, 307):
        return "Industrial"
    if pc in (303, 403):
        return "Common Element"
    return "Other"


parcel["property_land_use_category"] = parcel.apply(categorize, axis=1)

# ── 6. Exemptions: anything not 'TAXABLE' (incl. null) is exempt -> dropped ───
parcel["exemption_flag"] = (parcel["tax_status_norm"] != "TAXABLE").astype(int)
# Common-element / retired-split parcels carry no independent value to map.
parcel.loc[parcel["property_land_use_category"] == "Common Element", "exemption_flag"] = 1
n_ex = int(parcel["exemption_flag"].sum())
log(f"Exempt/non-taxable flagged: {n_ex:,} ({100 * n_ex / max(len(parcel), 1):.1f}%)")
ex = parcel[parcel["exemption_flag"] == 0].copy()
log(f"After exempt filter -> {len(ex):,}")

# ── 7. Refined under-utilization category (Vacant / Parking / Underdeveloped) ─
# Detroit is large; skip the Overture footprint cross-check (network/scale). The
# assessor's class already flags Vacant/Parking; `total_floor_area` guards the
# improvement==0 "Vacant" rule so built-but-unvalued parcels don't read as vacant.
ex["property_land_use_refined"] = classify_property_refined(
    ex,
    land_col="land_value",
    improvement_col="improvement_value",
    category_col="property_land_use_category",
    bld_ar_col="total_floor_area",
    state_class_col="__none__",          # Detroit has no state-class field
    exclude_categories=("Common Element",),
    fetch_footprints=False,
)

# ── 8. Canonical fields — reported lot sqft denominator (geodesic fallback) ───
def gis_area_sqft(g):
    if g is None or g.is_empty:
        return np.nan
    if g.geom_type == "Polygon":
        lon, lat = g.exterior.coords.xy
        a, _ = geod.polygon_area_perimeter(lon, lat)
        hole = 0.0
        for ring in g.interiors:
            lh, ph = ring.coords.xy
            ah, _ = geod.polygon_area_perimeter(lh, ph)
            hole += abs(ah)
        return max(abs(a) - hole, 0.0) * 10.763910416709722
    if g.geom_type == "MultiPolygon":
        return sum(gis_area_sqft(p) for p in g.geoms)
    return np.nan


ex["geometry"] = ex["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
log("Computing geodesic areas...")
ex["gis_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["gis_area_sqft"] < 1, "gis_area_sqft"] = np.nan
ex["reported_sqft"] = pd.to_numeric(ex.get("total_square_footage"), errors="coerce")
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

use_reported = ex["reported_sqft"] > 0
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["gis_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
ex["link"] = np.nan  # no stable public per-parcel deep link; popup omits when null

# ── 9. Export canonical schema ───────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs=ex.crs)
if final.crs is None or final.crs.to_epsg() != 4326:
    final = final.to_crs("EPSG:4326")

out = DATA_DIR / "detroit-mi-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"detroit-mi-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.2f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.2f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
