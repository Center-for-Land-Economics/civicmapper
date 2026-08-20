#!/usr/bin/env python3
"""
Build the City of Vancouver, BC (Canada) canonical parcel parquet.

Two-source city, but BOTH sources are open/free (no gating, no manual download) — the
City of Vancouver's own Open Data Portal (opendata.vancouver.ca, OpenDataSoft platform):

  GEOMETRY  — "Property parcel polygons" dataset. Assessment-based land polygons,
    already scoped to the City of Vancouver (this is the City's own portal, not a
    regional/Metro Vancouver layer, so no separate city-boundary clip is needed).
    Fields: civic_number, streetname, tax_coord (= first 8 digits of the BC Assessment
    folio number — the join key), site_id (Strata Plan number when the parcel is a
    condo building's shared footprint), geom (already EPSG:4326 GeoJSON), geo_point_2d.
    ~99.7k polygons, updated weekly.
    https://opendata.vancouver.ca/explore/dataset/property-parcel-polygons/

  VALUES    — "Property tax report" dataset. BC Assessment (BCA) current/previous land +
    improvement value per FOLIO, refreshed weekly, spans many `report_year`s (1.55M rows
    total) — filter to the latest `report_year` (~228.6k rows for one year). Join key is
    `land_coordinate` (= `tax_coord` above). Also carries `zoning_district` /
    `zoning_classification` (our land-use proxy — BC Assessment's actual-use code is NOT
    in this open extract) and `legal_type` (STRATA / LAND / OTHER).
    https://opendata.vancouver.ca/explore/dataset/property-tax-report/

  CONDO PATTERN (verified live, 2026-08-20): a condo building is ONE polygon in the
  geometry dataset but MANY rows in the tax report (one row per strata unit, its own
  folio). Unlike a Texas-style appraisal broadcast, BC Assessment's per-unit land value
  is a genuine PROPORTIONATE SHARE of the building's land (verified: units sharing one
  land_coordinate showed land values ranging $417k-$1.10M, not a duplicated constant) —
  so the correct aggregation is **groupby(land_coordinate).sum()** for land/improvement
  value, not `first`. This is the opposite of the usual Dallas-railroad dedup rule
  (playbook §2) because here the source already splits the value per unit; summing
  reconstitutes the whole-building value. Do not change this to `first` without re-
  verifying against a live building.

  EXEMPTION: government/park/crown/institutional land generally has NO row in the tax
  report at all (only 9 of 228,623 current-year rows have current_land_value=0), so a
  geometry parcel with no matching land_coordinate is exempt/unassessed by construction
  — no separate exemption-flag heuristic needed. These are simply left unjoined and
  dropped (exemption_flag=1).

  LAND USE: no actual-use code is published in this open extract, so
  `zoning_classification` (broad: Residential / Comprehensive Development / Commercial /
  Industrial / Multiple Family Dwelling / etc.) is the best available origCategoryField
  proxy. It is a legal-use bucket, not true land use, so refined
  Vacant/Underdeveloped/Parking classification leans on the land/improvement value ratio
  (classify_property_refined) rather than the category text, same as Vancouver, WA and
  most non-Texas cities.

Outputs:
- data/jurisidictions/data/vancouverbc/vancouver-bc-ca-parcels.parquet
- data/jurisidictions/data/vancouverbc/vancouver-bc-ca-parcels_YYYY_MM_DD.parquet

Currency: CAD ($CA). Units: imperial (feet) is fine — BC Assessment/City of Vancouver
publish areas in sqft colloquially same as US cities; no metric conversion needed for
this city's audience, so unitSystem is left at the default 'imperial'.

Run:
  PYTHONUTF8=1 python run_vancouverbc.py
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from datetime import datetime
from pathlib import Path
from shapely.ops import unary_union
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "vancouverbc"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "vancouver-bc-ca-geometry.parquet"
TAX_CACHE = DATA_DIR / "vancouver-bc-ca-tax-report.parquet"

BASE = "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets"
GEOM_DATASET = "property-parcel-polygons"
TAX_DATASET = "property-tax-report"
PAGE = 100  # ODS v2.1 records endpoint caps at 100/page regardless of requested limit
SQFT_PER_ACRE = 43560.0
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ods_export(dataset, select=None, where=None):
    """Full-dataset pull via the OpenDataSoft v2.1 /exports/json endpoint.

    The /records endpoint caps offset+limit at 10,000 (verified live — 99.7k geometry
    rows and 228.6k tax rows both exceed that), so bulk pulls must use /exports/json,
    which streams the entire filtered result in one request with no offset limit.
    """
    url = f"{BASE}/{dataset}/exports/json"
    params = {}
    if select:
        params["select"] = select
    if where:
        params["where"] = where
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=600)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Export pull failed for {dataset}")


def fetch_geometry():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    log(f"Pulling {GEOM_DATASET} (full export)...")
    rows = ods_export(GEOM_DATASET, select="civic_number,streetname,tax_coord,site_id,geom")
    log(f"  received {len(rows):,} rows")
    from shapely.geometry import shape
    recs = []
    for row in rows:
        g = row.get("geom")
        if not g:
            continue
        # exports/json wraps the geometry in a GeoJSON Feature: {"type":"Feature","geometry":{...}}
        geom_json = g.get("geometry") if isinstance(g, dict) and g.get("type") == "Feature" else g
        if not geom_json:
            continue
        try:
            geom = shape(geom_json)
        except Exception:  # noqa: BLE001
            continue
        recs.append({
            "civic_number": row.get("civic_number"),
            "streetname": row.get("streetname"),
            "tax_coord": row.get("tax_coord"),
            "site_id": row.get("site_id"),
            "geometry": geom,
        })
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
    gdf.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(gdf):,} rows)")
    return gdf


def latest_report_year():
    r = requests.get(f"{BASE}/{TAX_DATASET}/records", params={
        "limit": 1, "order_by": "report_year desc", "select": "report_year"}, timeout=60)
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0]["report_year"] if results else None


def fetch_tax_report(year):
    if TAX_CACHE.exists():
        log(f"Using cached tax report: {TAX_CACHE.name}")
        return pd.read_parquet(TAX_CACHE)
    log(f"Pulling {TAX_DATASET} for report_year={year} (full export)...")
    fields = ("land_coordinate,folio,legal_type,zoning_district,zoning_classification,"
              "current_land_value,current_improvement_value,year_built,tax_levy,"
              "street_name,to_civic_number")
    rows = ods_export(TAX_DATASET, select=fields, where=f'report_year="{year}"')
    log(f"  received {len(rows):,} rows")
    df = pd.DataFrame(rows)
    df.to_parquet(TAX_CACHE, index=False)
    log(f"  cached tax report -> {TAX_CACHE.name} ({len(df):,} rows)")
    return df


# ── 1. pull + cache both sources ───────────────────────────────────────────────
year = latest_report_year()
log(f"Latest report_year: {year}")
geom = fetch_geometry()
tax = fetch_tax_report(year)

# ── 2. clean geometry ───────────────────────────────────────────────────────────
if geom.crs is None:
    geom = geom.set_crs("EPSG:4326")
elif geom.crs.to_epsg() != 4326:
    geom = geom.to_crs("EPSG:4326")
geom["geometry"] = geom["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
geom = geom[geom["geometry"].notnull() & geom["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
geom["tax_coord"] = geom["tax_coord"].astype(str).str.strip()
geom = geom[geom["tax_coord"].ne("") & geom["tax_coord"].ne("None") & geom["tax_coord"].ne("nan")]
log(f"Valid geometry rows: {len(geom):,}")

# dedup multi-polygon-per-tax_coord (a few footprints are split into >1 GIS polygon) —
# geometry unioned, non-geometry attrs first (they're identical per tax_coord: civic
# address/site_id describe the physical footprint, not a value that could be inflated).
ndup = geom.duplicated(subset=["tax_coord"], keep=False).sum()
log(f"Geometry rows sharing a tax_coord (multi-polygon footprint): {ndup:,}")
if ndup:
    first_cols = [c for c in geom.columns if c not in ("geometry", "tax_coord")]
    coll = geom.groupby("tax_coord", dropna=False)[first_cols].first().reset_index()
    gu = geom.groupby("tax_coord", dropna=False)["geometry"].apply(
        lambda gs: unary_union([g for g in gs if g is not None]))
    coll["geometry"] = coll["tax_coord"].map(gu)
    geom = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After tax_coord dedup -> {len(geom):,} footprints")

# ── 3. aggregate tax report per land_coordinate (SUM values — proportionate strata
#      shares, verified live; NOT a broadcast duplicate) ──────────────────────────
tax["land_coordinate"] = tax["land_coordinate"].astype(str).str.strip()
tax["current_land_value"] = pd.to_numeric(tax["current_land_value"], errors="coerce").fillna(0)
tax["current_improvement_value"] = pd.to_numeric(tax["current_improvement_value"], errors="coerce").fillna(0)
n_units = tax.groupby("land_coordinate").size()
log(f"Tax report rows: {len(tax):,} across {tax['land_coordinate'].nunique():,} land parcels "
    f"(max units sharing one parcel: {int(n_units.max())})")

agg = tax.groupby("land_coordinate", dropna=False).agg(
    land_value=("current_land_value", "sum"),
    improvement_value=("current_improvement_value", "sum"),
    zoning_district=("zoning_district", "first"),
    zoning_classification=("zoning_classification", "first"),
    legal_type=("legal_type", "first"),
    year_built=("year_built", "max"),
    tax_levy=("tax_levy", "sum"),
    unit_count=("folio", "count"),
).reset_index()

# ── 4. join geometry <- tax values (left join: unmatched = exempt/unassessed) ─────
merged = geom.merge(agg, left_on="tax_coord", right_on="land_coordinate", how="left")
merged["exemption_flag"] = merged["land_value"].isna().astype(int)
log(f"Footprints with an assessed value: {(merged['exemption_flag'] == 0).sum():,} / "
    f"{len(merged):,} ({(merged['exemption_flag'] == 1).sum():,} exempt/unassessed, dropped)")

ex = merged[merged["exemption_flag"] == 0].copy()
ex["land_value"] = pd.to_numeric(ex["land_value"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["improvement_value"], errors="coerce")
ex["property_land_use_category"] = ex["zoning_classification"].fillna("Other").replace({"": "Other"})

# ── 5. classification ──────────────────────────────────────────────────────────
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=(),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)

# ── 6. canonical fields — geodesic area (no stated area field in this source) ─────
def gis_area_sqft(geom_):
    if geom_ is None or geom_.is_empty:
        return np.nan
    if geom_.geom_type == "Polygon":
        lon, lat = geom_.exterior.coords.xy
        a, _ = geod.polygon_area_perimeter(lon, lat)
        return abs(a) * 10.763910416709722
    if geom_.geom_type == "MultiPolygon":
        return sum(gis_area_sqft(p) for p in geom_.geoms)
    return np.nan


ex["geometry"] = ex["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
log("Computing geodesic areas...")
ex["land_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
ex["area_source"] = "gis"
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = ex["land_value"] + ex["improvement_value"]
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

ex["current_tax"] = pd.to_numeric(ex["tax_levy"], errors="coerce")
ex["current_tax_per_sqft"] = ex["current_tax"] / den

# BC Assessment's public property-info lookup takes a folio/PID, not land_coordinate;
# no reliable single-parcel deep link in this open extract, so link is omitted.

# ── 7. export ───────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "current_tax", "current_tax_per_sqft",
           "land_area_acres", "area_source", "likely_remnant", "zoning_district", "legal_type",
           "unit_count", "year_built"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].copy()
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs=ex.crs)
if final.crs is None or final.crs.to_epsg() != 4326:
    final = final.to_crs("EPSG:4326")

out = DATA_DIR / "vancouver-bc-ca-parcels.parquet"
final.to_parquet(out, index=False)
dated = DATA_DIR / f"vancouver-bc-ca-parcels_{datetime.now():%Y_%m_%d}.parquet"
final.to_parquet(dated, index=False)
log(f"Wrote {len(final):,} parcels -> {out}")

# ── 8. smoke checks (playbook §6a) ──────────────────────────────────────────────
log("--- smoke checks ---")
log(f"property_land_use_category counts:\n{final['property_land_use_category'].value_counts().head(15)}")
log(f"property_land_use_refined counts:\n{final['property_land_use_refined'].value_counts(dropna=False)}")
a = final["land_area_sqft"] if "land_area_sqft" in final.columns else ex["land_area_sqft"]
log(f"footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"sub-500 / sub-1000 sqft: {int((a < 500).sum())} / {int((a < 1000).sum())}")
lv = final["land_value_per_sqft"].replace([np.inf, -np.inf], np.nan).dropna()
log(f"land $/sqft p50/p99/max: {round(lv.median())} / {round(lv.quantile(.99))} / {round(lv.max())}")
log(f"exempt/unassessed dropped: {int(merged['exemption_flag'].sum()):,}")
