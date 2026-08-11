#!/usr/bin/env python3
"""
Build the Bryan / College Station canonical parcel parquet (the two cities as one unit).

Fully automated from a hosted Brazos County ArcGIS layer (geometry + values + class +
area in one layer), clipped to the UNION of the Bryan and College Station city limits.

Sources (reachable services*.arcgis.com hosts):
- Parcels (geometry + market/Land/Imprv value + state class + land area):
  "Brazos County, Texas Parcels" FeatureServer (PACS-schema, same publisher as the
  Bexar parcels layer used for San Antonio)
  https://services2.arcgis.com/82iS1Pc7dgs3LFZv/arcgis/rest/services/Brazos_County_Parcels/FeatureServer/0
  Pulled + cached on first run; reused after.
- City limits: "Brazos County City Limits" FeatureServer, ID IN ('Bryan','College Station'),
  unioned into one boundary (centroid-within clip).
  https://services1.arcgis.com/qr14biwnHA6Vis6l/arcgis/rest/services/Brazos_County_City_Limits/FeatureServer/0

Outputs:
- data/jurisidictions/data/bcs/bcs-tx-parcels.parquet
- data/jurisidictions/data/bcs/bcs-tx-parcels_YYYY_MM_DD.parquet

Notes:
- addr_city is the OWNER's mailing city (unreliable for jurisdiction), so the city
  filter is purely the authoritative Bryan ∪ College Station boundary clip.
- $/sqft denominator is the assessor's reported land_sqft (fallback land_acres, then the
  geodesic polygon area). Emits land_area_acres, area_source, likely_remnant (<500 sqft).
- The Brazos layer has no exemption flag, so exempt parcels are caught by state class
  (X*) plus an owner-keyword heuristic (city/county/state/ISD/Texas A&M/Blinn/etc.).
- A single account (PROP_ID) split into multiple GIS polygons keeps account-level values
  and reported area taken ONCE (first); geometry is unioned. (No summing — that was the
  Dallas N× bug.)
- state_cd is the Texas SPTB class (A1/B1/C1/F1...). hideUnderutilized + hideRemnants set.
- PMTiles bake should pass --drop-remnants (hideRemnants city):
    python data/scripts/parquet_to_pmtiles.py --city bcs --h3 --wsl --drop-remnants --upload
"""
from __future__ import annotations

import io
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
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "bcs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "bcs-tx-geometry.parquet"

PARCELS_URL = ("https://services2.arcgis.com/82iS1Pc7dgs3LFZv/arcgis/rest/services/"
               "Brazos_County_Parcels/FeatureServer/0/query")
BOUNDARY_URL = ("https://services1.arcgis.com/qr14biwnHA6Vis6l/arcgis/rest/services/"
                "Brazos_County_City_Limits/FeatureServer/0/query")
BOUNDARY_WHERE = "ID IN ('Bryan','College Station')"
GEOM_FIELDS = ("PROP_ID,geo_id,file_as_na,state_cd,Land_Val,Imprv_Val,market,"
               "land_sqft,land_acres")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── 1. Parcels (geometry + values + class + area), cached ─────────────────────
def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": "1=1", "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} Brazos County parcels (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        for attempt in range(4):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": "1=1", "outFields": GEOM_FIELDS, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326, "f": "geojson",
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                gdf = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt+1} @off {off}: {type(e).__name__}")
                time.sleep(5 * (attempt + 1))
                gdf = None
        if gdf is None:
            raise RuntimeError(f"Parcel pull failed at offset {off}")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if off % 20000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


geom = fetch_parcels()
geom["acct"] = geom["PROP_ID"].astype(str).str.strip()
geom = geom[geom["acct"].ne("") & geom["acct"].ne("None")]
for c in ["Land_Val", "Imprv_Val", "market", "land_sqft", "land_acres"]:
    geom[c] = pd.to_numeric(geom[c], errors="coerce")

parcel = geom.rename(columns={"Land_Val": "land_val", "Imprv_Val": "bld_val",
                              "file_as_na": "mailto", "state_cd": "state_class"})
parcel["tot_appr_val"] = parcel["market"].where(parcel["market"] > 0,
                                               parcel["land_val"] + parcel["bld_val"])
parcel["state_class"] = parcel["state_class"].astype(str).str.strip().str.upper()

# ── 2. Authoritative city-limits filter: Bryan ∪ College Station ─────────────
if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
rb = requests.get(BOUNDARY_URL, params={"where": BOUNDARY_WHERE, "outFields": "ID",
                  "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=120)
rb.raise_for_status()
bgdf = gpd.read_file(io.BytesIO(rb.content)).to_crs("EPSG:4326")
log(f"City boundary parts: {sorted(bgdf['ID'].astype(str).tolist())}")
boundary = unary_union(list(bgdf.geometry))
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
valid = parcel["geometry"].notnull() & parcel["geometry"].apply(lambda x: getattr(x, "is_valid", False))
cent = parcel.loc[valid, "geometry"].to_crs(3857).centroid.to_crs(4326)
inside = valid.copy()
inside[valid] = cent.within(boundary)
parcel = parcel[inside].copy()
log(f"City-limits filter (Bryan ∪ College Station) -> {len(parcel):,}")

# ── 3. dedup (account-level values + reported area first; geometry unioned) ──
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
if ndup:
    # All non-geometry fields are account-level (values) or reported (land_sqft/acres),
    # i.e. identical across an account's split polygons -> take first, never sum.
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs=parcel.crs)
log(f"After dedup -> {len(parcel):,}")


def categorize(v):
    """Texas SPTB state class (state_cd) -> coarse property category."""
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


parcel["PROPERTY_CATEGORY"] = parcel["state_class"].apply(categorize)
ex = parcel.copy()
ebs = ex["PROPERTY_CATEGORY"].isin(["Exempt"])
# No exemption flag in this source -> owner-keyword heuristic for public/institutional land.
KW = ["CITY OF BRYAN", "CITY OF COLLEGE STATION", "BRAZOS COUNTY", "STATE OF TEXAS",
      "TEXAS A&M", "TEXAS A & M", "TEXAS AM UNIVERSITY", "A&M UNIVERSITY",
      "BOARD OF REGENTS", "BRYAN ISD", "BRYAN IND", "COLLEGE STATION ISD",
      "BLINN COLLEGE", "BLINN JUNIOR COLLEGE", "BRAZOS TRANSIT", "UNITED STATES",
      "US GOVT", "U.S. GOVERNMENT", "HOUSING AUTHORITY", "BRAZOS VALLEY",
      "UNIVERSITY OF TEXAS"]
eown = ex["mailto"].astype(str).str.upper().str.contains("|".join(KW), na=False)
ex["exemption_flag"] = (ebs | eown).astype(int)
ex = ex[ex["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex = ex[~ex["property_land_use_category"].isin(
    {"Mineral / Oil & Gas", "Personal Property / Inventory", "Utility"})].copy()
ex["land_value"] = pd.to_numeric(ex.get("land_val", np.nan), errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex.get("bld_val", np.nan), errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(ex, fetch_footprints=False)
log(f"After exempt/refine -> {len(ex):,}")

# ── 4. Canonical fields — reported land_sqft denominator (geodesic fallback) ─
def gis_area_sqft(geom):
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        a, _ = geod.polygon_area_perimeter(lon, lat)
        return abs(a) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(gis_area_sqft(p) for p in geom.geoms)
    return np.nan


ex["geometry"] = ex["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
log("Computing GIS areas...")
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
# Reported land size: prefer land_sqft, fall back to land_acres.
rep = pd.to_numeric(ex.get("land_sqft", np.nan), errors="coerce")
rep = rep.where(rep > 0, pd.to_numeric(ex.get("land_acres", np.nan), errors="coerce") * SQFT_PER_ACRE)
ex["reported_sqft"] = rep
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

use_reported = ex["reported_sqft"] > 0
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex.get("tot_appr_val", np.nan), errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

# Brazos CAD detail page; PROP_ID like "R34801" -> numeric id.
ex["link"] = ("https://esearch.brazoscad.org/property/view/"
              + ex["acct"].astype(str).str.replace(r"^[A-Za-z]+", "", regex=True))

# ── 5. Export ────────────────────────────────────────────────────────────────
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
out = DATA_DIR / "bcs-tx-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"bcs-tx-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"p999=${final['land_value_per_sqft'].quantile(.999):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
