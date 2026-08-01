#!/usr/bin/env python3
"""
Build San Antonio's canonical parcel parquet from hosted Bexar (BCAD) ArcGIS layers.

Unlike Austin/Dallas (which need a manually-downloaded appraisal export), San Antonio
is fully automated: a public hosted layer carries geometry + values + class + situs.

Sources (all on reachable services*.arcgis.com / CoSAGIS hosts — the appraisal-district
hosts maps.bexar.org / maps.bcad.org are firewalled and intentionally avoided):
- Parcels (geometry + LAND/IMP/MKT value + state class + situs city + area):
  "Bexar_parcels_all" FeatureServer
  https://services2.arcgis.com/82iS1Pc7dgs3LFZv/arcgis/rest/services/Bexar_parcels_all/FeatureServer/0
  Pulled (where SITUS_CITY='SAN ANTONIO') + cached on first run; reused after.
- Exemptions (authoritative EX-* total-exemption flag), joined on GEO_ID:
  CoSAGIS "BCAD_Parcels" FeatureServer (attributes only, no geometry)
  https://services.arcgis.com/g1fRTDLeMgspWrYp/arcgis/rest/services/BCAD_Parcels/FeatureServer/0
- City limits: authoritative City of San Antonio "COSABoundary" FeatureServer (centroid within)
  https://services.arcgis.com/g1fRTDLeMgspWrYp/arcgis/rest/services/COSABoundary/FeatureServer/1

Outputs:
- data/jurisidictions/data/sanantonio/sanantonio-tx-parcels.parquet
- data/jurisidictions/data/sanantonio/sanantonio-tx-parcels_YYYY_MM_DD.parquet

Notes:
- Bexar_parcels_all is a TAX_YEAR 2022 snapshot — values are a few years old but it is
  the best reachable complete source (geometry + values in one layer).
- $/sqft denominator is GIS_AREA (reported in ACRES), falling back to the geodesic
  polygon area when GIS_AREA is missing. Emits QC columns land_area_acres, area_source,
  likely_remnant (<500 sqft fractional remnants).
- Exempt: BCAD Exemptions beginning with "EX" (EX-XV public, EX-XJ schools, EX-XG
  charitable, ...) are totally exempt and dropped; partial owner exemptions (HS, OV65,
  DV*, DP, HT, LIH) are NOT exclusions (normal taxable homes). Plus owner-keyword and
  state-class fallbacks.
- A single account (Prop_ID) can be split into several GIS polygons. Account values are
  taken ONCE (first) on dedup, never summed (summing would multiply by polygon count —
  the bug fixed for Dallas); area IS summed across the split polygons.
- STAT_LAND_ is the Texas SPTB state class (A1/B1/C1/F1...). hideUnderutilized is set.
- Upload + PMTiles are separate steps (San Antonio is large -> PMTiles):
    python data/scripts/parquet_to_pmtiles.py --city sanantonio --h3 --wsl --upload
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

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "sanantonio"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "sanantonio-tx-geometry.parquet"
EXEMPT_CACHE = DATA_DIR / "sanantonio-tx-exemptions.parquet"

PARCELS_URL = ("https://services2.arcgis.com/82iS1Pc7dgs3LFZv/arcgis/rest/services/"
               "Bexar_parcels_all/FeatureServer/0/query")
EXEMPT_URL = ("https://services.arcgis.com/g1fRTDLeMgspWrYp/arcgis/rest/services/"
              "BCAD_Parcels/FeatureServer/0/query")
BOUNDARY_URL = ("https://services.arcgis.com/g1fRTDLeMgspWrYp/arcgis/rest/services/"
                "COSABoundary/FeatureServer/1/query")
PARCEL_WHERE = "SITUS_CITY='SAN ANTONIO'"
GEOM_FIELDS = ("Prop_ID,GEO_ID,OWNER_NAME,STAT_LAND_,LAND_VALUE,IMP_VALUE,MKT_VALUE,"
               "GIS_AREA,SITUS_CITY")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm_geoid(s):
    return s.astype(str).str.strip().str.upper()


def _paginate_geojson(url, where, out_fields, return_geometry, cache, label):
    if cache.exists():
        log(f"Using cached {label}: {cache.name}")
        return gpd.read_parquet(cache) if return_geometry else pd.read_parquet(cache)
    total = requests.get(url, params={"where": where, "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} {label} rows (paginated)...")
    pages, off = [], 0
    fmt = "geojson" if return_geometry else "json"
    while off < total:
        for attempt in range(4):
            try:
                r = requests.get(url, params={
                    "where": where, "outFields": out_fields, "returnGeometry": str(return_geometry).lower(),
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326, "f": fmt,
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                if return_geometry:
                    chunk = gpd.read_file(io.BytesIO(r.content))
                else:
                    feats = r.json().get("features", [])
                    chunk = pd.DataFrame([f["attributes"] for f in feats])
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt+1} @off {off}: {type(e).__name__}")
                time.sleep(5 * (attempt + 1))
                chunk = None
        if chunk is None:
            raise RuntimeError(f"{label} pull failed at offset {off}")
        if not len(chunk):
            break
        pages.append(chunk)
        off += len(chunk)
        if off % 40000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(chunk) < PAGE:
            break
    if return_geometry:
        out = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    else:
        out = pd.concat(pages, ignore_index=True)
    out.to_parquet(cache, index=False)
    log(f"  cached {label} -> {cache.name} ({len(out):,} rows)")
    return out


# ── 1. Parcels (geometry + values + class + situs + area) ────────────────────
geom = _paginate_geojson(PARCELS_URL, PARCEL_WHERE, GEOM_FIELDS, True, GEOM_CACHE, "parcels")
geom["geoid"] = norm_geoid(geom["GEO_ID"])
geom["acct"] = geom["Prop_ID"].astype(str).str.strip()
geom = geom[geom["acct"].ne("") & geom["acct"].ne("None")]
for c in ["LAND_VALUE", "IMP_VALUE", "MKT_VALUE", "GIS_AREA"]:
    geom[c] = pd.to_numeric(geom[c], errors="coerce")

# ── 2. Exemptions (authoritative EX-* flag), joined on GEO_ID ────────────────
exdf = _paginate_geojson(EXEMPT_URL, "PropID>0", "Geo_id,Exemptions", False, EXEMPT_CACHE, "exemptions")
exdf["geoid"] = norm_geoid(exdf["Geo_id"])
exdf["ex_total"] = exdf["Exemptions"].astype(str).str.strip().str.upper().str.startswith("EX")
ex_map = exdf.drop_duplicates("geoid").set_index("geoid")["ex_total"]
geom["ex_total"] = geom["geoid"].map(ex_map).fillna(False)
log(f"Parcels: {len(geom):,} | EX-* totally-exempt: {int(geom['ex_total'].sum()):,}")

parcel = geom.rename(columns={"LAND_VALUE": "land_val", "IMP_VALUE": "bld_val",
                              "OWNER_NAME": "mailto", "STAT_LAND_": "state_class"})
parcel["tot_appr_val"] = parcel["MKT_VALUE"].where(parcel["MKT_VALUE"] > 0,
                                                   parcel["land_val"] + parcel["bld_val"])
parcel["state_class"] = parcel["state_class"].astype(str).str.strip().str.upper()

# ── 3. Authoritative city-limits filter (centroid within City of San Antonio) ─
if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
rb = requests.get(BOUNDARY_URL, params={"where": "1=1", "outFields": "*", "outSR": 4326,
                  "f": "geojson"}, headers=HEADERS, timeout=120)
rb.raise_for_status()
boundary = unary_union(list(gpd.read_file(io.BytesIO(rb.content)).to_crs("EPSG:4326").geometry))
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
valid = parcel["geometry"].notnull() & parcel["geometry"].apply(lambda x: getattr(x, "is_valid", False))
cent = parcel.loc[valid, "geometry"].to_crs(3857).centroid.to_crs(4326)
inside = valid.copy()
inside[valid] = cent.within(boundary)
parcel = parcel[inside].copy()
log(f"City-limits filter -> {len(parcel):,}")

# ── 4. dedup (values first, area summed — never multiply by polygon count) ───
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
if ndup:
    value_cols = [c for c in ["tot_appr_val", "land_val", "bld_val"] if c in parcel.columns]
    area_cols = [c for c in ["GIS_AREA"] if c in parcel.columns]
    first_cols = [c for c in parcel.columns
                  if c not in set(value_cols + area_cols + ["geometry", "acct"])]
    agg = {c: "first" for c in value_cols}
    agg.update({c: "sum" for c in area_cols})
    agg.update({c: "first" for c in first_cols})
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs=parcel.crs)
log(f"After dedup -> {len(parcel):,}")


def categorize(v):
    """Texas SPTB state class (STAT_LAND_) -> coarse property category."""
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
eex = ex.get("ex_total", pd.Series(False, index=ex.index)).fillna(False).astype(bool)
KW = ["CITY OF SAN ANTONIO", "BEXAR COUNTY", "STATE OF TEXAS", "SAN ANTONIO ISD",
      "SAN ANTONIO IND SCH", "NORTHSIDE ISD", "NORTH EAST ISD", "NORTHEAST ISD",
      "EDGEWOOD ISD", "SOUTHWEST ISD", "HARLANDALE ISD", "SOUTH SAN ANTONIO ISD",
      "ALAMO COMMUNITY COLLEGE", "ALAMO COLLEGES", "VIA METROPOLITAN", "CPS ENERGY",
      "CITY PUBLIC SERVICE", "SAN ANTONIO WATER", "SAWS", "BEXAR METROPOLITAN",
      "UNIVERSITY OF TEXAS", "UNIV OF TEXAS", "UT HEALTH", "TEXAS A&M",
      "UNITED STATES", "US GOVT", "U.S. GOVERNMENT", "HOUSING AUTHORITY",
      "SAN ANTONIO HOUSING", "OPPORTUNITY HOME"]
eown = ex["mailto"].astype(str).str.upper().str.contains("|".join(KW), na=False)
ex["exemption_flag"] = (ebs | eex | eown).astype(int)
ex = ex[ex["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex = ex[~ex["property_land_use_category"].isin(
    {"Mineral / Oil & Gas", "Personal Property / Inventory", "Utility"})].copy()
ex["land_value"] = pd.to_numeric(ex.get("land_val", np.nan), errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex.get("bld_val", np.nan), errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(ex, fetch_footprints=False)
log(f"After exempt/refine -> {len(ex):,}")

# ── 5. Canonical fields — GIS_AREA (acres) denominator, geodesic fallback ────
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
ex["reported_sqft"] = pd.to_numeric(ex.get("GIS_AREA", np.nan), errors="coerce") * SQFT_PER_ACRE
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

ex["link"] = "https://bexar.trueautomation.com/clientdb/Property.aspx?cid=110&prop_id=" + ex["acct"].astype(str)

# ── 6. Export ────────────────────────────────────────────────────────────────
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
out = DATA_DIR / "sanantonio-tx-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"sanantonio-tx-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"p999=${final['land_value_per_sqft'].quantile(.999):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
