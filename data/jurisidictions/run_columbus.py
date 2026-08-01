#!/usr/bin/env python3
"""
Build the City of Columbus, OH canonical parcel parquet.

Columbus spans Franklin (~94%), Delaware, and Fairfield counties. The City of
Columbus GIS "Central Ohio Parcels" layer is a one-stop source: it aggregates all
seven central-Ohio county auditors' parcels WITH appraised values (FCAO schema:
LNDVALUEBASE / BLDVALUEBASE / TOTVALUEBASE = 100% appraised market value, not the
35% assessed value) plus the statewide Ohio DTE land-use class code (CLASSCD) and a
per-county auditor detail link (HYPERLINK). No manual appraisal-roll join needed.

Sources (public, no token):
- Parcels (geometry + values + class + link), City of Columbus GIS:
  https://maps.columbus.gov/arcgis/rest/services/CityServices/KeyLayers/MapServer/3
  Pulled with an envelope pre-filter (Columbus bbox) + COUNTY IN (Franklin,
  Delaware, Fairfield), paginated GeoJSON; cached on first run.
- City of Columbus Corporate Boundary (authoritative, City of Columbus GIS):
  https://maps2.columbus.gov/arcgis/rest/services/Schemas/PublicService/MapServer/7
  CVT taxing-district text is NOT used for jurisdiction (it mixes in Gahanna /
  townships that merely share the Columbus school district).

Outputs:
- data/jurisidictions/data/columbus/columbus-oh-parcels.parquet
- data/jurisidictions/data/columbus/columbus-oh-parcels_YYYY_MM_DD.parquet

Notes:
- ~300k city parcels -> PMTiles + H3 (parquet_to_pmtiles.py --city columbus --wsl
  --drop-remnants --upload).
- Classification: Ohio DTE 3-digit CLASSCD (statewide scheme). 1xx ag, 3xx
  industrial, 4xx commercial (401-403 etc. -> Multifamily; 456/476 -> Parking so the
  refined classifier tags them Parking Lot), 5xx residential (550-559 condo), 6xx
  exempt, 899 ROW. PCLASS (A/C/E/I/R/U) is the fallback for the few null CLASSCD
  rows that carry value.
- Values are appraised market values (Columbus SFH sample medians match market;
  Ohio assessed value would be 35% of these).
- Condos: Franklin stacks per-unit condo parcels on one footprint (one coordinate
  carries 296 units downtown) -> same-footprint collapse with SUMMED values as in
  run_tulsa.py / run_olympia.py. Condo-ized rental complexes ("CONDO 40+ RENTAL
  UNITS", CLASSCD 553) include ~5.4k units with NULL values in the county source
  itself (e.g. Stone Lodge Apts) — no published value exists per unit; those merge
  into their stack (value = whatever the stack carries) or drop if the whole
  cluster is valueless. Documented undercount, not a join bug.
- $/sqft denominator is geodesic polygon area: STATEDAREA mixes units (sqft for
  some rows, acres for others) and ACRES is null in Franklin -> both unreliable.
- likely_remnant (<500 sqft) flagged; city ships with hideRemnants + --drop-remnants.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402
from pyproj import Geod  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "columbus"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "columbus-oh-geometry.parquet"

PARCELS_URL = ("https://maps.columbus.gov/arcgis/rest/services/CityServices/"
               "KeyLayers/MapServer/3/query")
BOUNDARY_URL = ("https://maps2.columbus.gov/arcgis/rest/services/Schemas/"
                "PublicService/MapServer/7/query")
OUT_FIELDS = ("PARCELID,COUNTY,CVTTXDSCRP,CLASSCD,CLASSDSCRP,PCLASS,ACRES,STATEDAREA,"
              "LNDVALUEBASE,BLDVALUEBASE,TOTVALUEBASE,CAUV,CAUVLNDBASE,"
              "RESFLRAREA,BLDGAREA,OWNERNME1,HYPERLINK")
COUNTY_WHERE = "COUNTY IN ('Franklin','Delaware','Fairfield')"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_boundary():
    """Columbus corporate boundary as a single multipolygon, EPSG:4326."""
    r = requests.get(BOUNDARY_URL, params={
        "where": "CITY_NAME='COLUMBUS'", "outFields": "CITY_NAME", "returnGeometry": "true",
        "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    bnd = gpd.read_file(io.BytesIO(r.content))
    if bnd.crs is None:
        bnd = bnd.set_crs(4326)
    elif bnd.crs.to_epsg() != 4326:
        bnd = bnd.to_crs(4326)
    poly = unary_union([g.buffer(0) for g in bnd.geometry if g is not None])
    log(f"Columbus boundary: {len(bnd)} feature(s); bbox {[round(x, 5) for x in bnd.total_bounds]}")
    return poly, list(bnd.total_bounds)


def fetch_parcels(bbox):
    """Pull Franklin/Delaware/Fairfield parcels intersecting the Columbus bbox."""
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    minx, miny, maxx, maxy = bbox
    spatial = {
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
    }
    total = requests.get(PARCELS_URL, params={**spatial, "where": COUNTY_WHERE,
                         "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} central-OH parcels in Columbus bbox (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    **spatial, "where": COUNTY_WHERE, "outFields": OUT_FIELDS,
                    "returnGeometry": "true", "resultOffset": off,
                    "resultRecordCount": PAGE, "outSR": 4326,
                    "orderByFields": "OBJECTID", "f": "geojson",
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                gdf = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt+1} @off {off}: {type(e).__name__}: {e}")
                time.sleep(5 * (attempt + 1))
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


boundary_poly, bbox = fetch_boundary()
geom = fetch_parcels(bbox)

# ── clip to City of Columbus (centroid within corporate boundary) ─────────────
if geom.crs is None:
    geom = geom.set_crs("EPSG:4326")
elif geom.crs.to_epsg() != 4326:
    geom = geom.to_crs("EPSG:4326")
geom["geometry"] = geom["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
geom = geom[geom["geometry"].notnull() & geom["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
rep = geom.geometry.representative_point()
inside = rep.within(boundary_poly)
log(f"Centroid-in-Columbus clip: {int(inside.sum()):,} of {len(geom):,} bbox parcels")
parcel = geom[inside.values].copy()

# ── field cleanup ─────────────────────────────────────────────────────────────
parcel["acct"] = parcel["PARCELID"].astype(str).str.strip().str.upper()
parcel = parcel[parcel["acct"].ne("") & parcel["acct"].ne("NONE") & parcel["acct"].ne("NAN")]
for c in ["LNDVALUEBASE", "BLDVALUEBASE", "TOTVALUEBASE", "RESFLRAREA", "BLDGAREA"]:
    parcel[c] = pd.to_numeric(parcel[c], errors="coerce")
parcel = parcel.rename(columns={"LNDVALUEBASE": "land_val", "BLDVALUEBASE": "bld_val"})
parcel["tot_appr_val"] = parcel["TOTVALUEBASE"].where(
    parcel["TOTVALUEBASE"] > 0, parcel["land_val"].fillna(0) + parcel["bld_val"].fillna(0))
parcel["bld_ar"] = parcel[["RESFLRAREA", "BLDGAREA"]].max(axis=1)
parcel["classcd"] = parcel["CLASSCD"].astype(str).str.strip()
parcel["pclass"] = parcel["PCLASS"].astype(str).str.strip().str.upper()

# GIS-only polygons with no assessor record at all (no class, no value): ROW and
# water slivers, plus the odd condo common ground. They carry nothing to map — drop,
# but only AFTER the count is logged (they are NOT a join failure).
no_record = parcel["classcd"].isin(["", "None", "nan"]) & ~(parcel["tot_appr_val"] > 0) \
    & parcel["pclass"].isin(["", "NONE", "NAN"])
log(f"Dropping {int(no_record.sum()):,} GIS-only polygons (no class, no value)")
parcel = parcel[~no_record].copy()

# ── dedup: one PARCELID split into multiple polygons (values first, geom union) ──
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing a PARCELID (multi-polygon): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After PARCELID dedup -> {len(parcel):,}")

# ── condo same-footprint collapse ─────────────────────────────────────────────
# Franklin stacks per-unit condo parcels (CLASSCD 550-559, plus office/retail/medical
# condo units 450/475/485/391 and condo parking 476) on one shared footprint — up to
# 296 units at a single coordinate downtown. Collapse stacks of >1 DISTINCT parcel at
# one representative point into a single footprint with SUMMED values (distinct units
# = distinct value shares). Same pattern as run_tulsa.py.
rp = parcel.geometry.representative_point()
parcel["_rpkey"] = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str))
vc = parcel["_rpkey"].value_counts()
stacked_keys = vc[vc > 1].index
log(f"Stacked footprints (>1 parcel at one point): {len(stacked_keys):,}; "
    f"max stack: {int(vc.max())}; parcels involved: {int(vc[vc > 1].sum()):,}")
if len(stacked_keys):
    is_stacked = parcel["_rpkey"].isin(stacked_keys)
    single = parcel[~is_stacked].copy()
    single["_collapsed"] = 0
    multi = parcel[is_stacked].copy()
    sum_cols = ["land_val", "bld_val", "tot_appr_val", "bld_ar"]
    first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
    agg = {c: "sum" for c in sum_cols if c in multi.columns}
    agg.update({c: "first" for c in first_cols})
    coll = multi.groupby("_rpkey", dropna=False).agg(agg).reset_index()
    gu = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    coll["_collapsed"] = 1
    coll = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
    parcel = pd.concat([single, coll], ignore_index=True)
    parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
else:
    parcel["_collapsed"] = 0
parcel = parcel.drop(columns=["_rpkey"], errors="ignore")
log(f"After condo footprint collapse -> {len(parcel):,}")


# ── classification: Ohio DTE land-use class code (CLASSCD) ────────────────────
def categorize(code, pclass):
    """Ohio DTE 3-digit class code -> coarse property category. Statewide scheme:
    1xx ag, 2xx mineral, 3xx industrial, 4xx commercial, 5xx residential, 6xx exempt,
    7xx abatement/TIF (kept, category by description), 8xx utility/ROW."""
    try:
        c = int(str(code).strip())
    except (TypeError, ValueError):
        p = str(pclass or "").strip().upper()
        return {"R": "Single Family", "C": "Commercial", "I": "Industrial",
                "A": "Agricultural / Rural", "E": "Exempt", "U": "Utility / ROW"}.get(p, "Other")
    if 100 <= c <= 199:
        return "Agricultural / Rural"
    if 200 <= c <= 299:
        return "Mineral"
    if c == 300:
        return "Vacant Industrial"
    if 301 <= c <= 399:
        return "Industrial"
    if c == 400:
        return "Vacant Commercial"
    if c in (401, 402, 403, 404, 409, 414, 418, 419, 496):
        return "Multifamily"
    if c == 415:
        return "Mobile Home Park"
    if c in (456, 476):
        return "Parking"          # "Parking" keys the refined classifier -> Parking Lot
    if 401 <= c <= 499:
        return "Commercial"
    if 500 <= c <= 505:
        return "Vacant Residential"
    if c in (510, 511, 512, 513, 514, 515, 560, 591, 599):
        return "Single Family"
    if 520 <= c <= 534 or c == 592:
        return "Two & Three Family"
    if c in (540, 555):
        return "Common Area / HOA"
    if 550 <= c <= 559:
        return "Condominium"
    if c == 585:
        return "Multifamily"
    if 500 <= c <= 599:
        return "Single Family"
    if 600 <= c <= 699:
        return "Exempt"
    if 700 <= c <= 799:
        return "Other"            # CRA abatement / TIF — private parcels, keep
    if 800 <= c <= 899:
        return "Utility / ROW"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(c, p) for c, p in zip(parcel["classcd"], parcel["pclass"])]
log(f"category (pre-filter): {parcel['PROPERTY_CATEGORY'].value_counts().to_dict()}")

# ── exemption flag ────────────────────────────────────────────────────────────
# Ohio 6xx class codes are the authoritative exempt list. A few 6xx rows carry
# PCLASS 'R' (e.g. 645 land-bank acquisitions) — still exempt. Owner-keyword
# heuristic catches government land miscoded into taxable classes.
_code = pd.to_numeric(parcel["classcd"], errors="coerce")
e_class = (_code >= 600) & (_code <= 699)
KW = ["CITY OF COLUMBUS", "COLUMBUS CITY", "FRANKLIN COUNTY", "STATE OF OHIO",
      "BOARD OF EDUCATION", "CITY SCHOOL", "SCHOOL DISTRICT", "OHIO STATE UNIV",
      "UNITED STATES", "U S A", "USA ", "METRO PARKS", "METROPOLITAN PARK",
      "CENTRAL OHIO TRANSIT", "COLUMBUS METROPOLITAN HOUSING", "HOUSING AUTHORITY",
      "REGIONAL AIRPORT AUTHORITY", "SOLID WASTE AUTHORITY", "BOARD OF COMMISSIONERS",
      "DEPT OF TRANSPORTATION", "DEPARTMENT OF TRANSPORTATION"]
e_own = parcel["OWNERNME1"].astype(str).str.upper().str.contains("|".join(KW), na=False)
parcel["exemption_flag"] = (e_class | e_own).astype(int)

ex = parcel[parcel["exemption_flag"] == 0].copy()
ex = ex[~ex["PROPERTY_CATEGORY"].isin({"Exempt", "Utility / ROW", "Mineral"})].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_state_class_prefixes=(),
    exclude_categories=("Agricultural / Rural", "Other", "Common Area / HOA"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar", fetch_footprints=False)
log(f"After exempt filter -> {len(ex):,} (dropped {int(parcel['exemption_flag'].sum()):,} exempt)")


# ── canonical fields — geodesic geometry area denominator ─────────────────────
# STATEDAREA mixes sqft and acres row-to-row and ACRES is null for Franklin, so the
# reported area is unusable; geometry is the denominator for every parcel.
def gis_area_sqft(g):
    if g is None or g.is_empty:
        return np.nan
    if g.geom_type == "Polygon":
        lon, lat = g.exterior.coords.xy
        a, _ = geod.polygon_area_perimeter(lon, lat)
        return abs(a) * 10.763910416709722
    if g.geom_type == "MultiPolygon":
        return sum(gis_area_sqft(p) for p in g.geoms)
    return np.nan


ex["geometry"] = ex["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
log("Computing GIS areas...")
ex["land_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
ex["area_source"] = "gis"
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex["tot_appr_val"], errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

# Per-county auditor detail link ships in the layer (Franklin -> FCAO, Delaware ->
# manatron, Fairfield -> its portal). Fallback: FCAO redirect by parcel id.
hl = ex["HYPERLINK"].astype(str).str.strip()
fallback = ("https://audr-apps.franklincountyohio.gov/redir/Link/Parcel/"
            + ex["acct"].str.replace("-", "", regex=False))
ex["link"] = np.where(hl.str.startswith("http"), hl, fallback)

# ── export ────────────────────────────────────────────────────────────────────
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
out = DATA_DIR / "columbus-oh-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"columbus-oh-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
lv = final["land_value_per_sqft"]
log(f"land_value_per_sqft: p50=${lv.median():.0f} p99=${lv.quantile(.99):.0f} "
    f"p999=${lv.quantile(.999):.0f} max=${lv.max():.0f}")
log("DONE")
