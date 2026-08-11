#!/usr/bin/env python3
"""
Build the City of Newport News, VA canonical parcel parquet.

Newport News is an independent city, so the city's own parcel layer is already city-only —
no county clip needed. Fully automated from the City's authoritative parcel MapServer
(geometry + current assessor land/improvement values + use/class + vacant/government flags +
ready-made assessor detail link, all in one layer).

Source (City of Newport News GIS, public, no token):
- Operational/Parcel MapServer/0:
  https://maps.nnva.gov/gis/rest/services/Operational/Parcel/MapServer/0
  ~54k parcels. Pulled (where 1=1) + cached on first run; reused after.

Outputs:
- data/jurisidictions/data/newportnews/newportnews-va-parcels.parquet
- data/jurisidictions/data/newportnews/newportnews-va-parcels_YYYY_MM_DD.parquet

Notes:
- ~54k parcels -> small enough for the browser GeoParquet path (no PMTiles / H3 hexes), like
  South Bend. Registered with usePmtiles omitted (default off) in newportnews.json.
- Values: CNTLNDVAL (current land value), CNTIMPVAL (current improvement value). full_market =
  land + improvement.
- Exempt: GOVERNMENT='Y' OR CITYOWNED='Y' (public/government land) -> exemption_flag=1, excluded.
- Category from CLASSDSCRP, with USEDSCRP overriding for PARKING (-> Parking) and VACANT
  (-> Vacant Land). Condominium units can stack on a footprint -> same-footprint collapse
  (SUM values) as in run_tulsa.py.
- $/sqft denominator is the assessor STATEDAREA (acres), fallback geodesic polygon area.
- link uses the parcel's own PublicLink (assessment.nnva.gov datalet).
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

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "newportnews"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "newportnews-va-geometry.parquet"

PARCELS_URL = ("https://maps.nnva.gov/gis/rest/services/"
               "Operational/Parcel/MapServer/0/query")
WHERE = "1=1"
OUT_FIELDS = ("PARCELID,USECD,USEDSCRP,CLASSCD,CLASSDSCRP,CNTLNDVAL,CNTIMPVAL,STATEDAREA,"
              "VACANT,GOVERNMENT,CITYOWNED,LIVUNIT,OWNERNME1,SITEADDRESS,PublicLink")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": WHERE, "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} Newport News parcels (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": WHERE, "outFields": OUT_FIELDS, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326,
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
        if off % 10000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


geom = fetch_parcels()
geom["acct"] = geom["PARCELID"].astype(str).str.strip()
geom = geom[geom["acct"].ne("") & geom["acct"].ne("None") & geom["acct"].ne("nan")]
for c in ["CNTLNDVAL", "CNTIMPVAL"]:
    geom[c] = pd.to_numeric(geom[c], errors="coerce")
geom["stated_acres"] = pd.to_numeric(geom["STATEDAREA"], errors="coerce")
for c in ["USEDSCRP", "CLASSDSCRP", "VACANT", "GOVERNMENT", "CITYOWNED"]:
    geom[c] = geom[c].astype(str).str.strip()

parcel = geom.rename(columns={"CNTLNDVAL": "land_val", "CNTIMPVAL": "bld_val"})
parcel["tot_appr_val"] = parcel["land_val"].fillna(0) + parcel["bld_val"].fillna(0)

if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
parcel = parcel[parcel["geometry"].notnull() & parcel["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
log(f"Valid-geometry parcels -> {len(parcel):,}")

# ── dedup multi-polygon parcels (values first, geometry unioned) ──────────────
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

# ── condo same-footprint collapse (>1 distinct account at one point -> SUM) ───
rp = parcel.geometry.representative_point()
parcel["_rpkey"] = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str))
vc = parcel["_rpkey"].value_counts()
stacked_keys = vc[vc > 1].index
log(f"Stacked footprints: {len(stacked_keys):,}; max stack: {int(vc.max())}; "
    f"parcels involved: {int(vc[vc > 1].sum()):,}")
if len(stacked_keys):
    is_stacked = parcel["_rpkey"].isin(stacked_keys)
    single = parcel[~is_stacked].copy()
    single["_collapsed"] = 0
    multi = parcel[is_stacked].copy()
    # Condo units are PER-UNIT shares of one shared parcel: sum the values AND the per-unit stated
    # area across the stack. stated_acres alone is unreliable as the land denominator (each unit's
    # 0.04 ac is a share, not the footprint), so the shared-footprint area is reconciled later as
    # max(summed stated area, union-polygon geodesic area) for `_collapsed` rows. Summing values
    # while taking one unit's area inflated $/sqft by the unit count (e.g. pin 277000105: ~14 units
    # -> $504/sqft instead of ~$35).
    sum_cols = ["land_val", "bld_val", "tot_appr_val", "stated_acres"]
    first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
    agg = {c: "sum" for c in sum_cols if c in multi.columns}
    agg.update({c: "first" for c in first_cols})
    coll = multi.groupby("_rpkey", dropna=False).agg(agg).reset_index()
    gu = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    coll["_collapsed"] = 1
    coll = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
    parcel = gpd.GeoDataFrame(pd.concat([single, coll], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
else:
    parcel["_collapsed"] = 0
parcel = parcel.drop(columns=["_rpkey"], errors="ignore")
log(f"After condo footprint collapse -> {len(parcel):,}")

# ── exemption flag + classification ──────────────────────────────────────────
parcel["exemption_flag"] = ((parcel["GOVERNMENT"].str.upper() == "Y")
                            | (parcel["CITYOWNED"].str.upper() == "Y")).astype(int)


def categorize(cls, use):
    cls = str(cls or "").strip()
    use = str(use or "").strip().upper()
    if "PARKING" in use:
        return "Parking"
    if use == "VACANT" or "VACANT" in use:
        return "Vacant Land"
    if cls == "Residential Single Family":
        return "Single Family"
    if cls == "Condominium":
        return "Condominium"
    if cls == "Commercial":
        return "Commercial"
    if cls == "Industrial":
        return "Industrial"
    if cls in ("Multi Family (2-4 dwellings)", "Apartment (over 4 dwellings)"):
        return "Multifamily"
    if cls == "Trailer court":
        return "Mobile Home"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(c, u) for c, u in
                               zip(parcel["CLASSDSCRP"], parcel["USEDSCRP"])]

ex = parcel[parcel["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other",),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)
log(f"After exempt filter -> {len(ex):,} (dropped {int(parcel['exemption_flag'].sum()):,} exempt)")

# ── canonical fields — STATEDAREA (acres) denominator, geodesic fallback ─────
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
rep = pd.to_numeric(ex.get("stated_acres", np.nan), errors="coerce") * SQFT_PER_ACRE
ex["reported_sqft"] = rep
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# Collapsed condo stacks: use the SHARED complex footprint as the land denominator (not one unit's
# stated area). Take the larger of the summed per-unit stated area and the union polygon's geodesic
# area; degenerate sub-500-sqft slivers fall through to likely_remnant.
col = ex["_collapsed"] == 1
ex.loc[col, "reported_sqft"] = np.maximum(
    pd.to_numeric(ex.loc[col, "reported_sqft"], errors="coerce").fillna(0.0),
    pd.to_numeric(ex.loc[col, "geom_area_sqft"], errors="coerce").fillna(0.0))
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

# Assessor datalet link (each parcel carries its own PublicLink); fall back to a built URL.
pl = ex.get("PublicLink").astype(str)
built = "https://assessment.nnva.gov/PT/datalets/datalet.aspx?UseSearch=no&pin=" + ex["acct"].astype(str) + "&jur=700"
ex["link"] = pl.where(pl.str.startswith("http"), built)

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
out = DATA_DIR / "newportnews-va-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"newportnews-va-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
