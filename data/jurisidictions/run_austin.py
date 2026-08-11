#!/usr/bin/env python3
"""
Build Austin's canonical parcel parquet by joining two free TCAD sources.

Sources:
- Appraisal values: TCAD appraisal-roll export PROP.TXT (fixed-width PACS "Legacy"
  layout). Download the ZIP from https://traviscad.org/publicinformation and place it at
  data/jurisidictions/data/austin/tcad-appraisal-export.zip (Cloudflare blocks scripted
  download).
- Parcel geometry: Travis County taxmaps ArcGIS parcels layer
  https://taxmaps.traviscountytx.gov/arcgis/rest/services/Parcels/MapServer/0
  (pulled + cached on first run; reused after).

Outputs:
- data/jurisidictions/data/austin/austin-tx-parcels.parquet
- data/jurisidictions/data/austin/austin-tx-parcels_YYYY_MM_DD.parquet

Notes:
- $/sqft denominator is the assessor's REPORTED land size (land_acres, fallback
  legal_acreage) when present, else the GIS polygon area — robust to fragment/sliver
  polygons (GIS ~ reported for ~99% of parcels). Emits QC columns land_area_acres,
  area_source, and likely_remnant (tiny <500 sqft fractional remnants).
- City limits via the authoritative City of Austin FULL PURPOSE jurisdiction boundary
  (not osmnx geocoding).
- Upload + PMTiles are separate steps:
    python data/upload_austin_dev.py
    python data/scripts/parquet_to_pmtiles.py --city austin --h3 --wsl --upload
- classify_property_refined(fetch_footprints=False): Austin hides the Vacant &
  Underdeveloped tab (hideUnderutilized), so the Overture footprint cross-check is
  skipped for speed; flip to True if that tab is ever enabled.

Mirrors data/jurisidictions/austin.ipynb (the notebook is the reference).
"""
from __future__ import annotations

import io
import os
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402
from cloud_utils import get_feature_data_with_geometry  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "austin"
ZIP = DATA_DIR / "tcad-appraisal-export.zip"
GEOM_CACHE = DATA_DIR / "austin-tx-geometry.parquet"
SQFT_PER_ACRE = 43560.0
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── 1. Parse PROP.TXT (values + state code + owner + situs + REPORTED ACREAGE) ──
PROP_FIELDS = [
    ("prop_id", 1, 12), ("prop_type_cd", 13, 17), ("sup_num", 23, 34),
    ("py_owner_name", 609, 678), ("situs_city", 1110, 1139),
    ("legal_acreage", 1660, 1675),
    ("land_hstd_val", 1796, 1810), ("land_non_hstd_val", 1811, 1825),
    ("imprv_hstd_val", 1826, 1840), ("imprv_non_hstd_val", 1841, 1855),
    ("appraised_val", 1916, 1930), ("ex_exempt", 2671, 2671),
    ("land_state_cd", 2742, 2751), ("land_acres", 2772, 2791),
    ("market_value", 4214, 4227),
]
COLSPECS = [(s - 1, e) for (_, s, e) in PROP_FIELDS]
NAMES = [n for (n, _, _) in PROP_FIELDS]
VALUE_COLS = ["land_hstd_val", "land_non_hstd_val", "imprv_hstd_val",
              "imprv_non_hstd_val", "appraised_val", "market_value"]

log("Parsing PROP.TXT (streamed)...")
kept, total = [], 0
with zipfile.ZipFile(ZIP) as zf:
    name = next(n for n in zf.namelist() if os.path.basename(n).upper() == "PROP.TXT")
    with zf.open(name) as fh:
        for ch in pd.read_fwf(io.TextIOWrapper(fh, encoding="latin-1", newline=""),
                              colspecs=COLSPECS, names=NAMES, dtype=str, chunksize=200_000):
            total += len(ch)
            for c in NAMES:
                ch[c] = ch[c].astype(str).str.strip()
            ch["sup_num"] = pd.to_numeric(ch["sup_num"], errors="coerce")
            ch = ch[(ch["sup_num"] == 0) & (ch["prop_type_cd"].str.upper() == "R")]
            if len(ch):
                kept.append(ch)
prop_df = pd.concat(kept, ignore_index=True)
for c in VALUE_COLS:
    prop_df[c] = pd.to_numeric(prop_df[c], errors="coerce").fillna(0)
# reported land size (4 implied decimals); prefer land_acres, fall back to legal_acreage
prop_df["land_acres_ac"] = pd.to_numeric(prop_df["land_acres"], errors="coerce") / 10000.0
prop_df["legal_acres_ac"] = pd.to_numeric(prop_df["legal_acreage"], errors="coerce") / 10000.0
prop_df["reported_ac"] = prop_df["land_acres_ac"].where(prop_df["land_acres_ac"] > 0, prop_df["legal_acres_ac"])
prop_df["prop_id"] = pd.to_numeric(prop_df["prop_id"], errors="coerce").astype("Int64")
prop_df = prop_df.dropna(subset=["prop_id"]).drop_duplicates("prop_id")
prop_df["_land_value"] = prop_df["land_hstd_val"] + prop_df["land_non_hstd_val"]
prop_df["_imp_value"] = prop_df["imprv_hstd_val"] + prop_df["imprv_non_hstd_val"]
prop_df["_market"] = prop_df["market_value"].where(prop_df["market_value"] > 0,
                                                    prop_df["_land_value"] + prop_df["_imp_value"])
appraisal = prop_df[["prop_id", "_land_value", "_imp_value", "_market", "reported_ac",
                     "land_state_cd", "py_owner_name", "situs_city", "ex_exempt"]].copy()
log(f"  {len(appraisal):,} properties | with reported acreage: {int((appraisal['reported_ac']>0).sum()):,}")

# ── 2. Geometry (taxmaps ArcGIS; cached after first pull) + join ─────────────
if GEOM_CACHE.exists():
    log(f"Using cached geometry: {GEOM_CACHE.name}")
    geom = gpd.read_parquet(GEOM_CACHE)
else:
    log("Pulling parcel geometry from taxmaps ArcGIS (paginated; keeps outer ring)...")
    geom = get_feature_data_with_geometry(
        dataset_name="Parcels",
        base_url="https://taxmaps.traviscountytx.gov/arcgis/rest/services",
        layer_id=0, paginate=True, out_epsg=4326, service_type="MapServer", verbose=True,
    )
    if geom is None or len(geom) == 0:
        raise RuntimeError("Geometry pull returned no features — check the taxmaps endpoint.")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name}")
g_id = next(c for c in ["PROP_ID", "prop_id", "property_id"] if c in geom.columns)
g_link = next((c for c in ["hyperlink", "link", "url"] if c in geom.columns), None)
geom["prop_id"] = pd.to_numeric(geom[g_id], errors="coerce").astype("Int64")
geom = geom.dropna(subset=["prop_id"])
if g_link and g_link != "hyperlink":
    geom = geom.rename(columns={g_link: "hyperlink"})
parcel = geom.merge(appraisal, on="prop_id", how="left")
parcel["land_val"] = parcel["_land_value"]
parcel["bld_val"] = parcel["_imp_value"]
parcel["tot_appr_val"] = parcel["_market"]
parcel["state_class"] = parcel["land_state_cd"]
parcel["mailto"] = parcel["py_owner_name"]
parcel["city"] = parcel["situs_city"]
parcel["acct"] = parcel["prop_id"].astype(str)
log(f"Joined {len(parcel):,} parcels | matched {int(parcel['_land_value'].notna().sum()):,}")

# ── 3. Authoritative city-limits filter (full-purpose jurisdiction) ──────────
if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
JURIS = ("https://services.arcgis.com/0L95CJ0VTaxqcmED/ArcGIS/rest/services/"
         "BOUNDARIES_jurisdictions/FeatureServer/0/query")
r = requests.get(JURIS, params={"where": "JURISDICTION_TYPE='FULL'", "outFields": "JURISDICTION_TYPE",
                                "outSR": 4326, "f": "geojson"}, timeout=120)
r.raise_for_status()
boundary = unary_union(list(gpd.read_file(io.BytesIO(r.content)).to_crs("EPSG:4326").geometry))
valid = parcel["geometry"].notnull() & parcel["geometry"].apply(lambda x: getattr(x, "is_valid", False))
cent = parcel.loc[valid, "geometry"].to_crs(3857).centroid.to_crs(4326)
inside = valid.copy()
inside[valid] = cent.within(boundary)
parcel = parcel[inside].copy()
log(f"City-limits filter -> {len(parcel):,}")

# ── 4. dedup, categorize, exempt, refined ───────────────────────────────────
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
if ndup:
    sum_cols = [c for c in ["tot_appr_val", "land_val", "bld_val"] if c in parcel.columns]
    cat_cols = [c for c in parcel.columns if c not in set(sum_cols + ["geometry", "acct"])]
    agg = {c: "sum" for c in sum_cols}; agg.update({c: "first" for c in cat_cols})
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs=parcel.crs)
log(f"After dedup -> {len(parcel):,}")


def categorize(v):
    raw = str(v or "").strip().upper()
    if raw == "X": return "Exempt"
    if raw == "W": return "Government / State"
    if raw in ("A1", "A2"): return "Single Family"
    if raw in ("B1", "B2", "B3", "B4"): return "Multifamily"
    if raw in ("C1", "C2", "C3"): return "Vacant Residential"
    if raw in ("D1", "D2", "E1", "E2", "E3"): return "Agricultural / Rural"
    if raw == "F1": return "Commercial"
    if raw == "F2": return "Industrial"
    if raw.startswith("G"): return "Mineral / Oil & Gas"
    if raw.startswith("J") or raw.startswith("U"): return "Utility"
    if raw in ("L1", "L2", "M1", "O1", "S"): return "Personal Property / Inventory"
    return "Other"


parcel["PROPERTY_CATEGORY"] = parcel["state_class"].apply(categorize)
ex = parcel.copy()
ebs = ex["PROPERTY_CATEGORY"].isin(["Exempt", "Government / State"])
eflag = ex.get("ex_exempt", pd.Series("", index=ex.index)).astype(str).str.upper().eq("T")
KW = ["CITY OF AUSTIN", "TRAVIS COUNTY", "STATE OF TEXAS", "AUSTIN ISD", "AISD",
      "AUSTIN COMMUNITY COLLEGE", "AUSTIN COMM COLL", "CENTRAL HEALTH", "LOWER COLORADO RIVER",
      "LCRA", "AUSTIN WATER", "CAPITAL METRO", "CAP METRO", "UNIVERSITY OF TEXAS", "UNIV OF TEXAS",
      "UNITED STATES", "US GOVT", "U.S. GOVERNMENT"]
eown = ex["mailto"].astype(str).str.upper().str.contains("|".join(KW), na=False)
ex["exemption_flag"] = (ebs | eflag | eown).astype(int)
ex = ex[ex["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex = ex[~ex["property_land_use_category"].isin({"Mineral / Oil & Gas", "Personal Property / Inventory"})].copy()
ex["land_value"] = pd.to_numeric(ex.get("land_val", np.nan), errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex.get("bld_val", np.nan), errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(ex, fetch_footprints=False)
log(f"After exempt/refine -> {len(ex):,}")

# ── 5. Canonical fields — REPORTED-land-size denominator (GIS fallback) ──────
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
ex["gis_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["gis_area_sqft"] < 1, "gis_area_sqft"] = np.nan
ex["reported_sqft"] = pd.to_numeric(ex.get("reported_ac", np.nan), errors="coerce") * SQFT_PER_ACRE

# Denominator: assessor reported size when present, else GIS polygon area.
use_reported = ex["reported_sqft"] > 0
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["gis_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
# Flag tiny fractional remnants/slivers: < 500 sqft can't be a standalone lot, so its
# $/sqft (assessor value / tiny area) is meaningless and renders as a false spike.
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex.get("tot_appr_val", np.nan), errors="coerce")
ex["land_value"] = pd.to_numeric(ex.get("land_val", np.nan), errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex.get("bld_val", np.nan), errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

if "hyperlink" in ex.columns and ex["hyperlink"].notna().any():
    ex["link"] = ex["hyperlink"].astype(str).str.replace("stage.travis.prodigycad.com", "travis.prodigycad.com", regex=False)
else:
    ex["link"] = "https://travis.prodigycad.com/property-detail/" + ex["acct"].astype(str)

# ── 6. Export ───────────────────────────────────────────────────────────────
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
out = DATA_DIR / "austin-tx-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"austin-tx-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"land_value_per_sqft now: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"p999=${final['land_value_per_sqft'].quantile(.999):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
# before/after on the two known artifacts
for pid in (547972, 262102):
    row = ex[ex["acct"] == str(pid)]
    if len(row):
        rr = row.iloc[0]
        log(f"  prop {pid}: gis={rr['gis_area_sqft']:,.0f}sqft reported={rr['reported_sqft']:,.0f}sqft "
            f"src={rr['area_source']} lvpsqft=${rr['land_value_per_sqft']:,.0f}")
log("DONE")
